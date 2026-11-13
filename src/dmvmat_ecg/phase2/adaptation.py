"""Synthetic-only selective adaptation with persistent module-mode enforcement."""

import copy
import hashlib
import io
import itertools
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn

from .contracts import DIMENSIONS, MODULES, Embeddings, SyntheticRoles


def state_hash(model):
    h = hashlib.sha256()
    for name, v in sorted(model.state_dict().items()):
        a = v.detach().cpu().numpy()
        h.update(str((name, a.shape, a.dtype)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def validate_owners(model):
    owners = model.selectable_modules()
    if set(owners) != set(MODULES):
        raise ValueError("Seven explicit modules required")
    all_owners = {**owners, "prediction_head": model.prediction_head}
    sets = [{id(p) for p in m.parameters()} for m in all_owners.values()]
    if sum(map(len, sets)) != len(set().union(*sets)) or set().union(*sets) != {
        id(p) for p in model.parameters()
    }:
        raise ValueError("Parameters must belong to exactly one view/fusion/head owner")
    # Cross-owner shared buffers would defeat frozen-state guarantees too.
    buffers = [{id(b) for b in m.buffers()} for m in all_owners.values()]
    if sum(map(len, buffers)) != len(set().union(*buffers)):
        raise ValueError("Buffers shared across adaptation owners")
    return owners


class SelectiveModel(nn.Module):
    """Clone source; SHARE stays eval even after enclosing wrappers call train()."""

    def __init__(self, source, mask, *, source_only=False):
        super().__init__()
        if set(mask) != set(MODULES) or any(v not in {"SHARE", "ADAPT"} for v in mask.values()):
            raise ValueError("Complete explicit SHARE/ADAPT mask required")
        if source_only and any(v != "SHARE" for v in mask.values()):
            raise ValueError("Source-only cannot adapt")
        validate_owners(source)
        self.model = copy.deepcopy(source)
        self.mask = dict(mask)
        self.source_only = source_only
        for name, module in self.model.selectable_modules().items():
            module.requires_grad_(mask[name] == "ADAPT")
        self.model.prediction_head.requires_grad_(not source_only)
        # Clear inherited gradients; copied checkpoint weights are initializations.
        for p in self.model.parameters():
            p.grad = None
        self._frozen = {
            name: {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
            for name, module in self.model.selectable_modules().items()
            if mask[name] == "SHARE"
        }
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        for name, module in self.model.selectable_modules().items():
            if self.mask[name] == "SHARE" or self.source_only:
                module.eval()
        if self.source_only:
            self.model.prediction_head.eval()
        return self

    def forward(self, **inputs):
        return self.model(**inputs)

    def optimizer(self, *, lr=0.0003):
        params = [p for p in self.parameters() if p.requires_grad]
        if not params:
            raise ValueError("Source-only has no optimizer")
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.01, betas=(0.9, 0.99))

    def verify_frozen(self):
        for name, snapshot in self._frozen.items():
            current = self.model.selectable_modules()[name].state_dict()
            if any(not torch.equal(v, current[k].detach().cpu()) for k, v in snapshot.items()):
                raise RuntimeError("Frozen weights or buffers changed: " + name)


def random_mask(source, reference, seed):
    owners = validate_owners(source)
    if set(reference) != set(MODULES) or any(
        v not in {"SHARE", "ADAPT"} for v in reference.values()
    ):
        raise ValueError("Invalid reference mask")
    counts = [sum(p.numel() for p in owners[n].parameters()) for n in MODULES]
    budget = sum(c for c, n in zip(counts, MODULES, strict=True) if reference[n] == "ADAPT")
    candidates = list(itertools.product((False, True), repeat=7))
    errors = [
        abs(sum(c * bit for c, bit in zip(counts, bits, strict=True)) - budget)
        for bits in candidates
    ]
    eligible = np.flatnonzero(np.asarray(errors) == min(errors))
    selected = int(np.random.default_rng(seed).choice(eligible))
    mask = {
        n: "ADAPT" if bit else "SHARE" for n, bit in zip(MODULES, candidates[selected], strict=True)
    }
    return mask, dict(
        parameter_error=errors[selected],
        candidate_count=len(eligible),
        equality_probability=1 / len(eligible),
        same_as_reference=mask == reference,
    )


@dataclass(frozen=True)
class Comparator:
    name: str
    cost_matched: bool
    needs_evidence_labels: bool


@dataclass(frozen=True)
class SourceLineage:
    checkpoint_hash: str
    training_groups: tuple
    selection_groups: tuple = ()
    origin: str = "synthetic_fixture"
    label_names: tuple = ()

    def validate(self, source, roles):
        if self.origin != "synthetic_fixture" or self.checkpoint_hash != state_hash(source):
            raise PermissionError("Only hash-bound synthetic source fixtures are supported")
        if not self.training_groups:
            raise PermissionError("Source patient-history declaration is required")
        roles.validate_source_history((*self.training_groups, *self.selection_groups))
        if tuple(self.label_names) != roles.label_names:
            raise PermissionError("Source task label schema is absent or incompatible")


COMPARATORS = {
    "source_only": Comparator("source_only", False, False),
    "head_only": Comparator("head_only", False, False),
    "full": Comparator("full", False, False),
    "full_cost": Comparator("full_cost", True, False),
    "head_cost": Comparator("head_cost", True, False),
    "random": Comparator("random", False, True),
    "selective": Comparator("selective", False, True),
}


def masked_bce(logits, targets, observed):
    if (
        logits.shape != targets.shape
        or logits.shape != observed.shape
        or observed.dtype != torch.bool
        or logits.ndim != 2
    ):
        raise ValueError("Aligned logits/targets/boolean observation mask required")
    if (
        not torch.isfinite(logits).all()
        or not torch.isin(targets[observed], targets.new_tensor([0, 1])).all()
    ):
        raise ValueError("Nonfinite logits or invalid observed targets")
    if not observed.any():
        raise ValueError("No verified labels in batch; fail rather than fabricate negatives")
    return nn.functional.binary_cross_entropy_with_logits(logits[observed], targets[observed])


def macro_ap(y, probabilities, observed):
    from .contracts import observed_labels

    y, observed = observed_labels(y, observed)
    p = np.asarray(probabilities)
    if p.shape != y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid continuous probabilities")
    scores = []
    for j in range(y.shape[1]):
        use = observed[:, j]
        if len(np.unique(y[use, j])) != 2:
            raise ValueError("Checkpoint objective undefined for an unsupported label")
        scores.append(average_precision_score(y[use, j], p[use, j]))
    return float(np.mean(scores))


class APCheckpoint:
    def __init__(self):
        self.best = None
        self.state = None
        self.epoch = None

    def consider(self, model, *, ap, bce, epoch):
        if not np.isfinite([ap, bce]).all() or not 0 <= ap <= 1 or bce < 0 or epoch < 0:
            raise ValueError("Invalid checkpoint score")
        if self.epoch is not None and epoch <= self.last_epoch:
            raise ValueError("Checkpoint looks must be chronological")
        self.last_epoch = epoch
        improved = (
            self.best is None
            or ap > self.best[0] + 1e-12
            or (abs(ap - self.best[0]) <= 1e-12 and bce < self.best[1])
        )
        if improved:
            self.best, self.epoch = (ap, bce), epoch
            self.state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        return improved


def _tensors(batch):
    return {k: torch.tensor(v.copy()) for k, v in batch.inputs.items()}


def extract_synthetic(source, roles: SyntheticRoles, role, preprocessing_hash, *, dimensions=None):
    batch = roles.access(role, "evidence")
    model = copy.deepcopy(source).cpu().eval()
    with torch.no_grad():
        reps = model.representations(**_tensors(batch))
    dimensions = dimensions or DIMENSIONS
    if set(reps) != set(MODULES) or any(
        reps[n].shape != (len(batch.y), dimensions[n]) for n in MODULES
    ):
        raise ValueError("Representation contract mismatch")
    return Embeddings(
        {n: z.detach().cpu().numpy() for n, z in reps.items()},
        batch.y,
        batch.observed,
        batch.groups,
        role,
        state_hash(source),
        preprocessing_hash,
        batch.identity,
        batch.label_names,
    )


def adapt_synthetic(
    source,
    roles: SyntheticRoles,
    *,
    lineage,
    method,
    evidence=None,
    updates=4,
    look_every=1,
    batch_size=8,
    patience=5,
    seed=101,
    lr=0.0003,
):
    """Bounded CPU fixture executor. No file loader, calibration or outer evaluation.

    It intentionally cannot execute the proposed real study. Persistent real-fit
    accounting, role-authorized real adapters and a new evaluation lock are future
    separately authorized infrastructure.
    """
    if method not in COMPARATORS or updates < 1 or min(look_every, batch_size, patience) < 1:
        raise ValueError("Invalid synthetic comparison/budget")
    plan = COMPARATORS[method]
    lineage.validate(source, roles)
    original = state_hash(source)
    mask = dict.fromkeys(MODULES, "SHARE")
    if plan.needs_evidence_labels:
        batch = roles.access("evidence", "evidence")
        if (
            evidence is None
            or evidence["checkpoint_hash"] != original
            or evidence.get("target_identity") != batch.identity
        ):
            raise PermissionError("Mask does not belong to this source initialization")
        mask = evidence["mask"]
        if method == "random":
            mask, _ = random_mask(source, mask, seed)
    elif method in {"full", "full_cost"}:
        mask = dict.fromkeys(MODULES, "ADAPT")
    controller = SelectiveModel(source, mask, source_only=method == "source_only").cpu()
    if method == "source_only":
        return controller, dict(
            scope="synthetic-only",
            namespace=roles.namespace,
            manifest_hash=roles.manifest_hash,
            method=method,
            selected_update=0,
            updates=0,
            source_hash=original,
            label_cost=roles.label_cost(),
        )
    fit_roles = ("fit", "evidence") if plan.cost_matched else ("fit",)
    batches = [
        roles.access(r, "fit_cost_matched" if plan.cost_matched else "fit") for r in fit_roles
    ]
    stop = roles.access("stop", "stop")
    inputs = {k: torch.cat([_tensors(b)[k] for b in batches]) for k in batches[0].inputs}
    y = torch.tensor(np.concatenate([b.y for b in batches]), dtype=torch.float32)
    obs = torch.tensor(np.concatenate([b.observed for b in batches]), dtype=torch.bool)
    optimizer = controller.optimizer(lr=lr)
    checkpoint, history, stale = APCheckpoint(), [], 0

    def score(epoch):
        controller.eval()
        with torch.no_grad():
            logits = controller(**_tensors(stop))
            value = float(
                masked_bce(
                    logits,
                    torch.tensor(stop.y.copy(), dtype=torch.float32),
                    torch.tensor(stop.observed.copy()),
                )
            )
            ap = macro_ap(stop.y, logits.sigmoid().numpy(), stop.observed)
        history.append(dict(epoch=epoch, stop_ap=ap, stop_bce=value))
        return checkpoint.consider(controller, ap=ap, bce=value, epoch=epoch)

    score(0)
    completed = 0
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        for step in range(1, updates + 1):
            ix = torch.randperm(len(y), generator=generator)[: min(batch_size, len(y))]
            controller.train()
            optimizer.zero_grad(set_to_none=True)
            loss = masked_bce(controller(**{k: v[ix] for k, v in inputs.items()}), y[ix], obs[ix])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                controller.parameters(), float("inf"), error_if_nonfinite=True
            )
            optimizer.step()
            if any(not torch.isfinite(v).all() for v in controller.parameters()):
                raise FloatingPointError("Nonfinite updated parameters")
            controller.verify_frozen()
            completed = step
            if step % look_every == 0 or step == updates:
                stale = 0 if score(step) else stale + 1
                if stale >= patience:
                    break
    controller.load_state_dict(checkpoint.state, strict=True)
    controller.eval()
    controller.verify_frozen()
    if state_hash(source) != original:
        raise RuntimeError("Source initialization was corrupted")
    return controller, dict(
        scope="synthetic-only",
        namespace=roles.namespace,
        manifest_hash=roles.manifest_hash,
        method=method,
        mask=mask,
        selected_update=checkpoint.epoch,
        updates=completed,
        history=history,
        source_hash=original,
        checkpoint_hash=state_hash(controller),
        label_cost=roles.label_cost(),
    )


def serialize_synthetic(model):
    """Return weight-only bytes+digest; never creates/discovers existing checkpoints."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    payload = buffer.getvalue()
    return payload, hashlib.sha256(payload).hexdigest()


def load_synthetic(model, payload, expected_hash):
    if hashlib.sha256(payload).hexdigest() != expected_hash:
        raise PermissionError("Checkpoint bytes differ from expected hash")
    state = torch.load(io.BytesIO(payload), weights_only=True)
    if any(not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Nonfinite checkpoint state")
    candidate = copy.deepcopy(model)
    candidate.load_state_dict(state, strict=True)
    if isinstance(candidate, SelectiveModel):
        candidate.verify_frozen()
    model.load_state_dict(candidate.state_dict(), strict=True)
    return model
