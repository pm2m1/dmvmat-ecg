"""Authorized real CPU development; adopted numerical/training recipe unchanged."""

import copy
import io
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from dmvmat_ecg.phase2.adaptation import SelectiveModel, macro_ap, masked_bce, state_hash
from dmvmat_ecg.phase2.contracts import MODULES
from dmvmat_ecg.phase3b.jobs import file_hash, source_code_hash
from dmvmat_ecg.phase3b.models import fresh_model, inputs, random_count_cost
from dmvmat_ecg.phase3b.protocol import (
    BATCH_SIZE,
    METHODS,
    OPTIMIZER,
    SOURCE_EPOCHS,
    UPDATES,
    adopted_settings,
    digest,
    seed,
)
from dmvmat_ecg.phase3b.roles import Normalization

from .boundary import implementation_hash, real_only
from .prepare import location


def pack(value):
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def unpack(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def finite_step(model, optimizer, loss):
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite loss; no earlier-checkpoint substitution")
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError("Nonfinite gradient")
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError("Nonfinite parameter")


@dataclass
class Source:
    model: torch.nn.Module
    normalization: Normalization
    checkpoint_hash: str
    train_identity: str
    stop_identity: str
    origin: str = "fresh-phase3b-two-proxy"
    source_seed: int = 15101
    file: str | None = None
    sha: str | None = None

    def validate(self, data):
        real_only(data.namespace)
        if self.file is None or not Path(self.file).resolve().is_relative_to(
            location(data.root) / "artifacts"
        ):
            raise PermissionError(
                "Only a fresh source artifact from this real namespace is allowed"
            )
        if (
            self.origin != "fresh-phase3b-two-proxy"
            or self.model.prediction_head.out_features != 2
            or state_hash(self.model) != self.checkpoint_hash
            or self.train_identity != data.role("source_train").identity
            or self.stop_identity != data.role("source_stop").identity
            or self.normalization.origin != "fresh-source-only"
            or self.normalization.training_identity != digest([self.train_identity])
        ):
            raise PermissionError("Historical, corrupted or membership-incompatible source")
        if self.file is not None:
            if file_hash(self.file) != self.sha:
                raise PermissionError("Source checkpoint file changed")
            saved = unpack(self.file)
            if (
                saved["normalization"] != self.normalization.__dict__
                or saved["state_hash"] != self.checkpoint_hash
                or saved["identity"]["seed"] != self.source_seed
                or saved["identity"]["data"] != [self.train_identity, self.stop_identity]
            ):
                raise PermissionError("Source weights/normalization differ from fitted artifact")


def train_source(data, log, slot, source_seed, *, factory=fresh_model, retry=False):
    real_only(data.namespace)
    if log.namespace != data.namespace:
        raise PermissionError("Mixed run namespaces")
    log.validate_slot(slot, kind="source", source_seed=source_seed)
    torch.set_num_threads(4)
    settings = adopted_settings()
    identity = dict(
        kind="source",
        data=[data.role(n).identity for n in ("source_train", "source_stop")],
        seed=source_seed,
        settings=settings,
        code=source_code_hash(),
        implementation=implementation_hash(),
        factory=f"{factory.__module__}.{factory.__qualname__}",
    )

    def fit():
        started = time.monotonic()
        normalization = Normalization.fit(data)
        x, y, obs = data.read("source_train", "fit_source")
        vx, vy, vm = data.read("source_stop", "stop_source")
        model = factory(source_seed).cpu().float()
        if model.prediction_head.out_features != 2:
            raise PermissionError("Fresh source must have exactly two proxy outputs")
        xx = inputs(model, normalization.transform(x))
        vv = inputs(model, normalization.transform(vx))
        yy, mm = torch.tensor(y, dtype=torch.float32), torch.tensor(obs)
        steps = (len(y) + 127) // 128
        optimizer = torch.optim.AdamW(model.parameters(), **settings["source_optimizer"])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.01,
            epochs=SOURCE_EPOCHS,
            steps_per_epoch=steps,
            **settings["source_schedule"],
        )
        best, best_state, history = None, None, []
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(source_seed)
            generator = torch.Generator().manual_seed(source_seed)
            for epoch in range(1, SOURCE_EPOCHS + 1):
                order = torch.randperm(len(y), generator=generator)
                model.train()
                losses = []
                for start in range(0, len(y), 128):
                    ix = order[start : start + 128]
                    optimizer.zero_grad(set_to_none=True)
                    loss = masked_bce(model(**{k: v[ix] for k, v in xx.items()}), yy[ix], mm[ix])
                    finite_step(model, optimizer, loss)
                    scheduler.step()
                    losses.append(float(loss.detach()))
                model.eval()
                with torch.no_grad():
                    logits = model(**vv)
                    bce = float(
                        masked_bce(logits, torch.tensor(vy, dtype=torch.float32), torch.tensor(vm))
                    )
                    ap = macro_ap(vy, logits.sigmoid().numpy(), vm)
                if (
                    best is None
                    or ap > best[0] + 1e-12
                    or (abs(ap - best[0]) <= 1e-12 and bce < best[1] - 1e-12)
                ):
                    best = ap, bce, epoch
                    best_state = copy.deepcopy(model.state_dict())
                print(
                    f"SOURCE {source_seed} epoch {epoch}/{SOURCE_EPOCHS} AP={ap:.6f} BCE={bce:.6f}",
                    flush=True,
                )
                history.append(
                    dict(
                        epoch=epoch,
                        training_bce=float(np.mean(losses)),
                        source_stop_ap=ap,
                        source_stop_bce=bce,
                    )
                )
        model.load_state_dict(best_state)
        model.eval()
        from sklearn.metrics import average_precision_score, roc_auc_score

        with torch.no_grad():
            retained = model(**vv).sigmoid().numpy()
        per_label = [
            dict(
                AP=float(average_precision_score(vy[vm[:, j], j], retained[vm[:, j], j])),
                AUROC=float(roc_auc_score(vy[vm[:, j], j], retained[vm[:, j], j])),
            )
            for j in range(2)
        ]
        return pack(
            dict(
                state=best_state,
                normalization=normalization.__dict__,
                selected_epoch=best[2],
                source_stop_per_label=per_label,
                history=history,
                identity=identity,
                state_hash=state_hash(model),
                runtime_seconds=time.monotonic() - started,
            )
        )

    path, alias = log.execute(slot, identity, fit, retry=retry)
    saved = unpack(path)
    model = factory(source_seed).cpu().float()
    model.load_state_dict(saved["state"])
    model.eval()
    source = Source(
        model,
        Normalization(**saved["normalization"]),
        saved["state_hash"],
        data.role("source_train").identity,
        data.role("source_stop").identity,
        source_seed=source_seed,
        file=str(path),
        sha=file_hash(path),
    )
    source.validate(data)
    return source, dict(
        alias=alias,
        artifact=str(path),
        **{
            k: saved[k]
            for k in ("selected_epoch", "history", "runtime_seconds", "source_stop_per_label")
        },
    )


def fit_target(
    data,
    log,
    slot,
    source,
    method,
    *,
    source_seed,
    draw_seed,
    budget,
    decision=None,
    factory=fresh_model,
    retry=False,
):
    real_only(data.namespace)
    if log.namespace != data.namespace or method not in METHODS:
        raise PermissionError("Wrong run namespace or unregistered method")
    log.validate_slot(
        slot,
        kind="target",
        source_seed=source_seed,
        draw_seed=draw_seed,
        budget=budget,
        method=method,
    )
    if source.source_seed != source_seed:
        raise PermissionError("Source seed lineage differs from the paired episode")
    torch.set_num_threads(4)
    source.validate(data)
    if decision is not None:
        from dataclasses import asdict

        from dmvmat_ecg.phase3b.evidence import configuration

        if decision.report.get("configuration") != asdict(configuration()):
            raise PermissionError("Real decisions require exact production Monte Carlo policy")
    if (
        len(data.role("F").records) != budget // 2
        or len(data.role("G").records) != budget // 2
        or budget not in (64, 128, 256)
    ):
        raise ValueError("Budget must be the registered half-F/half-G allocation")
    fit_seed = seed([78501, source_seed, draw_seed, budget])
    mask = dict.fromkeys(MODULES, "SHARE" if method == "head_cost" else "ADAPT")
    random_info = None
    if method in {"random_a", "selective_a"}:
        if decision is None:
            raise PermissionError("G-bound source decision required before fitting")
        mask = decision.validate(source, data)
        if method == "random_a":
            mask, random_info = random_count_cost(
                source.model, mask, seed([78601, source_seed, draw_seed, budget])
            )
    target_only = method == "target_only_cost"
    normalization = (
        Normalization.fit(data, target_only=True) if target_only else source.normalization
    )
    initial = factory(fit_seed).cpu().float() if target_only else source.model
    if initial.prediction_head.out_features != 2:
        raise ValueError("Two-proxy head required")
    # Method name and selector provenance are deliberately NOT predictive-job keys.
    # Data, initialization, module modes, optimizer and stochastic path ARE keys.
    identity = dict(
        kind="target",
        initialization=state_hash(initial),
        normalization=normalization.identity,
        mask=mask,
        data=[data.role(n).identity for n in ("F", "G")],
        seed=fit_seed,
        updates=UPDATES,
        batch_size=BATCH_SIZE,
        optimizer=OPTIMIZER,
        device="cpu",
        dtype="float32",
        code=source_code_hash(),
        implementation=implementation_hash(),
        labels=list(data.role("F").label_names),
        model_type=f"{type(initial).__module__}.{type(initial).__qualname__}",
        input_schema=initial.input_schema,
    )

    def fit():
        started = time.monotonic()
        batches = [data.read(n, "fit_target") for n in ("F", "G")]
        x, y, obs = [np.concatenate([b[j] for b in batches]) for j in range(3)]
        controller = SelectiveModel(initial, mask).cpu()
        xx = inputs(controller.model, normalization.transform(x))
        yy, mm = torch.tensor(y, dtype=torch.float32), torch.tensor(obs)
        optimizer = torch.optim.AdamW(
            [p for p in controller.parameters() if p.requires_grad], **OPTIMIZER
        )
        initial_state = copy.deepcopy(controller.model.state_dict())
        losses, exposure = [], np.zeros(len(y), dtype=int)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(fit_seed)
            generator = torch.Generator().manual_seed(fit_seed)
            for step in range(1, UPDATES + 1):
                ix = torch.randint(len(y), (BATCH_SIZE,), generator=generator)
                exposure += np.bincount(ix.numpy(), minlength=len(y))
                controller.train()
                optimizer.zero_grad(set_to_none=True)
                loss = masked_bce(controller(**{k: v[ix] for k, v in xx.items()}), yy[ix], mm[ix])
                finite_step(controller, optimizer, loss)
                controller.verify_frozen()
                losses.append(float(loss.detach()))
        controller.eval()
        source.validate(data)
        drift = {}
        for n, owner in controller.model.selectable_modules().items():
            drift[n] = float(
                np.sqrt(
                    sum(
                        float(torch.square(p.detach() - initial_state[f"{n}.{k}"]).sum())
                        for k, p in owner.named_parameters()
                    )
                )
            )
        return pack(
            dict(
                state=controller.model.state_dict(),
                normalization=normalization.__dict__,
                identity=identity,
                mask=mask,
                selected_update=UPDATES,
                completed_updates=UPDATES,
                history=losses,
                final_training_loss=losses[-1],
                drift=drift,
                state_hash=state_hash(controller.model),
                source_hash=source.checkpoint_hash,
                label_cost=data.label_cost(),
                fitting_exposures=dict(
                    F=int(exposure[: budget // 2].sum()), G=int(exposure[budget // 2 :].sum())
                ),
                runtime_seconds=time.monotonic() - started,
            )
        )

    path, alias = log.execute(slot, identity, fit, retry=retry)
    saved = unpack(path)
    model = copy.deepcopy(initial)
    model.load_state_dict(saved["state"])
    model.eval()
    source.validate(data)
    return model, dict(
        artifact=str(path),
        sha=file_hash(path),
        alias=alias,
        method=method,
        decision_hash=decision.identity if decision else None,
        random=random_info,
        **{
            k: saved[k]
            for k in (
                "mask",
                "final_training_loss",
                "selected_update",
                "completed_updates",
                "drift",
                "label_cost",
                "fitting_exposures",
                "runtime_seconds",
                "state_hash",
                "normalization",
            )
        },
    )
