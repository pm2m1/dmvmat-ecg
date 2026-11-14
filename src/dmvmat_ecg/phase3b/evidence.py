"""A-record-v1 wrapper around unchanged, pure numerical re-engineered evidence."""

import copy
from dataclasses import asdict, dataclass

import numpy as np
import torch

from dmvmat_ecg.phase2.adaptation import state_hash
from dmvmat_ecg.phase2.contracts import MODULES
from dmvmat_ecg.phase2.evidence import (
    EvidenceConfig,
    holm,
    logical_vote,
    permutation_mmd,
    relative_geometry,
)

from .models import inputs
from .protocol import LABELS, digest, ordered
from .roles import array_hash


def configuration(*, fixture=False):
    return EvidenceConfig(
        mmd_seed=78301,
        geometry_seed=78401,
        **(dict(permutations=19, bootstraps=99) if fixture else {}),
    )


@dataclass(frozen=True)
class RepresentationBank:
    values: dict
    y: np.ndarray
    observed: np.ndarray
    groups: tuple
    role: str
    group_kind: str
    patient_linkage: str
    data_identity: str
    checkpoint_hash: str
    normalization_hash: str
    labels: tuple = LABELS

    def __post_init__(self):
        if self.role == "source_train":
            expected = ("patient", "known")
        elif self.role == "G":
            expected = ("record_cluster", "unknown")
        else:
            raise PermissionError("Only source anchors and G may inform module masks")
        if (self.group_kind, self.patient_linkage) != expected or self.labels != LABELS:
            raise ValueError("Incorrect evidence sampling-unit or endpoint semantics")
        n = len(self.groups)
        if len(set(self.groups)) != n or self.y.shape != (n, 2):
            raise ValueError("One representation per canonical sampling unit required")
        if self.observed.shape != self.y.shape or self.observed.dtype != bool:
            raise ValueError("Explicit observation masks required")
        if not np.isin(self.y[self.observed], [0, 1]).all():
            raise ValueError("Invalid observed annotation")
        if set(self.values) != set(MODULES) or any(
            z.ndim != 2 or len(z) != n or not np.isfinite(z).all() for z in self.values.values()
        ):
            raise ValueError("Finite representations for all seven modules required")


def extract(source, normalization, data, role):
    source.validate(data)
    if normalization.identity != source.normalization.identity:
        raise PermissionError("Evidence must use the fitted source normalization")
    x, y, obs = data.read(role, "evidence")
    ix = data.canonical_source_indices() if role == "source_train" else np.arange(len(y))
    r = data.role(role)
    model = copy.deepcopy(source.model).cpu().eval()
    chunks = {n: [] for n in MODULES}
    with torch.no_grad():
        for start in range(0, len(ix), 128):
            batch = ix[start : start + 128]
            reps = model.representations(**inputs(model, normalization.transform(x[batch])))
            for n in MODULES:
                chunks[n].append(reps[n].cpu().numpy())
    if state_hash(source.model) != source.checkpoint_hash:
        raise PermissionError("Source changed during evidence extraction")
    return RepresentationBank(
        {n: np.concatenate(chunks[n]) for n in MODULES},
        y[ix],
        obs[ix],
        tuple(r.groups[i] for i in ix),
        role,
        r.group_kind,
        r.patient_linkage,
        r.identity,
        source.checkpoint_hash,
        normalization.identity,
    )


@dataclass(frozen=True)
class Decision:
    payload: bytes

    @property
    def report(self):
        import json

        return json.loads(self.payload)

    @property
    def identity(self):
        import hashlib

        return hashlib.sha256(self.payload).hexdigest()

    def validate(self, source, data):
        r = self.report
        if (
            r["source_checkpoint"] != source.checkpoint_hash
            or r["normalization"] != source.normalization.identity
            or r["source_identity"] != data.role("source_train").identity
            or r["G_identity"] != data.role("G").identity
            or r["policy"] != "A-record-v1"
        ):
            raise PermissionError("Decision is not bound to this source/G/normalization")
        # Fixture Monte Carlo settings are never silently promoted to the real policy.
        if r["sampling_unit"] != "record_cluster" or r["patient_linkage"] != "unknown":
            raise PermissionError("False patient-linkage claim")
        return r["mask"]


def analyze(source, target, *, fixture=False):
    from .protocol import canonical

    cfg = configuration(fixture=fixture)
    if source.role != "source_train" or target.role != "G":
        raise PermissionError("Wrong evidence roles")
    if (
        source.checkpoint_hash != target.checkpoint_hash
        or source.normalization_hash != target.normalization_hash
        or set(source.groups).intersection(target.groups)
    ):
        raise PermissionError("Evidence initialization/normalization/group mismatch")

    def sample(groups, count, rng_seed):
        chosen = set(ordered(groups, rng_seed)[:count])
        return np.array([i for i, g in enumerate(groups) if g in chosen], dtype=int)

    sm = np.zeros_like(source.observed)
    for j in range(2):
        for c in (0, 1):
            ix = np.flatnonzero(source.observed[:, j] & (source.y[:, j] == c))
            sub = sample(
                tuple(source.groups[i] for i in ix),
                cfg.source_class_cap,
                cfg.geometry_seed + 2 * j + c,
            )
            sm[ix[sub], j] = True
    union = sm.any(1)
    if not union.any():
        union[:] = True
    ns, nt = int(union.sum()), len(target.groups)
    weights = tuple(
        np.random.default_rng(s).multinomial(n, np.full(n, 1 / n), size=cfg.bootstraps)
        for n, s in ((ns, 78401), (nt, 78402))
    )
    si = sample(source.groups, cfg.mmd_cap, 78301)
    ti = sample(target.groups, cfg.mmd_cap, 78302)
    result = {
        n: dict(
            mmd=permutation_mmd(
                source.values[n][si], target.values[n][ti], config=cfg, seed=78301 + i
            )
        )
        for i, n in enumerate(MODULES)
    }
    adjusted = holm([result[n]["mmd"]["p"] if result[n]["mmd"]["valid"] else 1 for n in MODULES])
    mask = {}
    for n, q in zip(MODULES, adjusted, strict=True):
        e = result[n]
        e["mmd"]["q"] = float(q)
        e["geometry"] = [
            relative_geometry(
                source.values[n][union],
                source.y[union],
                sm[union],
                target.values[n],
                target.y,
                target.observed,
                label=j,
                formulation="A",
                config=cfg,
                weights=weights,
            )
            for j in range(2)
        ]
        e["votes"] = [logical_vote(g, e["mmd"], cfg) for g in e["geometry"]]
        e["abstained_labels"] = sum(not g["valid"] for g in e["geometry"])
        mask[n] = "ADAPT" if any(e["votes"]) else "SHARE"
    return Decision(
        canonical(
            dict(
                policy="A-record-v1",
                mask=mask,
                modules=result,
                configuration=asdict(cfg),
                scope="synthetic-reduced-monte-carlo" if fixture else "production-numerics",
                source_checkpoint=source.checkpoint_hash,
                normalization=source.normalization_hash,
                source_identity=source.data_identity,
                G_identity=target.data_identity,
                sampling_unit="record_cluster",
                patient_linkage="unknown",
                labels=LABELS,
                source_unit="canonical-patient",
                target_units=nt,
                source_representation_hash=array_hash(list(source.values.values())),
                target_representation_hash=array_hash(list(target.values.values())),
                source_class_counts=[
                    [int((sm[:, j] & (source.y[:, j] == c)).sum()) for c in (0, 1)]
                    for j in range(2)
                ],
                target_class_counts=[
                    [int((target.observed[:, j] & (target.y[:, j] == c)).sum()) for c in (0, 1)]
                    for j in range(2)
                ],
                target_unknown_counts=(~target.observed).sum(0).tolist(),
                sampling_hash=digest([list(source.groups), list(target.groups)]),
            )
        )
    )
