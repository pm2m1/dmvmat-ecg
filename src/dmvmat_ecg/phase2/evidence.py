"""Two explicit multi-label geometry policies; MMD is evidence, never a loss."""

from dataclasses import asdict, dataclass

import numpy as np

from .contracts import MODULES, Embeddings, hash_sample, observed_labels


@dataclass(frozen=True)
class EvidenceConfig:
    permutations: int = 999
    bootstraps: int = 4999
    mmd_cap: int = 256
    source_class_cap: int = 256
    minimum: int = 10
    alpha: float = 0.05
    source_d_floor: float = 0.1
    uncertainty_cutoff: float = 0.75
    retention_cutoff: float = 0.8
    confidence_cutoff: float = 1.0
    mmd_seed: int = 24007
    geometry_seed: int = 24011

    def __post_init__(self):
        if not np.isfinite(
            [
                self.alpha,
                self.source_d_floor,
                self.uncertainty_cutoff,
                self.retention_cutoff,
                self.confidence_cutoff,
            ]
        ).all():
            raise ValueError("Finite evidence thresholds required")
        for key in ("permutations", "bootstraps", "mmd_cap", "source_class_cap", "minimum"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 2:
                raise ValueError("Positive integer sample/draw counts ≥2 required")
        if self.permutations < 9 or self.bootstraps < 19 or not 0 < self.alpha < 0.5:
            raise ValueError("Invalid Monte Carlo configuration")
        if self.minimum > self.source_class_cap or self.source_d_floor <= 0:
            raise ValueError("Invalid source geometry support")
        if not 0 < self.uncertainty_cutoff < self.confidence_cutoff <= 1:
            raise ValueError("Invalid confidence thresholds")
        if not 0 < self.retention_cutoff < 1:
            raise ValueError("Invalid retention threshold")


def _array(z):
    z = np.asarray(z, float)
    if z.ndim != 2 or not z.shape[1] or not np.isfinite(z).all():
        raise ValueError("Finite [patient,feature] representations required")
    return z


def permutation_mmd(x, t, *, config=None, seed=None):
    """Biased V-statistic; pooled scalar RMS is fixed under label permutations."""
    cfg = config or EvidenceConfig()
    x, t = _array(x), _array(t)
    if x.shape[1] != t.shape[1]:
        raise ValueError("Representation dimensions differ")
    n, m = len(x), len(t)
    if min(n, m) < 2:
        return dict(valid=False, reason="insufficient_patients", p=1.0)
    if max(n, m) > cfg.mmd_cap:
        raise ValueError("Hash-sample distinct patients before capped MMD")
    pooled = np.concatenate([x, t])
    pooled = pooled - pooled.mean(0)
    radius = float(np.sqrt(np.square(pooled).sum(1).mean()))
    if radius <= 1e-8:
        return dict(valid=False, reason="constant_embeddings", p=1.0)
    pooled /= radius
    norms = np.square(pooled).sum(1)
    distance = np.maximum(norms[:, None] + norms[None] - 2 * pooled @ pooled.T, 0)
    kernel = sum(np.exp(-distance / (2 * b**2)) for b in (0.5, 1, 2, 4)) / 4
    rng = np.random.default_rng(cfg.mmd_seed if seed is None else seed)
    w = np.full((cfg.permutations + 1, n + m), -1 / m)
    w[0, :n] = 1 / n
    for i in range(1, len(w)):
        w[i, rng.permutation(n + m)[:n]] = 1 / n
    values = np.maximum(np.einsum("ij,ij->i", w @ kernel, w), 0)
    null = values[1:]
    # Preserve null samples in the evidence record for reproducible p-value checks.
    valid = bool(np.isfinite(values).all() and np.std(null) > 1e-12)
    return dict(
        valid=valid,
        reason="valid" if valid else "degenerate_null",
        statistic=float(values[0]),
        p=float((1 + (null >= values[0]).sum()) / len(w)),
        null=null.tolist(),
        pooled_radius=radius,
        n_source=n,
        n_target=m,
    )


def holm(p):
    p = np.asarray(p, float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid p-values")
    order = np.argsort(p, kind="stable")
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    result = np.empty_like(p)
    result[order] = np.minimum(1, adjusted)
    return result


def patient_weights(n, draws, seed):
    if n < 1:
        raise ValueError("No evidence patients")
    return np.random.default_rng(seed).multinomial(n, np.full(n, 1 / n), size=draws)


def _contrast(z, y, obs, label, formulation, minimum):
    """Return mean contrast, equal-cell scatter and patient-cell membership."""
    if formulation not in {"A", "B"}:
        raise ValueError("Choose formulation A or B")
    if formulation == "B" and y.shape[1] != 2:
        raise ValueError("B is defined only for the proposed two-label task")
    cells, contrasts, radii = [], [], []
    for c in range(2) if formulation == "B" else [None]:
        pair = []
        for outcome in (0, 1):
            mask = obs[:, label] & (y[:, label] == outcome)
            if c is not None:
                other = 1 - label
                mask &= obs[:, other] & (y[:, other] == c)
            if mask.sum() < minimum:
                return None, dict(
                    reason="insufficient_verified_class_or_joint_cell",
                    counts=[int(k.sum()) for k in cells] + [int(mask.sum())],
                )
            cells.append(mask)
            mean = z[mask].mean(0)
            pair.append(mean)
            radii.append(np.linalg.norm(z[mask] - mean, axis=1).mean())
        contrasts.append(pair[1] - pair[0])
    delta, scatter = np.mean(contrasts, axis=0), float(np.mean(radii))
    if scatter <= 1e-8:
        return None, dict(reason="degenerate_within_class_scatter")
    return (delta, scatter, cells), dict(counts=[int(k.sum()) for k in cells])


def _boot_contrast(projection, cells, weights):
    means, valid = [], np.ones(len(weights), dtype=bool)
    for cell in cells:
        denom = weights[:, cell].sum(1)
        valid &= denom > 0
        means.append(
            np.divide(
                weights[:, cell] @ projection[cell],
                denom,
                out=np.full(len(weights), np.nan),
                where=denom > 0,
            )
        )
    contrast = np.mean([means[i + 1] - means[i] for i in range(0, len(means), 2)], axis=0)
    return contrast, valid


def relative_geometry(x, sy, sm, t, ty, tm, *, label, formulation="A", config=None, weights=None):
    cfg = config or EvidenceConfig()
    x, t = _array(x), _array(t)
    sy, sm = observed_labels(sy, sm)
    ty, tm = observed_labels(ty, tm)
    if (
        x.shape[1] != t.shape[1]
        or len(x) != len(sy)
        or len(t) != len(ty)
        or sy.shape[1] != ty.shape[1]
        or label not in range(sy.shape[1])
    ):
        raise ValueError("Geometry shape/label mismatch")
    s, status_s = _contrast(x, sy, sm, label, formulation, cfg.minimum)
    u, status_t = _contrast(t, ty, tm, label, formulation, cfg.minimum)
    status = dict(valid=False, source=status_s, target=status_t, formulation=formulation)
    if s is None or u is None:
        return {**status, "reason": "insufficient_or_degenerate_geometry"}
    ds, ss, scells = s
    dt, st, tcells = u
    source_d = np.linalg.norm(ds) / ss
    if source_d < cfg.source_d_floor:
        return {**status, "reason": "weak_source_geometry", "source_d": float(source_d)}
    direction = ds / np.linalg.norm(ds)
    retention = float((dt @ direction / st) / source_d)
    if weights is None:
        weights = (
            patient_weights(len(x), cfg.bootstraps, cfg.geometry_seed),
            patient_weights(len(t), cfg.bootstraps, cfg.geometry_seed + 1),
        )
    ws, wt = weights
    for w, n in ((ws, len(x)), (wt, len(t))):
        if (
            w.shape != (cfg.bootstraps, n)
            or not np.isfinite(w).all()
            or (w < 0).any()
            or not np.all(w.sum(1) == n)
        ):
            raise ValueError("Invalid patient bootstrap multiplicities")
    bs, sv = _boot_contrast(x @ direction, scells, ws)
    bt, tv = _boot_contrast(t @ direction, tcells, wt)
    bs, bt = bs / ss, bt / st
    valid = sv & tv & (bs > 0.02)
    fraction = float(valid.mean())
    if fraction < 1 - cfg.alpha:
        return {
            **status,
            "reason": "uncertain_source_or_empty_bootstrap_cells",
            "retention": retention,
            "bootstrap_valid_fraction": fraction,
        }
    quantile = 1 - cfg.alpha / (7 * sy.shape[1])
    ratios = bt[valid] / bs[valid]
    upper = float(np.quantile(ratios, quantile))
    return dict(
        valid=True,
        reason="valid",
        source=status_s,
        target=status_t,
        formulation=formulation,
        retention=retention,
        upper=upper,
        quantile=quantile,
        source_d=float(source_d),
        target_d=float(np.linalg.norm(dt) / st),
        bootstrap_valid_fraction=fraction,
    )


def logical_vote(geometry, mmd, cfg):
    if not geometry["valid"]:
        return False
    r, u = geometry["retention"], geometry["upper"]
    return bool(
        u < cfg.uncertainty_cutoff
        or (
            mmd["valid"]
            and mmd["q"] <= cfg.alpha
            and r < cfg.retention_cutoff
            and u < cfg.confidence_cutoff
        )
    )


def analyze(source: Embeddings, target: Embeddings, *, formulation="A", config=None):
    cfg = config or EvidenceConfig()
    if formulation not in {"A", "B"} or (formulation == "B" and source.y.shape[1] != 2):
        raise ValueError("B requires exactly two labels; otherwise choose A")
    if source.role != "source_train" or target.role != "evidence":
        raise PermissionError("Wrong evidence roles")
    if set(source.groups).intersection(target.groups):
        raise PermissionError("Source/target patient overlap")
    if source.checkpoint_hash != target.checkpoint_hash:
        raise PermissionError("Evidence must use the identical source initialization")
    if source.preprocessing_hash != target.preprocessing_hash:
        raise PermissionError("Evidence preprocessing differs")
    if source.y.shape[1] != target.y.shape[1] or source.label_names != target.label_names:
        raise ValueError("Label schema differs")
    sm = np.zeros_like(source.observed)
    # Each label's source analysis mask caps each class, but bootstrap weights
    # are shared on their patient union; other-label masks retain joint evidence.
    for label in range(source.y.shape[1]):
        for outcome in (0, 1):
            ix = np.flatnonzero(source.observed[:, label] & (source.y[:, label] == outcome))
            chosen = hash_sample(
                [source.groups[i] for i in ix],
                cfg.source_class_cap,
                cfg.geometry_seed + label * 2 + outcome,
            )
            sm[ix[chosen], label] = True
    union = sm.any(1)
    if not union.any():
        # Retain records with all-false analysis masks so every geometry abstains.
        union = np.ones(len(sm), dtype=bool)
    sy, sg = source.y[union], tuple(np.asarray(source.groups)[union])
    sampled_sm = sm[union]
    # B needs the other label's verified status, not its independently capped
    # analysis membership. Restore it per label inside the geometry call below.
    weights = (
        patient_weights(len(sy), cfg.bootstraps, cfg.geometry_seed),
        patient_weights(len(target.y), cfg.bootstraps, cfg.geometry_seed + 1),
    )
    si = hash_sample(source.groups, cfg.mmd_cap, cfg.mmd_seed)
    ti = hash_sample(target.groups, cfg.mmd_cap, cfg.mmd_seed + 1)
    result, masks = {}, {}
    for index, module in enumerate(MODULES):
        result[module] = dict(
            mmd=permutation_mmd(
                source.features[module][si],
                target.features[module][ti],
                config=cfg,
                seed=cfg.mmd_seed + index,
            )
        )
    q = holm([result[m]["mmd"]["p"] if result[m]["mmd"]["valid"] else 1 for m in MODULES])
    for module, adjusted in zip(MODULES, q, strict=True):
        evidence = result[module]
        evidence["mmd"]["q"] = float(adjusted)
        geometry = []
        for label in range(sy.shape[1]):
            this_sm = source.observed[union].copy() if formulation == "B" else sampled_sm.copy()
            this_sm[:, label] = sampled_sm[:, label]
            geometry.append(
                relative_geometry(
                    source.features[module][union],
                    sy,
                    this_sm,
                    target.features[module],
                    target.y,
                    target.observed,
                    label=label,
                    formulation=formulation,
                    config=cfg,
                    weights=weights,
                )
            )
        votes = [logical_vote(g, evidence["mmd"], cfg) for g in geometry]
        masks[module] = "ADAPT" if any(votes) else "SHARE"
        evidence.update(
            geometry=geometry,
            label_votes=votes,
            abstained_labels=sum(not g["valid"] for g in geometry),
        )
    return dict(
        mask=masks,
        modules=result,
        configuration=asdict(cfg),
        formulation=formulation,
        source_anchor_groups=list(sg),
        source_class_counts=[
            [int((sampled_sm[:, j] & (sy[:, j] == c)).sum()) for c in (0, 1)]
            for j in range(sy.shape[1])
        ],
        target_patients=len(target.groups),
        target_class_counts=[
            [int((target.observed[:, j] & (target.y[:, j] == c)).sum()) for c in (0, 1)]
            for j in range(target.y.shape[1])
        ],
        target_unknown_counts=(~target.observed).sum(0).tolist(),
        checkpoint_hash=source.checkpoint_hash,
        target_identity=target.data_identity,
    )
