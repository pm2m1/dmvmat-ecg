"""Synthetic-only selector interface. Not a validated ECG transfer algorithm.

Multilabel ECG geometry must be explicitly conditioned on one selected label.
No argmax conversion or unregistered aggregation across labels is allowed.
"""

import numpy as np
from scipy.spatial.distance import cdist, pdist

MODULES = tuple([f"view_{i}" for i in range(1, 6)] + ["fusion_1", "fusion_2"])


def mmd(x, z):
    x, z = np.asarray(x, float), np.asarray(z, float)
    if x.ndim != 2 or z.ndim != 2 or x.shape[1] != z.shape[1]:
        raise ValueError("Aligned representations required")
    pooled = np.r_[x, z]
    radii = pdist(pooled)
    bandwidth = np.median(radii[radii > 0]) if np.any(radii > 0) else 1.0

    def kernel(a, b):
        d = cdist(a, b, "sqeuclidean")
        return np.mean(
            [np.exp(-d / (2 * (bandwidth * scale) ** 2)) for scale in (0.5, 1, 2, 4)], axis=0
        )

    return float(kernel(x, x).mean() + kernel(z, z).mean() - 2 * kernel(x, z).mean())


def permutation_mmd(x, z, permutations=199, seed=101):
    # Keep pooled bandwidth unchanged under permutation by construction.
    observed = mmd(x, z)
    rng = np.random.default_rng(seed)
    pooled = np.r_[x, z]
    null = []
    for _ in range(permutations):
        order = rng.permutation(len(pooled))
        null.append(mmd(pooled[order[: len(x)]], pooled[order[len(x) :]]))
    return dict(mmd=observed, p=(1 + np.sum(np.asarray(null) >= observed)) / (1 + permutations))


def logical_policy(*, valid, uncertainty, retention, p_mmd):
    return (
        "ADAPT"
        if valid and (uncertainty < 0.75 or (p_mmd <= 0.05 and retention < 0.8 and uncertainty < 1))
        else "SHARE"
    )


def binary_geometry(x, labels, *, label_index):
    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.shape[1] != 5 or label_index not in range(5):
        raise ValueError("An explicit ECG label index is mandatory")
    y = labels[:, label_index]
    if any(np.sum(y == k) < 5 for k in (0, 1)):
        return dict(valid=False, reason="Insufficient class support")
    means = [x[y == k].mean(0) for k in (0, 1)]
    within = np.mean([np.linalg.norm(x[y == k] - means[k], axis=1).mean() for k in (0, 1)])
    d = np.linalg.norm(means[1] - means[0]) / max(within, 1e-12)
    return dict(
        valid=bool(np.isfinite(d) and d >= 0.1),
        discriminability=float(d),
        direction=means[1] - means[0],
    )


def apply_mask(model, mask):
    if set(mask) != set(MODULES) or any(v not in ("SHARE", "ADAPT") for v in mask.values()):
        raise ValueError("Seven explicit SHARE/ADAPT decisions required")
    for name, module in model.selectable_modules().items():
        module.requires_grad_(mask[name] == "ADAPT")
        if mask[name] == "SHARE":
            module.eval()  # Frozen normalization buffers matter as well as weights.
    model.prediction_head.requires_grad_(True)


def frozen_snapshot(model, mask):
    return {
        name: {k: v.detach().clone() for k, v in module.state_dict().items()}
        for name, module in model.selectable_modules().items()
        if mask[name] == "SHARE"
    }
