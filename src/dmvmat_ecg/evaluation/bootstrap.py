"""Patient-cluster percentile intervals conditional on the fitted models.

All procedures and paired seeds share the same sampled patient multiplicities.
The estimand is mean seed-wise metric, not a score-averaged ensemble.
"""

import numpy as np


def weighted_ranking(y, p, weights):
    """Exact weighted AUROC and non-interpolated AP, including score ties."""
    order = np.argsort(-p, kind="stable")
    y, p = np.asarray(y)[order], np.asarray(p)[order]
    w = np.asarray(weights, dtype=float)
    if w.ndim == 1:
        w = w[None, :]
    w = w[:, order]
    ends = np.r_[np.flatnonzero(np.diff(p)), len(p) - 1]
    tp = np.cumsum(w * y, axis=1)[:, ends]
    fp = np.cumsum(w * (1 - y), axis=1)[:, ends]
    dp = np.diff(tp, prepend=0, axis=1)
    dn = np.diff(fp, prepend=0, axis=1)
    positives, negatives = tp[:, -1], fp[:, -1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ap = (
            np.sum(dp * np.divide(tp, tp + fp, out=np.zeros_like(tp), where=tp + fp > 0), axis=1)
            / positives
        )
        auc = np.sum(dp * (negatives[:, None] - fp + dn / 2), axis=1) / (positives * negatives)
    return auc, ap


def patient_bootstrap(y, predictions, patients, draws=2000, seed=70501, maximum=20000):
    """predictions maps procedure names to [seeds, records, labels] probabilities."""
    y = np.asarray(y)
    unique, inverse = np.unique(patients, return_inverse=True)
    rng = np.random.default_rng(seed)
    multiplicities = []
    attempted = 0
    while len(multiplicities) < draws and attempted < maximum:
        count = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)))
        w = count[inverse]
        positive = w @ y
        negative = w.sum() - positive
        attempted += 1
        if np.all(positive > 0) and np.all(negative > 0):
            multiplicities.append(w)
    if len(multiplicities) != draws:
        raise ValueError("Insufficient bootstrap samples with both classes in all labels")
    weights = np.array(multiplicities)
    result = {}
    for name, scores in predictions.items():
        scores = np.asarray(scores)
        if scores.ndim != 3 or scores.shape[1:] != y.shape:
            raise ValueError("Expected paired-seed predictions")
        auc = np.zeros((draws, y.shape[1]))
        ap = np.zeros_like(auc)
        for prediction in scores:
            for j in range(y.shape[1]):
                for start in range(0, draws, 100):
                    a, b = weighted_ranking(y[:, j], prediction[:, j], weights[start : start + 100])
                    auc[start : start + 100, j] += a / len(scores)
                    ap[start : start + 100, j] += b / len(scores)
        result[name] = dict(auroc=auc.mean(1), ap=ap.mean(1), per_label_auroc=auc, per_label_ap=ap)
    return result, dict(
        draws=draws,
        rejected_undefined=attempted - draws,
        patients=len(unique),
        seed=seed,
        unit="patient cluster",
        estimand="mean paired-seed metric",
        conditional=True,
    )


def interval(values, confidence=0.95):
    alpha = (1 - confidence) / 2
    return np.quantile(values, [alpha, 1 - alpha]).tolist()
