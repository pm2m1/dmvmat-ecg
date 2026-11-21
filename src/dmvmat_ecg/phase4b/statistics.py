"""Common record resamples, tie-aware AP/AUROC and A-only policy metrics."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from dmvmat_ecg.phase4.analysis import bootstrap_ap


def bootstrap_auc(y, p, weights, chunk_size=250):
    y, p, weights = np.asarray(y), np.asarray(p), np.asarray(weights)
    if (
        y.ndim != 1
        or p.shape != y.shape
        or weights.ndim != 2
        or weights.shape[1] != len(y)
        or not np.isin(y, [0, 1]).all()
        or not np.isfinite(p).all()
        or (weights < 0).any()
        or not np.equal(weights, np.floor(weights)).all()
    ):
        raise ValueError("Invalid AUROC bootstrap shapes")
    order = np.argsort(p, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(p[order]) != 0), len(y) - 1]
    result = np.full(len(weights), np.nan)
    for start in range(0, len(weights), chunk_size):
        w = weights[start : start + chunk_size, order].astype(np.float64)
        positives = np.cumsum(w * y[order], axis=1)[:, ends]
        negatives = np.cumsum(w * (1 - y[order]), axis=1)[:, ends]
        pg = np.diff(positives, axis=1, prepend=0)
        ng = np.diff(negatives, axis=1, prepend=0)
        numerator = (pg * (negatives - ng + 0.5 * ng)).sum(1)
        denom = positives[:, -1] * negatives[:, -1]
        result[start : start + len(w)] = np.divide(
            numerator, denom, out=np.full(len(w), np.nan), where=denom > 0
        )
    return result


def intervals(values, family=1):
    values = np.asarray(values)
    ok = np.isfinite(values)
    if ok.sum() < 9500:
        raise ValueError("Too few valid bootstrap draws")
    low, high = np.percentile(values[ok], [2.5 / family, 100 - 2.5 / family])
    return dict(low=float(low), high=float(high), valid=int(ok.sum()), undefined=int((~ok).sum()))


def measures(y, p):
    return np.stack(
        [average_precision_score(y, p, average=None), roc_auc_score(y, p, average=None)]
    )


def boots(y, p, weights, auc=True):
    ap = np.stack([bootstrap_ap(y[:, j], p[:, j], weights) for j in range(2)], axis=-1)
    if not auc:
        return ap
    au = np.stack([bootstrap_auc(y[:, j], p[:, j], weights) for j in range(2)], axis=-1)
    return np.stack([ap, au], axis=-2)
