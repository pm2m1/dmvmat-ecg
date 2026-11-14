"""Exact AP, conditional paired whole-vector bootstrap and fixed operating metrics."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from dmvmat_ecg.evaluation.bootstrap import weighted_ranking


def ratio(a, b):
    return float(a / b) if b else None


def metrics(y, p, observed):
    y, p, observed = np.asarray(y), np.asarray(p), np.asarray(observed)
    if y.shape != p.shape or observed.shape != y.shape or y.ndim != 2 or y.shape[1] != 2:
        raise ValueError("Aligned two-label outcomes/probabilities/masks required")
    if observed.dtype != bool or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Valid continuous probabilities required")
    result = []
    for j in range(2):
        yy, pp = y[observed[:, j], j], p[observed[:, j], j].astype(float)
        if not np.isin(yy, [0, 1]).all():
            raise ValueError("Unknown annotations cannot become negatives")
        n = len(yy)
        positive = int(yy.sum())
        negative = n - positive
        binary = pp >= 0.5
        tp = int(((yy == 1) & binary).sum())
        fp = int(((yy == 0) & binary).sum())
        tn = int(((yy == 0) & ~binary).sum())
        fn = int(((yy == 1) & ~binary).sum())
        clipped = np.clip(pp, 1e-7, 1 - 1e-7)
        bins = np.minimum((pp * 15).astype(int), 14)
        ece = (
            sum(
                int((bins == b).sum())
                * abs(float(pp[bins == b].mean()) - float(yy[bins == b].mean()))
                for b in range(15)
                if (bins == b).any()
            )
            / n
            if n
            else None
        )
        recall, specificity = ratio(tp, tp + fn), ratio(tn, tn + fp)
        result.append(
            dict(
                label=("CD-proxy", "STTC-proxy")[j],
                observed=n,
                unknown=int((~observed[:, j]).sum()),
                positive=positive,
                negative=negative,
                prevalence_AP_reference=ratio(positive, n),
                AP=float(average_precision_score(yy, pp)) if positive and negative else None,
                AUROC=float(roc_auc_score(yy, pp)) if positive and negative else None,
                Brier=float(np.mean((pp - yy) ** 2)) if n else None,
                log_loss=float(-np.mean(yy * np.log(clipped) + (1 - yy) * np.log1p(-clipped)))
                if n
                else None,
                ECE15=ece,
                mean_predicted=float(pp.mean()) if n else None,
                precision=ratio(tp, tp + fp),
                recall=recall,
                specificity=specificity,
                F1=ratio(2 * tp, 2 * tp + fp + fn),
                balanced_accuracy=(recall + specificity) / 2
                if recall is not None and specificity is not None
                else None,
                confusion=dict(TN=tn, FP=fp, FN=fn, TP=tp),
                threshold=0.5,
            )
        )
    macro = {}
    for key in (
        "AP",
        "AUROC",
        "Brier",
        "log_loss",
        "ECE15",
        "precision",
        "recall",
        "specificity",
        "F1",
        "balanced_accuracy",
    ):
        vals = [r[key] for r in result]
        macro[key] = float(np.mean(vals)) if all(v is not None for v in vals) else None
    return dict(per_label=result, macro=macro)


def shared_weights(groups, draws=10000, seed=78701):
    unique, inverse = np.unique(groups, return_inverse=True)
    counts = np.random.default_rng(seed).multinomial(
        len(unique), np.full(len(unique), 1 / len(unique)), size=draws
    )
    return counts[:, inverse]


def bootstrap_rankings(y, p, observed, weights, chunk=100):
    result = {"AP": np.full((len(weights), 2), np.nan), "AUROC": np.full((len(weights), 2), np.nan)}
    for j in range(2):
        use = observed[:, j]
        if not use.any():
            continue
        yy, pp = y[use, j], p[use, j]
        for start in range(0, len(weights), chunk):
            w = weights[start : start + chunk, use]
            a, b = weighted_ranking(yy, pp, w)
            valid = ((w @ yy) > 0) & ((w @ (1 - yy)) > 0)
            result["AP"][start : start + len(w), j] = np.where(valid, b, np.nan)
            result["AUROC"][start : start + len(w), j] = np.where(valid, a, np.nan)
    return result


def confidence(values, level=0.95, minimum_valid=9500):
    values = np.asarray(values)
    valid = np.isfinite(values)
    n = int(valid.sum())
    bounds = None
    if n >= minimum_valid:
        alpha = (1 - level) / 2
        bounds = np.quantile(values[valid], [alpha, 1 - alpha]).tolist()
    return dict(
        level=level,
        interval=bounds,
        valid=n,
        undefined=int((~valid).sum()),
        minimum_valid=minimum_valid,
        no_redrawing=True,
    )


def contrast(names, macro_ap, *, budget, comparator):
    """Equal mean of nine paired differences, never AP of averaged probabilities."""
    differences = []
    for source in (15101, 15102, 15103):
        for draw in (16101, 16102, 16103):
            a = names[f"{source}/{draw}/B{budget}/selective_a"]
            other = (
                names[f"source/{source}"]
                if comparator == "source_only"
                else names[f"{source}/{draw}/B{budget}/{comparator}"]
            )
            differences.append(macro_ap[a] - macro_ap[other])
    return np.mean(differences, axis=0)
