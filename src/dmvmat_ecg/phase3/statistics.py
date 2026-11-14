"""Record-level exploratory ranking intervals and label-free RBF shift evidence."""

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from dmvmat_ecg.evaluation.bootstrap import weighted_ranking


def ranking(y, p):
    return [
        {
            "auroc": float(roc_auc_score(y[:, j], p[:, j]))
            if len(np.unique(y[:, j])) == 2
            else None,
            "ap": float(average_precision_score(y[:, j], p[:, j])) if y[:, j].sum() else None,
            "prevalence_AP_reference": float(y[:, j].mean()),
            "positive": int(y[:, j].sum()),
            "negative": int(len(y) - y[:, j].sum()),
            "brier": float(brier_score_loss(y[:, j], p[:, j])),
            "mean_predicted": float(p[:, j].mean()),
        }
        for j in range(2)
    ]


def bootstrap(y, score_arrays, *, seed=74311, draws=2000):
    """Mean seed-wise metrics; shared record weights, no invented patient IDs."""
    n = len(y)
    weights = np.random.default_rng(seed).multinomial(n, np.full(n, 1 / n), size=draws)
    result = {m: {"ap": np.zeros((draws, 2)), "auroc": np.zeros((draws, 2))} for m in score_arrays}
    for name, scores in score_arrays.items():
        for p in scores:
            for j in range(2):
                for start in range(0, draws, 100):
                    auc, ap = weighted_ranking(y[:, j], p[:, j], weights[start : start + 100])
                    result[name]["ap"][start : start + 100, j] += ap / len(scores)
                    result[name]["auroc"][start : start + 100, j] += auc / len(scores)
    return result


def percentile(values):
    valid = np.isfinite(values)
    return {
        "lower": float(np.quantile(values[valid], 0.025)) if valid.any() else None,
        "upper": float(np.quantile(values[valid], 0.975)) if valid.any() else None,
        "undefined": int((~valid).sum()),
    }


def prior_adjusted_AP(source_y, source_scores, target_prevalence):
    values = []
    for j in range(2):
        prior = source_y[:, j].mean()
        if not 0 < prior < 1:
            values.append(None)
            continue
        weight = np.where(
            source_y[:, j] == 1,
            target_prevalence[j] / prior,
            (1 - target_prevalence[j]) / (1 - prior),
        )
        values.append(
            float(
                np.mean(
                    [
                        average_precision_score(source_y[:, j], p[:, j], sample_weight=weight)
                        for p in source_scores
                    ]
                )
            )
        )
    return values


def mmd(source, target, *, seed=74321, permutations=999):
    """Single RBF, pooled positive-distance median bandwidth, capped before call."""
    x = np.concatenate([source, target]).astype(float)
    n, k = len(source), len(target)
    norm = np.square(x).sum(1)
    distance = np.maximum(norm[:, None] + norm[None] - 2 * x @ x.T, 0)
    positive = distance[np.triu_indices(len(x), 1)]
    positive = positive[positive > 1e-12]
    if not len(positive):
        return {"valid": False, "reason": "constant pooled embeddings", "statistic": 0.0, "p": 1.0}
    sigma2 = float(np.median(positive))
    kernel = np.exp(-distance / (2 * sigma2))
    weights = np.full((permutations + 1, n + k), -1 / k)
    weights[0, :n] = 1 / n
    rng = np.random.default_rng(seed)
    for i in range(1, len(weights)):
        weights[i, rng.permutation(n + k)[:n]] = 1 / n
    values = np.maximum(np.einsum("ij,ij->i", weights @ kernel, weights), 0)
    return {
        "valid": True,
        "statistic": float(values[0]),
        "p": float((1 + (values[1:] >= values[0]).sum()) / (permutations + 1)),
        "sigma_squared_median": sigma2,
        "source_records": n,
        "target_records": k,
        "permutations": permutations,
        "seed": seed,
    }
