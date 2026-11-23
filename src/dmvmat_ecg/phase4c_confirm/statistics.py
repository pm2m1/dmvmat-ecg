"""Only the registered metrics/contrasts; frozen shared bootstrap, no selection."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from dmvmat_ecg.phase3b_eval.statistics import bootstrap_rankings, confidence, shared_weights
from dmvmat_ecg.phase3b_eval_v2.decoder import validity_gate

from .contract import BUDGETS, DRAWS, METHODS, SOURCES, gate_sequence, new_json, save_array

THRESHOLD = ("accuracy", "precision", "recall", "specificity", "F1", "balanced_accuracy", "NPV")


def ratio(a, b):
    return float(a / b) if b else None


def strict_mean(values):
    return float(np.mean(values)) if all(v is not None and np.isfinite(v) for v in values) else None


def measures(y, p, observed):
    y, p, observed = map(np.asarray, (y, p, observed))
    if (
        y.shape != p.shape
        or y.shape != observed.shape
        or y.ndim != 2
        or y.shape[1] != 2
        or observed.dtype != bool
        or not np.isfinite(p).all()
        or ((p < 0) | (p > 1)).any()
    ):
        raise ValueError("Invalid aligned two-label input")
    rows = []
    for j in range(2):
        a, s = y[observed[:, j], j], p[observed[:, j], j]
        if not np.isin(a, [0, 1]).all():
            raise ValueError("Unknowns cannot become negatives")
        z = s >= 0.5
        tp, tn, fp, fn = [
            int(v.sum()) for v in ((a == 1) & z, (a == 0) & ~z, (a == 0) & z, (a == 1) & ~z)
        ]
        recall, spec = ratio(tp, tp + fn), ratio(tn, tn + fp)
        rows.append(
            dict(
                label=("CD", "STTC")[j],
                observed=len(a),
                unknown=int((~observed[:, j]).sum()),
                positive=int((a == 1).sum()),
                negative=int((a == 0).sum()),
                confusion=dict(TP=tp, TN=tn, FP=fp, FN=fn),
                AP=float(average_precision_score(a, s))
                if (a == 1).any() and (a == 0).any()
                else None,
                AUROC=float(roc_auc_score(a, s)) if (a == 1).any() and (a == 0).any() else None,
                accuracy=ratio(tp + tn, len(a)),
                precision=ratio(tp, tp + fp),
                recall=recall,
                specificity=spec,
                F1=ratio(2 * tp, 2 * tp + fp + fn),
                balanced_accuracy=(recall + spec) / 2
                if recall is not None and spec is not None
                else None,
                NPV=ratio(tn, tn + fn),
            )
        )
    return dict(
        per_label=rows,
        macro={k: strict_mean([r[k] for r in rows]) for k in ("AP", "AUROC", *THRESHOLD)},
    )


def aggregate(values):
    return dict(
        macro={
            k: strict_mean([v["macro"][k] for v in values]) for k in ("AP", "AUROC", *THRESHOLD)
        },
        per_label=[
            {
                **{
                    k: values[0]["per_label"][j][k]
                    for k in ("label", "observed", "unknown", "positive", "negative")
                },
                **{
                    k: strict_mean([v["per_label"][j][k] for v in values])
                    for k in ("AP", "AUROC", *THRESHOLD)
                },
            }
            for j in range(2)
        ],
    )


def analyze(y, observed, predictions, roster, groups, folder, draws=10000, minimum=9500):
    gate = validity_gate(y, observed)
    if not gate["passed"]:
        raise PermissionError("Validity gate must pass before metrics")
    unique = {k: measures(y, p, observed) for k, p in predictions.items()}
    episode = {name: unique[key] for name, key in roster.items()}

    def names(m, b):
        return [f"{m}/B{b}/s{s}/d{d}" for s in SOURCES for d in DRAWS]

    table = {
        f"{m}/B{b}": aggregate([episode[n] for n in names(m, b)]) for m in METHODS for b in BUDGETS
    }
    weights = shared_weights(groups, draws=draws, seed=78701)
    save_array(folder / "bootstrap_weights.npy", weights)
    boot = {}
    for i, (key, p) in enumerate(predictions.items(), 1):
        boot[key] = bootstrap_rankings(y, p, observed, weights)
        if i % 5 == 0 or i == len(predictions):
            print(f"BOOTSTRAP_PROGRESS {i}/{len(predictions)}", flush=True)
    comparisons = {}
    for b in BUDGETS:
        for a, z in (("M5", "M3"), ("M4", "M3"), ("M5", "M4"), ("M3", "M2")):
            row = {}
            for metric in ("AP", "AUROC"):
                ar = np.mean([boot[roster[n]][metric] for n in names(a, b)], axis=0)
                zr = np.mean([boot[roster[n]][metric] for n in names(z, b)], axis=0)
                values = ar - zr
                x, zpoint = table[f"{a}/B{b}"], table[f"{z}/B{b}"]
                point = (
                    x["macro"][metric] - zpoint["macro"][metric]
                    if x["macro"][metric] is not None and zpoint["macro"][metric] is not None
                    else None
                )
                row[metric] = dict(
                    estimate=point,
                    CI95=confidence(values.mean(1), minimum_valid=minimum),
                    per_label={
                        label: dict(
                            estimate=x["per_label"][j][metric] - zpoint["per_label"][j][metric]
                            if x["per_label"][j][metric] is not None
                            and zpoint["per_label"][j][metric] is not None
                            else None,
                            CI95=confidence(values[:, j], minimum_valid=minimum),
                        )
                        for j, label in enumerate(("CD", "STTC"))
                    },
                )
            comparisons[f"{a}-minus-{z}/B{b}"] = row
    seq = gate_sequence(
        dict(
            primary=comparisons["M5-minus-M3/B128"]["AP"],
            S1=comparisons["M4-minus-M3/B128"]["AP"],
            S2=comparisons["M5-minus-M4/B128"]["AP"],
        )
    )
    result = dict(
        schema="phase4c-final-analysis-v1",
        records=len(y),
        validity=gate,
        methods=table,
        per_physical=unique,
        episode_aliases=roster,
        comparisons=comparisons,
        confirmatory_sequence=seq,
        bootstrap=dict(
            draws=draws,
            seed=78701,
            minimum_valid=minimum,
            known_groups=len(np.unique(groups)),
            same_weights_all_arms=True,
            conditional_on_models=True,
            no_redrawing=True,
        ),
        threshold=0.5,
    )
    new_json(folder / "ANALYSIS.json", result)
    return result
