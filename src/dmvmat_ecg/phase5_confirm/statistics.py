"""The one Georgia primary and the registered descriptive endpoints only."""

import numpy as np

from dmvmat_ecg.phase3b_eval.statistics import bootstrap_rankings, confidence
from dmvmat_ecg.phase3b_eval_v2.decoder import validity_gate
from dmvmat_ecg.phase4.safety import new_json
from dmvmat_ecg.phase4c_confirm.contract import save_array
from dmvmat_ecg.phase4c_confirm.statistics import aggregate, measures
from dmvmat_ecg.phase5.common import BUDGETS, DRAWS, SEEDS
from dmvmat_ecg.phase5.statistics import combine, weights

from .contract import METHODS


def verdict(point, ci):
    if point is None or ci["interval"] is None or ci["valid"] < 9500:
        return "invalid/incomplete"
    return "success" if ci["interval"][0] > 0 else "not successful"


def analyze(y, observed, predictions, roster, groups, folder, draws=10000, minimum=9500):
    gate = validity_gate(y, observed)
    if not gate["passed"]:
        raise PermissionError("V2 gate must pass before any metric")
    per = {key: measures(y, p, observed) for key, p in predictions.items()}

    def names(m, b, s=None):
        return [f"{m}/B{b}/s{k}/d{d}" for k in (SEEDS if s is None else (s,)) for d in DRAWS]

    def points(m, b, s=None):
        return aggregate([per[roster[n]] for n in names(m, b, s)])

    table = {f"{m}/B{b}": points(m, b) for m in METHODS for b in BUDGETS}
    sc, rc = weights(groups, nseeds=10, draws=draws, seed=20261201)
    save_array(folder / "bootstrap_source_counts.npy", sc)
    save_array(folder / "bootstrap_record_counts.npy", rc)
    boots = {}
    required = [k for k in predictions if not k.startswith("a/")]
    for i, key in enumerate(required, 1):
        boots[key] = bootstrap_rankings(y, predictions[key], observed, rc)
        if i % 5 == 0 or i == len(required):
            print(f"GEORGIA_BOOTSTRAP {i}/{len(required)}", flush=True)
    contrasts = {}
    perseed = []
    for b in BUDGETS:
        left, right = table[f"c/B{b}"], table[f"b/B{b}"]
        row = {}
        for metric in ("AP", "AUROC"):
            conditional = np.stack(
                [
                    np.mean([boots[roster[n]][metric] for n in names("c", b, s)], axis=0)
                    - np.mean([boots[roster[n]][metric] for n in names("b", b, s)], axis=0)
                    for s in SEEDS
                ],
                axis=1,
            )
            point = (
                left["macro"][metric] - right["macro"][metric]
                if left["macro"][metric] is not None and right["macro"][metric] is not None
                else None
            )
            row[metric] = dict(
                estimate=point,
                CI95=confidence(combine(conditional.mean(2), sc), minimum_valid=minimum),
                per_label={
                    label: dict(
                        estimate=left["per_label"][j][metric] - right["per_label"][j][metric]
                        if left["per_label"][j][metric] is not None
                        and right["per_label"][j][metric] is not None
                        else None,
                        CI95=confidence(combine(conditional[:, :, j], sc), minimum_valid=minimum),
                    )
                    for j, label in enumerate(("CD", "STTC"))
                },
            )
        contrasts[f"c-minus-b/B{b}"] = row
        for s in SEEDS:
            a, z = points("c", b, s), points("b", b, s)
            perseed.append(
                dict(
                    source_seed=s,
                    budget=b,
                    macro_AP_difference=a["macro"]["AP"] - z["macro"]["AP"]
                    if a["macro"]["AP"] is not None and z["macro"]["AP"] is not None
                    else None,
                    macro_AUROC_difference=a["macro"]["AUROC"] - z["macro"]["AUROC"]
                    if a["macro"]["AUROC"] is not None and z["macro"]["AUROC"] is not None
                    else None,
                    CD_AP_difference=a["per_label"][0]["AP"] - z["per_label"][0]["AP"]
                    if a["per_label"][0]["AP"] is not None and z["per_label"][0]["AP"] is not None
                    else None,
                    STTC_AP_difference=a["per_label"][1]["AP"] - z["per_label"][1]["AP"]
                    if a["per_label"][1]["AP"] is not None and z["per_label"][1]["AP"] is not None
                    else None,
                )
            )
    primary = contrasts["c-minus-b/B128"]["AP"]
    result = dict(
        schema="phase5-georgia-committed-analysis-v1",
        records=len(y),
        validity=gate,
        methods=table,
        per_physical=per,
        episode_aliases=roster,
        contrasts=contrasts,
        per_seed=perseed,
        primary=dict(**primary, verdict=verdict(primary["estimate"], primary["CI95"])),
        bootstrap=dict(
            replicates=draws,
            seed=20261201,
            minimum_valid=minimum,
            source_seeds=10,
            fixed_draws_per_seed=3,
            known_groups=len(np.unique(groups)),
            source_then_shared_records=True,
            no_redrawing=True,
        ),
        threshold=0.5,
    )
    new_json(folder / "ANALYSIS.json", result)
    return result
