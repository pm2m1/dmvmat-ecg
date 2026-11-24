"""Paired source-seed then shared-record bootstrap; validation only."""

import json

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from dmvmat_ecg.phase3b_eval.statistics import bootstrap_rankings, confidence
from dmvmat_ecg.phase4.safety import new_json

from .common import ARMS, BUDGETS, DRAWS, RUN, SEEDS, Scope
from .targets import read, target


def weights(groups, nseeds=10, draws=10000, seed=20261201):
    groups = np.asarray(groups)
    if groups.ndim != 1 or not len(groups) or nseeds < 2 or draws < 1:
        raise ValueError("Invalid hierarchical units")
    rng = np.random.default_rng(seed)
    source = rng.multinomial(nseeds, np.full(nseeds, 1 / nseeds), size=draws)
    units, inverse = np.unique(groups, return_inverse=True)
    records = rng.multinomial(len(units), np.full(len(units), 1 / len(units)), size=draws)
    return source, records[:, inverse]


def combine(per_seed, seed_counts):
    a, c = np.asarray(per_seed), np.asarray(seed_counts)
    if (
        a.ndim != 2
        or a.shape != c.shape
        or (c < 0).any()
        or not np.equal(c, np.floor(c)).all()
        or not (c.sum(1) > 0).all()
    ):
        raise ValueError("Paired replicate-by-seed arrays required")
    return np.sum(np.where(c > 0, a, 0) * c, axis=1) / c.sum(1)


def points(y, p):
    observed = np.isfinite(y)
    out = []
    for j in range(2):
        a, s = y[observed[:, j], j], p[observed[:, j], j]
        if not np.isin(a, [0, 1]).all() or not (a == 0).any() or not (a == 1).any():
            raise ValueError("Observed classes required for validation rankings")
        out.append([average_precision_score(a, s), roc_auc_score(a, s)])
    return np.asarray(out).T


def run(hospital):
    Scope(hospital).install()
    folder = RUN / hospital
    if (folder / "VALIDATION_ANALYSIS.json").exists():
        raise FileExistsError("Completed validation analysis preserved")
    if json.loads((folder / "TARGETS_COMPLETE.json").read_text())["status"] != "COMPLETE":
        raise PermissionError("All targets must finish")
    t = target(hospital)
    y = t.labels("B")
    observed = np.isfinite(y)
    seed_counts, record_counts = weights(t.ids("B"))
    metrics = []
    contrasts = []
    perseed = []
    for b in BUDGETS:
        conditional = {a: [] for a in ARMS}
        point = {a: [] for a in ARMS}
        a_points = {a: [] for a in ARMS}
        for s in SEEDS:
            for arm in ARMS:
                boot = []
                perdraw = []
                A = []
                for d in DRAWS:
                    p, _ = read(t, hospital, arm, s, d, b)
                    perdraw.append(points(y, p["B"]))
                    A.append(points(t.labels("A"), p["A"]))
                    boot.append(bootstrap_rankings(y, p["B"], observed, record_counts))
                point[arm].append(np.mean(perdraw, axis=0))
                a_points[arm].append(np.mean(A, axis=0))
                conditional[arm].append(
                    {k: np.mean([v[k] for v in boot], axis=0) for k in ("AP", "AUROC")}
                )
            delta = point["placebo"][-1] - point["plain"][-1]
            perseed.append(
                dict(
                    source_seed=s,
                    budget=b,
                    macro_AP_difference=float(delta[0].mean()),
                    macro_AUROC_difference=float(delta[1].mean()),
                    CD_AP_difference=float(delta[0, 0]),
                    STTC_AP_difference=float(delta[0, 1]),
                )
            )
            print("PHASE5_BOOTSTRAP", hospital, b, s, flush=True)
        for arm in ARMS:
            a = np.mean(point[arm], axis=0)
            aa = np.mean(a_points[arm], axis=0)
            metrics.append(
                dict(
                    arm=arm,
                    budget=b,
                    macro_AP=float(a[0].mean()),
                    macro_AUROC=float(a[1].mean()),
                    CD_AP=float(a[0, 0]),
                    CD_AUROC=float(a[1, 0]),
                    STTC_AP=float(a[0, 1]),
                    STTC_AUROC=float(a[1, 1]),
                    A_macro_AP=float(aa[0].mean()),
                    A_macro_AUROC=float(aa[1].mean()),
                )
            )
        contrast = dict(budget=b, per_label={})
        delta = np.mean(point["placebo"], axis=0) - np.mean(point["plain"], axis=0)
        for j, metric in enumerate(("AP", "AUROC")):
            # R x seed x label, each seed averaged over the SAME three draws.
            values = np.stack(
                [
                    conditional["placebo"][i][metric] - conditional["plain"][i][metric]
                    for i in range(len(SEEDS))
                ],
                axis=1,
            )
            replica = combine(values.mean(2), seed_counts)
            contrast[metric] = dict(estimate=float(delta[j].mean()), CI95=confidence(replica))
            for label, k in (("CD", 0), ("STTC", 1)):
                contrast["per_label"].setdefault(label, {})[metric] = dict(
                    estimate=float(delta[j, k]),
                    CI95=confidence(combine(values[:, :, k], seed_counts)),
                )
        contrasts.append(contrast)
    result = dict(
        hospital=hospital,
        validation_role="B",
        test_accesses=0,
        seed_count=10,
        draws_per_seed=3,
        metrics=metrics,
        contrasts=contrasts,
        per_seed=perseed,
        bootstrap=dict(
            replicates=10000,
            seed=20261201,
            unit="paired source seeds then shared B records/known groups",
            minimum_valid=9500,
            shared_record_resample_across_all_selected_seeds_draws_arms_labels=True,
            source_training_variability_included=True,
        ),
        observed=observed.sum(0).tolist(),
        unknown=(~observed).sum(0).tolist(),
    )
    new_json(folder / "VALIDATION_ANALYSIS.json", result)
    print("PHASE5_VALIDATION_ANALYSIS_COMPLETE", hospital, flush=True)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("hospital", choices=("CPSC", "Georgia"))
    run(parser.parse_args().hospital)
