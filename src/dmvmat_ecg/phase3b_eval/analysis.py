"""Registered equal-episode estimands; descriptive additions fixed before O access."""

import io

import numpy as np

from dmvmat_ecg.phase3b.jobs import atomic_new, save_json
from dmvmat_ecg.phase3b.protocol import BUDGETS, DRAW_SEEDS, METHODS, SOURCE_SEEDS

from .statistics import bootstrap_rankings, confidence, contrast, metrics, shared_weights


def safe_difference(a, b):
    return float(a - b) if a is not None and b is not None else None


def mean_metrics(values):
    result = dict(macro={}, per_label=[])
    for key in values[0]["macro"]:
        v = [a["macro"][key] for a in values]
        result["macro"][key] = float(np.mean(v)) if all(x is not None for x in v) else None
    for j in range(2):
        row = {}
        for key in values[0]["per_label"][j]:
            v = [a["per_label"][j][key] for a in values]
            if key in (
                "label",
                "threshold",
                "positive",
                "negative",
                "observed",
                "unknown",
                "prevalence_AP_reference",
            ):
                row[key] = v[0]
            elif key == "confusion":
                row[key] = {n: float(np.mean([k[n] for k in v])) for n in v[0]}
            else:
                row[key] = float(np.mean(v)) if all(x is not None for x in v) else None
        result["per_label"].append(row)
    return result


def analyze(
    y,
    observed,
    predictions,
    names,
    groups,
    folder,
    out,
    *,
    draws=10000,
    bootstrap_seed=78701,
    minimum_valid=9500,
):
    unique = {key: metrics(y, p, observed) for key, p in predictions.items()}
    episode = {name: dict(physical=key, **unique[key]) for name, key in names.items()}
    aggregate = {}
    for b in BUDGETS:
        for method in ("source_only", *METHODS):
            selected = (
                [episode[f"source/{s}"] for s in SOURCE_SEEDS]
                if method == "source_only"
                else [episode[f"{s}/{d}/B{b}/{method}"] for s in SOURCE_SEEDS for d in DRAW_SEEDS]
            )
            aggregate[f"B{b}/{method}"] = dict(
                mean_metrics(selected),
                unique_model_episodes=len(selected),
                estimand="mean episode-wise metrics, no probability ensemble",
            )
    save_json(
        out / "PER_EPISODE_METRICS.json",
        dict(scope="locked O annotation-proxy classification", procedures=episode),
    )
    save_json(
        out / "AGGREGATE_METRICS.json",
        dict(records=len(y), labels=["CD-proxy", "STTC-proxy"], methods=aggregate),
    )
    weights = shared_weights(groups, draws=draws, seed=bootstrap_seed)
    atomic_new(folder / "bootstrap_weights.npy", _npy(weights))
    boot = {}
    for index, (key, p) in enumerate(predictions.items(), 1):
        boot[key] = bootstrap_rankings(y, p, observed, weights)
        print(f"BOOTSTRAP {index}/{len(predictions)}", flush=True)
    point = {
        key: unique[key]["macro"]["AP"] if unique[key]["macro"]["AP"] is not None else np.nan
        for key in predictions
    }
    boot_ap = {key: a["AP"].mean(1) for key, a in boot.items()}
    per_comparison = {}
    comparator_keys = ("full_cost", "head_cost", "random_a", "source_only", "target_only_cost")
    for b in BUDGETS:
        for comparator in comparator_keys:
            values = contrast(names, boot_ap, budget=b, comparator=comparator)
            estimate = float(contrast(names, point, budget=b, comparator=comparator))
            estimate = estimate if np.isfinite(estimate) else None
            episodes = []
            for s in SOURCE_SEEDS:
                for d in DRAW_SEEDS:
                    a = episode[f"{s}/{d}/B{b}/selective_a"]
                    z = (
                        episode[f"source/{s}"]
                        if comparator == "source_only"
                        else episode[f"{s}/{d}/B{b}/{comparator}"]
                    )
                    episodes.append(
                        dict(
                            source_seed=s,
                            draw_seed=d,
                            delta_AP=safe_difference(a["macro"]["AP"], z["macro"]["AP"]),
                            delta_AUROC=safe_difference(a["macro"]["AUROC"], z["macro"]["AUROC"]),
                        )
                    )
            per_comparison[f"B{b}/A-minus-{comparator}"] = dict(
                delta_AP=estimate,
                delta_AUROC=float(np.mean([r["delta_AUROC"] for r in episodes]))
                if all(r["delta_AUROC"] is not None for r in episodes)
                else None,
                CI95=confidence(values, minimum_valid=minimum_valid),
                Bonferroni_CI9875=confidence(values, 0.9875, minimum_valid)
                if comparator != "full_cost"
                else None,
                episodes=episodes,
                primary=(b == 128 and comparator == "full_cost"),
                registered_secondary=(b == 128 and comparator != "full_cost"),
                exploratory=(b != 128),
            )
    # Save conditional resampling results privately for reproduction without rereading outcomes.
    payload = io.BytesIO()
    np.savez_compressed(payload, **{k.replace("/", "_") + "_AP": v["AP"] for k, v in boot.items()})
    atomic_new(folder / "bootstrap_AP_arrays.npz", payload.getvalue())
    save_json(
        out / "BOOTSTRAP_SUMMARIES.json",
        dict(
            draws=draws,
            seed=bootstrap_seed,
            unit="record/known-duplicate group; target patient linkage unknown",
            shared_weights=True,
            no_redrawing=True,
            conditional_on_fitted_models=True,
            comparisons=per_comparison,
        ),
    )
    recovery = {}
    negative_transfer = {}
    target_only_reference = {}
    for b in BUDGETS:
        for method in METHODS:
            differences = []
            auc = []
            for s in SOURCE_SEEDS:
                for d in DRAW_SEEDS:
                    a = episode[f"{s}/{d}/B{b}/{method}"]
                    z = episode[f"source/{s}"]
                    diff = safe_difference(a["macro"]["AP"], z["macro"]["AP"])
                    da = safe_difference(a["macro"]["AUROC"], z["macro"]["AUROC"])
                    differences.append(diff if diff is not None else np.nan)
                    auc.append(da if da is not None else np.nan)
            negative_transfer[f"B{b}/{method}"] = dict(
                episodes=9,
                AP_worse_count=int((np.array(differences) < 0).sum())
                if np.isfinite(differences).all()
                else None,
                AP_worse_fraction=float(np.mean(np.array(differences) < 0))
                if np.isfinite(differences).all()
                else None,
                mean_delta_AP=float(np.mean(differences))
                if np.isfinite(differences).all()
                else None,
                mean_delta_AUROC=float(np.mean(auc)) if np.isfinite(auc).all() else None,
                per_episode_delta_AP=[v if np.isfinite(v) else None for v in differences],
            )
            if method == "target_only_cost":
                target_only_reference[f"B{b}/{method}"] = negative_transfer.pop(f"B{b}/{method}")
            if method in ("head_cost", "full_cost"):
                recovery[f"B{b}/{method}-minus-source_only"] = dict(
                    descriptive_only=True,
                    delta_AP=float(np.mean(differences))
                    if np.isfinite(differences).all()
                    else None,
                    delta_AUROC=float(np.mean(auc)) if np.isfinite(auc).all() else None,
                )
    save_json(
        out / "NEGATIVE_TRANSFER_AND_RECOVERY.json",
        dict(
            negative_transfer=negative_transfer,
            recovery=recovery,
            target_only_relative_reference=target_only_reference,
        ),
    )
    return dict(
        episode=episode,
        aggregate=aggregate,
        comparisons=per_comparison,
        recovery=recovery,
        negative_transfer=negative_transfer,
        unique_metrics=unique,
    )


def _npy(x):
    stream = io.BytesIO()
    np.save(stream, x, allow_pickle=False)
    return stream.getvalue()
