"""Prewritten Georgia reporting: no endpoint or model selection."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dmvmat_ecg.phase4c_confirm.statistics import THRESHOLD
from dmvmat_ecg.phase5.common import BUDGETS, SEEDS

from .contract import METHODS, ROOT

NAMES = {"a": "Plain source-only", "b": "Plain retrain + full FT", "c": "Placebo MMD + full FT"}


def num(v):
    return "N/A" if v is None else f"{v:.6f}"


def ci(row):
    bounds = row["CI95"]["interval"]
    return "unavailable" if bounds is None else f"[{bounds[0]:+.6f}, {bounds[1]:+.6f}]"


def write(r, p):
    path = ROOT / "docs/phase5/PHASE5_GEORGIA_RESULTS.md"
    figpath = ROOT / "figures/phase5/phase5_georgia_test_per_seed_ap_differences.svg"
    if path.exists() or figpath.exists():
        raise FileExistsError("Existing test results preserved")
    figpath.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 9})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    for ax, b in zip(axes, BUDGETS, strict=True):
        rows = [v for v in r["per_seed"] if v["budget"] == b]
        ax.scatter(
            range(10),
            [
                float("nan") if v["macro_AP_difference"] is None else v["macro_AP_difference"]
                for v in rows
            ],
            s=32,
        )
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set(
            title=f"{b} labels",
            xticks=range(10),
            xticklabels=[str(s) for s in SEEDS],
            xlabel="Paired source seed",
        )
        ax.tick_params(axis="x", rotation=60)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Placebo minus plain-retrain macro AP")
    fig.suptitle("Georgia locked test: mean of three fixed draws per source seed")
    fig.tight_layout()
    fig.savefig(figpath)
    plt.close(fig)
    primary = r["primary"]
    lines = [
        "# Phase5 — single pre-registered Georgia test evaluation",
        "",
        f"Protocol commit: **{p['protocol']['commit']}**. Protocol SHA256: `{p['protocol']['sha256']}`. The committed protocol remained unchanged.",
        "",
        f"Prediction manifest: `{p['manifest_path']}`. SHA256: **`{p['manifest_sha256']}`**. UTC timestamp: {p['manifest_timestamp']}. All270 logical cells /190 physical predictors were complete and sealed before the single outcome-access marker.",
        "",
        f"Replay authentication: **{p['replay_matches']}/180** fits, A and B byte-for-byte plus final state,200-update history, source/normalization and recipe checks. Existing20 source checkpoints were hash-verified. No source was retrained, no settings were tuned and no target checkpoint was saved. The180 fixed target optimization replays reconstructed existing states only; this does not claim no optimization was performed.",
        "",
        "## Primary — placebo minus plain full FT at128 labels",
        "",
        f"Mean macro-AP difference: **{num(primary['estimate'])}**. Paired hierarchical95% CI: **{ci(primary)}**. Supported replicates: **{primary['CI95']['valid']}/10000**; undefined: {primary['CI95']['undefined']}. Registered verdict: **{primary['verdict']}**.",
        "",
        "Success requires lower95% bound strictly>0 and at least9500 supported draws. An interval crossing zero does not establish equivalence or no effect. This is the sole confirmatory endpoint; all other findings below are descriptive. The bootstrap resamples paired source seeds first, then one shared canonical-record/known-group sample across selected seeds, arms, draws, labels and budgets. It uses the frozen validation implementation and seed20261201, exact ties, strict means and no redrawing.",
        "",
        "## Main table",
        "",
        "Equal means of30 seed×draw episode metrics, not AP/AUROC of averaged probabilities or30 independent cohorts. Method(a) has10 physical zero-label source predictors aliased across nominal budgets/draws.",
        "",
        "| Method | Budget | Macro AP | Macro AUROC | CD AP | CD AUROC | STTC AP | STTC AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in METHODS:
        for b in BUDGETS:
            v = r["methods"][f"{m}/B{b}"]
            a, z = v["per_label"]
            vals = [v["macro"]["AP"], v["macro"]["AUROC"], a["AP"], a["AUROC"], z["AP"], z["AUROC"]]
            lines.append(f"| ({m}) {NAMES[m]} | {b} | " + " | ".join(map(num, vals)) + " |")
    lines += [
        "",
        "## Descriptive paired contrasts",
        "",
        "Only128-label macro AP above is confirmatory; all64/256 contrasts, AUROC and per-label results are descriptive pointwise intervals.",
        "",
        "| Budget | Macro AP difference [95% CI] | Macro AUROC difference [95% CI] |",
        "|---:|---|---|",
    ]
    for b in BUDGETS:
        v = r["contrasts"][f"c-minus-b/B{b}"]
        lines.append(
            f"| {b} | {num(v['AP']['estimate'])} {ci(v['AP'])} | {num(v['AUROC']['estimate'])} {ci(v['AUROC'])} |"
        )
    lines += [
        "",
        "### Per-label contrasts",
        "",
        "| Budget | Label | AP difference [95% CI] | AUROC difference [95% CI] |",
        "|---:|---|---|---|",
    ]
    for b in BUDGETS:
        v = r["contrasts"][f"c-minus-b/B{b}"]
        for label in ("CD", "STTC"):
            a, z = v["AP"]["per_label"][label], v["AUROC"]["per_label"][label]
            lines.append(
                f"| {b} | {label} | {num(a['estimate'])} {ci(a)} | {num(z['estimate'])} {ci(z)} |"
            )
    lines += [
        "",
        "## Descriptive per-seed effects",
        "",
        "![Per-seed AP differences](../../figures/phase5/phase5_georgia_test_per_seed_ap_differences.svg)",
        "",
        "Points are three-draw mean effects per seed. They are not per-seed confidence intervals. Every one of the ten predeclared seeds remains included.",
        "",
        "| Budget | Seed | Macro AP difference | Macro AUROC difference | CD AP difference | STTC AP difference |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for v in r["per_seed"]:
        lines.append(
            f"| {v['budget']} | {v['source_seed']} | "
            + " | ".join(
                num(v[k])
                for k in (
                    "macro_AP_difference",
                    "macro_AUROC_difference",
                    "CD_AP_difference",
                    "STTC_AP_difference",
                )
            )
            + " |"
        )
    lines += [
        "",
        "## Descriptive fixed0.5 threshold metrics",
        "",
        "Units0–1. These are strict equal-label/equal-episode means. N/A means mathematically undefined, not zero. No cutoff was optimized.",
        "",
        "| Method | Budget | Accuracy | Precision | Recall | Specificity | F1 | Balanced accuracy | NPV |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in METHODS:
        for b in BUDGETS:
            v = r["methods"][f"{m}/B{b}"]["macro"]
            lines.append(f"| {m} | {b} | " + " | ".join(num(v[k]) for k in THRESHOLD) + " |")
    lines += [
        "",
        "### Per-label threshold metrics",
        "",
        "| Method | Budget | Label | Accuracy | Precision | Recall | Specificity | F1 | Balanced accuracy | NPV |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in METHODS:
        for b in BUDGETS:
            for v in r["methods"][f"{m}/B{b}"]["per_label"]:
                lines.append(
                    f"| {m} | {b} | {v['label']} | "
                    + " | ".join(num(v[k]) for k in THRESHOLD)
                    + " |"
                )
    lines += [
        "",
        "## Post-seal outcome support and prevalence",
        "",
        "V2 observed-outcome/both-class gate passed before metrics. Unknowns were never converted to negatives. These test prevalences were first computed after the complete prediction seal.",
        "",
        "| Label | Observed | Unknown | Positive proxy | Negative proxy | Observed prevalence |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for v in next(iter(r["per_physical"].values()))["per_label"]:
        lines.append(
            f"| {v['label']} | {v['observed']} | {v['unknown']} | {v['positive']} | {v['negative']} | {v['positive'] / v['observed']:.6f} |"
        )
    lines += [
        "",
        "## Chronology and limits",
        "",
        "This was the single Phase5 pre-registered Georgia test evaluation:180 authentication replays, signal-only processing, all190 physical predictions sealed, then one outcome-access operation and the fixed analysis. No rerun, tuning, source replacement or endpoint/budget change followed test outcomes. No CPSC test data, previous test outputs or PTB-XL fold10 were accessed.",
        "",
        "The label-blind test assignment excluded the entire old survey subset. Historical Phase3 header/duplicate/mapping audits and original stratification had seen annotations; this is not a claim of never-decoded historical labels. Georgia is a dataset/domain, not proof of homogeneous single-hospital or patient-independent sampling. Outcomes are recorded-code CD/STTC annotation-union proxies, not adjudicated diagnoses or calibrated clinical risk. The ten-seed/shared-record bootstrap includes source-seed randomness but not training-cohort or label-draw selection uncertainty or unknown patient clustering. A positive result would support this fixed recipe/domain comparison, not universal benefit or clinical deployment.",
        "",
        "Per-fitted-vector integer confusion counts, episode aliases, unsupported-draw counts, predictions, raw inputs, IDs and operational bindings remain private and gitignored. Earlier data/results/protocols are unchanged. No commit, push or merge was performed.",
    ]
    path.write_text("\n".join(lines) + "\n")
