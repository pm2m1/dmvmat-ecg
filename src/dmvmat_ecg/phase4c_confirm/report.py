"""Prewritten reporting and one aggregate figure; no post-outcome selection."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .contract import BUDGETS, METHODS, ROOT
from .statistics import THRESHOLD

LABELS = {
    "M1": "Original source-only",
    "M2": "Original full FT",
    "M3": "Plain retrain + FT",
    "M4": "Placebo MMD + FT",
    "M5": "Aligned MMD + FT",
}


def number(v):
    return "N/A" if v is None else f"{v:.6f}"


def interval(ci):
    return (
        "unavailable"
        if ci["interval"] is None
        else f"[{ci['interval'][0]:+.6f}, {ci['interval'][1]:+.6f}]"
    )


def write(result, provenance):
    doc = ROOT / "docs/phase4/PHASE4C_RESULTS.md"
    if doc.exists():
        raise FileExistsError("Existing result report is preserved")
    figdir = ROOT / "figures/phase4"
    figdir.mkdir(parents=True, exist_ok=True)
    figure = figdir / "phase4c_macro_ap_vs_label_budget.svg"
    if figure.exists():
        raise FileExistsError("Existing figure is preserved")
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 10})
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for m in METHODS:
        ax.plot(
            BUDGETS,
            [result["methods"][f"{m}/B{b}"]["macro"]["AP"] for b in BUDGETS],
            marker="o",
            label=f"{m}: {LABELS[m]}",
        )
    ax.set(
        xticks=BUDGETS,
        xlabel="Nominal target label budget (M1 uses zero labels)",
        ylabel="Mean episode-wise CD/STTC macro AP",
        title="Phase4C CPSC test evaluation — fixed fitted models",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure)
    plt.close(fig)
    lines = [
        "# Phase4C — committed confirmatory evaluation",
        "",
        f"Protocol commit: **{provenance['protocol']['commit']}**. Protocol SHA-256: {provenance['protocol']['sha256']}.",
        "",
        "This was the single pre-registered **Phase4C** test evaluation. The same locked cohort had been evaluated previously in Experiment4; this is not globally untouched external validation. No test-based tuning or analysis changes occurred. No source was retrained and no hyperparameters were tuned. The108 fixed200-update target fits were regenerated solely for deterministic state authentication, as authorized by the committed protocol; this is not a claim that no optimization was performed.",
        "",
        f"Replay matches: **{provenance['replay_matches']}/108 fits**, both A and B bitwise; corresponding state/history/recipe bindings match. Source checkpoints were existing and hash-verified. No target checkpoints were written.",
        "",
        f"All135 logical predictions /111 physical vectors were frozen before labels. Prediction manifest: {provenance['manifest_path']}. SHA-256: **{provenance['manifest_sha256']}**. Manifest UTC timestamp: {provenance['manifest_timestamp']}.",
        "",
        f"Records: **{result['records']}**. Labels are recorded-code CD/STTC annotation-union proxies, not adjudicated diagnoses. V2 validity gate: PASS. Observed/unknown and class denominators are given below. Unknowns were not converted to negatives.",
        "",
        "## Confirmatory sequence — macro AP at128 labels",
        "",
        "| Order | Contrast | Difference | Paired95% CI | Valid draws | Confirmatory eligibility | Verdict |",
        "|---|---|---:|---|---:|---|---|",
    ]
    pairs = {"primary": "M5 minus M3", "S1": "M4 minus M3", "S2": "M5 minus M4"}
    for row in result["confirmatory_sequence"]:
        lines.append(
            f"| {row['name']} | {pairs[row['name']]} | {number(row['estimate'])} | {interval(row['CI95'])} | {row['CI95']['valid']} | {row['eligible']} | **{row['status']}** |"
        )
    lines += [
        "",
        "Success requires lower95% CI bound >0 and at least9500 supported draws. S1 requires primary success; S2 requires both primary and S1 success. A closed gate makes later evidence descriptive only even if its interval excludes zero. No Holm family or alternative test was used. Failure to establish benefit is not proof of equivalence or no effect.",
        "",
        "## Main table",
        "",
        "Equal means of nine episode-wise metrics, not probability ensembles or nine independent cohorts. All64/256 results, AUROC and per-label results are descriptive. M1 aliases three source models across draws/budgets and uses zero target labels.",
        "",
        "| Method | Budget | Macro AP | Macro AUROC | CD AP | CD AUROC | STTC AP | STTC AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in METHODS:
        for b in BUDGETS:
            r = result["methods"][f"{m}/B{b}"]
            v = [
                r["macro"]["AP"],
                r["macro"]["AUROC"],
                r["per_label"][0]["AP"],
                r["per_label"][0]["AUROC"],
                r["per_label"][1]["AP"],
                r["per_label"][1]["AUROC"],
            ]
            lines.append(f"| {m}: {LABELS[m]} | {b} | " + " | ".join(map(number, v)) + " |")
    lines += [
        "",
        "![Macro AP versus label budget](../../figures/phase4/phase4c_macro_ap_vs_label_budget.svg)",
        "",
        "Figure: mean fitted-episode macro AP at the three fixed label budgets; lines aid reading, not interpolation to untested budgets. No interval shading or invented uncertainty. M1 has zero adaptation labels regardless of nominal budget.",
        "",
        "## Descriptive threshold metrics at128 labels",
        "",
        "Threshold0.5, units0–1; strict label/episode means. N/A is mathematically undefined, never measured zero.",
        "",
        "| Method | Accuracy | Precision | Recall | Specificity | F1 | Balanced accuracy | NPV |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in METHODS:
        r = result["methods"][f"{m}/B128"]["macro"]
        lines.append(f"| {m} | " + " | ".join(number(r[k]) for k in THRESHOLD) + " |")
    lines += [
        "",
        "### Per-label threshold metrics at128",
        "",
        "| Method | Label | Accuracy | Precision | Recall | Specificity | F1 | Balanced accuracy | NPV |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in METHODS:
        for r in result["methods"][f"{m}/B128"]["per_label"]:
            lines.append(
                f"| {m} | {r['label']} | " + " | ".join(number(r[k]) for k in THRESHOLD) + " |"
            )
    lines += [
        "",
        "## Descriptive paired contrasts",
        "",
        "Pointwise CIs conditional on fixed models. Only the eligible128-label macro-AP sequence above is confirmatory; every other entry below is descriptive.",
        "",
        "| Contrast / budget | Macro AP difference [95% CI] | Macro AUROC difference [95% CI] |",
        "|---|---|---|",
    ]
    for key, r in result["comparisons"].items():
        lines.append(
            f"| {key} | {number(r['AP']['estimate'])} {interval(r['AP']['CI95'])} | {number(r['AUROC']['estimate'])} {interval(r['AUROC']['CI95'])} |"
        )
    lines += [
        "",
        "All intervals use10000 shared Experiment4 canonical-record/known-group resamples, seed78701, tie-aware weighted rankings and no redrawing. Unsupported draw counts are retained in the private analysis. Intervals omit training/selection uncertainty and unknown patient linkage. S2's domain substitution includes BN effects and does not isolate the MMD gradient alone.",
        "",
        "## Outcome denominators",
        "",
        "| Label | Observed | Unknown | Positive proxy | Negative proxy |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in next(iter(result["per_physical"].values()))["per_label"]:
        lines.append(
            f"| {r['label']} | {r['observed']} | {r['unknown']} | {r['positive']} | {r['negative']} |"
        )
    lines += [
        "",
        "## Per-fitted-vector integer confusion counts",
        "",
        "Source/draw numbers are experimental RNG seeds, not participant IDs. M1 vectors are explicitly aliased across nominal budgets/draws. Counts below describe individual fitted vectors, not mean fractional confusion matrices.",
        "",
        "| Predictor | Label | TP | TN | FP | FN |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key, r in result["per_physical"].items():
        for label in r["per_label"]:
            c = label["confusion"]
            lines.append(
                f"| {key} | {label['label']} | {c['TP']} | {c['TN']} | {c['FP']} | {c['FN']} |"
            )
    lines += [
        "",
        "Raw signals, labels, IDs, predictions, group maps, fitted states and operational manifests/receipts remain private and gitignored. Only the requested aggregate report/figure were produced here; no commit, push or merge. Scientific conclusions are limited to annotation-proxy classification under the committed conditional evaluation design.",
    ]
    doc.write_text("\n".join(lines) + "\n")
