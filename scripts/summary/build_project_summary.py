"""New thesis artifacts from existing aggregate JSON/Markdown ONLY.

No project data loaders, models, prediction arrays, ledgers or private receipts
are imported or opened. Arithmetic combines previously reported aggregate values.
"""

import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/summary"
DOC = ROOT / "docs/PROJECT_RESULTS_SUMMARY.md"
BRIEF = ROOT / "docs/SUPERVISOR_BRIEF.md"
PATHS = {
    "ecg": "results/locked_metrics.json",
    "p2": "results/phase2f/BOOTSTRAP_SUMMARIES.json",
    "p2abs": "results/phase2f/AGGREGATE_COMPARISON.json",
    "p2doc": "docs/phase2/INTERNAL_PILOT_LOCKED_EVALUATION_RESULTS.md",
    "p3a": "results/phase3a/METRICS.json",
    "p3": "results/phase3b_eval_v2/AGGREGATE_METRICS.json",
    "ci": "results/phase3b_eval_v2/BOOTSTRAP_SUMMARIES.json",
    "episodes": "results/phase3b_eval_v2/PER_EPISODE_METRICS.json",
    "gap": "results/phase3b_eval_v2/SOURCE_FOLD9_TRANSFER_REFERENCE.json",
    "recovery": "results/phase3b_eval_v2/NEGATIVE_TRANSFER_AND_RECOVERY.json",
    "masks": "results/phase3b/MODULE_EVIDENCE.json",
    "dev": "results/phase3b/DEVELOPMENT_SUMMARY.json",
    "receipt": "results/phase3b_eval_v2/CORRECTED_FINAL_VERIFICATION_RECEIPT.json",
    "headers": "results/phase3b_eval/HEADER_ONLY_SUMMARY.json",
    "k1": "../cvd-70k-multiview-benchmark/docs/results.md",
    "k1ci": "../cvd-70k-multiview-benchmark/results/aggregate/bootstrap_primary.json",
    "k2": "../cvd-70k-multiview-benchmark/docs/v2/OPTIMIZATION_RESULTS.md",
    "k3": "../cvd-70k-multiview-benchmark/docs/v3/PREDICTIVE_OPTIMIZATION.md",
    "k3age": "../cvd-70k-multiview-benchmark/docs/v3/ABLATION_RESULTS.md",
    "k4": "../cvd-70k-v4-synthetic/results/v4/evaluation/summary.json",
    "k4doc": "../cvd-70k-v4-synthetic/docs/v4/FINAL_EVALUATION_REPORT.md",
}
DATA, SOURCES = {}, {}
for alias, relative in PATHS.items():
    p = ROOT / relative
    if not p.exists():
        DATA[alias] = None
        continue
    raw = p.read_bytes()
    SOURCES[alias] = dict(path=relative, sha256=hashlib.sha256(raw).hexdigest())
    DATA[alias] = json.loads(raw) if p.suffix == ".json" else raw.decode()


def pointer(keys):
    return "/" + "/".join(str(k).replace("~", "~0").replace("/", "~1") for k in keys)


def ref(alias, *keys):
    x = DATA[alias]
    for k in keys:
        x = x[k]
    return dict(value=x, path=PATHS[alias], key=pointer(keys))


def combine(points, operation="arithmetic mean"):
    values = [p["value"] for p in points]
    value = float(np.mean(values)) if operation == "arithmetic mean" else float(sum(values))
    return dict(value=value, operation=operation, inputs=points)


def mdrow(alias, row, col):
    for line in DATA[alias].splitlines():
        cells = [s.strip() for s in line.split("|")[1:-1]]
        if cells and cells[0] == row:
            return dict(
                value=float(re.search(r"[-+]?\d*\.?\d+", cells[col]).group()),
                path=PATHS[alias],
                key=f"table[row={row}][column={col}]",
            )
    raise ValueError(f"Missing aggregate table row: {alias}/{row}")


def cite(alias, key=""):
    return f"[{PATHS[alias]}](../{PATHS[alias]})" + (f" → `{key}`" if key else "")


def number(x):
    return f"{x:.5f}"


def ci_text(a):
    return f"[{number(a[0])}, {number(a[1])}]"


def table(header, rows):
    return "\n".join(
        ["|" + "|".join(header) + "|", "|" + "|".join(["---"] * len(header)) + "|"]
        + ["|" + "|".join(map(str, row)) + "|" for row in rows]
    )


def write_new(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        f.write(
            content if isinstance(content, str) else json.dumps(content, indent=2, allow_nan=False)
        )


def main():
    if OUT.exists() or DOC.exists() or BRIEF.exists():
        raise FileExistsError("Do not overwrite existing summaries")
    OUT.mkdir(parents=True)
    figs = OUT / "figures"
    figs.mkdir()
    palette = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    provenance = {}

    def save(fig, name, points, description):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(figs / f"{name}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        provenance[name] = dict(description=description, numerical_data=points)

    kaggle = []
    for version, alias, names in [
        ("V1 fixed test", "k1", ["Five-view MLP", "Flat MLP", "XGBoost"]),
        ("V2 nested internal CV", "k2", ["multiview/selected", "mlp/selected", "xgboost/selected"]),
        (
            "V3 internal folds",
            "k3",
            ["flat_attention_matched", "selected_xgboost", "selected_lightgbm"],
        ),
    ]:
        if DATA[alias] is None:
            kaggle.append([version, "not available locally", "—", "—", "—"])
            continue
        for name in names:
            a, u = mdrow(alias, name, 1), mdrow(alias, name, 2)
            kaggle.append(
                [
                    version,
                    name,
                    number(a["value"]),
                    number(u["value"]),
                    cite(alias, a["key"] + "; AUROC column2"),
                ]
            )
    if DATA["k3age"] is not None:
        a, u = mdrow("k3age", "age_spline", 1), mdrow("k3age", "age_spline", 2)
        kaggle.append(
            [
                "V3 internal folds",
                "age-spline five-view",
                number(a["value"]),
                number(u["value"]),
                cite("k3age", a["key"] + "; AUROC column2"),
            ]
        )
    if DATA["k4"] is not None:
        for name in ("A-age", "NewSelected", "C", "Flat-raw", "XGBoost"):
            r = DATA["k4"]["leaderboard"][name]["raw_fixed_050"]
            kaggle.append(
                [
                    "V4 locked internal contexts",
                    name,
                    number(r["ap"]),
                    number(r["auroc"]),
                    cite("k4", f"/leaderboard/{name}/raw_fixed_050/{{ap,auroc}}"),
                ]
            )

    summary, comparisons = DATA["ecg"]["summary"], DATA["ecg"]["comparisons"]
    comp = comparisons["multiview-minus-flat_cnn"]
    p2 = DATA["p2"]["primary"]
    primary = DATA["ci"]["comparisons"]["B128/A-minus-full_cost"]
    masks = sorted(
        enumerate(DATA["masks"]["episodes"]),
        key=lambda z: (z[1]["budget"], z[1]["source_seed"], z[1]["draw_seed"]),
    )
    modules = [f"view_{i}" for i in range(1, 6)] + ["fusion_1", "fusion_2"]
    fusion_counts = [sum(e["mask"][m] == "ADAPT" for _, e in masks) for m in modules[-2:]]
    gap_means = {}
    for label in ("CD-proxy", "STTC-proxy"):
        indices = [i for i, r in enumerate(DATA["gap"]["rows"]) if r["label"] == label]
        gap_means[label] = {
            k: combine([ref("gap", "rows", i, k) for i in indices])
            for k in ("source_fold9_AUROC", "O_AUROC", "O_minus_fold9_AUROC")
        }

    # Figure 1: deliberately heterogeneous stages, never an improvement curve.
    journey = [
        ("Kaggle V3 XGBoost\ninternal CV, binary", mdrow("k3", "selected_xgboost", 2)),
        (
            "PTB-XL five-view\nlocked, five labels",
            ref("ecg", "summary", "multiview", "macro_auroc"),
        ),
        (
            "CPSC Source-only\ntwo proxies",
            ref("p3", "methods", "B128/source_only", "macro", "AUROC"),
        ),
        ("CPSC Full-cost\nB128", ref("p3", "methods", "B128/full_cost", "macro", "AUROC")),
        ("CPSC Full-cost\nB256", ref("p3", "methods", "B256/full_cost", "macro", "AUROC")),
    ]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(range(len(journey)), [v["value"] for _, v in journey], color=palette[:5])
    for i, (_, r) in enumerate(journey):
        ax.text(i, r["value"] + 0.008, f"{r['value']:.3f}", ha="center")
    ax.set(
        xticks=range(len(journey)),
        xticklabels=[n for n, _ in journey],
        ylim=(0, 1.03),
        ylabel="AUROC (binary or label-macro)",
        title="Project journey — different tasks/populations; not like-for-like gains",
    )
    save(
        fig,
        "project_journey",
        [dict(label=n, **r) for n, r in journey],
        "Kaggle value is descriptive highest selected XGBoost AUROC across historical versions, not a new model choice.",
    )

    fig, ax = plt.subplots(figsize=(7, 4.4))
    points = []
    for j, label in enumerate(gap_means):
        for k, field in enumerate(("source_fold9_AUROC", "O_AUROC")):
            x = j + (k - 0.5) * 0.3
            mean = gap_means[label][field]
            ax.bar(
                x,
                mean["value"],
                width=0.26,
                color=palette[k],
                label=("Source fold9", "CPSC O Source-only")[k] if j == 0 else None,
            )
            vals = [r[field] for r in DATA["gap"]["rows"] if r["label"] == label]
            ax.scatter(x + np.linspace(-0.04, 0.04, len(vals)), vals, color="black", s=22, zorder=3)
            ax.text(x, max(vals) + 0.015, f"{mean['value']:.3f}", ha="center")
            points.append(dict(label=label, field=field, **mean))
    ax.set(
        xticks=[0, 1],
        xticklabels=["CD proxy", "STTC proxy"],
        ylim=(0.5, 1.02),
        ylabel="AUROC",
        title="Transfer gap: means and the three source-seed points",
    )
    ax.legend(loc="lower left")
    save(
        fig, "transfer_gap", points, "Exact per-seed values are the inputs to each documented mean."
    )

    methods = [
        "source_only",
        "head_cost",
        "full_cost",
        "random_a",
        "selective_a",
        "target_only_cost",
    ]
    display = ["Source-only", "Head-cost", "Full-cost", "Random-A", "Selective-A", "Target-only"]
    budgets = sorted(int(k) for k in DATA["dev"]["budgets"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    points = []
    for i, (method, label) in enumerate(zip(methods, display, strict=True)):
        for ax, metric in zip(axes, ("AP", "AUROC"), strict=True):
            vals = []
            for b in budgets:
                p = ref("p3", "methods", f"B{b}/{method}", "macro", metric)
                vals.append(p["value"])
                points.append(dict(label=f"B{b}/{method}/{metric}", **p))
                selected = (
                    [n for n in DATA["episodes"]["procedures"] if n.startswith("source/")]
                    if method == "source_only"
                    else [
                        n for n in DATA["episodes"]["procedures"] if n.endswith(f"/B{b}/{method}")
                    ]
                )
                ep = [ref("episodes", "procedures", n, "macro", metric) for n in selected]
                assert abs(np.mean([v["value"] for v in ep]) - p["value"]) < 1e-12
                xs = b + (i - 2.5) * 1.25
                ax.vlines(
                    xs,
                    min(v["value"] for v in ep),
                    max(v["value"] for v in ep),
                    color=palette[i],
                    alpha=0.45,
                    lw=3,
                )
                points.extend(
                    dict(label=f"{b}/{n}/{metric}", **v) for n, v in zip(selected, ep, strict=True)
                )
            ax.plot(budgets, vals, marker="o", color=palette[i], label=label)
    for ax, metric in zip(axes, ("AP", "AUROC"), strict=True):
        ax.set(
            xlabel="Total target records available for labels (F∪G)",
            ylabel=f"Mean episode macro {metric}",
            xticks=budgets,
        )
    axes[1].legend(fontsize=8)
    fig.suptitle("CPSC locked O: means with episode min–max ranges (not confidence intervals)")
    save(
        fig,
        "label_efficiency",
        points,
        "Nine adapted episodes; three unique Source-only seeds repeated by design. Budget keys from DEVELOPMENT_SUMMARY.json/budgets.",
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharex=True, sharey=True)
    forest_points = []
    contrasts = ["full_cost", "head_cost", "random_a", "source_only", "target_only_cost"]
    for ax, b in zip(axes, budgets, strict=True):
        for j, control in enumerate(contrasts):
            key = f"B{b}/A-minus-{control}"
            r = DATA["ci"]["comparisons"][key]
            bounds = r["CI95"]["interval"]
            ax.hlines(j, *bounds, color=palette[0], lw=2)
            ax.plot(
                r["delta_AP"],
                j,
                "D" if (b == 128 and j == 0) else "o",
                color=palette[4] if j == 0 else palette[0],
            )
            if b == 128 and j > 0:
                ax.hlines(j, *r["Bonferroni_CI9875"]["interval"], color=palette[1], lw=5, alpha=0.5)
                forest_points.append(ref("ci", "comparisons", key, "Bonferroni_CI9875", "interval"))
            forest_points.extend(
                [
                    ref("ci", "comparisons", key, "delta_AP"),
                    ref("ci", "comparisons", key, "CI95", "interval"),
                ]
            )
        ax.axvline(0, color="black", ls="--", lw=1)
        ax.set(
            title=f"B{b}" + (" — primary budget" if b == 128 else " — exploratory"),
            xlabel="A minus comparator: Δ macro AP",
            yticks=range(5),
            yticklabels=["Full-cost", "Head-cost", "Random-A", "Source-only", "Target-only"],
        )
    axes[0].invert_yaxis()
    fig.suptitle(
        "Paired conditional intervals: blue 95%; orange 98.75% secondary family\nRed diamond: registered primary A−Full-cost at B128"
    )
    save(
        fig,
        "contrast_forest",
        forest_points,
        "All adopted contrasts and budgets. Bonferroni intervals only for the four B128 secondary contrasts.",
    )

    fig, ax = plt.subplots(figsize=(8, 4.4))
    ep = primary["episodes"]
    values = [r["delta_AP"] for r in ep]
    ax.scatter(range(len(ep)), values, color=palette[0], label="Paired episode difference")
    low, high = primary["CI95"]["interval"]
    ax.axhspan(
        low, high, color=palette[1], alpha=0.25, label="95% CI for equal mean (not episode CIs)"
    )
    ax.axhline(primary["delta_AP"], color=palette[4], label="Equal mean")
    ax.axhline(0, color="black", ls="--")
    ax.set(
        xticks=range(len(ep)),
        xticklabels=[f"{r['source_seed']}\n{r['draw_seed']}" for r in ep],
        xlabel="Source seed / target draw seed",
        ylabel="Δ mean(CD, STTC) raw AP",
        title="Primary B128: all paired A−Full-cost episode effects",
    )
    ax.legend(fontsize=8, loc="upper right")
    points = [
        ref("ci", "comparisons", "B128/A-minus-full_cost", "episodes", i, "delta_AP")
        for i in range(len(ep))
    ]
    save(
        fig,
        "primary_episodes",
        points
        + [
            ref("ci", "comparisons", "B128/A-minus-full_cost", "delta_AP"),
            ref("ci", "comparisons", "B128/A-minus-full_cost", "CI95", "interval"),
        ],
        "Seed identities are in the corresponding episode objects, not patient identifiers.",
    )

    matrix = np.array([[e["mask"][m] == "ADAPT" for m in modules] for _, e in masks], int)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(matrix, cmap=ListedColormap(["#E5E5E5", palette[0]]), vmin=0, vmax=1, aspect="auto")
    ax.set(
        xticks=range(7),
        xticklabels=["Limb", "V1–V3", "V4–V6", "Temporal", "Frequency", "Fusion 1", "Fusion 2"],
        yticks=range(len(masks)),
        yticklabels=[f"B{e['budget']} · {e['source_seed']}/{e['draw_seed']}" for _, e in masks],
        title="Frozen module masks: ADAPT (blue), SHARE (grey)",
    )
    for i in range(9, len(masks), 9):
        ax.axhline(i - 0.5, color="black", lw=1.5)
    ax.legend(
        handles=[Patch(color=palette[0], label="ADAPT"), Patch(color="#E5E5E5", label="SHARE")],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.07),
        ncol=2,
    )
    mask_inventory = [
        dict(
            source_path=PATHS["masks"],
            source_key=f"/episodes/{i}/mask",
            source_seed=e["source_seed"],
            draw_seed=e["draw_seed"],
            budget=e["budget"],
            mask=e["mask"],
            ADAPT_count=sum(v == "ADAPT" for v in e["mask"].values()),
        )
        for i, e in masks
    ]
    write_new(OUT / "MASK_INVENTORY.json", mask_inventory)
    save(
        fig,
        "mask_heatmap",
        mask_inventory,
        "All registered masks, no outcome-selected subset. Zero/one colour encodes SHARE/ADAPT, not numeric evidence strength.",
    )

    fig, ax = plt.subplots(figsize=(8, 3.5))
    for i, (point, bounds) in enumerate(
        [
            (p2["ap"]["difference"], [p2["ap"]["ci95"]["lower"], p2["ap"]["ci95"]["upper"]]),
            (primary["delta_AP"], primary["CI95"]["interval"]),
        ]
    ):
        ax.hlines(i, *bounds, color=palette[i], lw=2)
        ax.plot(point, i, "o", color=palette[i])
    ax.axvline(0, color="black", ls="--")
    ax.set(
        yticks=[0, 1],
        yticklabels=[
            "Phase 2 age pilot\nF512; total target labels differ",
            "Phase 3B CPSC\nB128 total target records",
        ],
        xlabel="A−Full-cost Δ mean raw AP (95% conditional CI)",
        title="Two negative primary findings — different tasks/designs; not pooled",
    )
    save(
        fig,
        "phase2_vs_phase3b",
        [
            ref("p2", "primary", "ap"),
            ref("ci", "comparisons", "B128/A-minus-full_cost", "delta_AP"),
            ref("ci", "comparisons", "B128/A-minus-full_cost", "CI95", "interval"),
        ],
        "No cross-study hypothesis test or meta-analysis; label budgets and source replication differ.",
    )

    claims = []

    def claim(n, status, text, evidence):
        claims.append(dict(claim=n, status=status, finding=text, evidence=evidence))

    g = gap_means
    claim(
        1,
        "VERIFIED",
        "; ".join(
            f"{lab}: {number(r['source_fold9_AUROC']['value'])} → {number(r['O_AUROC']['value'])}, DROP {number(r['O_minus_fold9_AUROC']['value'])}"
            for lab, r in g.items()
        )
        + f"; macro AP/AUROC {number(DATA['p3']['methods']['B128/source_only']['macro']['AP'])}/{number(DATA['p3']['methods']['B128/source_only']['macro']['AUROC'])}",
        cite("gap", "/rows/*/{source_fold9_AUROC,O_AUROC}; arithmetic means")
        + "; "
        + cite("p3", "/methods/B128~1source_only/macro"),
    )
    claim(
        2,
        "VERIFIED",
        "; ".join(
            f"{key}: {number(DATA['p3']['methods'][key]['macro']['AP'])}/{number(DATA['p3']['methods'][key]['macro']['AUROC'])}"
            for key in [
                "B128/source_only",
                "B128/target_only_cost",
                "B128/full_cost",
                "B256/full_cost",
            ]
        ),
        cite("p3", "/methods/{listed keys}/macro/{AP,AUROC}"),
    )
    rec = DATA["recovery"]["recovery"]["B128/full_cost-minus-source_only"]
    claim(
        3,
        "VERIFIED",
        f"Full-cost B128−Source-only: ΔAP {number(rec['delta_AP'])}; ΔAUROC {number(rec['delta_AUROC'])}",
        cite("recovery", "/recovery/B128~1full_cost-minus-source_only"),
    )
    claim(
        4,
        "VERIFIED",
        f"B128 ΔAP {number(primary['delta_AP'])}, CI {ci_text(primary['CI95']['interval'])}; all {len(ep)} episode AP effects negative ({sum(v < 0 for v in values)}/{len(ep)}). B64/B256: "
        + "/".join(
            number(DATA["ci"]["comparisons"][f"B{b}/A-minus-full_cost"]["delta_AP"])
            for b in (64, 256)
        ),
        cite("ci", "/comparisons/B{64,128,256}~1A-minus-full_cost; /episodes"),
    )
    secondary = [DATA["ci"]["comparisons"]["B128/A-minus-" + m] for m in contrasts[1:]]
    claim(
        5,
        "VERIFIED",
        "; ".join(
            f"A−{m}: {number(r['delta_AP'])}; adjusted CI {ci_text(r['Bonferroni_CI9875']['interval'])}"
            for m, r in zip(contrasts[1:], secondary, strict=True)
        )
        + ". A is below Random-A on AP; this is not equivalence.",
        cite("ci", "/comparisons/B128~1A-minus-{head_cost,random_a,source_only,target_only_cost}"),
    )
    claim(
        6,
        "CORRECTED",
        f"view_2 SHARE {len(masks)}/{len(masks)}; Fusion1 ADAPT {fusion_counts[0]}/{len(masks)}, Fusion2 {fusion_counts[1]}/{len(masks)}. Both always ADAPT at B128/B256; not at B64.",
        cite("masks", "/episodes/*/mask; counts derived without labels"),
    )
    claim(
        7,
        "VERIFIED",
        f"Phase2 ΔAP {number(p2['ap']['difference'])}, CI {ci_text([p2['ap']['ci95']['lower'], p2['ap']['ci95']['upper']])}",
        cite("p2", "/primary/ap"),
    )
    claim(
        8,
        "VERIFIED",
        f"Locked five-view AUROC/AP {number(summary['multiview']['macro_auroc'])}/{number(summary['multiview']['macro_ap'])}; flat {number(summary['flat_cnn']['macro_auroc'])}/{number(summary['flat_cnn']['macro_ap'])}; adjusted AUROC CI {ci_text(comp['auroc_multiplicity_ci'])}; AP CI includes zero.",
        cite("ecg", "/summary/{multiview,flat_cnn}; /comparisons/multiview-minus-flat_cnn"),
    )
    claim_table = table(
        ["Claim", "Status", "Verified finding / correction", "Aggregate evidence"],
        [[r["claim"], r["status"], r["finding"], r["evidence"]] for r in claims],
    )
    write_new(OUT / "CLAIM_VERIFICATION.json", claims)
    write_new(OUT / "FIGURE_DATA_PROVENANCE.json", provenance)
    write_new(OUT / "SOURCE_INDEX.json", SOURCES)

    ecg_rows = [
        [name, number(r["macro_ap"]), number(r["macro_auroc"]), ci_text(r["auroc_ci95"])]
        for name, r in summary.items()
        if name != "xresnet101_final"
    ]
    performance_rows = [
        [
            b,
            display[i],
            number(DATA["p3"]["methods"][f"B{b}/{m}"]["macro"]["AP"]),
            number(DATA["p3"]["methods"][f"B{b}/{m}"]["macro"]["AUROC"]),
        ]
        for b in budgets
        for i, m in enumerate(methods)
    ]
    contrast_rows = []
    for key, r in DATA["ci"]["comparisons"].items():
        contrast_rows.append(
            [
                key,
                number(r["delta_AP"]),
                ci_text(r["CI95"]["interval"]),
                ci_text(r["Bonferroni_CI9875"]["interval"])
                if key.startswith("B128/") and r["Bonferroni_CI9875"]
                else "not in B128 secondary family",
            ]
        )
    mask_rows = [
        [
            b,
            number(DATA["dev"]["budgets"][str(b)]["mean_ADAPT"]),
            DATA["dev"]["budgets"][str(b)]["all_SHARE"],
            number(DATA["dev"]["budgets"][str(b)]["module_ADAPT_frequency"]["fusion_1"]),
            number(DATA["dev"]["budgets"][str(b)]["module_ADAPT_frequency"]["fusion_2"]),
        ]
        for b in budgets
    ]
    negative_rows = [
        [key, f"{r['AP_worse_count']}/{r['episodes']}", number(r["mean_delta_AP"])]
        for key, r in DATA["recovery"]["negative_transfer"].items()
    ]
    phase2_rows = [
        [
            name,
            number(r["ap"]["difference"]),
            ci_text([r["ap"]["ci95"]["lower"], r["ap"]["ci95"]["upper"]]),
            ci_text([r["ap"]["ci99"]["lower"], r["ap"]["ci99"]["upper"]]),
        ]
        for name, r in DATA["p2"]["secondary"].items()
    ]
    drop_rows = [
        [domain, number(r["macro_proxy_matched_AUROC_DROP"]), r["status"]]
        for domain, r in DATA["p3a"]["verdicts"].items()
    ]
    source_index = "\n".join(
        f"- **{a}**: {cite(a)}; SHA256 `{s['sha256']}`." for a, s in SOURCES.items()
    )
    caption_sources = "\n".join(
        f"- **{name}**: `{provenance[name]['description']}`. All plotted numbers and their exact source keys are listed in [FIGURE_DATA_PROVENANCE.json](../results/summary/FIGURE_DATA_PROVENANCE.json), key `/{name}/numerical_data`."
        for name in provenance
    )

    text = f"""# Project results summary: multiview ECG annotation classification and selective transfer

## Project in one paragraph

This project investigates whether a structured multiview representation can support useful classification and evidence-guided adaptation. It moves from binary cross-sectional tabular cardiovascular labels to multilabel ECG annotation classification, then compares selective module updates with conventional fine-tuning under matched target-label access. The ECG backbone separates limb, early and late precordial, learned temporal and frequency views, followed by selectable fusion modules and an always-trainable head. The implemented MMD/permutation selector is a re-engineering inspired by DMV-MAT, not proof of faithful replication of every published component. The scientific endpoint is detection/classification of annotations on recorded ECGs, not prediction of future clinical events. Architecture and provenance: [ECG_ARCHITECTURE](ECG_ARCHITECTURE.md), [SELECTOR_IMPLEMENTATION_REPORT](phase2/SELECTOR_IMPLEMENTATION_REPORT.md). This summary reads existing aggregate JSON/Markdown only; no participant-level data, predictions, models or ledgers were opened.

## Chapter 1 — Kaggle: a tabular ranking plateau, not a universal performance ceiling

{table(["Evaluation basis", "Model/procedure", "AP", "AUROC", "Source path and key"], kaggle)}

Historical versions use different evaluation roles: V1 is its fixed held-out test; V2 uses nested grouped internal cross-validation; V3 uses grouped outer contexts with development holdouts; V4 uses locked internal contexts from the original V1 training pool. The scores cannot be subtracted across versions as paired improvements. The selected XGBoost V3 AUROC is the descriptive Kaggle point shown in the journey figure, not a newly chosen deployment model. V3 LightGBM has the slightly larger selected-tree AP point estimate; those metrics need not identify the same tree family. Table rows preserve the source precision, including rounded Markdown values.

The defensible conclusion is **no reproducible broad AP advantage from the multiview structure on these predictors**, not literal absence of every gain. V1 five-view−flat AP has interval {ci_text(DATA["k1ci"]["ap_difference_vs_mlp/standard"]["multiview/standard"]["ci95"])} ({cite("k1ci", "/ap_difference_vs_mlp~1standard/multiview~1standard/ci95")}). V2 optimization improves its matched legacy reference, but multiview−flat AP is inconclusive; its AUROC contrast is positive in that internal comparison ({cite("k2", "paired-comparison table")}). In V4 the prespecified NewSelected−A-age AP contrast is {number(DATA["k4"]["primary"]["ap_difference"])}, CI {ci_text(DATA["k4"]["primary"]["ap_95_ci"])} ({cite("k4", "/primary/{ap_difference,ap_95_ci}")}). Neither numerical embeddings nor modular ensembling established the registered primary AP improvement. This is evidence of a local plateau under tested features/budgets, not an information-theoretic limit or medical impossibility claim.

![Different tasks and evaluation populations](../results/summary/figures/project_journey.png)

## Chapter 2 — PTB-XL locked ECG classification

{table(["Procedure", "Macro AP", "Macro AUROC", "95% AUROC CI"], ecg_rows)}

Source: {cite("ecg", "/summary/<procedure>/{macro_ap,macro_auroc,auroc_ci95}")}. These are locked official fold-10 results, averaging seed-wise metrics, not validation scores or a probability ensemble. The two training seeds are identified in `/runs` and the original report ({cite("ecg", "/runs")}). Five-view−flat ΔAUROC is {number(comp["delta_auroc"])}, with the registered Bonferroni97.5% CI {ci_text(comp["auroc_multiplicity_ci"])}; ΔAP is {number(comp["delta_ap"])}, exploratory95% CI {ci_text(comp["ap_ci95"])} ({cite("ecg", "/comparisons/multiview-minus-flat_cnn")}). Thus there is limited conditional AUROC evidence over the matched flat control, while AP superiority remains inconclusive. The lower adjusted AUROC bound is near zero and omits retraining/selection uncertainty.

The stronger XResNet pipeline remains ahead of five-view on both ranking metrics: see `/comparisons/multiview-minus-xresnet101`. It uses its own registered inference pipeline; the lightweight CNNs use full-record inference. The augmentation variant did not establish an improvement over the unaugmented network (`/comparisons/multiview_aug-minus-multiview`). These observations justify a modular backbone as a research instrument, not a claim that multiview is the best ECG architecture. The original paper's benchmark is context, not an achieved transfer target or an independent paired comparator.

## Chapter 3 — Phase 2 internal age-subgroup pilot

On the newly trained, patient-disjoint source/target age roles, primary A−Full-cost at F512 was ΔAP {number(p2["ap"]["difference"])},95% CI {ci_text([p2["ap"]["ci95"]["lower"], p2["ap"]["ci95"]["upper"]])}; ΔAUROC {number(p2["auroc"]["difference"])},95% CI {ci_text([p2["auroc"]["ci95"]["lower"], p2["auroc"]["ci95"]["upper"]])} ({cite("p2", "/primary/{ap,auroc}")}). The intervals favor Full-cost for these fitted procedures; the draw-specific AP differences differ in direction (`/primary/by_draw`), so the mean is not evidence of uniform episode behavior.

{table(["Secondary contrast", "ΔAP", "95% CI", "Bonferroni99% CI"], phase2_rows)}

Source: {cite("p2", "/secondary/<contrast>/ap")}. Head/random/source and B−A comparisons remain inconclusive; Full-F is a different label-access comparison. Epoch-zero selection frequently retained source weights and source-only already generalized strongly ({cite("p2doc", "Selector, epoch-zero and negative-transfer observations")}). This describes checkpoint behavior, not the cause of failure. At F512 the primary matched access total is1,024 target participants, including F, evidence, stopping and calibration; at F128 it is640. These are not512- or128-total-label conditions ({cite("p2doc", "Label and compute fairness")}).

![Two distinct locked primary findings](../results/summary/figures/phase2_vs_phase3b.png)

## Chapter 4 — Cross-dataset annotation-proxy transfer

### Feasibility informed the target choice

{table(["Target domain", "Five-view proxy-matched macro AUROC DROP", "Recorded feasibility verdict"], drop_rows)}

Source: {cite("p3a", "/verdicts/<domain>/{macro_proxy_matched_AUROC_DROP,status}")}. These earlier FEAS results used historical source checkpoints and proxy-matched fold9 labels. They are not the later freshly trained source models' O results. CPSC was selected after feasibility inspection; this history reduces confirmatory strength. Georgia was also suitable by the adopted rule; Chapman–Ningbo was borderline. A drop can reflect reporting/label mismatch as well as waveform distribution shift.

### Fresh sources show an asymmetric transfer gap

{table(["Label", "Mean fold9 AUROC", "Mean O Source-only AUROC", "Mean O−fold9"], [[label] + [number(r[k]["value"]) for k in ["source_fold9_AUROC", "O_AUROC", "O_minus_fold9_AUROC"]] for label, r in gap_means.items()])}

Source: {cite("gap", "/rows/*; arithmetic means within label across source seeds")}. Fold9 selected source checkpoints, so it is an optimistic development reference. Prevalence differs and AP cannot measure a pure signal-shift effect. All per-seed points remain visible below.

![Transfer gap](../results/summary/figures/transfer_gap.png)

### Complete budget comparison

{table(["Total F∪G budget", "Procedure", "Mean macro AP", "Mean macro AUROC"], performance_rows)}

Source: {cite("p3", "/methods/B<budget>~1<method>/macro/{AP,AUROC}")}. Source-only averages the source-seed metrics; each adapted row averages the paired source×draw episodes. Target-only is the registered scratch-training control, not an optimally tuned target-only benchmark. Budget includes both fitting and selector-evidence records: selective and random masks were determined on G, then all adapted arms fit F∪G. There is no hidden target stopping or calibration set in this design. Terminal checkpoints are fixed; these labels cannot be compared directly with Phase2's larger all-role costs ({cite("p2doc", "Label and compute fairness")}; [Phase3B development report](phase3/PHASE3B_DEVELOPMENT_RESULTS.md)).

![Budget curves with episode ranges](../results/summary/figures/label_efficiency.png)

Full-cost at B128 recovers ΔAP {number(rec["delta_AP"])} and ΔAUROC {number(rec["delta_AUROC"])} over Source-only ({cite("recovery", "/recovery/B128~1full_cost-minus-source_only")}). These descriptive recovery contrasts do not prove clinical gap closure or isolate hospital causality. No interpolation identifies the minimum sufficient label budget.

### Primary and secondary comparisons

{table(["Contrast", "Δ mean raw AP", "Ordinary95% CI", "B128 secondary Bonferroni98.75% CI"], contrast_rows)}

Source: {cite("ci", "/comparisons/<contrast>/{delta_AP,CI95,Bonferroni_CI9875}")}. Only B128 A−Full-cost is primary; other budgets are exploratory. All nine primary episode differences are negative (`/comparisons/B128~1A-minus-full_cost/episodes`). The intervals are conditional paired record/known-group bootstraps with shared whole-label-vector weights, not independent-clinical-population inference. The stored plan uses{DATA["ci"]["draws"]:,} draws, seed{DATA["ci"]["seed"]}; all are valid for the primary result and none were redrawn ({cite("ci", "/{draws,seed}; /comparisons/B128~1A-minus-full_cost/CI95")}). A exceeds head-only and source-only, but underperforms Full-cost and Random-A on AP at the primary budget. This fails the stipulated evidence-guided-selection interpretation.

![All registered contrasts](../results/summary/figures/contrast_forest.png)
![Primary episode effects](../results/summary/figures/primary_episodes.png)

### Masks, abstention, negative transfer and execution

{table(["Budget", "Mean ADAPT modules", "All-SHARE episodes", "Fusion1 ADAPT fraction", "Fusion2 ADAPT fraction"], mask_rows)}

Source: {cite("dev", "/budgets/<budget>/{mean_ADAPT,all_SHARE,module_ADAPT_frequency}")}. All registered masks are preserved in [MASK_INVENTORY.json](../results/summary/MASK_INVENTORY.json), with exact `/episodes/<index>/mask` citations into {cite("masks")}. The early-precordial encoder remains SHARE throughout; fusion ADAPT is common but not universal. All-SHARE fallback remains in the comparisons rather than being discarded. Larger support changes masks, but mask stability is not predictive benefit.

![Every frozen module mask](../results/summary/figures/mask_heatmap.png)

{table(["Budget/procedure", "AP below paired Source-only", "Mean AP−Source-only"], negative_rows)}

Source: {cite("recovery", "/negative_transfer/<procedure>/{AP_worse_count,episodes,mean_delta_AP}")}. Target-only is not counted as negative transfer because it does not initialize from source. These episode counts are descriptive and dependent, not repeated independent trials. Actual fits used{DATA["dev"]["attempts"]}/{DATA["dev"]["ceiling"]};{DATA["dev"]["aliases"]} of{DATA["dev"]["target_logical_procedures"]} target logical procedures alias existing artifacts ({cite("dev", "/{attempts,ceiling,aliases,target_logical_procedures}")}). Alias savings never funded extra search.

## What the results establish

1. Tabular multiview did not establish a broad AP advantage over strong flat/tree controls; the local AUROC plateau does not imply a universal ceiling (Kaggle evidence above).
2. Five-view ECG improves the matched flat comparator's AUROC conditionally, while its AP interval crosses zero; the stronger XResNet remains superior under the registered benchmark comparisons ({cite("ecg", "/comparisons")}).
3. Source-only performance can degrade substantially across annotation-proxy datasets, especially for STTC in CPSC ({cite("gap", "/rows")}).
4. Limited target supervision substantially improves full fine-tuning's ranking over Source-only in CPSC ({cite("recovery", "/recovery")}).
5. Neither locked transfer study supports selective superiority over matched Full-cost; CPSC also supplies evidence against an AP advantage over parameter-matched Random-A ({cite("p2", "/primary")}; {cite("ci", "/comparisons/B128~1A-minus-{full_cost,random_a}")}).

**One-line conclusion:** A useful multiview ECG representation and a real transfer gap were demonstrated, but the tested evidence-guided freezing policy did not improve on fair conventional full fine-tuning.

## Exploratory interpretations — NOT TESTED

“Freezing precordial views hurts STTC transfer”, “this small model does not overfit, so Full-cost wins”, and “the selector needs at least20 labels while freezing helps only below that support” are hypotheses, not established mechanisms. Masks are not randomized per-module interventions; Full-cost differs in many updates; selected representations and geometry do not identify causal effects. The proposed support boundary is not estimated by these budgets. A better AUC on one metric or an increased ADAPT frequency cannot validate those explanations. A selector can detect shift yet select harmful constraints; this remains a possible explanation rather than a proved cause.

## Thesis claims and limitations

The thesis can claim reproducible implementation, bounded matched-label experiments, limited conditional representation benefits, and two negative primary selective-versus-full findings. It cannot claim faithful replication of all DMV-MAT components, selective-transfer superiority, equivalence to random selection, general clinical transfer, diagnostic deployment utility, independently verified disease negatives, or future-event prediction. The distinction between annotation classification and clinical diagnosis must remain explicit in title and abstract.

CinC target patient identities are unknown: bootstrap units are records/known duplicate groups, not verified patients. The study is CPSC-only for adaptation; source-target linkage and near-duplicates are not fully excluded. Codes are annotation proxies, absence is not clinically adjudicated negative, and incomplete harmonization can produce apparent shift. Truncation affected{DATA["headers"]["truncated_first10"]} O recordings;{DATA["headers"]["excluded_short"]} were excluded as short before scoring ({cite("headers", "/{truncated_first10,excluded_short}")}). The retained segment may omit findings annotated on the full tracing. FEAS informed target/design selection; reused PTB-XL benchmarks and validation selection also limit independence. Phase2 O overlaps an earlier audited holdout by790 people, disclosed before evaluation ({cite("p2doc", "Population, provenance and evidence limits")}).

There were{DATA["receipt"]["O_outcome_access_count"]} Phase3B O reads: a failed decoder read produced zero observed labels, then an explicit human-authorized corrected read used the already sealed unchanged predictions. The pre-access validation requirement was prospectively corrected to frozen-reference agreement on711 R records plus live-adapter agreement on3,859 R records; no O feedback informed that change ([V2 disclosure](phase3/PHASE3B_LOCKED_EVALUATION_RESULTS_V2.md), pre-access revision and decoder-validation receipts cited there). The failed V1 outputs are superseded, not deleted. No new predictions or fitting occurred in the correction ({cite("receipt", "/{O_outcome_access_count,corrected_O_access_count,new_inference_runs,new_predictor_fits}")}). Conditional intervals omit retraining and policy-selection uncertainty; seeds are not new clinical cohorts. Neither a negative primary contrast nor an inconclusive secondary contrast is universal proof that selective adaptation never helps.

## Possible future work — not executed

A separately preregistered source/target alignment experiment could test MK-MMD-aligned pretraining with unlabelled target ECGs, then assess module-selection benefit against strong conventional regularization. This untested part must be distinguished from the current selector's MMD evidence; alignment must not silently be added to the existing results. The faithful-paper arm was deferred in the adopted Phase3B protocol ([method gates](phase3/PHASE3B_METHOD_AND_EVIDENCE_GATES.md)). Cross-cohort work first needs defensible label-negative semantics, participant linkage and overlap control. If future-event “prediction” is required, investigate whether resources such as MIMIC-IV-ECG or CODE-15% provide suitable longitudinal outcomes and permissions; neither resource has been audited or evaluated here, and their names alone do not establish future-outcome availability. Write-up can present the negative result without launching another experiment.

## Claim verification

{claim_table}

## Figure provenance and source-file index

Every displayed data value, interval, seed-point input and mask state has an explicit source path/key in [FIGURE_DATA_PROVENANCE.json](../results/summary/FIGURE_DATA_PROVENANCE.json). Arithmetic is limited to averages/ranges/counts of existing aggregate results; no metrics are recomputed from participant-level labels or predictions. The machine-readable source index records source hashes. Sources from adjacent tabular workspaces are read-only references and are not copied or modified; availability outside this checkout is required to rebuild those rows.

{caption_sources}

{source_index}
"""
    extra_gap_rows = []
    for label in ("CD-proxy", "STTC-proxy"):
        rows = [r for r in DATA["gap"]["rows"] if r["label"] == label]
        extra_gap_rows.append(
            [label]
            + [
                number(float(np.mean([r[k] for r in rows])))
                for k in (
                    "source_fold9_AP",
                    "O_AP",
                    "O_minus_fold9_AP",
                    "source_fold9_prevalence",
                    "O_prevalence",
                )
            ]
        )
    gap_detail = table(
        [
            "Label",
            "Fold9 AP",
            "O Source-only AP",
            "O−fold9 AP",
            "Fold9 AP baseline",
            "O AP baseline",
        ],
        extra_gap_rows,
    )
    gap_detail += (
        "\n\nSource: "
        + cite(
            "gap",
            "/rows/*/{source_fold9_AP,O_AP,O_minus_fold9_AP,source_fold9_prevalence,O_prevalence}; means by label",
        )
        + ". AP drops are prevalence-dependent.\n\n"
    )
    text = text.replace("![Transfer gap]", gap_detail + "![Transfer gap]", 1)
    recovery_detail = table(
        ["Budget / recovery contrast", "ΔAP", "ΔAUROC"],
        [
            [k, number(v["delta_AP"]), number(v["delta_AUROC"])]
            for k, v in DATA["recovery"]["recovery"].items()
        ],
    )
    recovery_detail += (
        "\n\nSource: "
        + cite("recovery", "/recovery/<contrast>/{delta_AP,delta_AUROC}")
        + ". All recovery contrasts are descriptive, without additional superiority claims.\n\n"
    )
    text = text.replace(
        "### Primary and secondary comparisons",
        recovery_detail + "### Primary and secondary comparisons",
        1,
    )
    write_new(DOC, text)
    brief = f"""# Supervisor brief — multiview ECG classification and selective transfer

**Recommended thesis emphasis:** evidence-guided selective adaptation under matched target-label budgets, with transparent negative findings. Use “ECG annotation classification/detection”; the project has no future-event endpoint.

**Representation result.** Locked PTB-XL five-view macro AUROC **{number(summary["multiview"]["macro_auroc"])}**, AP **{number(summary["multiview"]["macro_ap"])}**. Flat CNN AUROC **{number(summary["flat_cnn"]["macro_auroc"])}**; XResNet **{number(summary["xresnet101"]["macro_auroc"])}**. Five-view−flat AUROC adjusted97.5% CI **{ci_text(comp["auroc_multiplicity_ci"])}** is positive but near zero at its lower end; the AP CI crosses zero. Thus there is limited conditional benefit over the matched flat control, not architecture supremacy. Source: {cite("ecg", "/summary; /comparisons/multiview-minus-flat_cnn")}.

**Adaptation works; this selector does not beat Full-cost.** CPSC B128 Full-cost AP/AUROC **{number(DATA["p3"]["methods"]["B128/full_cost"]["macro"]["AP"])}/{number(DATA["p3"]["methods"]["B128/full_cost"]["macro"]["AUROC"])}**, versus Source-only **{number(DATA["p3"]["methods"]["B128/source_only"]["macro"]["AP"])}/{number(DATA["p3"]["methods"]["B128/source_only"]["macro"]["AUROC"])}**. Recovery is **+{number(rec["delta_AP"])} AP / +{number(rec["delta_AUROC"])} AUROC** using128 total labelled target records (all F∪G), not128 positive examples. Sources: {cite("p3", "/methods/B128~1{full_cost,source_only}/macro")}; {cite("recovery", "/recovery/B128~1full_cost-minus-source_only")}.

**Two locked negative primary results.** Age pilot A−Full-cost AP **{number(p2["ap"]["difference"])}**,95% CI **{ci_text([p2["ap"]["ci95"]["lower"], p2["ap"]["ci95"]["upper"]])}**; CPSC B128 **{number(primary["delta_AP"])}**,95% CI **{ci_text(primary["CI95"]["interval"])}**. Source: {cite("p2", "/primary/ap")}; {cite("ci", "/comparisons/B128~1A-minus-full_cost")}. A is numerically close to Random-A in CPSC but **worse**, ΔAP **{number(DATA["ci"]["comparisons"]["B128/A-minus-random_a"]["delta_AP"])}**, adjusted98.75% CI **{ci_text(DATA["ci"]["comparisons"]["B128/A-minus-random_a"]["Bonferroni_CI9875"]["interval"])}**; do not write “equivalent to random.” Source: {cite("ci", "/comparisons/B128~1A-minus-random_a")}. Phase2 random comparison is inconclusive. Neither finding is a universal no-benefit theorem.

**Integrity disclosure.** Phase3B O was read twice: first the decoder returned zero observed labels; the corrected human-authorized read used unchanged predictions sealed beforehand. The original markers/reports remain. No retraining or new inference followed outcome access. This is annotation-proxy, record-level CPSC evidence, not external clinical validation; patient IDs are unavailable. Phase2 holdout overlap and FEAS-informed target choice are disclosed in the full summary. Source: {cite("receipt", "/{O_outcome_access_count,new_inference_runs,new_predictor_fits}")}; {cite("p2doc", "Population, provenance and evidence limits")}.

**Three questions for supervision**

1. Should the title explicitly use “annotation classification/detection” rather than “prediction”, given the absence of longitudinal event outcomes?
2. Is a separately designed MK-MMD-aligned pretraining experiment scientifically necessary before submission, or should the thesis present the current negative selector result and defer that untested component?
3. What is the write-up/submission deadline, and which additional analyses are essential within it?

See [full source-cited summary](PROJECT_RESULTS_SUMMARY.md) and [aggregate-only figures/provenance](../results/summary/FIGURE_DATA_PROVENANCE.json). No new training or evaluation was run for this synthesis.
"""
    write_new(BRIEF, brief)
    write_new(
        OUT / "BUILD_RECEIPT.json",
        dict(
            reads="Existing aggregate JSON/Markdown only",
            raw_data_access=False,
            sealed_data_access=False,
            new_training=False,
            new_inference=False,
            new_calibration=False,
            new_metric_computation_from_raw=False,
            figures=list(provenance),
            source_count=len(SOURCES),
            summary_words=len(text.split()),
            brief_words=len(brief.split()),
        ),
    )
    print(
        json.dumps(
            dict(
                figures=list(provenance),
                summary_words=len(text.split()),
                brief_words=len(brief.split()),
            )
        )
    )


if __name__ == "__main__":
    main()
