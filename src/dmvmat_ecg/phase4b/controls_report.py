"""Secondary oracle/control ranks, after all A-only choices and primary analysis."""

import json

import matplotlib
import numpy as np
from sklearn.metrics import average_precision_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dmvmat_ecg.phase4.safety import new_json, sha

from .common import ROOT, RUN, setup
from .statistics import boots, intervals
from .target_fit import key


def ranks(values):
    return np.argsort(np.lexsort((np.arange(128), -np.asarray(values)))) + 1


def report():
    t = setup()
    primary = json.loads((ROOT / "docs/phase4/PHASE4B_AGGREGATES.json").read_text())
    controls = json.loads((RUN / "RECONSTRUCTED_ORIGINAL_CONTROLS.json").read_text())
    binding = json.loads((RUN / "ORIGINAL_CONTROL_REPLAY_BINDING.json").read_text())
    if (
        controls["status"] != "PASS"
        or controls["B_used"]
        or not controls["all_nine_B128_selector_and_random_masks_match_retained_records"]
        or binding["reconstruction_sha256"] != sha(RUN / "RECONSTRUCTED_ORIGINAL_CONTROLS.json")
    ):
        raise PermissionError("Unverified control reconstruction")
    for name, expected in binding["code_sha256"].items():
        if sha(ROOT / name) != expected:
            raise PermissionError("Reconstruction code changed")
    selected = {
        (v["source_seed"], v["draw_seed"]): v for v in controls["rows"] if v["budget"] == 64
    }
    y = t.labels("B")
    weights = (
        np.random.default_rng(20261127)
        .multinomial(500, np.full(500, 1 / 500), size=10000)
        .astype(np.int16)
    )
    rows = []
    point = {n: [] for n in ("full", "selector", "random")}
    sampled = {n: [] for n in point}
    all_b = []
    for s in (15101, 15102, 15103):
        for d in (16101, 16102, 16103):
            a, b, predictions = [], [], {}
            chosen = {
                "full": 127,
                "selector": int(selected[s, d]["selector_mask"], 2),
                "random": int(selected[s, d]["random_mask"], 2),
            }
            for m in range(128):
                name = key(s, d, 64, "oracle", mask=m)
                r = json.loads((RUN / "fits" / (name + ".json")).read_text())
                path = RUN / "predictions" / (name + ".npz")
                if sha(path) != r["predictions_sha256"] or r["status"] != "COMPLETE":
                    raise PermissionError("Oracle prediction integrity differs")
                with np.load(path, allow_pickle=False) as data:
                    p = data["B"]
                a.append(r["A_macro_AP"])
                b.append(float(average_precision_score(y, p, average=None).mean()))
                if m in chosen.values():
                    predictions[m] = p
            ra, rb = ranks(a), ranks(b)
            all_b.append(b)
            row = dict(source_seed=s, draw_seed=d, full_B_rank=int(rb[127]), head_B_rank=int(rb[0]))
            for label, m in chosen.items():
                row[label] = dict(mask=f"{m:07b}", A_rank=int(ra[m]), B_rank=int(rb[m]), B_AP=b[m])
                p = predictions[m]
                point[label].append(average_precision_score(y, p, average=None))
                sampled[label].append(boots(y, p, weights, False))
            rows.append(row)
    means = np.asarray(all_b).mean(0)
    if not np.allclose(
        means, [v["B_AP"] for v in primary["oracle64"]["all_mask_means"]], atol=1e-12, rtol=0
    ):
        raise PermissionError("Control rescore differs from primary oracle results")
    summaries = {}
    for name in ("selector", "random"):
        delta = np.mean(point[name], axis=0) - np.mean(point["full"], axis=0)
        draws = np.mean(sampled[name], axis=0) - np.mean(sampled["full"], axis=0)
        summaries[name] = dict(
            macro_AP=float(np.mean(point[name])),
            difference=float(delta.mean()),
            ci=intervals(draws.mean(-1)),
            per_label={
                label: dict(difference=float(delta[j]), ci=intervals(draws[:, j]))
                for j, label in enumerate(("CD", "STTC"))
            },
        )
    result = dict(
        schema="phase4b-reconstructed-control-ranks-v1",
        B64_original_receipts_available=False,
        all_nine_B128_replay_pairs_match_retained_records=True,
        masks_reconstructed_without_A_B=True,
        secondary_descriptive_analysis=True,
        rows=rows,
        summaries=summaries,
    )
    destination = ROOT / "docs/phase4/PHASE4B_CONTROL_RANKS.json"
    if destination.exists():
        raise FileExistsError("Completed control-rank report cannot be overwritten")
    new_json(destination, result)
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    for name, label, marker, color in (
        ("selector", "Reconstructed selector", "o", "#28627a"),
        ("random", "Reconstructed random", "s", "#ad7738"),
        ("full", "Full fine-tuning", "D", "#a94340"),
    ):
        ax.scatter(
            range(1, 10), [r[name]["B_rank"] for r in rows], marker=marker, color=color, label=label
        )
    ax.set(
        xlabel="Fixed source/draw episode (ordered by source seed, then draw seed)",
        ylabel="Split-B AP rank among 128 masks (1 = highest)",
        title="64-label control masks: production replay, validation only",
        xticks=range(1, 10),
        ylim=(130, 0),
    )
    ax.legend()
    fig.savefig(ROOT / "figures/phase4/phase4b_oracle64_control_ranks.svg")
    plt.close(fig)
    lines = [
        "",
        "## Reconstructed original-source controls at 64 labels",
        "",
        "All nine B128 production selector/random pairs reproduced the retained bundle masks exactly. The same unchanged source/G pipeline then reconstructed B64 masks; no independently retained original B64 receipts were available. These controls are not B128 masks relabeled as B64. No A/B outcomes selected these masks; the following ranks are secondary descriptive B evaluation after primary fitting and A-only selection finished.",
        "",
        "![Control ranks](../../figures/phase4/phase4b_oracle64_control_ranks.svg)",
        "",
        "Ranks are within each episode's 128 masks, not ranks of independent test cohorts. Each control may choose a different mask in each episode.",
        "",
        "| Source | Draw | Selector mask | A / B rank | Random mask | A / B rank | Full B rank | Head B rank |",
        "|---:|---:|---|---|---|---|---:|---:|",
    ]
    for r in rows:
        a, b = r["selector"], r["random"]
        lines.append(
            f"| {r['source_seed']} | {r['draw_seed']} | `{a['mask']}` | {a['A_rank']} / {a['B_rank']} | `{b['mask']}` | {b['A_rank']} / {b['B_rank']} | {r['full_B_rank']} | {r['head_B_rank']} |"
        )
    lines += [
        "",
        "| Reconstructed policy | B macro AP | Difference versus full [pointwise 95% CI] |",
        "|---|---:|---|",
    ]
    for name, r in summaries.items():
        lines.append(
            f"| {name} | {r['macro_AP']:.6f} | {r['difference']:+.6f} [{r['ci']['low']:+.6f}, {r['ci']['high']:+.6f}] |"
        )
    lines += [
        "",
        "### CD/STTC detail for the A-ranked oracle masks",
        "",
        "| Mask | A CD AP | B CD AP | A STTC AP | B STTC AP |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in primary["oracle64"]["top_5"]:
        a, b = r["A_CD_STTC_AP"], r["B_CD_STTC_AP"]
        lines.append(f"| `{r['mask']}` | {a[0]:.6f} | {b[0]:.6f} | {a[1]:.6f} | {b[1]:.6f} |")
    lines += [
        "",
        "Primary A-selected mask, separate-label differences versus full:",
        "",
        "| Label | AP difference [95% CI] |",
        "|---|---|",
    ]
    for name, r in primary["oracle64"]["label_differences"].items():
        lines.append(
            f"| {name} | {r['difference']:+.6f} [{r['ci']['low']:+.6f}, {r['ci']['high']:+.6f}] |"
        )
    with (ROOT / "docs/phase4/PHASE4B_RESULTS.md").open("a") as stream:
        stream.write("\n".join(lines) + "\n")
    print("SECONDARY_CONTROL_RANK_REPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    report()
