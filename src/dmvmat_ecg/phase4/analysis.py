"""Predeclared A selection / B evaluation; conditional paired record bootstrap."""

import argparse
import json
import time

import matplotlib
import numpy as np
from sklearn.metrics import average_precision_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .runtime import BUNDLE, OUTPUT, ROOT, Bundle, mask_index, prepare_split, split_summary
from .safety import bundle_only_audit, new_json
from .study import check_complete, paths


def best_mask(a):
    """Ascending-index ties, using A only. Never pass B into this function."""
    a = np.asarray(a)
    if a.shape != (128,) or not np.isfinite(a).all():
        raise ValueError("128 finite split-A mask means required")
    return int(np.argmax(a))


def bootstrap_ap(y, p, weights, chunk_size=250):
    """Exact tie-aware AP on resampled records without expanding duplicates.

    weights are common multinomial record counts, not case/control weights.
    Both classes are required for inference even if sklearn defines all-one AP.
    """
    y, p, weights = np.asarray(y), np.asarray(p), np.asarray(weights)
    if (
        y.ndim != 1
        or p.shape != y.shape
        or weights.ndim != 2
        or weights.shape[1] != len(y)
        or not np.isin(y, [0, 1]).all()
        or not np.isfinite(p).all()
        or (weights < 0).any()
        or not np.equal(weights, np.floor(weights)).all()
    ):
        raise ValueError("Invalid bootstrap input")
    order = np.argsort(-p, kind="stable")
    end = np.r_[np.flatnonzero(np.diff(p[order]) != 0), len(y) - 1]
    result = np.full(len(weights), np.nan)
    for start in range(0, len(weights), chunk_size):
        w = weights[start : start + chunk_size, order].astype(np.float64)
        tp = np.cumsum(w * y[order], axis=1)[:, end]
        total = np.cumsum(w, axis=1)[:, end]
        precision = np.divide(tp, total, out=np.zeros_like(tp), where=total > 0)
        increments = np.diff(tp, axis=1, prepend=0)
        numerator = (precision * increments).sum(1)
        positives = tp[:, -1]
        valid = (positives > 0) & (positives < total[:, -1])
        result[start : start + len(w)] = np.divide(
            numerator, positives, out=np.full(len(w), np.nan), where=valid
        )
    return result


def ci(values):
    values = np.asarray(values)
    finite = np.isfinite(values)
    if finite.sum() < 9500:
        raise ValueError("Insufficient supported paired bootstrap replicates")
    low, high = np.percentile(values[finite], [2.5, 97.5])
    return dict(
        low=float(low),
        high=float(high),
        valid_replicates=int(finite.sum()),
        undefined_replicates=int((~finite).sum()),
    )


def interpretation(low, high):
    matches = []
    if low > 0.005:
        matches.append("a")
    if low <= 0 <= high or high < 0.005:
        matches.append("b")
    if high < 0:
        matches.append("c")
    return dict(
        matching_requested_rules=matches,
        specific_requested_rule="c"
        if "c" in matches
        else matches[0]
        if matches
        else "UNCLASSIFIED",
    )


def analyze():
    started = time.monotonic()
    bundle_only_audit(ROOT, BUNDLE, OUTPUT)
    bundle = Bundle()
    split = prepare_split(bundle)
    complete = json.loads((OUTPUT / "RUN_COMPLETE.json").read_text())
    if (
        complete["completed_fits"] != 1152
        or complete["parallel_reverification"]["status"] != "PASS"
    ):
        raise PermissionError("All fits and parallel reproduction must pass")
    destination = ROOT / "docs/phase4/ORACLE_MASK_AGGREGATES.json"
    if destination.exists():
        raise FileExistsError("Analysis already exists; do not overwrite")
    episodes = [(s, d) for s in (15101, 15102, 15103) for d in (16101, 16102, 16103)]
    scores = np.empty((128, 9, 2, 2))  # mask, episode, A/B, CD/STTC
    predictions_b = np.empty((128, 9, 500, 2), dtype=np.float32)
    reports = []
    for e, (s, d) in enumerate(episodes):
        for m in range(128):
            report = check_complete(s, d, m, bundle, split)
            if report is None:
                raise PermissionError("Missing completed fit")
            reports.append(report)
            with np.load(paths(s, d, m)[1], allow_pickle=False) as arrays:
                for k, name in enumerate(("A", "B")):
                    y = bundle.y[bundle.indices(split[name])]
                    score = average_precision_score(y, arrays[name], average=None)
                    if not np.array_equal(
                        score, [report["scores"][name]["CD"], report["scores"][name]["STTC"]]
                    ):
                        raise ValueError("Saved prediction/AP reconciliation failed")
                    scores[m, e, k] = score
                predictions_b[m, e] = arrays["B"]
    means = scores.mean(axis=1)
    macro = means.mean(axis=-1)
    selected = best_mask(macro[:, 0])
    order_a = np.lexsort((np.arange(128), -macro[:, 0]))
    order_b = np.lexsort((np.arange(128), -macro[:, 1]))
    rank_a = np.argsort(order_a) + 1
    rank_b = np.argsort(order_b) + 1
    episode_selected = np.array([best_mask(scores[:, e, 0].mean(axis=-1)) for e in range(9)])
    yb = bundle.y[bundle.indices(split["B"])]
    weights = (
        np.random.default_rng(20261124)
        .multinomial(500, np.full(500, 1 / 500), size=10000)
        .astype(np.int16)
    )
    bootstrap = np.empty((128, 9, 10000, 2), dtype=np.float64)
    for m in range(128):
        for e in range(9):
            for label in range(2):
                bootstrap[m, e, :, label] = bootstrap_ap(
                    yb[:, label], predictions_b[m, e, :, label], weights
                )
        if (m + 1) % 16 == 0:
            print(f"PAIRED_RECORD_BOOTSTRAP_MASKS={m + 1}/128", flush=True)
    mean_boot = bootstrap.mean(axis=1)
    delta = mean_boot[selected] - mean_boot[127]
    primary = dict(
        mask_index=selected,
        mask_bits=f"{selected:07b}",
        split_A_mean_macro_AP=float(macro[selected, 0]),
        split_B_mean_macro_AP=float(macro[selected, 1]),
        full_split_B_mean_macro_AP=float(macro[127, 1]),
        difference=float(macro[selected, 1] - macro[127, 1]),
        ci=ci(delta.mean(axis=-1)),
        labels={
            name: dict(
                selected_AP=float(means[selected, 1, j]),
                full_AP=float(means[127, 1, j]),
                difference=float(means[selected, 1, j] - means[127, 1, j]),
                ci=ci(delta[:, j]),
            )
            for j, name in enumerate(("CD", "STTC"))
        },
    )
    primary.update(interpretation(primary["ci"]["low"], primary["ci"]["high"]))
    episode_point = np.array(
        [scores[m, e, 1].mean() - scores[127, e, 1].mean() for e, m in enumerate(episode_selected)]
    )
    episode_boot = np.stack(
        [bootstrap[m, e] - bootstrap[127, e] for e, m in enumerate(episode_selected)]
    ).mean(axis=(0, 2))
    secondary = dict(
        episode_selected_masks=episode_selected.tolist(),
        per_episode_difference=episode_point.tolist(),
        mean_difference=float(episode_point.mean()),
        ci=ci(episode_boot),
    )
    bits = np.array([[int(b) for b in f"{m:07b}"] for m in range(128)], dtype=float)
    design = np.concatenate(
        [np.ones((1152, 1)), np.repeat(bits, 9, axis=0), np.tile(np.eye(9)[:, 1:], (128, 1))],
        axis=1,
    )
    response = scores[:, :, 1].mean(axis=-1).reshape(-1)
    coef = np.linalg.lstsq(design, response, rcond=None)[0]
    effects = []
    for j, module in enumerate(
        ("view_1", "view_2", "view_3", "view_4", "view_5", "fusion_1", "fusion_2")
    ):
        effect_boot = mean_boot[bits[:, j] == 1].mean(axis=(0, 2)) - mean_boot[
            bits[:, j] == 0
        ].mean(axis=(0, 2))
        point = macro[bits[:, j] == 1, 1].mean() - macro[bits[:, j] == 0, 1].mean()
        if not np.isclose(point, coef[1 + j], rtol=1e-10, atol=1e-12):
            raise ValueError("Factorial marginal effect differs from episode-FE regression")
        effects.append(dict(module=module, adapt_minus_freeze=float(point), ci=ci(effect_boot)))
    selectors = []
    for e, (s, d) in enumerate(episodes):
        key = f"{s}/{d}"
        row = dict(source_seed=s, draw_seed=d)
        for method in ("selective_a", "random_a"):
            m = mask_index(bundle.recorded_masks[key][method])
            ep_a = scores[:, e, 0].mean(axis=-1)
            ep_b = scores[:, e, 1].mean(axis=-1)
            row[method] = dict(
                mask_index=m,
                mask_bits=f"{m:07b}",
                episode_rank_A=int(np.flatnonzero(np.lexsort((np.arange(128), -ep_a)) == m)[0] + 1),
                episode_rank_B=int(np.flatnonzero(np.lexsort((np.arange(128), -ep_b)) == m)[0] + 1),
                fixed_mask_rank_mean_A=int(rank_a[m]),
                fixed_mask_rank_mean_B=int(rank_b[m]),
                split_A_macro_AP=float(ep_a[m]),
                split_B_macro_AP=float(ep_b[m]),
            )
        selectors.append(row)
    top = []
    for m in order_a[:10]:
        top.append(
            dict(
                mask_index=int(m),
                mask_bits=f"{m:07b}",
                rank_A=int(rank_a[m]),
                rank_B=int(rank_b[m]),
                A_macro_AP=float(macro[m, 0]),
                B_macro_AP=float(macro[m, 1]),
                A_CD_AP=float(means[m, 0, 0]),
                A_STTC_AP=float(means[m, 0, 1]),
                B_CD_AP=float(means[m, 1, 0]),
                B_STTC_AP=float(means[m, 1, 1]),
            )
        )
    label_top = {}
    for j, name in enumerate(("CD", "STTC")):
        label_order_a = np.lexsort((np.arange(128), -means[:, 0, j]))
        label_order_b = np.lexsort((np.arange(128), -means[:, 1, j]))
        label_rank_a = np.argsort(label_order_a) + 1
        label_rank_b = np.argsort(label_order_b) + 1
        label_top[name] = dict(
            full_rank_A=int(label_rank_a[127]),
            full_rank_B=int(label_rank_b[127]),
            head_rank_A=int(label_rank_a[0]),
            head_rank_B=int(label_rank_b[0]),
            top_10=[
                dict(
                    mask_bits=f"{m:07b}",
                    rank_A=int(label_rank_a[m]),
                    rank_B=int(label_rank_b[m]),
                    A_AP=float(means[m, 0, j]),
                    B_AP=float(means[m, 1, j]),
                )
                for m in label_order_a[:10]
            ],
        )
    progress = [json.loads(line) for line in (OUTPUT / "progress.jsonl").read_text().splitlines()]
    reprod = json.loads((OUTPUT / "REPRODUCIBILITY.json").read_text())
    extra_seconds = sum(r["runtime_seconds"] for r in reprod["repeats"]) + sum(
        r["runtime_seconds"] for r in complete["parallel_reverification"]["repeats"]
    )
    summary = dict(
        schema="phase4-validation-only-oracle-results-v1",
        primary=primary,
        validation=split_summary(bundle, split),
        top_10_by_A=top,
        label_specific_top_10_by_A=label_top,
        full=dict(
            rank_A=int(rank_a[127]),
            rank_B=int(rank_b[127]),
            A_macro_AP=float(macro[127, 0]),
            B_macro_AP=float(macro[127, 1]),
        ),
        head_only=dict(
            rank_A=int(rank_a[0]),
            rank_B=int(rank_b[0]),
            A_macro_AP=float(macro[0, 0]),
            B_macro_AP=float(macro[0, 1]),
        ),
        per_episode_oracle=secondary,
        module_effects=effects,
        recorded_selector_ranks=selectors,
        all_mask_means=[
            dict(
                mask_index=m,
                mask_bits=f"{m:07b}",
                A_macro_AP=float(macro[m, 0]),
                B_macro_AP=float(macro[m, 1]),
            )
            for m in range(128)
        ],
        fits=dict(
            planned=1152,
            complete=len(reports),
            failed=sum(r["status"] == "FAILED_STOP" for r in progress),
            reproducibility_extra_fits=5,
            fit_seconds_sum=sum(r["runtime_seconds"] for r in reports),
            extra_fit_seconds=extra_seconds,
            study_wall_seconds=complete["session_elapsed_seconds"],
        ),
        bootstrap=dict(
            replicates=10000,
            seed=20261124,
            unit="split-B record",
            same_record_resample_for_every_mask_episode_label=True,
            interval="percentile; conditional on A selection and fixed fitted episodes",
        ),
        reproducibility=dict(
            full_gate="PASS",
            parallel_gate="PASS",
            matched_original_reference_fits=sum(
                bool(r["original_reference_methods_matched"]) for r in reports
            ),
        ),
        no_test_signal_label_prediction_metric_access=True,
        no_new_fitted_checkpoint=True,
    )
    figure_dir = ROOT / "figures/phase4"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.6, 4.5), layout="constrained")
    ax.hist(macro[:, 1], bins=20, color="#91a9bc", edgecolor="white")
    for m, label, color in (
        (127, "Full", "#b43c39"),
        (0, "Head only", "#286a4c"),
        (selected, "A-selected", "#5a3483"),
    ):
        ax.axvline(macro[m, 1], label=f"{label}: {macro[m, 1]:.4f}", color=color, linewidth=1.8)
    for method, label, color in (
        ("selective_a", "Original selector policy", "#a27318"),
        ("random_a", "Original random policy", "#355780"),
    ):
        value = np.mean([r[method]["split_B_macro_AP"] for r in selectors])
        ax.axvline(value, label=f"{label}: {value:.4f}", color=color, linestyle="--")
        unique = sorted({r[method]["mask_index"] for r in selectors})
        ax.plot(
            macro[unique, 1],
            np.full(len(unique), 0.4 if method == "selective_a" else 0.9),
            "v" if method == "selective_a" else "^",
            color=color,
            markersize=5,
            label=f"{label.replace(' policy', '')} fixed-mask positions",
        )
    ax.set(
        xlabel="Mean split-B CD/STTC macro AP over nine episodes",
        ylabel="Number of fixed masks",
        title="Validation-only oracle: 128 masks; identical A/B records across episodes",
    )
    ax.legend(fontsize=8)
    fig.savefig(figure_dir / "mask_distribution.svg")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.6, 4.5), layout="constrained")
    names = [e["module"] for e in effects]
    vals = np.array([e["adapt_minus_freeze"] for e in effects])
    lower = np.array([e["ci"]["low"] for e in effects])
    upper = np.array([e["ci"]["high"] for e in effects])
    ax.hlines(np.arange(7), lower, upper, color="#234e70")
    ax.scatter(vals, np.arange(7), color="#234e70")
    ax.set_yticks(np.arange(7), names)
    ax.axvline(0, color="gray", linestyle="--")
    ax.set(
        xlabel="Adapt minus freeze effect on split-B macro AP",
        title="Factorial module effects with episode fixed effects\n95% paired record-bootstrap intervals; exploratory",
    )
    fig.savefig(figure_dir / "module_effects.svg")
    plt.close(fig)
    summary["analysis_runtime_seconds"] = time.monotonic() - started
    new_json(destination, summary)
    write_report(summary)
    print(
        json.dumps(dict(primary=primary, full=summary["full"], fits=summary["fits"]), indent=2),
        flush=True,
    )


def write_report(s):
    p = s["primary"]
    interval = p["ci"]
    lines = [
        "# Phase 4A — validation-only oracle module-mask results",
        "",
        "Completed using only the adaptation bundle and new Phase 4 outputs. No CPSC test outcome, signal, prediction, metric or evaluation artifact was read. Test IDs were used only as an exclusion list during building; the bundle stores exclusion hashes, not those IDs.",
        "",
        "## Protocol, bundle and validation split",
        "",
        "[Prespecified protocol](ORACLE_MASK_PROTOCOL.md) · [Split counts and list hashes](VALIDATION_SPLIT_SUMMARY.json) · [Full aggregate results](ORACLE_MASK_AGGREGATES.json)",
        "",
        "All 711 unique original label-draw records across budgets were excluded. The remaining adaptation pool has 3,148 records. A/B each contain 500 records, with zero mutual overlap, zero overlap with original label draws, and zero overlap with test IDs. All labels use the unchanged verified V2 CD/STTC mapping. Positive is recorded annotation, not an adjudicated clinical diagnosis.",
        "",
        "| Split | Records | CD positives | STTC positives |",
        "|---|---:|---:|---:|",
    ]
    for name in ("A", "B"):
        v = s["validation"][name]
        lines.append(
            f"| {name} | {v['records']} | {v['positive_counts'][0]} ({v['prevalence'][0]:.1%}) | {v['positive_counts'][1]} ({v['prevalence'][1]:.1%}) |"
        )
    lines += [
        "",
        "The builder copied three existing source checkpoint files byte-for-byte. No new fitted checkpoint was saved. All source hashes/normalization matched; V2 labels and preprocessed signals matched the original development caches for all 711 original draw records. Record IDs, waveforms, labels and per-fit validation probabilities stay under gitignored `private/phase4_oracle_*` paths. SHA-256 values bind identities; they do not establish clinical validation.",
        "",
        "## Reproducibility and execution",
        "",
        f"Full-mask repeated A/B predictions, final state and loss history were bitwise identical. Final state, drift and loss exactly matched the original Experiment 4 full-cost record. All reference-compatible masks were reconciled during fitting ({s['reproducibility']['matched_original_reference_fits']} distinct fits). Three prespecified random mask/episode fits from the parallel run were repeated serially with identical predictions, state, drift and history.",
        "",
        f"Completed **{s['fits']['complete']}/1,152** distinct mask/episode fits; failed fits **{s['fits']['failed']}**. Five extra reproducibility fits. Three processes × original four threads. Main study wall time **{s['fits']['study_wall_seconds']:.1f} s**; summed fit durations **{s['fits']['fit_seconds_sum']:.1f} s**, extra verification fits **{s['fits']['extra_fit_seconds']:.1f} s**. Summed concurrent durations are not elapsed wall time.",
        "",
        "## Primary A-selected fixed mask versus full fine-tuning",
        "",
        f"A-selected mask **`{p['mask_bits']}`** (index {p['mask_index']}); bit order is view_1..view_5, fusion_1, fusion_2. One means ADAPT, zero SHARE; head always trainable. Ascending mask index resolves exact A ties.",
        "",
        f"Mean A macro AP: **{p['split_A_mean_macro_AP']:.6f}**. Mean B macro AP: selected **{p['split_B_mean_macro_AP']:.6f}**, full **{p['full_split_B_mean_macro_AP']:.6f}**. Primary difference **{p['difference']:+.6f}**, 95% paired record-bootstrap CI **[{interval['low']:+.6f}, {interval['high']:+.6f}]** ({interval['valid_replicates']}/10,000 valid replicates).",
        "",
        f"Requested rule match(es): **{', '.join(p['matching_requested_rules']) or 'none'}**; specific requested category: **{p['specific_requested_rule']}**. The exact requested statements and overlap are preserved in the protocol. A CI including zero is inconclusive and cannot prove that no useful mask exists; a negative contrast of the A-selected mask cannot prove that every freezing choice hurts. This is conditional validation evidence, not a CPSC test result or an attribution of selector failure by itself.",
        "",
        "| Label | Selected B AP | Full B AP | Difference | 95% paired CI |",
        "|---|---:|---:|---:|---|",
    ]
    for name, v in p["labels"].items():
        lines.append(
            f"| {name} | {v['selected_AP']:.6f} | {v['full_AP']:.6f} | {v['difference']:+.6f} | [{v['ci']['low']:+.6f}, {v['ci']['high']:+.6f}] |"
        )
    lines += [
        "",
        "## Top ten by mean split-A macro AP",
        "",
        "Ranks are ordinal, with ascending mask index as tie breaker. B ranks are descriptive and were never used to select a model.",
        "",
        "| A rank | Mask | A macro AP | B macro AP | A CD | A STTC | B CD | B STTC | B rank |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for v in s["top_10_by_A"]:
        lines.append(
            f"| {v['rank_A']} | `{v['mask_bits']}` | {v['A_macro_AP']:.6f} | {v['B_macro_AP']:.6f} | {v['A_CD_AP']:.6f} | {v['A_STTC_AP']:.6f} | {v['B_CD_AP']:.6f} | {v['B_STTC_AP']:.6f} | {v['rank_B']} |"
        )
    for name, result in s["label_specific_top_10_by_A"].items():
        lines += [
            "",
            f"### {name}-specific A rankings (secondary)",
            "",
            "These rank the individual label on A; they do not replace the macro-selected primary mask.",
            "",
            "| A rank | Mask | A AP | B AP | B rank |",
            "|---:|---|---:|---:|---:|",
        ]
        for v in result["top_10"]:
            lines.append(
                f"| {v['rank_A']} | `{v['mask_bits']}` | {v['A_AP']:.6f} | {v['B_AP']:.6f} | {v['rank_B']} |"
            )
        lines += [
            "",
            f"{name}: full A/B rank {result['full_rank_A']}/{result['full_rank_B']}; head-only A/B rank {result['head_rank_A']}/{result['head_rank_B']}.",
        ]
    lines += [
        "",
        f"Full fine-tuning: A rank **{s['full']['rank_A']}**, B rank **{s['full']['rank_B']}**. Head-only: A rank **{s['head_only']['rank_A']}**, B rank **{s['head_only']['rank_B']}**.",
        "",
        "![Distribution of all fixed-mask B AP means](../../figures/phase4/mask_distribution.svg)",
        "",
        "Dashed selector/random lines are the original episode-varying policy means on B, not necessarily any single fixed mask. Their exact recorded mask identities and ranks are listed below. All fixed-mask means are available in the aggregate JSON.",
        "",
        "## Per-episode A-selected oracle (secondary)",
        "",
        f"Mean B oracle-minus-full difference **{s['per_episode_oracle']['mean_difference']:+.6f}**, 95% paired CI **[{s['per_episode_oracle']['ci']['low']:+.6f}, {s['per_episode_oracle']['ci']['high']:+.6f}]**. Episode masks (source-major, draw-minor): `{s['per_episode_oracle']['episode_selected_masks']}`. Every episode mask is selected on its own A scores only. Reused B records and nine fitted models are not nine independent cohorts.",
        "",
        "## Module effects",
        "",
        "OLS: B macro AP ~ seven mask bits + episode fixed effects. The complete factorial design makes each coefficient equal its adapt-minus-freeze average across the other bits and episodes. CIs resample B records jointly across every fit, rather than treating 1,152 correlated AP measurements as independent observations. Seven exploratory intervals are not multiplicity-adjusted and are not causal clinical importance.",
        "",
        "| Module | Adapt minus freeze | 95% paired record-bootstrap CI |",
        "|---|---:|---|",
    ]
    for v in s["module_effects"]:
        lines.append(
            f"| {v['module']} | {v['adapt_minus_freeze']:+.6f} | [{v['ci']['low']:+.6f}, {v['ci']['high']:+.6f}] |"
        )
    lines += [
        "",
        "![Module factorial effects](../../figures/phase4/module_effects.svg)",
        "",
        "## Original Experiment 4 selector and random masks",
        "",
        "| Source seed | Draw seed | Selector mask | Its episode A/B rank | Random mask | Its episode A/B rank |",
        "|---:|---:|---|---|---|---|",
    ]
    for v in s["recorded_selector_ranks"]:
        a, b = v["selective_a"], v["random_a"]
        lines.append(
            f"| {v['source_seed']} | {v['draw_seed']} | `{a['mask_bits']}` | {a['episode_rank_A']} / {a['episode_rank_B']} | `{b['mask_bits']}` | {b['episode_rank_A']} / {b['episode_rank_B']} |"
        )
    lines += [
        "",
        "## Limits and preservation",
        "",
        "This study selects on A and evaluates once on B; B-ranked distributions/effects remain descriptive. The primary interval holds the A-selected mask and nine fitted source/draw episodes fixed and does not capture source-training uncertainty, future A-split selection variability or unknown patient clustering. The seven modules are scientific computational units, not independent clinical risk factors. No hyperparameters, label budget, target hospital, seeds or thresholds were tuned. Every fit used the exact original 128 F∪G records, source normalization, AdamW settings, 200 updates and persistent frozen BatchNorm statistics.",
        "",
        "No CPSC test waveform, label, probability, metric, seal or evaluation output was accessed. No PTB-XL fold 10 or sealed/reserved payload was accessed. Test IDs were used only to reject forbidden records. No old result, config, ledger, report or source was modified. No new fitted checkpoint was written; only byte-identical copies of the three existing source checkpoints are in the bundle. Nothing was committed, pushed, merged or submitted as a PR. Stop here; no 64-/256-label runs or test evaluation are performed.",
    ]
    (ROOT / "docs/phase4/ORACLE_MASK_RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    analyze()
