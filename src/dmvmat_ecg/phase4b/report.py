"""Final validation-only table, low-label oracle and paired uncertainty."""

import json
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dmvmat_ecg.phase4.safety import new_json, sha

from .common import ROOT, RUN, setup
from .statistics import boots, intervals, measures
from .target_fit import key
from .train_alignment import tag


def analyze():
    started = time.monotonic()
    t = setup()
    destination = ROOT / "docs/phase4/PHASE4B_AGGREGATES.json"
    if destination.exists():
        raise FileExistsError("Completed analysis cannot be overwritten")
    if (
        not (RUN / "ORACLE64_COMPLETE.json").exists()
        or not (RUN / "BASELINES_COMPLETE.json").exists()
    ):
        raise PermissionError("Oracle and baseline fits must finish")
    alpha = json.loads((RUN / "ALPHA_SELECTIONS.json").read_text())
    lam = json.loads((RUN / "LAMBDA_SELECTION.json").read_text())
    if lam["B_used_for_selection"] or any(v["selection_uses_B"] for v in alpha.values()):
        raise PermissionError("B was used for selection")
    if not (RUN / "ALIGNED_TARGETS_COMPLETE.json").exists():
        raise PermissionError("Aligned-source stage must finish or record its collapse decision")
    y = t.labels("B")
    weights = (
        np.random.default_rng(20261127)
        .multinomial(500, np.full(500, 1 / 500), size=10000)
        .astype(np.int16)
    )
    episodes = [(s, d) for s in (15101, 15102, 15103) for d in (16101, 16102, 16103)]
    oracleA = np.empty((128, 9, 2))
    oracleB = np.empty((128, 9, 2))
    oracleBoot = np.empty((128, 9, 10000, 2))
    cache = {}

    def read(name, with_auc=True):
        if (name, with_auc) in cache:
            return cache[name, with_auc]
        r = json.loads((RUN / "fits" / (name + ".json")).read_text())
        p = RUN / "predictions" / (name + ".npz")
        if r["status"] != "COMPLETE" or sha(p) != r["predictions_sha256"]:
            raise PermissionError("Completed fit identity/prediction mismatch")
        with np.load(p, allow_pickle=False) as arrays:
            a, b = arrays["A"], arrays["B"]
        if a.shape != (500, 2) or b.shape != (500, 2):
            raise ValueError("Not validation-only probability arrays")
        a_measure = measures(t.labels("A"), a)
        if not np.array_equal(a_measure[0], r["A_AP"]):
            raise ValueError("A AP differs from sealed selection record")
        result = dict(A=a_measure, B=measures(y, b), bootstrap=boots(y, b, weights, with_auc))
        cache[name, with_auc] = result
        return result

    for m in range(128):
        for e, (s, d) in enumerate(episodes):
            row = read(key(s, d, 64, "oracle", mask=m), False)
            oracleA[m, e] = row["A"][0]
            oracleB[m, e] = row["B"][0]
            oracleBoot[m, e] = row["bootstrap"]
        if (m + 1) % 16 == 0:
            print(f"PHASE4B_BOOTSTRAP_ORACLE_MASKS={m + 1}/128", flush=True)
    am = oracleA.mean(axis=(1, 2))
    bm = oracleB.mean(axis=(1, 2))
    rankA = np.argsort(np.lexsort((np.arange(128), -am))) + 1
    rankB = np.argsort(np.lexsort((np.arange(128), -bm))) + 1
    selected = int(np.argmax(am))
    meanboot = oracleBoot.mean(axis=1)
    delta = meanboot[selected] - meanboot[127]
    bits = np.array([[int(b) for b in f"{m:07b}"] for m in range(128)])
    effects = []
    design = np.c_[np.ones(1152), np.repeat(bits, 9, axis=0), np.tile(np.eye(9)[:, 1:], (128, 1))]
    coef = np.linalg.lstsq(design, oracleB.mean(axis=-1).ravel(), rcond=None)[0]
    for j, n in enumerate(
        ["view_1", "view_2", "view_3", "view_4", "view_5", "fusion_1", "fusion_2"]
    ):
        diff = bm[bits[:, j] == 1].mean() - bm[bits[:, j] == 0].mean()
        if not np.isclose(diff, coef[j + 1], atol=1e-12):
            raise ValueError("Oracle module regression mismatch")
        values = meanboot[bits[:, j] == 1].mean(axis=(0, 2)) - meanboot[bits[:, j] == 0].mean(
            axis=(0, 2)
        )
        effects.append(dict(module=n, effect=float(diff), ci=intervals(values)))
    chosen_e = np.argmax(oracleA.mean(axis=-1), axis=0)
    secondary_boot = np.stack(
        [oracleBoot[m, e] - oracleBoot[127, e] for e, m in enumerate(chosen_e)]
    ).mean(axis=(0, 2))
    oracle = dict(
        selected_mask=f"{selected:07b}",
        selected_mask_index=selected,
        A_macro_AP=float(am[selected]),
        B_macro_AP=float(bm[selected]),
        full_rank_A=int(rankA[127]),
        full_rank_B=int(rankB[127]),
        head_rank_A=int(rankA[0]),
        head_rank_B=int(rankB[0]),
        difference=float(bm[selected] - bm[127]),
        ci=intervals(delta.mean(axis=-1)),
        top_5=[
            dict(
                mask=f"{m:07b}",
                rank_A=int(rankA[m]),
                rank_B=int(rankB[m]),
                A_macro_AP=float(am[m]),
                B_macro_AP=float(bm[m]),
                A_CD_STTC_AP=oracleA[m].mean(axis=0).tolist(),
                B_CD_STTC_AP=oracleB[m].mean(axis=0).tolist(),
            )
            for m in np.lexsort((np.arange(128), -am))[:5]
        ],
        module_effects=effects,
        per_episode_oracle=dict(
            masks=chosen_e.tolist(),
            difference=float(
                np.mean(
                    [oracleB[m, e].mean() - oracleB[127, e].mean() for e, m in enumerate(chosen_e)]
                )
            ),
            ci=intervals(secondary_boot),
        ),
        label_differences={
            label: dict(
                difference=float(oracleB[selected, :, j].mean() - oracleB[127, :, j].mean()),
                ci=intervals(delta[:, j]),
            )
            for j, label in enumerate(["CD", "STTC"])
        },
        all_mask_means=[
            dict(mask=f"{m:07b}", A_AP=float(am[m]), B_AP=float(bm[m])) for m in range(128)
        ],
    )
    oracle["freezing_beats_full"] = selected != 127 and oracle["ci"]["low"] > 0
    nonfull = max(range(127), key=lambda m: (bm[m], -m))
    oracle["descriptive_best_nonfull_B"] = dict(
        mask=f"{nonfull:07b}",
        B_AP=float(bm[nonfull]),
        difference=float(bm[nonfull] - bm[127]),
        nonfull_masks_above_full_mean=int((bm[:127] > bm[127]).sum()),
        selected_on_B_for_description_only=True,
    )

    def filename(s, d, b, method):
        if method == "full":
            return key(s, d, 64, "oracle", mask=127) if b == 64 else key(s, d, b, "full")
        if method in ("source", "adabn"):
            return key(s, 16101, 0, method)
        if method == "l2sp":
            return key(s, d, b, "l2sp", alpha=alpha[str(b)]["alpha"])
        if method in ("aligned_source", "aligned_full", "aligned_selector"):
            a = tag(lam["selected_lambda"])
            if method == "aligned_source":
                return key(s, 16101, 0, "source", aligned=a)
            if method == "aligned_full":
                return key(s, d, b, "full", aligned=a)
            decision = json.loads(
                (RUN / "aligned_decisions" / f"{a}_{s}_{d}_B{b}.json").read_text()
            )
            mask = int(
                "".join(
                    "1" if decision["mask"][n] == "ADAPT" else "0"
                    for n in [
                        "view_1",
                        "view_2",
                        "view_3",
                        "view_4",
                        "view_5",
                        "fusion_1",
                        "fusion_2",
                    ]
                ),
                2,
            )
            return key(s, d, b, "aligned_selector", aligned=a, mask=mask)
        return key(s, d, b, method)

    methods = ["full", "source", "adabn", "adabn_full", "lpft", "l2sp"]
    if lam["selected_lambda"] is not None:
        methods += ["aligned_source", "aligned_full", "aligned_selector"]
    rows = []
    for b in (64, 128, 256):
        base = [read(filename(s, d, b, "full")) for s, d in episodes]
        base_mean = np.mean([r["B"] for r in base], axis=0)
        base_boot = np.mean([r["bootstrap"] for r in base], axis=0)
        for method in methods:
            reports = [read(filename(s, d, b, method)) for s, d in episodes]
            point = np.mean([r["B"] for r in reports], axis=0)
            sampled = np.mean([r["bootstrap"] for r in reports], axis=0)
            dboot = sampled - base_boot
            ap_delta = float((point[0] - base_mean[0]).mean())
            ci = intervals(dboot[:, 0].mean(axis=-1))
            rows.append(
                dict(
                    method=method,
                    budget=b,
                    optimization_labels=0 if method in ("source", "adabn", "aligned_source") else b,
                    macro_AP=float(point[0].mean()),
                    macro_AUROC=float(point[1].mean()),
                    AP_difference=ap_delta,
                    AP_ci=ci,
                    AUROC_difference=float((point[1] - base_mean[1]).mean()),
                    AUROC_ci=intervals(dboot[:, 1].mean(axis=-1)),
                    CD_AP=float(point[0, 0]),
                    STTC_AP=float(point[0, 1]),
                    CD_AUROC=float(point[1, 0]),
                    STTC_AUROC=float(point[1, 1]),
                    label_differences={
                        label: dict(
                            AP_difference=float(point[0, j] - base_mean[0, j]),
                            AP_ci=intervals(dboot[:, 0, j]),
                            AUROC_difference=float(point[1, j] - base_mean[1, j]),
                            AUROC_ci=intervals(dboot[:, 1, j]),
                        )
                        for j, label in enumerate(["CD", "STTC"])
                    },
                    beats_full_pointwise=method != "full" and ci["low"] > 0,
                    familywise_AP_ci=intervals(dboot[:, 0].mean(axis=-1), 7)
                    if b == 128 and method not in ("full", "source")
                    else None,
                    distinct_prediction_vectors=len(
                        {filename(s, d, b, method) for s, d in episodes}
                    ),
                )
            )
    winners = [
        r["method"]
        for r in rows
        if r["budget"] == 128 and r["method"] != "full" and r["beats_full_pointwise"]
    ]
    original = json.loads((RUN / "ORIGINAL_FOLD9_BASELINES.json").read_text())
    source_rows = []
    for a in (0.1, 1.0):
        for s in (15101, 15102, 15103):
            entry = json.loads((RUN / "aligned_sources" / f"{tag(a)}_{s}.json").read_text())
            source_rows.append(
                dict(
                    seed=s,
                    lambda_value=a,
                    original_fold9_AUROC=original[str(s)]["AUROC"],
                    aligned_fold9_AUROC=entry["fold9_AUROC"],
                    selected_epoch=entry["selected_epoch"],
                    checkpoint_sha256=entry["sha256"],
                    runtime_seconds=entry["runtime_seconds"],
                )
            )
    result = dict(
        schema="phase4b-validation-only-results-v1",
        rows=rows,
        oracle64=oracle,
        alpha_selection=alpha,
        lambda_selection=lam,
        source_fold9=source_rows,
        primary128_pointwise_winners=winners,
        any_method_beats_full128=bool(winners),
        bootstrap=dict(
            replicates=10000,
            seed=20261127,
            unit="same split-B record across every arm/episode/label",
        ),
        validation_B_is_reused_from_phase4A=True,
        no_CPSC_test_data_access=True,
        U_labels_used=False,
        analysis_seconds=time.monotonic() - started,
    )
    new_json(destination, result)
    figures(result)
    write_report(result)
    print(
        json.dumps(
            dict(
                primary128_winners=winners,
                oracle64=oracle["selected_mask"],
                oracle64_difference=oracle["difference"],
                oracle64_ci=oracle["ci"],
            ),
            indent=2,
        ),
        flush=True,
    )


def figures(result):
    directory = ROOT / "figures/phase4"
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    primary = [r for r in result["rows"] if r["budget"] == 128]
    for i, row in enumerate(primary):
        ax.hlines(i, row["AP_ci"]["low"], row["AP_ci"]["high"], color="#275976")
        ax.plot(row["AP_difference"], i, "o", color="#275976")
    ax.set_yticks(range(len(primary)), [r["method"] for r in primary])
    ax.axvline(0, color="gray", linestyle="--")
    ax.set(
        xlabel="Mean split-B macro AP minus original-source full fine-tuning",
        title="Phase 4B: 128 optimization labels; validation-only\nPointwise 95% paired record-bootstrap intervals",
    )
    fig.savefig(directory / "phase4b_primary_comparisons.svg")
    plt.close(fig)
    values = result["oracle64"]["all_mask_means"]
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    ax.hist([r["B_AP"] for r in values], bins=20, color="#9db2c2", edgecolor="white")
    for index, label, color in [
        (0, "Head-only", "#3e7457"),
        (127, "Full", "#a94340"),
        (result["oracle64"]["selected_mask_index"], "A-selected", "#66428d"),
    ]:
        ax.axvline(values[index]["B_AP"], label=label, color=color)
    ax.legend()
    ax.set(
        xlabel="Mean split-B CD/STTC macro AP over nine episodes",
        ylabel="Fixed mask count",
        title="64-label oracle: all 128 masks, same validation records",
    )
    fig.savefig(directory / "phase4b_oracle64_distribution.svg")
    plt.close(fig)


def write_report(r):
    o = r["oracle64"]
    lines = [
        "# Phase 4B — validation-only adaptation results",
        "",
        "[Protocol](PHASE4B_PROTOCOL.md) · [Full numeric results](PHASE4B_AGGREGATES.json)",
        "",
        f"**Any new method beats full at 128 labels under the requested pointwise rule: {r['any_method_beats_full128']}.** Winners: {', '.join(r['primary128_pointwise_winners']) or 'none'}.",
        "",
        "The rule is a paired-bootstrap 95% lower bound above zero. Pointwise findings are not automatically familywise-confirmed: seven-method Bonferroni intervals are in the aggregate JSON. B was already examined in Phase 4A; this is exploratory validation reuse, not a fresh held-out test or clinical validation. All comparisons share the same 500 B records and nine fixed source/draw episodes. A=500 is used for selection; optimization label budgets exclude these extra labels.",
        "",
        "## Main method × budget table",
        "",
        "Zero-label methods have three distinct source vectors reused across nine paired cells and across nominal budgets; they do not consume 64/128/256 labels. AP/AUROC are per-model values averaged over cells, not scores pooled or ensembled. CD/STTC are annotation-union proxies, not adjudicated disease diagnoses.",
        "",
        "| Budget | Method | Fit labels | Macro AP | Macro AUROC | AP difference [95% CI] | CD AP / AUROC | STTC AP / AUROC |",
        "|---:|---|---:|---:|---:|---|---|---|",
    ]
    for v in r["rows"]:
        lines.append(
            f"| {v['budget']} | {v['method']} | {v['optimization_labels']} | {v['macro_AP']:.6f} | {v['macro_AUROC']:.6f} | {v['AP_difference']:+.6f} [{v['AP_ci']['low']:+.6f}, {v['AP_ci']['high']:+.6f}] | {v['CD_AP']:.6f} / {v['CD_AUROC']:.6f} | {v['STTC_AP']:.6f} / {v['STTC_AUROC']:.6f} |"
        )
    lines += [
        "",
        "![Primary comparisons](../../figures/phase4/phase4b_primary_comparisons.svg)",
        "",
        "## A-only hyperparameter choices",
        "",
        f"L2-SP alpha selections: `{r['alpha_selection']}`.",
        "",
        f"Alignment lambda selection: `{r['lambda_selection']}`. The source collapse screen was fixed before fitting; its .05 AUROC tolerance is an engineering safeguard, not a clinical margin. No B labels or scores entered alpha/lambda selection.",
        "",
        "## Low-label oracle",
        "",
        f"A-selected 64-label mask: `{o['selected_mask']}`; B AP difference versus full **{o['difference']:+.6f}**, 95% CI **[{o['ci']['low']:+.6f}, {o['ci']['high']:+.6f}]**. Freezing wins under the lower-bound-above-zero criterion: **{o['freezing_beats_full']}**.",
        "",
        f"Full ranks A/B: {o['full_rank_A']}/{o['full_rank_B']}; head-only ranks A/B: {o['head_rank_A']}/{o['head_rank_B']}. A-selected full would give a degenerate identity interval, not an equivalence test. Neither an interval including zero nor this finite grid proves that useful freezing is universally impossible.",
        "",
        "| A rank | Mask | A AP | B AP | B rank |",
        "|---:|---|---:|---:|---:|",
    ]
    for v in o["top_5"]:
        lines.append(
            f"| {v['rank_A']} | `{v['mask']}` | {v['A_macro_AP']:.6f} | {v['B_macro_AP']:.6f} | {v['rank_B']} |"
        )
    lines += [
        "",
        "![64-label distribution](../../figures/phase4/phase4b_oracle64_distribution.svg)",
        "",
        "Bit order: view_1..view_5, fusion_1, fusion_2; 1 adapts. The per-episode A oracle and CD/STTC contrasts are in the aggregate JSON. Original recorded B64 selector masks were not included in the adaptation bundle; no B128 mask is mislabeled as B64.",
        "",
        "### Module effects at 64 labels",
        "",
        "B macro AP regressed on the seven bits plus episode fixed effects. Intervals use joint record resamples, not independent regression rows; exploratory and not multiplicity-adjusted.",
        "",
        "| Module | Adapt minus freeze | 95% CI |",
        "|---|---:|---|",
    ]
    for v in o["module_effects"]:
        lines.append(
            f"| {v['module']} | {v['effect']:+.6f} | [{v['ci']['low']:+.6f}, {v['ci']['high']:+.6f}] |"
        )
    lines += [
        "",
        "## Source fold-9 performance and alignment",
        "",
        "| Seed | Lambda | Original CD/STTC AUROC | Aligned CD/STTC AUROC | Selected epoch |",
        "|---:|---:|---|---|---:|",
    ]
    for v in r["source_fold9"]:
        a, b = v["original_fold9_AUROC"], v["aligned_fold9_AUROC"]
        lines.append(
            f"| {v['seed']} | {v['lambda_value']} | {a[0]:.6f} / {a[1]:.6f} | {b[0]:.6f} / {b[1]:.6f} | {v['selected_epoch']} |"
        )
    lines += [
        "",
        "Six lambda/seed candidates were trained from fresh initialization; one selected lambda supplies three aligned models. Source normalization, eligibility, 50-epoch OneCycle schedule and fold9 AP checkpoint selection match the original recipe. Source batches are 128, unlabelled target batches 32. Alignment uses source labels only; U contains 2,859 adaptation signals and excludes both A and B. Target passes also update source BN statistics, so improvement cannot be attributed uniquely to MMD without an additional matched ablation. The selected-source selector uses the original production MMD/geometry logic on source anchors and G, not a mask reused from another source.",
        "",
        "## Reproducibility, runtime and isolation",
        "",
        "See PHASE4B_VALIDATION_REPORT.md for actual replay checks, fit/checkpoint counts, runtimes and preservation. All per-record arrays and the six permitted aligned-source checkpoints remain gitignored. No target or AdaBN checkpoint was saved. Test exclusion guards and OS-open audit guards remain active. PTB-XL fold 10 and CPSC test-half signals/labels/predictions/metrics/seals were never read. The only CPSC inputs are the isolated adaptation bundle; source inputs were verified folds1–9. No commits, pushes or merges.",
    ]
    (ROOT / "docs/phase4/PHASE4B_RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    analyze()
