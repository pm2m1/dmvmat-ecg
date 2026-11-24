"""Write aggregate validation evidence and the uncommitted Georgia protocol."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dmvmat_ecg.phase4.safety import digest, new_json, sha

from .common import ARMS, BUDGETS, GEORGIA, ROOT, RUN, SEEDS, Scope
from .sources import descriptor


def interval(row):
    c = row["CI95"]
    return (
        "unavailable"
        if c["interval"] is None
        else f"[{c['interval'][0]:+.6f}, {c['interval'][1]:+.6f}]"
    )


def perseed_plot(result):
    folder = ROOT / "figures/phase5"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"phase5_{result['hospital'].lower()}_per_seed_ap_differences.svg"
    if path.exists():
        raise FileExistsError("Existing aggregate figure preserved")
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 9})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    for ax, b in zip(axes, BUDGETS, strict=True):
        rows = [r for r in result["per_seed"] if r["budget"] == b]
        if [r["source_seed"] for r in rows] != list(SEEDS):
            raise PermissionError("Incomplete per-seed points")
        ax.scatter(range(10), [r["macro_AP_difference"] for r in rows], s=32)
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
    fig.suptitle(result["hospital"] + " validation B: mean of three fixed draws per source seed")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def run():
    # Read-only aggregation of completed development artifacts. No raw/test loader.
    Scope("Georgia").install()
    sources = json.loads((RUN / "SOURCES_COMPLETE.json").read_text())
    if sources["source_models"] != 20 or sources["status"] != "COMPLETE":
        raise PermissionError("Sources incomplete")
    datasets = {}
    for hospital in ("CPSC", "Georgia"):
        # These new private outputs are in the explicitly allowed RUN subtree.
        datasets[hospital] = json.loads((RUN / hospital / "VALIDATION_ANALYSIS.json").read_text())
    summary = json.loads((GEORGIA / "MANIFEST.json").read_text())["summary"]
    source_rows = [descriptor(a, s)[3] for s in SEEDS for a in ARMS]
    doc = ROOT / "docs/phase5/PHASE5_VALIDATION_RESULTS.md"
    protocol = ROOT / "docs/phase5/PHASE5_GEORGIA_PROTOCOL.md"
    if doc.exists() or protocol.exists():
        raise FileExistsError("Existing final report/protocol preserved")
    figures = {h: perseed_plot(r) for h, r in datasets.items()}
    lines = [
        "# Phase5 — multi-seed and Georgia development replication",
        "",
        "Completed fixed validation analyses only. No CPSC test data or Phase4C test outputs were loaded. No Georgia test signal/label or PTB-XL fold10 was loaded. No commit, push, merge, tuning or test evaluation occurred.",
        "",
        "## Part A — source training",
        "",
        "Ten paired source seeds per arm. Seeds15101–15103 reuse the existing hash-verified Phase4C sources;15104–15110 are new fixed50-epoch source fits. Both arms use PTB-XL training folds1–8 and checkpoint selection on fold9 AP/BCE. Fold9 AUROC below is descriptive, not the checkpoint-selection criterion. Neither arm used target inputs during pretraining. Source checkpoint SHA-256/state/history identities remain in the gitignored source manifest.",
        "",
        "| Source seed | Plain retrain fold9 macro AUROC | Placebo fold9 macro AUROC | Plain selected epoch | Placebo selected epoch | Origin |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for s in SEEDS:
        pair = {r["arm"]: r for r in source_rows if r["source_seed"] == s}
        a, b = pair["plain"], pair["placebo"]
        lines.append(
            f"| {s} | {a['fold9_macro_AUROC']:.6f} | {b['fold9_macro_AUROC']:.6f} | {a['selected_epoch']} | {b['selected_epoch']} | {'Existing' if s <= 15103 else 'New'} |"
        )
    lines += [
        "",
        f"Mean fold9 macro AUROC: plain **{np.mean([r['fold9_macro_AUROC'] for r in source_rows if r['arm'] == 'plain']):.6f}**; placebo **{np.mean([r['fold9_macro_AUROC'] for r in source_rows if r['arm'] == 'placebo']):.6f}**.",
    ]
    for hospital, r in datasets.items():
        lines += [
            "",
            f"## {'Part B' if hospital == 'CPSC' else 'Part C'} — {hospital} validation",
            "",
            "All10 paired source seeds ×3 fixed label draws ×3 budgets per arm:180 fitted-vector slots per hospital. CPSC reuses54 authenticated historical validation fits and adds126 fits; Georgia has180 new fits. Target fits use the unchanged200-update full-FT recipe and save no checkpoints. A/B are disjoint500-record validation roles; all draws are disjoint from both. Every model is evaluated on the same B records, not independent cohorts.",
            "",
            "| Budget | Arm | A macro AP | A macro AUROC | B macro AP | B macro AUROC | B CD AP | B STTC AP |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
        for v in r["metrics"]:
            lines.append(
                f"| {v['budget']} | {v['arm']} | {v['A_macro_AP']:.6f} | {v['A_macro_AUROC']:.6f} | {v['macro_AP']:.6f} | {v['macro_AUROC']:.6f} | {v['CD_AP']:.6f} | {v['STTC_AP']:.6f} |"
            )
        lines += [
            "",
            "| Budget | Placebo − plain macro AP | Paired hierarchical95% CI | Valid replicates | Macro AUROC difference [95% CI] |",
            "|---:|---:|---|---:|---|",
        ]
        for v in r["contrasts"]:
            lines.append(
                f"| {v['budget']} | {v['AP']['estimate']:+.6f} | {interval(v['AP'])} | {v['AP']['CI95']['valid']} | {v['AUROC']['estimate']:+.6f} {interval(v['AUROC'])} |"
            )
        lines += [
            "",
            f"![Per-seed AP effects](../../figures/phase5/{figures[hospital].name})",
            "",
            "Points show per-seed differences averaged over three paired draws, not per-seed confidence intervals. Hierarchical intervals first resample paired source seeds, then one shared B-record/known-group resample across both arms, all selected seeds, draws and labels.10000 replicates, seed20261201, percentile95%, minimum9500 supported; no redraw or undefined-value imputation. Source-training randomness is included; uncertainty remains conditional on the available training cohort, fixed label draws, source-selection rule and known record linkage. These reused validation results are descriptive development evidence, not confirmatory hospital-test findings.",
        ]
    lines += [
        "",
        "## Georgia label-blind preparation",
        "",
        "| Role | Records | CD positive proxy | STTC positive proxy | CD prevalence | STTC prevalence |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for role in ("pool", "A", "B"):
        v = summary[role]
        lines.append(
            f"| {role} | {v['records']} | {v['positive'][0]} | {v['positive'][1]} | {v['prevalence'][0]:.6f} | {v['prevalence'][1]:.6f} |"
        )
    lines += [
        f"| Locked test | {summary['test_records']} | Not accessed | Not accessed | Not accessed | Not accessed |",
        "",
        f"All{summary['survey_records']} original20% survey records are in the pool. Remaining canonical IDs were split label-blind, seed79101; no future test label or signal was opened. Known native duplicates had already been canonicalized under the Experiment3 audit; groups stay together. Pool labels alone stratified A/B, seed79102. The nested label-blind draws contain{summary['all_draws_union']} distinct records across budgets/draws. A/B, validation/draw, pool/test, survey/test and known-group overlaps are all zero.",
        "",
        "Membership lists, their SHA-256 values, raw-input fingerprints and adaptation-only bundle manifest are private. Test IDs live only in the separate ID-only assignment; the model-facing bundle stores only one-way exclusion hashes. The test prevalence will be reported only after a later authorized scoring operation.",
        "",
        "## Interpretation and preservation",
        "",
        "Georgia is a dataset/domain and not proof of a single homogeneous hospital/device population. Endpoints are recorded-code CD/STTC annotation-union proxies, not adjudicated diagnoses; omitted target annotations are not clinical proof of absence. Patient linkage and PTB-target waveform overlap remain unverified under the original audit's limitations. More source seeds do not remove those limitations. The old Phase3 survey had already been used for domain feasibility; it is excluded from the future test. Historical all-domain code/duplicate audits and original stratification had seen header labels; the present ID-only split does not erase that history.",
        "",
        "The Georgia confirmatory protocol is written after these development results and must be committed by the author before explicit continuation. No Georgia test evaluation is performed here. All earlier source checkpoints/results/protocols and consumed operations are preserved unchanged. Public source code, aggregates and figures are uncommitted; private arrays/checkpoints/memberships remain gitignored.",
    ]
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("\n".join(lines) + "\n")
    p = [
        "# Phase5 — Georgia pre-registered test protocol",
        "",
        "Status: fixed after Phase5 development; awaiting author commit and explicit continuation. **STOP: no test access is authorized in this task.**",
        "",
        "## Question and fixed sources",
        "",
        "Does placebo within-source MK-MMD pretraining improve Georgia CD/STTC macro AP over the matched lambda=0 plain-retrain pipeline after full FT? Use only the20 existing hash-verified Phase5 sources:10 per arm, seeds15101–15110. Both arms used PTB-XL only; no target-domain source pretraining. No source retraining/replacement, lambda search, hyperparameter changes, calibration, threshold selection or tuning is permitted.",
        "",
        "Methods: **(a)** plain-retrain source-only, zero target labels; **(b)** plain retrain +full FT; **(c)** placebo lambda0.1 +full FT. Budgets64/128/256; draw seeds16101/16102/16103;30 seed×draw episodes at each budget. Reuse exact recorded Georgia nested F/G memberships. Source-only has10 physical predictors, aliased across budgets/draws. Full FT uses200 updates, replacement batch32, constant AdamW lr3e-4, weight decay0.01, betas(0.9,0.99), eps1e-8, seed([78501,source_seed,draw_seed,budget]), all modules/head trainable, source-only normalization and frozen source settings. Source pretraining remains the matched two-forward pipeline,50 epochs, source128/independent PTB32 batches, seven-module biased MK-MMD, bandwidth multipliers0.5/1/2/4, and existing fold9 AP/BCE checkpoint selection.",
        "",
        "## Test population and label protection",
        "",
        f"Locked Georgia test: **{summary['test_records']} canonical records**, fixed by the ID-only assignment. The original{summary['survey_records']}-record20% survey subset is wholly in the{summary['pool']['records']}-record adaptation pool and never test. The remaining records were allocated50/50 **label-blind** using fixed seed79101 and deterministic SHA256 group ordering. Known groups remain together. Label-blind allocation was adopted explicitly to keep test labels untouched during this pre-registration phase. Test prevalence is deliberately unknown and will be reported after scoring, not estimated from the pool.",
        "",
        "Preserve the Experiment3 native-duplicate canonicalization, canonical lead order, physical mV conversion, first10s100Hz12x1000 processing and short-record exclusions. Do not introduce post-outcome exclusions or a new duplicate search. Stable patient linkage is unknown; retained canonical record IDs are the existing known singleton groups. This is annotation-proxy domain replication, not clinical validation or proven person-independent sampling. Historical Phase3 audits/20–80 stratification used headers, but only the old survey was scored; the future test was not used for model scoring in this task. No claim of never-decoded historical outcomes is made.",
        "",
        "## Single primary",
        "",
        "**(c) minus(b), mean episode-wise CD/STTC macro AP at128 labels**, equally over10 source seeds and three fixed draws per seed. Never compute AP of mean scores or pool repeated participants into a larger cohort. Use10000 paired hierarchical bootstrap replicates, fixed seed20261201: first multinomial-resample10 paired source seeds; then one shared canonical-record/known-group resample across all selected seeds, paired arms, draws and labels. Preserve shared-cohort dependence. Use the validation implementation's exact tie-aware ranking calculation, strict means, percentile2.5/97.5 bounds, minimum9500 supported finite draws, unsupported counts reported and no redrawing/imputation. **Success requires the two-sided95% CI lower bound strictly greater than zero.** Undefined/insufficient-support results are invalid and cannot establish success. This includes source-seed variability but omits training-cohort, label-draw-selection and unknown patient-linkage uncertainty.",
        "",
        "## Descriptive only",
        "",
        "All64/256 results, macro AUROC, CD/STTC AP/AUROC, fixed0.5 accuracy/precision/recall/specificity/F1/balanced accuracy/NPV and per-seed differences are descriptive only. Source-only is a zero-target-update reference. No larger hypothesis family, test-set threshold optimization or post-hoc primary replacement. Threshold decision: probability>=0.5. Undefined denominators remain N/A; unknown labels never become negatives. CD/STTC order and the frozen V2 annotation-union decoder/mapping stay fixed; require>=95% observed per label and both classes before metrics, reporting observed/unknown and class denominators. Failed validity gates stop without changing rules.",
        "",
        "## Replay, prediction freeze and one evaluation",
        "",
        "Only after this final protocol is committed and the author explicitly says continue: verify exact committed protocol bytes and fresh unused private output. Verify all20 existing source checkpoint/state/normalization hashes, the Georgia bundle/assignment/draw/split hashes and frozen numerical dependencies. Regenerate all180 full-FT states by deterministic replay against their stored A AND B validation arrays **BITWISE**, including shape/dtype/order, final state hash,200-update history, recipe and source bindings. Any mismatch stops before test signals; no tolerance or setting adjustment is permitted. Authentication optimization only, no new target training experiment and no target checkpoints.",
        "",
        "Training/replay reads only the adaptation-only bundle under the test-exclusion guard. Only after all180 replay matches may a separate final-prediction capability load test SIGNALS ONLY. Use CPUfloat32/four threads, eval/no_grad,batch128 and unchanged preprocessing/normalization; verify state/BN hashes before/after. Read only unbuffered physical header prefixes and signal-only MAT values; no full annotation/header checksum before the seal. Generate all270 logical method×budget×episode arrays, with190 physical predictors (180 adapted states+10 source-only aliases), shape(test_records,2), finite[0,1]. Save privately and SHA256-hash the complete roster, source/state/code/normalization/row-order/group identities and timestamp before one immutable prediction seal. No labels unlock on a partial roster.",
        "",
        "Only then consume one outcome-access operation and decode labels with frozen V2 semantics. Score exactly the primary and listed descriptive outputs using shared seed/record resamples. Single evaluation: no automatic retry/reset, prediction regeneration after labels, tuning, metric switching or analysis changes after results. Preserve any failure and report incomplete/invalid. Never load CPSC test data, PTB-XL fold10 or unrelated sealed material. No reuse of Phase4C authorizations/receipts. Raw inputs, IDs, predictions, model weights and operational manifests remain private; no commit/push/merge by the agent.",
        "",
        "## Fixed development evidence",
        "",
        "The evidence below motivated this test but is not its result. No favorable source seed/lambda/budget was selected; all10 paired seeds remain in the test estimand.",
        "",
        f"All14 new source fits completed alongside the six preserved historical sources. Mean PTB-XL fold9 macro AUROC across10 sources per arm: plain retrain **{np.mean([r['fold9_macro_AUROC'] for r in source_rows if r['arm'] == 'plain']):.6f}**, placebo **{np.mean([r['fold9_macro_AUROC'] for r in source_rows if r['arm'] == 'placebo']):.6f}**. The full per-seed/arm table and checkpoint-selection epochs are retained in the development report; these source metrics did not select or exclude seeds.",
        "",
        "| Validation domain | Budget | Placebo − plain macro AP | Hierarchical95% CI |",
        "|---|---:|---:|---|",
    ]
    for h, r in datasets.items():
        for v in r["contrasts"]:
            p.append(f"| {h} | {v['budget']} | {v['AP']['estimate']:+.6f} | {interval(v['AP'])} |")
    p += [
        "",
        f"Development report SHA256: `{sha(doc)}`. Source identities, validation fitted-state hashes, A/B prediction hashes and record bindings are frozen in the new private release binding. Validation plan: [PHASE5_VALIDATION_PLAN.md](PHASE5_VALIDATION_PLAN.md). Full evidence: [PHASE5_VALIDATION_RESULTS.md](PHASE5_VALIDATION_RESULTS.md).",
        "",
        "Stop for author commitment and explicit continuation. No Georgia test signal, label, prediction, prevalence or metric was loaded/computed to write this protocol.",
    ]
    protocol.write_text("\n".join(p) + "\n")
    refs = {
        h: [
            dict(
                report=str(p.relative_to(RUN)),
                sha256=sha(p),
                predictions_sha256=json.loads(p.read_text())["predictions_sha256"],
                state_hash=json.loads(p.read_text())["state_hash"],
            )
            for p in sorted((RUN / h / "fits").glob("*.json"))
        ]
        for h in datasets
    }
    new_json(
        RUN / "VALIDATION_RELEASE_BINDING.json",
        dict(
            protocol_sha256=sha(protocol),
            report_sha256=sha(doc),
            sources=source_rows,
            reference_fits=refs,
            Georgia_bundle_sha256=sha(GEORGIA / "MANIFEST.json"),
            references_sha256=digest(refs),
            test_accesses=0,
        ),
    )
    print("PHASE5_VALIDATION_AND_PROTOCOL_COMPLETE", str(protocol), flush=True)


if __name__ == "__main__":
    run()
