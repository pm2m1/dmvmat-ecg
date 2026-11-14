"""Aggregate-only reporting, consuming immutable FEAS results, never RESERVED."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .download import sha
from .prepare import write_new


def number(value):
    return "undefined" if value is None else f"{value:.4f}"


def report(root):
    root = Path(root)
    out = root / "results/phase3a"
    metrics = json.loads((out / "METRICS.json").read_text())
    split = json.loads((out / "SPLIT_SUMMARY.json").read_text())
    mapping = json.loads((out / "LABEL_MAPPING.json").read_text())
    audit = json.loads((out / "DATASET_AUDIT.json").read_text())
    download = json.loads(
        (root / "data/raw/cinc2021_v1_0_3/DOWNLOAD_VERIFICATION.json").read_text()
    )
    shift = json.loads((out / "MODULE_MMD.json").read_text())
    lines = [
        "# Cross-hospital zero-shot annotation-proxy feasibility",
        "",
        "Executed inference only. These are exploratory annotation-classification results, not clinical diagnostic negatives, external clinical validation, or future-event prediction. Source: PTB-XL v1.0.3. Targets: CinC2021 v1.0.3. No training, fine-tuning, normalization fitting or calibration fitting was performed.",
        "",
        "## Download and dataset audit",
        "",
        "Official release files: https://physionet.org/files/challenge-2021/1.0.3/. Eight maximum concurrent single-stream downloads, checksum-aware skip/resume, at most ten total tries per curl transfer, five-second retry wait. Initial aria2 downloads used max-tries10; runtime low-speed aborts prompted removal of that floor, then the permitted curl fallback with reusable HTTP2 connections. A16-file probe passed all hashes. A premature completion check during curl startup hashing was archived, not used as final verification; see results/phase3a/STARTUP_RACE_INCIDENT.json. No model ran during that incident. PTB, PTB-XL and INCART copies deliberately excluded: source overlap and non-target design. Publisher checksum list SHA256: `"
        + audit["publisher_index_sha256"]
        + "`.",
        "",
        "| Source | Expected files | Verified files | Bytes | Mismatches | Records | Short<10s | Long>10s | Flat-lead records | Duplicates removed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for source, value in download["sources"].items():
        a = audit["sources"][source]
        lines.append(
            f"| {source} | {value['expected_files']} | {value['downloaded_files']} | {value['bytes']} | {value['mismatches']} | {a['records']} | {a['shorter_than_10s']} | {a['longer_than_10s']} | {a['flat_lead_records']} | {a['duplicate_removed']} |"
        )
    lines += [
        "",
        "Full rate, length, lead-order, unit and gain inventories: [DATASET_AUDIT.json](../../results/phase3a/DATASET_AUDIT.json). Header/MAT/finite checks failed for "
        + str(len(audit["failures"]))
        + " records. Native exact duplicate totals: "
        + str(audit["native_exact_duplicate_total"])
        + ". Pair counts: `"
        + json.dumps(audit["duplicates_by_source_pair"], sort_keys=True)
        + "`. First lexicographic complete native-wave identity retained, independently of diagnoses; labels are never merged. Canonical lead/mV hash rounded at1e-9mV includes native rate and full length. It does not detect near-duplicates, crops or common patients.",
        "",
        "## Frozen annotation mapping and proxy-matched reference",
        "",
    ]
    for label in ["CD", "STTC"]:
        lines.append(
            f"- **{label}-proxy SNOMED**: "
            + ", ".join(mapping["proxy_snomed"][label])
            + ". **Matched fold9 SCP keys**: "
            + ", ".join(mapping["proxy_scp"][label])
            + "."
        )
    lines += [
        "",
        "Mapping hash: `"
        + sha(out / "LABEL_MAPPING.json")
        + "`. [LABEL_MAPPING.json](../../results/phase3a/LABEL_MAPPING.json) lists EVERY observed code, included or explicitly excluded, condition families and descriptive per-code tasks. Code164909002 is excluded because of the documented LBBB/fascicular conversion collision. AF is not a model output. Absent codes mean **not annotated**, never clinically negative. Source SCP keys are used as recorded, without inventing a likelihood cutoff. Only fold9 IDs are admitted before SCP decoding. Form-only SCP tags support the restricted ST/T proxy; this is deliberately different from the original full diagnostic superclass.",
        "",
        "The same named-condition unions are applied on both sides. Subtype/reporting completeness remains uncertain, including unspecified RBBB and nonspecific T/ST annotations. Therefore **part of any DROP can result from label-definition/reporting mismatch rather than true signal shift**. No clinically verified negative endpoint was established.",
        "",
        "## Frozen FEAS/RESERVED split",
        "",
        "Fixed seed74301, one deterministic joint-pattern Hamilton20/80 allocation after native duplicate and header-only short-record exclusions. No seed search. ID-only manifests remain git-ignored. RESERVED labels were used only for stratification, then discarded from model-facing caches; no RESERVED inference or subsequent label analysis.",
        "",
        "| Domain | FEAS | CD+ | STTC+ | RESERVED count only | Expected RESERVED CD+ | Expected RESERVED STTC+ | FEAS truncated | All short excluded | Split SHA256 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for domain, value in split["domains"].items():
        lines.append(
            f"| {domain} | {value['FEAS_records']} | {value['FEAS_positive_CD_STTC'][0]} | {value['FEAS_positive_CD_STTC'][1]} | {value['RESERVED_records']} | {value['EXPECTED_RESERVED_positive_CD_STTC'][0]:.1f} | {value['EXPECTED_RESERVED_positive_CD_STTC'][1]:.1f} | {value['FEAS_truncated_first10']} | {value['all_short_excluded_before_split']} | `{value['split_sha256']}` |"
        )
    lines += [
        "",
        "Expected RESERVED positives are **estimates**, calculated from FEAS prevalence times RESERVED size; they are not observed RESERVED label counts.",
        "",
        "## Frozen inference and reference",
        "",
        "Historical no-augmentation five-view and matched flat CNN; both already-trained seeds101/202, selected previously on five-class fold9 macro AUROC. No target-based checkpoint selection. Mean seed-wise metrics are reported, not a new ensemble. Full10s [12,1000] inference uses exact historical train-only scalar normalization; rhythm and frequency transforms are unchanged. Longer records use first10s; shorter records are excluded, never padded. Anti-aliasing: scipy.resample_poly with default Kaiser5 filter and line boundary extension. Lead reordering and header gain/unit conversion precede resampling. No target normalization fit. All checkpoint and statistics hashes appear in [RUN_PROTOCOL.json](../../results/phase3a/RUN_PROTOCOL.json); per-pass runtimes in [PREDICTION_SEAL.json](../../results/phase3a/PREDICTION_SEAL.json). Five-view212261 parameters; flat206117.",
        "",
        "## Fold9 references and FEAS metrics",
        "",
        "Continuous scores throughout. AP random-ranking reference is the annotation-positive prevalence. Brier and mean predicted probability are diagnostic only: the model outputs full superclass probabilities rather than calibrated probabilities of these restricted proxies. Source fold9 full-class and proxy metrics share the same frozen scores. DROP uses **proxy-matched fold9 only**.",
        "",
        "| Population/definition | Model | Label | AUROC | AP | Prevalence | Brier | Mean predicted | AUROC95% CI | AP95% CI |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for population, methods in metrics["metrics"].items():
        for model, rows in methods.items():
            for j, row in enumerate(rows):
                ci = row.get("CI95", {})

                def fmt(key):
                    return (
                        "—"
                        if key not in ci
                        else "[" + number(ci[key]["lower"]) + ", " + number(ci[key]["upper"]) + "]"
                    )

                lines.append(
                    f"| {population} | {model} | {['CD', 'STTC'][j]} | {number(row['auroc'])} | {number(row['ap'])} | {number(row['prevalence_AP_reference'])} | {number(row['brier'])} | {number(row['mean_predicted'])} | {fmt('auroc')} | {fmt('ap')} |"
                )
    lines += [
        "",
        "| Domain | Model | Label | AUROC DROP | AP DROP | Prior-adjusted source AP |",
        "|---|---|---|---:|---:|---:|",
    ]
    for domain, methods in metrics["drops"].items():
        for model, rows in methods.items():
            for j, row in enumerate(rows):
                lines.append(
                    f"| {domain} | {model} | {['CD', 'STTC'][j]} | {number(row['auroc_DROP'])} | {number(row['ap_DROP'])} | {number(row['source_AP_prior_reweighted_to_target'])} |"
                )
    lines += [
        "",
        "DROP intervals, undefined bootstrap counts and all descriptive per-code prevalence/AUROC/AP values are in [METRICS.json](../../results/phase3a/METRICS.json). Per-code AUROC uses the associated CD/STTC output, **not a dedicated per-code predictor**. Missing-class tasks have undefined AUROC and zero-positive AP is undefined, never converted into success. Pure-prior reweighting of source AP is a descriptive benchmark, not proof that annotation semantics match. Raw AP drops are prevalence-dependent.",
        "",
        "## Descriptive per-code results",
        "",
        "Every task is a recorded code versus its absence; absence is not a clinical negative. No dedicated subtype head is invented. The LBBB task uses unambiguous complete-LBBB code733534002 only; ambiguous164909002 is excluded.",
        "",
        "| Population | Code/condition | Proxy score | Annotated+ | Prevalence | Five-view AUROC | Flat AUROC |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for population, codes in metrics["per_code_descriptive"].items():
        for code, value in codes.items():
            lines.append(
                f"| {population} | {code} {value['name']} | {value['score_output']} | {value['positive']} | {number(value['annotation_prevalence'])} | {number(value['models']['multiview']['auroc'])} | {number(value['models']['flat_cnn']['auroc'])} |"
            )
    lines += [
        "",
        "Bootstrap:2000 record-level resamples, shared across labels/models/training seeds within each population; seeds do not create extra people. Conditional percentile intervals; undefined samples excluded from quantiles with exact counts, no invented negatives. No patient IDs in targets; correlated repeated patients may make record intervals too narrow. Fold9 also has repeated patients; the requested record-level reference CI shares that limitation. Source and target resamples independent. No confirmatory multiplicity or transfer advantage claim.",
        "",
        "## Seven-module shift evidence",
        "",
        "Primary five-view only, both source seeds. Deterministic label-free hash sampling at most256 records/group, pooled median RBF bandwidth,999 permutations, (1+extreme)/1000; Holm7 within domain/seed. Native feature scales differ, so MMD values across modules are descriptive and bandwidth-normalized, not a causal adaptation ranking.",
        "",
        "| Domain | Seed | Module | MMD² | Permutation p | Holm7 p |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for domain, seeds in shift.items():
        for seed, modules in seeds.items():
            for module, value in modules.items():
                lines.append(
                    f"| {domain} | {seed} | {module} | {value['statistic']:.5f} | {value['p']:.4f} | {value['p_Holm7']:.4f} |"
                )
    lines.append("")
    for domain, seeds in shift.items():
        names = list(next(iter(seeds.values())))
        means = {
            name: float(np.mean([row[name]["statistic"] for row in seeds.values()]))
            for name in names
        }
        ordered = sorted(means, key=means.get, reverse=True)
        significant = {
            name: sum(row[name]["p_Holm7"] <= 0.05 for row in seeds.values()) for name in names
        }
        lines.append(
            f"{domain}: largest descriptive median-kernel MMD values in {', '.join(ordered[:3])}. Holm-significant seed counts out of2: `{json.dumps(significant, sort_keys=True)}`. This is representation-shift evidence, not measured adaptation benefit."
        )
        lines.append("")
    lines += [
        "",
        "Significant module shift is selection evidence only. Different module shifts may motivate selective adaptation; uniform shift provides less differentiation. Neither proves that updating any module improves prediction.",
        "",
        "## Pre-stated feasibility verdict",
        "",
        "Rule frozen in RUN_PROTOCOL.json before scoring: primary five-view macro proxy AUROC DROP≤−0.05, OR both target AP95% upper bounds below source AP reweighted to target priors; AND expected RESERVED positives≥150 each; AND preprocessing/mapping checks pass. These criteria refer only to annotation proxies. No clinical-negative equivalence is claimed.",
        "",
        "| Domain | Verdict | Macro proxy AUROC DROP | AP alternative | Expected support |",
        "|---|---|---:|---|---|",
    ]
    for domain, value in metrics["verdicts"].items():
        lines.append(
            f"| {domain} | {value['status']} | {number(value['macro_proxy_matched_AUROC_DROP'])} | {value['AP_prior_adjusted_condition']} | {value['expected_reserved_support_at_least150_each']} |"
        )
    suitable = [d for d, v in metrics["verdicts"].items() if v["status"] == "SUITABLE"]
    suitable.sort(
        key=lambda d: (
            metrics["verdicts"][d]["macro_proxy_matched_AUROC_DROP"] is None,
            metrics["verdicts"][d]["macro_proxy_matched_AUROC_DROP"] or 0,
            -min(split["domains"][d]["EXPECTED_RESERVED_positive_CD_STTC"]),
        )
    )
    lines += [
        "",
        "Recommended primary target: "
        + (suitable[0] if suitable else "none under frozen criterion")
        + ". Backup: "
        + (suitable[1] if len(suitable) > 1 else "none meets all criteria")
        + ". Domain selection on FEAS makes any later project a development-selected target study, not confirmation that selection generalizes.",
        "",
        "## Phase3B design proposal — NOT executed",
        "",
        "Source PTB-XL; target only an eligible RESERVED cohort after a separate reviewed label/patient-independence protocol. Budget32/64/128 distinct labeled records, three fresh source seeds if permitted, repeated paired draws. Arms: source-only, head-only, full fine-tune, parameter-matched random, re-engineered MMD/FormulationA, and a conditional literature-faithful DMV-MAT arm. Prefer fixed target update schedules to avoid concealing hundreds of stopping labels; otherwise stopping≥256 must be counted separately. Every evidence, fitting, stopping and calibration label counts; Full-cost must access the same union. Keep an untouched future target evaluation, account for lack of patient IDs, overlapping code annotations, teacher/source training history and record duplicates. No future split is frozen by this report.",
        "",
        "**Faithful-arm evidence blocker:** the local secondary review `DMV-MAT Paper and Official Code Teardown.pdf` describes Cui etal., TNNLS36(2):2917–2930 (2025), DOI10.1109/TNNLS.2024.3350085. It reports JS transfer scores and a two-component GMM selection for feature extractors, but released model/GMM definitions are missing, repository licensing unresolved, and MMD loss calls are commented out. MK-MMD pretraining is NOT verified from the available offline evidence. The requested MK-MMD-pretraining+transfer-score+GMM feature-only arm must therefore remain conditional on primary-paper/code/license verification; otherwise name it an inspired variant, not a faithful reproduction. MMD module-selection evidence and an alignment training loss are distinct. No network was used for this Part2 inquiry.",
        "",
        "## Limitations and integrity",
        "",
        "1. **Explicit deviation:** use actual full10s five-view/flat pipeline, replacing original2.5s-window request. No windows or test-time averaging.",
        "2. **Explicit deviation:** no PTB-XL-vs-target waveform duplicate audit, as authorized. Excluding named PTB/PTBXL/INCART archives does not empirically prove absence of overlap or shared patients.",
        "3. No stable target patient identities; exact native duplicates do not establish clinical independence. Multi-source domains are dataset groupings, not verified homogeneous hospitals/devices.",
        "4. Unlisted diagnoses are unannotated, not confirmed absence. Apparent loss may arise from label mismatch, coding completeness, prevalence or truncation, as well as signal shift. Clinical external validation is out of scope.",
        "5. PTB-XL fold9 was used for historical checkpoint selection; its reference is optimistic development evidence. Fold10, Phase2 O and all protected Phase2 artifacts were untouched.",
        "6. RESERVED was used only for stratification and ID manifests, never inference/diagnostics; expected counts are estimates. Future patient grouping and endpoint adjudication remain conditions.",
        "7. Bootstrap intervals condition on already-trained models, omit source-training/selection uncertainty and may understate within-person dependence. No new transfer experiment or clinical utility established.",
        "",
        "## Files, tests and repository",
        "",
        "New source: `src/dmvmat_ecg/phase3/`; new scripts: `scripts/phase3/`; synthetic tests: `tests/phase3/`; aggregate artifacts: `results/phase3a/`; this report. Raw data, caches, split identities, predictions and representations stay under git-ignored `data/`. No raw/patient artifacts staged, no commit/push/merge/PR. Preservation and test receipts are recorded separately in results/phase3a; old tracked files are checked against the starting hashes.",
        "",
    ]
    path = root / "docs/phase3/CROSS_HOSPITAL_FEASIBILITY.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write("\n".join(lines))
    # All points below originate in actual frozen metrics.
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for j, label in enumerate(["CD-proxy", "STTC-proxy"]):
        populations = ["fold9", *metrics["verdicts"]]
        for offset, model in enumerate(["multiview", "flat_cnn"]):
            values = [metrics["metrics"][p][model][j]["auroc"] for p in populations]
            axes[j].plot(range(4), [np.nan if v is None else v for v in values], "o-", label=model)
        axes[j].set(title=label, ylabel="AUROC", xticks=range(4), xticklabels=populations)
        axes[j].tick_params(axis="x", rotation=20)
        axes[j].legend()
        axes[j].grid(alpha=0.2)
    for extension in ["png", "pdf"]:
        fig.savefig(figdir / ("zero_shot_proxy_AUROC." + extension), dpi=240)
    plt.close(fig)
    write_new(
        out / "REPORT_COMPLETE.json",
        {
            "report_sha256": sha(path),
            "suitable_domains": suitable,
            "final_outcome": "FEASIBLE — PROCEED TO PHASE 3B DESIGN"
            if suitable
            else "NOT FEASIBLE — no domain satisfied the frozen annotation-proxy criterion",
        },
    )
