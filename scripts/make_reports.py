"""Generate scientific tables from completed immutable experimental records."""

import json
from pathlib import Path

from dmvmat_ecg.evaluation.state import Study
from dmvmat_ecg.labels.superclasses import CLASSES
from dmvmat_ecg.utils.io import sha, write_json

root = Path(__file__).resolve().parents[1]
study = Study(root)
if not (root / "results/evaluation_completion.json").exists():
    raise PermissionError("No completed locked evaluation")
result = json.loads((root / "results/locked_metrics.json").read_text())
config = json.loads((root / "configs/study.json").read_text())
audit = json.loads((root / "results/dataset_audit.json").read_text())
full = [r for r in study.runs() if r["status"] == "successful" and r["metadata"]["stage"] == "full"]
names = ["xresnet101", "xresnet101_final", "flat_cnn", "multiview", "multiview_aug"]


def table(headers, rows):
    return (
        "| "
        + " | ".join(headers)
        + " |\n|"
        + "|".join(["---"] * len(headers))
        + "|\n"
        + "\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows)
        + "\n"
    )


def ci(x):
    return f"[{x[0]:+.5f}, {x[1]:+.5f}]"


def scientific_contrast(key):
    c = result["comparisons"][key]
    interval = c["auroc_multiplicity_ci"]
    interpretation = (
        "positive conditional evidence"
        if interval[0] > 0
        else "negative conditional evidence"
        if interval[1] < 0
        else "inconclusive conditional evidence"
    )
    return f"{key}: Δmacro AUROC={c['delta_auroc']:+.5f}, {('primary97.5% Bonferroni CI' if c['primary'] else 'secondary95% unadjusted CI')} {ci(interval)} ({interpretation}); Δmacro AP={c['delta_ap']:+.5f}, exploratory95%CI {ci(c['ap_ci95'])}."


sections = [
    "# Completed ECG experiments — locked fold10 results\n",
    "All numerical results below derive from the completed PTB-XL1.0.3 fits and one joint locked test evaluation. Scores are arithmetic means of seed-specific metrics, not probability ensembles. Classification is multilabel ECG diagnosis, not future events, clinical risk or cross-cohort validation.\n",
    "## Population and prevalence\n",
    f"Train {audit['partitions']['train']['records']:,}, validation {audit['partitions']['validation']['records']:,}, test {audit['partitions']['test']['records']:,} recordings. Test patients {audit['partitions']['test']['patients']:,}. No detected patient or exact waveform overlap.\n",
    table(
        ["Label", "Test positives", "Prevalence / AP random-ranking reference"],
        [
            [
                label,
                audit["partitions"]["test"]["positives"][label],
                f"{result['prevalence'][j]:.5f}",
            ]
            for j, label in enumerate(CLASSES)
        ],
    ),
    "## Complete registered leaderboard\n",
    "The order below is prespecified, not an outer-score-selected winner. XResNet-final is a checkpoint alias, not another model fit.\n",
    table(
        [
            "Procedure",
            "Macro AUROC (95% CI)",
            "Macro AP (95% CI)",
            "Micro AP",
            "Brier",
            "ECE",
            "Macro F1",
        ],
        [
            [
                n,
                f"{result['summary'][n]['macro_auroc']:.5f} {ci(result['summary'][n]['auroc_ci95'])}",
                f"{result['summary'][n]['macro_ap']:.5f} {ci(result['summary'][n]['ap_ci95'])}",
                *[
                    f"{result['summary'][n][m]:.5f}"
                    for m in ("micro_ap", "mean_brier", "mean_ece", "macro_f1")
                ],
            ]
            for n in names
        ],
    ),
    "F1 uses independently validation-selected per-label thresholds. Brier/ECE use raw sigmoid probabilities: no probability calibration model was fitted. ECE has ten equal-width bins; empty bins contribute zero. The XResNet max-window procedure has different probability semantics from single full-record CNN prediction.\n",
    "## Registered paired comparisons\n",
]
for key in result["comparisons"]:
    sections.append(scientific_contrast(key) + "\n")
sections += [
    "Two primary AUROC contrasts use97.5% Bonferroni percentile intervals. Augmentation is secondary95%. AP, per-class comparisons and all exploratory intervals have no familywise superiority interpretation. Shared patient-cluster resampling,2,000 valid draws, seed70501; all patients' recordings and paired seeds share multiplicities. Intervals condition on fitted models, validation selection and this split. They omit retraining, model-selection and external-population uncertainty. Seeds are not independent patients.\n",
    "## Per-label ranking\n",
]
for metric, title in [("per_label_auroc", "AUROC"), ("per_label_ap", "AP")]:
    sections.append(
        table(
            ["Procedure"] + list(CLASSES),
            [[n] + [f"{x:.5f}" for x in result["summary"][n][metric]] for n in names],
        )
    )
    sections.append(
        f"{title}: additional per-label conditional95% intervals are in `results/locked_metrics.json`; per-class differences are exploratory.\n"
    )
sections.append("## Thresholds, probability quality and confusion counts\n")
for name in names:
    sections.append(f"### {name}\n")
    for seed in config["seeds"]:
        item = result["runs"][f"{name}:{seed}"]["validation_thresholds"]
        sections.append(f"Seed {seed}; thresholds chosen on fold9, never fold10.\n")
        sections.append(
            table(
                [
                    "Label",
                    "Threshold",
                    "Precision",
                    "Recall",
                    "Specificity",
                    "F1",
                    "Brier",
                    "ECE",
                    "TN/FP/FN/TP",
                ],
                [
                    [
                        label,
                        f"{v['threshold']:.2f}",
                        *[
                            f"{v[m]:.5f}"
                            for m in ("precision", "recall", "specificity", "f1", "brier", "ece")
                        ],
                        "/".join(str(x) for row in v["confusion_matrix"] for x in row),
                    ]
                    for label, v in item["labels"].items()
                ],
            )
        )
    sections.append(
        "Fixed0.50 results are retained for every seed/label in the machine-readable metrics. Threshold changes do not change AP/AUROC. No deployment threshold is recommended.\n"
    )
sections += [
    "## Development checkpoints and compute\n",
    table(
        [
            "Model",
            "Seed",
            "Best epoch",
            "Validation AUROC",
            "Validation AP",
            "Parameters",
            "Training min (no validation)",
            "Validation min",
        ],
        [
            [
                r["metadata"]["model"],
                r["metadata"]["seed"],
                r["metadata"]["best_epoch"],
                f"{r['metadata']['validation_macro_auroc']:.5f}",
                f"{max(r['metadata']['history'], key=lambda h: h['validation_metrics']['macro_auroc'])['validation_metrics']['macro_ap']:.5f}",
                r["metadata"]["parameters"],
                f"{r['metadata']['training_seconds'] / 60:.2f}",
                f"{sum(h['validation_prediction_seconds'] for h in r['metadata']['history']) / 60:.2f}",
            ]
            for r in full
        ],
    ),
    "All full fits completed50 epochs. CPU float32, four threads per fit, up to three concurrent workers. Reported training durations exclude validation and are not the study wall time. Best checkpoint is earliest maximal validation macro AUROC; final epoch also preserved. Reference matches pinned architecture but modern software/dataset/optimizer/selection differences prevent exact numerical reproduction.\n",
    table(
        ["Procedure", "Seed", "Test inference seconds"],
        [
            [n, s, f"{result['runs'][f'{n}:{s}']['inference_seconds']:.3f}"]
            for n in names
            for s in config["seeds"]
        ],
    ),
    "## Integrity and limitations\n",
    f"Actual predictor attempts: {len(study.runs())}; successful {sum(r['status'] == 'successful' for r in study.runs())}; failed {sum(r['status'] == 'failed' for r in study.runs())}. This includes three training-only smoke fits, separate from eight full fits. Synthetic test fits do not use the real ledger. Unused reserve is not used after test opening.\n",
    "All registered predictions were generated/sealed before metrics. See `results/evaluation_lock.json`, `results/prediction_seal.json`, `results/evaluation_completion.json` and the ignored SQLite ledger. No post-test fitting, selection or recalibration. Raw data, checkpoints and per-person predictions remain local.\n",
    "Known published fold10 results inform the benchmark choice, so this is not independent hypothesis confirmation. Only two seeds, one official split and no external cohort. Unknown linkage/near-duplicates, annotation uncertainty, ECG version differences, derived-view redundancy and limited tuning remain. Patient bootstrap captures test-sampling variability conditional on models; no clinical utility or selective-transfer superiority is established.\n",
    "## Figures and underlying records\n",
    "PNG/PDF research figures are under `results/figures/`, with captions/checksums in `results/figure_inventory.json`. Per-fit aggregate histories in `results/baseline/` and `results/multiview/`; complete test metrics in `results/locked_metrics.json`. Individual predictions/checkpoints are excluded.\n",
]
(root / "docs/EXPERIMENT_RESULTS.md").write_text("\n".join(sections))

mv = result["summary"]["multiview"]
reference = result["summary"]["xresnet101_final"]
summary = [
    "# Final research summary\n",
    "## Executed work\n",
    f"New independent PTB-XL1.0.3 project; {audit['raw_records']:,} raw recordings, {audit['raw_patients']:,} patient IDs. {audit['eligible_records']:,} eligible diagnostic recordings, official folds1–8/9/10, patient-disjoint and no detected exact waveform leakage. All43,598 required100Hz files match publisher checksums; no500Hz data downloaded.\n",
    f"Completed {len(full)} full50-epoch fits plus three training-only smoke fits. Test access consumed once after the complete artifact lock; ten model/seed prediction sets sealed before metrics. All numerical findings: [experiment results](EXPERIMENT_RESULTS.md).\n",
    "## Baseline and our architecture\n",
    f"Architecture-faithful modern PyTorch XResNet1D101 adaptation: final-epoch macro AUROC {reference['macro_auroc']:.5f}, macro AP {reference['macro_ap']:.5f}. Difference from published approximately0.928 macro AUROC: {reference['macro_auroc'] - 0.928:+.5f}. This is contextual, not a paired study or proof of exact historical reproduction. Validation-best variant is separately reported.\n",
    f"Five-view no-augmentation model: macro AUROC {mv['macro_auroc']:.5f}, macro AP {mv['macro_ap']:.5f}; 212,261 parameters versus206,117 for the flat CNN and1,872,261 for XResNet. Five signal-appropriate CNN encoders, 64-dimensional embeddings, fusion320→128→64, five-label head. Rhythm is a learned temporal view; spectrum is deterministic Hann FFT, not independent information.\n",
    scientific_contrast("multiview-minus-flat_cnn") + "\n",
    scientific_contrast("multiview-minus-xresnet101") + "\n",
    scientific_contrast("multiview_aug-minus-multiview") + "\n",
    "A positive point estimate alone does not establish superiority. Individual diagnostic-label changes are tabulated with exploratory intervals; no blanket claim of improvement across labels. No probability ensemble across seeds and no test-fitted calibration. Macro AP is not macro AUROC; prevalence baselines accompany each AP.\n",
    "## Interpretation and next research\n",
    "This study tests a finite lightweight multiview design and a bounded augmentation against a matched flat CNN and strong reference pipeline. Any advantage is conditional on this benchmark and design. A negative result motivates inspecting view redundancy, model capacity and optimization with a new development-only protocol, not tuning against this opened test. Numerical proximity to0.92–0.93 AUROC does not establish clinical readiness or generalization.\n",
    "Future useful work: separately preregister view ablations, compare budgets/representations using newly separated development roles (still exploratory after dataset reuse), and establish an independently harmonized external ECG cohort. Do not claim selective transfer from this classification study. Seven module interfaces and synthetic selector checks are available; patient overlap removal and five-class endpoint harmonization for CinC2021 are unresolved. See [feasibility](TRANSFER_FEASIBILITY.md).\n",
    "## Reproducibility and release\n",
    "See [protocol](EXPERIMENT_PROTOCOL.md), [audit](DATASET_AUDIT.md), [architecture](ECG_ARCHITECTURE.md), [reference fidelity](BASELINE_REPRODUCTION.md) and [reproduction](REPRODUCIBILITY.md). Hardware AppleM5Pro48GiB, CPU fallback following failed XResNet synthetic MPS-gradient tolerance. Existing research repositories remain read-only. No GitHub remote or push; no data/checkpoints/individual scores published.\n",
]
(root / "docs/FINAL_RESEARCH_SUMMARY.md").write_text("\n".join(summary))
readme = [
    "# DMV-MAT ECG\n",
    "Independent PTB-XL1.0.3 research comparing a modern PyTorch XResNet1D101 adaptation, a capacity-matched flat CNN, and our modular five-view ECG network. The task is five-superclass multilabel ECG diagnosis; this is not future cardiovascular-event prediction, a clinical tool or an external transfer validation.\n",
    "## Verified dataset and split\n",
    "[PTB-XL1.0.3](https://physionet.org/content/ptb-xl/1.0.3/), CC-BY4.0, DOI10.13026/kfzx-aw45; Wagner et al., [dataset paper](https://doi.org/10.1038/s41597-020-0495-6). All43,598 required100Hz header/data files verified against publisher SHA256 checksums. Raw21,799 ECGs/18,869 patient IDs; eligible21,388 ECGs/18,617 patient IDs. Exclude411 recordings with no diagnostic superclass. Official train folds1–8:17,084; validation9:2,146; test10:2,158. No detected cross-role patient/exact-waveform overlap; unknown identity and near-duplicates remain limitations. No500Hz files downloaded.\n",
    "## Architecture and training\n",
    "Five CNN views: limb leads; V1–V3; V4–V6; learned temporal/rhythm representation (no fabricated RR measurements); Hann-FFT spectrum0.5–40Hz. Five64-dimensional embeddings, concatenation320, fusion128→64, five sigmoid outputs. Seven named internal modules remain available for future selective-transfer studies.212,261 parameters versus206,117 for the matched flat control. Derived views do not add independent measurements.\n",
    "Four prespecified variants ×two seeds101/202,50 epochs each, BCEWithLogitsLoss, AdamW lr0.01/weight decay0.01/betas0.9,0.99, OneCycleLR, batch128. Common CPU float32 after XResNet failed a preset synthetic MPS-gradient tolerance; Apple M5 Pro48GiB. Per-seed checkpoints selected by fold9record-level macro AUROC. Three training-only smoke fits precede eight full fits. No successful fit repeated, no test-guided tuning.\n",
    "The [reference benchmark](https://github.com/helme/ecg_ptbxl_benchmarking) is pinned atcdbf4e66d7e57d9b6a2657b6024716212b8d0afa (GPL3). Its architecture is reused with a compatible head helper; obsolete FastAI1 training is adapted to modern PyTorch. Reference uses random2.5s training crops and max probabilities over seven held-out windows; flat/multiview controls use full10s. Thus reference comparisons are pipeline comparisons. Our implementation is not exact historical numerical reproduction. Published~0.928 macro AUROC is context, not our achieved metric or an AP value.\n",
    "## Actual locked results\n",
    "Same fold10recordings, mean seed-wise metrics, no probability ensemble. All ten registered checkpoint/seed prediction sets sealed before metrics. XResNet-final is a no-extra-fit alias of the original final-epoch convention.\n",
    table(
        ["Procedure", "Macro AUROC", "Macro AP", "Mean Brier", "Mean ECE"],
        [
            [n]
            + [
                f"{result['summary'][n][metric]:.5f}"
                for metric in ("macro_auroc", "macro_ap", "mean_brier", "mean_ece")
            ]
            for n in names
        ],
    ),
    scientific_contrast("multiview-minus-flat_cnn") + "\n",
    scientific_contrast("multiview-minus-xresnet101") + "\n",
    "Primary AUROC intervals use97.5% Bonferroni coverage for two contrasts; AP intervals are exploratory95%. Patient-cluster bootstrap2,000 draws, paired seeds/shared test patients, conditional on fitted models. Positive point estimates alone do not establish superiority; inconclusive intervals do not establish equivalence. See complete per-label/threshold/calibration/seed results in the research report.\n",
    "## Reproduce and inspect\n",
    "Python3.12.14 with uv and `uv.lock`.\n\n```bash\nuv sync --frozen\nuv run pytest -q\nuv run ruff check src scripts tests\nuv run ruff format --check src scripts tests\n```\n",
    "Data acquisition, one-time audit, finite training and single-use evaluation commands are documented in [reproducibility](docs/REPRODUCIBILITY.md). Reproduce in a new independent workspace/ledger; do not delete an evaluation receipt or rerun data preparation to reset the existing experiment.\n",
    "Reports: [final summary](docs/FINAL_RESEARCH_SUMMARY.md), [complete results](docs/EXPERIMENT_RESULTS.md), [protocol](docs/EXPERIMENT_PROTOCOL.md), [data audit](docs/DATASET_AUDIT.md), [architecture](docs/ECG_ARCHITECTURE.md), [reference fidelity](docs/BASELINE_REPRODUCTION.md), [transfer feasibility](docs/TRANSFER_FEASIBILITY.md). FiguresPNG/PDF under `results/figures`, with captions/checksums in `results/figure_inventory.json`; aggregate histories and metrics under `results`.\n",
    "## Limits and future work\n",
    "Known published benchmark scores, two seeds, one official split, label uncertainty, repeated/unknown person linkage and limited tuning constrain inference. The lightweight CNN's local receptive field does not directly estimate RR intervals; view-specific usefulness requires separate ablations. Raw probabilities have not been recalibrated. Thresholds are validation-F1 choices, not clinical recommendations. No transfer superiority, external validity or clinical utility is established.\n",
    "CinC2021 contains PTB-XL/PTB; the full Challenge collection is not independent. Person/segment overlap and diagnostic-label harmonization must be resolved before a separately approved transfer study. Synthetic MMD/selector interfaces do not substitute for cross-cohort evidence.\n",
    "## License and release boundaries\n",
    "Code GPL-3.0-only, including attributed upstream architecture; source data CC-BY4.0 with separate attribution obligations. Raw signals/metadata, patient manifests, predictions, fitted checkpoints, local operational files and environments are ignored. `results/public_release_inventory.json` lists eligible aggregate/code artifacts for separate release review. No GitHub remote or publication authorization; previous cardiovascular repositories remain unchanged.\n",
]
(root / "README.md").write_text("\n".join(readme))
inventory = {}
for base in ["src", "scripts", "tests", "configs", "docs", "results"]:
    for p in sorted((root / base).rglob("*")):
        if (
            p.is_file()
            and "__pycache__" not in p.parts
            and p.suffix in (".py", ".json", ".md", ".png", ".pdf")
            and p.name != "public_release_inventory.json"
        ):
            inventory[str(p.relative_to(root))] = sha(p)
for filename in (
    "README.md",
    "LICENSE",
    "pyproject.toml",
    "uv.lock",
    ".gitignore",
    "data/README.md",
):
    inventory[filename] = sha(root / filename)
write_json(
    root / "results/public_release_inventory.json",
    dict(
        eligible_files=inventory,
        excluded=[
            "data/raw",
            "data/manifests",
            "private",
            ".venv",
            "checkpoints",
            "individual predictions",
        ],
        license="code GPL-3.0; source dataset CC-BY4.0",
        remote_publication_authorized=False,
    ),
)
print(json.dumps(dict(status="reports_complete", rows=len(full), attempts=len(study.runs()))))
