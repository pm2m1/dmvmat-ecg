# DMV-MAT ECG

Independent PTB-XL1.0.3 research comparing a modern PyTorch XResNet1D101 adaptation, a capacity-matched flat CNN, and our modular five-view ECG network. The task is five-superclass multilabel ECG diagnosis; this is not future cardiovascular-event prediction, a clinical tool or an external transfer validation.

## Verified dataset and split

[PTB-XL1.0.3](https://physionet.org/content/ptb-xl/1.0.3/), CC-BY4.0, DOI10.13026/kfzx-aw45; Wagner et al., [dataset paper](https://doi.org/10.1038/s41597-020-0495-6). All43,598 required100Hz header/data files verified against publisher SHA256 checksums. Raw21,799 ECGs/18,869 patient IDs; eligible21,388 ECGs/18,617 patient IDs. Exclude411 recordings with no diagnostic superclass. Official train folds1–8:17,084; validation9:2,146; test10:2,158. No detected cross-role patient/exact-waveform overlap; unknown identity and near-duplicates remain limitations. No500Hz files downloaded.

## Architecture and training

Five CNN views: limb leads; V1–V3; V4–V6; learned temporal/rhythm representation (no fabricated RR measurements); Hann-FFT spectrum0.5–40Hz. Five64-dimensional embeddings, concatenation320, fusion128→64, five sigmoid outputs. Seven named internal modules remain available for future selective-transfer studies.212,261 parameters versus206,117 for the matched flat control. Derived views do not add independent measurements.

Four prespecified variants ×two seeds101/202,50 epochs each, BCEWithLogitsLoss, AdamW lr0.01/weight decay0.01/betas0.9,0.99, OneCycleLR, batch128. Common CPU float32 after XResNet failed a preset synthetic MPS-gradient tolerance; Apple M5 Pro48GiB. Per-seed checkpoints selected by fold9record-level macro AUROC. Three training-only smoke fits precede eight full fits. No successful fit repeated, no test-guided tuning.

The [reference benchmark](https://github.com/helme/ecg_ptbxl_benchmarking) is pinned atcdbf4e66d7e57d9b6a2657b6024716212b8d0afa (GPL3). Its architecture is reused with a compatible head helper; obsolete FastAI1 training is adapted to modern PyTorch. Reference uses random2.5s training crops and max probabilities over seven held-out windows; flat/multiview controls use full10s. Thus reference comparisons are pipeline comparisons. Our implementation is not exact historical numerical reproduction. Published~0.928 macro AUROC is context, not our achieved metric or an AP value.

## Actual locked results

Same fold10recordings, mean seed-wise metrics, no probability ensemble. All ten registered checkpoint/seed prediction sets sealed before metrics. XResNet-final is a no-extra-fit alias of the original final-epoch convention.

| Procedure | Macro AUROC | Macro AP | Mean Brier | Mean ECE |
|---|---|---|---|---|
| xresnet101 | 0.92985 | 0.82759 | 0.08588 | 0.05153 |
| xresnet101_final | 0.92800 | 0.82241 | 0.09258 | 0.06734 |
| flat_cnn | 0.91733 | 0.80657 | 0.08811 | 0.04066 |
| multiview | 0.92139 | 0.81405 | 0.08694 | 0.03775 |
| multiview_aug | 0.92068 | 0.81698 | 0.08513 | 0.03789 |

multiview-minus-flat_cnn: Δmacro AUROC=+0.00406, primary97.5% Bonferroni CI [+0.00009, +0.00786] (positive conditional evidence); Δmacro AP=+0.00748, exploratory95%CI [-0.00078, +0.01600].

multiview-minus-xresnet101: Δmacro AUROC=-0.00847, primary97.5% Bonferroni CI [-0.01312, -0.00400] (negative conditional evidence); Δmacro AP=-0.01354, exploratory95%CI [-0.02253, -0.00408].

Primary AUROC intervals use97.5% Bonferroni coverage for two contrasts; AP intervals are exploratory95%. Patient-cluster bootstrap2,000 draws, paired seeds/shared test patients, conditional on fitted models. Positive point estimates alone do not establish superiority; inconclusive intervals do not establish equivalence. See complete per-label/threshold/calibration/seed results in the research report.

## Reproduce and inspect

Python3.12.14 with uv and `uv.lock`.

```bash
uv sync --frozen
uv run pytest -q
uv run ruff check src scripts tests
uv run ruff format --check src scripts tests
```

Data acquisition, one-time audit, finite training and single-use evaluation commands are documented in [reproducibility](docs/REPRODUCIBILITY.md). Reproduce in a new independent workspace/ledger; do not delete an evaluation receipt or rerun data preparation to reset the existing experiment.

Reports: [final summary](docs/FINAL_RESEARCH_SUMMARY.md), [complete results](docs/EXPERIMENT_RESULTS.md), [protocol](docs/EXPERIMENT_PROTOCOL.md), [data audit](docs/DATASET_AUDIT.md), [architecture](docs/ECG_ARCHITECTURE.md), [reference fidelity](docs/BASELINE_REPRODUCTION.md), [transfer feasibility](docs/TRANSFER_FEASIBILITY.md). FiguresPNG/PDF under `results/figures`, with captions/checksums in `results/figure_inventory.json`; aggregate histories and metrics under `results`.

## Limits and future work

Known published benchmark scores, two seeds, one official split, label uncertainty, repeated/unknown person linkage and limited tuning constrain inference. The lightweight CNN's local receptive field does not directly estimate RR intervals; view-specific usefulness requires separate ablations. Raw probabilities have not been recalibrated. Thresholds are validation-F1 choices, not clinical recommendations. No transfer superiority, external validity or clinical utility is established.

CinC2021 contains PTB-XL/PTB; the full Challenge collection is not independent. Person/segment overlap and diagnostic-label harmonization must be resolved before a separately approved transfer study. Synthetic MMD/selector interfaces do not substitute for cross-cohort evidence.

## License and release boundaries

Code GPL-3.0-only, including attributed upstream architecture; source data CC-BY4.0 with separate attribution obligations. Raw signals/metadata, patient manifests, predictions, fitted checkpoints, local operational files and environments are ignored. `results/public_release_inventory.json` lists eligible aggregate/code artifacts for separate release review. No GitHub remote or publication authorization; previous cardiovascular repositories remain unchanged.
