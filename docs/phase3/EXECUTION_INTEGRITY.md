# Execution integrity and interpretation supplement

This supplements CROSS_HOSPITAL_FEASIBILITY.md without modifying its frozen metrics or prediction seal. No new predictive analysis, calibration, training, mask selection or holdout inference is introduced.

## Principal finding

CPSC is the primary candidate for a separately reviewed annotation-proxy transfer design, Georgia the backup; Chapman–Ningbo is borderline under the pre-stated rule. This does not establish that selective adaptation will outperform full fine-tuning.

| Domain | Five-view macro proxy AUROC | Proxy-matched macro DROP | Verdict |
|---|---:|---:|---|
| CPSC | 0.78891 | -0.09866 | SUITABLE |
| Chapman-Ningbo | 0.85159 | -0.03598 | BORDERLINE |
| Georgia | 0.83558 | -0.05199 | SUITABLE |

The same five-view fold9 scores produce STTC AUROC0.92574/AP0.78814 under the full superclass, but AUROC0.85045/AP0.44598 under the restricted named-condition proxy. Thus the endpoint definition itself changes measured performance substantially even within the source population. All feasibility DROP calculations use the latter reference. Georgia’s raw STTC AP rises to0.73467 alongside a prevalence increase from0.1710 to0.4407; that is not evidence that AUROC or clinical diagnostic ability improved. Its AUROC falls to0.80796. No favorable endpoint or code was selected after seeing these results.

The aggregate CD/STTC scores are not dedicated IAVB, RBBB, LBBB or ST-elevation classifiers. The per-code tables are descriptive; rare codes with one or a few positives cannot support robust phenotype-specific conclusions. Absence of a code is not a clinically confirmed negative. Future source training should explicitly address the mismatch between the original five-superclass head and the restricted proxy task through a separately approved, clinically justified protocol; do not prune codes to improve the observed AUROC.

All MMD statistics are exploratory and can reflect annotation prevalence/case-mix as well as acquisition differences. The strongest CPSC shift is early precordial view_2 in both seeds, followed by frequency/fusion features; this motivates a hypothesis, not an adaptation benefit claim. Georgia shifts broadly across all modules. Geometry, abstention and label-cost-matched controls remain necessary in any future study.

## Acquisition audit details

| Source | Sampling rate counts | Duration min/median/max seconds | Unit sets | ADC gain range | Header lead orders |
|---|---|---|---|---|---|
| chapman_shaoxing | `{"500": 10247}` | 10.000/10.000/10.000 | `{"mV,mV,mV,mV,mV,mV,mV,mV,mV,mV,mV,mV": 10247}` | 1000–1000 | 1 |
| cpsc_2018 | `{"500": 6877}` | 6.000/12.000/144.000 | `{"mV,mV,mV,mV,mV,mV,mV,mV,mV,mV,mV,mV": 6877}` | 1000–1000 | 1 |
| cpsc_2018_extra | `{"500": 3453}` | 8.000/12.000/98.000 | `{"mV,mV,mV,mV,mV,mV,mV,mV,mV,mV,mV,mV": 3453}` | 1000–1000 | 1 |
| georgia | `{"500": 10344}` | 5.000/10.000/10.000 | `{"mV,mV,mV,mV,mV,mV,mV,mV,mV,mV,mV,mV": 10344}` | 1000–1000 | 1 |
| ningbo | `{"500": 34905}` | 10.000/10.000/10.000 | `{"mV,mV,mV,mV,mV,mV,mV,mV,mV,mV,mV,mV": 34905}` | 1000–1000 | 1 |

Raw recordings65826; exact duplicate exclusions840; unique native recordings64986; eligible records64915. There are74 raw short recordings but only71 unique short exclusions: three short Georgia records were also duplicate copies. These are overlapping exclusion flags, not additive disjoint counts. The eligibility set contains12983 FEAS records and51932 RESERVED records. RESERVED outcomes were used only to stratify, never for later model scoring or label summaries. The report gives only RESERVED sizes and prevalence-based expected positives.

Flat-lead flags denote exactly constant leads and were reported without adding a post-score exclusion. Particularly,1480 Ningbo raw records have at least one such lead. Eligible records with finite, correctly decoded flat leads remain included. No waveform is repaired, padded, duplicated or replaced with synthetic data.

The one initial audit error was a loader assumption: S23074.hea declares JS23074.mat. The corrected loader follows the header file_name, leaving both raw filenames and all labels unchanged. No exclusion was added. The initial failed audit and resolution are preserved in DATASET_AUDIT_INITIAL_ALIAS_FAILURE.json and PARSER_ALIAS_RESOLUTION.json. Independent WFDB-versus-MAT checks on one FEAS record from each source agreed exactly in mV and produced finite source-normalized100Hz views; see REAL_ADAPTER_CHECKS.json.

The full10s inference and omitted PTBXL waveform cross-match are the two human-approved deviations. Full native target duplicate checks were performed; waveform matches to PTBXL were not. Source overlap and repeated patients are therefore not empirically ruled out. No clinical external-validation claim is supported.

## Verification and resource record

The final pre-inference code passed226 tests, with seven documented upstream/synthetic-fixture warnings; no test failures. Commands: `.venv/bin/python -B -m pytest -q`, `.venv/bin/ruff check .`, and `.venv/bin/ruff format --check .` (132 Python files). The full historical suite contains tiny fictional training fixtures; zero real ECG predictor fits or fine-tuning runs occurred. The alias regression and fictional four-model inference/report rehearsal are included. No synthetic fixture is reported as cardiovascular evidence.

Actual inference:16 dataset/model/seed passes, 60516 recording-model passes, CPU float32/four Torch threads. Summed recorded in-function inference runtime: 42.33s; excludes raw decoding/resampling, checkpoint loading and statistical analysis. There are20 sealed private NPZ artifacts (16 prediction/representation files and four outcome arrays), all under git-ignored data/. Checkpoint bytes and all hashes still match. No Phase2 ledger or approvals were accessed.

The initial curl completion watcher raced the startup checksum scan and made a premature incomplete-download check. That diagnostic was archived; no model ran then. The running-status write now precedes startup hashing. Final verification independently checked all131721 target files, zero mismatches, shasum returncode0. The bounded curl probe checked16/16 files. Raw data and all intermediate failures are preserved locally.

Bootstrap draws2000; source seed74311, target seed74312; record-level conditional resampling, shared weights across paired models/labels/seeds. Main proxy AP/AUROC and DROP intervals had zero undefined resamples. Per-code zero-positive or one-class tasks are explicitly undefined rather than manufactured. Permutations999, seven-module Holm correction within each domain/source seed. The intervals do not cover source-training/model-selection uncertainty, target-selection uncertainty, clinical-label validity or unknown patient clustering.

## Repository state

Branch: `phase3-cross-hospital`. HEAD: `c7b795553ee9806caeb1835e7a829dc0469872eb`. All169 pre-existing tracked files match starting hashes. All original69 Phase2 inventory paths are preserved as part of that tracked set; no claim is made that the expanded live source inventory has the old Phase2 code_identity. Protected private Phase2 artifacts were kept closed, not rehashed. Source fold9 caches and the specifically authorized historical source checkpoints were used read-only; no fold10 waveforms, individual targets or predictions were loaded for this study. Historical metadata/cache checksum manifests were read for lineage only.

Staged files: none. No commits, remote configuration, pushes, merges or PRs. Existing untracked Phase2 reports/results are user work and were preserved, not treated as new Phase3 outputs.

```text
?? docs/phase2/INTERNAL_PILOT_DEVELOPMENT_RESULTS.md
?? docs/phase2/INTERNAL_PILOT_EVALUATION_READINESS.md
?? docs/phase2/INTERNAL_PILOT_INTEGRITY_REPORT.md
?? docs/phase2/INTERNAL_PILOT_LOCKED_EVALUATION_RESULTS.md
?? docs/phase3/
?? results/phase2e/
?? results/phase2f/
?? results/phase3a/
?? scripts/phase3/
?? src/dmvmat_ecg/phase3/
?? tests/phase3/
```

## Files created by this task

New versioned source/scripts/tests and aggregate artifacts:

- `docs/phase3/CROSS_HOSPITAL_FEASIBILITY.md`
- `docs/phase3/DOWNLOAD_STARTUP_RACE_CHECK.md`
- `docs/phase3/EXECUTION_INTEGRITY.md`
- `results/phase3a/DATASET_AUDIT.json`
- `results/phase3a/DATASET_AUDIT_INITIAL_ALIAS_FAILURE.json`
- `results/phase3a/DOWNLOAD_VERIFICATION_PREMATURE_CHECK.json`
- `results/phase3a/DOWNLOAD_VERIFICATION_SUMMARY.json`
- `results/phase3a/FINAL_PRE_SCORE_VERIFICATION.json`
- `results/phase3a/LABEL_MAPPING.json`
- `results/phase3a/METRICS.json`
- `results/phase3a/MODULE_MMD.json`
- `results/phase3a/PARSER_ALIAS_RESOLUTION.json`
- `results/phase3a/PREDICTION_SEAL.json`
- `results/phase3a/PRESERVATION_FINAL.json`
- `results/phase3a/PRESERVATION_START.json`
- `results/phase3a/PROXY_DEFINITIONS_PRE_SCORE.json`
- `results/phase3a/REAL_ADAPTER_CHECKS.json`
- `results/phase3a/REPORT_COMPLETE.json`
- `results/phase3a/RUN_PROTOCOL.json`
- `results/phase3a/SOURCE_IDENTITY_FINAL_PRE_SCORE_VERIFICATION.json`
- `results/phase3a/SOURCE_METADATA_IDENTITY.json`
- `results/phase3a/SPLIT_SUMMARY.json`
- `results/phase3a/STARTUP_RACE_INCIDENT.json`
- `results/phase3a/SYNTHETIC_VERIFICATION.json`
- `results/phase3a/SYNTHETIC_VERIFICATION_CURL.json`
- `results/phase3a/SYNTHETIC_VERIFICATION_RECOVERY2.json`
- `results/phase3a/figures/zero_shot_proxy_AUROC.pdf`
- `results/phase3a/figures/zero_shot_proxy_AUROC.png`
- `scripts/phase3/curl_fallback.py`
- `scripts/phase3/download_cinc.py`
- `scripts/phase3/finish_after_download.py`
- `scripts/phase3/integrity_check.py`
- `scripts/phase3/offline_feasibility.py`
- `src/dmvmat_ecg/phase3/__init__.py`
- `src/dmvmat_ecg/phase3/curl_fallback.py`
- `src/dmvmat_ecg/phase3/download.py`
- `src/dmvmat_ecg/phase3/inference.py`
- `src/dmvmat_ecg/phase3/labels.py`
- `src/dmvmat_ecg/phase3/prepare.py`
- `src/dmvmat_ecg/phase3/report.py`
- `src/dmvmat_ecg/phase3/run.py`
- `src/dmvmat_ecg/phase3/signals.py`
- `src/dmvmat_ecg/phase3/splits.py`
- `src/dmvmat_ecg/phase3/statistics.py`
- `tests/phase3/test_phase3.py`

Additional git-ignored outputs: official CinC files under data/raw/cinc2021_v1_0_3; checksum/index/license/RECORDS, queues, configs, resumable logs and recovery diagnostics; FEAS metadata and ID-only split manifests under data/manifests/phase3a_v1; sealed predictions/representations under its predictions/ folder. Private record-level material is not in aggregate results. Homebrew aria2 installation is operational tooling; the project Python dependency pins were not changed.

## Required next review

Proceed only to Phase3B protocol design. Keep CPSC primary and Georgia backup as FEAS-selected targets. Review annotation-positive/unannotated-negative endpoint semantics, unknown patient identity/overlap, and the source head’s proxy-task mismatch. Preserve RESERVED for a separately frozen future design. Do not treat the gap as proof that selective adaptation can close it. Verify the original Cui paper/code/licensing before calling any MK-MMD/transfer-score/GMM arm faithful. No transfer training or new clinical evaluation is authorized by this report.
