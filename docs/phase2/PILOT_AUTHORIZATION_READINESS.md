# Phase 2B authorization readiness

**Execution-readiness decision: BLOCKED.**

The engineering safeguards and synthetic rehearsal are complete, but the audited
draft splitter structurally removes joint CD/STTC positives from the target
fitting/evidence reservoir and concentrates source joint positives in stopping.
Do not authorize training on this candidate without prospective protocol review.
Passing marginal support or software tests does not resolve that scientific issue.

## Completed checks and blocker

The endpoint is the existing proposal's **two-label CD/STTC annotation task**,
not five classes or IAVB/IRBBB. No endpoint, age rule, threshold, sample size or
fit family was changed. Folds1–9 only, actual patient IDs, one canonical eligible
ECG/patient, fresh source weights/normalization required. The original fold10
is permanently excluded. No real signal was loaded by the new execution path.

Metadata identity and all role-capacity/marginal floors pass:19,230 eligible
development records,18,857 age-eligible,16,415 canonical patients; source7,638,
target8,777. Source train/stop6,466/1,172, O2,668, E/C128 each, A4,208.
All active source/F/G/E/C/O patient/record sets are disjoint; deterministic
generation and known native-duplicate isolation pass. Exact hashes and all
per-label/joint counts are in [preflight](PILOT_PREFLIGHT_REPORT.md).

**Scientific findings requiring review:** target A has zero joint positives,
so every proposed G draw has zero and B necessarily abstains, whatever the
representation. Source_train has19 joint positives versus142 in source_stop.
The draft's rare-pattern-first marginal objective does not preserve co-occurrence.
These are label-feasibility observations, not real model/MMD/geometry results.
No alternative assignment was substituted, controls dropped, labels fabricated
or split seeds searched. The failed candidate is preserved. Review a prospective
joint-pattern stratification amendment, then authorize a new metadata-only
preflight; do not silently modify or clear the current blocked ledger.

## Engineering implementation verified synthetically

- Namespace/context-bound real data adapter and label-free worker manifests;
  separate per-role label vaults. Training/selector workers do not preload
  per-record O/C/unused-role outcomes. Permission precedes label/signal reads;
  membership and signal-content hashes bind preprocessing inputs.
- Completely new source constructor, seed3101, no historical checkpoint argument,
  source-only train/stopping hashes, fresh normalization provenance; target-only
  initializes separately and fits its own F∪G statistics. Old normalization,
  foreign source IDs, wrong lineage and changed artifacts are rejected.
- Guarded source training/adaptation primitives; exact two-label AP stopping,
  correct scratch/pretrained epoch-zero conventions; all-SHARE/all-ADAPT,
  each of seven individual masks and mixed masks. Prediction-head ownership,
  frozen parameters/BN affine/running statistics/counters/dropout and source
  checkpoint integrity are tested. No MMD alignment loss was added.
-33-slot scheduler dry run, atomically reserved fits, hard36 ceiling including
  failures/retries; concurrency, crash-before/during-fit, incomplete write,
  corrupted checkpoints, immutable terminal jobs and fresh restart rules.
  Actual optimizer/RNG continuation is unsupported and never claimed.
- E/checkpoint selection separate from F/G fitting, C-only calibration and
  operating thresholds, distinct source/audit/evaluation costs. Full-cost gets
  the selector's G labels for gradient use and the same40/160 update budget.
  Equal steps do not imply equal FLOPs or elapsed time.
- Independent development lock, separate single-use evaluation grant, O-label
  redaction during prediction, complete prediction sealing before labels/metrics,
  second-access refusal and failure-consumed lease protection across restart.
  Live source/config/dependency integrity is checked at data access.

These controls govern approved code paths, not an adversarial OS sandbox against
an operator who can arbitrarily open files or edit SQLite. Outcomes/assignments,
waveforms, checkpoints and operational ledgers remain private/ignored. No private
vaults or patient-level rows appear in public reports/results.

## Synthetic rehearsal and verification commands

One complete isolated synthetic namespace trained a tiny source and all eight
controls/policies for one draw/tier (nine tiny predictor fits), source-only
preprocessing, permitted evidence extraction, A/B/MMD/masks, E stopping,
C calibration, artifact lock, sealed fictional O predictions and synthetic
metric access. It then rejected a second lease. Separate fixtures exercised
interruptions/retries/concurrent reservations to36 and evaluation failure/restart.
All source/target identities, ECG tensors and outcomes were fictional; reduced
steps/Monte Carlo overrides are forbidden in a real namespace. These are software
tests, not ECG transfer findings or real study attempts.

Commands actually executed:

```bash
.venv/bin/python -m pytest tests/phase2b/test_preflight.py -q
.venv/bin/python -m pytest tests/phase2b -q
.venv/bin/ruff format src/dmvmat_ecg/phase2b tests/phase2b scripts/phase2b_preflight.py
.venv/bin/ruff check src scripts tests
.venv/bin/ruff format --check src scripts tests
.venv/bin/python -m pytest -q
```

Final counts: **33 Phase2B tests;142 full-suite tests passed**, zero failed/skipped,
seven pre-existing warnings. Full Ruff and formatting checks pass. Warnings are
WFDB/pandas deprecations, an intentionally unsupported-label legacy metric fixture
and an unchanged legacy synthetic scalar-loss conversion. Initial development
checks found style issues and one synthetic O-reader fixture failure: it tried
to use redacted labels to generate signals. The fixture now derives fictional
signals without reading labels; redaction was preserved. Its failed synthetic
lease was not rearmed; corrected tests ran in fresh temporary namespaces.
The system Python lacked NumPy; the existing pinned venv was used without
installing/upgrading anything. Hardware-query sandbox denial was resolved by an
approved read-only sysctl query, not by altering execution settings.

## Real state and required future authority

New real ledger: `private/phase2b_internal/study.sqlite`, **draft,0/36**,
no development authorization, no evaluation authorization/access and immutable
preflight-review blockers. No new real model, fitted preprocessing, calibrator,
prediction or performance metric exists. Model constructors/parameter-count and
forward/backward checks were synthetic only. The historical ledger remains
evaluated,11 successful fits, one consumed evaluation receipt.

The proposed matrix remains33 fits: one fresh source plus eight methods×two
draws×two sizes; three shared technical reserve attempts, total ceiling36.
This is not adoption or training permission. Before any later authorization:

1. Review the actual CD/STTC task, source/target joint-cell allocation, splitter
   objective/ties,20 post-update looks plus epoch zero, and calibration-fallback
   threshold convention. Approve a prospective revision if required.
2. Preserve this candidate/audit/ledger. Separately authorize revised metadata-only
   preparation in a new versioned context, verify joint/marginal support and
   disjointness without predictive feedback, and adopt the reviewed contract.
3. Only after a clean preflight explicitly authorize fresh source pretraining,
   the bounded33-fit development matrix and C-only calibration. No historical
   source weights/statistics, extra architectures, favorable reruns or budget
   borrowing. Any real smoke/probe must be prospectively specified and counted.
4. After immutable development locking, obtain a separate one-time exploratory O
   authorization. External clinical transfer still needs its independent label/
   patient/overlap evidence; internal success cannot clear that gate.

Resources: Apple M5 Pro15 cores/48GiB, approximately710GiB free; CPU float32,
four threads/fit, at most two real active fits. Python3.12.14, torch2.15.0,
NumPy2.5.4, pandas3.1.1, SciPy1.18.1, scikit-learn1.9.1, WFDB4.3.1,
pytest9.2.0, Ruff0.17.4; uv.lock unchanged. Current sandbox MPS availability is
false (built true); no training-device switch or new MPS parity claim. New real
training/evidence timing and peak memory are unmeasured. Historical timings and
candidate workload counts are planning context only; reserve several GB locally,
with no external archive/cloud requirement.

## File inventory and preservation

Created source under `src/dmvmat_ecg/phase2b/`:
`__init__.py`, `contract.py`, `preflight.py`, `ledger.py`, `access.py`,
`integration.py`, `execution.py`, `runtime.py`, `sealing.py`.
Created tests under `tests/phase2b/`: `__init__.py`, `test_preflight.py`,
`test_execution.py`; script `scripts/phase2b_preflight.py`; draft configuration
`configs/phase2b/internal_pilot_draft.json`; the four requested `PILOT_*.md`
reports; safe aggregate receipt `results/phase2b/PREFLIGHT_AGGREGATE.json`.
All are intentionally visible to Git, with no commits/staging required.

Private/ignored outputs: the original candidate full auditor manifest,
label-free worker manifest,16 per-role metadata-label vaults, execution-integrity
inventory and the new zero-fit SQLite ledger. No participant assignments/labels
or real per-record predictions are published. Temporary synthetic fit artifacts
are confined to pytest namespaces. No prior file was overwritten or deleted.

The starting112-file inventory (95 tracked plus17 prior Phase2 files) remains
byte-identical. All48 historical lock artifacts, including16 best/final checkpoint
files, match stored digests; none was deserialized as a new source. Added code
changes the current source-tree hash by design; historical stored hashes/receipts
were not rewritten. Baseline/Phase2A source, tests, plans, results and dependencies
remain unchanged. Other research repositories were not modified.

Git remains `main`, HEAD `1f0ff9031e9e1e2adc97962922fd97705570baf2`;
tracked diff empty, prior user files and new Phase2B files untracked. The exact
code/dependency/manifest/role hashes and real permission snapshot are in
[PREFLIGHT_AGGREGATE.json](../../results/phase2b/PREFLIGHT_AGGREGATE.json).
No real training, fine-tuning, calibration, inference or new evaluation occurred.
No fold10 record/label/prediction loader was used. No commit, push, merge or PR.

**BLOCKED — stop here for prospective split/protocol review, not training.**
