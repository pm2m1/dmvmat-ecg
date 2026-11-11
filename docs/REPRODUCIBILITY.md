# Reproducibility and artifact boundaries

Independent workspace `~/Desktop/dmvmat-ecg`, local branchmain. Initial source
revision `4b18037d346f3bd45330d7d68ffdb07637c920b1`. Existing cardiovascular
repositories are read-only references; no checkpoints or private participant
datasets copied. Local source history is independent; no GitHub remote created.

## Environment and commands

Python3.12.14, uv0.12.9, exact resolution in `uv.lock`. Direct research pins:
PyTorch2.15.0, NumPy2.5.4, pandas3.1.1, SciPy1.18.1, scikit-learn1.9.1,
matplotlib3.11.2, pytest9.2.0, Ruff0.17.4. WFDB4.3.1/httpx0.28.1 resolved in lock.
Apple M5 Pro,15physical cores,48GiB unified memory. Common CPU float32 fallback;
four intra-op threads per seed process, zero data-loader workers. Thread-level
reductions and framework changes can affect numerical reproducibility; no claim
of identical MPS/CPU training. Deterministic operations requested warn-only;
random initialization, dropout, permutation, crops and noise seeded explicitly.

```bash
uv sync --frozen
uv run python scripts/download_ptbxl.py
uv run python scripts/audit_dataset.py
uv run pytest -q
uv run ruff check src scripts tests
uv run ruff format --check src scripts tests
uv run python scripts/device_probe.py
uv run python scripts/train_study.py --stage smoke
uv run python scripts/train_study.py --stage full --seed 101
uv run python scripts/train_study.py --stage full --seed 202
# Only after all development artifacts are final; consumes a single test lease:
uv run python scripts/locked_evaluate.py
uv run python scripts/make_figures.py
```

The two full seed commands may run concurrently, with separate logs and SQLite
reservations. Do not concurrently schedule duplicate seed jobs. Do not rerun an
auditor or delete a ledger to bypass an evaluation lease. To reproduce from
scratch, use a **new independent workspace and ledger**, retain previous results,
and label any experiments informed by our test results as exploratory.

## Real attempts and immutable artifacts

`private/study.sqlite` WAL/FULL synchronous commits reserve each real predictor
attempt before fitting, including smokes and failures. Hard ceiling13; planned
3smoke+8full,2shared failure reserve. Model/seed/stage slots are unique; successful
fits cannot be rescheduled. Failures stay recorded, one identical-config fresh
repair permitted per failed slot subject to global cap. No exact optimizer/RNG
continuation from model-only checkpoints. Active/interrupted jobs require review,
not blind resume. Never use unused repairs to chase a better score.

UUIDrun folders contain entry, epoch history, best/final checkpoints, validation
scores/targets, thresholds, completion or failure receipt. These are ignored.
Safe completion metadata/history aggregates appear under `results/baseline` or
`results/multiview`. Source/config hashes checked before/after each fit. Final
lock additionally binds scripts, protocol, uv.lock, preprocessing, memberships,
checkpoint checksums and threshold-bearing completion files. Individual scores,
ECG IDs, patient IDs and source rows never enter public aggregate reports.

All test scores are generated and hashed before metrics. One durable access
receipt is consumed before first test read; closing/restarting cannot obtain a
second lease. A failure after access preserves the record and requires explicit
review. No recalibration, fitting or selection follows test opening. Reporting
uses the same sealed prediction artifacts; any correction must be recorded.

## Tests versus scientific evidence

Synthetic fixtures test loading, lead order, multilabel semantics, train-only
normalization/augmentation, architecture/gradient/serialization, parameter counts,
exact AP/AUC, checkpoint direction, threshold selection, role locks, finite
attempts, shared-patient bootstrap, MMD and synthetic freezing. MPS parity is a
separate explicit synthetic device probe; one failure causes common CPU fallback.
WFDB emits upstream pandas deprecation warnings. A known scalar-loss conversion
warning does not imply nonfinite gradients; finite guards run every batch.
Tests do not establish clinical validity or substitute for actual training.

## Publication inventory

Eligible for separate release review: source/configuration/tests, GPL3 notices,
dependency lock, dataset download instructions, aggregate audit/metrics/curves,
PNG/PDF figures and research reports. Check dataset attribution at release time.

Local only: rawWFDB/metadata, membership manifests, signal caches, checkpoints,
patient-level predictions/labels, SQLiteoperational ledger, reference clone,
download logs, environment, temporary files and any credentials. `.gitignore`
excludes these. No remote publication has been authorized. A source-code commit
is not publication and contains no raw records or model weights.
