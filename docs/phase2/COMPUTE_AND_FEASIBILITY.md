# Compute, storage and execution feasibility

**DRAFT — estimates and future stages only; no transfer profiling or model execution.**

## 1. Measured local evidence and its limits

Read-only system inspection: Apple M5 Pro, 15 CPU cores, 48 GiB unified memory
(51,539,607,552 bytes), macOS 26.6 arm64; approximately **710 GiB free** at audit.
Current environment reports Python 3.12.14, torch 2.15.0, NumPy 2.5.4, pandas
3.1.1, SciPy 1.18.1, scikit-learn 1.9.1, matplotlib 3.11.2, wfdb 4.3.1.
The existing [uv.lock](../../uv.lock) is the dependency record, not a reason to
upgrade packages during an audit. Future versioned transfer code must record
its own tested lock and code revision without altering the completed environment.

The completed study's [device receipt](../../results/device_probe.json)
reports MPS built/available. Flat/five-view synthetic forward/backward checks
passed; XResNet had one gradient parity failure (absolute difference about
0.00529 versus atol 0.003, rtol 0.01). This is not proof of an unsupported
operator, or of identical trained CPU/MPS weights. Completed neural fits used
CPU, four threads per fit, up to three concurrent workers. Default future
comparison to **common CPU execution** pending new synthetic masked-loss,
freezing/BatchNorm and serialization checks; do not silently change operators
or pretrained preprocessing to obtain MPS speed.

Historical timings below are existing measurements, not new benchmarks. Evidence:
[results](../EXPERIMENT_RESULTS.md),
[development lineage](../../results/development_lineage.json), and the completed
run metadata recorded there.

| Completed family | Per-seed optimization/preparation, 50 epochs | Validation across training | Context |
|---|---:|---:|---|
| Five-view, no augmentation |44.92 / 45.60 min|3.44 / 3.51 min|17,084 train, 2,146 validation; full 1,000 points |
| Five-view, augmentation |46.04 / 48.14 min|recorded in lineage|Not a planned transfer augmentation arm |
| Flat CNN |18.79 / 19.34 min|recorded in lineage|Smaller inference pipeline |
| XResNet |about 168 min each|about 95 min each|250-point train crops, seven-window validation |

Historical totals: 9.314 worker-hours optimization/preparation, 3.459
worker-hours validation; overlapping execution wall time about 4.418 hours.
These are not eight sequential fits' wall time. Mean completed fold-10 inference
time for 2,158 records: five-view about 2.218 s, flat 0.943 s, XResNet about
75.05 s (seven windows). They are frozen aggregate log values, not a new test pass.
The initial full 100 Hz WFDB audit took about 54.8 s; 500 Hz download/resampling
and cross-dataset duplicate checks have not been timed.

## 2. External acquisition and cache sizes

No full external waveforms, large archive or new participant-label table was
downloaded in this sprint. Estimates assume twelve leads, and must be replaced
by actual publisher file sizes/checksums before authorization.

| Candidate | Native signal payload lower-level calculation | Acquisition caveats |
|---|---|---|
| Georgia public 10,344 |0.621–1.241 GB at 5–10 s, 500 Hz, 16-bit samples|MAT/container/compression change actual bytes; eligible 10 s fraction unknown |
| Ningbo 34,905 |4.189 GB at 10 s, 500 Hz, 16-bit samples|Do not infer actual storage/ADC solely from this arithmetic |
| Combined 45,152 |5.418 GB (5.046 GiB) under same assumption|Publisher lists about 5.1 GB uncompressed / 2.3 GB ZIP; contains Chapman/Ningbo copies |
| Original Chapman raw item |Official API ZIP size 754,581,129 bytes|Diagnostics item 975,796 bytes; original CSV/labels differ from Challenge conversion |

Official size/licence sources and release distinctions are in
[the audit](EXTERNAL_ECG_DATASET_AUDIT.md), D2–D6. Native 32-bit/float storage
can double the illustrative 16-bit payload; header calibration controls physical
units. Download only one pinned candidate distribution; avoid storing full
Challenge, its combined copy and original Chapman in parallel without purpose.

At 100 Hz float32, one 12×1,000 ECG is 48,000 bytes. Full waveform caches would
be 0.497 GB Georgia, 1.675 GB Ningbo, 2.167 GB combined, before headers/indexes.
500 Hz float32 caches cost five times more. Existing source raw 100 Hz data are
roughly 612 MB plus approximately 1.027 GB signal cache. Prefer memory-mapped
streaming; do not copy every waveform/derived view into accelerator memory.
Patient/group manifests, labels and per-record predictions are private artifacts.

Seven embeddings total 512 float32 coordinates (5×64+128+64) = 2,048 bytes/record.
Georgia all-record embeddings would be 21.18 MB/checkpoint; Ningbo 71.49 MB.
Two source seeds double this. Only authorized source/G evidence is extracted
before evaluation; these all-record figures are capacity bounds, not permission
to compute or inspect outer embeddings.

Model state sizes measured from existing checkpoint bytes: MV best about
0.885 MB, XResNet best about 7.822 MB; two-label head changes are tiny relative
to convolution weights. A full 166-fit plan's weights alone are under roughly
1 GB even with multiple checkpoint files under this family mix. Optimizer/RNG,
history, cache versions and crash-safe temporary writes add storage. Reserve
approximately 2 GB pilot / 10 GB full-study artifact space and 10–25 GB raw/cache
headroom depending on the selected distribution. These are planning allowances,
not measured peak requirements. Current free disk is adequate; battery/power
and thermal behavior must be checked before future sustained execution.

## 3. Neural workload and fit ceilings

See [protocol](TRANSFER_EXPERIMENT_PROTOCOL.md) for the exact method matrix.

| Proposed stage | Predictor fits | What it can establish |
|---|---:|---|
| Small engineering/transfer pilot |18 planned, ceiling **20** with technical reserve|Loading, labels, source conversion, frozen buffers, selector abstention and bounded development behavior; no final O claim |
| Full one-target study |166 planned, ceiling **172**|162 adaptation/target-only fits plus four source-task conversions, 18 paired episodes, one sealed external-target evaluation |
| Both stages if separately approved |≤192 actual attempts|Can reuse identical verified conversion artifacts; unused reserve cannot fund new searches |
| Optional second target, separate approval |162 planned +4 reserve = **166**, if four original conversions remain identical|Independent additional database evaluation with its own endpoint/linkage gates and sealed holdout |

Calibrator fits are not predictor fits but must be logged. Full study up to
162×2=324 per-label Platt mappings plus eight mappings for four source-only
checkpoint variants (MV/XResNet × two seeds × two labels), before immutable-job
deduplication/fallbacks. Source-only inference adds no predictor fit. Artifact
aliases caused by identical random/logical masks reduce actual fits, not the
registered comparison slots. All attempted/failed/interrupted predictor fits
count toward the respective hard ceiling; no automatic continuation from old
weights as an allegedly preserved optimizer trajectory.

At F=128/512/2,048, adaptation ceilings are **40/160/640 optimizer steps** with
batch64. Larger F+G controls receive the same steps, not more epochs. At the
large tier this is at most 40,960 gradient-example exposures, versus 854,200
in a historical 50-epoch source fit. E validation up to 20×512 records can
dominate small adaptation fits, particularly seven-window XResNet. Early stopping
and epoch-zero retention may reduce updates, but no favorable timing is assumed.

For scale only, historical MV optimization was about 0.0032 seconds per
record-epoch, XResNet about 0.0118, under the old batch/worker conditions.
Four new 20-epoch source conversions, if throughput and validation cost scaled
unchanged, correspond to roughly four worker-hours including validation. Batch64
instead of128, masks, different labels and concurrency can materially change
that estimate. This is a **workload-equivalent extrapolation, not a wall-clock
promise**. Adaptation timings are unmeasured. Full CPU execution may occupy an
overnight/multiple-day window; it must be planned from the separately approved
pilot, not asserted finished or replaced by cheaper models without amendment.

## 4. Evidence and uncertainty computation

For a 256+256 patient MMD cap, a float64 512² Gram matrix is about 2 MiB per
module; seven simultaneously about 14 MiB. Reuse kernels across 999 permutations.
Naive quadratic-form work across seven modules is on the order of 1.8 billion
element-products per episode, about 33 billion across 18 episodes. Optimized
block matrix operations may help; no local timing for this proposal exists.
Restrict BLAS threads when neural workers run to prevent oversubscription.

Geometry uses 4,999 patient bootstrap draws and seven small embedding spaces.
A 4,999×1,024 float64 weight block is about 41 MB; stream label/module blocks
rather than duplicate full embeddings per draw. Fixed direction/scatter
conditional bounds permit class-mean projection operations. B's rare joint cells
may invalidate many episodes; do not spend compute fabricating covariance or
populating empty strata. Record invalidity before evaluating targets.

Final 10,000-draw paired patient bootstrap of O must preserve label vectors and
episode pairing. It may be CPU-heavy because exact AP/AUROC requires weighted
ranking; pre-sort fixed scores and reuse weights where mathematically equivalent.
Validate optimized weighted metrics against scikit-learn synthetic fixtures.
Bootstrap is not predictor training and does not generate new independent
patients. Measure timing only after separate execution authorization.

## 5. Pipeline feasibility and dominant risks

Naive all-pairs sample-by-sample near-duplicate comparison across 21,799 source
and tens of thousands of target records is unnecessarily expensive. Use exact
file/physical-signal hashes first, then label-blind candidate fingerprints and
an indexed near-duplicate search, followed by detailed aligned waveform review.
Verify sensitivity on known cross-release conversions/synthetic transformations.
No absence of hash matches establishes person independence for distinct ECGs.

Parallelization: at most three CPU4-thread neural workers replicates the measured
project setting, but start the pilot conservatively and record memory/throughput.
Separate dataset audit, neural fitting, representation extraction and evidence
phases; cap NumPy/BLAS threads, keep disk writes atomic. Do not run multiple
unvalidated MPS processes competing for unified memory. CPU remains the default;
any hardware change is prospectively logged, with common paired-method settings.

Local capacity likely supports the bounded pilot and small models. Cloud CUDA
may accelerate the larger matrix, but requires separate authorization, data-use
review, validated numerics, secure artifact transfer and measured pricing; no
external uploads or cloud spending is authorized here. Different devices cannot
be compared as if runtime were algorithm-only. No precise dollar forecast is
supported by current evidence.

**The immediate blockers are clinical endpoint completeness and patient/overlap
evidence, not disk space or neural parameter memory.** Expert adjudication may
cost more than fitting. No realistic runtime estimate can substitute for those
gates. A 172-attempt project is technically plausible on this machine only after
the label/cohort audit and a separately approved bounded pilot; its scientific
validity remains unestablished until the gates pass.
