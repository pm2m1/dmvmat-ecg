# Phase 3B compute allocation and human decision record

**ADOPTED v1 — 2026-11-14 — protocol owner: Parth Mishra (git user.name).**

This document transcribes the owner's explicit decisions; it is not an invented
signature. Only implementation and synthetic tests are currently authorized.
No RESERVED access, metadata preparation of R/O, real fitting or evaluation.
The pre-adoption drafts are archived unchanged under `archive/phase3b_pre_adoption/`.

## 1. Adopted decisions

| Review item | Owner's decision | Adopted scope |
|---|---|---|
|1 Endpoint|APPROVED as drafted|Exact Phase3A CD/STTC annotation proxies, not clinical negatives|
|2 Independence|APPROVED as drafted|Exploratory record/known-duplicate-group separation; patient linkage unknown|
|3 Sources|APPROVED as drafted|Fresh two-output five-view sources, source-only norm, source fold9 AP stopping, no fold10|
|4 Allocation|MODIFIED|B64/B128(primary)/B256, half F/half G, nested label-blind; A/random select on G then fit F∪G|
|5 Checkpoint|APPROVED as drafted|Target fixed200 updates, no E/C/P, no target checkpoint selection|
|6 Methods|APPROVED as drafted|A−Full-cost primary; head/random/source/target-only secondary; count-and-cost random, inclusive aliases/abstentions|
|7 Compute|MODIFIED|3 sources ×3 draws; no XResNet extension; faithful-paper arm deferred to Phase3C|
|8 Statistics|APPROVED as drafted|Conditional record/group bootstrap, four-secondary Bonferroni family, +0.01AP engineering margin, no established power|
|CPSC truncation|ACCEPTED and disclosed|First10s identically for all arms; annotations may refer to later signal|

**Prospective pre-data amendment:** the owner cites Cui et al., TNNLS2025
Algorithm1 as motivation for target data reuse in scoring and fine-tuning.
Record that rationale without conflating the re-engineered A policy with a
verified faithful-paper implementation. No patient metadata, RESERVED labels,
waveforms, learned representations or predictions were used to make this
amendment. It is not a post-result revision.

## 2. Exact fit matrix, aliases and ceiling

Source seeds15101/15102/15103; draws16101/16102/16103. Fitting randomness
is paired across methods, independently of random-mask randomness.

| Component | Per source/draw cell | Across nine cells |
|---|---:|---:|
|Head-cost, budgets64/128/256|3|27|
|Full-cost, budgets64/128/256|3|27|
|Random-A, budgets64/128/256|3|27|
|Selective-A, budgets64/128/256|3|27|
|Target-only-cost, budgets64/128/256|3|27|
|Target fits total|15|135|
|Fresh source pretraining|one per source seed|3|
|**Planned logical fit slots / conservative unique maximum**| |**138**|
|Technical reserves, independently capped| |6|
|**Hard ceiling actual predictor-fit attempts, including failures**| |**144**|

Source-only uses the three new source models: no extra training. There are no
guaranteed aliases after removing B32. Actual deterministic equivalences are
recorded once and reuse an immutable artifact: A all-SHARE and Random-A
all-SHARE alias Head-cost; all-ADAPT alias Full-cost; an identical mixed Random
mask aliases A. The job key includes source initialization, actual data/order,
normalization, mask/modes, seed, optimizer, terminal budget, code and task.
Method names do not defeat equivalence. Alias savings do not add candidates,
retries or favorable reruns; reserves remain six, at most one fresh technical
retry per interrupted job. Deterministic nonfinite/scientific failures stop for
review. No exact mid-trajectory continuation is claimed.

All target arms use the SAME F∪G gradient data (B records), not merely the same
accessible label count. Only A and Random-A use G for mask construction first.
B128 is the primary contrast, mean over all nine paired cells, raw mean CD/STTC
AP. Abstentions stay included and equal Head-cost when identical. No target-only
or source-only ensemble, no best-seed reporting, no G-based stopping.

The source-label/pretraining cost is separate. A fully observed B-record episode
has2B annotation entries. Across three B256 draws, maximum new adaptation
union is768 records before overlap, reused by all seeds/methods. Public annotated
data access is simulated label restriction, not new clinical adjudication.

### Support-probability calculation (planning only)

Use ONLY Phase3A CPSC FEAS CD506/1930 and STTC494/1930. For each label:
`P(10 <= K <= G-10), K ~ Binomial(G,p_FEAS)`.

|B|F|G|CD probability|STTC probability|
|---:|---:|---:|---:|---:|
|64|32|32|0.318770116748|0.289995614110|
|128|64|64|0.984909732369|0.980301019758|
|256|128|128|0.999999972870|0.999999940173|

This is per-label count feasibility, not joint independence, actual RESERVED
support, geometry validity or predicted ranking improvement. Unknown labels,
weak source separation and bootstrap uncertainty can cause abstention at any
budget. No enriched G sampling or seed search is permitted.

## 3. Measured local evidence and conditional feasibility

Historical **CPU**, four-thread,50-epoch full source runs provide these anchors;
they are not measurements of the proposed two-label task or adaptation fits:

| Existing model | Seed101 training seconds | Seed202 training seconds | Peak process GiB | Parameters |
|---|---:|---:|---:|---:|
| Five-view |2695.22|2736.22|1.519|212,261|
| Flat CNN |1127.69|1160.20|1.472|206,117|
| XResNet101 |10077.06|10084.64|1.824–1.826|1,872,261|

Evidence is in the immutable `private/runs/<run_id>/completion.json` files:
five-view `541ceff8cfbd45b2b8f16db2c3fc3275` and
`1419561ecfdd4d329559e33e1b3a322f`; flat
`5defa4e903534533aaf0106451f77412` and
`a33ff61f65864b30b258966cf85ade6d`; XResNet
`9f340af838904c878b9ee7e47b2cf558` and
`1cd7d36ecdb844abba9092c7960bcfb4`. Only completion metadata, not test predictions,
were read for these estimates.

Three comparable five-view source fits sum to roughly136 minutes in historical
training time. XResNet is not part of the adopted study. These are workload anchors,
**not promised runtimes**: new proxy labels, checkpoint scoring, machine load,
caching and software behavior can change cost. Target200-update fit throughput
has not been measured; do not scale inference throughput into training claims.

Phase 3A measured16 source/target/model/seed inference passes over60,516
recording-model inputs in42.33 seconds inside the inference functions, excluding
decoding/resampling and model loading; see
[`PREDICTION_SEAL.json`](../../results/phase3a/PREDICTION_SEAL.json).
That supports inexpensive representation extraction, not exact adaptation-time
prediction. Geometry4999 and permutation999 calculations must be timed in the
first authorized episode without changing thresholds or labels to save time.

The existing48GiB Apple Silicon machine and CPU-only policy are plausible for
the adopted pilot. Use one four-thread training worker initially; do not assume
CUDA or identical MPS/CPU fitted weights. Original common CPU fallback is
documented in [`configs/study.json`](../../configs/study.json). MPS is not a new
experiment factor in this proposal.

Already verified CinC raw bytes total8,682,129,257. No new external waveform
download is needed. CPSC RESERVED resampled float32 storage would be about
353.3MiB for7,719×12×1000 values; the source training cache is about782.0MiB
for17,084 records, excluding views and metadata. Batch-wise view construction
avoids unnecessary full-copy caches. These are size calculations, not permission
to load or preprocess RESERVED now.

A two-output five-view state is about0.81MiB of float32 parameters. AdamW
states/gradients and activations add memory; filesystem serialization, buffers,
source epoch snapshots and failed fits add disk usage. Reserve at least10GiB
free beyond existing raw data before future preparation; record measured usage
instead of claiming an exact peak. Keep all checkpoints, split identities and
per-record predictions private and git-ignored. Raw data are not rewritten.

Sequential CPU execution is the default. Any later parallelization must retain
paired seeds, worker/thread settings, immutable artifacts and the same attempt
ceiling. If resources cannot finish the matrix, report incomplete; do not remove
an inconvenient comparator or shrink its training budget after seeing results.

## 4. What changes now; what still requires authorization

The adopted configuration and all138 logical fit entries are in
`configs/phase3b/adopted_v1.json` and `fit_matrix_v1.json`. Numerical support
arithmetic is in `support_probability_v1.json`. The new implementation is in
`src/dmvmat_ecg/phase3b/`; fixtures/tests are in `tests/phase3b/`.
The execution surface in this sprint accepts fictional in-memory providers only;
real namespaces fail before provider reads. No new real approval system or
Phase2 ledger is created. This is deliberate staging, not a claim that a unit
test authorizes real cohort access.

Separate next-stage authorization must cover metadata-only R/O and nested-draw
preparation plus reviewed real adapters/lineage. Source/target waveform loading,
new source training, target adaptation and locked evaluation each remain
unexecuted and unauthorized. No existing Phase2/3A files may be modified.

The full synthetic verification commands, results, file inventory, preservation
hashes and remaining execution conditions are recorded in
`PHASE3B_IMPLEMENTATION_VERIFICATION.md`. Independent external clinical
validation remains **BLOCKED** despite adoption of this exploratory proxy study.
