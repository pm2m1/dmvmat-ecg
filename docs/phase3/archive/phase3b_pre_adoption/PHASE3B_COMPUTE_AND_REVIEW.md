# Phase 3B compute allocation and review memo

**DRAFT — NO IMPLEMENTATION, DATA PREPARATION, TRAINING OR EVALUATION GRANT.**

Recommendation: review the CPSC-only, fixed-schedule, annotation-proxy pilot in
[`PHASE3B_TRANSFER_PROTOCOL_DRAFT.md`](PHASE3B_TRANSFER_PROTOCOL_DRAFT.md).
Keep geometry A primary. Georgia is a future backup, not a second target to
try after an unfavorable CPSC result. Do not optimize the classification
architecture or reopen the completed PTB-XL test.

## 1. What Phase 3A establishes and leaves open

All131,721 requested target files passed publisher SHA-256 verification;
840 native exact duplicates were excluded before splitting. CPSC FEAS had
1,930 records, CD prevalence0.2622 and STTC prevalence0.2560. Its five-view
proxy-matched macro AUROC drop was−0.09866. Georgia’s drop was−0.05199;
Chapman–Ningbo’s was−0.03598. Source proxy semantics materially change the
reference: fold9 STTC AUROC was0.85045 rather than full-superclass0.92574.
These results justify a **design discussion**, not a transfer-success claim.

No reliable clinical negatives, complete target patient identities or empirical
PTB-XL-versus-target overlap exclusion were established. RESERVED was not
scored. A future record-disjoint annotation pilot can be exploratory only;
independent external clinical validation remains blocked. Do not interpret
dataset names or exact-record deduplication as proof of patient independence.

## 2. Finite budget choices — none approved

Core target training procedures: Head-cost, Full-cost, Random-A, Selective-A,
Target-only-cost. Source-only requires no additional fit. Three target-label
budgets are fixed at32/64/128 newly exposed episode labels, counting F and G.
This is not the project's total target-data history: Phase3A used12,983 FEAS
outcome vectors for model-aware feasibility (including1,930 CPSC records), and
the cohort auditor used eligible annotation vectors for stratification.
The protocol gives overlapping historical counts separately. No method may
reopen FEAS for new source selection, stopping or tuning. Report this as a
conditional incremental-label comparison, not as a pristine32-label study.

| Choice | Source seeds × draws | Source slots | Target logical slots | Structural B32 aliases | Maximum distinct planned fits | Technical reserves | Hard maximum actual attempts |
|---|---|---:|---:|---:|---:|---:|---:|
| **Recommended reduced pilot** |2×2|2|60|4|58|4|**62**|
| Larger pilot, choose before any O access |3×3|3|135|9|129|6|**135**|

The alias reduction is prospective, not based on model performance: G=16
cannot satisfy10 positive plus10 zero observations, so A32 is all-SHARE;
Random-A32 must have zero adapted modules and is the identical head-on-F job.
With paired stochastic seeds it reuses the same artifact. Other exact matches
may reduce real fits further. Do not count logical comparisons as independent
fits or spend saved slots on extra searches. Every failed/interrupted real fit
and real training smoke run consumes the actual ceiling; retries are fresh
attempts unless exact continuation states have been validated.

The recommended reduced source seeds are15101/15102 and draws16101/16102.
The larger option adds15103 and16103, not favorable-seed replacement. It must
be chosen prospectively; after reduced O results, a later larger run is a new
exploratory study, not a way to repair the original inference.

### Optional strong conventional reference

An XResNet101 two-output reference could add fresh source pretraining and
Head-cost/Full-cost adaptation under the same source/target label roles.
Its original2.5s source training and seven-window maximum-probability inference
would be retained and clearly separated from the full10s five-view pipeline.
It is a stronger architecture reference, not a capacity-matched control.

| Extension | Added source fits | Added target fits | Added planned fits | Added reserves | Combined maximum with core |
|---|---:|---:|---:|---:|---:|
| Reduced2×2 |2|24|26|2|**90** =58+26+4+2|
| Larger3×3 |3|54|57|3|**195** =129+57+6+3|

No XResNet fits are automatically authorized. Verify its two-output constructor,
parameter count, checkpoint policy and source-task mapping synthetically first.
The requested literature-faithful arm is not in these counts: its source
objective and required initialization-matched controls are unresolved, so an
honest finite allocation cannot yet be claimed. It needs a separate prospective
matrix, not a reserve-slot substitution. See the method/evidence companion.

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

Two comparable five-view source fits are roughly90.5 minutes in summed
historical training time, three roughly136 minutes. Two XResNet source fits
alone add about5.6 hours, three about8.4 hours. These are workload anchors,
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
the reduced pilot. Use one four-thread training worker initially; do not assume
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

## 4. Choices needing explicit human review

1. **Scientific endpoint:** adopt the exact Phase3A annotation proxies and
   source population eligibility, including annotation absence semantics and
   the excluded ambiguous code. No clinical-negative or clinical-utility claim.
2. **Independence:** accept a clearly exploratory record-disjoint pilot despite
   unknown target patients and the waived source waveform cross-match, or
   require stronger original patient/overlap metadata before proceeding.
3. **Source training:** approve new two-output source models, source-only
   normalization and source-fold9 AP checkpoint selection, without opening fold10.
   The old five-superclass source gap is not guaranteed to persist after task
   alignment; Source-only remains a mandatory locked comparator, with no new
   target-based source screening.
4. **Label allocation:** choose the half-F/half-G32/64/128 total-label design,
   recognizing deterministic B32 abstention and low B64 support. Do not silently
   replace it with enriched evidence draws. B32 G labels have diagnostic cost
   but cannot influence its module mask; decide whether this control is worth
   the access. Also accept the explicit historical FEAS/stratification disclosure.
5. **Checkpoint convention:** approve fixed update200, E=C=P=0 and no target
   epoch-zero/AP selection, or request a different explicitly costed protocol.
6. **Methods and claim:** A primary versus Full-cost at B128; secondary head,
   random, source-only and target-only contrasts; both-count-and-cost random
   matching, inclusive abstentions and transparent degenerate aliases.
7. **Compute:** choose reduced62-attempt ceiling or larger135 before training;
   optional XResNet means a distinct90/195 total ceiling, not free extra fits.
   The faithful-paper arm remains gated and unbudgeted until its method is verified.
8. **Statistics:** review conditional record-level bootstrap, the four-secondary
   Bonferroni family, +0.01 AP engineering margin and absence of established power
   for that margin. No threshold of0.93 or significance-by-point-estimate rule.

The CPSC first-10s policy also needs scientific acceptance: 1,203/1,930 FEAS
records were truncated although annotations may describe a longer record.
Relative arm comparisons share that input, but some labels may not be observable
in the selected window. If this is unacceptable, amend the design before new
data access; do not automatically move to Georgia or change windowing after
observing a pilot result.

Do not fill old Phase2 approval records, create invented signatures or assume
that a favorable readiness statement adopts these choices. After review, the
next authorized action should be implementation plus synthetic verification;
then separate metadata-only preflight, bounded development training, artifact
locking and one-time evaluation. None of those stages begins in this document.

## 5. Current state and verification

This continuation created only these three new planning documents. It performed
read-only code/report inspections and non-training budget/binomial calculations.
No source or target model was initialized or fitted, no signals or new
predictions were loaded, no RESERVED labels/memberships were inspected, and no
network research or downloads occurred. No tests containing synthetic fits were
rerun. The226-test record belongs to completed Phase3A, not a new implementation.

Branch remains `phase3-cross-hospital`, HEAD
`c7b795553ee9806caeb1835e7a829dc0469872eb`. Existing tracked files, Phase3A
code/results and prior reports are to remain byte-identical. No commits,
staging, push, merge, PR, ledger changes, approval grants or evaluation access.

Completed document checks: relative file links resolve, the fit-budget and
support-floor arithmetic was checked without real-data fitting, and all217
pre-existing tracked/Phase3A public files in the preservation set retained the
same combined SHA-256 before and after writing these drafts:
`9c7cf45945e11c1bc1c7998a593978a39429ad1c43e2407ab5b421638558fd2a`.
The three new drafts are the only project-file changes from this continuation;
protected private Phase2 data were not opened or rehashed. This check is not a
claim that new transfer execution code has been implemented or tested.

**Design status: READY FOR HUMAN PROTOCOL REVIEW.**
**Training/implementation authorization: NOT GRANTED.**
**Independent external clinical validation: BLOCKED.**
