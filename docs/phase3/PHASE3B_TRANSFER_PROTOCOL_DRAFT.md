# Phase 3B: label-cost-matched selective transfer

**ADOPTED v1 — 2026-11-14 — protocol owner: Parth Mishra (git user.name).**

Adoption transcribes the owner's explicit review decisions in the current instruction.
Implementation and synthetic verification are authorized; real metadata preparation,
RESERVED access, training and evaluation are NOT authorized. The filename is retained
for link continuity. The previous draft is archived byte-for-byte under
`archive/phase3b_pre_adoption/`. No earlier approval record is changed.

**Prospective, pre-data amendment:** budgets become B64/B128/B256; Selective-A
and Random-A determine masks using G, then fine-tune on F∪G. The owner supplied
Cui et al., TNNLS 2025 Algorithm 1 as the rationale for reusing target training
data for scoring and fine-tuning. This is the owner's rationale, not a claim
that the complete paper or faithful algorithm was independently reproduced in
this offline sprint. The faithful arm is deferred to a separate Phase 3C.
The 3×3 design is adopted; no XResNet extension. CPSC first-10s truncation is
explicitly accepted and disclosed, identically for all arms.

This document specifies the adopted cross-dataset **ECG annotation-proxy** pilot. It
does not reopen Phase 2, PTB-XL fold 10, or the completed Phase 3A experiment.
No RESERVED labels, waveforms or predictions were accessed to prepare it.

## 1. Question and interpretation

Primary question: does the re-engineered evidence-guided seven-module policy
improve target mean CD-proxy/STTC-proxy Average Precision over full fine-tuning
with the **same incremental episode-level target-label budget**, including
evidence labels, and optimizer-update budget? Historical target-label access
is separately disclosed below; it cannot be erased. Architecture optimization
and attaining a particular AUROC are outside
scope. Freezing alone, informed selection, and improved clinical diagnosis are
different claims; this pilot must distinguish them.

Phase 3A selected CPSC as the primary target, Georgia as backup. Its five-view
proxy-matched macro AUROC differences were −0.09866, −0.05199 and −0.03598 for
CPSC, Georgia and Chapman–Ningbo respectively. Those are feasibility results,
not selective-transfer improvements. References:
[feasibility report](CROSS_HOSPITAL_FEASIBILITY.md),
[interpretation supplement](EXECUTION_INTEGRITY.md), and
[`METRICS.json`](../../results/phase3a/METRICS.json).

The pilot is development-selected, exploratory, cross-dataset annotation
classification. It cannot establish independent external **clinical** validation:
target patient identities and clinically confirmed negative labels are absent,
and the Phase 3A source-versus-target waveform cross-match was waived.

## 2. Adopted endpoint

Two independent sigmoid outputs, in order `CD-proxy-v1`, `STTC-proxy-v1`.
Retain exactly the Phase 3A named-condition unions; do not prune conditions
because their observed AUROC was low.

| Proxy | Target SNOMED codes | Source SCP statement keys |
|---|---|---|
| CD | 270492004, 445118002, 59118001, 698252002, 713426002, 713427006, 733534002 | 1AVB, CLBBB, CRBBB, IRBBB, IVCD, LAFB |
| STTC | 164930006, 164934002, 428750005, 429622005, 59931005 | INVT, NDT, NST_, NT_, STD_, STE_, TAB_ |

Binding reference: [`LABEL_MAPPING.json`](../../results/phase3a/LABEL_MAPPING.json),
SHA-256 `ecce890b4678b8f2a06cf0796476221adceb8156c7d476ecfbeba910a43891d6`.
Code 164909002 remains excluded because of the documented conversion ambiguity.
The restricted labels are not the original full PTB-XL superclasses. Source
statement likelihoods retain the existing any-key convention, including zero.

A positive means an included annotation was recorded. A zero means no included
annotation was recorded, **not clinical absence**. A valid, nonempty publisher
annotation list permits an observed annotation target; genuinely missing,
malformed or uninterpretable annotation data remain unknown, with a false
observation mask, never fabricated negatives. The implementation must distinguish
`annotation_observed` from clinical ascertainment. No patient-level OR of
different ECG diagnoses is permitted.

Source eligibility remains the historical diagnostic-superclass-eligible
PTB-XL population, while target eligibility is the valid CinC annotation
population. This population-selection difference also requires review; matched
condition names do not remove it or make reporting completeness equivalent.

## 3. Adopted role definitions — real memberships NOT generated

**Source:** PTB-XL v1.0.3, official folds 1–8 for training and 9 for source
checkpoint selection. Reuse the established eligibility/membership definition
(historically 17,084 training and 2,146 validation recordings); verify all hashes
and patient separation in a separately authorized preflight. Fold 10 is excluded.
All serial source recordings retain their own labels. Source evidence anchors
use the lowest eligible ECG ID per source-training patient, independently of
diagnoses and predictions; do not pool serial diagnoses.

**Target:** only the CPSC RESERVED ID set from Phase 3A. Its manifest hash is
`84bea540fef6e04d5638f7b09f1107de5b0335a0e58df1d2b93007e8bed16cf1`
and it contains 7,719 records. This hash identifies the complete Phase 3A
FEAS/RESERVED manifest; future preparation must select its RESERVED member set,
never regenerate the Phase 3A split. The 1,930 CPSC FEAS records are excluded
from all pilot fitting and final evaluation. They informed this design only.

After approval, sort RESERVED native-record/duplicate-group IDs by
`SHA256(canonical_JSON([77101, group_id]))`, with literal ID as a collision
tie-break. Assign the first `floor(N/2)` to adaptation reservoir R and the rest
to sealed evaluation O: expected counts 3,859 and 3,860, subject to membership
verification. No labels or model outputs enter this allocation. Freeze both
membership hashes once; do not search seeds or retry splits for support.

All known native exact duplicates were already removed before Phase 3A splitting.
Retain their original grouping/exclusion provenance. There are **no verified
target patient IDs**: this is record/known-duplicate-group separation, not proven
patient separation. Do not put record IDs in a field named `patient_id` and claim
otherwise. Clinical validation remains blocked unless identity and label evidence
are strengthened. Human acceptance of this limitation is required even for the
exploratory annotation pilot.

The FEAS prevalence-based expectation for the proposed O is approximately
1,012 CD annotations and 988 STTC annotations. These are planning estimates,
not observed O counts. No O labels have been queried for this proposal.

Use canonical JSON encoding for these adopted hashes: UTF-8,
`json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)`.
For a stochastic integer seed, take the first eight SHA-256 bytes as an unsigned
big-endian integer, modulo 2^32. Membership ordering uses the complete hexadecimal
digest. Draw ordering uses `[draw_seed, group_id]`; stochastic fitting uses
`[78501, source_seed, draw_seed, B]`; random masks use
`[78601, source_seed, draw_seed, B]`. These are adopted conventions, not generated real memberships or execution grants.

CPSC also has a duration limitation: 1,203 of its 1,930 Phase3A FEAS records
required first-10s truncation, while the annotation can describe the entire
recording. A diagnosis may therefore refer to signal outside the visible input.
Keep the same deterministic input in every arm and disclose this possible
irreducible label/window mismatch. A different hospital or whole-record
inference policy requires a prospective amendment, not a post-result switch.

## 4. Budgets and label-blind episodes

Source seeds **15101/15102/15103** and draw seeds **16101/16102/16103**
are fixed prospectively. For each draw, hash-order R known-group IDs with the
specified draw seed, take 256 without replacement, assign the first 128 to the
F master list and the next 128 to G. At budget B take the first B/2 from each
master list. Thus F/G are disjoint and nested across budgets; no labels enter
ordering, no quotas, replenishment, enrichment or seed shopping are permitted.
The same memberships are reused for all source seeds and methods.

| Incremental episode labels B | F | G for evidence AND fitting | E | C | P |
|---:|---:|---:|---:|---:|---:|---:|
|64|32|32|0|0|0|
|128 — primary|64|64|0|0|0|
|256|128|128|0|0|0|

All five target procedures train on F∪G. Selective-A/Random-A first select a
mask from source anchors and G, under the untouched source model, then reuse
F∪G for gradient updates. This is explicitly permitted development reuse, not
independent validation of the selector. No G checkpoint selection is allowed.
There are at most **768 distinct new adaptation records** across the three
B256 draws before overlap; repetitions across methods/seeds do not add unique
labels. Report records and observed record-label entries (two per fully
observed record), not presumed patient counts. Missing annotations stay masked;
never replace a record to top up an unknown annotation. O acquisition is separate.

**Historical label-access accounting — overlapping sets, not additive costs:**
the release is already annotated; B is a simulated restriction on a new
procedure's access, not evidence that only B ECGs were ever annotated or viewed
by this research project.

| Previous exposure | Records | Role and limitation |
|---|---:|---|
| All five target-source raw header annotations | 65,826 | Historical dataset/duplicate/code audit, not new predictor fitting |
| Eligible target annotation vectors used for Phase3A stratification | 64,915 | Cohort-auditor access, including RESERVED; not sealed-from-all-humans labels |
| Phase3A model-aware FEAS outcomes across all domains | 12,983 | Used for zero-shot metrics and domain/design selection |
| CPSC subset of those FEAS outcomes | 1,930 | Common target-specific development history |
| Phase3A RESERVED across domains | 51,932 | Stratification history only; no model predictions or later label summaries |
| CPSC RESERVED subset | 7,719 | Future R/O parent population, not pristine clinical confirmation |

The methods share this frozen prior research history. In the adopted new
training stage, none may fit, stop, calibrate, tune or choose checkpoints using
FEAS; all access to new R annotations is constrained to the paired B-record
union. Full-cost can train on every record in that union. This supports a
**conditional incremental-label comparison**, not an end-to-end claim that the
entire project used only 64/128/256 target labels. Report the historical exposure
and the maximum new 768-record adaptation union separately; do not sum overlapping
cohort-audit counts or call public annotation access new clinical adjudication.

**Support limit:** the unchanged A floor is ten observed positive and ten
observed annotation-zero groups per label. B64/G32 frequently fails this floor;
B128/G64 and B256/G128 improve expected count support without guaranteeing
valid geometry, low uncertainty, useful adaptation or predictive improvement.
All abstentions remain in the primary procedure. See the companion support table.
B32 is removed prospectively and has no slots or guaranteed aliases.

## 5. Models, initialization and preprocessing

Train a **new source model** per source seed on the fixed proxy task: the
unchanged five ECG encoders, two fusion modules, and a two-logit head. Expected
parameter count is 212,066, obtained from the existing 212,261-parameter model
by replacing the 325-parameter five-output head with a 130-parameter head.
Verify this synthetically before any real fit. Do not initialize from Phase 2
or historical five-superclass checkpoints, nor train a new architecture.

Fit a fresh scalar normalization on source training only. Source validation,
G, target R and O cannot fit it. Each pretrained comparator uses the exact same
source checkpoint and normalization within its paired episode. Target-only
starts from fresh random weights and fits normalization using its own F∪G only.
Report the source-label/pretraining cost separately from target-label cost.

Retain 12 canonical leads, physical mV conversion, 100 Hz, first 10 seconds for
long records, and exclusion of short records. No padding, augmentation, target
normalization refit for pretrained arms, or CNN sliding windows. Frequency and
temporal views remain exactly as in
[`ECG_ARCHITECTURE.md`](../ECG_ARCHITECTURE.md). Follow the actual WFDB signal
filename, including the documented header alias; never rename raw release data.

Source training: CPU float32, four Torch threads, 50 completed epochs, batch
128, AdamW max LR 0.01, weight decay 0.01, betas (0.9,0.99), epsilon 1e-8;
OneCycleLR with pct_start 0.3, div_factor 25, final_div_factor 10000,
base/max momentum 0.85/0.95. BCE over observed annotation entries. Select by
source fold-9 **mean proxy AP**, then smaller BCE, then earlier epoch, with
1e-12 metric tie tolerance. This is a declared task/checkpoint change from the
historical five-class benchmark, not an exact reproduction claim. No source
hyperparameter search is included.
Scratch epoch zero may be logged but is not an eligible source checkpoint.

The Phase3A gap was measured with a five-superclass-trained head. It is unknown
whether it survives task-aligned source pretraining. Do not select a source,
training recipe, target or execution decision by reopening FEAS after the new
source fits. Source-only is a mandatory locked comparison; if it transfers
well, report that finding and the remaining adaptation effects without assuming
the old gap still applies. A fresh source-FEAS gating/tuning exercise would be
additional target model-development access and requires an explicitly costed
different protocol. New code must handle two logits, not reuse Phase3A's
five-output indices. None of this alters the old scores or initiates source fits.

## 6. Exact core comparator matrix

| Procedure | Initialization | Gradient labels | Evidence label access | Newly exposed target development labels | Predictor fits per episode |
|---|---|---|---|---:|---:|
| Source-only | Paired new source | None | None | 0 | 0 |
| Head-cost | Same source | F∪G | Same labels available | B | 1 |
| Full-cost — primary comparator | Same source | F∪G | Same labels available | B | 1 |
| Random-A | Same source | F∪G | G through A mask/cost calculation, before fitting | B | 1 or identical-job reuse |
| Selective-A — primary method | Same source | F∪G | G, before fitting | B | 1 or identical-job reuse |
| Target-only-cost | Fresh weights | F∪G | Same labels available | B | 1 |

Thus Full-cost and Head-cost may use every permitted evidence label for
gradient fitting; the selector does not get free extra supervision. Source-only
has no adaptation-label requirement. Equal target labels do not imply equal
source-data cost or equal computational cost, and those distinctions must be
reported. A separate Full-F is not needed to claim the primary cost comparison
and is outside this reduced matrix.

Adaptation: 200 optimizer updates, batch 32 sampled with replacement from the
permitted fitting set, AdamW constant LR 0.0003, weight decay 0.01, betas
(0.9,0.99), epsilon 1e-8, CPU float32/four threads. Set the same deterministic
sampler and Torch stochastic seed for paired methods using a hash of source
seed/draw/B in a separate RNG namespace. Random-mask RNG must not perturb it.
Do not add clipping, weighting, resampling by diagnosis, or loss terms.

**Fixed terminal checkpoint:** retain update 200, not a target-validation best.
There is no target early stopping, target AP selection or epoch-zero selection.
Source-only is the explicit untouched reference; negative transfer stays in the
results. Epoch zero may be saved for integrity diagnostics but is not chosen
using G or O. Nonfinite loss/gradients are technical failures, not permission to
select an earlier favorable checkpoint. This differs from the Phase 2
validation/epoch-zero policy and must be approved explicitly.

All-SHARE means Head-cost on F∪G, with the same initialization, sampler and seed; reuse its artifact. It is not Source-only.
SHARE parameters and buffers remain bitwise unchanged, including BatchNorm
statistics and affine parameters; frozen dropout stays in eval mode. ADAPT
modules and their BatchNorm follow training mode. The head is always trainable
except Source-only. All arms share update counts, **not guaranteed wall time**.

## 7. Selection policy and random control

Use Formulation A only as the primary selective policy. It is the re-engineered
MMD/permutation/relative-geometry method, not an original-paper reproduction.
Here A means the adopted **A-record-v1**: the same numerical decision rule,
but a declared target record/group sampling unit rather than verified patients.
The owner accepted this unit change for exploratory work; it does not establish patient-level validity.
The companion specifies exact thresholds, invalid-evidence handling and the
record-versus-patient adapter gap. Formulation B, learned masks, architecture
search and new regularizers are not core arms.

For Random-A enumerate 128 masks, first require the same number of ADAPT
modules as A and then minimize absolute trainable-parameter-count error.
Because A itself is allowed, zero mismatch is always attainable. Draw uniformly
from all exact-cost, same-count masks, **including A**. One fixed random draw
per source/draw/B, seed namespace 78601; do not redraw until different. Report
candidate-set size, equality probability, chosen mask, counts and parameter
cost. This is an explicitly adopted tightening of the current parameter-only
random helper, not a change to the historical implementation.

Identical fully specified jobs share one immutable artifact. Equivalence
requires the same initialization, normalization, F∪G identity/order, seven
module modes, optimizer, sampler/stochastic seed, 200 updates, code and dtype.
Method names and evidence receipt names do not make identical jobs distinct.
All-SHARE A/Random-A alias Head-cost; all-ADAPT alias Full-cost; Random-A may
alias A for a mixed mask. Include these episodes and zero contrasts in analysis.
There are **no guaranteed aliases** at the new budgets; do not subtract a
speculative alias count from the prospective maximum.

## 8. Adopted finite attempt allocation

| Component | Arithmetic | Planned fit slots |
|---|---|---:|
| Fresh source models |3 source seeds|3|
| B64 target fits |5 procedures ×3 sources ×3 draws|45|
| B128 target fits |5 ×3 ×3|45|
| B256 target fits |5 ×3 ×3|45|
| **Planned maximum before outcome-dependent aliases** |3+135|**138**|
| Technical retry reserve |at most one fresh retry per eligible interrupted job|6|
| **Hard ceiling, failures/interruptions/smoke fits included** |138+6|**144**|

Source-only predictions reuse the three source fits; no additional source-only
training slots. Exact aliases reduce actual fits, never authorize extra work.
Reserve use remains separately capped at six even when alias savings leave
headroom. No exact optimizer/RNG resumption is claimed: permitted environmental
interruptions require a new attempt ID and fresh initialization under the
unchanged declared seed. Deterministic numerical/scientific failures require
review, not repeated retries. Stop incomplete if required controls cannot finish.
An XResNet extension is rejected for this design. The faithful arm is deferred
and unbudgeted in a separate Phase3C, not a reserve use.

## 9. Primary and secondary analysis

Primary budget is B128. For each source seed s and draw d, compute

`Δ_s,d = 0.5*(AP_CD(A)+AP_STTC(A)) − 0.5*(AP_CD(Full-cost)+AP_STTC(Full-cost))`.

The primary estimand is the equal mean of the nine Δ values. Use exact sklearn
non-interpolated AP from continuous raw scores. Do not substitute pooled OOF
scores, a probability ensemble, or a best-performing seed/draw/budget.
Here raw ranking scores mean unclipped sigmoid probabilities, without fitted
calibration. Do not change to logits after inspecting which result is favorable.

Use 10,000 paired evaluation-record/group bootstrap draws, seed 78701. Resample
the same available record groups across all methods, source seeds, draws,
budgets and both labels, preserving complete label vectors. After Phase 3A
canonicalization these are effectively singleton record groups, not known
patients. Calculate a 95% percentile interval for the primary contrast. Report
undefined replicate counts; do not invent negatives or quietly resample to make
a difficult label disappear. A rare unsupported label makes its metric and
the corresponding macro contrast unsupported. Require both annotation classes
for both labels in the evaluation population, even if a degenerate mathematical
AP value exists. Do not redraw undefined bootstrap samples: use percentiles of
valid contrasts, report their denominator and conditional interpretation, and
do not report a primary CI if fewer than9,500 of10,000 draws are valid.

At B128 the four secondary AP contrasts are A−Head-cost, A−Random-A,
A−Source-only and A−Target-only-cost. Report ordinary 95% and Bonferroni
98.75% intervals for that fixed family. Per-label AP/AUROC, mean AUROC, Brier,
log loss, fixed-bin ECE, fixed-0.5 operating metrics, negative-transfer frequency,
parameter drift, mask stability, invalid-geometry frequency, training time and
label efficiency are descriptive secondary results. B64/B256 contrasts are
exploratory and must not replace the primary budget or be promoted as a new
primary finding. No probability calibration fitting is in scope.
For log loss only, clip probabilities to [1e-7,1−1e-7]; never clip ranking scores
or Brier inputs. ECE uses15 fixed equal-width probability bins, left-closed and
right-open except for the last closed bin, weighted by observed record count.
Operating metrics use0.5; undefined denominators are NA. Report these as
annotation-classification statistics, not clinical sensitivity or risk calibration.

Positive primary point estimates are not sufficient. A primary interval entirely
above zero provides **conditional exploratory evidence** over Full-cost only.
The adopted practical margin is +0.01 mean AP, not a clinically validated margin;
distinguish a +0.01 point estimate from an interval entirely above +0.01. A CI
crossing zero is inconclusive, not proof of no effect. Evidence-guided selection
specifically is not established by beating full fine-tuning while tying ordinary
head-only or a degenerate random mask.

The intervals condition on fitted models and selected episodes. Nine paired
source/draw cells are not nine independent clinical populations. Report seed/draw contrasts
and their spread separately; do not run a patient-population t-test on model
seeds. Unknown patient clustering, source training/model selection and FEAS-based
target selection are not captured by these CIs. Power for +0.01 is not established.

## 10. Authorization stages and stopping rules

1. Human adoption is transcribed above. Items1/2/3/5/6/8 were approved as
   drafted; items4/7 were modified explicitly. No synthetic result adopts or
   modifies another human decision.
2. This sprint implements the adopted methods in `src/dmvmat_ecg/phase3b/`
   and tests fictional tensors only. It neither prepares R/O nor enables a
   real-data backend. Phase2/3A source and results remain unchanged.
3. **Separate authorization required:** metadata-only R/O and nested draw
   preparation, lineage preflight and supported real execution integration.
   Verify source/target hashes, known-group separation, annotation mapping,
   and episode label access without model fitting or O outcome summaries.
4. **Separate bounded training authorization required:** fresh sources,
   permitted G/source-anchor extraction, full registered adaptation matrix,
   immutable artifacts, exact-job aliases, durable attempt accounting ≤144.
   No Phase2 ledger/approvals/checkpoints may be reused.
5. **Separate evaluation authorization required:** finalize all registered
   fits/aliases, freeze artifacts, then seal the complete 138-logical-predictor
   roster (three Source-only plus135 target procedures) before any O outcomes.
   Reuse Source-only vectors across budgets/draws; no ensemble or reselection.
6. After one-time outcome access: no fits, recalibration, changed masks,
   changed hypotheses or outcome-guided retries. Use immutable sealed predictions
   for the registered analysis only.

Stop on hash/schema mismatch, prohibited group overlap, historical/task-lineage
error, nonfinite training, hidden role access, freeze-policy change or insufficient
attempt budget. Geometry abstention is a valid result, not a technical failure.
Do not change the target hospital/window/thresholds after viewing a result.
The clinical-validation gate remains BLOCKED. This adopted annotation-proxy
study does not establish verified patient independence or clinical negatives.
