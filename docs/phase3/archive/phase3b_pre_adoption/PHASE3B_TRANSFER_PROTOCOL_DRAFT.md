# Phase 3B: label-cost-matched selective transfer

**DRAFT — REQUIRES HUMAN REVIEW AND SEPARATE IMPLEMENTATION/TRAINING
AUTHORIZATION. Not adopted, registered, implemented or executed.**

This document proposes a new cross-dataset **ECG annotation-proxy** pilot. It
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

## 2. Frozen endpoint proposed for adoption

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

## 3. Source and target roles — proposed, not generated

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

Use canonical JSON encoding for these proposed hashes: UTF-8,
`json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)`.
For a stochastic integer seed, take the first eight SHA-256 bytes as an unsigned
big-endian integer, modulo 2^32. Membership ordering uses the complete hexadecimal
digest. Draw ordering uses `[draw_seed, group_id]`; stochastic fitting uses
`[78501, source_seed, draw_seed, B]`; random masks use
`[78601, source_seed, draw_seed, B]`. These are proposed conventions, not generated
manifests or active configuration.

CPSC also has a duration limitation: 1,203 of its 1,930 Phase3A FEAS records
required first-10s truncation, while the annotation can describe the entire
recording. A diagnosis may therefore refer to signal outside the visible input.
Keep the same deterministic input in every arm and disclose this possible
irreducible label/window mismatch. A different hospital or whole-record
inference policy requires a prospective amendment, not a post-result switch.

## 4. Budgets and label-blind episodes

Recommended reduced study: source seeds **15101, 15102** and target-draw seeds
**16101, 16102**. For each draw, hash-order R IDs using that draw seed and take
128 records without replacement, without reading their labels first. Assign
the first 64 to F and the next 64 to G. For budget B, take the first B/2 from
each corresponding list. Budgets are nested; F and G never overlap.

| Newly exposed episode labels B | F: gradient fitting for A/random | G: evidence | E: target stopping | C: calibration | P: policy selection |
|---:|---:|---:|---:|---:|---:|
| 32 | 16 | 16 | 0 | 0 | 0 |
| 64 | 32 | 32 | 0 | 0 | 0 |
| 128 — primary | 64 | 64 | 0 | 0 | 0 |

The same draw memberships are used for both source seeds and all methods. Do
not query extra labels to enforce positive quotas, replenish an inconvenient
sample, or repair selector support. There are at most 256 distinct adaptation
labels across the two maximum-size draws before overlaps; repeated seeds and
methods do not create additional unique labeled records. O outcome acquisition
is reported separately and must never influence development.
Each fully observed record supplies two annotation entries, so report both
record counts and observed record-label entries. Do not top up missing entries
by querying extra records outside B.

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

The methods share this frozen prior research history. In the proposed new
training stage, none may fit, stop, calibrate, tune or choose checkpoints using
FEAS; all access to new R annotations is constrained to the paired B-record
union. Full-cost can train on every record in that union. This supports a
**conditional incremental-label comparison**, not an end-to-end claim that the
entire project used only 32/64/128 target labels. Report the historical exposure
and the maximum new 256-record adaptation union separately; do not sum overlapping
cohort-audit counts or call public annotation access new clinical adjudication.

**Important structural limit:** the unchanged Formulation A floor is 10 observed
positives **and** 10 observed negatives per label. With G=16, neither label can
ever pass. B=32 therefore tests the declared head-only abstention fallback, not
nontrivial evidence-guided module selection. Keep this fact explicit. Do not
lower the floor to make the experiment look active. B=64 has limited support;
B=128 is the primary contrast. Further geometry/uncertainty checks can abstain
even when raw class counts pass; see the method companion document.

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
| Random-A | Same source | F | G through A mask/cost calculation | B | 1 or identical-job reuse |
| Selective-A — primary method | Same source | F | G | B | 1 |
| Target-only-cost | Fresh weights | F∪G | Same labels available | B | 1 |

Thus Full-cost and Head-cost may use every permitted evidence label for
gradient fitting; the selector does not get free extra supervision. Source-only
has no adaptation-label requirement. Equal target labels do not imply equal
source-data cost or equal gradient-data usage, and those distinctions must be
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

An alternative E=256 validation/epoch-zero design would consume totals
288/320/384 development labels, not 32/64/128, and requires a different approved
protocol. It is not executed or included in this matrix.

All-SHARE means head-only fitting on F, not source-only and not Head-cost.
SHARE parameters and buffers remain bitwise unchanged, including BatchNorm
statistics and affine parameters; frozen dropout stays in eval mode. ADAPT
modules and their BatchNorm follow training mode. The head is always trainable
except Source-only. All arms share update counts, **not guaranteed wall time**.

## 7. Selection policy and random control

Use Formulation A only as the primary selective policy. It is the re-engineered
MMD/permutation/relative-geometry method, not an original-paper reproduction.
Here A means the proposed **A-record-v1**: the same numerical decision rule,
but a declared target record/group sampling unit rather than verified patients.
This unit change requires review and is not evidence of patient-level validity.
The companion specifies exact thresholds, invalid-evidence handling and the
record-versus-patient adapter gap. Formulation B, learned masks, architecture
search and new regularizers are not core arms.

For Random-A enumerate 128 masks, first require the same number of ADAPT
modules as A and then minimize absolute trainable-parameter-count error.
Because A itself is allowed, zero mismatch is always attainable. Draw uniformly
from all exact-cost, same-count masks, **including A**. One fixed random draw
per source/draw/B, seed namespace 78601; do not redraw until different. Report
candidate-set size, equality probability, chosen mask, counts and parameter
cost. This is an explicitly proposed tightening of the current parameter-only
random helper, not a change to the historical implementation.

Identical fully specified fitting jobs reuse one immutable completed artifact.
At B=32, A and Random-A are structurally identical; their contrast is zero by
design. Log the alias rather than perform a redundant fit or pretend the two
results are independent. Other equality/abstention episodes remain included.

## 8. Attempt allocation and source-only reuse

Recommended: two source seeds × two paired draws × three budgets.

| Component | Logical fit slots | Maximum distinct planned fits |
|---|---:|---:|
| Fresh source training | 2 | 2 |
| B32: five target training procedures × four source/draw pairs | 20 | 16 |
| B64 | 20 | 20 |
| B128 | 20 | 20 |
| Total planned | 62 | 58 |
| Technical failure/retry reserve | — | 4 |
| **Hard ceiling of actual predictor-fit attempts** | — | **62** |

Further exact-job aliases can reduce the actual count; unused capacity cannot
be spent on new candidates or favorable reruns. Failed, interrupted and real
training smoke fits count, reserved before fitting. No exact trajectory resume
claim without validated optimizer/RNG/sampler state; otherwise retry fresh under
a new ID. If required controls cannot finish within the ceiling, stop incomplete.
Source-only predictions reuse each of the two new source models without fits.
The larger and optional-reference budgets are separate choices, not authorized
extensions; see the compute document.
The four technical attempts are a separate cap, even if artifact aliases leave
unused headroom. Permit at most one fresh retry per registered job for a logged
environmental interruption under unchanged predictive settings. Deterministic
nonfinite behavior or a needed scientific/code change requires review, not
repeated retries.

## 9. Primary and secondary analysis

Primary budget is B128. For each source seed s and draw d, compute

`Δ_s,d = 0.5*(AP_CD(A)+AP_STTC(A)) − 0.5*(AP_CD(Full-cost)+AP_STTC(Full-cost))`.

The primary estimand is the equal mean of the four Δ values. Use exact sklearn
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
label efficiency are descriptive secondary results. B32/B64 contrasts are
exploratory and must not replace the primary budget or be promoted as a new
primary finding. No probability calibration fitting is in scope.
For log loss only, clip probabilities to [1e-7,1−1e-7]; never clip ranking scores
or Brier inputs. ECE uses15 fixed equal-width probability bins, left-closed and
right-open except for the last closed bin, weighted by observed record count.
Operating metrics use0.5; undefined denominators are NA. Report these as
annotation-classification statistics, not clinical sensitivity or risk calibration.

Positive primary point estimates are not sufficient. A primary interval entirely
above zero provides **conditional exploratory evidence** over Full-cost only.
A proposed practical margin is +0.01 mean AP, not a clinically validated margin;
distinguish a +0.01 point estimate from an interval entirely above +0.01. A CI
crossing zero is inconclusive, not proof of no effect. Evidence-guided selection
specifically is not established by beating full fine-tuning while tying ordinary
head-only or a degenerate random mask.

The intervals condition on fitted models and selected episodes. Four paired
fits are not four independent clinical populations. Report seed/draw contrasts
and their spread separately; do not run a patient-population t-test on model
seeds. Unknown patient clustering, source training/model selection and FEAS-based
target selection are not captured by these CIs. Power for +0.01 is not established.

## 10. Future execution order and stopping rules

1. Human reviews endpoint semantics, record-level dependence, allocation,
   fixed-checkpoint policy, policy provenance and budget. This document grants
   no authority and contains no approval fields to auto-populate.
2. Separately authorize implementation and synthetic tests of a new adapter,
   fixed-update runner and both-count-and-cost random control. Do not modify
   Phase 2 or Phase 3A code/artifacts. No real fits during that sprint.
3. Separately authorize metadata-only R/O/draw preparation and frozen-hash
   preflight. Verify the original source/target checksum and exclusion lineage.
   No O outcomes or signals are needed for development preparation.
4. After a bounded training authorization, initialize a fresh run namespace;
   use a simple durable attempt log, hash-bound role manifests and existing
   tested safety patterns. Do not reuse the canonical Phase 2 ledger,
   approvals, checkpoints or evaluation session. No new signature bureaucracy
   is proposed here.
5. Train sources, compute permitted evidence and execute the full registered matrix;
   preserve failures and exact-job aliases. Do not make new model/policy
   selections from target evaluation. No optional arm may be added after O.
6. Freeze all artifacts and request separate one-time evaluation authorization.
   Seal complete registered O predictions before outcome/metric access; then
   perform the fixed analysis. No further fits, recalibration, altered masks
   or outcome-guided retries after access.

Stop before real fitting on a hash/schema mismatch, cross-role known-group
overlap, source/task-lineage error, unsupported model operation, hidden label
access, changed freeze policy, or insufficient attempt budget. Do not silently
relax floors, fix a poor result by changing preprocessing, or substitute another
hospital. Geometry abstention itself is a valid result, not a technical failure.

## 11. Approval scope still needed

The recommended choice is the reduced, CPSC-only annotation pilot with the
58-distinct-fit upper bound plus four technical attempts, fixed terminal target
checkpoints and A primary. Approval must explicitly accept or reject the B32
structural abstention and the residual patient/clinical-label limitations.
Neither the optional XResNet extension nor a purported faithful-paper arm is
included automatically. Georgia backup training, clinical endpoint changes,
literature-network access and any real data preparation/fitting/evaluation
require their own clearly scoped subsequent instructions.
