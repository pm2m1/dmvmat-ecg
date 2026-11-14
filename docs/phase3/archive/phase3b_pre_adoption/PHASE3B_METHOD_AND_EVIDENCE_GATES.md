# Phase 3B method provenance and evidence gates

**DRAFT — design only. No new method implemented, no thresholds adopted.**

## 1. Keep three evidence levels separate

| Level | Existing evidence | Permitted conclusion |
|---|---|---|
| Synthetic engineering | Phase 2 synthetic selector/freezing tests and Phase 3A fixture tests | Tested software behavior on fictional data |
| Cross-dataset annotation-proxy evidence | Phase 3A FEAS-only frozen-model scores and shift diagnostics | A development-selected proxy ranking gap exists |
| Independent clinical transfer validation | Still missing confirmed-negative semantics, dependable target patient linkage and resolved source overlap | **BLOCKED** |

Successful engineering or an annotation pilot cannot clear the clinical gate.
The previous PTB-XL classification and internal age-subgroup study are closed
references, not new target selection resources. No Phase 2 O, fold 10, or
Phase 3A RESERVED outcome/prediction data were opened for this design.

## 2. Re-engineered A policy proposed for the core pilot

Proposed version name: **A-record-v1**. Numerical thresholds match the existing
policy; the target sampling unit becomes record/known-duplicate group because
patient identity is unverified. This is an explicit statistical-unit adaptation,
not a claim that the old patient contract can be used unchanged.

Static implementation evidence:
[`phase2/evidence.py`](../../src/dmvmat_ecg/phase2/evidence.py), particularly
`EvidenceConfig`, `permutation_mmd`, `relative_geometry`, `logical_vote` and
`analyze`. Historical design/provenance:
[`MULTILABEL_DMV_MAT_METHOD.md`](../phase2/MULTILABEL_DMV_MAT_METHOD.md).
These old files remain unchanged.

Owners and output dimensions are view_1…view_5:64 each, fusion_1:128,
fusion_2:64. The two-output prediction head is separate and always trainable
in adaptation. Compute all evidence from the same source initialization in
eval/no-grad mode, never from O or an already-adapted candidate.

For label l, use observed **annotation** groups:

`δ_d,l = μ_d,l,1 − μ_d,l,0`

`s_d,l = 0.5*(mean within-positive Euclidean distance + mean within-zero distance)`

`D_s,l = ||δ_s,l||/s_s,l`, `v_s,l=δ_s,l/||δ_s,l||`

`R_l = (v_s,l · δ_t,l/s_t,l)/D_s,l`.

Zero is annotation absence, not verified clinical disease absence. An unknown
annotation list is not a zero group. Require 10 observed examples in both
classes in both domains, scatter >1e-8, and source D≥0.1. Signed retention is
not replaced by its absolute value.

Conditional geometry bootstrap: 4,999 draws; source direction/scatters fixed;
preserve complete multi-label vectors and share resampling weights across
modules/labels. Require ≥95% valid draws, including positive projected source
contrast >0.02. The proposed upper percentile remains
`1−0.05/(7*2)=0.9964285714`. This has only about 18 draws in its upper tail;
record tail instability and do not describe it as an exact simultaneous bound.

MMD: biased multikernel V-statistic, pooled centering and scalar RMS radius;
bandwidths {0.5,1,2,4}; 999 permutations, `(1+extreme)/1000`, cap256 records or
known groups/domain, label-blind hash sampling, Holm across seven modules.
Invalid/degenerate tests retain invalid status and enter Holm with p=1.
Source geometry caps256 canonical source patients per class/label, with their
union and whole labels retained; G uses all permitted target evidence records.

This **differs from Phase 3A's diagnostic single-RBF median-bandwidth MMD**.
Do not substitute its statistics for the selection policy's registered kernel.
RNG namespaces proposed: MMD78301+module index; shared geometry78401 for
source and78402 for target, separated from split/draw/evaluation RNGs.

For each valid label, vote ADAPT if

`U < 0.75 OR (valid MMD AND q≤0.05 AND R<0.8 AND U<1)`.

The module adapts if any valid label votes. Invalid geometry never votes;
all-invalid means SHARE plus an abstention flag. Invalid MMD prevents the
corroborative branch, but does not veto a valid `U<0.75` geometry vote.
All-SHARE adapts only the head on F. It is not positive proof of invariance.
Log per-label support, D/R/U, bootstrap validity, MMD/p/q, seven decisions,
partial-label validity and effective parameter/buffer drift for every episode.

## 3. Critical new adapter requirement

The existing evidence API names its sampling units patients and requires unique
groups (`Embeddings` and `analyze`). CinC record IDs are not patient IDs.
Pure numerical routines may be reused, but a versioned Phase 3B adapter must
carry explicit `group_kind` and `patient_linkage=unknown` metadata, use canonical
source patients and target record/known-duplicate groups, and label target
bootstrap/permutation assumptions accordingly. This adapter is **not implemented
by this document**. Passing a record ID into a patient field would not prove
patient-disjoint clinical evaluation.
Accordingly the target floor counts10 positive and10 zero **record groups**,
not proven distinct patients. The source uses canonical known patients. The
target exchangeability assumptions of the p-values and uncertainty bounds can
fail under unidentified repeat patients; the adapter cannot repair this by
renaming its fields.

Likewise, source and target label names must explicitly identify the restricted
proxy version, not silently reuse full-superclass `CD/STTC` names. Bind mapping,
annotation observation semantics, source normalization, source initialization,
role memberships, group kind and policy version to every evidence record.
No learned representation from the future O may enter any selection.
Do not label real records or source artifacts as `synthetic_fixture` to satisfy
an old synthetic-only guard. A new real-data adapter requires its own reviewed
role and lineage contract; the old canonical Phase2 ledger is not that adapter.

## 4. Support implications before querying RESERVED labels

The following planning calculation uses **only** published Phase 3A CPSC FEAS
prevalences, CD506/1930 and STTC494/1930. For each label, approximate support
probability is `P(10≤K≤G−10)` with `K~Binomial(G,p_FEAS)`. It does not assume
independence between the two labels or verify actual RESERVED support.

| B | F | G | CD marginal-floor probability, approximate | STTC probability, approximate |
|---:|---:|---:|---:|---:|
|32|16|16|0, structurally impossible|0, structurally impossible|
|64|32|32|0.3188|0.2900|
|128|64|64|0.9849|0.9803|

Actual source D, uncertainty and co-occurrence can still invalidate geometry.
This table predicts substantial abstention without promising useful adaptation.
Do not enrich G, sample until support passes, or lower minimum10 after observing
results. Keeping B32 makes the hard information requirement visible but does
not test the efficacy of active module selection at that budget. A reviewer
who wants a nontrivial B32 selector must approve a distinct evidence-allocation
or policy study before training; it is not silently incorporated here.
Under the proposed B32 procedure G labels are still queried for the declared
support diagnostics and counted, although they cannot change the module mask.
Do not claim useful selector information at that budget. If an implementation
instead short-circuits before querying G, report its actual smaller label access
and treat it as a different allocation policy rather than silently calling it
the same B32 procedure.

Formulation B is a different, secondary co-occurrence-standardized policy
requiring10 observations in each of four joint cells. It is not part of this
reduced Phase 3B matrix and must not be substituted when A disappoints.

## 5. Comparator/freezing implementation gaps to verify next

The existing
[`SelectiveModel`](../../src/dmvmat_ecg/phase2/adaptation.py) deep-copies the
source and reapplies eval mode to SHARE modules on every train-mode change.
It provides parameter/buffer equality checks. Reuse the tested pattern in a new
namespace, with explicit two-output task ownership; do not alter the old study.

The existing `random_mask` minimizes parameter error only. It **does not enforce
the same ADAPT-module count**. The new proposed Random-A first filters that
count, then cost, includes the A mask, and uses one fixed draw. This requires
new synthetic tests; it is not already verified merely because old tests pass.

The existing `APCheckpoint` and earlier real runner use stopping data and
epoch-zero eligibility. They cannot be called unchanged for the proposed
zero-E fixed-update experiment. The new runner must forbid target checkpoint
selection, preserve the final fixed-update artifact, and report Source-only
separately. No G-as-validation loophole is allowed.

Required synthetic checks before execution: complete owner partition; same
initialization/output equivalence; all-SHARE/all-ADAPT/mixed masks; frozen
BatchNorm affine/running statistics and dropout modes; fair optimizer ownership;
record/patient unit labels; unknown annotation masks; class-support abstention;
both-count-and-cost random matching; deterministic identical-job reuse; fixed
terminal checkpoint with no E/G/O selection access; label-union accounting;
correct full-matrix prediction sealing; source-file integrity and failed-attempt
counting. Existing Phase 3A's226 passing tests do not establish these new gaps
are already closed.

## 6. Literature-faithful arm — evidence gate, not a claimed implementation

The earlier project records the inspected authors' repository revision
`4693df334bf34cc9037ea38390efb31d539a5584` and DOI
`10.1109/TNNLS.2024.3350085` (Cui et al.; online2024, journal2025). At that
inspected revision the documented selector uses Jensen–Shannon divergence;
our MMD/permutation/A policy is a later re-engineering. The local teardown is
secondary evidence and describes incomplete model/GMM definitions, commented
MMD calls and no verified reuse licence. This is not a verification of the
current remote repository or the complete primary paper.

The requested arm—MK-MMD pretraining loss, transfer score and two-component
GMM freezing/fine-tuning of feature-extraction modules—remains **conditional**:

1. Verify exact primary-paper objective, MMD kernels/weights, where target
   data enter, score formula/direction, GMM inputs/initialization/cluster mapping,
   eligible modules, and whether fusion/head are selected or always updated.
2. Pin the authoritative source revision, dependencies and usable licence;
   distinguish paper-described mathematics from code actually wired into
   training. Obtain permitted access to the primary material; no network
   research was performed in this offline design sprint.
3. Resolve task/view differences: EEG frequency views are not ECG lead/rhythm
   views. At most this can reproduce a verified policy in an ECG adaptation,
   not reproduce the original brain-decoding experiment.
4. If MK-MMD changes source initialization, add paired head/full/random controls
   from that same initialization and account for all target labels and unlabeled
   target access. A standalone aligned-source arm versus ordinary-source A
   cannot isolate module selection. No O signals may enter pretraining, even
   unlabeled, unless a distinct transductive protocol is explicitly approved.
5. Register its complete finite fit allocation and synthetic tests before O is
   accessed. It has **no core pilot fit slots** and may not use technical reserves.

Until these gates pass, label the arm NOT EXECUTED / PROVENANCE UNRESOLVED.
Do not silently implement JS as MK-MMD, invent a GMM score, or describe an
inspired hybrid as a faithful reproduction. This restriction does not prevent
separate review of the clearly named re-engineered A pilot.

## 7. Interpretation gates

Beating Full-cost alone supports a policy-level exploratory comparison, not
proof that MMD evidence chose uniquely beneficial modules. Tying Head-cost or
an identical random mask limits that interpretation. Increased ADAPT counts,
stable masks, significant MMD and better class geometry are not predictive
success metrics. Preserve negative transfer and abstentions in the primary
mean. Clinical-negative adjudication and patient/source independence remain
separate unresolved scientific requirements.

One-vs-rest annotation classes also mix the other diagnosis. Changes in
co-occurrence or reporting can change R and MMD without destroying useful
conditional features. The future results must distinguish mask stability,
representation shift, predictive benefit and clinical validity. The existing
FEAS-informed protocol history is common to the methods and must be disclosed
alongside their new B-record information budgets.
