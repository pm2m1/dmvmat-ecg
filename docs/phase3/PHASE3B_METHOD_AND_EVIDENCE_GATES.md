# Phase 3B method provenance and evidence gates

**ADOPTED v1 — 2026-11-14 — Parth Mishra, protocol owner (git user.name).**

Transcribed from explicit human decisions; this sprint implements/tests fictional
data only. No real metadata preparation, RESERVED access, training or evaluation
is authorized. The pre-adoption version is preserved in `archive/phase3b_pre_adoption/`.
Prospective amendment: B64/B128/B256, three sources ×three draws, and masks from
G followed by 200-update F∪G fitting. Faithful-paper work is deferred to Phase3C.

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

## 2. Adopted re-engineered A-record-v1 policy

Adopted version name: **A-record-v1**. Numerical thresholds match the existing
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
contrast >0.02. The adopted upper percentile remains
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
Adopted RNG namespaces: MMD78301+module index; shared geometry78401 for
source and78402 for target, separated from split/draw/evaluation RNGs.

For each valid label, vote ADAPT if

`U < 0.75 OR (valid MMD AND q≤0.05 AND R<0.8 AND U<1)`.

The module adapts if any valid label votes. Invalid geometry never votes;
all-invalid means SHARE plus an abstention flag. Invalid MMD prevents the
corroborative branch, but does not veto a valid `U<0.75` geometry vote.
All-SHARE adapts only the head on F∪G and aliases Head-cost under an identical job. It is not positive proof of invariance.
Log per-label support, D/R/U, bootstrap validity, MMD/p/q, seven decisions,
partial-label validity and effective parameter/buffer drift for every episode.

## 3. Typed record-versus-patient adapter

The existing evidence API names its sampling units patients and requires unique
groups (`Embeddings` and `analyze`). CinC record IDs are not patient IDs.
Pure numerical routines may be reused, but a versioned Phase 3B adapter must
carry explicit `group_kind` and `patient_linkage=unknown` metadata, use canonical
source patients and target record/known-duplicate groups, and label target
bootstrap/permutation assumptions accordingly. The new typed adapter is implemented in `phase3b/roles.py` and `phase3b/evidence.py`; only fictional in-memory execution is enabled this sprint. Passing a record ID into a patient field would not prove
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
an old synthetic-only guard. A new real-data adapter requires a separately authorized metadata preflight and reviewed
role and lineage contract; the old canonical Phase2 ledger is not that adapter.

## 4. Support implications before querying RESERVED labels

The following planning calculation uses **only** published Phase 3A CPSC FEAS
prevalences, CD506/1930 and STTC494/1930. For each label, approximate support
probability is `P(10≤K≤G−10)` with `K~Binomial(G,p_FEAS)`. It does not assume
independence between the two labels or verify actual RESERVED support.

| B | F | G | CD probability | STTC probability |
|---:|---:|---:|---:|---:|
|64|32|32|0.318770116748|0.289995614110|
|128 — primary|64|64|0.984909732369|0.980301019758|
|256|128|128|0.999999972870|0.999999940173|

Exact binomial planning calculation saved in
`configs/phase3b/support_probability_v1.json`. These are marginal probabilities,
not a joint-label independence assumption or observed RESERVED counts. The
computation ignores finite-reservoir uncertainty and unidentified patient
clusters. Floors passing does not ensure source D, retention/uncertainty validity
or predictive improvement. Do not enrich/retry G or lower minimum10. B32 has
been removed prospectively; no structural alias subtraction remains.

Formulation B is a different, secondary co-occurrence-standardized policy
requiring10 observations in each of four joint cells. It is not part of this
reduced Phase 3B matrix and must not be substituted when A disappoints.

## 5. Required synthetic execution verification

Implementation lives only in `src/dmvmat_ecg/phase3b/`. The unchanged Phase2
`SelectiveModel` freezing primitive and pure numerical evidence functions are
reused; its patient-named `Embeddings` contract and target stopping runner are
not repurposed as record/patient or fixed-update equivalents.

Tests must verify the following before any real integration is considered:

1. Complete, disjoint ownership of all seven modules and the two-output head;
   full-backbone212,066 parameters and exact10s view shapes.
2. Deterministic fresh initialization and untouched source output equivalence.
3. All-SHARE/all-ADAPT/single/mixed masks; head gradients; frozen BatchNorm
   affine parameters/running buffers and frozen dropout mode on every train call.
4. Optimizer ownership, finite gradients and unchanged source checkpoint bytes/state.
5. Explicit source canonical-patient and target record_cluster/unknown linkage;
   exact-record labels, unknown observation masks and label-union accounting.
6. Class-support/degenerate-geometry abstention, identical/shifted/class-dependent
   synthetic controls, permutation p-value arithmetic, seven-module Holm and
   fixed shared whole-vector bootstrap weights. Production MC999/4999 stays fixed;
   explicitly named reduced-MC fictional fixtures are not the production policy.
7. Count AND trainable-cost random matching, inclusion of A, one deterministic
   draw and degenerate same-mask cases; random RNG does not alter fitting RNG.
8. Mask selection bound to unmodified source/G before fitting; every target arm
   then trains on F∪G. No hidden E/C/P access, no G/O checkpoint selection.
9. Exactly200 updates with replacement, retained terminal checkpoint; no
   epoch-zero/AP-best substitution. Source selection uses only source stopping.
10. Predictive-job equivalence/immutable aliases, failed-attempt consumption,
    concurrency/restart/technical-reserve ceiling, corrupted artifact rejection.
11. Full registered prediction-roster verification, no lazy O outcome-provider
    invocation until predictions are sealed/hashed, one-time outcome access,
    and rejection of post-lock artifact/code/model changes.
12. Real namespaces denied before provider access; no old approval/ledger reuse.

The current implementation and exact executed test results are documented in
`PHASE3B_IMPLEMENTATION_VERIFICATION.md`. Unit/integration fixtures do not
establish clinical or real execution readiness, and consume no real attempts.

## 6. Phase3C-deferred literature-faithful arm — no Phase3B slots

The earlier project records the inspected authors' repository revision
`4693df334bf34cc9037ea38390efb31d539a5584` and DOI
`10.1109/TNNLS.2024.3350085` (Cui et al.; online2024, journal2025). At that
inspected revision the documented selector uses Jensen–Shannon divergence;
our MMD/permutation/A policy is a later re-engineering. The local teardown is
secondary evidence and describes incomplete model/GMM definitions, commented
MMD calls and no verified reuse licence. This is not a verification of the
current remote repository or the complete primary paper.

The requested arm—MK-MMD pretraining loss, transfer score and two-component
GMM freezing/fine-tuning of feature-extraction modules—is **DEFERRED TO A SEPARATE PHASE 3C**. Before such a protocol:

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
