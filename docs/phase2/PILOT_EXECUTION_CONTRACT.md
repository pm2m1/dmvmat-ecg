# Phase 2B execution contract — prospective, not adopted

**DRAFT. Preparation and synthetic verification only. Training and evaluation
permissions remain DENY. Scientific role-allocation review is required.**

## Endpoint and evidence boundary

The actual [internal proposal](INTERNAL_TRANSFER_PILOT_PROPOSAL.md) uses the ordered
pair **CD, STTC**, not five outputs and not IAVB/IRBBB. The five original classes
are the source annotation vocabulary; this pilot projects cached development
columns3/2 into CD/STTC order after verifying that vocabulary. This is recorded
superclass-annotation classification. A zero means no corresponding recorded
superclass tag, not a clinician-confirmed negative. Any-key/likelihood-zero
semantics are inherited explicitly; clinical unknowns must not be relabeled
negative. The general selector handles observation masks; this particular
annotation task has two observed tag-presence statuses per eligible record.

This endpoint is sufficiently specified for an **exploratory internal benchmark**,
subject to approval, but does not resolve external clinical harmonization. A
handles arbitrary label counts; its present multiplicity quantile is
`1−.05/(7×2)=.99642857`, and MMD uses seven-module Holm adjustment. B remains
restricted to two labels and all four verified joint cells require≥10 patients.
It is not generalized to five labels. Any endpoint/matrix expansion needs a
prospective amendment and new synthetic verification, never a favorable-score
substitution. Original-author JS divergence and our later MMD re-engineering
remain distinguished in [the method](MULTILABEL_DMV_MAT_METHOD.md).

## Dataset, units and roles

PTB-XL1.0.3 eligible original folds1–9 only. No fold10 waveform, label array or
prediction is loaded. Demographic parsing selects only ECG ID, patient ID, age,
fold and100Hz filename from the original CSV; scp_codes is not selected/decoded.
CD/STTC labels come only from the immutable original train/validation metadata
caches. Hashing the publisher CSV as bytes is provenance verification, not
reading its test outcomes for model development.

Age18–89 eligible; lowest eligible ECG ID per actual patient. Source18–59,
target60–89. No age input is added to the network; no hospital/site is invented.
Patient/record/native-waveform identity isolation is checked; unresolved
cross-person identical signals stop preparation. Original exact physical-duplicate
audit reported zero; no new near-duplicate signal search is claimed.

Proposed source split85/15, seed33002. Target O/E-reserve/C-reserve/A fractions
30/10/10/50, seed33001. The implemented interpretation is rare joint-label-pattern
first, then seeded-hash ties, with equal-weight **squared relative deviations**
of group total and two positive/two negative counts from desired counts.
It does not include joint-cell counts as objective terms. This interpretation
is explicitly submitted for review; it is not an adopted registration.
Its observed joint-cell distortion is a blocker in the preflight report.
No alternative splitter, seed or membership was tried to repair the finding.

E/C128 by label-independent hash ordering, seeds33003/33005. Draws4101/4102
permutations of A: first256 G, next512 F reservoir; F128/512 nested. Distinct
active roles: source_train, source_stop, F, G, E, C, O. E handles checkpoint
selection; there is **no additional architecture-selection S search**. C cannot
select architecture, masks or checkpoints. O labels are available to the
isolated metadata auditor for counts/stratification, not predictive development.
Do not describe these audit labels as globally never inspected.

## Fresh source and training policy, if separately authorized

Exactly one new random source model, seed3101, two-head five-view architecture,
212,066 parameters. No old weights, heads, optimizer states or normalization.
Source pretraining and its checkpoint stopping use only their newly assigned
source patient roles. Fresh scalar population mean/std must be fit on source_train
physical mV signals only; no real preprocessing statistics were fitted here.
All pretrained target methods inherit that source normalization and checkpoint.
Target-only starts randomly, with its own F∪G normalization.

Source maximum50epochs, batch128, AdamW lr.01/wd.01/betas(.9,.99)/eps1e-8,
OneCycleLR max_lr.01/pct_start.3/div_factor25/final_div_factor10,000/
base_momentum.85/max_momentum.95/cosine/cycle_momentum=True/three_phase=False.
Pinned PyTorch signature was inspected; scratch epoch zero is diagnostic only.
Select exact scikit-learn macro AP, lower BCE tie, earliest checkpoint.

Target: batch64, AdamW.0003/wd.01/betas(.9,.99)/eps1e-8, constant LR,
40/160 maximum updates for F128/512; E looks each2/8 updates, patience5.
Step budgets match across each method pair; FLOPs/wall time need separate
measurement. No augmentation, alignment loss or hyperparameter search.
Pretrained epoch zero is eligible; target-only scratch epoch zero is diagnostic.
Implementation logs20 post-update looks plus the initial look; this resolves an
ambiguous shorthand in the prior draft and requires review before execution.
AP ties use the existing1e-12 numerical tolerance. CPU float32, four threads,
at most two active real fit attempts. MPS is not enabled.

## Matrix and label budget

One source fit plus eight methods×two draws×two F sizes = **33 planned fits**,
hard ceiling**36 including failures/retries/probes**. Source-only is a prediction
procedure on the one source checkpoint, not an extra fit.

| Method | Gradient pool | Selector supervision | Target union including E128/C128 |
|---|---|---|---:|
| Head-F |F|none|384 /768|
| Full-F |F|none|384 /768|
| Head-cost |F∪G|none|640 /1,024|
| **Full-cost** |**F∪G**|none|**640 /1,024**|
| Random-A |F|G-derived matched parameter cost|640 /1,024|
| Selective A |F|G|640 /1,024|
| Selective B |F|G|640 /1,024|
| Target-only |F∪G|none|640 /1,024|

Primary contrast remains A−Full-cost at F512, mean of paired draw-specific raw
macro AP. Source-only raw uses zero target labels; calibrated source-only uses
C128 and is not label-free. Source pretraining/selection labels and audit-level
label access are reported separately. Actual access receipts count distinct
record-label cells, not only gradient rows or repeated copies of the same patient.
Calibration artifacts augment consumed-label counts only after authorized C access.
Evaluation-label costs are separately recorded after sealing predictions.

## Execution boundary and lock

New ledger/manifest/hash identity in `private/phase2b_internal`, separate from
the closed historical study. Every operation resolves rows from a bound manifest,
checks patient/group/content hashes, active job and live code/dependency pins.
Role strings cannot authorize different data. Raw data reader verifies native
header/data checksums, physical units, lead order and12×1,000 shape before use.
Its real access paths are currently denied, so no real signals were loaded.

F/G are gradient-accessible only as their registered job specifies; source anchors
and G representations require a selector job. E/source_stop select checkpoints
only. SHARE parameters and buffers remain frozen across train calls; the head
is always trainable except source-only. Frozen state, optimizer ownership,
BatchNorm/dropout modes and source identity are verified synthetically.

C alone fits positive-slope Platt mappings and F1 operating thresholds; insufficient
support/convergence failures produce logged identity fallbacks. The implementation
uses threshold.5 on any calibration fallback, an explicitly proposed clarification
requiring review. No C mappings/thresholds were fitted on real records.

All33 outcomes/checkpoints/lineages and33 calibration-or-fallback artifacts must
be verified before development lock. Code/config/dependencies, memberships,
source history, preprocessing and checkpoint hashes are bound. A separate exact
evaluation authorization is then required. The lease is consumed before O access.
O prediction access redacts labels; **all registered predictions must be sealed
before O outcome/metric access**. No model selection or fitting after lock.
Use the failure-closing evaluation context; failures preserve the consumed lease
and require review, not automatic rearming.

The proposed10,000-draw patient-cluster paired analysis and five secondary
Bonferroni comparisons remain as the prior pilot draft; no real evaluation or
bootstrap was run. Confidence intervals would be conditional on the one fitted
source, overlapping target draws and reused internal population. No external
clinical validity, clinical utility or selective superiority is established.

## Adoption boundary

This document describes implemented guards and proposed science, not approval.
The current candidate manifest is **not accepted for training** because of
structural co-occurrence allocation. Review the splitter prospectively, preserve
this failed preflight, and separately authorize any revised metadata preparation.
Do not silently remove B, change labels, reseed or spend reserve fits to proceed.
