# Phase 2D — human protocol adoption review package

**UNSIGNED — NOT ADOPTED. No real training/evaluation permission.**

This package recommends a bounded **internal exploratory selective-transfer**
pilot. It cannot sign for a user, professor, principal investigator, supervisor
or clinical reviewer. All decisions in
[HUMAN_REVIEW_UNSIGNED.json](../../configs/phase2d/HUMAN_REVIEW_UNSIGNED.json)
are `approved=null`, with reviewer/role/date/notes null. Training and evaluation
flags are false. A readiness result does not fill or accept this record.

## A–L scientific review

| Item | Evidence/recommendation | Human decision required |
|---|---|---|
|A. Original split failure|[Root cause](SPLIT_FAILURE_ROOT_CAUSE.md): rare11-first sorting plus marginal relative-error greed favored small roles, creating19/142 source joint split and zero in target A|Accept within-domain allocation correction, distinct from real age-domain prevalence difference|
|B. Seed44301|One [prospective specification](../../configs/phase2c/joint_pattern_v1.json), written before candidate generation; no split search or predictive feedback|Accept this seed/algorithm version; no result-based alternative seeds|
|C. Distributions|Source train6,466→6,492, stop1,172→1,146; target A4,208→4,388, O2,668→2,633; joint positives137/24/293/176|Review complete tables in [feasibility](JOINT_PATTERN_FEASIBILITY.md)|
|D. Domains|Same canonical16,415 patients; source18–59, target60–89; recorded age only, no invented sites|Accept internal age-shift interpretation, not independent cohorts|
|E. Allocation|Within-domain observed joint-stratum Hamilton quotas, each cell/role within one patient of natural fractional target|Accept approximate integer role sizes and label-aware stratification; no G enrichment|
|F. Label audit/inference|Old and replacement O labels were audited for feasibility; O has790 shared patients; no pilot predictions exist|Accept that replacement O is internal exploratory, not independent confirmation|
|G. Canonical semantics|[Serial audit](SERIAL_RECORD_ANNOTATION_AUDIT.md): exact-record labels;625 groups/1,593 records; no OR/adjudication; minimum eligible ID remains key|Accept annotation-presence task and serial/likelihood0 limitations, or request a new amendment|
|H. A/B validity|A≥10/class/domain, B≥10/four cells/domain, source D≥.1, finite scatter and bootstrap validity; invalid defaults SHARE|Accept abstention as an outcome, not a reason to enrich/reselect|
|I. Matched labels|Primary A and Full-cost both640/1,024 target participants including G/E/C; Full-cost gradients F∪G|Accept union-based accounting plus separate source, audit and evaluation label costs|
|J. Budget|One fresh source +8 methods×2 draws×2 sizes=33; technical reserve3; hard36 including failures|Accept no hidden predictor smoke fits, favorable retries or expansion|
|K. Access/schedules|Folds1–9 only; source stop isolated; E checkpoint selection; C calibration; separate O lock; pretrained zero eligible|Accept20 post-update stopping looks plus initial zero, fixed schedules and calibration identity/.5 fallback on failure|
|L. Interpretation|Primary mean paired-draw raw macro AP(A−Full-cost) at F512; full/head/random/source controls; fixed conditional paired bootstrap|Accept internal/conditional inference, negative results and no clinical deployment claim|

These are review recommendations. There is no adopted protocol merely because
a JSON configuration or synthetic run exists. The older external20/172-fit
plans remain separate and blocked; no P role or external source conversion is
part of this33-fit internal matrix.

## Explicit review record and separate authorizations

Proposed execution contract semantic hash:
`dbc0afe9bee29e7323a3a353b37660b8dff20733e3fe62fa94958d95400ecb28`.
Exact candidate auditor hash:
`c809d6db691d7ad3094dcd3635546ac58941529d1c6a3d06be35129f8ca0f4c9`.
Canonical policy: `lowest-age-eligible-ecg-id-exact-record-annotations-v1`.

Copy the blank template to a separate private review record; **do not edit the
pinned unsigned template in place**. For each listed decision, an actual reviewer
must supply approval/rejection, real identity/role/date and rationale. The
protocol owner/PI (or explicitly delegated responsible researcher) must approve
the complete protocol. Clinical/methodology review should explicitly accept the
annotation-versus-clinical-truth distinction and canonical policy. A professor
or clinical expert's approval must be recorded only if actually obtained;
Python cannot authenticate a signature or manufacture authority. A rejected or
unresolved item is not adopted. If clinical adjudication is required, stop for
its separate policy rather than labeling this review complete.

The ledger verifies a complete attributable decision record and binds its hash;
this attests operator-supplied authority, not cryptographic identity verification.
`record_adoption()` leaves phase draft and grants **no fit permission**. An exact
separate development authorization bound to code/dependency/manifest/contract
identities is required to enter development. Evaluation requires another exact
one-time grant after all development artifacts are locked. None was recorded
for the real study in this sprint.

## Proposed fixed training/analysis settings requiring adoption

Source: five-view two-head212,066 parameters, fresh seed3101, source_train-only
scalar mean/std, source_stop exact macro AP selection. AdamW lr.01/wd.01/
betas(.9,.99)/eps1e-8, batch128, maximum50 epochs; explicit OneCycle settings
in [execution configuration](../../configs/phase2d/amended_execution_draft.json).
Scratch epoch zero diagnostic only. No historical weights, head, optimizer or
normalization can initialize this source or target-only control.

Target: AdamW.0003/wd.01/betas(.9,.99)/eps1e-8, batch64,40/160 update ceilings
for F128/512, E looks each2/8 updates, patience5. Exact macro AP, tie lower BCE
then earlier; pretrained epoch zero eligible, target-only scratch zero diagnostic.
This explicitly retains the Phase2B proposed interpretation:20 post-update looks
plus initial diagnostic/eligible zero. It is submitted for review, not silently
adopted. SHARE parameters and BN buffers/dropout remain frozen; head updates.

C-only positive-slope Platt and C-F1 thresholds on.05–.95 step.01; insufficient
support/convergence failure yields identity and threshold.5. C cannot choose
architectures, masks or checkpoints. Source-only raw has no target supervision;
its C-calibrated variant consumes C128. Equal updates do not imply equal FLOPs.

Primary: raw two-label macro AP(A−Full-cost), F512, mean of two paired draw
metrics. Patient-cluster10,000 bootstrap draws seed64001 preserve full label
vectors, masks and paired predictions across methods/draws;95% primary interval.
Five secondary AP contrasts A−Head-cost/A−Random/A−Full-F/B−A/A−raw source-only
use95% and Bonferroni99% intervals. No pooled-OOF substitution, best-draw selection
or post-test tuning. Intervals condition on this split/source/model choices and
omit design-amendment, retraining and external-population uncertainty.

## What later results would mean

A positive point estimate is not superiority. A positive conditional interval
and stable comparison against Full-cost/head/random would support this particular
internal adaptation procedure;+.005 AP is an engineering relevance signal, not
medical utility or a guaranteed outcome. Negative/inconclusive effects show no
established advantage under these conditions. All-SHARE/epoch-zero retention or
B abstention is a legitimate result; it does not justify new thresholds, masks,
sampling or favorable reruns. Greater mask diversity alone is not predictive
success. Source-only strength and negative-transfer frequency must be reported.
No outcome here can clear external clinical label/person/overlap gates.
