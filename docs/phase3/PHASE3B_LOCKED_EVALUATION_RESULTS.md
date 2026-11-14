# Phase3B one-time locked CPSC O evaluation

## Executive finding

Primary A−Full-cost at B128: Δ mean raw AP **NA**;
95% paired record-group interval **not estimable**.
Mean paired ΔAUROC: **NA** (descriptive).
Primary uncertainty is not estimable under the valid-replicate rule.
Evidence-guided selection beyond ordinary freezing requires A also to beat
Head-cost and Random-A. Both Bonferroni98.75% lower bounds exceed zero:
**False**. Beating Full-cost alone is a policy-level result.
No model, mask, checkpoint, threshold or seed was changed after O access.

## Population and pre-outcome signal audit

Frozen O membership3,860 records; evaluated **3860**;
short records excluded **0**; first10s truncation
**2436**. All exclusions were determined from the
physical WFDB header prefix before any score/outcome access. No padding or
windows. Inputs12 canonical leads,mV,100Hz; source or target-only frozen
normalization as registered. Diagnosis comments were not read during signal
metadata preparation; only the first13 unbuffered header lines were parsed.
Every MAT signal passed its publisher checksum. Full header checksums and
diagnoses were read only inside the single post-seal outcome access.

CD-proxy observed prevalence/random AP baseline **NA**;
STTC-proxy **NA**. An unlisted code means not annotated,
not clinically absent. Unknown labels remain masked; observed counts and
unknowns appear in every per-label metric record.

## Complete mean episode-wise performance

Source-only means three source models, reused across budgets/draws. Other
rows average nine source×draw episodes. These are means of metrics, not
probability ensembles, pooled AP or an outer-selected winning model.

|B|Procedure|Mean AP|Mean AUROC|Mean Brier|Mean log loss|Mean ECE15|
|---:|---|---:|---:|---:|---:|---:|
|64|source_only|NA|NA|NA|NA|NA|
|64|head_cost|NA|NA|NA|NA|NA|
|64|full_cost|NA|NA|NA|NA|NA|
|64|random_a|NA|NA|NA|NA|NA|
|64|selective_a|NA|NA|NA|NA|NA|
|64|target_only_cost|NA|NA|NA|NA|NA|
|128|source_only|NA|NA|NA|NA|NA|
|128|head_cost|NA|NA|NA|NA|NA|
|128|full_cost|NA|NA|NA|NA|NA|
|128|random_a|NA|NA|NA|NA|NA|
|128|selective_a|NA|NA|NA|NA|NA|
|128|target_only_cost|NA|NA|NA|NA|NA|
|256|source_only|NA|NA|NA|NA|NA|
|256|head_cost|NA|NA|NA|NA|NA|
|256|full_cost|NA|NA|NA|NA|NA|
|256|random_a|NA|NA|NA|NA|NA|
|256|selective_a|NA|NA|NA|NA|NA|
|256|target_only_cost|NA|NA|NA|NA|NA|

|B|Procedure|Label|AP|AUROC|Prevalence/AP baseline|Brier|Log loss|ECE15|Precision|Recall|Specificity|F1|Balanced accuracy|
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|64|source_only|CD-proxy|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|
|64|source_only|STTC-proxy|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|
|64|head_cost|CD-proxy|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|
|64|head_cost|STTC-proxy|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|
|64|full_cost|CD-proxy|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|
|64|full_cost|STTC-proxy|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|
|64|random_a|CD-proxy|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|
|64|random_a|STTC-proxy|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|
|64|selective_a|CD-proxy|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|
|64|selective_a|STTC-proxy|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|
|64|target_only_cost|CD-proxy|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|
|64|target_only_cost|STTC-proxy|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|
|128|source_only|CD-proxy|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|
|128|source_only|STTC-proxy|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|
|128|head_cost|CD-proxy|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|
|128|head_cost|STTC-proxy|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|
|128|full_cost|CD-proxy|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|
|128|full_cost|STTC-proxy|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|
|128|random_a|CD-proxy|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|
|128|random_a|STTC-proxy|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|
|128|selective_a|CD-proxy|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|
|128|selective_a|STTC-proxy|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|
|128|target_only_cost|CD-proxy|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|
|128|target_only_cost|STTC-proxy|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|
|256|source_only|CD-proxy|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|
|256|source_only|STTC-proxy|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|
|256|head_cost|CD-proxy|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|
|256|head_cost|STTC-proxy|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|
|256|full_cost|CD-proxy|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|
|256|full_cost|STTC-proxy|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|
|256|random_a|CD-proxy|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|
|256|random_a|STTC-proxy|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|
|256|selective_a|CD-proxy|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|
|256|selective_a|STTC-proxy|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|
|256|target_only_cost|CD-proxy|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|
|256|target_only_cost|STTC-proxy|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|

All138 logical procedures, per-label values, unknown counts and confusion
counts are in PER_EPISODE_METRICS.json. No exclusions of abstaining episodes
or masks equal to controls. Precision etc use0.5; undefined denominators are
NA. Probabilities are raw, uncalibrated frozen sigmoid outputs. Only log loss
clips to[1e-7,1−1e-7]. AP/AUROC/Brier/ECE inputs are not clipped. ECE uses15
fixed equal-width bins; these are diagnostic statistics, not fitted calibrators.

## Primary paired episodes

|Source|Draw|ΔAP A−Full-cost|ΔAUROC|
|---:|---:|---:|---:|
|15101|16101|NA|NA|
|15101|16102|NA|NA|
|15101|16103|NA|NA|
|15102|16101|NA|NA|
|15102|16102|NA|NA|
|15102|16103|NA|NA|
|15103|16101|NA|NA|
|15103|16102|NA|NA|
|15103|16103|NA|NA|

Primary is the equal mean of all nine differences at B128. Budget64 and256
do not replace it. Practical AP margin+0.01 is an engineering criterion, not
a clinical threshold; distinguish a point estimate≥0.01 from a CI lower bound>0.01.

## Registered comparison family

|Contrast|Δ mean AP|Δ mean AUROC|95% AP interval|Bonferroni98.75% AP interval|
|---|---:|---:|---|---|
|B64/A-minus-full_cost|NA|NA|not estimable|not estimable|
|B64/A-minus-head_cost|NA|NA|not estimable|not estimable|
|B64/A-minus-random_a|NA|NA|not estimable|not estimable|
|B64/A-minus-source_only|NA|NA|not estimable|not estimable|
|B64/A-minus-target_only_cost|NA|NA|not estimable|not estimable|
|B128/A-minus-full_cost|NA|NA|not estimable|not estimable|
|B128/A-minus-head_cost|NA|NA|not estimable|not estimable|
|B128/A-minus-random_a|NA|NA|not estimable|not estimable|
|B128/A-minus-source_only|NA|NA|not estimable|not estimable|
|B128/A-minus-target_only_cost|NA|NA|not estimable|not estimable|
|B256/A-minus-full_cost|NA|NA|not estimable|not estimable|
|B256/A-minus-head_cost|NA|NA|not estimable|not estimable|
|B256/A-minus-random_a|NA|NA|not estimable|not estimable|
|B256/A-minus-source_only|NA|NA|not estimable|not estimable|
|B256/A-minus-target_only_cost|NA|NA|not estimable|not estimable|

Only B128 A−Full-cost is primary. The four B128 secondary AP comparisons
form the registered Bonferroni family. Other budgets are exploratory; intervals
there do not establish simultaneous control across the entire result package.

Bootstrap10,000 draws,seed78701; shared record/known-duplicate-group weights
across all labels,methods,seeds,draws and budgets. Whole CD/STTC vectors are
preserved. Primary valid draws **0**, undefined
**10000**; no redraws. No primary interval when fewer
than9,500 replicates are valid. Training seeds/draws are not independent patients.
Intervals are conditional on the fitted models and realized design; they do not
capture model/selector refitting uncertainty, target selection from FEAS,
unknown within-patient correlation, source checkpoint selection or label error.

## Source transfer gap (fold9 reference versus O Source-only)

|Seed|Label|Fold9 AP|Fold9 AUROC|Fold9 AP baseline|O AP|O AUROC|O AP baseline|O−fold9 AUROC|
|---:|---|---:|---:|---:|---:|---:|---:|---:|
|15101|CD-proxy|0.84326|0.93455|0.22088|NA|NA|NA|NA|
|15101|STTC-proxy|0.57729|0.87646|0.17102|NA|NA|NA|NA|
|15102|CD-proxy|0.83572|0.93293|0.22088|NA|NA|NA|NA|
|15102|STTC-proxy|0.56187|0.87770|0.17102|NA|NA|NA|NA|
|15103|CD-proxy|0.83573|0.92428|0.22088|NA|NA|NA|NA|
|15103|STTC-proxy|0.55902|0.87130|0.17102|NA|NA|NA|NA|

Fold9 selected source checkpoints and is an optimistic development reference.
Source and target populations/prevalences and annotation practices differ.
AP changes cannot be interpreted as pure hospital signal-shift effects.

## Descriptive recovery contrasts fixed before O access

|Contrast|Mean ΔAP|Mean ΔAUROC|
|---|---:|---:|
|B64/head_cost-minus-source_only|NA|NA|
|B64/full_cost-minus-source_only|NA|NA|
|B128/head_cost-minus-source_only|NA|NA|
|B128/full_cost-minus-source_only|NA|NA|
|B256/head_cost-minus-source_only|NA|NA|
|B256/full_cost-minus-source_only|NA|NA|

These show how much adaptation changes ranking relative to Source-only;
they are not additional primary hypotheses or clinical recovery estimates.

## Negative transfer

|Procedure|AP worse than paired Source-only|Mean ΔAP|
|---|---:|---:|
|B64/head_cost|None/9|NA|
|B64/full_cost|None/9|NA|
|B64/random_a|None/9|NA|
|B64/selective_a|None/9|NA|
|B128/head_cost|None/9|NA|
|B128/full_cost|None/9|NA|
|B128/random_a|None/9|NA|
|B128/selective_a|None/9|NA|
|B256/head_cost|None/9|NA|
|B256/full_cost|None/9|NA|
|B256/random_a|None/9|NA|
|B256/selective_a|None/9|NA|

Counts are paired fitted episodes on the same O population, not independent
clinical events. Equality from aliases stays included. Increased ADAPT counts,
valid geometry, MMD significance and mask stability do not by themselves show
predictive benefit. Module masks/drift and label costs remain the frozen
development findings; all target arms received the same F∪G labels and200 updates.

## Execution integrity and limits

Human authorization was transcribed under private/phase3b_cpsc_v1/. Locked
study hash30a0c75ab9a159773bc87e3941ddfdee91c1860482d08e20a7b87ae601d25073.
Complete138-procedure prediction seal **b7c7e795b9641979858c5050bd6f7db9c2f1340a1f466b4828a0e4088eec5cde** preceded all O outcome
access. Aliases resolve to shared artifacts without disappearing from the roster.
Unique inference runs **121**; summed forward time
**439.67s**, excluding preparation/load.
Predictor-fit ledger remains121/144; no training,fine-tuning,recalibration,
threshold selection,mask/seed/checkpoint changes or reserve use. Post-evaluation
tests and the final integrity receipt document re-entry refusal and unchanged
frozen artifacts. No Phase2 artifact,PTBXL fold10 or CPSC FEAS data were accessed.

This is **CPSC-only exploratory annotation-proxy evidence at record level**.
Target patient identifiers are unknown; exact native deduplication is not proof
of patient independence. PTBXL-versus-target waveform overlap was not audited
by explicit prior waiver. Phase3A FEAS outcomes chose the target and informed
the protocol; RESERVED annotations were previously used for stratification.
This history precludes a pristine independent confirmation claim. Labels are
recorded-code unions, not clinically adjudicated positives/negatives; label
definition/reporting mismatch can contribute to the apparent hospital gap.
First10s truncation can omit findings described by full-record annotations.
No external clinical validation, clinical utility, causal transfer mechanism
or future cardiovascular-event prediction is established. A CI crossing zero
is inconclusive, not proof of no effect. No further experiment is executed here.
