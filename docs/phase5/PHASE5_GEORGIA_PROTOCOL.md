# Phase5 — Georgia pre-registered test protocol

Status: fixed after Phase5 development; awaiting author commit and explicit continuation. **STOP: no test access is authorized in this task.**

## Question and fixed sources

Does placebo within-source MK-MMD pretraining improve Georgia CD/STTC macro AP over the matched lambda=0 plain-retrain pipeline after full FT? Use only the20 existing hash-verified Phase5 sources:10 per arm, seeds15101–15110. Both arms used PTB-XL only; no target-domain source pretraining. No source retraining/replacement, lambda search, hyperparameter changes, calibration, threshold selection or tuning is permitted.

Methods: **(a)** plain-retrain source-only, zero target labels; **(b)** plain retrain +full FT; **(c)** placebo lambda0.1 +full FT. Budgets64/128/256; draw seeds16101/16102/16103;30 seed×draw episodes at each budget. Reuse exact recorded Georgia nested F/G memberships. Source-only has10 physical predictors, aliased across budgets/draws. Full FT uses200 updates, replacement batch32, constant AdamW lr3e-4, weight decay0.01, betas(0.9,0.99), eps1e-8, seed([78501,source_seed,draw_seed,budget]), all modules/head trainable, source-only normalization and frozen source settings. Source pretraining remains the matched two-forward pipeline,50 epochs, source128/independent PTB32 batches, seven-module biased MK-MMD, bandwidth multipliers0.5/1/2/4, and existing fold9 AP/BCE checkpoint selection.

## Test population and label protection

Locked Georgia test: **4081 canonical records**, fixed by the ID-only assignment. The original2040-record20% survey subset is wholly in the6120-record adaptation pool and never test. The remaining records were allocated50/50 **label-blind** using fixed seed79101 and deterministic SHA256 group ordering. Known groups remain together. Label-blind allocation was adopted explicitly to keep test labels untouched during this pre-registration phase. Test prevalence is deliberately unknown and will be reported after scoring, not estimated from the pool.

Preserve the Experiment3 native-duplicate canonicalization, canonical lead order, physical mV conversion, first10s100Hz12x1000 processing and short-record exclusions. Do not introduce post-outcome exclusions or a new duplicate search. Stable patient linkage is unknown; retained canonical record IDs are the existing known singleton groups. This is annotation-proxy domain replication, not clinical validation or proven person-independent sampling. Historical Phase3 audits/20–80 stratification used headers, but only the old survey was scored; the future test was not used for model scoring in this task. No claim of never-decoded historical outcomes is made.

## Single primary

**(c) minus(b), mean episode-wise CD/STTC macro AP at128 labels**, equally over10 source seeds and three fixed draws per seed. Never compute AP of mean scores or pool repeated participants into a larger cohort. Use10000 paired hierarchical bootstrap replicates, fixed seed20261201: first multinomial-resample10 paired source seeds; then one shared canonical-record/known-group resample across all selected seeds, paired arms, draws and labels. Preserve shared-cohort dependence. Use the validation implementation's exact tie-aware ranking calculation, strict means, percentile2.5/97.5 bounds, minimum9500 supported finite draws, unsupported counts reported and no redrawing/imputation. **Success requires the two-sided95% CI lower bound strictly greater than zero.** Undefined/insufficient-support results are invalid and cannot establish success. This includes source-seed variability but omits training-cohort, label-draw-selection and unknown patient-linkage uncertainty.

## Descriptive only

All64/256 results, macro AUROC, CD/STTC AP/AUROC, fixed0.5 accuracy/precision/recall/specificity/F1/balanced accuracy/NPV and per-seed differences are descriptive only. Source-only is a zero-target-update reference. No larger hypothesis family, test-set threshold optimization or post-hoc primary replacement. Threshold decision: probability>=0.5. Undefined denominators remain N/A; unknown labels never become negatives. CD/STTC order and the frozen V2 annotation-union decoder/mapping stay fixed; require>=95% observed per label and both classes before metrics, reporting observed/unknown and class denominators. Failed validity gates stop without changing rules.

## Replay, prediction freeze and one evaluation

Only after this final protocol is committed and the author explicitly says continue: verify exact committed protocol bytes and fresh unused private output. Verify all20 existing source checkpoint/state/normalization hashes, the Georgia bundle/assignment/draw/split hashes and frozen numerical dependencies. Regenerate all180 full-FT states by deterministic replay against their stored A AND B validation arrays **BITWISE**, including shape/dtype/order, final state hash,200-update history, recipe and source bindings. Any mismatch stops before test signals; no tolerance or setting adjustment is permitted. Authentication optimization only, no new target training experiment and no target checkpoints.

Training/replay reads only the adaptation-only bundle under the test-exclusion guard. Only after all180 replay matches may a separate final-prediction capability load test SIGNALS ONLY. Use CPUfloat32/four threads, eval/no_grad,batch128 and unchanged preprocessing/normalization; verify state/BN hashes before/after. Read only unbuffered physical header prefixes and signal-only MAT values; no full annotation/header checksum before the seal. Generate all270 logical method×budget×episode arrays, with190 physical predictors (180 adapted states+10 source-only aliases), shape(test_records,2), finite[0,1]. Save privately and SHA256-hash the complete roster, source/state/code/normalization/row-order/group identities and timestamp before one immutable prediction seal. No labels unlock on a partial roster.

Only then consume one outcome-access operation and decode labels with frozen V2 semantics. Score exactly the primary and listed descriptive outputs using shared seed/record resamples. Single evaluation: no automatic retry/reset, prediction regeneration after labels, tuning, metric switching or analysis changes after results. Preserve any failure and report incomplete/invalid. Never load CPSC test data, PTB-XL fold10 or unrelated sealed material. No reuse of Phase4C authorizations/receipts. Raw inputs, IDs, predictions, model weights and operational manifests remain private; no commit/push/merge by the agent.

## Fixed development evidence

The evidence below motivated this test but is not its result. No favorable source seed/lambda/budget was selected; all10 paired seeds remain in the test estimand.

All14 new source fits completed alongside the six preserved historical sources. Mean PTB-XL fold9 macro AUROC across10 sources per arm: plain retrain **0.902664**, placebo **0.905465**. The full per-seed/arm table and checkpoint-selection epochs are retained in the development report; these source metrics did not select or exclude seeds.

| Validation domain | Budget | Placebo − plain macro AP | Hierarchical95% CI |
|---|---:|---:|---|
| CPSC | 64 | +0.024793 | [+0.005832, +0.043429] |
| CPSC | 128 | +0.026575 | [+0.009361, +0.042475] |
| CPSC | 256 | +0.023419 | [+0.006481, +0.039447] |
| Georgia | 64 | +0.019469 | [-0.004297, +0.042703] |
| Georgia | 128 | +0.023946 | [-0.001419, +0.048311] |
| Georgia | 256 | +0.031272 | [+0.007667, +0.054146] |

Development report SHA256: `caf6a0b8a240f4124bde90c6edf86dc42bf5b4ae531cf6a2f81410556f93cbb5`. Source identities, validation fitted-state hashes, A/B prediction hashes and record bindings are frozen in the new private release binding. Validation plan: [PHASE5_VALIDATION_PLAN.md](PHASE5_VALIDATION_PLAN.md). Full evidence: [PHASE5_VALIDATION_RESULTS.md](PHASE5_VALIDATION_RESULTS.md).

Stop for author commitment and explicit continuation. No Georgia test signal, label, prediction, prevalence or metric was loaded/computed to write this protocol.
