# Phase4C amendment — source/source placebo, validation only

Written before new input preparation, source fitting or validation scoring. Branch `phase4c-confirm`; no commit/push/merge. No CPSC test membership, headers, signals, labels, predictions, metrics or old evaluation/seal material may be opened. PTB-XL fold10 and other reserved material remain excluded.

## Fixed source control

Train three fresh source models, seeds 15101/15102/15103, using exactly the Phase4B aligned architecture, two CD/STTC outputs, source folds1–8 (17,084) and fold9 (2,146), source normalization, 50 epochs, source batch128, original AdamW/OneCycle settings and fold9 AP selection with BCE ties. Lambda is fixed at 0.1; no search or model selection from CPSC validation results.

The only loss-input change is the second stream: replace the unlabelled CPSC batch32 with a **PTB-XL training batch32**, sampled by a separate seeded generator. Keep the seven-module differentiable MK-MMD, detached median bandwidth estimation, biased four-kernel statistic and multipliers {0.5,1,2,4} unchanged. First source stream remains the original once-per-epoch shuffle. Second stream uses the original independent namespace `seed([20261126, source_seed])`, cycles independent PTB training permutations with tails retained and uses no second-stream labels. Independent streams may overlap by chance; do not force disjoint batches or condition on class. Keep source-first, second-stream-second train-mode forwards, including dropout and BN updates. Source batch128 versus second batch32 preserves the aligned-source computation shape.

Pretraining must be genuinely source-only: do not instantiate the CPSC Target, load its bundle/labels/IDs/U cache, or use the old Features loader that opens U. Build source-only normalization/features from the verified PTB-only source bundle, using the original normalization algorithm and input-view functions. A source-only OS-open guard permits only that bundle and new placebo storage among project data/private paths. Source fold metadata is checked before signals. No CPSC data or target validation criterion enters source training/checkpoint selection. Report all three fold9 AP/AUROC values and selected epochs without replacing a weak source.

Run one complete deterministic source-training repeat (no extra saved checkpoint) before the other two source fits. Require exact state/history/fold9 prediction agreement. Retain exactly three unique placebo source checkpoints with SHA-256, source-role/normalization/code bindings and fold9 summaries, all ignored/private. Source/target numerical modules from Phase4B are not edited.

## Fixed target validation

After source checkpoints are complete, full-fine-tune each placebo model in the same nine source/draw episodes, budgets64/128/256, exact F+G memberships, same source normalization, target seed `seed([78501,s,d,b])`, batch32 replacement sampler, AdamW lr3e-4 constant, weight_decay0.01, betas(0.9,0.99), eps1e-8, 200 updates and all modules/head trainable. Use the identical Phase4A A/B splits and runtime test-ID exclusion guard. No test data/provider is available. A/B are validation data; their previous inspection is acknowledged.

Store 27 validation A/B probability archives and completed fit records, including source file/state hashes, final state hash, losses, drift and input bindings, in a fresh ignored private namespace. Save no adapted target checkpoints. Repeat one target fit before the full grid and require bitwise A/B/state/history equality. Source-only numerical reference and full-FT engine parity are tested with invented fixtures before real fitting. One CLI controller and an OS stage mutex prevent duplicate launches; do not delete partial outputs to force resume.

## Contrasts fixed before placebo outcomes

At each budget, calculate the mean of nine episode-wise CD/STTC macro AP values, never AP of mean probabilities. Two contrasts are fixed: aligned(lambda0.1)+full minus placebo+full; placebo+full minus original-source full. Report their paired percentile95% CIs using 10,000 common split-B record resamples, seed20261127, as in Phase4B. Reuse authentic stored Phase4B prediction arrays after hash/input verification; do not rerun their inference. Also report macro AUROC, per-label metrics and supported resample counts. No threshold or hyperparameter choice from B. These six macro-AP comparisons are pointwise descriptive validation analyses, not the future confirmatory Holm family.

Append one clearly marked **Placebo-alignment control** section to PHASE4B_RESULTS.md, preserving its previous text/results. New structured aggregates and validation records must be separately named. Any difference measures the second-stream domain substitution under this matched recipe, including domain-dependent BN effects; it is not proof that MK-MMD alone is the causal mechanism. Keep the Phase4C roster/contrast definitions fixed even if this control is unfavorable.

## AdaBN and future replay amendment

Remove AdaBN+full as the planned Phase4C control because its reported AP/AUROC coincide with ordinary full at all three budgets after 200 updates, with identical loss histories/drift norms. Be precise: only 10/27 A/B vector pairs were bitwise-identical; BN buffers/continuous probabilities are not proved universally equal. Running statistics are largely overwritten during FT. The placebo replaces it without relabeling or deleting old results.

Future Phase4C M2/M3/M4 target states may be regenerated only as explicitly authorized deterministic authentication replays, after the amended protocol is committed and the author supplies the continuation phrase. Each must match its stored A/B probabilities **bitwise**, final state hash, input/normalization/source bindings and original recipe before **any test prediction** (and, conservatively, before loading test signals). M4 references are the new 27 placebo validation records; M2/M3 references remain immutable Phase4B records. Any mismatch stops; no repair/tuning against test outcomes. Replays perform the fixed optimization to reconstruct states; they are not described as inference or new training experiments. No such confirmation replay or test evaluation is run in this amendment task.

Stop after validation reporting and amending PHASE4C_CONFIRMATORY_PROTOCOL.md. The primary/secondary estimands, 95-secondary Holm family, prediction-before-label seal, known-group bootstrap and prior Experiment4 test-use qualification remain unchanged except that method4 now means placebo+full.
