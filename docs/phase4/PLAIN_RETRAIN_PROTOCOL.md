# Amendment2 — plain-retrain control, validation only

Written before any amendment2 fitting or new validation scoring. Stay on `phase4c-confirm`; do not commit, push or merge. Never open CPSC test data/outputs, PTB-XL fold10, or sealed/reserved material. Use only the existing PTB folds1–9 isolated bundle, source-only feature cache, and adaptation-only bundle/fixed A/B references.

## Fixed lambda0 source control

Train seeds15101/15102/15103 with the same source architecture, initialization, source normalization, source-training order, 50-epoch AdamW/OneCycle schedule, source batch128, independent PTB second-stream batch32, and fold9 AP selection/BCE tie rule as the source/source placebo. The new parameterized numerical loop must reproduce the frozen placebo lambda0.1 loop on invented fixtures before execution.

**Lambda0 means BCE-only optimization.** Keep the second PTB training forward, independent sampling generator `seed([20261126,s])`, train-mode BN updates and dropout/RNG consumption. Compute the same four-kernel/seven-module MMD diagnostically under no-grad, with no MMD term in the optimization loss and no MMD gradient. This intentionally matches the placebo's forward-pass pipeline; removing the second forward would confound the requested placebo-minus-retrain gradient comparison. No CPSC input or target criterion enters source fitting or source checkpoint selection. Independent source streams may overlap by chance.

Keep the old aligned/placebo code and artifacts unchanged. Reuse verified source-only feature files read-only; do not rebuild them or use the old feature loader that opens CPSC U. Retain exactly three new source checkpoints privately with SHA-256 hashes, selected epochs, loss histories, source identities and fold9 metrics. Repeat one full source run without retaining another checkpoint; require identical model state, history and fold9 predictions before the remaining source runs.

Compare each retained checkpoint directly with its original Experiment4 source: distinguish file-byte identity from full tensor-state byte identity, trainable-parameter identity and BN-buffer identity. Verify original registered checkpoint hashes and source/normalization identities. Inspect only the allowlisted original source checkpoints and source descriptors; never an old run directory or test artifact. Describe pipeline differences from the original public real-source executor, including the added independent second forward, dropout RNG consumption, BN buffers, preprocessing/cache path and provenance/checkpoint packaging. Do not attribute every difference to one mechanism merely from final states.

## Target validation and fixed analysis

Run full FT on the same nine episodes at budgets64/128/256: exact F+G memberships, original source normalization, fit seed `seed([78501,s,d,b])`, batch32 replacement sampling, AdamW lr3e-4/weight_decay0.01/betas(0.9,0.99)/eps1e-8, exactly200 updates, all modules and head trainable. Preserve the adaptation-only runtime record-exclusion guard and existing A/B500/500 splits. Save only 27 A/B probability archives and fit records privately; no target checkpoints. Repeat one target fit and require byte-level A/B equality and identical state/history/recipe bindings.

The three fixed contrasts at each budget are: retrain+FT minus original full FT; placebo+FT minus retrain+FT; aligned(lambda0.1)+FT minus retrain+FT. Compute equal means of nine episode-wise macro AP and AUROC, not metrics from mean probabilities. Use10,000 common paired split-B record resamples, seed20261127 (the placebo amendment's frozen weights), pointwise percentile95% CIs with at least9500 supported replicates. Preserve CD/STTC separately and A/B metrics. These are reused validation findings, not confirmatory test claims. No hyperparameter, source, budget or threshold selection from B.

Append `Plain-retrain control` to PHASE4B_RESULTS.md while preserving its existing byte prefix, including the placebo section. Retain structured new aggregates, source comparison and integrity/replay references in separately named outputs. Fail closed on any integrity or replay mismatch; do not overwrite partial/completed outputs or run concurrent duplicate controllers.

## Draft only; no test authorization

Prepare PHASE4C_CONFIRMATORY_PROTOCOL_v2_DRAFT.md with original source-only/original full FT/retrain+FT/placebo+FT/aligned+FT. The author must choose P-reg (placebo−retrain macro AP128) or P-align (aligned−placebo macro AP128) and lock the final inferential specification before any test access. Preserve the bootstrap/Holm safeguards, strict stored-A/B replay authentication, prediction hashing before labels, single evaluation and prior Experiment4 cohort-use qualification. Do not finalize the draft, execute any confirmatory replay or access test data. Stop after amendment2 validation and drafting.
