# Phase 4A — validation-only oracle module-mask protocol

Status: PRESPECIFIED before any Phase 4A data processing/fitting; execution is now complete. See ORACLE_MASK_RESULTS.md.

## Question and immutable boundaries

Can any fixed freeze/adapt mask improve CD/STTC macro average precision over full fine-tuning on a separate validation evaluation split at the Experiment 4 128-label budget? This is a validation-only diagnostic of attainable selective-freezing performance, not a new test evaluation or clinical claim.

Amended by the owner's explicit Phase 4A instruction before any data processing or fits: `private/` is not prohibited as a folder. The bundle builder alone may read the original three Experiment 4 source checkpoints/normalization, episode memberships/seeds, recorded selector/random evidence, CPSC role assignments and adaptation-pool signals/V2 labels. Test-half IDs may be read solely as an exclusion list. Never load test-half signals, labels, predictions, metrics or derived evaluation files; never access PTB-XL fold 10, sealed or reserved payloads. A mixed metadata file must be field-projected to permitted fields; stop if this cannot isolate them. Never invoke old evaluation, seal-verification or source-training paths. After construction, all Phase 4 code reads only the new adaptation-only bundle. The test exclusion list is separate from the bundle (no test ID is stored in the bundle); a guard closes over that list during building and uses non-reversible per-ID exclusion hashes thereafter. Do not modify any existing experiment source, configuration, result, ledger or report. Do not commit, push, merge or open a PR.

## Masks and fixed training setup

Enumerate all 128 bit masks in module order `(view_1, view_2, view_3, view_4, view_5, fusion_1, fusion_2)`. A bit of one adapts that module; zero freezes it. `0000000` is head-only; `1111111` is full fine-tuning. The head is always trainable. Frozen modules retain both parameters and BatchNorm running statistics.

Reuse exactly the Experiment 4 three source models and three label draws per source model (nine episodes), original 128-label record lists, V2-decoded CD/STTC labels, source normalization, original initialization/sampling/training seeds and original update/loss implementation. AdamW, learning rate 3e-4 constant, batch 32 sampled with replacement, exactly 200 updates. No source training, hyperparameter tuning, 64-label fits or 256-label fits. Inspect public definitions to establish all unspecified optimizer/default details; do not substitute guesses. No newly fitted checkpoints may be written; the owner-authorized byte-identical copies of the three existing source checkpoints are the only exception.

## Validation design

From the 3,859-record adaptation pool only, exclude the union of every record in every original 64/128/256-label draw, all three draws and all source episodes. Sample 1,000 remaining records stratified by the joint CD × STTC label combination using validation seed 20261121. Split equally into A and B with joint-label stratification and split seed 20261122. Fix and record any integer stratum allocation by the deterministic stratified-split implementation before fitting. If support or remaining population is insufficient, stop rather than reduce size or change the design.

Save selected record lists and small per-fit A/B probability arrays only in the new gitignored `private/phase4_oracle_run/` location allowed by the amendment, separate from the bundle and all old experiment files. Save SHA-256 hashes of the lists and aggregate A/B counts/prevalences in this protocol directory. Never hash or load held-out waveforms, outcomes, predictions or evaluation files. Establish held-out separation by using role-assignment test IDs only as an exclusion list; no other test-half field is allowed. Never silently treat a membership assumption as an observed zero overlap.

## Reproducibility gate

Before the 1,152-fit run, repeat all-one training for one identical episode and require bitwise-identical A/B predictions. If original safe Experiment 4 artifacts contain final-parameter hashes, drifts or loss traces, reproduce their exact semantics and require a match. A mismatch stops the study. Only the allowlisted original source checkpoints, episode definitions and development-only completion reference fields may be read by the builder; no old ledger/evaluation material may be opened. Record unavailable evidence explicitly. If parallelism is used, verify three deterministically sampled mask/episode fits (seed 20261123) by rerunning with identical inputs and comparing A/B arrays; otherwise run serially. Never replace an original Experiment 4 seed with a newly chosen training seed.

## Primary estimand and inference

Choose one mask maximizing mean split-A macro AP over the nine episodes. Break exact ties by ascending binary mask index in the module order above, without looking at B. Calculate its split-B macro AP minus all-one split-B macro AP separately for each episode, then average the nine paired differences.

Use 10,000 paired bootstrap replicates (seed 20261124), sampling 500 split-B records with replacement. Use the same sampled record indices for both arms and all nine episodes; do not resample the nine episodes as if they were independent cohorts. Retain the A-selected mask throughout bootstrapping. Report the percentile 2.5th/97.5th limits and any undefined class-AP replicates. Stop inferential reporting if a class is absent from B or undefined replicates prevent the specified estimator. The interval is conditional on A selection and the fixed nine fitted episodes; it does not capture all source-training/adaptation uncertainty.

## Requested interpretation rules — fixed before execution

(a) CI entirely above +0.005: selective freezing CAN beat full fine-tuning; our selector failed to find good masks.

(b) CI includes 0, or the upper bound is below +0.005: no practically useful mask exists at this budget; full fine-tuning is the right default.

(c) CI entirely below 0: freezing anything hurts.

These requested rules overlap: a CI entirely below zero also satisfies (b). Report every matching rule and give (c) priority as the more specific numerical case; do not conceal the overlap. A positive interval crossing +0.005 but excluding zero satisfies none of the rules and will be reported as unclassified. The numerical rule labels are prespecified decision statements, not proof of universal claims: a CI including zero does not establish no useful mask exists, and a negative contrast for the A-selected mask does not prove that every non-full mask hurts. Full fine-tuning is itself in the candidate set. Comparisons do not establish failure of the original selector specifically unless its valid masks and evidence are independently available under these boundaries.

## Secondary and descriptive analysis

Report all-mask mean split-B AP distribution, markings for all-one/all-zero and safe original Experiment 4 selector/random masks; top ten by mean A AP with B macro and class-specific AP; A/B ranks of full and head-only; episode-specific A-selected oracle versus full on B; class-specific primary CD/STTC contrasts. Use the same prespecified A tie rule per episode. B-ranked displays and module effects are descriptive, never a new selection criterion.

Estimate split-B macro AP on seven mask bits plus episode fixed effects using all 128 masks × nine episodes. Report coefficients and record-bootstrap percentile intervals using common B-record resamples, not independent-row regression standard errors on reused participants. Interpret these as factorial average marginal effects across masks at this exact training setup, not causal clinical importance. Report selector ranks per episode when masks differ, rather than inventing a single selector mask.

## Outputs and preservation

Only new code and files. Aggregate results: docs/phase4/ORACLE_MASK_RESULTS.md; figures: figures/phase4/. Prediction arrays/record lists: gitignored. No checkpoints. Each fit must have immutable mask, episode, source/label/train-seed/config bindings, completion status and runtime; skip only verified compatible complete fits and never silently reuse incomplete or mismatched outputs. Maintain a new progress log. Record failures and total fits/runtime. Preserve all existing results/configuration/ledger/source behavior. Stop after Phase 4A validation-only analysis. No test or other label-budget run is authorized.

## Bundle amendment execution details (before processing)

Builder: `src/dmvmat_ecg/phase4/build_bundle.py`. New gitignored destination: `private/phase4_oracle_bundle/` (already covered by the original `.gitignore`). Source checkpoint copies are byte-identical copies of the three existing artifacts, not newly trained/saved model states. No target checkpoint is read or saved. The manifest records allowed-file/field identities, pool-only raw/header/cache and label hashes, source checkpoint checks, development-only reference projections, mapping/code fingerprints and zero test overlap. Test IDs are not included; only exclusion hashes are retained for the runtime guard. The builder must compare V2 labels and preprocessed waveforms against original development-only caches for every used draw record. Unknown labels remain unknown and cause a stop if needed support is unavailable. All runtime file opens during splitting/fitting/analysis must be audited to deny the old experiment directory, raw datasets, sealed/reserved paths and unrelated arrays; only the new bundle, Phase 4 output locations, public code and standard libraries are readable.

## Analysis presentation clarification (before aggregate analysis)

In addition to the top ten masks ranked by mean A macro AP, report separate CD-only and STTC-only top-ten A rankings and their B values. These label-specific rankings are descriptive secondary displays; the primary common mask is selected only by mean CD/STTC macro AP on A. Show actual recorded selector/random mask positions in the fixed-mask B distribution, and distinguish them from episode-varying policy means. This clarification does not change any fitting, seed, primary estimand or interpretation rule.
