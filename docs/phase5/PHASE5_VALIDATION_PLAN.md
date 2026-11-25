# Phase 5 — prospective validation replication plan

Written before new fitting. Parent: Phase4C commit `53276002997f67d2bed149625335396a668e13eb`; branch `phase5-replication`. No commit, push, merge or test evaluation is authorized in this task.

## Fixed experiment

Compare plain retrain (lambda=0) with within-source placebo MK-MMD (lambda=0.1). Retain the exact Phase4C new two-forward pipeline: architecture, all seven representation terms, Gaussian bandwidth multipliers 0.5/1/2/4, source-only normalization, PTB folds1–8 training/fold9 checkpoint selection, 50 epochs, source batch128, independent PTB second batch32, original OneCycle/AdamW recipe and AP/BCE checkpoint rule. Neither source arm receives target inputs. Reuse seeds15101–15103 unchanged; add15104–15110 in both arms without seed selection. No tuning or source replacement based on target results.

Use draw seeds16101/16102/16103, budgets64/128/256 and full FT with all seven modules/head trainable, 200 updates, batch32 replacement, constant AdamW lr3e-4, weight decay0.01, betas(0.9,0.99), eps1e-8 and fit seed seed([78501,source_seed,draw_seed,budget]). Store source checkpoints only; target states are not saved. Validation A/B predictions and all record memberships stay gitignored. Completed immutable fits may resume only after source, numerical-code, membership and prediction hashes verify; partial outputs are preserved and refused.

## CPSC validation

Reuse the existing adaptation-only bundle, label-blind nested F/G draws and A/B500/500 splits. The old three-seed validation fits are reused only after integrity/source/recipe checks. The 14 new sources produce126 additional CPSC fits, yielding180 per-hospital arm/seed/draw/budget fits. Never read CPSC test signals, labels, predictions, outcomes, manifests or results, including the Phase4C test operation.

## Paired seed-and-record bootstrap

Primary validation summary is placebo minus plain-retrain mean macro AP on B at128 labels, with descriptive counterparts at64/256 and AUROC/per-label results. Point estimate: equal mean of10 paired source-seed effects, each the equal mean of three paired draw effects and two observed labels. Never use AP of an averaged probability vector.

Use10000 replicates, fixed seed `20261201`. First multinomial-resample10 source seeds with replacement, retaining paired arms and all three label draws. Then draw one paired validation-record/known-group resample per replicate, shared across all selected source seeds, draws, arms, labels and budgets. This preserves dependence from evaluating every model on the same people; independently resampling a different cohort inside each selected seed would erase that shared-cohort dependence. Compute tie-aware ranking metrics, strict equal-label/draw means, and the seed-count-weighted mean. This is a two-level paired source-seed/shared-record bootstrap, not independent test cohorts or cross-validation. Percentile2.5/97.5% intervals; require9500 supported draws, retain unsupported counts, no redraw/imputation. Do not resample/select the three fixed label draws or tune on B. Source seeds include historical and new fits; uncertainty remains conditional on this source cohort and checkpoint-selection pipeline.

Per-seed plots show the equal-three-draw B macro-AP differences at each fixed budget, with a zero reference and no fabricated per-seed confidence intervals. All current A/B analyses are development evidence, not Georgia test confirmation.

## Georgia access gate and fixed intended design

Georgia denotes the dataset/domain, not independently verified single-hospital sampling. Preserve Experiment3 native canonical duplicate handling, canonical leads/mV, first10s100Hz12x1000 processing and short-record exclusions; retain CD/STTC V2 annotation-union proxy definitions and unknowns. The old20% FEAS survey records must enter the adaptation pool, never test. Per the author's explicit correction, allocate the remaining eligible canonical record/known-group IDs50/50 **label-blind**, fixed seed `79101`, with seeded pseudorandom SHA256 ordering of sorted group identities as in the existing membership allocator. Keep known groups together; canonical retained native-duplicate representatives are singleton groups under the original audit. Put floor(n/2) remaining singleton records in the pool and the rest in locked test. No Georgia test prevalence is computed or reported before the later commitment/continuation gate. Pool label draws use the existing draw seeds, label-blind deterministic ordering and nested F/G allocation as Experiment4. A/B500/500 are stratified using **pool labels only**, fixed seed `79102`, disjoint from every label draw. No seed search, outcome-dependent physical exclusion or changing the allocation to get favorable metrics.

**Resolved by author correction before source fitting:** do not stratify the pool/test allocation and do not read future test labels for split preparation. Only ID-only historical Georgia memberships are read before pool allocation. Read raw signals/labels solely for the newly designated pool under a record-exclusion guard. Report pool/A/B counts and prevalence, but locked-test counts only. The Georgia test prediction/scoring gate remains closed: protocol commit plus explicit continuation, future bitwise validation replay, prediction sealing before outcome access and one evaluation. No split-only test-label exception is authorized or used.

All earlier artifacts remain read-only. Test/raw sources and PTB fold10 are denied by runtime capability guards. New source checkpoints, validation predictions and memberships are private; only aggregate development results/figures are placed in docs/phase5 and figures/phase5.
