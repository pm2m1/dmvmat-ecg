# Internal pilot development results — Phase 2E

**DEVELOPMENT ONLY. No O predictions, outcomes, performance metrics or confidence intervals were accessed/generated.**

## Execution and scope

The adopted two-label CD/STTC ECG annotation-classification pilot completed **33/33 registered predictor fits**, **33/33 C calibration artifacts** (66 fitted label mappings), with **0 failed/interrupted/retried fits** and **0 reserve attempts used**. The hard ceiling remains36. This is an internal age-subgroup study, not external clinical validation, future-event prediction or clinical utility.

Authority: [signed human adoption](../../private/phase2d_joint44301_v1/approvals/PROTOCOL_ADOPTION_SIGNED.json), [separate development grant](../../private/phase2d_joint44301_v1/approvals/DEVELOPMENT_AUTHORIZATION.json), and [signing receipt](../../private/phase2d_joint44301_v1/approvals/SIGNING_RECEIPT.md). No new approval was generated. The draft source/config files were not rewritten; adoption is the separate human-attested ledger record.

## Population and training

| Role | Patients | CD positive | STTC positive | CD/STTC jointly positive |
|---|---:|---:|---:|---:|
| source_train | 6492 | 905 | 933 | 137 |
| source_stop | 1146 | 159 | 165 | 24 |
| A | 4388 | 1262 | 1331 | 293 |
| E | 128 | 35 | 43 | 6 |
| C | 128 | 32 | 43 | 7 |
| d4101:n128:F | 128 | 34 | 39 | 9 |
| d4101:n512:F | 512 | 141 | 157 | 29 |
| d4101:n128:G | 256 | 65 | 67 | 17 |
| d4102:n128:F | 128 | 35 | 37 | 6 |
| d4102:n512:F | 512 | 145 | 157 | 32 |
| d4102:n128:G | 256 | 72 | 83 | 21 |

Source ages18–59, target60–89. Folds1–9 only; original fold10 excluded. One deterministic canonical ECG per patient retains its own exact recording annotations; serial findings were not merged. Unknown clinical diagnosis status is not converted to a verified negative: these endpoints are recorded tag presence/absence. All actual pilot tag masks were observed.

The new source starts randomly (seed3101), fits scalar normalization only on6492 source-training records, and uses1146 disjoint source-stopping records. CPU float32; four threads per worker, at most two concurrent predictor workers; AdamW/OneCycle and all target settings exactly as the adopted [execution contract](../../src/dmvmat_ecg/phase2d/contract.py) specifies. Source ran50 epochs/2550 updates; source-stopping AP retained epoch21/update1071. Model size212066 parameters. No augmentation, architecture search or alignment loss was added.

All pretrained target methods start at that same retained source checkpoint and normalization. Target-only starts afresh with its own F∪G statistics. Draw seeds4101/4102, paired training seeds5101/5102; F128/512; G256; shared E128/C128. Maximum40/160 updates, checkpoint looks every2/8 updates, patience5, pretrained epoch zero eligible.

## Source reference and prevalence

Source-stopping macro AP 0.75710440, macro AUROC 0.92215572. Target-E source-only raw AP 0.78334620, AUROC 0.87638863. These are different populations/prevalences and their nominal AP difference is not a transfer gain.

E contains128 patients,35 CD positives (AP random-ranking reference0.2734375) and43 STTC positives (0.3359375); mean prevalence0.3046875. All target model rows below use these same E patients. E selected checkpoints, so these are optimistic development scores, not unbiased final performance.

## Development model comparisons

Arithmetic means over the two paired draws, never pooled or best-draw scores. Per-label results and all individual runs appear below and in the aggregate files.

| F | Method | Mean raw AP | Mean AUROC | Retained updates (4101,4102) |
|---:|---|---:|---:|---|
| 128 | Head-F | 0.78361671 | 0.87646544 | 14,0 |
| 128 | Head-cost | 0.78368546 | 0.87660224 | 0,14 |
| 128 | Full-F | 0.78411300 | 0.87700307 | 2,2 |
| 128 | Full-cost | 0.78479469 | 0.87789111 | 16,0 |
| 128 | Random-A | 0.78351089 | 0.87661064 | 2,0 |
| 128 | Selective A (primary) | 0.78334620 | 0.87638863 | 0,0 |
| 128 | Selective B (secondary) | 0.78388034 | 0.87691786 | 4,0 |
| 128 | Target-only | 0.61223084 | 0.75836123 | 40,36 |
| 512 | Head-F | 0.78342487 | 0.87651703 | 0,32 |
| 512 | Head-cost | 0.78405392 | 0.87680744 | 32,24 |
| 512 | Full-F | 0.78914100 | 0.88491713 | 88,48 |
| 512 | Full-cost | 0.78862192 | 0.88156672 | 0,96 |
| 512 | Random-A | 0.78334620 | 0.87638863 | 0,0 |
| 512 | Selective A (primary) | 0.78609544 | 0.88046623 | 72,0 |
| 512 | Selective B (secondary) | 0.78726182 | 0.87950023 | 64,0 |
| 512 | Target-only | 0.67362199 | 0.79425161 | 80,160 |

### Prespecified primary contrast, described on development E only

Formulation A minus Full-cost, mean CD/STTC raw AP at F512, averaged across both draws. This preserves the registered primary definition but does not replace its future sealed-O assessment.

| Draw | ΔAP | ΔAUROC |
|---|---:|---:|
| 4101 | +0.00549848 | +0.00815519 |
| 4102 | -0.01055143 | -0.01035616 |
| Mean | -0.00252648 | -0.00110049 |

The mean development difference favors Full-cost, while the two draws have opposite signs. No superiority, equivalence, significance or absence-of-effect conclusion is justified. No paired holdout bootstrap/CI was run. No method was discarded or rerun in response.

## Selector evidence, abstention and retained changes

Each evidence calculation used999 MMD permutations,4999 geometry bootstraps, MMD cap256, source class caps256, minimum10/class or joint cell, seven-module Holm correction, and geometry upper quantile1−0.05/14. MMD remains selection evidence, not a loss. Complete sanitized module statistics are in [the audit JSON](../../results/phase2e/COMPLETED_DEVELOPMENT_AUDIT.json); private evidence artifacts additionally retain null samples and source-anchor identities.

| Method/draw | ADAPT modules | Trainable parameters | Retained update F128/F512 | Invalid label-module geometries |
|---|---:|---:|---|---:|
| Selective A (primary), 4101 | 4 | 115682 | 0/72 | 0/14 |
| Selective B (secondary), 4101 | 3 | 81762 | 4/64 | 0/14 |
| Selective A (primary), 4102 | 6 | 180162 | 0/0 | 0/14 |
| Selective B (secondary), 4102 | 5 | 146242 | 0/0 | 0/14 |

No invalid-geometry abstentions or all-SHARE selector masks occurred. This differs from retaining epoch zero: A retained source weights in3/4 episodes, B in2/4, and13/28 pretrained comparator/adaptation episodes retained epoch zero overall. A therefore requested adaptation without retaining it in most episodes. No threshold was changed to force decisions.

G is fixed within each draw across F128/F512, so masks/evidence repeat by construction; this does not show increasing selector-support stability. A adapts view3/view5/fusion1/fusion2 in draw4101 and additionally view1/view2 in draw4102. B omits view5 in both draws. View4 remains SHARE throughout. These are task-specific policy outputs, not clinical importance claims.

Random-A exactly matched A on parameters and module count. Its mask differs from A in draw4101 and equals A in draw4102 (both F tiers). The exact-cost set has3 masks; equality probability1/3. No rejection sampling was used to force a different mask. Every SHARE parameter/buffer matched the source exactly;1560 tensor comparisons passed. Effective per-owner L2 changes for every run are retained in its aggregate summary.

## Label and compute accounting

| Method | Gradient labels | Evidence | E | C | Total distinct target people, F128/F512 |
|---|---|---|---:|---:|---:|
| Head-F / Full-F | F | none |128|128|384 /768|
| Head-cost / Full-cost / Target-only | F∪G | none |128|128|640 /1024|
| Random-A / Selective A / Selective B | F |G256|128|128|640 /1024|

All label unions and observed-label cell counts (two per person) passed audit. Full-cost receives G as gradient supervision; no selector-only labels are hidden. Raw source-only has no target fitting labels; calibrated source-only uses C128. Its separate descriptive target-E report additionally accesses128 assessment labels, shared with checkpoint selection; they are disclosed separately from fitting/calibration costs. Prior label-based cohort/split audits are also separate and mean these are conditional predictive-access budgets, not real-world annotation-acquisition costs.

Summed end-to-end predictor elapsed time 1414.29s (23.57min); selector portions total 391.27s. These sums include role loading/initialization/selection and overlap across two workers, so they are not wall time, pure optimization time or proof of compute equivalence. Update counts and trainable parameters are logged for every attempt. Observed source process RSS was about2GiB; no full peak-memory/inference-cost benchmark was run.

## Calibration

All66 positive-slope C-only Platt mappings converged; no identity fallback was needed. Each mapping, C-selected threshold and hash is preserved. No C result selected a model, mask or checkpoint. Raw scores determine the primary AP. Descriptive E operating metrics in the per-run summaries use0.50; frozen C-selected thresholds are retained for the future locked assessment.

For the source reference, mean raw E Brier 0.12891730 versus calibrated 0.12945186: calibration did not improve this descriptive E value. This is not a reason to refit calibration. Calibrated E performance for every target model was not persisted and was not reconstructed after locking.

## Complete per-label attempt results

AP/AUROC are continuous-score metrics. Source row uses source stopping; every target row uses E. Full checkpoint/config/preprocessing/calibration SHA-256 values are in [the index](../../results/phase2e/CONFIGURATION_AND_ARTIFACT_INDEX.json); all other metrics and histories are in [per-attempt aggregates](../../results/phase2e/attempts/).

| Job | CD AP | STTC AP | CD AUROC | STTC AUROC | Selected update | Executed updates |
|---|---:|---:|---:|---:|---:|---:|
| source | 0.79571355 | 0.71849526 | 0.91818164 | 0.92612980 | 1071 | 2550 |
| d4101:n128:head_f | 0.83874329 | 0.72903113 | 0.92380952 | 0.82927497 | 14 | 24 |
| d4101:n128:head_cost | 0.83858897 | 0.72810344 | 0.92350230 | 0.82927497 | 0 | 10 |
| d4101:n128:full_f | 0.84059995 | 0.72817668 | 0.92473118 | 0.82927497 | 2 | 12 |
| d4101:n128:full_cost | 0.84734026 | 0.72514612 | 0.92841782 | 0.83036936 | 16 | 26 |
| d4101:n128:random_a | 0.83699036 | 0.73036079 | 0.92411674 | 0.82954856 | 2 | 12 |
| d4101:n128:selective_a | 0.83858897 | 0.72810344 | 0.92350230 | 0.82927497 | 0 | 10 |
| d4101:n128:selective_b | 0.84045299 | 0.72837595 | 0.92534562 | 0.82954856 | 4 | 14 |
| d4101:n128:target_only | 0.64044121 | 0.62952119 | 0.77296467 | 0.73543092 | 40 | 40 |
| d4101:n512:head_f | 0.83858897 | 0.72810344 | 0.92350230 | 0.82927497 | 0 | 40 |
| d4101:n512:head_cost | 0.83857791 | 0.72974265 | 0.92380952 | 0.83009576 | 32 | 72 |
| d4101:n512:full_f | 0.84054725 | 0.73883196 | 0.92503840 | 0.84815321 | 88 | 128 |
| d4101:n512:full_cost | 0.83858897 | 0.72810344 | 0.92350230 | 0.82927497 | 0 | 40 |
| d4101:n512:random_a | 0.83858897 | 0.72810344 | 0.92350230 | 0.82927497 | 0 | 40 |
| d4101:n512:selective_a | 0.83778330 | 0.73990606 | 0.92503840 | 0.84404925 | 72 | 112 |
| d4101:n512:selective_b | 0.84178528 | 0.74056960 | 0.92473118 | 0.84049248 | 64 | 104 |
| d4101:n512:target_only | 0.67828689 | 0.64543497 | 0.82058372 | 0.77318741 | 80 | 120 |
| d4102:n128:head_f | 0.83858897 | 0.72810344 | 0.92350230 | 0.82927497 | 0 | 10 |
| d4102:n128:head_cost | 0.83874329 | 0.72930616 | 0.92380952 | 0.82982216 | 14 | 24 |
| d4102:n128:full_f | 0.83962304 | 0.72805234 | 0.92473118 | 0.82927497 | 2 | 12 |
| d4102:n128:full_cost | 0.83858897 | 0.72810344 | 0.92350230 | 0.82927497 | 0 | 10 |
| d4102:n128:random_a | 0.83858897 | 0.72810344 | 0.92350230 | 0.82927497 | 0 | 10 |
| d4102:n128:selective_a | 0.83858897 | 0.72810344 | 0.92350230 | 0.82927497 | 0 | 10 |
| d4102:n128:selective_b | 0.83858897 | 0.72810344 | 0.92350230 | 0.82927497 | 0 | 10 |
| d4102:n128:target_only | 0.62954942 | 0.54941155 | 0.81013825 | 0.71491108 | 36 | 40 |
| d4102:n512:head_f | 0.83756287 | 0.72944419 | 0.92319508 | 0.83009576 | 32 | 72 |
| d4102:n512:head_cost | 0.83858897 | 0.72930616 | 0.92350230 | 0.82982216 | 24 | 64 |
| d4102:n512:full_f | 0.85477734 | 0.72240743 | 0.93118280 | 0.83529412 | 48 | 88 |
| d4102:n512:full_cost | 0.85586360 | 0.73193167 | 0.93026114 | 0.84322845 | 96 | 136 |
| d4102:n512:random_a | 0.83858897 | 0.72810344 | 0.92350230 | 0.82927497 | 0 | 40 |
| d4102:n512:selective_a | 0.83858897 | 0.72810344 | 0.92350230 | 0.82927497 | 0 | 40 |
| d4102:n512:selective_b | 0.83858897 | 0.72810344 | 0.92350230 | 0.82927497 | 0 | 40 |
| d4102:n512:target_only | 0.72692886 | 0.64383723 | 0.82181260 | 0.76142271 | 160 | 160 |

## Limitations and evidence boundary

One source seed, two overlapping target draws, shared E128 and C128, checkpoint selection on E, prior reuse of PTB-XL, annotation uncertainty and unknown linkage beyond supplied patient IDs limit interpretation. The short target-only schedule is a compute-matched scratch control, not an optimized target-only benchmark. No epoch-mean training loss was persisted by the frozen engine; actual stopping BCE/AP curves and finite-gradient/update checks were retained.

The replacement O has a disclosed metadata-feasibility audit history and shares790 patients with the earlier holdout. It remains sealed for predictive work, but must not be called pristine or independent. External clinical transfer remains blocked. No performance claim is based on original fold10 or the old five-superclass benchmark.

## Figures

Actual development figures, each PNG/PDF: [source stopping](../../results/phase2e/figures/source_stopping_curves.png), [budget comparison](../../results/phase2e/figures/development_budget_comparison.png), [selector masks](../../results/phase2e/figures/selector_masks.png). No synthetic or holdout curve is included.
