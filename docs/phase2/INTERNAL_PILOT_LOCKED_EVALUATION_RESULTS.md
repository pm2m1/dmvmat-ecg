# Internal pilot — single locked O evaluation (Phase 2F)

**LOCKED EVALUATION COMPLETE. Internal exploratory ECG annotation classification only.**

## Executive result

The prespecified primary selective procedure did not outperform matched-label Full-cost fine-tuning. At F512, mean raw CD/STTC AP across both paired draws was **0.79805929 for A** and **0.80433859 for Full-cost**. A−Full-cost ΔAP **-0.00627930**, conditional paired95% CI **[-0.00900938, -0.00335019]**. ΔAUROC **-0.00316186**,95% CI **[-0.00464399, -0.00169761]**. The intervals lie below zero and favor Full-cost for these fitted models on this internal population. This is not a universal claim about selective adaptation.

No selective-superiority claim is made. A primary interval must exclude0 in favor of A before such a conditional claim would be considered. An interval including0 is **inconclusive**, never evidence of “no effect.” No candidate, mask, checkpoint, calibration or threshold was changed after seeing O.

## Population, provenance and evidence limits

One new patient-disjoint source model (source ages18–59), target ages60–89, PTB-XL1.0.3 folds1–9, ordered CD/STTC annotation labels. Task: exact-record ECG annotation classification, not independently confirmed diagnoses, future events or clinical utility. One canonical ECG per supplied patient ID. The completed five-superclass fold10 benchmark is different and was not reopened or used to tune this study.

Role O has2633 records/2633 patients; all5266 tag-observation cells are observed. CD:757 positive/1876 negative; STTC:799 positive/1834 negative. The labels are non-exclusive.

AP prevalence references: CD **0.28750475**, STTC **0.30345613**; label-macro reference **0.29548044**. These apply to every absolute AP below, since all models use the same O population.

**Required history disclosure:** O shares790 patients with an earlier proposed holdout and had a label-feasibility audit before adoption. The broader PTB-XL data and research direction informed earlier work. O is not pristine or independent external validation. This is an internal age-subgroup pilot in one dataset, not a hospital/device or cross-country transfer experiment. Unknown person linkage beyond published patient IDs remains a limitation.

## All registered model families

Arithmetic means of paired draw-specific metrics; no probability ensembling, best-draw selection or pooled-OOF replacement. Source-only is one source checkpoint, not another predictor fit. Every one of the33 frozen models is retained/reported, even when its weights equal source. Raw and already-C-calibrated versions were sealed together.

| F | Method | Frozen scores | Macro AP | AP reference | Macro AUROC | Mean Brier | Mean log loss | Mean ECE10 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| — | Source-only | raw | 0.79706698 | 0.29548044 | 0.88482728 | 0.11270753 | 0.37539379 | 0.03996491 |
| — | Source-only | calibrated | 0.79706698 | 0.29548044 | 0.88482728 | 0.11253521 | 0.37674435 | 0.03681908 |
| 128 | Head-F | raw | 0.79726769 | 0.29548044 | 0.88496387 | 0.11249103 | 0.37423133 | 0.03745390 |
| 128 | Head-F | calibrated | 0.79726769 | 0.29548044 | 0.88496387 | 0.11248530 | 0.37628133 | 0.03771104 |
| 128 | Head-cost | raw | 0.79728720 | 0.29548044 | 0.88497690 | 0.11249874 | 0.37418968 | 0.03772213 |
| 128 | Head-cost | calibrated | 0.79728720 | 0.29548044 | 0.88497690 | 0.11247664 | 0.37617760 | 0.03718710 |
| 128 | Full-F | raw | 0.79709577 | 0.29548044 | 0.88475507 | 0.11267215 | 0.37508337 | 0.03787006 |
| 128 | Full-F | calibrated | 0.79709577 | 0.29548044 | 0.88475507 | 0.11268906 | 0.37640666 | 0.03689586 |
| 128 | Full-cost | raw | 0.79769530 | 0.29548044 | 0.88565877 | 0.11198513 | 0.37216277 | 0.03504891 |
| 128 | Full-cost | calibrated | 0.79769530 | 0.29548044 | 0.88565877 | 0.11239599 | 0.37506811 | 0.03751509 |
| 128 | Random-A | raw | 0.79694376 | 0.29548044 | 0.88478297 | 0.11266772 | 0.37521834 | 0.03919107 |
| 128 | Random-A | calibrated | 0.79694376 | 0.29548044 | 0.88478297 | 0.11256698 | 0.37665073 | 0.03664577 |
| 128 | Selective A | raw | 0.79706698 | 0.29548044 | 0.88482728 | 0.11270753 | 0.37539379 | 0.03996491 |
| 128 | Selective A | calibrated | 0.79706698 | 0.29548044 | 0.88482728 | 0.11253521 | 0.37674435 | 0.03681908 |
| 128 | Selective B | raw | 0.79728277 | 0.29548044 | 0.88489529 | 0.11242129 | 0.37424518 | 0.03727787 |
| 128 | Selective B | calibrated | 0.79728277 | 0.29548044 | 0.88489529 | 0.11249628 | 0.37687064 | 0.03683232 |
| 128 | Target-only | raw | 0.60519133 | 0.29548044 | 0.75034218 | 0.17309019 | 0.52834456 | 0.07842851 |
| 128 | Target-only | calibrated | 0.60519133 | 0.29548044 | 0.75034218 | 0.16720891 | 0.51063419 | 0.04452053 |
| 512 | Head-F | raw | 0.79734628 | 0.29548044 | 0.88499820 | 0.11244337 | 0.37351086 | 0.03712669 |
| 512 | Head-F | calibrated | 0.79734628 | 0.29548044 | 0.88499820 | 0.11247971 | 0.37605922 | 0.03769873 |
| 512 | Head-cost | raw | 0.79753777 | 0.29548044 | 0.88516599 | 0.11217178 | 0.37206278 | 0.03388675 |
| 512 | Head-cost | calibrated | 0.79753777 | 0.29548044 | 0.88516599 | 0.11241462 | 0.37550200 | 0.03778928 |
| 512 | Full-F | raw | 0.80493738 | 0.29548044 | 0.88911925 | 0.11045376 | 0.36506175 | 0.03004921 |
| 512 | Full-F | calibrated | 0.80493738 | 0.29548044 | 0.88911925 | 0.11126712 | 0.36970286 | 0.03712795 |
| 512 | Full-cost | raw | 0.80433859 | 0.29548044 | 0.88852033 | 0.11090312 | 0.36626355 | 0.03298677 |
| 512 | Full-cost | calibrated | 0.80433859 | 0.29548044 | 0.88852033 | 0.11183629 | 0.37207975 | 0.03962941 |
| 512 | Random-A | raw | 0.79706698 | 0.29548044 | 0.88482728 | 0.11270753 | 0.37539379 | 0.03996491 |
| 512 | Random-A | calibrated | 0.79706698 | 0.29548044 | 0.88482728 | 0.11253521 | 0.37674435 | 0.03681908 |
| 512 | Selective A | raw | 0.79805929 | 0.29548044 | 0.88535848 | 0.11199259 | 0.37048602 | 0.03329103 |
| 512 | Selective A | calibrated | 0.79805929 | 0.29548044 | 0.88535848 | 0.11223361 | 0.37516447 | 0.03743675 |
| 512 | Selective B | raw | 0.79841375 | 0.29548044 | 0.88535812 | 0.11221964 | 0.37088475 | 0.03474532 |
| 512 | Selective B | calibrated | 0.79841375 | 0.29548044 | 0.88535820 | 0.11213812 | 0.37564708 | 0.03663233 |
| 512 | Target-only | raw | 0.67100494 | 0.29548044 | 0.79857057 | 0.15764581 | 0.50653521 | 0.07555309 |
| 512 | Target-only | calibrated | 0.67100494 | 0.29548044 | 0.79857057 | 0.15120217 | 0.47427925 | 0.04211719 |

## Primary paired contrast

Fixed estimand: mean over draws4101/4102 of [mean(AP_CD,AP_STTC)(A)−mean(AP_CD,AP_STTC)(Full-cost)] at F512, raw scores. Each label has equal weight. Seeds5101/5102 are paired within draws; the one source checkpoint is shared. No independent-source replication is implied.

| Draw | Δ raw macro AP | Δ macro AUROC |
|---|---:|---:|
| 4101 | +0.00198461 | +0.00106240 |
| 4102 | -0.01454321 | -0.00738612 |
| Mean | -0.00627930 | -0.00316186 |

The draw directions differ. The negative mean is dominated by draw4102, in which A retained epoch zero and Full-cost retained update96. In draw4101 A retained update72 while Full-cost retained epoch zero. This does not identify a causal per-module failure: masks, learned updates and checkpoint selection all differ. Do not infer that any particular frozen lead/temporal view caused the loss without a separate controlled experiment.

| Exploratory label-specific A−Full-cost | ΔAP | Ordinary95% CI | ΔAUROC | Ordinary95% CI |
|---|---:|---|---:|---|
| CD | -0.00078571 | [-0.00297855, +0.00141781] | -0.00023343 | [-0.00180212, +0.00137291] |
| STTC | -0.01177289 | [-0.01669501, -0.00633877] | -0.00609029 | [-0.00857404, -0.00360359] |

The STTC comparison contributes most of the negative macro contrast. CD label-specific intervals include0 and are inconclusive. These label-specific intervals are exploratory, not additional multiplicity-adjusted primary hypotheses. The proposed+0.005 AP engineering relevance signal was not achieved; it is not a clinical margin.

## Five registered secondary contrasts — F512 raw scores

Bonferroni99% intervals correspond to five secondary AP contrasts (familywise α0.05). Ordinary95% intervals are also shown. AUROC intervals are accompanying secondary descriptions, not a new confirmatory metric family. All raw-score contrasts preserve the registered direction.

| Contrast | ΔAP | 95% CI | Bonferroni99% CI | Interpretation |
|---|---:|---|---|---|
| A_minus_Head_cost | +0.00052152 | [-0.00106826, +0.00218345] | [-0.00156379, +0.00273420] | Inconclusive |
| A_minus_Random_A | +0.00099231 | [-0.00043495, +0.00249419] | [-0.00088786, +0.00299927] | Inconclusive |
| A_minus_Full_F | -0.00687809 | [-0.01069933, -0.00284350] | [-0.01189117, -0.00155885] | Conditional contrast favors Full-F; unequal label budget |
| B_minus_A | +0.00035446 | [-0.00055019, +0.00124300] | [-0.00081049, +0.00151581] | Inconclusive |
| A_minus_raw_Source_only | +0.00099231 | [-0.00043495, +0.00249419] | [-0.00088786, +0.00299927] | Inconclusive |

| Contrast | ΔAUROC | 95% CI | 99% CI |
|---|---:|---|---|
| A_minus_Head_cost | +0.00019248 | [-0.00066963, +0.00105119] | [-0.00091808, +0.00130453] |
| A_minus_Random_A | +0.00053120 | [-0.00035341, +0.00140907] | [-0.00060390, +0.00171997] |
| A_minus_Full_F | -0.00376077 | [-0.00580509, -0.00174458] | [-0.00644140, -0.00103077] |
| B_minus_A | -0.00000036 | [-0.00052024, +0.00052915] | [-0.00070301, +0.00070279] |
| A_minus_raw_Source_only | +0.00053120 | [-0.00035341, +0.00140907] | [-0.00060390, +0.00171997] |

A−Head-cost, A−Random, B−A and A−raw Source-only are inconclusive. Do not switch the primary method to B after inspection. A−Full-F is negative, but Full-F consumes a different total label budget and is not the primary fairness comparison. Random-A at F512 retained epoch zero in both draws, so its frozen raw predictions equal source; their identical contrasts are not independent corroboration.

## Bootstrap and statistical interpretation

Exactly10000 unstratified patient-cluster resamples, seed64001. Whole CD/STTC vectors and paired predictions stay together; one patient-multiplicity matrix is shared by every method, label and episode. Metrics are computed within each fitted episode, then averaged, never from averaged probabilities. Non-interpolated AP and AUROC use continuous scores and exact tie handling. Weighted ranking was verified against scikit-learn on fictional tied-score examples.

Ordinary intervals use2.5/97.5 percentiles;99% intervals use0.5/99.5 percentiles. Missing either class makes a label/replicate undefined for macro comparisons; undefined replicates would not be replaced, and conclusive CI interpretation would be withheld if>1%. **Observed undefined counts: CD0, STTC0; all10000 primary and secondary replicates finite.**

Intervals condition on the fitted source, target fits, split and reused E/C. They omit variability from new source training, new target adaptation draws, policy/research selection, annotation uncertainty and external populations. The paired conditional interval can be narrow because predictions are highly correlated; it does not overturn the pre-adoption warning that this is a small feasibility/effect-size pilot for general algorithm conclusions. Two draws are not two independent clinical populations, and seeds are not patients.

## Calibration and operating points

Brier/log loss/ECE above use the sealed frozen predictions. Ten equal-width reliability bins cover[0,.1),…,[.9,1]. The original33 C-only mappings and thresholds were loaded unchanged; no C or O outcome selected a new threshold/model. Raw operating metrics use0.50. Calibrated operating metrics use each frozen C-selected threshold, with calibrated0.50 metrics additionally reported.

Source-only mean Brier changed from0.11270753 raw to0.11253521 with frozen C calibration, while log loss worsened from0.37539379 to0.37674435. At F512, A mean Brier changed0.11199259→0.11223361; Full-cost0.11090312→0.11183629. Calibration did not uniformly improve probability quality and was not revised.

All132 label/method/score-version regressions converged and are labeled **“diagnostic calibration statistics (not applied)”**. They were estimated only after the complete prediction seal, against logit(sealed probabilities) with1e-6 numerical clipping. They assess joint logistic intercept/slope; no statistic was applied to a prediction, used for selection or saved as a calibrator. These are not the frozen C mapping coefficients.

Complete diagnostics: [132-entry diagnostic table](../../results/phase2f/DIAGNOSTIC_CALIBRATION_STATISTICS.md). All confusion counts, precision, recall, specificity, F1 and balanced accuracy: [operating-point table](../../results/phase2f/OPERATING_POINT_METRICS.md) and [full metrics JSON](../../results/phase2f/LOCKED_METRICS.json). Every confusion matrix totals2633. Thresholds are research operating points, not clinical recommendations.

## Selector, epoch-zero and negative-transfer observations

A masks requested4/6 internal ADAPT modules by draw; B requested3/5. No invalid-geometry abstention or all-SHARE selector mask occurred during development. G256 is identical across F128/F512 within each draw, so repeated masks are expected rather than independent stability evidence. A retained epoch zero in3/4 episodes; B in2/4. All13 retained-zero pretrained checkpoints equal source exactly. Every mask and abstaining/zero-retained episode remained included; no favorable subset was chosen.

Across28 pretrained adaptation episodes (not independent populations), raw macro AP was below source in2, identical in13, and above in13. The two lower point estimates were F128 Full-F draw4101 (−0.00001795) and F128 Random-A draw4101 (−0.00024645). These are descriptive negative-transfer counts, not per-episode significance claims. Source remains a strong baseline; masks requesting adaptation do not themselves establish benefit.

## Label and compute fairness

At F512, A and Full-cost each access1024 distinct target people: F512+G256+E128+C128 (2048 observed tag cells). At F128 the matched total is640. Full-cost receives G for gradient updates; A/Random/B use G for evidence. Head-cost and Target-only also use F∪G. Head-F/Full-F use384/768 total and are secondary resource comparisons. Source-only raw uses no target fitting labels; calibrated source-only uses C128; assessment O labels are disclosed separately for every method. Prior split-audit label access is separate from these conditional predictive-label budgets.

All methods retain their predeclared maximum40/160 target-update budgets and E stopping rules. Actual update counts/elapsed cost differ and are logged; matching optimizer steps is not compute equivalence. Target-only is a short-schedule scratch control, not an optimized target-only ECG benchmark.

No new predictor fit was run in Phase2F. Predictor count remains33/36 withzero failed/interrupted/retried fits; allthree reserves unused. Evaluation/analysis elapsed 132.99s (excluding report-generation/tests); the first inference timing includes physical O loading and later calls reuse the checked label-free cache, so timings are not an architecture speed benchmark.

## Complete per-episode AP/AUROC results

Every raw and frozen-C-calibrated procedure is listed; prevalence baselines repeat next to the corresponding AP. Per-procedure conditional95% intervals and label intervals are in [bootstrap summaries](../../results/phase2f/BOOTSTRAP_SUMMARIES.json).

| Job | Scores | CD AP | CD AP reference | CD AUROC | STTC AP | STTC AP reference | STTC AUROC |
|---|---|---:|---:|---:|---:|---:|---:|
| source | raw | 0.84560335 | 0.28750475 | 0.89686100 | 0.74853061 | 0.30345613 | 0.87279355 |
| source | calibrated | 0.84560335 | 0.28750475 | 0.89686100 | 0.74853061 | 0.30345613 | 0.87279355 |
| d4101:n128:head_f | raw | 0.84563851 | 0.28750475 | 0.89685255 | 0.74929827 | 0.30345613 | 0.87334836 |
| d4101:n128:head_f | calibrated | 0.84563851 | 0.28750475 | 0.89685255 | 0.74929827 | 0.30345613 | 0.87334836 |
| d4101:n128:head_cost | raw | 0.84560335 | 0.28750475 | 0.89686100 | 0.74853061 | 0.30345613 | 0.87279355 |
| d4101:n128:head_cost | calibrated | 0.84560335 | 0.28750475 | 0.89686100 | 0.74853061 | 0.30345613 | 0.87279355 |
| d4101:n128:full_f | raw | 0.84534012 | 0.28750475 | 0.89666313 | 0.74875794 | 0.30345613 | 0.87308427 |
| d4101:n128:full_f | calibrated | 0.84534012 | 0.28750475 | 0.89666313 | 0.74875794 | 0.30345613 | 0.87308427 |
| d4101:n128:full_cost | raw | 0.84453578 | 0.28750475 | 0.89672368 | 0.75211147 | 0.30345613 | 0.87625685 |
| d4101:n128:full_cost | calibrated | 0.84453578 | 0.28750475 | 0.89672368 | 0.75211147 | 0.30345613 | 0.87625685 |
| d4101:n128:random_a | raw | 0.84535112 | 0.28750475 | 0.89653638 | 0.74828995 | 0.30345613 | 0.87294096 |
| d4101:n128:random_a | calibrated | 0.84535112 | 0.28750475 | 0.89653638 | 0.74828995 | 0.30345613 | 0.87294096 |
| d4101:n128:selective_a | raw | 0.84560335 | 0.28750475 | 0.89686100 | 0.74853061 | 0.30345613 | 0.87279355 |
| d4101:n128:selective_a | calibrated | 0.84560335 | 0.28750475 | 0.89686100 | 0.74853061 | 0.30345613 | 0.87279355 |
| d4101:n128:selective_b | raw | 0.84561394 | 0.28750475 | 0.89698634 | 0.74938317 | 0.30345613 | 0.87294028 |
| d4101:n128:selective_b | calibrated | 0.84561394 | 0.28750475 | 0.89698634 | 0.74938317 | 0.30345613 | 0.87294028 |
| d4101:n128:target_only | raw | 0.72305849 | 0.28750475 | 0.81451020 | 0.46756582 | 0.30345613 | 0.66902228 |
| d4101:n128:target_only | calibrated | 0.72305849 | 0.28750475 | 0.81451020 | 0.46756582 | 0.30345613 | 0.66902228 |
| d4101:n512:head_f | raw | 0.84560335 | 0.28750475 | 0.89686100 | 0.74853061 | 0.30345613 | 0.87279355 |
| d4101:n512:head_f | calibrated | 0.84560335 | 0.28750475 | 0.89686100 | 0.74853061 | 0.30345613 | 0.87279355 |
| d4101:n512:head_cost | raw | 0.84568479 | 0.28750475 | 0.89684058 | 0.74943810 | 0.30345613 | 0.87353330 |
| d4101:n512:head_cost | calibrated | 0.84568479 | 0.28750475 | 0.89684058 | 0.74943810 | 0.30345613 | 0.87353330 |
| d4101:n512:full_f | raw | 0.84873839 | 0.28750475 | 0.89617092 | 0.75690055 | 0.30345613 | 0.88087618 |
| d4101:n512:full_f | calibrated | 0.84873839 | 0.28750475 | 0.89617092 | 0.75690055 | 0.30345613 | 0.88087618 |
| d4101:n512:full_cost | raw | 0.84560335 | 0.28750475 | 0.89686100 | 0.74853061 | 0.30345613 | 0.87279355 |
| d4101:n512:full_cost | calibrated | 0.84560335 | 0.28750475 | 0.89686100 | 0.74853061 | 0.30345613 | 0.87279355 |
| d4101:n512:random_a | raw | 0.84560335 | 0.28750475 | 0.89686100 | 0.74853061 | 0.30345613 | 0.87279355 |
| d4101:n512:random_a | calibrated | 0.84560335 | 0.28750475 | 0.89686100 | 0.74853061 | 0.30345613 | 0.87279355 |
| d4101:n512:selective_a | raw | 0.84750270 | 0.28750475 | 0.89636597 | 0.75060049 | 0.30345613 | 0.87541338 |
| d4101:n512:selective_a | calibrated | 0.84750270 | 0.28750475 | 0.89636597 | 0.75060049 | 0.30345613 | 0.87541338 |
| d4101:n512:selective_b | raw | 0.84760852 | 0.28750475 | 0.89727011 | 0.75191253 | 0.30345613 | 0.87450780 |
| d4101:n512:selective_b | calibrated | 0.84760852 | 0.28750475 | 0.89727011 | 0.75191253 | 0.30345613 | 0.87450814 |
| d4101:n512:target_only | raw | 0.77194196 | 0.28750475 | 0.84841691 | 0.55272129 | 0.30345613 | 0.73851379 |
| d4101:n512:target_only | calibrated | 0.77194196 | 0.28750475 | 0.84841691 | 0.55272129 | 0.30345613 | 0.73851379 |
| d4102:n128:head_f | raw | 0.84560335 | 0.28750475 | 0.89686100 | 0.74853061 | 0.30345613 | 0.87279355 |
| d4102:n128:head_f | calibrated | 0.84560335 | 0.28750475 | 0.89686100 | 0.74853061 | 0.30345613 | 0.87279355 |
| d4102:n128:head_cost | raw | 0.84566554 | 0.28750475 | 0.89681663 | 0.74934928 | 0.30345613 | 0.87343640 |
| d4102:n128:head_cost | calibrated | 0.84566554 | 0.28750475 | 0.89681663 | 0.74934928 | 0.30345613 | 0.87343640 |
| d4102:n128:full_f | raw | 0.84538753 | 0.28750475 | 0.89636879 | 0.74889751 | 0.30345613 | 0.87290411 |
| d4102:n128:full_f | calibrated | 0.84538753 | 0.28750475 | 0.89636879 | 0.74889751 | 0.30345613 | 0.87290411 |
| d4102:n128:full_cost | raw | 0.84560335 | 0.28750475 | 0.89686100 | 0.74853061 | 0.30345613 | 0.87279355 |
| d4102:n128:full_cost | calibrated | 0.84560335 | 0.28750475 | 0.89686100 | 0.74853061 | 0.30345613 | 0.87279355 |
| d4102:n128:random_a | raw | 0.84560335 | 0.28750475 | 0.89686100 | 0.74853061 | 0.30345613 | 0.87279355 |
| d4102:n128:random_a | calibrated | 0.84560335 | 0.28750475 | 0.89686100 | 0.74853061 | 0.30345613 | 0.87279355 |
| d4102:n128:selective_a | raw | 0.84560335 | 0.28750475 | 0.89686100 | 0.74853061 | 0.30345613 | 0.87279355 |
| d4102:n128:selective_a | calibrated | 0.84560335 | 0.28750475 | 0.89686100 | 0.74853061 | 0.30345613 | 0.87279355 |
| d4102:n128:selective_b | raw | 0.84560335 | 0.28750475 | 0.89686100 | 0.74853061 | 0.30345613 | 0.87279355 |
| d4102:n128:selective_b | calibrated | 0.84560335 | 0.28750475 | 0.89686100 | 0.74853061 | 0.30345613 | 0.87279355 |
| d4102:n128:target_only | raw | 0.71374122 | 0.28750475 | 0.81050987 | 0.51639978 | 0.30345613 | 0.70732636 |
| d4102:n128:target_only | calibrated | 0.71374122 | 0.28750475 | 0.81050987 | 0.51639978 | 0.30345613 | 0.70732636 |
| d4102:n512:head_f | raw | 0.84567547 | 0.28750475 | 0.89677016 | 0.74957570 | 0.30345613 | 0.87356811 |
| d4102:n512:head_f | calibrated | 0.84567547 | 0.28750475 | 0.89677016 | 0.74957570 | 0.30345613 | 0.87356811 |
| d4102:n512:head_cost | raw | 0.84567630 | 0.28750475 | 0.89682368 | 0.74935189 | 0.30345613 | 0.87346642 |
| d4102:n512:head_cost | calibrated | 0.84567630 | 0.28750475 | 0.89682368 | 0.74935189 | 0.30345613 | 0.87346642 |
| d4102:n512:full_f | raw | 0.84671758 | 0.28750475 | 0.89596460 | 0.76739299 | 0.30345613 | 0.88346529 |
| d4102:n512:full_f | calibrated | 0.84671758 | 0.28750475 | 0.89596460 | 0.76739299 | 0.30345613 | 0.88346529 |
| d4102:n512:full_cost | raw | 0.84907412 | 0.28750475 | 0.89683283 | 0.77414627 | 0.30345613 | 0.88759395 |
| d4102:n512:full_cost | calibrated | 0.84907412 | 0.28750475 | 0.89683283 | 0.77414627 | 0.30345613 | 0.88759395 |
| d4102:n512:random_a | raw | 0.84560335 | 0.28750475 | 0.89686100 | 0.74853061 | 0.30345613 | 0.87279355 |
| d4102:n512:random_a | calibrated | 0.84560335 | 0.28750475 | 0.89686100 | 0.74853061 | 0.30345613 | 0.87279355 |
| d4102:n512:selective_a | raw | 0.84560335 | 0.28750475 | 0.89686100 | 0.74853061 | 0.30345613 | 0.87279355 |
| d4102:n512:selective_a | calibrated | 0.84560335 | 0.28750475 | 0.89686100 | 0.74853061 | 0.30345613 | 0.87279355 |
| d4102:n512:selective_b | raw | 0.84560335 | 0.28750475 | 0.89686100 | 0.74853061 | 0.30345613 | 0.87279355 |
| d4102:n512:selective_b | calibrated | 0.84560335 | 0.28750475 | 0.89686100 | 0.74853061 | 0.30345613 | 0.87279355 |
| d4102:n512:target_only | raw | 0.75245006 | 0.28750475 | 0.84017190 | 0.60690646 | 0.30345613 | 0.76717967 |
| d4102:n512:target_only | calibrated | 0.75245006 | 0.28750475 | 0.84017190 | 0.60690646 | 0.30345613 | 0.76717967 |

## Execution integrity, authority and repository

Reviewer **Parth Mishra**, local date **2026-11-13**. [Signed evaluation authorization](../../private/phase2d_joint44301_v1/approvals/EVALUATION_AUTHORIZATION_SIGNED.json) was transcribed from the explicit human grant using the exact ledger schema/statement, then recorded via the existing API. Its semantic hash is `f48c313c7ff0f54ee95e37865b356880787c6a5e4820e98793ef048e3f83bec9`. Diagnostic-only permission was separately supplied by the human and is recorded in the pre-access receipt.

```text
lock_hash = 1f31e3835fa3ee270d4866313af91107ac2d654b3832469a369bd08326a8d609
code_hash = ed812868495e91aac456f955315635932031a67e250c798fad5127a831bc3afa
ledger_id = d6f784e8-5073-4fad-a804-1011820f92ef
canonical_path = /Users/kiyo/Desktop/dmvmat-ecg/private/phase2d_joint44301_v1/study.sqlite
contract_hash = dbc0afe9bee29e7323a3a353b37660b8dff20733e3fe62fa94958d95400ecb28
dependency_hash = 195cd5ae983b6a3e196bc52ac0cf0997b26f60f45fb801e53256b8b57f881424
manifest_hash = 65cf652084396403077918e0e9be560eaedd009ab4fc8587260c833ddcf4f816
```

All111 frozen artifacts matched before access and after completion. All33 prediction files containing66 score procedures were sealed/hashes recorded before the sole outcome-read event. The last prediction event is497, outcome-read event498. There was one physical O signal materialization, followed by integrity-checked cache reuse, and one outcome accessor call. No row-level signals, outcomes or predictions appear in public outputs.

Ledger now **evaluated**, **33 predictor attempts**, **one evaluation lease**, authorization consumed. A second evaluation entry was rejected without a second access or O read. No training, fine-tuning, normalization fit, predictive calibration fit, threshold/mask/checkpoint change or method removal occurred. The separately authorized132 diagnostic statistical estimations never altered forecasts or calibrator artifacts.

Post-evaluation verification: `.venv/bin/python -B -m pytest tests private/phase2f_run_v1/test_analysis.py -q -p no:cacheprovider`: **215 passed,7 existing warnings**,36.56s. Ruff and formatting checks passed for the repository and private evaluation helpers. No inventoried file changed. No commit, push, merge or PR. HEAD remains `c7b795553ee9806caeb1835e7a829dc0469872eb`. New Phase2F outputs are untracked; existing Phase2E outputs preserved.

Receipts: [pre-access integrity](../../results/phase2f/PRE_ACCESS_VERIFICATION.json), [prediction seal](../../results/phase2f/PREDICTION_SEAL_RECEIPT.json), [completion](../../results/phase2f/EVALUATION_COMPLETION_RECEIPT.json); private [evaluation receipt](../../private/phase2f_run_v1/EVALUATION_RECEIPT.json). All original signals, outcomes, learned weights and row predictions remain ignored/private. Bootstrap metric distributions are retained privately; their entries are resampling statistics, not new patient records.

New file inventory: this report; aggregate metrics/bootstrap/leaderboard/negative-transfer JSONs, two aggregate diagnostic/operating Markdown tables and five PNG/PDF figure pairs under `results/phase2f/`; three private evaluator/statistics/synthetic-test helpers, evaluation log, bootstrap statistic archive and receipt under `private/phase2f_run_v1/`; the signed evaluation record under private approvals;33 private immutable prediction NPZs and their seal under the canonical ledger evaluation namespace. Existing inventoried code, manifests, approvals other than the newly authorized evaluation record, all111 locked artifacts and historical experiments were preserved.

## Figures and conclusion

Actual locked aggregate figures (PNG/PDF): [per-label AP and prevalence](../../results/phase2f/figures/locked_per_label_AP.png), [primary paired differences](../../results/phase2f/figures/primary_paired_differences.png), [secondary AP intervals](../../results/phase2f/figures/secondary_AP_intervals.png), [STTC reliability](../../results/phase2f/figures/STTC_reliability_frozen_scores.png), [C-threshold confusion matrices](../../results/phase2f/figures/frozen_C_threshold_confusion_matrices.png). No new outcome read was needed to generate them; all use saved aggregate statistics.

This is a defensible negative primary result for the fitted internal pilot: evidence-guided A did not beat matched-label full fine-tuning. Head/random/source comparisons and B−A are inconclusive, not “no effect.” A future study would need a new approved protocol with more independently trained sources/target draws and clinically defensible external endpoint/identity evidence. No follow-up fitting, selector redesign or repeated evaluation is performed here.
