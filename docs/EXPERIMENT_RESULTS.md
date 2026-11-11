# Completed ECG experiments — locked fold10 results

All numerical results below derive from the completed PTB-XL1.0.3 fits and one joint locked test evaluation. Scores are arithmetic means of seed-specific metrics, not probability ensembles. Classification is multilabel ECG diagnosis, not future events, clinical risk or cross-cohort validation.

## Population and prevalence

Train 17,084, validation 2,146, test 2,158 recordings. Test patients 1,877. No detected patient or exact waveform overlap.

| Label | Test positives | Prevalence / AP random-ranking reference |
|---|---|---|
| NORM | 963 | 0.44625 |
| MI | 550 | 0.25487 |
| STTC | 521 | 0.24143 |
| CD | 496 | 0.22984 |
| HYP | 262 | 0.12141 |

## Complete registered leaderboard

The order below is prespecified, not an outer-score-selected winner. XResNet-final is a checkpoint alias, not another model fit.

| Procedure | Macro AUROC (95% CI) | Macro AP (95% CI) | Micro AP | Brier | ECE | Macro F1 |
|---|---|---|---|---|---|---|
| xresnet101 | 0.92985 [+0.92306, +0.93617] | 0.82759 [+0.81215, +0.84303] | 0.86784 | 0.08588 | 0.05153 | 0.74524 |
| xresnet101_final | 0.92800 [+0.92104, +0.93450] | 0.82241 [+0.80715, +0.83830] | 0.86483 | 0.09258 | 0.06734 | 0.74744 |
| flat_cnn | 0.91733 [+0.90974, +0.92440] | 0.80657 [+0.78983, +0.82287] | 0.84430 | 0.08811 | 0.04066 | 0.73436 |
| multiview | 0.92139 [+0.91396, +0.92852] | 0.81405 [+0.79706, +0.83004] | 0.84552 | 0.08694 | 0.03775 | 0.74299 |
| multiview_aug | 0.92068 [+0.91311, +0.92787] | 0.81698 [+0.80031, +0.83296] | 0.85262 | 0.08513 | 0.03789 | 0.74031 |

F1 uses independently validation-selected per-label thresholds. Brier/ECE use raw sigmoid probabilities: no probability calibration model was fitted. ECE has ten equal-width bins; empty bins contribute zero. The XResNet max-window procedure has different probability semantics from single full-record CNN prediction.

## Registered paired comparisons

multiview-minus-flat_cnn: Δmacro AUROC=+0.00406, primary97.5% Bonferroni CI [+0.00009, +0.00786] (positive conditional evidence); Δmacro AP=+0.00748, exploratory95%CI [-0.00078, +0.01600].

multiview-minus-xresnet101: Δmacro AUROC=-0.00847, primary97.5% Bonferroni CI [-0.01312, -0.00400] (negative conditional evidence); Δmacro AP=-0.01354, exploratory95%CI [-0.02253, -0.00408].

multiview_aug-minus-multiview: Δmacro AUROC=-0.00071, secondary95% unadjusted CI [-0.00333, +0.00191] (inconclusive conditional evidence); Δmacro AP=+0.00293, exploratory95%CI [-0.00247, +0.00803].

Two primary AUROC contrasts use97.5% Bonferroni percentile intervals. Augmentation is secondary95%. AP, per-class comparisons and all exploratory intervals have no familywise superiority interpretation. Shared patient-cluster resampling,2,000 valid draws, seed70501; all patients' recordings and paired seeds share multiplicities. Intervals condition on fitted models, validation selection and this split. They omit retraining, model-selection and external-population uncertainty. Seeds are not independent patients.

## Per-label ranking

| Procedure | NORM | MI | STTC | CD | HYP |
|---|---|---|---|---|---|
| xresnet101 | 0.94908 | 0.93792 | 0.93555 | 0.92620 | 0.90054 |
| xresnet101_final | 0.94778 | 0.93451 | 0.93358 | 0.92659 | 0.89754 |
| flat_cnn | 0.94268 | 0.92326 | 0.92995 | 0.90486 | 0.88592 |
| multiview | 0.94393 | 0.92274 | 0.93031 | 0.91293 | 0.89704 |
| multiview_aug | 0.94422 | 0.92221 | 0.93118 | 0.91409 | 0.89170 |

AUROC: additional per-label conditional95% intervals are in `results/locked_metrics.json`; per-class differences are exploratory.

| Procedure | NORM | MI | STTC | CD | HYP |
|---|---|---|---|---|---|
| xresnet101 | 0.92800 | 0.85440 | 0.83446 | 0.85877 | 0.66233 |
| xresnet101_final | 0.92479 | 0.84687 | 0.82486 | 0.85869 | 0.65681 |
| flat_cnn | 0.92148 | 0.83493 | 0.81916 | 0.82671 | 0.63055 |
| multiview | 0.92306 | 0.82670 | 0.81918 | 0.83891 | 0.66239 |
| multiview_aug | 0.92361 | 0.83374 | 0.81752 | 0.84339 | 0.66664 |

AP: additional per-label conditional95% intervals are in `results/locked_metrics.json`; per-class differences are exploratory.

## Thresholds, probability quality and confusion counts

### xresnet101

Seed 101; thresholds chosen on fold9, never fold10.

| Label | Threshold | Precision | Recall | Specificity | F1 | Brier | ECE | TN/FP/FN/TP |
|---|---|---|---|---|---|---|---|---|
| CD | 0.61 | 0.76171 | 0.75403 | 0.92960 | 0.75785 | 0.08721 | 0.06876 | 1545/117/122/374 |
| HYP | 0.29 | 0.51917 | 0.67176 | 0.91403 | 0.58569 | 0.06790 | 0.01775 | 1733/163/86/176 |
| MI | 0.46 | 0.70579 | 0.84182 | 0.87998 | 0.76783 | 0.09012 | 0.05648 | 1415/193/87/463 |
| NORM | 0.58 | 0.81079 | 0.92108 | 0.82678 | 0.86242 | 0.09558 | 0.06539 | 988/207/76/887 |
| STTC | 0.37 | 0.64146 | 0.87908 | 0.84362 | 0.74170 | 0.08845 | 0.05753 | 1381/256/63/458 |

Seed 202; thresholds chosen on fold9, never fold10.

| Label | Threshold | Precision | Recall | Specificity | F1 | Brier | ECE | TN/FP/FN/TP |
|---|---|---|---|---|---|---|---|---|
| CD | 0.67 | 0.85063 | 0.67742 | 0.96450 | 0.75421 | 0.08007 | 0.03702 | 1603/59/160/336 |
| HYP | 0.25 | 0.55128 | 0.65649 | 0.92616 | 0.59930 | 0.06503 | 0.01842 | 1756/140/90/172 |
| MI | 0.48 | 0.69219 | 0.83818 | 0.87251 | 0.75822 | 0.09568 | 0.06623 | 1403/205/89/461 |
| NORM | 0.62 | 0.80752 | 0.93666 | 0.82008 | 0.86731 | 0.10350 | 0.09082 | 980/215/61/902 |
| STTC | 0.46 | 0.71356 | 0.80806 | 0.89676 | 0.75788 | 0.08524 | 0.03694 | 1468/169/100/421 |

Fixed0.50 results are retained for every seed/label in the machine-readable metrics. Threshold changes do not change AP/AUROC. No deployment threshold is recommended.

### xresnet101_final

Seed 101; thresholds chosen on fold9, never fold10.

| Label | Threshold | Precision | Recall | Specificity | F1 | Brier | ECE | TN/FP/FN/TP |
|---|---|---|---|---|---|---|---|---|
| CD | 0.60 | 0.76707 | 0.77016 | 0.93020 | 0.76861 | 0.08545 | 0.05887 | 1546/116/114/382 |
| HYP | 0.37 | 0.55298 | 0.63740 | 0.92880 | 0.59220 | 0.07223 | 0.03860 | 1761/135/95/167 |
| MI | 0.58 | 0.74533 | 0.79818 | 0.90672 | 0.77085 | 0.09481 | 0.05754 | 1458/150/111/439 |
| NORM | 0.66 | 0.80359 | 0.93043 | 0.81674 | 0.86237 | 0.10876 | 0.09596 | 976/219/67/896 |
| STTC | 0.57 | 0.70149 | 0.81190 | 0.89004 | 0.75267 | 0.09873 | 0.07354 | 1457/180/98/423 |

Seed 202; thresholds chosen on fold9, never fold10.

| Label | Threshold | Precision | Recall | Specificity | F1 | Brier | ECE | TN/FP/FN/TP |
|---|---|---|---|---|---|---|---|---|
| CD | 0.82 | 0.87200 | 0.65927 | 0.97112 | 0.75086 | 0.08975 | 0.06173 | 1614/48/169/327 |
| HYP | 0.54 | 0.60324 | 0.56870 | 0.94831 | 0.58546 | 0.07260 | 0.04671 | 1798/98/113/149 |
| MI | 0.63 | 0.74825 | 0.77818 | 0.91045 | 0.76292 | 0.09775 | 0.06523 | 1464/144/122/428 |
| NORM | 0.56 | 0.79020 | 0.95431 | 0.79582 | 0.86453 | 0.10507 | 0.09290 | 951/244/44/919 |
| STTC | 0.70 | 0.74908 | 0.77927 | 0.91692 | 0.76388 | 0.10060 | 0.08228 | 1501/136/115/406 |

Fixed0.50 results are retained for every seed/label in the machine-readable metrics. Threshold changes do not change AP/AUROC. No deployment threshold is recommended.

### flat_cnn

Seed 101; thresholds chosen on fold9, never fold10.

| Label | Threshold | Precision | Recall | Specificity | F1 | Brier | ECE | TN/FP/FN/TP |
|---|---|---|---|---|---|---|---|---|
| CD | 0.34 | 0.74233 | 0.73185 | 0.92419 | 0.73706 | 0.08495 | 0.02739 | 1536/126/133/363 |
| HYP | 0.20 | 0.54448 | 0.58397 | 0.93249 | 0.56354 | 0.07028 | 0.03211 | 1768/128/109/153 |
| MI | 0.28 | 0.70063 | 0.81273 | 0.88122 | 0.75253 | 0.09398 | 0.04364 | 1417/191/103/447 |
| NORM | 0.53 | 0.81485 | 0.90031 | 0.83515 | 0.85545 | 0.10176 | 0.05089 | 998/197/96/867 |
| STTC | 0.24 | 0.71261 | 0.81382 | 0.89554 | 0.75986 | 0.08780 | 0.04768 | 1466/171/97/424 |

Seed 202; thresholds chosen on fold9, never fold10.

| Label | Threshold | Precision | Recall | Specificity | F1 | Brier | ECE | TN/FP/FN/TP |
|---|---|---|---|---|---|---|---|---|
| CD | 0.22 | 0.78261 | 0.68952 | 0.94284 | 0.73312 | 0.09158 | 0.06346 | 1567/95/154/342 |
| HYP | 0.23 | 0.55254 | 0.62214 | 0.93038 | 0.58528 | 0.06839 | 0.02203 | 1764/132/99/163 |
| MI | 0.34 | 0.67325 | 0.80545 | 0.86629 | 0.73344 | 0.09594 | 0.02714 | 1393/215/107/443 |
| NORM | 0.53 | 0.80533 | 0.91070 | 0.82259 | 0.85478 | 0.09954 | 0.05710 | 983/212/86/877 |
| STTC | 0.26 | 0.71195 | 0.83493 | 0.89249 | 0.76855 | 0.08688 | 0.03515 | 1461/176/86/435 |

Fixed0.50 results are retained for every seed/label in the machine-readable metrics. Threshold changes do not change AP/AUROC. No deployment threshold is recommended.

### multiview

Seed 101; thresholds chosen on fold9, never fold10.

| Label | Threshold | Precision | Recall | Specificity | F1 | Brier | ECE | TN/FP/FN/TP |
|---|---|---|---|---|---|---|---|---|
| CD | 0.61 | 0.71954 | 0.75000 | 0.91276 | 0.73445 | 0.10137 | 0.09946 | 1517/145/124/372 |
| HYP | 0.32 | 0.61265 | 0.59160 | 0.94831 | 0.60194 | 0.06570 | 0.01436 | 1798/98/107/155 |
| MI | 0.41 | 0.71858 | 0.73818 | 0.90112 | 0.72825 | 0.09537 | 0.01947 | 1449/159/144/406 |
| NORM | 0.41 | 0.82962 | 0.88993 | 0.85272 | 0.85872 | 0.09353 | 0.02777 | 1019/176/106/857 |
| STTC | 0.25 | 0.74101 | 0.79079 | 0.91203 | 0.76509 | 0.08859 | 0.04361 | 1493/144/109/412 |

Seed 202; thresholds chosen on fold9, never fold10.

| Label | Threshold | Precision | Recall | Specificity | F1 | Brier | ECE | TN/FP/FN/TP |
|---|---|---|---|---|---|---|---|---|
| CD | 0.18 | 0.80275 | 0.70565 | 0.94826 | 0.75107 | 0.09013 | 0.06784 | 1576/86/146/350 |
| HYP | 0.34 | 0.60448 | 0.61832 | 0.94409 | 0.61132 | 0.06557 | 0.01499 | 1790/106/100/162 |
| MI | 0.39 | 0.77344 | 0.72000 | 0.92786 | 0.74576 | 0.09109 | 0.02426 | 1492/116/154/396 |
| NORM | 0.49 | 0.81256 | 0.91381 | 0.83013 | 0.86022 | 0.09603 | 0.04231 | 992/203/83/880 |
| STTC | 0.39 | 0.72559 | 0.82726 | 0.90043 | 0.77309 | 0.08205 | 0.02346 | 1474/163/90/431 |

Fixed0.50 results are retained for every seed/label in the machine-readable metrics. Threshold changes do not change AP/AUROC. No deployment threshold is recommended.

### multiview_aug

Seed 101; thresholds chosen on fold9, never fold10.

| Label | Threshold | Precision | Recall | Specificity | F1 | Brier | ECE | TN/FP/FN/TP |
|---|---|---|---|---|---|---|---|---|
| CD | 0.44 | 0.81100 | 0.68347 | 0.95247 | 0.74179 | 0.08230 | 0.03393 | 1583/79/157/339 |
| HYP | 0.50 | 0.69951 | 0.54198 | 0.96783 | 0.61075 | 0.06612 | 0.02756 | 1835/61/120/142 |
| MI | 0.45 | 0.75092 | 0.74000 | 0.91604 | 0.74542 | 0.09026 | 0.02546 | 1473/135/143/407 |
| NORM | 0.62 | 0.83155 | 0.88681 | 0.85523 | 0.85829 | 0.09761 | 0.05620 | 1022/173/109/854 |
| STTC | 0.16 | 0.66914 | 0.86564 | 0.86378 | 0.75481 | 0.08787 | 0.04201 | 1414/223/70/451 |

Seed 202; thresholds chosen on fold9, never fold10.

| Label | Threshold | Precision | Recall | Specificity | F1 | Brier | ECE | TN/FP/FN/TP |
|---|---|---|---|---|---|---|---|---|
| CD | 0.57 | 0.82082 | 0.68347 | 0.95548 | 0.74587 | 0.07992 | 0.02988 | 1588/74/157/339 |
| HYP | 0.21 | 0.57565 | 0.59542 | 0.93935 | 0.58537 | 0.06699 | 0.03160 | 1781/115/106/156 |
| MI | 0.21 | 0.69494 | 0.77455 | 0.88371 | 0.73259 | 0.09845 | 0.05819 | 1421/187/124/426 |
| NORM | 0.53 | 0.81354 | 0.91070 | 0.83180 | 0.85938 | 0.10092 | 0.05960 | 994/201/86/877 |
| STTC | 0.31 | 0.71382 | 0.83301 | 0.89371 | 0.76882 | 0.08083 | 0.01448 | 1463/174/87/434 |

Fixed0.50 results are retained for every seed/label in the machine-readable metrics. Threshold changes do not change AP/AUROC. No deployment threshold is recommended.

## Development checkpoints and compute

| Model | Seed | Best epoch | Validation AUROC | Validation AP | Parameters | Training min (no validation) | Validation min |
|---|---|---|---|---|---|---|---|
| xresnet101 | 101 | 37 | 0.93399 | 0.82759 | 1872261 | 167.95 | 95.30 |
| xresnet101 | 202 | 37 | 0.93502 | 0.83031 | 1872261 | 168.08 | 95.16 |
| flat_cnn | 101 | 18 | 0.92269 | 0.79918 | 206117 | 18.79 | 1.53 |
| flat_cnn | 202 | 18 | 0.92359 | 0.80529 | 206117 | 19.34 | 1.57 |
| multiview | 101 | 13 | 0.92264 | 0.80453 | 212261 | 44.92 | 3.44 |
| multiview | 202 | 18 | 0.92436 | 0.80786 | 212261 | 45.60 | 3.51 |
| multiview_aug | 101 | 19 | 0.92329 | 0.80869 | 212261 | 46.04 | 3.44 |
| multiview_aug | 202 | 22 | 0.92424 | 0.81124 | 212261 | 48.14 | 3.60 |

All full fits completed50 epochs. CPU float32, four threads per fit, up to three concurrent workers. Reported training durations exclude validation and are not the study wall time. Best checkpoint is earliest maximal validation macro AUROC; final epoch also preserved. Reference matches pinned architecture but modern software/dataset/optimizer/selection differences prevent exact numerical reproduction.

| Procedure | Seed | Test inference seconds |
|---|---|---|
| xresnet101 | 101 | 74.758 |
| xresnet101 | 202 | 75.350 |
| xresnet101_final | 101 | 75.206 |
| xresnet101_final | 202 | 75.104 |
| flat_cnn | 101 | 0.942 |
| flat_cnn | 202 | 0.943 |
| multiview | 101 | 2.211 |
| multiview | 202 | 2.224 |
| multiview_aug | 101 | 2.218 |
| multiview_aug | 202 | 2.228 |

## Integrity and limitations

Actual predictor attempts: 11; successful 11; failed 0. This includes three training-only smoke fits, separate from eight full fits. Synthetic test fits do not use the real ledger. Unused reserve is not used after test opening.

All registered predictions were generated/sealed before metrics. See `results/evaluation_lock.json`, `results/prediction_seal.json`, `results/evaluation_completion.json` and the ignored SQLite ledger. No post-test fitting, selection or recalibration. Raw data, checkpoints and per-person predictions remain local.

Known published fold10 results inform the benchmark choice, so this is not independent hypothesis confirmation. Only two seeds, one official split and no external cohort. Unknown linkage/near-duplicates, annotation uncertainty, ECG version differences, derived-view redundancy and limited tuning remain. Patient bootstrap captures test-sampling variability conditional on models; no clinical utility or selective-transfer superiority is established.

## Figures and underlying records

PNG/PDF research figures are under `results/figures/`, with captions/checksums in `results/figure_inventory.json`. Per-fit aggregate histories in `results/baseline/` and `results/multiview/`; complete test metrics in `results/locked_metrics.json`. Individual predictions/checkpoints are excluded.

## Explicit seed-wise locked ranking

| Procedure | Seed | Macro AUROC | Macro AP |
|---|---:|---:|---:|
| flat_cnn | 101 | 0.916626 | 0.807199 |
| flat_cnn | 202 | 0.918043 | 0.805935 |
| multiview | 101 | 0.920855 | 0.811025 |
| multiview | 202 | 0.921924 | 0.817071 |
| multiview_aug | 101 | 0.920676 | 0.817834 |
| multiview_aug | 202 | 0.920682 | 0.816126 |
| xresnet101 | 101 | 0.929463 | 0.826396 |
| xresnet101 | 202 | 0.930246 | 0.828786 |
| xresnet101_final | 101 | 0.927283 | 0.819603 |
| xresnet101_final | 202 | 0.928720 | 0.825211 |

See the final summary's direct answers for interpretation. These seeds share one test population and do not constitute independent validation cohorts.
