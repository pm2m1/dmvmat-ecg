# Confusion matrices — existing scored ECG experiments

Aggregate counts only. This is a transcription and exact arithmetic summary of previously saved results, not a new test evaluation. No signals, per-record predictions, label arrays, model checkpoints or participant IDs were opened to prepare it. No training, inference, threshold search, calibration, bootstrap or tuning was performed. Existing files were read-only; no private file was copied into this document or staged.

## Reading the tables

- Columns use **TN, FP, FN, TP**, with rows = actual class and columns = predicted class (`[[TN, FP], [FN, TP]]`). Each label is a separate binary task; these are not mutually exclusive diagnostic classes.
- The saved decision rule is score **>= threshold**. Thresholds are frozen historical choices, not selected again here.
- **N is the original test-record count for each label and each run**, never the number of runs times the test size. Every saved per-run row was checked for integer, nonnegative counts, `TN + FP + FN + TP = N`, and constant positive/negative support. All included labels are fully observed; no unknown value was treated as negative.
- Mean tables are equal arithmetic means of counts over the indicated draws/episodes. **Fractional values are means over runs**, not a confusion matrix from pooled participants or an ensemble. Repeated runs share one test population.
- Means are exact: terminating decimals are shown directly; otherwise mixed fractions such as `123 + 2/9` mean exactly `123 + (2/9)`. No rounding correction, residual assignment or fractional participant was introduced. Thus every displayed mean row also sums exactly to N.
- Source-only aliases reproduce one unchanged source predictor across the corresponding draws/budgets. They do not represent new fits or new label use.

## Coverage and row checks

| Study | Test N per label | Per-run rows | Mean rows | Status |
|---|---:|---:|---:|---|
| Experiment 1: PTB-XL fold 10 | 2158 | 100 | 0 | All rows pass |
| Experiment 2: age transfer | 2633 | 264 | 136 | All rows pass |
| Experiment 4: CPSC, corrected V2 | 3860 | 324 | 36 | All rows pass |
| Phase 4C: CPSC | 3860 | 270 | 30 | All rows pass |
| Phase 5: Georgia | 4081 | 540 | 18 | All rows pass |

**Experiment 3 has no confusion counts: it was threshold-free by design.** No cutoff or counts were added retrospectively. The later CPSC adaptation/test experiment is listed here as Experiment 4, consistent with the project summary.

## 1. Experiment 1 — PTB-XL fold 10

Source: [locked_metrics.json](../results/locked_metrics.json), `runs/*/{validation_thresholds,fixed_half}/labels/*/confusion_matrix`. All five saved procedures, both seeds (101, 202), and all five diagnostic superclasses are included. `xresnet101_final` is the already-recorded final-epoch checkpoint alias, not an additional training fit. Validation-selected thresholds came from fold 9; no fold-10 threshold optimization was performed.

| Model | Seed | Class | Threshold policy | Threshold | TN | FP | FN | TP | N |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| flat_cnn | 101 | CD | validation-selected | 0.34 | 1536 | 126 | 133 | 363 | 2158 |
| flat_cnn | 101 | HYP | validation-selected | 0.2 | 1768 | 128 | 109 | 153 | 2158 |
| flat_cnn | 101 | MI | validation-selected | 0.28 | 1417 | 191 | 103 | 447 | 2158 |
| flat_cnn | 101 | NORM | validation-selected | 0.53 | 998 | 197 | 96 | 867 | 2158 |
| flat_cnn | 101 | STTC | validation-selected | 0.24 | 1466 | 171 | 97 | 424 | 2158 |
| flat_cnn | 101 | CD | fixed 0.5 | 0.5 | 1590 | 72 | 166 | 330 | 2158 |
| flat_cnn | 101 | HYP | fixed 0.5 | 0.5 | 1870 | 26 | 168 | 94 | 2158 |
| flat_cnn | 101 | MI | fixed 0.5 | 0.5 | 1487 | 121 | 152 | 398 | 2158 |
| flat_cnn | 101 | NORM | fixed 0.5 | 0.5 | 991 | 204 | 91 | 872 | 2158 |
| flat_cnn | 101 | STTC | fixed 0.5 | 0.5 | 1568 | 69 | 184 | 337 | 2158 |
| flat_cnn | 202 | CD | validation-selected | 0.22 | 1567 | 95 | 154 | 342 | 2158 |
| flat_cnn | 202 | HYP | validation-selected | 0.23 | 1764 | 132 | 99 | 163 | 2158 |
| flat_cnn | 202 | MI | validation-selected | 0.34 | 1393 | 215 | 107 | 443 | 2158 |
| flat_cnn | 202 | NORM | validation-selected | 0.53 | 983 | 212 | 86 | 877 | 2158 |
| flat_cnn | 202 | STTC | validation-selected | 0.26 | 1461 | 176 | 86 | 435 | 2158 |
| flat_cnn | 202 | CD | fixed 0.5 | 0.5 | 1644 | 18 | 228 | 268 | 2158 |
| flat_cnn | 202 | HYP | fixed 0.5 | 0.5 | 1858 | 38 | 154 | 108 | 2158 |
| flat_cnn | 202 | MI | fixed 0.5 | 0.5 | 1468 | 140 | 159 | 391 | 2158 |
| flat_cnn | 202 | NORM | fixed 0.5 | 0.5 | 971 | 224 | 75 | 888 | 2158 |
| flat_cnn | 202 | STTC | fixed 0.5 | 0.5 | 1549 | 88 | 175 | 346 | 2158 |
| multiview | 101 | CD | validation-selected | 0.61 | 1517 | 145 | 124 | 372 | 2158 |
| multiview | 101 | HYP | validation-selected | 0.32 | 1798 | 98 | 107 | 155 | 2158 |
| multiview | 101 | MI | validation-selected | 0.41 | 1449 | 159 | 144 | 406 | 2158 |
| multiview | 101 | NORM | validation-selected | 0.41 | 1019 | 176 | 106 | 857 | 2158 |
| multiview | 101 | STTC | validation-selected | 0.25 | 1493 | 144 | 109 | 412 | 2158 |
| multiview | 101 | CD | fixed 0.5 | 0.5 | 1479 | 183 | 93 | 403 | 2158 |
| multiview | 101 | HYP | fixed 0.5 | 0.5 | 1841 | 55 | 133 | 129 | 2158 |
| multiview | 101 | MI | fixed 0.5 | 0.5 | 1494 | 114 | 184 | 366 | 2158 |
| multiview | 101 | NORM | fixed 0.5 | 0.5 | 1049 | 146 | 137 | 826 | 2158 |
| multiview | 101 | STTC | fixed 0.5 | 0.5 | 1585 | 52 | 198 | 323 | 2158 |
| multiview | 202 | CD | validation-selected | 0.18 | 1576 | 86 | 146 | 350 | 2158 |
| multiview | 202 | HYP | validation-selected | 0.34 | 1790 | 106 | 100 | 162 | 2158 |
| multiview | 202 | MI | validation-selected | 0.39 | 1492 | 116 | 154 | 396 | 2158 |
| multiview | 202 | NORM | validation-selected | 0.49 | 992 | 203 | 83 | 880 | 2158 |
| multiview | 202 | STTC | validation-selected | 0.39 | 1474 | 163 | 90 | 431 | 2158 |
| multiview | 202 | CD | fixed 0.5 | 0.5 | 1642 | 20 | 232 | 264 | 2158 |
| multiview | 202 | HYP | fixed 0.5 | 0.5 | 1849 | 47 | 140 | 122 | 2158 |
| multiview | 202 | MI | fixed 0.5 | 0.5 | 1526 | 82 | 199 | 351 | 2158 |
| multiview | 202 | NORM | fixed 0.5 | 0.5 | 996 | 199 | 85 | 878 | 2158 |
| multiview | 202 | STTC | fixed 0.5 | 0.5 | 1527 | 110 | 122 | 399 | 2158 |
| multiview_aug | 101 | CD | validation-selected | 0.44 | 1583 | 79 | 157 | 339 | 2158 |
| multiview_aug | 101 | HYP | validation-selected | 0.5 | 1835 | 61 | 120 | 142 | 2158 |
| multiview_aug | 101 | MI | validation-selected | 0.45 | 1473 | 135 | 143 | 407 | 2158 |
| multiview_aug | 101 | NORM | validation-selected | 0.62 | 1022 | 173 | 109 | 854 | 2158 |
| multiview_aug | 101 | STTC | validation-selected | 0.16 | 1414 | 223 | 70 | 451 | 2158 |
| multiview_aug | 101 | CD | fixed 0.5 | 0.5 | 1599 | 63 | 164 | 332 | 2158 |
| multiview_aug | 101 | HYP | fixed 0.5 | 0.5 | 1835 | 61 | 120 | 142 | 2158 |
| multiview_aug | 101 | MI | fixed 0.5 | 0.5 | 1490 | 118 | 155 | 395 | 2158 |
| multiview_aug | 101 | NORM | fixed 0.5 | 0.5 | 978 | 217 | 78 | 885 | 2158 |
| multiview_aug | 101 | STTC | fixed 0.5 | 0.5 | 1563 | 74 | 188 | 333 | 2158 |
| multiview_aug | 202 | CD | validation-selected | 0.57 | 1588 | 74 | 157 | 339 | 2158 |
| multiview_aug | 202 | HYP | validation-selected | 0.21 | 1781 | 115 | 106 | 156 | 2158 |
| multiview_aug | 202 | MI | validation-selected | 0.21 | 1421 | 187 | 124 | 426 | 2158 |
| multiview_aug | 202 | NORM | validation-selected | 0.53 | 994 | 201 | 86 | 877 | 2158 |
| multiview_aug | 202 | STTC | validation-selected | 0.31 | 1463 | 174 | 87 | 434 | 2158 |
| multiview_aug | 202 | CD | fixed 0.5 | 0.5 | 1567 | 95 | 134 | 362 | 2158 |
| multiview_aug | 202 | HYP | fixed 0.5 | 0.5 | 1867 | 29 | 152 | 110 | 2158 |
| multiview_aug | 202 | MI | fixed 0.5 | 0.5 | 1526 | 82 | 204 | 346 | 2158 |
| multiview_aug | 202 | NORM | fixed 0.5 | 0.5 | 987 | 208 | 80 | 883 | 2158 |
| multiview_aug | 202 | STTC | fixed 0.5 | 0.5 | 1535 | 102 | 130 | 391 | 2158 |
| xresnet101 | 101 | CD | validation-selected | 0.61 | 1545 | 117 | 122 | 374 | 2158 |
| xresnet101 | 101 | HYP | validation-selected | 0.29 | 1733 | 163 | 86 | 176 | 2158 |
| xresnet101 | 101 | MI | validation-selected | 0.46 | 1415 | 193 | 87 | 463 | 2158 |
| xresnet101 | 101 | NORM | validation-selected | 0.58 | 988 | 207 | 76 | 887 | 2158 |
| xresnet101 | 101 | STTC | validation-selected | 0.37 | 1381 | 256 | 63 | 458 | 2158 |
| xresnet101 | 101 | CD | fixed 0.5 | 0.5 | 1495 | 167 | 93 | 403 | 2158 |
| xresnet101 | 101 | HYP | fixed 0.5 | 0.5 | 1821 | 75 | 123 | 139 | 2158 |
| xresnet101 | 101 | MI | fixed 0.5 | 0.5 | 1438 | 170 | 96 | 454 | 2158 |
| xresnet101 | 101 | NORM | fixed 0.5 | 0.5 | 952 | 243 | 49 | 914 | 2158 |
| xresnet101 | 101 | STTC | fixed 0.5 | 0.5 | 1461 | 176 | 94 | 427 | 2158 |
| xresnet101 | 202 | CD | validation-selected | 0.67 | 1603 | 59 | 160 | 336 | 2158 |
| xresnet101 | 202 | HYP | validation-selected | 0.25 | 1756 | 140 | 90 | 172 | 2158 |
| xresnet101 | 202 | MI | validation-selected | 0.48 | 1403 | 205 | 89 | 461 | 2158 |
| xresnet101 | 202 | NORM | validation-selected | 0.62 | 980 | 215 | 61 | 902 | 2158 |
| xresnet101 | 202 | STTC | validation-selected | 0.46 | 1468 | 169 | 100 | 421 | 2158 |
| xresnet101 | 202 | CD | fixed 0.5 | 0.5 | 1546 | 116 | 120 | 376 | 2158 |
| xresnet101 | 202 | HYP | fixed 0.5 | 0.5 | 1853 | 43 | 144 | 118 | 2158 |
| xresnet101 | 202 | MI | fixed 0.5 | 0.5 | 1408 | 200 | 95 | 455 | 2158 |
| xresnet101 | 202 | NORM | fixed 0.5 | 0.5 | 927 | 268 | 38 | 925 | 2158 |
| xresnet101 | 202 | STTC | fixed 0.5 | 0.5 | 1483 | 154 | 111 | 410 | 2158 |
| xresnet101_final | 101 | CD | validation-selected | 0.6 | 1546 | 116 | 114 | 382 | 2158 |
| xresnet101_final | 101 | HYP | validation-selected | 0.37 | 1761 | 135 | 95 | 167 | 2158 |
| xresnet101_final | 101 | MI | validation-selected | 0.58 | 1458 | 150 | 111 | 439 | 2158 |
| xresnet101_final | 101 | NORM | validation-selected | 0.66 | 976 | 219 | 67 | 896 | 2158 |
| xresnet101_final | 101 | STTC | validation-selected | 0.57 | 1457 | 180 | 98 | 423 | 2158 |
| xresnet101_final | 101 | CD | fixed 0.5 | 0.5 | 1511 | 151 | 91 | 405 | 2158 |
| xresnet101_final | 101 | HYP | fixed 0.5 | 0.5 | 1803 | 93 | 117 | 145 | 2158 |
| xresnet101_final | 101 | MI | fixed 0.5 | 0.5 | 1423 | 185 | 97 | 453 | 2158 |
| xresnet101_final | 101 | NORM | fixed 0.5 | 0.5 | 926 | 269 | 43 | 920 | 2158 |
| xresnet101_final | 101 | STTC | fixed 0.5 | 0.5 | 1426 | 211 | 86 | 435 | 2158 |
| xresnet101_final | 202 | CD | validation-selected | 0.82 | 1614 | 48 | 169 | 327 | 2158 |
| xresnet101_final | 202 | HYP | validation-selected | 0.54 | 1798 | 98 | 113 | 149 | 2158 |
| xresnet101_final | 202 | MI | validation-selected | 0.63 | 1464 | 144 | 122 | 428 | 2158 |
| xresnet101_final | 202 | NORM | validation-selected | 0.56 | 951 | 244 | 44 | 919 | 2158 |
| xresnet101_final | 202 | STTC | validation-selected | 0.7 | 1501 | 136 | 115 | 406 | 2158 |
| xresnet101_final | 202 | CD | fixed 0.5 | 0.5 | 1493 | 169 | 97 | 399 | 2158 |
| xresnet101_final | 202 | HYP | fixed 0.5 | 0.5 | 1783 | 113 | 104 | 158 | 2158 |
| xresnet101_final | 202 | MI | fixed 0.5 | 0.5 | 1418 | 190 | 97 | 453 | 2158 |
| xresnet101_final | 202 | NORM | fixed 0.5 | 0.5 | 934 | 261 | 34 | 929 | 2158 |
| xresnet101_final | 202 | STTC | fixed 0.5 | 0.5 | 1406 | 231 | 67 | 454 | 2158 |

## 2. Experiment 2 — internal age transfer

Source: [Phase 2F LOCKED_METRICS.json](../results/phase2f/LOCKED_METRICS.json), the completed 2,633-record locked O evaluation. The eight adapted methods use F budgets 128/512 and paired draw seeds 4101/4102. Source-only has one predictor and no target-adaptation budget. Budget is the recorded F budget; it is not a claim of equal total label cost across methods.

Both existing score variants and both saved operating policies are retained:

- `raw`, `fixed 0.5`: original uncalibrated scores at 0.5.
- `calibrated`, `fixed 0.5`: already-C-calibrated scores at 0.5.
- `raw`, `C-selected / raw default`: the historical raw-default 0.5 (duplicates the raw fixed-0.5 counts; it is not a C-selected raw threshold).
- `calibrated`, `C-selected / raw default`: each fitted procedure/class uses its previously frozen C-only threshold, including the original identity/default fallback where applicable. The per-draw thresholds are in Appendix A. No new calibration was fitted and no calibration diagnostic was applied.

Means are over the two draws; source-only is listed once (Runs = 1). Different frozen per-draw thresholds remain different policies applied to their own fitted scores; the mean is not evaluated at an averaged threshold.

| Method | Budget | Scores | Threshold policy | Class | Runs | TN | FP | FN | TP | N |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| full_cost | 128 | calibrated | C-selected / raw default | CD | 2 | 1752.5 | 123.5 | 194 | 563 | 2633 |
| full_cost | 128 | calibrated | fixed 0.5 | CD | 2 | 1783 | 93 | 221 | 536 | 2633 |
| full_cost | 128 | calibrated | C-selected / raw default | STTC | 2 | 1626 | 208 | 260 | 539 | 2633 |
| full_cost | 128 | calibrated | fixed 0.5 | STTC | 2 | 1608 | 226 | 252.5 | 546.5 | 2633 |
| full_cost | 128 | raw | C-selected / raw default | CD | 2 | 1785 | 91 | 221 | 536 | 2633 |
| full_cost | 128 | raw | fixed 0.5 | CD | 2 | 1785 | 91 | 221 | 536 | 2633 |
| full_cost | 128 | raw | C-selected / raw default | STTC | 2 | 1643.5 | 190.5 | 274.5 | 524.5 | 2633 |
| full_cost | 128 | raw | fixed 0.5 | STTC | 2 | 1643.5 | 190.5 | 274.5 | 524.5 | 2633 |
| full_f | 128 | calibrated | C-selected / raw default | CD | 2 | 1754.5 | 121.5 | 194 | 563 | 2633 |
| full_f | 128 | calibrated | fixed 0.5 | CD | 2 | 1784.5 | 91.5 | 221.5 | 535.5 | 2633 |
| full_f | 128 | calibrated | C-selected / raw default | STTC | 2 | 1588.5 | 245.5 | 231 | 568 | 2633 |
| full_f | 128 | calibrated | fixed 0.5 | STTC | 2 | 1617.5 | 216.5 | 251.5 | 547.5 | 2633 |
| full_f | 128 | raw | C-selected / raw default | CD | 2 | 1793.5 | 82.5 | 231 | 526 | 2633 |
| full_f | 128 | raw | fixed 0.5 | CD | 2 | 1793.5 | 82.5 | 231 | 526 | 2633 |
| full_f | 128 | raw | C-selected / raw default | STTC | 2 | 1640.5 | 193.5 | 277 | 522 | 2633 |
| full_f | 128 | raw | fixed 0.5 | STTC | 2 | 1640.5 | 193.5 | 277 | 522 | 2633 |
| head_cost | 128 | calibrated | C-selected / raw default | CD | 2 | 1749 | 127 | 188 | 569 | 2633 |
| head_cost | 128 | calibrated | fixed 0.5 | CD | 2 | 1783 | 93 | 222 | 535 | 2633 |
| head_cost | 128 | calibrated | C-selected / raw default | STTC | 2 | 1591 | 243 | 233.5 | 565.5 | 2633 |
| head_cost | 128 | calibrated | fixed 0.5 | STTC | 2 | 1616.5 | 217.5 | 258 | 541 | 2633 |
| head_cost | 128 | raw | C-selected / raw default | CD | 2 | 1790.5 | 85.5 | 228 | 529 | 2633 |
| head_cost | 128 | raw | fixed 0.5 | CD | 2 | 1790.5 | 85.5 | 228 | 529 | 2633 |
| head_cost | 128 | raw | C-selected / raw default | STTC | 2 | 1633.5 | 200.5 | 269.5 | 529.5 | 2633 |
| head_cost | 128 | raw | fixed 0.5 | STTC | 2 | 1633.5 | 200.5 | 269.5 | 529.5 | 2633 |
| head_f | 128 | calibrated | C-selected / raw default | CD | 2 | 1749 | 127 | 188 | 569 | 2633 |
| head_f | 128 | calibrated | fixed 0.5 | CD | 2 | 1783 | 93 | 222 | 535 | 2633 |
| head_f | 128 | calibrated | C-selected / raw default | STTC | 2 | 1591 | 243 | 233.5 | 565.5 | 2633 |
| head_f | 128 | calibrated | fixed 0.5 | STTC | 2 | 1616.5 | 217.5 | 258 | 541 | 2633 |
| head_f | 128 | raw | C-selected / raw default | CD | 2 | 1790.5 | 85.5 | 227.5 | 529.5 | 2633 |
| head_f | 128 | raw | fixed 0.5 | CD | 2 | 1790.5 | 85.5 | 227.5 | 529.5 | 2633 |
| head_f | 128 | raw | C-selected / raw default | STTC | 2 | 1634.5 | 199.5 | 270 | 529 | 2633 |
| head_f | 128 | raw | fixed 0.5 | STTC | 2 | 1634.5 | 199.5 | 270 | 529 | 2633 |
| random_a | 128 | calibrated | C-selected / raw default | CD | 2 | 1750.5 | 125.5 | 190.5 | 566.5 | 2633 |
| random_a | 128 | calibrated | fixed 0.5 | CD | 2 | 1784 | 92 | 221.5 | 535.5 | 2633 |
| random_a | 128 | calibrated | C-selected / raw default | STTC | 2 | 1594.5 | 239.5 | 237 | 562 | 2633 |
| random_a | 128 | calibrated | fixed 0.5 | STTC | 2 | 1619 | 215 | 256 | 543 | 2633 |
| random_a | 128 | raw | C-selected / raw default | CD | 2 | 1792 | 84 | 228.5 | 528.5 | 2633 |
| random_a | 128 | raw | fixed 0.5 | CD | 2 | 1792 | 84 | 228.5 | 528.5 | 2633 |
| random_a | 128 | raw | C-selected / raw default | STTC | 2 | 1636 | 198 | 269 | 530 | 2633 |
| random_a | 128 | raw | fixed 0.5 | STTC | 2 | 1636 | 198 | 269 | 530 | 2633 |
| selective_a | 128 | calibrated | C-selected / raw default | CD | 2 | 1749 | 127 | 188 | 569 | 2633 |
| selective_a | 128 | calibrated | fixed 0.5 | CD | 2 | 1783 | 93 | 222 | 535 | 2633 |
| selective_a | 128 | calibrated | C-selected / raw default | STTC | 2 | 1595 | 239 | 237 | 562 | 2633 |
| selective_a | 128 | calibrated | fixed 0.5 | STTC | 2 | 1619 | 215 | 258 | 541 | 2633 |
| selective_a | 128 | raw | C-selected / raw default | CD | 2 | 1791 | 85 | 228 | 529 | 2633 |
| selective_a | 128 | raw | fixed 0.5 | CD | 2 | 1791 | 85 | 228 | 529 | 2633 |
| selective_a | 128 | raw | C-selected / raw default | STTC | 2 | 1630 | 204 | 264 | 535 | 2633 |
| selective_a | 128 | raw | fixed 0.5 | STTC | 2 | 1630 | 204 | 264 | 535 | 2633 |
| selective_b | 128 | calibrated | C-selected / raw default | CD | 2 | 1748.5 | 127.5 | 188 | 569 | 2633 |
| selective_b | 128 | calibrated | fixed 0.5 | CD | 2 | 1783.5 | 92.5 | 221.5 | 535.5 | 2633 |
| selective_b | 128 | calibrated | C-selected / raw default | STTC | 2 | 1600 | 234 | 238.5 | 560.5 | 2633 |
| selective_b | 128 | calibrated | fixed 0.5 | STTC | 2 | 1620 | 214 | 256 | 543 | 2633 |
| selective_b | 128 | raw | C-selected / raw default | CD | 2 | 1792 | 84 | 227.5 | 529.5 | 2633 |
| selective_b | 128 | raw | fixed 0.5 | CD | 2 | 1792 | 84 | 227.5 | 529.5 | 2633 |
| selective_b | 128 | raw | C-selected / raw default | STTC | 2 | 1642.5 | 191.5 | 277.5 | 521.5 | 2633 |
| selective_b | 128 | raw | fixed 0.5 | STTC | 2 | 1642.5 | 191.5 | 277.5 | 521.5 | 2633 |
| target_only | 128 | calibrated | C-selected / raw default | CD | 2 | 1627.5 | 248.5 | 273 | 484 | 2633 |
| target_only | 128 | calibrated | fixed 0.5 | CD | 2 | 1759 | 117 | 380 | 377 | 2633 |
| target_only | 128 | calibrated | C-selected / raw default | STTC | 2 | 421.5 | 1412.5 | 73 | 726 | 2633 |
| target_only | 128 | calibrated | fixed 0.5 | STTC | 2 | 1774 | 60 | 698 | 101 | 2633 |
| target_only | 128 | raw | C-selected / raw default | CD | 2 | 1807.5 | 68.5 | 443 | 314 | 2633 |
| target_only | 128 | raw | fixed 0.5 | CD | 2 | 1807.5 | 68.5 | 443 | 314 | 2633 |
| target_only | 128 | raw | C-selected / raw default | STTC | 2 | 1634.5 | 199.5 | 602 | 197 | 2633 |
| target_only | 128 | raw | fixed 0.5 | STTC | 2 | 1634.5 | 199.5 | 602 | 197 | 2633 |
| full_cost | 512 | calibrated | C-selected / raw default | CD | 2 | 1740.5 | 135.5 | 185 | 572 | 2633 |
| full_cost | 512 | calibrated | fixed 0.5 | CD | 2 | 1786 | 90 | 228.5 | 528.5 | 2633 |
| full_cost | 512 | calibrated | C-selected / raw default | STTC | 2 | 1571 | 263 | 216.5 | 582.5 | 2633 |
| full_cost | 512 | calibrated | fixed 0.5 | STTC | 2 | 1596.5 | 237.5 | 238.5 | 560.5 | 2633 |
| full_cost | 512 | raw | C-selected / raw default | CD | 2 | 1789.5 | 86.5 | 228.5 | 528.5 | 2633 |
| full_cost | 512 | raw | fixed 0.5 | CD | 2 | 1789.5 | 86.5 | 228.5 | 528.5 | 2633 |
| full_cost | 512 | raw | C-selected / raw default | STTC | 2 | 1623.5 | 210.5 | 257.5 | 541.5 | 2633 |
| full_cost | 512 | raw | fixed 0.5 | STTC | 2 | 1623.5 | 210.5 | 257.5 | 541.5 | 2633 |
| full_f | 512 | calibrated | C-selected / raw default | CD | 2 | 1747.5 | 128.5 | 198 | 559 | 2633 |
| full_f | 512 | calibrated | fixed 0.5 | CD | 2 | 1783.5 | 92.5 | 221.5 | 535.5 | 2633 |
| full_f | 512 | calibrated | C-selected / raw default | STTC | 2 | 1473.5 | 360.5 | 170 | 629 | 2633 |
| full_f | 512 | calibrated | fixed 0.5 | STTC | 2 | 1596.5 | 237.5 | 239 | 560 | 2633 |
| full_f | 512 | raw | C-selected / raw default | CD | 2 | 1759.5 | 116.5 | 204 | 553 | 2633 |
| full_f | 512 | raw | fixed 0.5 | CD | 2 | 1759.5 | 116.5 | 204 | 553 | 2633 |
| full_f | 512 | raw | C-selected / raw default | STTC | 2 | 1653 | 181 | 279 | 520 | 2633 |
| full_f | 512 | raw | fixed 0.5 | STTC | 2 | 1653 | 181 | 279 | 520 | 2633 |
| head_cost | 512 | calibrated | C-selected / raw default | CD | 2 | 1749 | 127 | 188 | 569 | 2633 |
| head_cost | 512 | calibrated | fixed 0.5 | CD | 2 | 1783.5 | 92.5 | 222 | 535 | 2633 |
| head_cost | 512 | calibrated | C-selected / raw default | STTC | 2 | 1587 | 247 | 230 | 569 | 2633 |
| head_cost | 512 | calibrated | fixed 0.5 | STTC | 2 | 1614 | 220 | 258 | 541 | 2633 |
| head_cost | 512 | raw | C-selected / raw default | CD | 2 | 1789.5 | 86.5 | 224.5 | 532.5 | 2633 |
| head_cost | 512 | raw | fixed 0.5 | CD | 2 | 1789.5 | 86.5 | 224.5 | 532.5 | 2633 |
| head_cost | 512 | raw | C-selected / raw default | STTC | 2 | 1638.5 | 195.5 | 276.5 | 522.5 | 2633 |
| head_cost | 512 | raw | fixed 0.5 | STTC | 2 | 1638.5 | 195.5 | 276.5 | 522.5 | 2633 |
| head_f | 512 | calibrated | C-selected / raw default | CD | 2 | 1766.5 | 109.5 | 205 | 552 | 2633 |
| head_f | 512 | calibrated | fixed 0.5 | CD | 2 | 1783.5 | 92.5 | 222 | 535 | 2633 |
| head_f | 512 | calibrated | C-selected / raw default | STTC | 2 | 1591 | 243 | 233.5 | 565.5 | 2633 |
| head_f | 512 | calibrated | fixed 0.5 | STTC | 2 | 1616 | 218 | 258 | 541 | 2633 |
| head_f | 512 | raw | C-selected / raw default | CD | 2 | 1789.5 | 86.5 | 226 | 531 | 2633 |
| head_f | 512 | raw | fixed 0.5 | CD | 2 | 1789.5 | 86.5 | 226 | 531 | 2633 |
| head_f | 512 | raw | C-selected / raw default | STTC | 2 | 1631.5 | 202.5 | 267.5 | 531.5 | 2633 |
| head_f | 512 | raw | fixed 0.5 | STTC | 2 | 1631.5 | 202.5 | 267.5 | 531.5 | 2633 |
| random_a | 512 | calibrated | C-selected / raw default | CD | 2 | 1749 | 127 | 188 | 569 | 2633 |
| random_a | 512 | calibrated | fixed 0.5 | CD | 2 | 1783 | 93 | 222 | 535 | 2633 |
| random_a | 512 | calibrated | C-selected / raw default | STTC | 2 | 1595 | 239 | 237 | 562 | 2633 |
| random_a | 512 | calibrated | fixed 0.5 | STTC | 2 | 1619 | 215 | 258 | 541 | 2633 |
| random_a | 512 | raw | C-selected / raw default | CD | 2 | 1791 | 85 | 228 | 529 | 2633 |
| random_a | 512 | raw | fixed 0.5 | CD | 2 | 1791 | 85 | 228 | 529 | 2633 |
| random_a | 512 | raw | C-selected / raw default | STTC | 2 | 1630 | 204 | 264 | 535 | 2633 |
| random_a | 512 | raw | fixed 0.5 | STTC | 2 | 1630 | 204 | 264 | 535 | 2633 |
| selective_a | 512 | calibrated | C-selected / raw default | CD | 2 | 1752 | 124 | 192 | 565 | 2633 |
| selective_a | 512 | calibrated | fixed 0.5 | CD | 2 | 1781.5 | 94.5 | 219.5 | 537.5 | 2633 |
| selective_a | 512 | calibrated | C-selected / raw default | STTC | 2 | 1559.5 | 274.5 | 214 | 585 | 2633 |
| selective_a | 512 | calibrated | fixed 0.5 | STTC | 2 | 1618 | 216 | 259 | 540 | 2633 |
| selective_a | 512 | raw | C-selected / raw default | CD | 2 | 1780 | 96 | 216 | 541 | 2633 |
| selective_a | 512 | raw | fixed 0.5 | CD | 2 | 1780 | 96 | 216 | 541 | 2633 |
| selective_a | 512 | raw | C-selected / raw default | STTC | 2 | 1662 | 172 | 298 | 501 | 2633 |
| selective_a | 512 | raw | fixed 0.5 | STTC | 2 | 1662 | 172 | 298 | 501 | 2633 |
| selective_b | 512 | calibrated | C-selected / raw default | CD | 2 | 1753.5 | 122.5 | 189.5 | 567.5 | 2633 |
| selective_b | 512 | calibrated | fixed 0.5 | CD | 2 | 1782.5 | 93.5 | 219.5 | 537.5 | 2633 |
| selective_b | 512 | calibrated | C-selected / raw default | STTC | 2 | 1560.5 | 273.5 | 212.5 | 586.5 | 2633 |
| selective_b | 512 | calibrated | fixed 0.5 | STTC | 2 | 1624 | 210 | 258.5 | 540.5 | 2633 |
| selective_b | 512 | raw | C-selected / raw default | CD | 2 | 1780.5 | 95.5 | 216 | 541 | 2633 |
| selective_b | 512 | raw | fixed 0.5 | CD | 2 | 1780.5 | 95.5 | 216 | 541 | 2633 |
| selective_b | 512 | raw | C-selected / raw default | STTC | 2 | 1672 | 162 | 311 | 488 | 2633 |
| selective_b | 512 | raw | fixed 0.5 | STTC | 2 | 1672 | 162 | 311 | 488 | 2633 |
| target_only | 512 | calibrated | C-selected / raw default | CD | 2 | 1727 | 149 | 297.5 | 459.5 | 2633 |
| target_only | 512 | calibrated | fixed 0.5 | CD | 2 | 1754.5 | 121.5 | 315 | 442 | 2633 |
| target_only | 512 | calibrated | C-selected / raw default | STTC | 2 | 1363.5 | 470.5 | 283.5 | 515.5 | 2633 |
| target_only | 512 | calibrated | fixed 0.5 | STTC | 2 | 1617 | 217 | 445 | 354 | 2633 |
| target_only | 512 | raw | C-selected / raw default | CD | 2 | 1744 | 132 | 311 | 446 | 2633 |
| target_only | 512 | raw | fixed 0.5 | CD | 2 | 1744 | 132 | 311 | 446 | 2633 |
| target_only | 512 | raw | C-selected / raw default | STTC | 2 | 1658 | 176 | 485.5 | 313.5 | 2633 |
| target_only | 512 | raw | fixed 0.5 | STTC | 2 | 1658 | 176 | 485.5 | 313.5 | 2633 |
| source_only | — | calibrated | C-selected / raw default | CD | 1 | 1749 | 127 | 188 | 569 | 2633 |
| source_only | — | calibrated | fixed 0.5 | CD | 1 | 1783 | 93 | 222 | 535 | 2633 |
| source_only | — | calibrated | C-selected / raw default | STTC | 1 | 1595 | 239 | 237 | 562 | 2633 |
| source_only | — | calibrated | fixed 0.5 | STTC | 1 | 1619 | 215 | 258 | 541 | 2633 |
| source_only | — | raw | C-selected / raw default | CD | 1 | 1791 | 85 | 228 | 529 | 2633 |
| source_only | — | raw | fixed 0.5 | CD | 1 | 1791 | 85 | 228 | 529 | 2633 |
| source_only | — | raw | C-selected / raw default | STTC | 1 | 1630 | 204 | 264 | 535 | 2633 |
| source_only | — | raw | fixed 0.5 | STTC | 1 | 1630 | 204 | 264 | 535 | 2633 |

## 3. Experiment 4 — CPSC locked test, corrected V2 labels

Source: [corrected V2 PER_EPISODE_METRICS.json](../results/phase3b_eval_v2/PER_EPISODE_METRICS.json); cross-checked against [AGGREGATE_METRICS.json](../results/phase3b_eval_v2/AGGREGATE_METRICS.json). The superseded V1 zero-observed-label output is not used. Classes are the frozen V2 CD/STTC annotation-union proxies, not adjudicated diagnoses.

All thresholds are 0.5. Methods retain their recorded identifiers: `source_only`, `head_cost`, `full_cost`, `random_a`, `selective_a`, `target_only_cost`. Budgets 64/128/256 are the recorded nominal label budgets. Each mean uses the nine source-seed (15101–15103) × draw-seed (16101–16103) slots. The three source-only vectors are each aliased over all three draw slots and all budgets; their nine-slot mean equals the saved three-source mean.

| Method | Budget | Class | Runs | TN | FP | FN | TP | N |
|---|---|---|---|---:|---:|---:|---:|---:|
| source_only | 64 | CD | 9 | 2489 + 2/3 | 342 + 1/3 | 284 + 2/3 | 743 + 1/3 | 3860 |
| source_only | 64 | STTC | 9 | 2442 | 476 | 691 + 1/3 | 250 + 2/3 | 3860 |
| source_only | 128 | CD | 9 | 2489 + 2/3 | 342 + 1/3 | 284 + 2/3 | 743 + 1/3 | 3860 |
| source_only | 128 | STTC | 9 | 2442 | 476 | 691 + 1/3 | 250 + 2/3 | 3860 |
| source_only | 256 | CD | 9 | 2489 + 2/3 | 342 + 1/3 | 284 + 2/3 | 743 + 1/3 | 3860 |
| source_only | 256 | STTC | 9 | 2442 | 476 | 691 + 1/3 | 250 + 2/3 | 3860 |
| head_cost | 64 | CD | 9 | 2514 + 2/3 | 317 + 1/3 | 309 + 1/3 | 718 + 2/3 | 3860 |
| head_cost | 64 | STTC | 9 | 2410 + 2/9 | 507 + 7/9 | 633 + 7/9 | 308 + 2/9 | 3860 |
| head_cost | 128 | CD | 9 | 2512 + 8/9 | 319 + 1/9 | 307 + 1/3 | 720 + 2/3 | 3860 |
| head_cost | 128 | STTC | 9 | 2435 + 4/9 | 482 + 5/9 | 660 | 282 | 3860 |
| head_cost | 256 | CD | 9 | 2520 + 1/9 | 311 + 8/9 | 312 + 2/3 | 715 + 1/3 | 3860 |
| head_cost | 256 | STTC | 9 | 2414 + 2/3 | 503 + 1/3 | 639 + 2/9 | 302 + 7/9 | 3860 |
| full_cost | 64 | CD | 9 | 2565 + 1/9 | 266 + 8/9 | 290 | 738 | 3860 |
| full_cost | 64 | STTC | 9 | 2409 + 4/9 | 508 + 5/9 | 495 + 1/9 | 446 + 8/9 | 3860 |
| full_cost | 128 | CD | 9 | 2602 + 1/9 | 229 + 8/9 | 283 + 4/9 | 744 + 5/9 | 3860 |
| full_cost | 128 | STTC | 9 | 2534 + 1/3 | 383 + 2/3 | 540 + 5/9 | 401 + 4/9 | 3860 |
| full_cost | 256 | CD | 9 | 2641 + 4/9 | 190 + 5/9 | 284 + 5/9 | 743 + 4/9 | 3860 |
| full_cost | 256 | STTC | 9 | 2536 + 8/9 | 381 + 1/9 | 493 + 5/9 | 448 + 4/9 | 3860 |
| random_a | 64 | CD | 9 | 2561 + 1/3 | 270 + 2/3 | 334 + 2/3 | 693 + 1/3 | 3860 |
| random_a | 64 | STTC | 9 | 2410 + 4/9 | 507 + 5/9 | 572 + 1/9 | 369 + 8/9 | 3860 |
| random_a | 128 | CD | 9 | 2566 + 2/3 | 265 + 1/3 | 320 + 2/9 | 707 + 7/9 | 3860 |
| random_a | 128 | STTC | 9 | 2572 + 1/3 | 345 + 2/3 | 624 + 5/9 | 317 + 4/9 | 3860 |
| random_a | 256 | CD | 9 | 2624 | 208 | 321 + 4/9 | 706 + 5/9 | 3860 |
| random_a | 256 | STTC | 9 | 2560 + 4/9 | 357 + 5/9 | 557 + 4/9 | 384 + 5/9 | 3860 |
| selective_a | 64 | CD | 9 | 2562 + 7/9 | 269 + 2/9 | 333 + 1/9 | 694 + 8/9 | 3860 |
| selective_a | 64 | STTC | 9 | 2395 + 8/9 | 522 + 1/9 | 564 + 8/9 | 377 + 1/9 | 3860 |
| selective_a | 128 | CD | 9 | 2562 | 270 | 321 + 5/9 | 706 + 4/9 | 3860 |
| selective_a | 128 | STTC | 9 | 2559 + 4/9 | 358 + 5/9 | 610 + 1/3 | 331 + 2/3 | 3860 |
| selective_a | 256 | CD | 9 | 2625 + 8/9 | 206 + 1/9 | 321 + 4/9 | 706 + 5/9 | 3860 |
| selective_a | 256 | STTC | 9 | 2542 + 2/9 | 375 + 7/9 | 520 | 422 | 3860 |
| target_only_cost | 64 | CD | 9 | 2710 + 8/9 | 121 + 1/9 | 523 | 505 | 3860 |
| target_only_cost | 64 | STTC | 9 | 2623 + 2/3 | 294 + 1/3 | 664 + 1/3 | 277 + 2/3 | 3860 |
| target_only_cost | 128 | CD | 9 | 2699 | 133 | 490 + 4/9 | 537 + 5/9 | 3860 |
| target_only_cost | 128 | STTC | 9 | 2656 + 4/9 | 261 + 5/9 | 681 | 261 | 3860 |
| target_only_cost | 256 | CD | 9 | 2652 + 1/9 | 179 + 8/9 | 441 + 8/9 | 586 + 1/9 | 3860 |
| target_only_cost | 256 | STTC | 9 | 2603 + 1/3 | 314 + 2/3 | 583 + 5/9 | 358 + 4/9 | 3860 |

## 4. Phase 4C — CPSC confirmatory evaluation

Source: the completed private Phase 4C `ANALYSIS.json`, `per_physical/*/per_label/*/confusion`, expanded through its frozen `episode_aliases`. Its SHA-256 was verified against the original completion receipt before extraction. No private source file, prediction, label array, participant identity or private fingerprint is reproduced here. Scientific context: [PHASE4C_RESULTS.md](phase4/PHASE4C_RESULTS.md).

Threshold 0.5; nine episodes per method/budget. Method identifiers:

- **M1:** original Experiment 4 source-only (three physical predictors aliased across draws/budgets; zero adaptation labels).
- **M2:** original sources + full fine-tuning.
- **M3:** matched new-pipeline plain retrain (lambda = 0) + full fine-tuning.
- **M4:** placebo within-source MMD (lambda = 0.1) + full fine-tuning.
- **M5:** target-aligned MMD (lambda = 0.1) + full fine-tuning.

This is the same CPSC test cohort previously scored in Experiment 4, not an additional independent population.

| Method | Budget | Class | Runs | TN | FP | FN | TP | N |
|---|---|---|---|---:|---:|---:|---:|---:|
| M1 | 64 | CD | 9 | 2489 + 2/3 | 342 + 1/3 | 284 + 2/3 | 743 + 1/3 | 3860 |
| M1 | 64 | STTC | 9 | 2442 | 476 | 691 + 1/3 | 250 + 2/3 | 3860 |
| M1 | 128 | CD | 9 | 2489 + 2/3 | 342 + 1/3 | 284 + 2/3 | 743 + 1/3 | 3860 |
| M1 | 128 | STTC | 9 | 2442 | 476 | 691 + 1/3 | 250 + 2/3 | 3860 |
| M1 | 256 | CD | 9 | 2489 + 2/3 | 342 + 1/3 | 284 + 2/3 | 743 + 1/3 | 3860 |
| M1 | 256 | STTC | 9 | 2442 | 476 | 691 + 1/3 | 250 + 2/3 | 3860 |
| M2 | 64 | CD | 9 | 2565 + 1/9 | 266 + 8/9 | 290 | 738 | 3860 |
| M2 | 64 | STTC | 9 | 2409 + 4/9 | 508 + 5/9 | 495 + 1/9 | 446 + 8/9 | 3860 |
| M2 | 128 | CD | 9 | 2602 + 1/9 | 229 + 8/9 | 283 + 4/9 | 744 + 5/9 | 3860 |
| M2 | 128 | STTC | 9 | 2534 + 1/3 | 383 + 2/3 | 540 + 5/9 | 401 + 4/9 | 3860 |
| M2 | 256 | CD | 9 | 2641 + 4/9 | 190 + 5/9 | 284 + 5/9 | 743 + 4/9 | 3860 |
| M2 | 256 | STTC | 9 | 2536 + 8/9 | 381 + 1/9 | 493 + 5/9 | 448 + 4/9 | 3860 |
| M3 | 64 | CD | 9 | 2567 + 8/9 | 264 + 1/9 | 275 + 2/9 | 752 + 7/9 | 3860 |
| M3 | 64 | STTC | 9 | 2419 | 499 | 485 + 1/3 | 456 + 2/3 | 3860 |
| M3 | 128 | CD | 9 | 2597 + 5/9 | 234 + 4/9 | 256 | 772 | 3860 |
| M3 | 128 | STTC | 9 | 2547 | 371 | 522 + 5/9 | 419 + 4/9 | 3860 |
| M3 | 256 | CD | 9 | 2628 + 1/9 | 203 + 8/9 | 247 + 5/9 | 780 + 4/9 | 3860 |
| M3 | 256 | STTC | 9 | 2528 + 4/9 | 389 + 5/9 | 488 | 454 | 3860 |
| M4 | 64 | CD | 9 | 2573 + 2/9 | 258 + 7/9 | 286 + 1/9 | 741 + 8/9 | 3860 |
| M4 | 64 | STTC | 9 | 2456 + 7/9 | 461 + 2/9 | 477 + 1/3 | 464 + 2/3 | 3860 |
| M4 | 128 | CD | 9 | 2600 + 2/3 | 231 + 1/3 | 280 + 8/9 | 747 + 1/9 | 3860 |
| M4 | 128 | STTC | 9 | 2595 + 5/9 | 322 + 4/9 | 540 + 1/9 | 401 + 8/9 | 3860 |
| M4 | 256 | CD | 9 | 2621 + 1/3 | 210 + 2/3 | 258 + 4/9 | 769 + 5/9 | 3860 |
| M4 | 256 | STTC | 9 | 2561 + 7/9 | 356 + 2/9 | 466 + 7/9 | 475 + 2/9 | 3860 |
| M5 | 64 | CD | 9 | 2576 + 4/9 | 255 + 5/9 | 267 + 2/3 | 760 + 1/3 | 3860 |
| M5 | 64 | STTC | 9 | 2425 | 493 | 443 + 2/9 | 498 + 7/9 | 3860 |
| M5 | 128 | CD | 9 | 2609 + 7/9 | 222 + 2/9 | 256 + 1/9 | 771 + 8/9 | 3860 |
| M5 | 128 | STTC | 9 | 2573 + 7/9 | 344 + 2/9 | 512 + 1/3 | 429 + 2/3 | 3860 |
| M5 | 256 | CD | 9 | 2637 + 1/3 | 194 + 2/3 | 242 + 2/3 | 785 + 1/3 | 3860 |
| M5 | 256 | STTC | 9 | 2536 + 2/3 | 381 + 1/3 | 446 + 1/3 | 495 + 2/3 | 3860 |

## 5. Phase 5 — Georgia confirmatory replication

Source: the completed private Phase 5 Georgia `ANALYSIS.json`, `per_physical/*/per_label/*/confusion`, expanded through its frozen `episode_aliases`. Its SHA-256 was verified against the original completion receipt. No new outcome read or prediction computation was necessary. Scientific context: [PHASE5_GEORGIA_RESULTS.md](phase5/PHASE5_GEORGIA_RESULTS.md).

Threshold 0.5; 30 slots = ten source seeds (15101–15110) × three fixed draw seeds (16101–16103), at each budget. Methods: **a** = plain-retrain source-only; **b** = plain retrain + full FT; **c** = placebo MMD + full FT. Method a has ten physical source predictors aliased across draws/budgets and uses zero target labels. The 30-slot mean gives each source equal weight.

| Method | Budget | Class | Runs | TN | FP | FN | TP | N |
|---|---|---|---|---:|---:|---:|---:|---:|
| a | 64 | CD | 30 | 2755.2 | 605.8 | 195.2 | 524.8 | 4081 |
| a | 64 | STTC | 30 | 2088.1 | 216.9 | 1185.2 | 590.8 | 4081 |
| a | 128 | CD | 30 | 2755.2 | 605.8 | 195.2 | 524.8 | 4081 |
| a | 128 | STTC | 30 | 2088.1 | 216.9 | 1185.2 | 590.8 | 4081 |
| a | 256 | CD | 30 | 2755.2 | 605.8 | 195.2 | 524.8 | 4081 |
| a | 256 | STTC | 30 | 2088.1 | 216.9 | 1185.2 | 590.8 | 4081 |
| b | 64 | CD | 30 | 2914 + 11/30 | 446 + 19/30 | 220 + 1/30 | 499 + 29/30 | 4081 |
| b | 64 | STTC | 30 | 1668 + 11/15 | 636 + 4/15 | 539.1 | 1236.9 | 4081 |
| b | 128 | CD | 30 | 2988 + 1/3 | 372 + 2/3 | 248.9 | 471.1 | 4081 |
| b | 128 | STTC | 30 | 1693 + 1/3 | 611 + 2/3 | 559 + 11/30 | 1216 + 19/30 | 4081 |
| b | 256 | CD | 30 | 3069 + 1/6 | 291 + 5/6 | 295 + 1/15 | 424 + 14/15 | 4081 |
| b | 256 | STTC | 30 | 1725 + 2/3 | 579 + 1/3 | 521 + 23/30 | 1254 + 7/30 | 4081 |
| c | 64 | CD | 30 | 2927.5 | 433.5 | 218 + 4/15 | 501 + 11/15 | 4081 |
| c | 64 | STTC | 30 | 1662 + 29/30 | 642 + 1/30 | 520.7 | 1255.3 | 4081 |
| c | 128 | CD | 30 | 2986 | 375 | 236.5 | 483.5 | 4081 |
| c | 128 | STTC | 30 | 1697 + 7/30 | 607 + 23/30 | 538 + 1/15 | 1237 + 14/15 | 4081 |
| c | 256 | CD | 30 | 3075 + 1/30 | 285 + 29/30 | 289.5 | 430.5 | 4081 |
| c | 256 | STTC | 30 | 1758 + 23/30 | 546 + 7/30 | 524 + 1/6 | 1251 + 5/6 | 4081 |

## Appendix A — Experiment 2 per-draw counts

Each fitted draw retains its actual historical threshold. A dash for source-only draw/budget denotes a single unchanged source predictor, not a missing result. Raw-default policy duplication is retained explicitly.

| Method | Budget | Draw seed | Scores | Threshold policy | Class | Threshold | TN | FP | FN | TP | N |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| full_cost | 128 | 4101 | calibrated | C-selected / raw default | CD | 0.43 | 1756 | 120 | 200 | 557 | 2633 |
| full_cost | 128 | 4101 | calibrated | fixed 0.5 | CD | 0.5 | 1783 | 93 | 220 | 537 | 2633 |
| full_cost | 128 | 4101 | calibrated | C-selected / raw default | STTC | 0.56 | 1657 | 177 | 283 | 516 | 2633 |
| full_cost | 128 | 4101 | calibrated | fixed 0.5 | STTC | 0.5 | 1597 | 237 | 247 | 552 | 2633 |
| full_cost | 128 | 4101 | raw | C-selected / raw default | CD | 0.5 | 1779 | 97 | 214 | 543 | 2633 |
| full_cost | 128 | 4101 | raw | fixed 0.5 | CD | 0.5 | 1779 | 97 | 214 | 543 | 2633 |
| full_cost | 128 | 4101 | raw | C-selected / raw default | STTC | 0.5 | 1657 | 177 | 285 | 514 | 2633 |
| full_cost | 128 | 4101 | raw | fixed 0.5 | STTC | 0.5 | 1657 | 177 | 285 | 514 | 2633 |
| full_f | 128 | 4101 | calibrated | C-selected / raw default | CD | 0.39 | 1757 | 119 | 196 | 561 | 2633 |
| full_f | 128 | 4101 | calibrated | fixed 0.5 | CD | 0.5 | 1784 | 92 | 222 | 535 | 2633 |
| full_f | 128 | 4101 | calibrated | C-selected / raw default | STTC | 0.47 | 1588 | 246 | 230 | 569 | 2633 |
| full_f | 128 | 4101 | calibrated | fixed 0.5 | STTC | 0.5 | 1618 | 216 | 252 | 547 | 2633 |
| full_f | 128 | 4101 | raw | C-selected / raw default | CD | 0.5 | 1794 | 82 | 230 | 527 | 2633 |
| full_f | 128 | 4101 | raw | fixed 0.5 | CD | 0.5 | 1794 | 82 | 230 | 527 | 2633 |
| full_f | 128 | 4101 | raw | C-selected / raw default | STTC | 0.5 | 1646 | 188 | 279 | 520 | 2633 |
| full_f | 128 | 4101 | raw | fixed 0.5 | STTC | 0.5 | 1646 | 188 | 279 | 520 | 2633 |
| head_cost | 128 | 4101 | calibrated | C-selected / raw default | CD | 0.37 | 1749 | 127 | 188 | 569 | 2633 |
| head_cost | 128 | 4101 | calibrated | fixed 0.5 | CD | 0.5 | 1783 | 93 | 222 | 535 | 2633 |
| head_cost | 128 | 4101 | calibrated | C-selected / raw default | STTC | 0.47 | 1595 | 239 | 237 | 562 | 2633 |
| head_cost | 128 | 4101 | calibrated | fixed 0.5 | STTC | 0.5 | 1619 | 215 | 258 | 541 | 2633 |
| head_cost | 128 | 4101 | raw | C-selected / raw default | CD | 0.5 | 1791 | 85 | 228 | 529 | 2633 |
| head_cost | 128 | 4101 | raw | fixed 0.5 | CD | 0.5 | 1791 | 85 | 228 | 529 | 2633 |
| head_cost | 128 | 4101 | raw | C-selected / raw default | STTC | 0.5 | 1630 | 204 | 264 | 535 | 2633 |
| head_cost | 128 | 4101 | raw | fixed 0.5 | STTC | 0.5 | 1630 | 204 | 264 | 535 | 2633 |
| head_f | 128 | 4101 | calibrated | C-selected / raw default | CD | 0.37 | 1749 | 127 | 188 | 569 | 2633 |
| head_f | 128 | 4101 | calibrated | fixed 0.5 | CD | 0.5 | 1783 | 93 | 222 | 535 | 2633 |
| head_f | 128 | 4101 | calibrated | C-selected / raw default | STTC | 0.46 | 1587 | 247 | 230 | 569 | 2633 |
| head_f | 128 | 4101 | calibrated | fixed 0.5 | STTC | 0.5 | 1614 | 220 | 258 | 541 | 2633 |
| head_f | 128 | 4101 | raw | C-selected / raw default | CD | 0.5 | 1790 | 86 | 227 | 530 | 2633 |
| head_f | 128 | 4101 | raw | fixed 0.5 | CD | 0.5 | 1790 | 86 | 227 | 530 | 2633 |
| head_f | 128 | 4101 | raw | C-selected / raw default | STTC | 0.5 | 1639 | 195 | 276 | 523 | 2633 |
| head_f | 128 | 4101 | raw | fixed 0.5 | STTC | 0.5 | 1639 | 195 | 276 | 523 | 2633 |
| random_a | 128 | 4101 | calibrated | C-selected / raw default | CD | 0.38 | 1752 | 124 | 193 | 564 | 2633 |
| random_a | 128 | 4101 | calibrated | fixed 0.5 | CD | 0.5 | 1785 | 91 | 221 | 536 | 2633 |
| random_a | 128 | 4101 | calibrated | C-selected / raw default | STTC | 0.47 | 1594 | 240 | 237 | 562 | 2633 |
| random_a | 128 | 4101 | calibrated | fixed 0.5 | STTC | 0.5 | 1619 | 215 | 254 | 545 | 2633 |
| random_a | 128 | 4101 | raw | C-selected / raw default | CD | 0.5 | 1793 | 83 | 229 | 528 | 2633 |
| random_a | 128 | 4101 | raw | fixed 0.5 | CD | 0.5 | 1793 | 83 | 229 | 528 | 2633 |
| random_a | 128 | 4101 | raw | C-selected / raw default | STTC | 0.5 | 1642 | 192 | 274 | 525 | 2633 |
| random_a | 128 | 4101 | raw | fixed 0.5 | STTC | 0.5 | 1642 | 192 | 274 | 525 | 2633 |
| selective_a | 128 | 4101 | calibrated | C-selected / raw default | CD | 0.37 | 1749 | 127 | 188 | 569 | 2633 |
| selective_a | 128 | 4101 | calibrated | fixed 0.5 | CD | 0.5 | 1783 | 93 | 222 | 535 | 2633 |
| selective_a | 128 | 4101 | calibrated | C-selected / raw default | STTC | 0.47 | 1595 | 239 | 237 | 562 | 2633 |
| selective_a | 128 | 4101 | calibrated | fixed 0.5 | STTC | 0.5 | 1619 | 215 | 258 | 541 | 2633 |
| selective_a | 128 | 4101 | raw | C-selected / raw default | CD | 0.5 | 1791 | 85 | 228 | 529 | 2633 |
| selective_a | 128 | 4101 | raw | fixed 0.5 | CD | 0.5 | 1791 | 85 | 228 | 529 | 2633 |
| selective_a | 128 | 4101 | raw | C-selected / raw default | STTC | 0.5 | 1630 | 204 | 264 | 535 | 2633 |
| selective_a | 128 | 4101 | raw | fixed 0.5 | STTC | 0.5 | 1630 | 204 | 264 | 535 | 2633 |
| selective_b | 128 | 4101 | calibrated | C-selected / raw default | CD | 0.37 | 1748 | 128 | 188 | 569 | 2633 |
| selective_b | 128 | 4101 | calibrated | fixed 0.5 | CD | 0.5 | 1784 | 92 | 221 | 536 | 2633 |
| selective_b | 128 | 4101 | calibrated | C-selected / raw default | STTC | 0.48 | 1605 | 229 | 240 | 559 | 2633 |
| selective_b | 128 | 4101 | calibrated | fixed 0.5 | STTC | 0.5 | 1621 | 213 | 254 | 545 | 2633 |
| selective_b | 128 | 4101 | raw | C-selected / raw default | CD | 0.5 | 1793 | 83 | 227 | 530 | 2633 |
| selective_b | 128 | 4101 | raw | fixed 0.5 | CD | 0.5 | 1793 | 83 | 227 | 530 | 2633 |
| selective_b | 128 | 4101 | raw | C-selected / raw default | STTC | 0.5 | 1655 | 179 | 291 | 508 | 2633 |
| selective_b | 128 | 4101 | raw | fixed 0.5 | STTC | 0.5 | 1655 | 179 | 291 | 508 | 2633 |
| target_only | 128 | 4101 | calibrated | C-selected / raw default | CD | 0.23 | 1586 | 290 | 259 | 498 | 2633 |
| target_only | 128 | 4101 | calibrated | fixed 0.5 | CD | 0.5 | 1749 | 127 | 360 | 397 | 2633 |
| target_only | 128 | 4101 | calibrated | C-selected / raw default | STTC | 0.25 | 500 | 1334 | 106 | 693 | 2633 |
| target_only | 128 | 4101 | calibrated | fixed 0.5 | STTC | 0.5 | 1777 | 57 | 721 | 78 | 2633 |
| target_only | 128 | 4101 | raw | C-selected / raw default | CD | 0.5 | 1777 | 99 | 384 | 373 | 2633 |
| target_only | 128 | 4101 | raw | fixed 0.5 | CD | 0.5 | 1777 | 99 | 384 | 373 | 2633 |
| target_only | 128 | 4101 | raw | C-selected / raw default | STTC | 0.5 | 1820 | 14 | 782 | 17 | 2633 |
| target_only | 128 | 4101 | raw | fixed 0.5 | STTC | 0.5 | 1820 | 14 | 782 | 17 | 2633 |
| full_cost | 512 | 4101 | calibrated | C-selected / raw default | CD | 0.37 | 1749 | 127 | 188 | 569 | 2633 |
| full_cost | 512 | 4101 | calibrated | fixed 0.5 | CD | 0.5 | 1783 | 93 | 222 | 535 | 2633 |
| full_cost | 512 | 4101 | calibrated | C-selected / raw default | STTC | 0.47 | 1595 | 239 | 237 | 562 | 2633 |
| full_cost | 512 | 4101 | calibrated | fixed 0.5 | STTC | 0.5 | 1619 | 215 | 258 | 541 | 2633 |
| full_cost | 512 | 4101 | raw | C-selected / raw default | CD | 0.5 | 1791 | 85 | 228 | 529 | 2633 |
| full_cost | 512 | 4101 | raw | fixed 0.5 | CD | 0.5 | 1791 | 85 | 228 | 529 | 2633 |
| full_cost | 512 | 4101 | raw | C-selected / raw default | STTC | 0.5 | 1630 | 204 | 264 | 535 | 2633 |
| full_cost | 512 | 4101 | raw | fixed 0.5 | STTC | 0.5 | 1630 | 204 | 264 | 535 | 2633 |
| full_f | 512 | 4101 | calibrated | C-selected / raw default | CD | 0.51 | 1782 | 94 | 219 | 538 | 2633 |
| full_f | 512 | 4101 | calibrated | fixed 0.5 | CD | 0.5 | 1780 | 96 | 217 | 540 | 2633 |
| full_f | 512 | 4101 | calibrated | C-selected / raw default | STTC | 0.31 | 1385 | 449 | 133 | 666 | 2633 |
| full_f | 512 | 4101 | calibrated | fixed 0.5 | STTC | 0.5 | 1602 | 232 | 246 | 553 | 2633 |
| full_f | 512 | 4101 | raw | C-selected / raw default | CD | 0.5 | 1748 | 128 | 191 | 566 | 2633 |
| full_f | 512 | 4101 | raw | fixed 0.5 | CD | 0.5 | 1748 | 128 | 191 | 566 | 2633 |
| full_f | 512 | 4101 | raw | C-selected / raw default | STTC | 0.5 | 1663 | 171 | 283 | 516 | 2633 |
| full_f | 512 | 4101 | raw | fixed 0.5 | STTC | 0.5 | 1663 | 171 | 283 | 516 | 2633 |
| head_cost | 512 | 4101 | calibrated | C-selected / raw default | CD | 0.37 | 1749 | 127 | 188 | 569 | 2633 |
| head_cost | 512 | 4101 | calibrated | fixed 0.5 | CD | 0.5 | 1784 | 92 | 222 | 535 | 2633 |
| head_cost | 512 | 4101 | calibrated | C-selected / raw default | STTC | 0.46 | 1587 | 247 | 230 | 569 | 2633 |
| head_cost | 512 | 4101 | calibrated | fixed 0.5 | STTC | 0.5 | 1613 | 221 | 258 | 541 | 2633 |
| head_cost | 512 | 4101 | raw | C-selected / raw default | CD | 0.5 | 1789 | 87 | 224 | 533 | 2633 |
| head_cost | 512 | 4101 | raw | fixed 0.5 | CD | 0.5 | 1789 | 87 | 224 | 533 | 2633 |
| head_cost | 512 | 4101 | raw | C-selected / raw default | STTC | 0.5 | 1642 | 192 | 282 | 517 | 2633 |
| head_cost | 512 | 4101 | raw | fixed 0.5 | STTC | 0.5 | 1642 | 192 | 282 | 517 | 2633 |
| head_f | 512 | 4101 | calibrated | C-selected / raw default | CD | 0.37 | 1749 | 127 | 188 | 569 | 2633 |
| head_f | 512 | 4101 | calibrated | fixed 0.5 | CD | 0.5 | 1783 | 93 | 222 | 535 | 2633 |
| head_f | 512 | 4101 | calibrated | C-selected / raw default | STTC | 0.47 | 1595 | 239 | 237 | 562 | 2633 |
| head_f | 512 | 4101 | calibrated | fixed 0.5 | STTC | 0.5 | 1619 | 215 | 258 | 541 | 2633 |
| head_f | 512 | 4101 | raw | C-selected / raw default | CD | 0.5 | 1791 | 85 | 228 | 529 | 2633 |
| head_f | 512 | 4101 | raw | fixed 0.5 | CD | 0.5 | 1791 | 85 | 228 | 529 | 2633 |
| head_f | 512 | 4101 | raw | C-selected / raw default | STTC | 0.5 | 1630 | 204 | 264 | 535 | 2633 |
| head_f | 512 | 4101 | raw | fixed 0.5 | STTC | 0.5 | 1630 | 204 | 264 | 535 | 2633 |
| random_a | 512 | 4101 | calibrated | C-selected / raw default | CD | 0.37 | 1749 | 127 | 188 | 569 | 2633 |
| random_a | 512 | 4101 | calibrated | fixed 0.5 | CD | 0.5 | 1783 | 93 | 222 | 535 | 2633 |
| random_a | 512 | 4101 | calibrated | C-selected / raw default | STTC | 0.47 | 1595 | 239 | 237 | 562 | 2633 |
| random_a | 512 | 4101 | calibrated | fixed 0.5 | STTC | 0.5 | 1619 | 215 | 258 | 541 | 2633 |
| random_a | 512 | 4101 | raw | C-selected / raw default | CD | 0.5 | 1791 | 85 | 228 | 529 | 2633 |
| random_a | 512 | 4101 | raw | fixed 0.5 | CD | 0.5 | 1791 | 85 | 228 | 529 | 2633 |
| random_a | 512 | 4101 | raw | C-selected / raw default | STTC | 0.5 | 1630 | 204 | 264 | 535 | 2633 |
| random_a | 512 | 4101 | raw | fixed 0.5 | STTC | 0.5 | 1630 | 204 | 264 | 535 | 2633 |
| selective_a | 512 | 4101 | calibrated | C-selected / raw default | CD | 0.42 | 1755 | 121 | 196 | 561 | 2633 |
| selective_a | 512 | 4101 | calibrated | fixed 0.5 | CD | 0.5 | 1780 | 96 | 217 | 540 | 2633 |
| selective_a | 512 | 4101 | calibrated | C-selected / raw default | STTC | 0.4 | 1524 | 310 | 191 | 608 | 2633 |
| selective_a | 512 | 4101 | calibrated | fixed 0.5 | STTC | 0.5 | 1617 | 217 | 260 | 539 | 2633 |
| selective_a | 512 | 4101 | raw | C-selected / raw default | CD | 0.5 | 1769 | 107 | 204 | 553 | 2633 |
| selective_a | 512 | 4101 | raw | fixed 0.5 | CD | 0.5 | 1769 | 107 | 204 | 553 | 2633 |
| selective_a | 512 | 4101 | raw | C-selected / raw default | STTC | 0.5 | 1694 | 140 | 332 | 467 | 2633 |
| selective_a | 512 | 4101 | raw | fixed 0.5 | STTC | 0.5 | 1694 | 140 | 332 | 467 | 2633 |
| selective_b | 512 | 4101 | calibrated | C-selected / raw default | CD | 0.41 | 1758 | 118 | 191 | 566 | 2633 |
| selective_b | 512 | 4101 | calibrated | fixed 0.5 | CD | 0.5 | 1782 | 94 | 217 | 540 | 2633 |
| selective_b | 512 | 4101 | calibrated | C-selected / raw default | STTC | 0.4 | 1526 | 308 | 188 | 611 | 2633 |
| selective_b | 512 | 4101 | calibrated | fixed 0.5 | STTC | 0.5 | 1629 | 205 | 259 | 540 | 2633 |
| selective_b | 512 | 4101 | raw | C-selected / raw default | CD | 0.5 | 1770 | 106 | 204 | 553 | 2633 |
| selective_b | 512 | 4101 | raw | fixed 0.5 | CD | 0.5 | 1770 | 106 | 204 | 553 | 2633 |
| selective_b | 512 | 4101 | raw | C-selected / raw default | STTC | 0.5 | 1714 | 120 | 358 | 441 | 2633 |
| selective_b | 512 | 4101 | raw | fixed 0.5 | STTC | 0.5 | 1714 | 120 | 358 | 441 | 2633 |
| target_only | 512 | 4101 | calibrated | C-selected / raw default | CD | 0.37 | 1716 | 160 | 289 | 468 | 2633 |
| target_only | 512 | 4101 | calibrated | fixed 0.5 | CD | 0.5 | 1769 | 107 | 322 | 435 | 2633 |
| target_only | 512 | 4101 | calibrated | C-selected / raw default | STTC | 0.34 | 1341 | 493 | 286 | 513 | 2633 |
| target_only | 512 | 4101 | calibrated | fixed 0.5 | STTC | 0.5 | 1624 | 210 | 468 | 331 | 2633 |
| target_only | 512 | 4101 | raw | C-selected / raw default | CD | 0.5 | 1792 | 84 | 345 | 412 | 2633 |
| target_only | 512 | 4101 | raw | fixed 0.5 | CD | 0.5 | 1792 | 84 | 345 | 412 | 2633 |
| target_only | 512 | 4101 | raw | C-selected / raw default | STTC | 0.5 | 1710 | 124 | 552 | 247 | 2633 |
| target_only | 512 | 4101 | raw | fixed 0.5 | STTC | 0.5 | 1710 | 124 | 552 | 247 | 2633 |
| full_cost | 128 | 4102 | calibrated | C-selected / raw default | CD | 0.37 | 1749 | 127 | 188 | 569 | 2633 |
| full_cost | 128 | 4102 | calibrated | fixed 0.5 | CD | 0.5 | 1783 | 93 | 222 | 535 | 2633 |
| full_cost | 128 | 4102 | calibrated | C-selected / raw default | STTC | 0.47 | 1595 | 239 | 237 | 562 | 2633 |
| full_cost | 128 | 4102 | calibrated | fixed 0.5 | STTC | 0.5 | 1619 | 215 | 258 | 541 | 2633 |
| full_cost | 128 | 4102 | raw | C-selected / raw default | CD | 0.5 | 1791 | 85 | 228 | 529 | 2633 |
| full_cost | 128 | 4102 | raw | fixed 0.5 | CD | 0.5 | 1791 | 85 | 228 | 529 | 2633 |
| full_cost | 128 | 4102 | raw | C-selected / raw default | STTC | 0.5 | 1630 | 204 | 264 | 535 | 2633 |
| full_cost | 128 | 4102 | raw | fixed 0.5 | STTC | 0.5 | 1630 | 204 | 264 | 535 | 2633 |
| full_f | 128 | 4102 | calibrated | C-selected / raw default | CD | 0.38 | 1752 | 124 | 192 | 565 | 2633 |
| full_f | 128 | 4102 | calibrated | fixed 0.5 | CD | 0.5 | 1785 | 91 | 221 | 536 | 2633 |
| full_f | 128 | 4102 | calibrated | C-selected / raw default | STTC | 0.47 | 1589 | 245 | 232 | 567 | 2633 |
| full_f | 128 | 4102 | calibrated | fixed 0.5 | STTC | 0.5 | 1617 | 217 | 251 | 548 | 2633 |
| full_f | 128 | 4102 | raw | C-selected / raw default | CD | 0.5 | 1793 | 83 | 232 | 525 | 2633 |
| full_f | 128 | 4102 | raw | fixed 0.5 | CD | 0.5 | 1793 | 83 | 232 | 525 | 2633 |
| full_f | 128 | 4102 | raw | C-selected / raw default | STTC | 0.5 | 1635 | 199 | 275 | 524 | 2633 |
| full_f | 128 | 4102 | raw | fixed 0.5 | STTC | 0.5 | 1635 | 199 | 275 | 524 | 2633 |
| head_cost | 128 | 4102 | calibrated | C-selected / raw default | CD | 0.37 | 1749 | 127 | 188 | 569 | 2633 |
| head_cost | 128 | 4102 | calibrated | fixed 0.5 | CD | 0.5 | 1783 | 93 | 222 | 535 | 2633 |
| head_cost | 128 | 4102 | calibrated | C-selected / raw default | STTC | 0.46 | 1587 | 247 | 230 | 569 | 2633 |
| head_cost | 128 | 4102 | calibrated | fixed 0.5 | STTC | 0.5 | 1614 | 220 | 258 | 541 | 2633 |
| head_cost | 128 | 4102 | raw | C-selected / raw default | CD | 0.5 | 1790 | 86 | 228 | 529 | 2633 |
| head_cost | 128 | 4102 | raw | fixed 0.5 | CD | 0.5 | 1790 | 86 | 228 | 529 | 2633 |
| head_cost | 128 | 4102 | raw | C-selected / raw default | STTC | 0.5 | 1637 | 197 | 275 | 524 | 2633 |
| head_cost | 128 | 4102 | raw | fixed 0.5 | STTC | 0.5 | 1637 | 197 | 275 | 524 | 2633 |
| head_f | 128 | 4102 | calibrated | C-selected / raw default | CD | 0.37 | 1749 | 127 | 188 | 569 | 2633 |
| head_f | 128 | 4102 | calibrated | fixed 0.5 | CD | 0.5 | 1783 | 93 | 222 | 535 | 2633 |
| head_f | 128 | 4102 | calibrated | C-selected / raw default | STTC | 0.47 | 1595 | 239 | 237 | 562 | 2633 |
| head_f | 128 | 4102 | calibrated | fixed 0.5 | STTC | 0.5 | 1619 | 215 | 258 | 541 | 2633 |
| head_f | 128 | 4102 | raw | C-selected / raw default | CD | 0.5 | 1791 | 85 | 228 | 529 | 2633 |
| head_f | 128 | 4102 | raw | fixed 0.5 | CD | 0.5 | 1791 | 85 | 228 | 529 | 2633 |
| head_f | 128 | 4102 | raw | C-selected / raw default | STTC | 0.5 | 1630 | 204 | 264 | 535 | 2633 |
| head_f | 128 | 4102 | raw | fixed 0.5 | STTC | 0.5 | 1630 | 204 | 264 | 535 | 2633 |
| random_a | 128 | 4102 | calibrated | C-selected / raw default | CD | 0.37 | 1749 | 127 | 188 | 569 | 2633 |
| random_a | 128 | 4102 | calibrated | fixed 0.5 | CD | 0.5 | 1783 | 93 | 222 | 535 | 2633 |
| random_a | 128 | 4102 | calibrated | C-selected / raw default | STTC | 0.47 | 1595 | 239 | 237 | 562 | 2633 |
| random_a | 128 | 4102 | calibrated | fixed 0.5 | STTC | 0.5 | 1619 | 215 | 258 | 541 | 2633 |
| random_a | 128 | 4102 | raw | C-selected / raw default | CD | 0.5 | 1791 | 85 | 228 | 529 | 2633 |
| random_a | 128 | 4102 | raw | fixed 0.5 | CD | 0.5 | 1791 | 85 | 228 | 529 | 2633 |
| random_a | 128 | 4102 | raw | C-selected / raw default | STTC | 0.5 | 1630 | 204 | 264 | 535 | 2633 |
| random_a | 128 | 4102 | raw | fixed 0.5 | STTC | 0.5 | 1630 | 204 | 264 | 535 | 2633 |
| selective_a | 128 | 4102 | calibrated | C-selected / raw default | CD | 0.37 | 1749 | 127 | 188 | 569 | 2633 |
| selective_a | 128 | 4102 | calibrated | fixed 0.5 | CD | 0.5 | 1783 | 93 | 222 | 535 | 2633 |
| selective_a | 128 | 4102 | calibrated | C-selected / raw default | STTC | 0.47 | 1595 | 239 | 237 | 562 | 2633 |
| selective_a | 128 | 4102 | calibrated | fixed 0.5 | STTC | 0.5 | 1619 | 215 | 258 | 541 | 2633 |
| selective_a | 128 | 4102 | raw | C-selected / raw default | CD | 0.5 | 1791 | 85 | 228 | 529 | 2633 |
| selective_a | 128 | 4102 | raw | fixed 0.5 | CD | 0.5 | 1791 | 85 | 228 | 529 | 2633 |
| selective_a | 128 | 4102 | raw | C-selected / raw default | STTC | 0.5 | 1630 | 204 | 264 | 535 | 2633 |
| selective_a | 128 | 4102 | raw | fixed 0.5 | STTC | 0.5 | 1630 | 204 | 264 | 535 | 2633 |
| selective_b | 128 | 4102 | calibrated | C-selected / raw default | CD | 0.37 | 1749 | 127 | 188 | 569 | 2633 |
| selective_b | 128 | 4102 | calibrated | fixed 0.5 | CD | 0.5 | 1783 | 93 | 222 | 535 | 2633 |
| selective_b | 128 | 4102 | calibrated | C-selected / raw default | STTC | 0.47 | 1595 | 239 | 237 | 562 | 2633 |
| selective_b | 128 | 4102 | calibrated | fixed 0.5 | STTC | 0.5 | 1619 | 215 | 258 | 541 | 2633 |
| selective_b | 128 | 4102 | raw | C-selected / raw default | CD | 0.5 | 1791 | 85 | 228 | 529 | 2633 |
| selective_b | 128 | 4102 | raw | fixed 0.5 | CD | 0.5 | 1791 | 85 | 228 | 529 | 2633 |
| selective_b | 128 | 4102 | raw | C-selected / raw default | STTC | 0.5 | 1630 | 204 | 264 | 535 | 2633 |
| selective_b | 128 | 4102 | raw | fixed 0.5 | STTC | 0.5 | 1630 | 204 | 264 | 535 | 2633 |
| target_only | 128 | 4102 | calibrated | C-selected / raw default | CD | 0.29 | 1669 | 207 | 287 | 470 | 2633 |
| target_only | 128 | 4102 | calibrated | fixed 0.5 | CD | 0.5 | 1769 | 107 | 400 | 357 | 2633 |
| target_only | 128 | 4102 | calibrated | C-selected / raw default | STTC | 0.23 | 343 | 1491 | 40 | 759 | 2633 |
| target_only | 128 | 4102 | calibrated | fixed 0.5 | STTC | 0.5 | 1771 | 63 | 675 | 124 | 2633 |
| target_only | 128 | 4102 | raw | C-selected / raw default | CD | 0.5 | 1838 | 38 | 502 | 255 | 2633 |
| target_only | 128 | 4102 | raw | fixed 0.5 | CD | 0.5 | 1838 | 38 | 502 | 255 | 2633 |
| target_only | 128 | 4102 | raw | C-selected / raw default | STTC | 0.5 | 1449 | 385 | 422 | 377 | 2633 |
| target_only | 128 | 4102 | raw | fixed 0.5 | STTC | 0.5 | 1449 | 385 | 422 | 377 | 2633 |
| full_cost | 512 | 4102 | calibrated | C-selected / raw default | CD | 0.34 | 1732 | 144 | 182 | 575 | 2633 |
| full_cost | 512 | 4102 | calibrated | fixed 0.5 | CD | 0.5 | 1789 | 87 | 235 | 522 | 2633 |
| full_cost | 512 | 4102 | calibrated | C-selected / raw default | STTC | 0.46 | 1547 | 287 | 196 | 603 | 2633 |
| full_cost | 512 | 4102 | calibrated | fixed 0.5 | STTC | 0.5 | 1574 | 260 | 219 | 580 | 2633 |
| full_cost | 512 | 4102 | raw | C-selected / raw default | CD | 0.5 | 1788 | 88 | 229 | 528 | 2633 |
| full_cost | 512 | 4102 | raw | fixed 0.5 | CD | 0.5 | 1788 | 88 | 229 | 528 | 2633 |
| full_cost | 512 | 4102 | raw | C-selected / raw default | STTC | 0.5 | 1617 | 217 | 251 | 548 | 2633 |
| full_cost | 512 | 4102 | raw | fixed 0.5 | STTC | 0.5 | 1617 | 217 | 251 | 548 | 2633 |
| full_f | 512 | 4102 | calibrated | C-selected / raw default | CD | 0.32 | 1713 | 163 | 177 | 580 | 2633 |
| full_f | 512 | 4102 | calibrated | fixed 0.5 | CD | 0.5 | 1787 | 89 | 226 | 531 | 2633 |
| full_f | 512 | 4102 | calibrated | C-selected / raw default | STTC | 0.46 | 1562 | 272 | 207 | 592 | 2633 |
| full_f | 512 | 4102 | calibrated | fixed 0.5 | STTC | 0.5 | 1591 | 243 | 232 | 567 | 2633 |
| full_f | 512 | 4102 | raw | C-selected / raw default | CD | 0.5 | 1771 | 105 | 217 | 540 | 2633 |
| full_f | 512 | 4102 | raw | fixed 0.5 | CD | 0.5 | 1771 | 105 | 217 | 540 | 2633 |
| full_f | 512 | 4102 | raw | C-selected / raw default | STTC | 0.5 | 1643 | 191 | 275 | 524 | 2633 |
| full_f | 512 | 4102 | raw | fixed 0.5 | STTC | 0.5 | 1643 | 191 | 275 | 524 | 2633 |
| head_cost | 512 | 4102 | calibrated | C-selected / raw default | CD | 0.37 | 1749 | 127 | 188 | 569 | 2633 |
| head_cost | 512 | 4102 | calibrated | fixed 0.5 | CD | 0.5 | 1783 | 93 | 222 | 535 | 2633 |
| head_cost | 512 | 4102 | calibrated | C-selected / raw default | STTC | 0.46 | 1587 | 247 | 230 | 569 | 2633 |
| head_cost | 512 | 4102 | calibrated | fixed 0.5 | STTC | 0.5 | 1615 | 219 | 258 | 541 | 2633 |
| head_cost | 512 | 4102 | raw | C-selected / raw default | CD | 0.5 | 1790 | 86 | 225 | 532 | 2633 |
| head_cost | 512 | 4102 | raw | fixed 0.5 | CD | 0.5 | 1790 | 86 | 225 | 532 | 2633 |
| head_cost | 512 | 4102 | raw | C-selected / raw default | STTC | 0.5 | 1635 | 199 | 271 | 528 | 2633 |
| head_cost | 512 | 4102 | raw | fixed 0.5 | STTC | 0.5 | 1635 | 199 | 271 | 528 | 2633 |
| head_f | 512 | 4102 | calibrated | C-selected / raw default | CD | 0.5 | 1784 | 92 | 222 | 535 | 2633 |
| head_f | 512 | 4102 | calibrated | fixed 0.5 | CD | 0.5 | 1784 | 92 | 222 | 535 | 2633 |
| head_f | 512 | 4102 | calibrated | C-selected / raw default | STTC | 0.46 | 1587 | 247 | 230 | 569 | 2633 |
| head_f | 512 | 4102 | calibrated | fixed 0.5 | STTC | 0.5 | 1613 | 221 | 258 | 541 | 2633 |
| head_f | 512 | 4102 | raw | C-selected / raw default | CD | 0.5 | 1788 | 88 | 224 | 533 | 2633 |
| head_f | 512 | 4102 | raw | fixed 0.5 | CD | 0.5 | 1788 | 88 | 224 | 533 | 2633 |
| head_f | 512 | 4102 | raw | C-selected / raw default | STTC | 0.5 | 1633 | 201 | 271 | 528 | 2633 |
| head_f | 512 | 4102 | raw | fixed 0.5 | STTC | 0.5 | 1633 | 201 | 271 | 528 | 2633 |
| random_a | 512 | 4102 | calibrated | C-selected / raw default | CD | 0.37 | 1749 | 127 | 188 | 569 | 2633 |
| random_a | 512 | 4102 | calibrated | fixed 0.5 | CD | 0.5 | 1783 | 93 | 222 | 535 | 2633 |
| random_a | 512 | 4102 | calibrated | C-selected / raw default | STTC | 0.47 | 1595 | 239 | 237 | 562 | 2633 |
| random_a | 512 | 4102 | calibrated | fixed 0.5 | STTC | 0.5 | 1619 | 215 | 258 | 541 | 2633 |
| random_a | 512 | 4102 | raw | C-selected / raw default | CD | 0.5 | 1791 | 85 | 228 | 529 | 2633 |
| random_a | 512 | 4102 | raw | fixed 0.5 | CD | 0.5 | 1791 | 85 | 228 | 529 | 2633 |
| random_a | 512 | 4102 | raw | C-selected / raw default | STTC | 0.5 | 1630 | 204 | 264 | 535 | 2633 |
| random_a | 512 | 4102 | raw | fixed 0.5 | STTC | 0.5 | 1630 | 204 | 264 | 535 | 2633 |
| selective_a | 512 | 4102 | calibrated | C-selected / raw default | CD | 0.37 | 1749 | 127 | 188 | 569 | 2633 |
| selective_a | 512 | 4102 | calibrated | fixed 0.5 | CD | 0.5 | 1783 | 93 | 222 | 535 | 2633 |
| selective_a | 512 | 4102 | calibrated | C-selected / raw default | STTC | 0.47 | 1595 | 239 | 237 | 562 | 2633 |
| selective_a | 512 | 4102 | calibrated | fixed 0.5 | STTC | 0.5 | 1619 | 215 | 258 | 541 | 2633 |
| selective_a | 512 | 4102 | raw | C-selected / raw default | CD | 0.5 | 1791 | 85 | 228 | 529 | 2633 |
| selective_a | 512 | 4102 | raw | fixed 0.5 | CD | 0.5 | 1791 | 85 | 228 | 529 | 2633 |
| selective_a | 512 | 4102 | raw | C-selected / raw default | STTC | 0.5 | 1630 | 204 | 264 | 535 | 2633 |
| selective_a | 512 | 4102 | raw | fixed 0.5 | STTC | 0.5 | 1630 | 204 | 264 | 535 | 2633 |
| selective_b | 512 | 4102 | calibrated | C-selected / raw default | CD | 0.37 | 1749 | 127 | 188 | 569 | 2633 |
| selective_b | 512 | 4102 | calibrated | fixed 0.5 | CD | 0.5 | 1783 | 93 | 222 | 535 | 2633 |
| selective_b | 512 | 4102 | calibrated | C-selected / raw default | STTC | 0.47 | 1595 | 239 | 237 | 562 | 2633 |
| selective_b | 512 | 4102 | calibrated | fixed 0.5 | STTC | 0.5 | 1619 | 215 | 258 | 541 | 2633 |
| selective_b | 512 | 4102 | raw | C-selected / raw default | CD | 0.5 | 1791 | 85 | 228 | 529 | 2633 |
| selective_b | 512 | 4102 | raw | fixed 0.5 | CD | 0.5 | 1791 | 85 | 228 | 529 | 2633 |
| selective_b | 512 | 4102 | raw | C-selected / raw default | STTC | 0.5 | 1630 | 204 | 264 | 535 | 2633 |
| selective_b | 512 | 4102 | raw | fixed 0.5 | STTC | 0.5 | 1630 | 204 | 264 | 535 | 2633 |
| target_only | 512 | 4102 | calibrated | C-selected / raw default | CD | 0.49 | 1738 | 138 | 306 | 451 | 2633 |
| target_only | 512 | 4102 | calibrated | fixed 0.5 | CD | 0.5 | 1740 | 136 | 308 | 449 | 2633 |
| target_only | 512 | 4102 | calibrated | C-selected / raw default | STTC | 0.34 | 1386 | 448 | 281 | 518 | 2633 |
| target_only | 512 | 4102 | calibrated | fixed 0.5 | STTC | 0.5 | 1610 | 224 | 422 | 377 | 2633 |
| target_only | 512 | 4102 | raw | C-selected / raw default | CD | 0.5 | 1696 | 180 | 277 | 480 | 2633 |
| target_only | 512 | 4102 | raw | fixed 0.5 | CD | 0.5 | 1696 | 180 | 277 | 480 | 2633 |
| target_only | 512 | 4102 | raw | C-selected / raw default | STTC | 0.5 | 1606 | 228 | 419 | 380 | 2633 |
| target_only | 512 | 4102 | raw | fixed 0.5 | STTC | 0.5 | 1606 | 228 | 419 | 380 | 2633 |
| source_only | — | — | calibrated | C-selected / raw default | CD | 0.37 | 1749 | 127 | 188 | 569 | 2633 |
| source_only | — | — | calibrated | fixed 0.5 | CD | 0.5 | 1783 | 93 | 222 | 535 | 2633 |
| source_only | — | — | calibrated | C-selected / raw default | STTC | 0.47 | 1595 | 239 | 237 | 562 | 2633 |
| source_only | — | — | calibrated | fixed 0.5 | STTC | 0.5 | 1619 | 215 | 258 | 541 | 2633 |
| source_only | — | — | raw | C-selected / raw default | CD | 0.5 | 1791 | 85 | 228 | 529 | 2633 |
| source_only | — | — | raw | fixed 0.5 | CD | 0.5 | 1791 | 85 | 228 | 529 | 2633 |
| source_only | — | — | raw | C-selected / raw default | STTC | 0.5 | 1630 | 204 | 264 | 535 | 2633 |
| source_only | — | — | raw | fixed 0.5 | STTC | 0.5 | 1630 | 204 | 264 | 535 | 2633 |

## Appendix B — Experiment 4 per-episode counts

Threshold 0.5 throughout. Source seed and draw seed identify fitted-run slots, not participants. Source-only repeated slots are aliases, not additional fits.

| Method | Budget | Source seed | Draw seed | Class | TN | FP | FN | TP | N |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| source_only | 64 | 15101 | 16101 | CD | 2474 | 358 | 245 | 783 | 3860 |
| source_only | 64 | 15101 | 16101 | STTC | 2522 | 396 | 753 | 189 | 3860 |
| source_only | 64 | 15101 | 16102 | CD | 2474 | 358 | 245 | 783 | 3860 |
| source_only | 64 | 15101 | 16102 | STTC | 2522 | 396 | 753 | 189 | 3860 |
| source_only | 64 | 15101 | 16103 | CD | 2474 | 358 | 245 | 783 | 3860 |
| source_only | 64 | 15101 | 16103 | STTC | 2522 | 396 | 753 | 189 | 3860 |
| source_only | 64 | 15102 | 16101 | CD | 2554 | 278 | 332 | 696 | 3860 |
| source_only | 64 | 15102 | 16101 | STTC | 2289 | 629 | 578 | 364 | 3860 |
| source_only | 64 | 15102 | 16102 | CD | 2554 | 278 | 332 | 696 | 3860 |
| source_only | 64 | 15102 | 16102 | STTC | 2289 | 629 | 578 | 364 | 3860 |
| source_only | 64 | 15102 | 16103 | CD | 2554 | 278 | 332 | 696 | 3860 |
| source_only | 64 | 15102 | 16103 | STTC | 2289 | 629 | 578 | 364 | 3860 |
| source_only | 64 | 15103 | 16101 | CD | 2441 | 391 | 277 | 751 | 3860 |
| source_only | 64 | 15103 | 16101 | STTC | 2515 | 403 | 743 | 199 | 3860 |
| source_only | 64 | 15103 | 16102 | CD | 2441 | 391 | 277 | 751 | 3860 |
| source_only | 64 | 15103 | 16102 | STTC | 2515 | 403 | 743 | 199 | 3860 |
| source_only | 64 | 15103 | 16103 | CD | 2441 | 391 | 277 | 751 | 3860 |
| source_only | 64 | 15103 | 16103 | STTC | 2515 | 403 | 743 | 199 | 3860 |
| source_only | 128 | 15101 | 16101 | CD | 2474 | 358 | 245 | 783 | 3860 |
| source_only | 128 | 15101 | 16101 | STTC | 2522 | 396 | 753 | 189 | 3860 |
| source_only | 128 | 15101 | 16102 | CD | 2474 | 358 | 245 | 783 | 3860 |
| source_only | 128 | 15101 | 16102 | STTC | 2522 | 396 | 753 | 189 | 3860 |
| source_only | 128 | 15101 | 16103 | CD | 2474 | 358 | 245 | 783 | 3860 |
| source_only | 128 | 15101 | 16103 | STTC | 2522 | 396 | 753 | 189 | 3860 |
| source_only | 128 | 15102 | 16101 | CD | 2554 | 278 | 332 | 696 | 3860 |
| source_only | 128 | 15102 | 16101 | STTC | 2289 | 629 | 578 | 364 | 3860 |
| source_only | 128 | 15102 | 16102 | CD | 2554 | 278 | 332 | 696 | 3860 |
| source_only | 128 | 15102 | 16102 | STTC | 2289 | 629 | 578 | 364 | 3860 |
| source_only | 128 | 15102 | 16103 | CD | 2554 | 278 | 332 | 696 | 3860 |
| source_only | 128 | 15102 | 16103 | STTC | 2289 | 629 | 578 | 364 | 3860 |
| source_only | 128 | 15103 | 16101 | CD | 2441 | 391 | 277 | 751 | 3860 |
| source_only | 128 | 15103 | 16101 | STTC | 2515 | 403 | 743 | 199 | 3860 |
| source_only | 128 | 15103 | 16102 | CD | 2441 | 391 | 277 | 751 | 3860 |
| source_only | 128 | 15103 | 16102 | STTC | 2515 | 403 | 743 | 199 | 3860 |
| source_only | 128 | 15103 | 16103 | CD | 2441 | 391 | 277 | 751 | 3860 |
| source_only | 128 | 15103 | 16103 | STTC | 2515 | 403 | 743 | 199 | 3860 |
| source_only | 256 | 15101 | 16101 | CD | 2474 | 358 | 245 | 783 | 3860 |
| source_only | 256 | 15101 | 16101 | STTC | 2522 | 396 | 753 | 189 | 3860 |
| source_only | 256 | 15101 | 16102 | CD | 2474 | 358 | 245 | 783 | 3860 |
| source_only | 256 | 15101 | 16102 | STTC | 2522 | 396 | 753 | 189 | 3860 |
| source_only | 256 | 15101 | 16103 | CD | 2474 | 358 | 245 | 783 | 3860 |
| source_only | 256 | 15101 | 16103 | STTC | 2522 | 396 | 753 | 189 | 3860 |
| source_only | 256 | 15102 | 16101 | CD | 2554 | 278 | 332 | 696 | 3860 |
| source_only | 256 | 15102 | 16101 | STTC | 2289 | 629 | 578 | 364 | 3860 |
| source_only | 256 | 15102 | 16102 | CD | 2554 | 278 | 332 | 696 | 3860 |
| source_only | 256 | 15102 | 16102 | STTC | 2289 | 629 | 578 | 364 | 3860 |
| source_only | 256 | 15102 | 16103 | CD | 2554 | 278 | 332 | 696 | 3860 |
| source_only | 256 | 15102 | 16103 | STTC | 2289 | 629 | 578 | 364 | 3860 |
| source_only | 256 | 15103 | 16101 | CD | 2441 | 391 | 277 | 751 | 3860 |
| source_only | 256 | 15103 | 16101 | STTC | 2515 | 403 | 743 | 199 | 3860 |
| source_only | 256 | 15103 | 16102 | CD | 2441 | 391 | 277 | 751 | 3860 |
| source_only | 256 | 15103 | 16102 | STTC | 2515 | 403 | 743 | 199 | 3860 |
| source_only | 256 | 15103 | 16103 | CD | 2441 | 391 | 277 | 751 | 3860 |
| source_only | 256 | 15103 | 16103 | STTC | 2515 | 403 | 743 | 199 | 3860 |
| head_cost | 64 | 15101 | 16101 | CD | 2526 | 306 | 319 | 709 | 3860 |
| head_cost | 64 | 15101 | 16101 | STTC | 2459 | 459 | 682 | 260 | 3860 |
| head_cost | 64 | 15101 | 16102 | CD | 2474 | 358 | 250 | 778 | 3860 |
| head_cost | 64 | 15101 | 16102 | STTC | 2460 | 458 | 682 | 260 | 3860 |
| head_cost | 64 | 15101 | 16103 | CD | 2527 | 305 | 320 | 708 | 3860 |
| head_cost | 64 | 15101 | 16103 | STTC | 2496 | 422 | 715 | 227 | 3860 |
| head_cost | 64 | 15102 | 16101 | CD | 2595 | 237 | 356 | 672 | 3860 |
| head_cost | 64 | 15102 | 16101 | STTC | 2245 | 673 | 503 | 439 | 3860 |
| head_cost | 64 | 15102 | 16102 | CD | 2532 | 300 | 316 | 712 | 3860 |
| head_cost | 64 | 15102 | 16102 | STTC | 2253 | 665 | 507 | 435 | 3860 |
| head_cost | 64 | 15102 | 16103 | CD | 2581 | 251 | 345 | 683 | 3860 |
| head_cost | 64 | 15102 | 16103 | STTC | 2374 | 544 | 631 | 311 | 3860 |
| head_cost | 64 | 15103 | 16101 | CD | 2476 | 356 | 296 | 732 | 3860 |
| head_cost | 64 | 15103 | 16101 | STTC | 2411 | 507 | 589 | 353 | 3860 |
| head_cost | 64 | 15103 | 16102 | CD | 2449 | 383 | 288 | 740 | 3860 |
| head_cost | 64 | 15103 | 16102 | STTC | 2462 | 456 | 640 | 302 | 3860 |
| head_cost | 64 | 15103 | 16103 | CD | 2472 | 360 | 294 | 734 | 3860 |
| head_cost | 64 | 15103 | 16103 | STTC | 2532 | 386 | 755 | 187 | 3860 |
| head_cost | 128 | 15101 | 16101 | CD | 2531 | 301 | 324 | 704 | 3860 |
| head_cost | 128 | 15101 | 16101 | STTC | 2479 | 439 | 692 | 250 | 3860 |
| head_cost | 128 | 15101 | 16102 | CD | 2474 | 358 | 246 | 782 | 3860 |
| head_cost | 128 | 15101 | 16102 | STTC | 2512 | 406 | 720 | 222 | 3860 |
| head_cost | 128 | 15101 | 16103 | CD | 2521 | 311 | 309 | 719 | 3860 |
| head_cost | 128 | 15101 | 16103 | STTC | 2481 | 437 | 705 | 237 | 3860 |
| head_cost | 128 | 15102 | 16101 | CD | 2595 | 237 | 357 | 671 | 3860 |
| head_cost | 128 | 15102 | 16101 | STTC | 2279 | 639 | 547 | 395 | 3860 |
| head_cost | 128 | 15102 | 16102 | CD | 2532 | 300 | 313 | 715 | 3860 |
| head_cost | 128 | 15102 | 16102 | STTC | 2378 | 540 | 620 | 322 | 3860 |
| head_cost | 128 | 15102 | 16103 | CD | 2574 | 258 | 340 | 688 | 3860 |
| head_cost | 128 | 15102 | 16103 | STTC | 2321 | 597 | 579 | 363 | 3860 |
| head_cost | 128 | 15103 | 16101 | CD | 2485 | 347 | 306 | 722 | 3860 |
| head_cost | 128 | 15103 | 16101 | STTC | 2455 | 463 | 631 | 311 | 3860 |
| head_cost | 128 | 15103 | 16102 | CD | 2432 | 400 | 277 | 751 | 3860 |
| head_cost | 128 | 15103 | 16102 | STTC | 2534 | 384 | 746 | 196 | 3860 |
| head_cost | 128 | 15103 | 16103 | CD | 2472 | 360 | 294 | 734 | 3860 |
| head_cost | 128 | 15103 | 16103 | STTC | 2480 | 438 | 700 | 242 | 3860 |
| head_cost | 256 | 15101 | 16101 | CD | 2521 | 311 | 308 | 720 | 3860 |
| head_cost | 256 | 15101 | 16101 | STTC | 2470 | 448 | 685 | 257 | 3860 |
| head_cost | 256 | 15101 | 16102 | CD | 2509 | 323 | 296 | 732 | 3860 |
| head_cost | 256 | 15101 | 16102 | STTC | 2479 | 439 | 692 | 250 | 3860 |
| head_cost | 256 | 15101 | 16103 | CD | 2521 | 311 | 308 | 720 | 3860 |
| head_cost | 256 | 15101 | 16103 | STTC | 2475 | 443 | 694 | 248 | 3860 |
| head_cost | 256 | 15102 | 16101 | CD | 2586 | 246 | 347 | 681 | 3860 |
| head_cost | 256 | 15102 | 16101 | STTC | 2297 | 621 | 561 | 381 | 3860 |
| head_cost | 256 | 15102 | 16102 | CD | 2556 | 276 | 333 | 695 | 3860 |
| head_cost | 256 | 15102 | 16102 | STTC | 2306 | 612 | 566 | 376 | 3860 |
| head_cost | 256 | 15102 | 16103 | CD | 2573 | 259 | 340 | 688 | 3860 |
| head_cost | 256 | 15102 | 16103 | STTC | 2296 | 622 | 571 | 371 | 3860 |
| head_cost | 256 | 15103 | 16101 | CD | 2469 | 363 | 292 | 736 | 3860 |
| head_cost | 256 | 15103 | 16101 | STTC | 2462 | 456 | 642 | 300 | 3860 |
| head_cost | 256 | 15103 | 16102 | CD | 2471 | 361 | 293 | 735 | 3860 |
| head_cost | 256 | 15103 | 16102 | STTC | 2469 | 449 | 651 | 291 | 3860 |
| head_cost | 256 | 15103 | 16103 | CD | 2475 | 357 | 297 | 731 | 3860 |
| head_cost | 256 | 15103 | 16103 | STTC | 2478 | 440 | 691 | 251 | 3860 |
| full_cost | 64 | 15101 | 16101 | CD | 2628 | 204 | 359 | 669 | 3860 |
| full_cost | 64 | 15101 | 16101 | STTC | 2174 | 744 | 358 | 584 | 3860 |
| full_cost | 64 | 15101 | 16102 | CD | 2460 | 372 | 182 | 846 | 3860 |
| full_cost | 64 | 15101 | 16102 | STTC | 2467 | 451 | 443 | 499 | 3860 |
| full_cost | 64 | 15101 | 16103 | CD | 2600 | 232 | 285 | 743 | 3860 |
| full_cost | 64 | 15101 | 16103 | STTC | 2500 | 418 | 557 | 385 | 3860 |
| full_cost | 64 | 15102 | 16101 | CD | 2683 | 149 | 369 | 659 | 3860 |
| full_cost | 64 | 15102 | 16101 | STTC | 2178 | 740 | 321 | 621 | 3860 |
| full_cost | 64 | 15102 | 16102 | CD | 2480 | 352 | 219 | 809 | 3860 |
| full_cost | 64 | 15102 | 16102 | STTC | 2498 | 420 | 536 | 406 | 3860 |
| full_cost | 64 | 15102 | 16103 | CD | 2621 | 211 | 337 | 691 | 3860 |
| full_cost | 64 | 15102 | 16103 | STTC | 2529 | 389 | 560 | 382 | 3860 |
| full_cost | 64 | 15103 | 16101 | CD | 2612 | 220 | 352 | 676 | 3860 |
| full_cost | 64 | 15103 | 16101 | STTC | 2215 | 703 | 374 | 568 | 3860 |
| full_cost | 64 | 15103 | 16102 | CD | 2453 | 379 | 231 | 797 | 3860 |
| full_cost | 64 | 15103 | 16102 | STTC | 2603 | 315 | 660 | 282 | 3860 |
| full_cost | 64 | 15103 | 16103 | CD | 2549 | 283 | 276 | 752 | 3860 |
| full_cost | 64 | 15103 | 16103 | STTC | 2521 | 397 | 647 | 295 | 3860 |
| full_cost | 128 | 15101 | 16101 | CD | 2672 | 160 | 363 | 665 | 3860 |
| full_cost | 128 | 15101 | 16101 | STTC | 2402 | 516 | 416 | 526 | 3860 |
| full_cost | 128 | 15101 | 16102 | CD | 2518 | 314 | 181 | 847 | 3860 |
| full_cost | 128 | 15101 | 16102 | STTC | 2711 | 207 | 648 | 294 | 3860 |
| full_cost | 128 | 15101 | 16103 | CD | 2604 | 228 | 293 | 735 | 3860 |
| full_cost | 128 | 15101 | 16103 | STTC | 2422 | 496 | 448 | 494 | 3860 |
| full_cost | 128 | 15102 | 16101 | CD | 2685 | 147 | 336 | 692 | 3860 |
| full_cost | 128 | 15102 | 16101 | STTC | 2471 | 447 | 427 | 515 | 3860 |
| full_cost | 128 | 15102 | 16102 | CD | 2530 | 302 | 202 | 826 | 3860 |
| full_cost | 128 | 15102 | 16102 | STTC | 2732 | 186 | 672 | 270 | 3860 |
| full_cost | 128 | 15102 | 16103 | CD | 2649 | 183 | 321 | 707 | 3860 |
| full_cost | 128 | 15102 | 16103 | STTC | 2402 | 516 | 412 | 530 | 3860 |
| full_cost | 128 | 15103 | 16101 | CD | 2644 | 188 | 343 | 685 | 3860 |
| full_cost | 128 | 15103 | 16101 | STTC | 2392 | 526 | 464 | 478 | 3860 |
| full_cost | 128 | 15103 | 16102 | CD | 2530 | 302 | 220 | 808 | 3860 |
| full_cost | 128 | 15103 | 16102 | STTC | 2774 | 144 | 807 | 135 | 3860 |
| full_cost | 128 | 15103 | 16103 | CD | 2587 | 245 | 292 | 736 | 3860 |
| full_cost | 128 | 15103 | 16103 | STTC | 2503 | 415 | 571 | 371 | 3860 |
| full_cost | 256 | 15101 | 16101 | CD | 2659 | 173 | 305 | 723 | 3860 |
| full_cost | 256 | 15101 | 16101 | STTC | 2524 | 394 | 415 | 527 | 3860 |
| full_cost | 256 | 15101 | 16102 | CD | 2632 | 200 | 256 | 772 | 3860 |
| full_cost | 256 | 15101 | 16102 | STTC | 2624 | 294 | 508 | 434 | 3860 |
| full_cost | 256 | 15101 | 16103 | CD | 2649 | 183 | 283 | 745 | 3860 |
| full_cost | 256 | 15101 | 16103 | STTC | 2411 | 507 | 422 | 520 | 3860 |
| full_cost | 256 | 15102 | 16101 | CD | 2663 | 169 | 280 | 748 | 3860 |
| full_cost | 256 | 15102 | 16101 | STTC | 2532 | 386 | 451 | 491 | 3860 |
| full_cost | 256 | 15102 | 16102 | CD | 2623 | 209 | 238 | 790 | 3860 |
| full_cost | 256 | 15102 | 16102 | STTC | 2600 | 318 | 513 | 429 | 3860 |
| full_cost | 256 | 15102 | 16103 | CD | 2656 | 176 | 306 | 722 | 3860 |
| full_cost | 256 | 15102 | 16103 | STTC | 2525 | 393 | 481 | 461 | 3860 |
| full_cost | 256 | 15103 | 16101 | CD | 2631 | 201 | 297 | 731 | 3860 |
| full_cost | 256 | 15103 | 16101 | STTC | 2487 | 431 | 468 | 474 | 3860 |
| full_cost | 256 | 15103 | 16102 | CD | 2645 | 187 | 283 | 745 | 3860 |
| full_cost | 256 | 15103 | 16102 | STTC | 2603 | 315 | 601 | 341 | 3860 |
| full_cost | 256 | 15103 | 16103 | CD | 2615 | 217 | 313 | 715 | 3860 |
| full_cost | 256 | 15103 | 16103 | STTC | 2526 | 392 | 583 | 359 | 3860 |
| random_a | 64 | 15101 | 16101 | CD | 2606 | 226 | 365 | 663 | 3860 |
| random_a | 64 | 15101 | 16101 | STTC | 2231 | 687 | 392 | 550 | 3860 |
| random_a | 64 | 15101 | 16102 | CD | 2474 | 358 | 250 | 778 | 3860 |
| random_a | 64 | 15101 | 16102 | STTC | 2460 | 458 | 682 | 260 | 3860 |
| random_a | 64 | 15101 | 16103 | CD | 2526 | 306 | 319 | 709 | 3860 |
| random_a | 64 | 15101 | 16103 | STTC | 2544 | 374 | 632 | 310 | 3860 |
| random_a | 64 | 15102 | 16101 | CD | 2652 | 180 | 381 | 647 | 3860 |
| random_a | 64 | 15102 | 16101 | STTC | 2426 | 492 | 563 | 379 | 3860 |
| random_a | 64 | 15102 | 16102 | CD | 2532 | 300 | 316 | 712 | 3860 |
| random_a | 64 | 15102 | 16102 | STTC | 2253 | 665 | 507 | 435 | 3860 |
| random_a | 64 | 15102 | 16103 | CD | 2680 | 152 | 403 | 625 | 3860 |
| random_a | 64 | 15102 | 16103 | STTC | 2508 | 410 | 608 | 334 | 3860 |
| random_a | 64 | 15103 | 16101 | CD | 2575 | 257 | 358 | 670 | 3860 |
| random_a | 64 | 15103 | 16101 | STTC | 2248 | 670 | 419 | 523 | 3860 |
| random_a | 64 | 15103 | 16102 | CD | 2449 | 383 | 288 | 740 | 3860 |
| random_a | 64 | 15103 | 16102 | STTC | 2462 | 456 | 640 | 302 | 3860 |
| random_a | 64 | 15103 | 16103 | CD | 2558 | 274 | 332 | 696 | 3860 |
| random_a | 64 | 15103 | 16103 | STTC | 2562 | 356 | 706 | 236 | 3860 |
| random_a | 128 | 15101 | 16101 | CD | 2627 | 205 | 390 | 638 | 3860 |
| random_a | 128 | 15101 | 16101 | STTC | 2508 | 410 | 523 | 419 | 3860 |
| random_a | 128 | 15101 | 16102 | CD | 2424 | 408 | 211 | 817 | 3860 |
| random_a | 128 | 15101 | 16102 | STTC | 2738 | 180 | 736 | 206 | 3860 |
| random_a | 128 | 15101 | 16103 | CD | 2599 | 233 | 374 | 654 | 3860 |
| random_a | 128 | 15101 | 16103 | STTC | 2432 | 486 | 545 | 397 | 3860 |
| random_a | 128 | 15102 | 16101 | CD | 2701 | 131 | 375 | 653 | 3860 |
| random_a | 128 | 15102 | 16101 | STTC | 2536 | 382 | 596 | 346 | 3860 |
| random_a | 128 | 15102 | 16102 | CD | 2479 | 353 | 234 | 794 | 3860 |
| random_a | 128 | 15102 | 16102 | STTC | 2759 | 159 | 765 | 177 | 3860 |
| random_a | 128 | 15102 | 16103 | CD | 2650 | 182 | 389 | 639 | 3860 |
| random_a | 128 | 15102 | 16103 | STTC | 2473 | 445 | 526 | 416 | 3860 |
| random_a | 128 | 15103 | 16101 | CD | 2606 | 226 | 355 | 673 | 3860 |
| random_a | 128 | 15103 | 16101 | STTC | 2430 | 488 | 498 | 444 | 3860 |
| random_a | 128 | 15103 | 16102 | CD | 2461 | 371 | 208 | 820 | 3860 |
| random_a | 128 | 15103 | 16102 | STTC | 2768 | 150 | 815 | 127 | 3860 |
| random_a | 128 | 15103 | 16103 | CD | 2553 | 279 | 346 | 682 | 3860 |
| random_a | 128 | 15103 | 16103 | STTC | 2507 | 411 | 617 | 325 | 3860 |
| random_a | 256 | 15101 | 16101 | CD | 2660 | 172 | 365 | 663 | 3860 |
| random_a | 256 | 15101 | 16101 | STTC | 2535 | 383 | 497 | 445 | 3860 |
| random_a | 256 | 15101 | 16102 | CD | 2614 | 218 | 315 | 713 | 3860 |
| random_a | 256 | 15101 | 16102 | STTC | 2635 | 283 | 583 | 359 | 3860 |
| random_a | 256 | 15101 | 16103 | CD | 2589 | 243 | 355 | 673 | 3860 |
| random_a | 256 | 15101 | 16103 | STTC | 2481 | 437 | 480 | 462 | 3860 |
| random_a | 256 | 15102 | 16101 | CD | 2632 | 200 | 329 | 699 | 3860 |
| random_a | 256 | 15102 | 16101 | STTC | 2556 | 362 | 602 | 340 | 3860 |
| random_a | 256 | 15102 | 16102 | CD | 2630 | 202 | 268 | 760 | 3860 |
| random_a | 256 | 15102 | 16102 | STTC | 2604 | 314 | 525 | 417 | 3860 |
| random_a | 256 | 15102 | 16103 | CD | 2632 | 200 | 360 | 668 | 3860 |
| random_a | 256 | 15102 | 16103 | STTC | 2555 | 363 | 591 | 351 | 3860 |
| random_a | 256 | 15103 | 16101 | CD | 2627 | 205 | 300 | 728 | 3860 |
| random_a | 256 | 15103 | 16101 | STTC | 2509 | 409 | 503 | 439 | 3860 |
| random_a | 256 | 15103 | 16102 | CD | 2620 | 212 | 288 | 740 | 3860 |
| random_a | 256 | 15103 | 16102 | STTC | 2610 | 308 | 616 | 326 | 3860 |
| random_a | 256 | 15103 | 16103 | CD | 2612 | 220 | 313 | 715 | 3860 |
| random_a | 256 | 15103 | 16103 | STTC | 2559 | 359 | 620 | 322 | 3860 |
| selective_a | 64 | 15101 | 16101 | CD | 2624 | 208 | 362 | 666 | 3860 |
| selective_a | 64 | 15101 | 16101 | STTC | 2157 | 761 | 384 | 558 | 3860 |
| selective_a | 64 | 15101 | 16102 | CD | 2474 | 358 | 250 | 778 | 3860 |
| selective_a | 64 | 15101 | 16102 | STTC | 2460 | 458 | 682 | 260 | 3860 |
| selective_a | 64 | 15101 | 16103 | CD | 2526 | 306 | 319 | 709 | 3860 |
| selective_a | 64 | 15101 | 16103 | STTC | 2544 | 374 | 632 | 310 | 3860 |
| selective_a | 64 | 15102 | 16101 | CD | 2652 | 180 | 381 | 647 | 3860 |
| selective_a | 64 | 15102 | 16101 | STTC | 2426 | 492 | 563 | 379 | 3860 |
| selective_a | 64 | 15102 | 16102 | CD | 2532 | 300 | 316 | 712 | 3860 |
| selective_a | 64 | 15102 | 16102 | STTC | 2253 | 665 | 507 | 435 | 3860 |
| selective_a | 64 | 15102 | 16103 | CD | 2680 | 152 | 403 | 625 | 3860 |
| selective_a | 64 | 15102 | 16103 | STTC | 2508 | 410 | 608 | 334 | 3860 |
| selective_a | 64 | 15103 | 16101 | CD | 2582 | 250 | 347 | 681 | 3860 |
| selective_a | 64 | 15103 | 16101 | STTC | 2236 | 682 | 397 | 545 | 3860 |
| selective_a | 64 | 15103 | 16102 | CD | 2449 | 383 | 288 | 740 | 3860 |
| selective_a | 64 | 15103 | 16102 | STTC | 2462 | 456 | 640 | 302 | 3860 |
| selective_a | 64 | 15103 | 16103 | CD | 2546 | 286 | 332 | 696 | 3860 |
| selective_a | 64 | 15103 | 16103 | STTC | 2517 | 401 | 671 | 271 | 3860 |
| selective_a | 128 | 15101 | 16101 | CD | 2649 | 183 | 391 | 637 | 3860 |
| selective_a | 128 | 15101 | 16101 | STTC | 2451 | 467 | 482 | 460 | 3860 |
| selective_a | 128 | 15101 | 16102 | CD | 2424 | 408 | 211 | 817 | 3860 |
| selective_a | 128 | 15101 | 16102 | STTC | 2738 | 180 | 736 | 206 | 3860 |
| selective_a | 128 | 15101 | 16103 | CD | 2569 | 263 | 358 | 670 | 3860 |
| selective_a | 128 | 15101 | 16103 | STTC | 2427 | 491 | 481 | 461 | 3860 |
| selective_a | 128 | 15102 | 16101 | CD | 2683 | 149 | 400 | 628 | 3860 |
| selective_a | 128 | 15102 | 16101 | STTC | 2491 | 427 | 585 | 357 | 3860 |
| selective_a | 128 | 15102 | 16102 | CD | 2479 | 353 | 234 | 794 | 3860 |
| selective_a | 128 | 15102 | 16102 | STTC | 2759 | 159 | 765 | 177 | 3860 |
| selective_a | 128 | 15102 | 16103 | CD | 2650 | 182 | 389 | 639 | 3860 |
| selective_a | 128 | 15102 | 16103 | STTC | 2473 | 445 | 526 | 416 | 3860 |
| selective_a | 128 | 15103 | 16101 | CD | 2590 | 242 | 357 | 671 | 3860 |
| selective_a | 128 | 15103 | 16101 | STTC | 2421 | 497 | 486 | 456 | 3860 |
| selective_a | 128 | 15103 | 16102 | CD | 2461 | 371 | 208 | 820 | 3860 |
| selective_a | 128 | 15103 | 16102 | STTC | 2768 | 150 | 815 | 127 | 3860 |
| selective_a | 128 | 15103 | 16103 | CD | 2553 | 279 | 346 | 682 | 3860 |
| selective_a | 128 | 15103 | 16103 | STTC | 2507 | 411 | 617 | 325 | 3860 |
| selective_a | 256 | 15101 | 16101 | CD | 2660 | 172 | 365 | 663 | 3860 |
| selective_a | 256 | 15101 | 16101 | STTC | 2535 | 383 | 497 | 445 | 3860 |
| selective_a | 256 | 15101 | 16102 | CD | 2599 | 233 | 256 | 772 | 3860 |
| selective_a | 256 | 15101 | 16102 | STTC | 2616 | 302 | 502 | 440 | 3860 |
| selective_a | 256 | 15101 | 16103 | CD | 2605 | 227 | 339 | 689 | 3860 |
| selective_a | 256 | 15101 | 16103 | STTC | 2446 | 472 | 429 | 513 | 3860 |
| selective_a | 256 | 15102 | 16101 | CD | 2650 | 182 | 372 | 656 | 3860 |
| selective_a | 256 | 15102 | 16101 | STTC | 2524 | 394 | 550 | 392 | 3860 |
| selective_a | 256 | 15102 | 16102 | CD | 2630 | 202 | 268 | 760 | 3860 |
| selective_a | 256 | 15102 | 16102 | STTC | 2604 | 314 | 525 | 417 | 3860 |
| selective_a | 256 | 15102 | 16103 | CD | 2639 | 193 | 375 | 653 | 3860 |
| selective_a | 256 | 15102 | 16103 | STTC | 2540 | 378 | 526 | 416 | 3860 |
| selective_a | 256 | 15103 | 16101 | CD | 2623 | 209 | 311 | 717 | 3860 |
| selective_a | 256 | 15103 | 16101 | STTC | 2479 | 439 | 457 | 485 | 3860 |
| selective_a | 256 | 15103 | 16102 | CD | 2620 | 212 | 288 | 740 | 3860 |
| selective_a | 256 | 15103 | 16102 | STTC | 2610 | 308 | 616 | 326 | 3860 |
| selective_a | 256 | 15103 | 16103 | CD | 2607 | 225 | 319 | 709 | 3860 |
| selective_a | 256 | 15103 | 16103 | STTC | 2526 | 392 | 578 | 364 | 3860 |
| target_only_cost | 64 | 15101 | 16101 | CD | 2753 | 79 | 530 | 498 | 3860 |
| target_only_cost | 64 | 15101 | 16101 | STTC | 2405 | 513 | 504 | 438 | 3860 |
| target_only_cost | 64 | 15101 | 16102 | CD | 2693 | 139 | 566 | 462 | 3860 |
| target_only_cost | 64 | 15101 | 16102 | STTC | 2767 | 151 | 779 | 163 | 3860 |
| target_only_cost | 64 | 15101 | 16103 | CD | 2758 | 74 | 539 | 489 | 3860 |
| target_only_cost | 64 | 15101 | 16103 | STTC | 2691 | 227 | 724 | 218 | 3860 |
| target_only_cost | 64 | 15102 | 16101 | CD | 2759 | 73 | 526 | 502 | 3860 |
| target_only_cost | 64 | 15102 | 16101 | STTC | 2521 | 397 | 590 | 352 | 3860 |
| target_only_cost | 64 | 15102 | 16102 | CD | 2708 | 124 | 514 | 514 | 3860 |
| target_only_cost | 64 | 15102 | 16102 | STTC | 2665 | 253 | 666 | 276 | 3860 |
| target_only_cost | 64 | 15102 | 16103 | CD | 2753 | 79 | 565 | 463 | 3860 |
| target_only_cost | 64 | 15102 | 16103 | STTC | 2745 | 173 | 744 | 198 | 3860 |
| target_only_cost | 64 | 15103 | 16101 | CD | 2736 | 96 | 549 | 479 | 3860 |
| target_only_cost | 64 | 15103 | 16101 | STTC | 2433 | 485 | 511 | 431 | 3860 |
| target_only_cost | 64 | 15103 | 16102 | CD | 2649 | 183 | 484 | 544 | 3860 |
| target_only_cost | 64 | 15103 | 16102 | STTC | 2729 | 189 | 737 | 205 | 3860 |
| target_only_cost | 64 | 15103 | 16103 | CD | 2589 | 243 | 434 | 594 | 3860 |
| target_only_cost | 64 | 15103 | 16103 | STTC | 2657 | 261 | 724 | 218 | 3860 |
| target_only_cost | 128 | 15101 | 16101 | CD | 2660 | 172 | 477 | 551 | 3860 |
| target_only_cost | 128 | 15101 | 16101 | STTC | 2665 | 253 | 686 | 256 | 3860 |
| target_only_cost | 128 | 15101 | 16102 | CD | 2677 | 155 | 436 | 592 | 3860 |
| target_only_cost | 128 | 15101 | 16102 | STTC | 2815 | 103 | 800 | 142 | 3860 |
| target_only_cost | 128 | 15101 | 16103 | CD | 2770 | 62 | 530 | 498 | 3860 |
| target_only_cost | 128 | 15101 | 16103 | STTC | 2347 | 571 | 493 | 449 | 3860 |
| target_only_cost | 128 | 15102 | 16101 | CD | 2732 | 100 | 533 | 495 | 3860 |
| target_only_cost | 128 | 15102 | 16101 | STTC | 2624 | 294 | 615 | 327 | 3860 |
| target_only_cost | 128 | 15102 | 16102 | CD | 2660 | 172 | 459 | 569 | 3860 |
| target_only_cost | 128 | 15102 | 16102 | STTC | 2794 | 124 | 766 | 176 | 3860 |
| target_only_cost | 128 | 15102 | 16103 | CD | 2756 | 76 | 561 | 467 | 3860 |
| target_only_cost | 128 | 15102 | 16103 | STTC | 2420 | 498 | 554 | 388 | 3860 |
| target_only_cost | 128 | 15103 | 16101 | CD | 2697 | 135 | 483 | 545 | 3860 |
| target_only_cost | 128 | 15103 | 16101 | STTC | 2818 | 100 | 788 | 154 | 3860 |
| target_only_cost | 128 | 15103 | 16102 | CD | 2560 | 272 | 389 | 639 | 3860 |
| target_only_cost | 128 | 15103 | 16102 | STTC | 2797 | 121 | 787 | 155 | 3860 |
| target_only_cost | 128 | 15103 | 16103 | CD | 2779 | 53 | 546 | 482 | 3860 |
| target_only_cost | 128 | 15103 | 16103 | STTC | 2628 | 290 | 640 | 302 | 3860 |
| target_only_cost | 256 | 15101 | 16101 | CD | 2667 | 165 | 462 | 566 | 3860 |
| target_only_cost | 256 | 15101 | 16101 | STTC | 2749 | 169 | 604 | 338 | 3860 |
| target_only_cost | 256 | 15101 | 16102 | CD | 2668 | 164 | 465 | 563 | 3860 |
| target_only_cost | 256 | 15101 | 16102 | STTC | 2653 | 265 | 627 | 315 | 3860 |
| target_only_cost | 256 | 15101 | 16103 | CD | 2773 | 59 | 537 | 491 | 3860 |
| target_only_cost | 256 | 15101 | 16103 | STTC | 2534 | 384 | 596 | 346 | 3860 |
| target_only_cost | 256 | 15102 | 16101 | CD | 2707 | 125 | 461 | 567 | 3860 |
| target_only_cost | 256 | 15102 | 16101 | STTC | 2606 | 312 | 560 | 382 | 3860 |
| target_only_cost | 256 | 15102 | 16102 | CD | 2425 | 407 | 326 | 702 | 3860 |
| target_only_cost | 256 | 15102 | 16102 | STTC | 2691 | 227 | 605 | 337 | 3860 |
| target_only_cost | 256 | 15102 | 16103 | CD | 2708 | 124 | 447 | 581 | 3860 |
| target_only_cost | 256 | 15102 | 16103 | STTC | 2543 | 375 | 545 | 397 | 3860 |
| target_only_cost | 256 | 15103 | 16101 | CD | 2615 | 217 | 434 | 594 | 3860 |
| target_only_cost | 256 | 15103 | 16101 | STTC | 2563 | 355 | 531 | 411 | 3860 |
| target_only_cost | 256 | 15103 | 16102 | CD | 2710 | 122 | 473 | 555 | 3860 |
| target_only_cost | 256 | 15103 | 16102 | STTC | 2660 | 258 | 631 | 311 | 3860 |
| target_only_cost | 256 | 15103 | 16103 | CD | 2596 | 236 | 372 | 656 | 3860 |
| target_only_cost | 256 | 15103 | 16103 | STTC | 2431 | 487 | 553 | 389 | 3860 |

## Appendix C — Phase 4C per-episode counts

Threshold 0.5 throughout. Source seed and draw seed identify fitted-run slots, not participants. Source-only repeated slots are aliases, not additional fits.

| Method | Budget | Source seed | Draw seed | Class | TN | FP | FN | TP | N |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| M1 | 64 | 15101 | 16101 | CD | 2474 | 358 | 245 | 783 | 3860 |
| M1 | 64 | 15101 | 16101 | STTC | 2522 | 396 | 753 | 189 | 3860 |
| M1 | 64 | 15101 | 16102 | CD | 2474 | 358 | 245 | 783 | 3860 |
| M1 | 64 | 15101 | 16102 | STTC | 2522 | 396 | 753 | 189 | 3860 |
| M1 | 64 | 15101 | 16103 | CD | 2474 | 358 | 245 | 783 | 3860 |
| M1 | 64 | 15101 | 16103 | STTC | 2522 | 396 | 753 | 189 | 3860 |
| M1 | 64 | 15102 | 16101 | CD | 2554 | 278 | 332 | 696 | 3860 |
| M1 | 64 | 15102 | 16101 | STTC | 2289 | 629 | 578 | 364 | 3860 |
| M1 | 64 | 15102 | 16102 | CD | 2554 | 278 | 332 | 696 | 3860 |
| M1 | 64 | 15102 | 16102 | STTC | 2289 | 629 | 578 | 364 | 3860 |
| M1 | 64 | 15102 | 16103 | CD | 2554 | 278 | 332 | 696 | 3860 |
| M1 | 64 | 15102 | 16103 | STTC | 2289 | 629 | 578 | 364 | 3860 |
| M1 | 64 | 15103 | 16101 | CD | 2441 | 391 | 277 | 751 | 3860 |
| M1 | 64 | 15103 | 16101 | STTC | 2515 | 403 | 743 | 199 | 3860 |
| M1 | 64 | 15103 | 16102 | CD | 2441 | 391 | 277 | 751 | 3860 |
| M1 | 64 | 15103 | 16102 | STTC | 2515 | 403 | 743 | 199 | 3860 |
| M1 | 64 | 15103 | 16103 | CD | 2441 | 391 | 277 | 751 | 3860 |
| M1 | 64 | 15103 | 16103 | STTC | 2515 | 403 | 743 | 199 | 3860 |
| M1 | 128 | 15101 | 16101 | CD | 2474 | 358 | 245 | 783 | 3860 |
| M1 | 128 | 15101 | 16101 | STTC | 2522 | 396 | 753 | 189 | 3860 |
| M1 | 128 | 15101 | 16102 | CD | 2474 | 358 | 245 | 783 | 3860 |
| M1 | 128 | 15101 | 16102 | STTC | 2522 | 396 | 753 | 189 | 3860 |
| M1 | 128 | 15101 | 16103 | CD | 2474 | 358 | 245 | 783 | 3860 |
| M1 | 128 | 15101 | 16103 | STTC | 2522 | 396 | 753 | 189 | 3860 |
| M1 | 128 | 15102 | 16101 | CD | 2554 | 278 | 332 | 696 | 3860 |
| M1 | 128 | 15102 | 16101 | STTC | 2289 | 629 | 578 | 364 | 3860 |
| M1 | 128 | 15102 | 16102 | CD | 2554 | 278 | 332 | 696 | 3860 |
| M1 | 128 | 15102 | 16102 | STTC | 2289 | 629 | 578 | 364 | 3860 |
| M1 | 128 | 15102 | 16103 | CD | 2554 | 278 | 332 | 696 | 3860 |
| M1 | 128 | 15102 | 16103 | STTC | 2289 | 629 | 578 | 364 | 3860 |
| M1 | 128 | 15103 | 16101 | CD | 2441 | 391 | 277 | 751 | 3860 |
| M1 | 128 | 15103 | 16101 | STTC | 2515 | 403 | 743 | 199 | 3860 |
| M1 | 128 | 15103 | 16102 | CD | 2441 | 391 | 277 | 751 | 3860 |
| M1 | 128 | 15103 | 16102 | STTC | 2515 | 403 | 743 | 199 | 3860 |
| M1 | 128 | 15103 | 16103 | CD | 2441 | 391 | 277 | 751 | 3860 |
| M1 | 128 | 15103 | 16103 | STTC | 2515 | 403 | 743 | 199 | 3860 |
| M1 | 256 | 15101 | 16101 | CD | 2474 | 358 | 245 | 783 | 3860 |
| M1 | 256 | 15101 | 16101 | STTC | 2522 | 396 | 753 | 189 | 3860 |
| M1 | 256 | 15101 | 16102 | CD | 2474 | 358 | 245 | 783 | 3860 |
| M1 | 256 | 15101 | 16102 | STTC | 2522 | 396 | 753 | 189 | 3860 |
| M1 | 256 | 15101 | 16103 | CD | 2474 | 358 | 245 | 783 | 3860 |
| M1 | 256 | 15101 | 16103 | STTC | 2522 | 396 | 753 | 189 | 3860 |
| M1 | 256 | 15102 | 16101 | CD | 2554 | 278 | 332 | 696 | 3860 |
| M1 | 256 | 15102 | 16101 | STTC | 2289 | 629 | 578 | 364 | 3860 |
| M1 | 256 | 15102 | 16102 | CD | 2554 | 278 | 332 | 696 | 3860 |
| M1 | 256 | 15102 | 16102 | STTC | 2289 | 629 | 578 | 364 | 3860 |
| M1 | 256 | 15102 | 16103 | CD | 2554 | 278 | 332 | 696 | 3860 |
| M1 | 256 | 15102 | 16103 | STTC | 2289 | 629 | 578 | 364 | 3860 |
| M1 | 256 | 15103 | 16101 | CD | 2441 | 391 | 277 | 751 | 3860 |
| M1 | 256 | 15103 | 16101 | STTC | 2515 | 403 | 743 | 199 | 3860 |
| M1 | 256 | 15103 | 16102 | CD | 2441 | 391 | 277 | 751 | 3860 |
| M1 | 256 | 15103 | 16102 | STTC | 2515 | 403 | 743 | 199 | 3860 |
| M1 | 256 | 15103 | 16103 | CD | 2441 | 391 | 277 | 751 | 3860 |
| M1 | 256 | 15103 | 16103 | STTC | 2515 | 403 | 743 | 199 | 3860 |
| M2 | 64 | 15101 | 16101 | CD | 2628 | 204 | 359 | 669 | 3860 |
| M2 | 64 | 15101 | 16101 | STTC | 2174 | 744 | 358 | 584 | 3860 |
| M2 | 64 | 15101 | 16102 | CD | 2460 | 372 | 182 | 846 | 3860 |
| M2 | 64 | 15101 | 16102 | STTC | 2467 | 451 | 443 | 499 | 3860 |
| M2 | 64 | 15101 | 16103 | CD | 2600 | 232 | 285 | 743 | 3860 |
| M2 | 64 | 15101 | 16103 | STTC | 2500 | 418 | 557 | 385 | 3860 |
| M2 | 64 | 15102 | 16101 | CD | 2683 | 149 | 369 | 659 | 3860 |
| M2 | 64 | 15102 | 16101 | STTC | 2178 | 740 | 321 | 621 | 3860 |
| M2 | 64 | 15102 | 16102 | CD | 2480 | 352 | 219 | 809 | 3860 |
| M2 | 64 | 15102 | 16102 | STTC | 2498 | 420 | 536 | 406 | 3860 |
| M2 | 64 | 15102 | 16103 | CD | 2621 | 211 | 337 | 691 | 3860 |
| M2 | 64 | 15102 | 16103 | STTC | 2529 | 389 | 560 | 382 | 3860 |
| M2 | 64 | 15103 | 16101 | CD | 2612 | 220 | 352 | 676 | 3860 |
| M2 | 64 | 15103 | 16101 | STTC | 2215 | 703 | 374 | 568 | 3860 |
| M2 | 64 | 15103 | 16102 | CD | 2453 | 379 | 231 | 797 | 3860 |
| M2 | 64 | 15103 | 16102 | STTC | 2603 | 315 | 660 | 282 | 3860 |
| M2 | 64 | 15103 | 16103 | CD | 2549 | 283 | 276 | 752 | 3860 |
| M2 | 64 | 15103 | 16103 | STTC | 2521 | 397 | 647 | 295 | 3860 |
| M2 | 128 | 15101 | 16101 | CD | 2672 | 160 | 363 | 665 | 3860 |
| M2 | 128 | 15101 | 16101 | STTC | 2402 | 516 | 416 | 526 | 3860 |
| M2 | 128 | 15101 | 16102 | CD | 2518 | 314 | 181 | 847 | 3860 |
| M2 | 128 | 15101 | 16102 | STTC | 2711 | 207 | 648 | 294 | 3860 |
| M2 | 128 | 15101 | 16103 | CD | 2604 | 228 | 293 | 735 | 3860 |
| M2 | 128 | 15101 | 16103 | STTC | 2422 | 496 | 448 | 494 | 3860 |
| M2 | 128 | 15102 | 16101 | CD | 2685 | 147 | 336 | 692 | 3860 |
| M2 | 128 | 15102 | 16101 | STTC | 2471 | 447 | 427 | 515 | 3860 |
| M2 | 128 | 15102 | 16102 | CD | 2530 | 302 | 202 | 826 | 3860 |
| M2 | 128 | 15102 | 16102 | STTC | 2732 | 186 | 672 | 270 | 3860 |
| M2 | 128 | 15102 | 16103 | CD | 2649 | 183 | 321 | 707 | 3860 |
| M2 | 128 | 15102 | 16103 | STTC | 2402 | 516 | 412 | 530 | 3860 |
| M2 | 128 | 15103 | 16101 | CD | 2644 | 188 | 343 | 685 | 3860 |
| M2 | 128 | 15103 | 16101 | STTC | 2392 | 526 | 464 | 478 | 3860 |
| M2 | 128 | 15103 | 16102 | CD | 2530 | 302 | 220 | 808 | 3860 |
| M2 | 128 | 15103 | 16102 | STTC | 2774 | 144 | 807 | 135 | 3860 |
| M2 | 128 | 15103 | 16103 | CD | 2587 | 245 | 292 | 736 | 3860 |
| M2 | 128 | 15103 | 16103 | STTC | 2503 | 415 | 571 | 371 | 3860 |
| M2 | 256 | 15101 | 16101 | CD | 2659 | 173 | 305 | 723 | 3860 |
| M2 | 256 | 15101 | 16101 | STTC | 2524 | 394 | 415 | 527 | 3860 |
| M2 | 256 | 15101 | 16102 | CD | 2632 | 200 | 256 | 772 | 3860 |
| M2 | 256 | 15101 | 16102 | STTC | 2624 | 294 | 508 | 434 | 3860 |
| M2 | 256 | 15101 | 16103 | CD | 2649 | 183 | 283 | 745 | 3860 |
| M2 | 256 | 15101 | 16103 | STTC | 2411 | 507 | 422 | 520 | 3860 |
| M2 | 256 | 15102 | 16101 | CD | 2663 | 169 | 280 | 748 | 3860 |
| M2 | 256 | 15102 | 16101 | STTC | 2532 | 386 | 451 | 491 | 3860 |
| M2 | 256 | 15102 | 16102 | CD | 2623 | 209 | 238 | 790 | 3860 |
| M2 | 256 | 15102 | 16102 | STTC | 2600 | 318 | 513 | 429 | 3860 |
| M2 | 256 | 15102 | 16103 | CD | 2656 | 176 | 306 | 722 | 3860 |
| M2 | 256 | 15102 | 16103 | STTC | 2525 | 393 | 481 | 461 | 3860 |
| M2 | 256 | 15103 | 16101 | CD | 2631 | 201 | 297 | 731 | 3860 |
| M2 | 256 | 15103 | 16101 | STTC | 2487 | 431 | 468 | 474 | 3860 |
| M2 | 256 | 15103 | 16102 | CD | 2645 | 187 | 283 | 745 | 3860 |
| M2 | 256 | 15103 | 16102 | STTC | 2603 | 315 | 601 | 341 | 3860 |
| M2 | 256 | 15103 | 16103 | CD | 2615 | 217 | 313 | 715 | 3860 |
| M2 | 256 | 15103 | 16103 | STTC | 2526 | 392 | 583 | 359 | 3860 |
| M3 | 64 | 15101 | 16101 | CD | 2612 | 220 | 349 | 679 | 3860 |
| M3 | 64 | 15101 | 16101 | STTC | 2233 | 685 | 349 | 593 | 3860 |
| M3 | 64 | 15101 | 16102 | CD | 2531 | 301 | 209 | 819 | 3860 |
| M3 | 64 | 15101 | 16102 | STTC | 2499 | 419 | 471 | 471 | 3860 |
| M3 | 64 | 15101 | 16103 | CD | 2613 | 219 | 299 | 729 | 3860 |
| M3 | 64 | 15101 | 16103 | STTC | 2549 | 369 | 543 | 399 | 3860 |
| M3 | 64 | 15102 | 16101 | CD | 2631 | 201 | 326 | 702 | 3860 |
| M3 | 64 | 15102 | 16101 | STTC | 2181 | 737 | 343 | 599 | 3860 |
| M3 | 64 | 15102 | 16102 | CD | 2373 | 459 | 133 | 895 | 3860 |
| M3 | 64 | 15102 | 16102 | STTC | 2541 | 377 | 521 | 421 | 3860 |
| M3 | 64 | 15102 | 16103 | CD | 2628 | 204 | 230 | 798 | 3860 |
| M3 | 64 | 15102 | 16103 | STTC | 2510 | 408 | 555 | 387 | 3860 |
| M3 | 64 | 15103 | 16101 | CD | 2649 | 183 | 372 | 656 | 3860 |
| M3 | 64 | 15103 | 16101 | STTC | 2235 | 683 | 356 | 586 | 3860 |
| M3 | 64 | 15103 | 16102 | CD | 2462 | 370 | 211 | 817 | 3860 |
| M3 | 64 | 15103 | 16102 | STTC | 2534 | 384 | 577 | 365 | 3860 |
| M3 | 64 | 15103 | 16103 | CD | 2612 | 220 | 348 | 680 | 3860 |
| M3 | 64 | 15103 | 16103 | STTC | 2489 | 429 | 653 | 289 | 3860 |
| M3 | 128 | 15101 | 16101 | CD | 2630 | 202 | 320 | 708 | 3860 |
| M3 | 128 | 15101 | 16101 | STTC | 2425 | 493 | 394 | 548 | 3860 |
| M3 | 128 | 15101 | 16102 | CD | 2524 | 308 | 188 | 840 | 3860 |
| M3 | 128 | 15101 | 16102 | STTC | 2714 | 204 | 583 | 359 | 3860 |
| M3 | 128 | 15101 | 16103 | CD | 2595 | 237 | 274 | 754 | 3860 |
| M3 | 128 | 15101 | 16103 | STTC | 2488 | 430 | 421 | 521 | 3860 |
| M3 | 128 | 15102 | 16101 | CD | 2627 | 205 | 215 | 813 | 3860 |
| M3 | 128 | 15102 | 16101 | STTC | 2437 | 481 | 399 | 543 | 3860 |
| M3 | 128 | 15102 | 16102 | CD | 2513 | 319 | 168 | 860 | 3860 |
| M3 | 128 | 15102 | 16102 | STTC | 2713 | 205 | 636 | 306 | 3860 |
| M3 | 128 | 15102 | 16103 | CD | 2619 | 213 | 205 | 823 | 3860 |
| M3 | 128 | 15102 | 16103 | STTC | 2420 | 498 | 433 | 509 | 3860 |
| M3 | 128 | 15103 | 16101 | CD | 2673 | 159 | 368 | 660 | 3860 |
| M3 | 128 | 15103 | 16101 | STTC | 2450 | 468 | 463 | 479 | 3860 |
| M3 | 128 | 15103 | 16102 | CD | 2528 | 304 | 186 | 842 | 3860 |
| M3 | 128 | 15103 | 16102 | STTC | 2794 | 124 | 821 | 121 | 3860 |
| M3 | 128 | 15103 | 16103 | CD | 2669 | 163 | 380 | 648 | 3860 |
| M3 | 128 | 15103 | 16103 | STTC | 2482 | 436 | 553 | 389 | 3860 |
| M3 | 256 | 15101 | 16101 | CD | 2610 | 222 | 273 | 755 | 3860 |
| M3 | 256 | 15101 | 16101 | STTC | 2575 | 343 | 447 | 495 | 3860 |
| M3 | 256 | 15101 | 16102 | CD | 2590 | 242 | 221 | 807 | 3860 |
| M3 | 256 | 15101 | 16102 | STTC | 2603 | 315 | 488 | 454 | 3860 |
| M3 | 256 | 15101 | 16103 | CD | 2592 | 240 | 250 | 778 | 3860 |
| M3 | 256 | 15101 | 16103 | STTC | 2436 | 482 | 413 | 529 | 3860 |
| M3 | 256 | 15102 | 16101 | CD | 2634 | 198 | 181 | 847 | 3860 |
| M3 | 256 | 15102 | 16101 | STTC | 2495 | 423 | 424 | 518 | 3860 |
| M3 | 256 | 15102 | 16102 | CD | 2592 | 240 | 161 | 867 | 3860 |
| M3 | 256 | 15102 | 16102 | STTC | 2581 | 337 | 501 | 441 | 3860 |
| M3 | 256 | 15102 | 16103 | CD | 2660 | 172 | 215 | 813 | 3860 |
| M3 | 256 | 15102 | 16103 | STTC | 2409 | 509 | 441 | 501 | 3860 |
| M3 | 256 | 15103 | 16101 | CD | 2683 | 149 | 330 | 698 | 3860 |
| M3 | 256 | 15103 | 16101 | STTC | 2555 | 363 | 538 | 404 | 3860 |
| M3 | 256 | 15103 | 16102 | CD | 2644 | 188 | 282 | 746 | 3860 |
| M3 | 256 | 15103 | 16102 | STTC | 2624 | 294 | 597 | 345 | 3860 |
| M3 | 256 | 15103 | 16103 | CD | 2648 | 184 | 315 | 713 | 3860 |
| M3 | 256 | 15103 | 16103 | STTC | 2478 | 440 | 543 | 399 | 3860 |
| M4 | 64 | 15101 | 16101 | CD | 2619 | 213 | 349 | 679 | 3860 |
| M4 | 64 | 15101 | 16101 | STTC | 2216 | 702 | 364 | 578 | 3860 |
| M4 | 64 | 15101 | 16102 | CD | 2432 | 400 | 172 | 856 | 3860 |
| M4 | 64 | 15101 | 16102 | STTC | 2583 | 335 | 551 | 391 | 3860 |
| M4 | 64 | 15101 | 16103 | CD | 2688 | 144 | 352 | 676 | 3860 |
| M4 | 64 | 15101 | 16103 | STTC | 2465 | 453 | 543 | 399 | 3860 |
| M4 | 64 | 15102 | 16101 | CD | 2652 | 180 | 351 | 677 | 3860 |
| M4 | 64 | 15102 | 16101 | STTC | 2319 | 599 | 368 | 574 | 3860 |
| M4 | 64 | 15102 | 16102 | CD | 2495 | 337 | 205 | 823 | 3860 |
| M4 | 64 | 15102 | 16102 | STTC | 2534 | 384 | 561 | 381 | 3860 |
| M4 | 64 | 15102 | 16103 | CD | 2669 | 163 | 343 | 685 | 3860 |
| M4 | 64 | 15102 | 16103 | STTC | 2544 | 374 | 564 | 378 | 3860 |
| M4 | 64 | 15103 | 16101 | CD | 2609 | 223 | 352 | 676 | 3860 |
| M4 | 64 | 15103 | 16101 | STTC | 2341 | 577 | 288 | 654 | 3860 |
| M4 | 64 | 15103 | 16102 | CD | 2385 | 447 | 172 | 856 | 3860 |
| M4 | 64 | 15103 | 16102 | STTC | 2543 | 375 | 494 | 448 | 3860 |
| M4 | 64 | 15103 | 16103 | CD | 2610 | 222 | 279 | 749 | 3860 |
| M4 | 64 | 15103 | 16103 | STTC | 2566 | 352 | 563 | 379 | 3860 |
| M4 | 128 | 15101 | 16101 | CD | 2692 | 140 | 375 | 653 | 3860 |
| M4 | 128 | 15101 | 16101 | STTC | 2519 | 399 | 480 | 462 | 3860 |
| M4 | 128 | 15101 | 16102 | CD | 2507 | 325 | 174 | 854 | 3860 |
| M4 | 128 | 15101 | 16102 | STTC | 2748 | 170 | 668 | 274 | 3860 |
| M4 | 128 | 15101 | 16103 | CD | 2648 | 184 | 362 | 666 | 3860 |
| M4 | 128 | 15101 | 16103 | STTC | 2405 | 513 | 483 | 459 | 3860 |
| M4 | 128 | 15102 | 16101 | CD | 2670 | 162 | 335 | 693 | 3860 |
| M4 | 128 | 15102 | 16101 | STTC | 2632 | 286 | 527 | 415 | 3860 |
| M4 | 128 | 15102 | 16102 | CD | 2515 | 317 | 193 | 835 | 3860 |
| M4 | 128 | 15102 | 16102 | STTC | 2800 | 118 | 752 | 190 | 3860 |
| M4 | 128 | 15102 | 16103 | CD | 2602 | 230 | 256 | 772 | 3860 |
| M4 | 128 | 15102 | 16103 | STTC | 2307 | 611 | 354 | 588 | 3860 |
| M4 | 128 | 15103 | 16101 | CD | 2650 | 182 | 358 | 670 | 3860 |
| M4 | 128 | 15103 | 16101 | STTC | 2598 | 320 | 431 | 511 | 3860 |
| M4 | 128 | 15103 | 16102 | CD | 2522 | 310 | 186 | 842 | 3860 |
| M4 | 128 | 15103 | 16102 | STTC | 2777 | 141 | 676 | 266 | 3860 |
| M4 | 128 | 15103 | 16103 | CD | 2600 | 232 | 289 | 739 | 3860 |
| M4 | 128 | 15103 | 16103 | STTC | 2574 | 344 | 490 | 452 | 3860 |
| M4 | 256 | 15101 | 16101 | CD | 2632 | 200 | 298 | 730 | 3860 |
| M4 | 256 | 15101 | 16101 | STTC | 2584 | 334 | 480 | 462 | 3860 |
| M4 | 256 | 15101 | 16102 | CD | 2612 | 220 | 230 | 798 | 3860 |
| M4 | 256 | 15101 | 16102 | STTC | 2645 | 273 | 521 | 421 | 3860 |
| M4 | 256 | 15101 | 16103 | CD | 2600 | 232 | 305 | 723 | 3860 |
| M4 | 256 | 15101 | 16103 | STTC | 2464 | 454 | 470 | 472 | 3860 |
| M4 | 256 | 15102 | 16101 | CD | 2665 | 167 | 257 | 771 | 3860 |
| M4 | 256 | 15102 | 16101 | STTC | 2588 | 330 | 446 | 496 | 3860 |
| M4 | 256 | 15102 | 16102 | CD | 2615 | 217 | 212 | 816 | 3860 |
| M4 | 256 | 15102 | 16102 | STTC | 2637 | 281 | 541 | 401 | 3860 |
| M4 | 256 | 15102 | 16103 | CD | 2642 | 190 | 243 | 785 | 3860 |
| M4 | 256 | 15102 | 16103 | STTC | 2450 | 468 | 462 | 480 | 3860 |
| M4 | 256 | 15103 | 16101 | CD | 2641 | 191 | 306 | 722 | 3860 |
| M4 | 256 | 15103 | 16101 | STTC | 2601 | 317 | 400 | 542 | 3860 |
| M4 | 256 | 15103 | 16102 | CD | 2596 | 236 | 217 | 811 | 3860 |
| M4 | 256 | 15103 | 16102 | STTC | 2574 | 344 | 419 | 523 | 3860 |
| M4 | 256 | 15103 | 16103 | CD | 2589 | 243 | 258 | 770 | 3860 |
| M4 | 256 | 15103 | 16103 | STTC | 2513 | 405 | 462 | 480 | 3860 |
| M5 | 64 | 15101 | 16101 | CD | 2659 | 173 | 379 | 649 | 3860 |
| M5 | 64 | 15101 | 16101 | STTC | 2269 | 649 | 325 | 617 | 3860 |
| M5 | 64 | 15101 | 16102 | CD | 2485 | 347 | 199 | 829 | 3860 |
| M5 | 64 | 15101 | 16102 | STTC | 2514 | 404 | 459 | 483 | 3860 |
| M5 | 64 | 15101 | 16103 | CD | 2614 | 218 | 256 | 772 | 3860 |
| M5 | 64 | 15101 | 16103 | STTC | 2559 | 359 | 500 | 442 | 3860 |
| M5 | 64 | 15102 | 16101 | CD | 2632 | 200 | 324 | 704 | 3860 |
| M5 | 64 | 15102 | 16101 | STTC | 2176 | 742 | 319 | 623 | 3860 |
| M5 | 64 | 15102 | 16102 | CD | 2464 | 368 | 172 | 856 | 3860 |
| M5 | 64 | 15102 | 16102 | STTC | 2567 | 351 | 561 | 381 | 3860 |
| M5 | 64 | 15102 | 16103 | CD | 2612 | 220 | 249 | 779 | 3860 |
| M5 | 64 | 15102 | 16103 | STTC | 2491 | 427 | 525 | 417 | 3860 |
| M5 | 64 | 15103 | 16101 | CD | 2619 | 213 | 364 | 664 | 3860 |
| M5 | 64 | 15103 | 16101 | STTC | 2299 | 619 | 309 | 633 | 3860 |
| M5 | 64 | 15103 | 16102 | CD | 2460 | 372 | 180 | 848 | 3860 |
| M5 | 64 | 15103 | 16102 | STTC | 2492 | 426 | 469 | 473 | 3860 |
| M5 | 64 | 15103 | 16103 | CD | 2643 | 189 | 286 | 742 | 3860 |
| M5 | 64 | 15103 | 16103 | STTC | 2458 | 460 | 522 | 420 | 3860 |
| M5 | 128 | 15101 | 16101 | CD | 2698 | 134 | 374 | 654 | 3860 |
| M5 | 128 | 15101 | 16101 | STTC | 2510 | 408 | 428 | 514 | 3860 |
| M5 | 128 | 15101 | 16102 | CD | 2564 | 268 | 207 | 821 | 3860 |
| M5 | 128 | 15101 | 16102 | STTC | 2736 | 182 | 635 | 307 | 3860 |
| M5 | 128 | 15101 | 16103 | CD | 2588 | 244 | 248 | 780 | 3860 |
| M5 | 128 | 15101 | 16103 | STTC | 2496 | 422 | 468 | 474 | 3860 |
| M5 | 128 | 15102 | 16101 | CD | 2631 | 201 | 245 | 783 | 3860 |
| M5 | 128 | 15102 | 16101 | STTC | 2519 | 399 | 500 | 442 | 3860 |
| M5 | 128 | 15102 | 16102 | CD | 2572 | 260 | 183 | 845 | 3860 |
| M5 | 128 | 15102 | 16102 | STTC | 2817 | 101 | 800 | 142 | 3860 |
| M5 | 128 | 15102 | 16103 | CD | 2609 | 223 | 220 | 808 | 3860 |
| M5 | 128 | 15102 | 16103 | STTC | 2291 | 627 | 364 | 578 | 3860 |
| M5 | 128 | 15103 | 16101 | CD | 2672 | 160 | 365 | 663 | 3860 |
| M5 | 128 | 15103 | 16101 | STTC | 2543 | 375 | 377 | 565 | 3860 |
| M5 | 128 | 15103 | 16102 | CD | 2532 | 300 | 186 | 842 | 3860 |
| M5 | 128 | 15103 | 16102 | STTC | 2750 | 168 | 606 | 336 | 3860 |
| M5 | 128 | 15103 | 16103 | CD | 2622 | 210 | 277 | 751 | 3860 |
| M5 | 128 | 15103 | 16103 | STTC | 2502 | 416 | 433 | 509 | 3860 |
| M5 | 256 | 15101 | 16101 | CD | 2625 | 207 | 276 | 752 | 3860 |
| M5 | 256 | 15101 | 16101 | STTC | 2603 | 315 | 470 | 472 | 3860 |
| M5 | 256 | 15101 | 16102 | CD | 2642 | 190 | 231 | 797 | 3860 |
| M5 | 256 | 15101 | 16102 | STTC | 2643 | 275 | 495 | 447 | 3860 |
| M5 | 256 | 15101 | 16103 | CD | 2630 | 202 | 261 | 767 | 3860 |
| M5 | 256 | 15101 | 16103 | STTC | 2439 | 479 | 422 | 520 | 3860 |
| M5 | 256 | 15102 | 16101 | CD | 2638 | 194 | 216 | 812 | 3860 |
| M5 | 256 | 15102 | 16101 | STTC | 2551 | 367 | 458 | 484 | 3860 |
| M5 | 256 | 15102 | 16102 | CD | 2648 | 184 | 206 | 822 | 3860 |
| M5 | 256 | 15102 | 16102 | STTC | 2699 | 219 | 600 | 342 | 3860 |
| M5 | 256 | 15102 | 16103 | CD | 2642 | 190 | 232 | 796 | 3860 |
| M5 | 256 | 15102 | 16103 | STTC | 2390 | 528 | 431 | 511 | 3860 |
| M5 | 256 | 15103 | 16101 | CD | 2656 | 176 | 287 | 741 | 3860 |
| M5 | 256 | 15103 | 16101 | STTC | 2524 | 394 | 355 | 587 | 3860 |
| M5 | 256 | 15103 | 16102 | CD | 2631 | 201 | 229 | 799 | 3860 |
| M5 | 256 | 15103 | 16102 | STTC | 2529 | 389 | 413 | 529 | 3860 |
| M5 | 256 | 15103 | 16103 | CD | 2624 | 208 | 246 | 782 | 3860 |
| M5 | 256 | 15103 | 16103 | STTC | 2452 | 466 | 373 | 569 | 3860 |

## Appendix D — Phase 5 per-episode counts

Threshold 0.5 throughout. Source seed and draw seed identify fitted-run slots, not participants. Source-only repeated slots are aliases, not additional fits.

| Method | Budget | Source seed | Draw seed | Class | TN | FP | FN | TP | N |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| a | 64 | 15101 | 16101 | CD | 2792 | 569 | 251 | 469 | 4081 |
| a | 64 | 15101 | 16101 | STTC | 2130 | 175 | 1308 | 468 | 4081 |
| a | 64 | 15101 | 16102 | CD | 2792 | 569 | 251 | 469 | 4081 |
| a | 64 | 15101 | 16102 | STTC | 2130 | 175 | 1308 | 468 | 4081 |
| a | 64 | 15101 | 16103 | CD | 2792 | 569 | 251 | 469 | 4081 |
| a | 64 | 15101 | 16103 | STTC | 2130 | 175 | 1308 | 468 | 4081 |
| a | 64 | 15102 | 16101 | CD | 2671 | 690 | 119 | 601 | 4081 |
| a | 64 | 15102 | 16101 | STTC | 2022 | 283 | 1081 | 695 | 4081 |
| a | 64 | 15102 | 16102 | CD | 2671 | 690 | 119 | 601 | 4081 |
| a | 64 | 15102 | 16102 | STTC | 2022 | 283 | 1081 | 695 | 4081 |
| a | 64 | 15102 | 16103 | CD | 2671 | 690 | 119 | 601 | 4081 |
| a | 64 | 15102 | 16103 | STTC | 2022 | 283 | 1081 | 695 | 4081 |
| a | 64 | 15103 | 16101 | CD | 2785 | 576 | 215 | 505 | 4081 |
| a | 64 | 15103 | 16101 | STTC | 2167 | 138 | 1278 | 498 | 4081 |
| a | 64 | 15103 | 16102 | CD | 2785 | 576 | 215 | 505 | 4081 |
| a | 64 | 15103 | 16102 | STTC | 2167 | 138 | 1278 | 498 | 4081 |
| a | 64 | 15103 | 16103 | CD | 2785 | 576 | 215 | 505 | 4081 |
| a | 64 | 15103 | 16103 | STTC | 2167 | 138 | 1278 | 498 | 4081 |
| a | 64 | 15104 | 16101 | CD | 2802 | 559 | 217 | 503 | 4081 |
| a | 64 | 15104 | 16101 | STTC | 1870 | 435 | 854 | 922 | 4081 |
| a | 64 | 15104 | 16102 | CD | 2802 | 559 | 217 | 503 | 4081 |
| a | 64 | 15104 | 16102 | STTC | 1870 | 435 | 854 | 922 | 4081 |
| a | 64 | 15104 | 16103 | CD | 2802 | 559 | 217 | 503 | 4081 |
| a | 64 | 15104 | 16103 | STTC | 1870 | 435 | 854 | 922 | 4081 |
| a | 64 | 15105 | 16101 | CD | 2653 | 708 | 144 | 576 | 4081 |
| a | 64 | 15105 | 16101 | STTC | 2158 | 147 | 1265 | 511 | 4081 |
| a | 64 | 15105 | 16102 | CD | 2653 | 708 | 144 | 576 | 4081 |
| a | 64 | 15105 | 16102 | STTC | 2158 | 147 | 1265 | 511 | 4081 |
| a | 64 | 15105 | 16103 | CD | 2653 | 708 | 144 | 576 | 4081 |
| a | 64 | 15105 | 16103 | STTC | 2158 | 147 | 1265 | 511 | 4081 |
| a | 64 | 15106 | 16101 | CD | 2701 | 660 | 137 | 583 | 4081 |
| a | 64 | 15106 | 16101 | STTC | 2072 | 233 | 1129 | 647 | 4081 |
| a | 64 | 15106 | 16102 | CD | 2701 | 660 | 137 | 583 | 4081 |
| a | 64 | 15106 | 16102 | STTC | 2072 | 233 | 1129 | 647 | 4081 |
| a | 64 | 15106 | 16103 | CD | 2701 | 660 | 137 | 583 | 4081 |
| a | 64 | 15106 | 16103 | STTC | 2072 | 233 | 1129 | 647 | 4081 |
| a | 64 | 15107 | 16101 | CD | 2759 | 602 | 228 | 492 | 4081 |
| a | 64 | 15107 | 16101 | STTC | 2085 | 220 | 1106 | 670 | 4081 |
| a | 64 | 15107 | 16102 | CD | 2759 | 602 | 228 | 492 | 4081 |
| a | 64 | 15107 | 16102 | STTC | 2085 | 220 | 1106 | 670 | 4081 |
| a | 64 | 15107 | 16103 | CD | 2759 | 602 | 228 | 492 | 4081 |
| a | 64 | 15107 | 16103 | STTC | 2085 | 220 | 1106 | 670 | 4081 |
| a | 64 | 15108 | 16101 | CD | 2783 | 578 | 219 | 501 | 4081 |
| a | 64 | 15108 | 16101 | STTC | 2045 | 260 | 1075 | 701 | 4081 |
| a | 64 | 15108 | 16102 | CD | 2783 | 578 | 219 | 501 | 4081 |
| a | 64 | 15108 | 16102 | STTC | 2045 | 260 | 1075 | 701 | 4081 |
| a | 64 | 15108 | 16103 | CD | 2783 | 578 | 219 | 501 | 4081 |
| a | 64 | 15108 | 16103 | STTC | 2045 | 260 | 1075 | 701 | 4081 |
| a | 64 | 15109 | 16101 | CD | 2798 | 563 | 226 | 494 | 4081 |
| a | 64 | 15109 | 16101 | STTC | 2246 | 59 | 1565 | 211 | 4081 |
| a | 64 | 15109 | 16102 | CD | 2798 | 563 | 226 | 494 | 4081 |
| a | 64 | 15109 | 16102 | STTC | 2246 | 59 | 1565 | 211 | 4081 |
| a | 64 | 15109 | 16103 | CD | 2798 | 563 | 226 | 494 | 4081 |
| a | 64 | 15109 | 16103 | STTC | 2246 | 59 | 1565 | 211 | 4081 |
| a | 64 | 15110 | 16101 | CD | 2808 | 553 | 196 | 524 | 4081 |
| a | 64 | 15110 | 16101 | STTC | 2086 | 219 | 1191 | 585 | 4081 |
| a | 64 | 15110 | 16102 | CD | 2808 | 553 | 196 | 524 | 4081 |
| a | 64 | 15110 | 16102 | STTC | 2086 | 219 | 1191 | 585 | 4081 |
| a | 64 | 15110 | 16103 | CD | 2808 | 553 | 196 | 524 | 4081 |
| a | 64 | 15110 | 16103 | STTC | 2086 | 219 | 1191 | 585 | 4081 |
| a | 128 | 15101 | 16101 | CD | 2792 | 569 | 251 | 469 | 4081 |
| a | 128 | 15101 | 16101 | STTC | 2130 | 175 | 1308 | 468 | 4081 |
| a | 128 | 15101 | 16102 | CD | 2792 | 569 | 251 | 469 | 4081 |
| a | 128 | 15101 | 16102 | STTC | 2130 | 175 | 1308 | 468 | 4081 |
| a | 128 | 15101 | 16103 | CD | 2792 | 569 | 251 | 469 | 4081 |
| a | 128 | 15101 | 16103 | STTC | 2130 | 175 | 1308 | 468 | 4081 |
| a | 128 | 15102 | 16101 | CD | 2671 | 690 | 119 | 601 | 4081 |
| a | 128 | 15102 | 16101 | STTC | 2022 | 283 | 1081 | 695 | 4081 |
| a | 128 | 15102 | 16102 | CD | 2671 | 690 | 119 | 601 | 4081 |
| a | 128 | 15102 | 16102 | STTC | 2022 | 283 | 1081 | 695 | 4081 |
| a | 128 | 15102 | 16103 | CD | 2671 | 690 | 119 | 601 | 4081 |
| a | 128 | 15102 | 16103 | STTC | 2022 | 283 | 1081 | 695 | 4081 |
| a | 128 | 15103 | 16101 | CD | 2785 | 576 | 215 | 505 | 4081 |
| a | 128 | 15103 | 16101 | STTC | 2167 | 138 | 1278 | 498 | 4081 |
| a | 128 | 15103 | 16102 | CD | 2785 | 576 | 215 | 505 | 4081 |
| a | 128 | 15103 | 16102 | STTC | 2167 | 138 | 1278 | 498 | 4081 |
| a | 128 | 15103 | 16103 | CD | 2785 | 576 | 215 | 505 | 4081 |
| a | 128 | 15103 | 16103 | STTC | 2167 | 138 | 1278 | 498 | 4081 |
| a | 128 | 15104 | 16101 | CD | 2802 | 559 | 217 | 503 | 4081 |
| a | 128 | 15104 | 16101 | STTC | 1870 | 435 | 854 | 922 | 4081 |
| a | 128 | 15104 | 16102 | CD | 2802 | 559 | 217 | 503 | 4081 |
| a | 128 | 15104 | 16102 | STTC | 1870 | 435 | 854 | 922 | 4081 |
| a | 128 | 15104 | 16103 | CD | 2802 | 559 | 217 | 503 | 4081 |
| a | 128 | 15104 | 16103 | STTC | 1870 | 435 | 854 | 922 | 4081 |
| a | 128 | 15105 | 16101 | CD | 2653 | 708 | 144 | 576 | 4081 |
| a | 128 | 15105 | 16101 | STTC | 2158 | 147 | 1265 | 511 | 4081 |
| a | 128 | 15105 | 16102 | CD | 2653 | 708 | 144 | 576 | 4081 |
| a | 128 | 15105 | 16102 | STTC | 2158 | 147 | 1265 | 511 | 4081 |
| a | 128 | 15105 | 16103 | CD | 2653 | 708 | 144 | 576 | 4081 |
| a | 128 | 15105 | 16103 | STTC | 2158 | 147 | 1265 | 511 | 4081 |
| a | 128 | 15106 | 16101 | CD | 2701 | 660 | 137 | 583 | 4081 |
| a | 128 | 15106 | 16101 | STTC | 2072 | 233 | 1129 | 647 | 4081 |
| a | 128 | 15106 | 16102 | CD | 2701 | 660 | 137 | 583 | 4081 |
| a | 128 | 15106 | 16102 | STTC | 2072 | 233 | 1129 | 647 | 4081 |
| a | 128 | 15106 | 16103 | CD | 2701 | 660 | 137 | 583 | 4081 |
| a | 128 | 15106 | 16103 | STTC | 2072 | 233 | 1129 | 647 | 4081 |
| a | 128 | 15107 | 16101 | CD | 2759 | 602 | 228 | 492 | 4081 |
| a | 128 | 15107 | 16101 | STTC | 2085 | 220 | 1106 | 670 | 4081 |
| a | 128 | 15107 | 16102 | CD | 2759 | 602 | 228 | 492 | 4081 |
| a | 128 | 15107 | 16102 | STTC | 2085 | 220 | 1106 | 670 | 4081 |
| a | 128 | 15107 | 16103 | CD | 2759 | 602 | 228 | 492 | 4081 |
| a | 128 | 15107 | 16103 | STTC | 2085 | 220 | 1106 | 670 | 4081 |
| a | 128 | 15108 | 16101 | CD | 2783 | 578 | 219 | 501 | 4081 |
| a | 128 | 15108 | 16101 | STTC | 2045 | 260 | 1075 | 701 | 4081 |
| a | 128 | 15108 | 16102 | CD | 2783 | 578 | 219 | 501 | 4081 |
| a | 128 | 15108 | 16102 | STTC | 2045 | 260 | 1075 | 701 | 4081 |
| a | 128 | 15108 | 16103 | CD | 2783 | 578 | 219 | 501 | 4081 |
| a | 128 | 15108 | 16103 | STTC | 2045 | 260 | 1075 | 701 | 4081 |
| a | 128 | 15109 | 16101 | CD | 2798 | 563 | 226 | 494 | 4081 |
| a | 128 | 15109 | 16101 | STTC | 2246 | 59 | 1565 | 211 | 4081 |
| a | 128 | 15109 | 16102 | CD | 2798 | 563 | 226 | 494 | 4081 |
| a | 128 | 15109 | 16102 | STTC | 2246 | 59 | 1565 | 211 | 4081 |
| a | 128 | 15109 | 16103 | CD | 2798 | 563 | 226 | 494 | 4081 |
| a | 128 | 15109 | 16103 | STTC | 2246 | 59 | 1565 | 211 | 4081 |
| a | 128 | 15110 | 16101 | CD | 2808 | 553 | 196 | 524 | 4081 |
| a | 128 | 15110 | 16101 | STTC | 2086 | 219 | 1191 | 585 | 4081 |
| a | 128 | 15110 | 16102 | CD | 2808 | 553 | 196 | 524 | 4081 |
| a | 128 | 15110 | 16102 | STTC | 2086 | 219 | 1191 | 585 | 4081 |
| a | 128 | 15110 | 16103 | CD | 2808 | 553 | 196 | 524 | 4081 |
| a | 128 | 15110 | 16103 | STTC | 2086 | 219 | 1191 | 585 | 4081 |
| a | 256 | 15101 | 16101 | CD | 2792 | 569 | 251 | 469 | 4081 |
| a | 256 | 15101 | 16101 | STTC | 2130 | 175 | 1308 | 468 | 4081 |
| a | 256 | 15101 | 16102 | CD | 2792 | 569 | 251 | 469 | 4081 |
| a | 256 | 15101 | 16102 | STTC | 2130 | 175 | 1308 | 468 | 4081 |
| a | 256 | 15101 | 16103 | CD | 2792 | 569 | 251 | 469 | 4081 |
| a | 256 | 15101 | 16103 | STTC | 2130 | 175 | 1308 | 468 | 4081 |
| a | 256 | 15102 | 16101 | CD | 2671 | 690 | 119 | 601 | 4081 |
| a | 256 | 15102 | 16101 | STTC | 2022 | 283 | 1081 | 695 | 4081 |
| a | 256 | 15102 | 16102 | CD | 2671 | 690 | 119 | 601 | 4081 |
| a | 256 | 15102 | 16102 | STTC | 2022 | 283 | 1081 | 695 | 4081 |
| a | 256 | 15102 | 16103 | CD | 2671 | 690 | 119 | 601 | 4081 |
| a | 256 | 15102 | 16103 | STTC | 2022 | 283 | 1081 | 695 | 4081 |
| a | 256 | 15103 | 16101 | CD | 2785 | 576 | 215 | 505 | 4081 |
| a | 256 | 15103 | 16101 | STTC | 2167 | 138 | 1278 | 498 | 4081 |
| a | 256 | 15103 | 16102 | CD | 2785 | 576 | 215 | 505 | 4081 |
| a | 256 | 15103 | 16102 | STTC | 2167 | 138 | 1278 | 498 | 4081 |
| a | 256 | 15103 | 16103 | CD | 2785 | 576 | 215 | 505 | 4081 |
| a | 256 | 15103 | 16103 | STTC | 2167 | 138 | 1278 | 498 | 4081 |
| a | 256 | 15104 | 16101 | CD | 2802 | 559 | 217 | 503 | 4081 |
| a | 256 | 15104 | 16101 | STTC | 1870 | 435 | 854 | 922 | 4081 |
| a | 256 | 15104 | 16102 | CD | 2802 | 559 | 217 | 503 | 4081 |
| a | 256 | 15104 | 16102 | STTC | 1870 | 435 | 854 | 922 | 4081 |
| a | 256 | 15104 | 16103 | CD | 2802 | 559 | 217 | 503 | 4081 |
| a | 256 | 15104 | 16103 | STTC | 1870 | 435 | 854 | 922 | 4081 |
| a | 256 | 15105 | 16101 | CD | 2653 | 708 | 144 | 576 | 4081 |
| a | 256 | 15105 | 16101 | STTC | 2158 | 147 | 1265 | 511 | 4081 |
| a | 256 | 15105 | 16102 | CD | 2653 | 708 | 144 | 576 | 4081 |
| a | 256 | 15105 | 16102 | STTC | 2158 | 147 | 1265 | 511 | 4081 |
| a | 256 | 15105 | 16103 | CD | 2653 | 708 | 144 | 576 | 4081 |
| a | 256 | 15105 | 16103 | STTC | 2158 | 147 | 1265 | 511 | 4081 |
| a | 256 | 15106 | 16101 | CD | 2701 | 660 | 137 | 583 | 4081 |
| a | 256 | 15106 | 16101 | STTC | 2072 | 233 | 1129 | 647 | 4081 |
| a | 256 | 15106 | 16102 | CD | 2701 | 660 | 137 | 583 | 4081 |
| a | 256 | 15106 | 16102 | STTC | 2072 | 233 | 1129 | 647 | 4081 |
| a | 256 | 15106 | 16103 | CD | 2701 | 660 | 137 | 583 | 4081 |
| a | 256 | 15106 | 16103 | STTC | 2072 | 233 | 1129 | 647 | 4081 |
| a | 256 | 15107 | 16101 | CD | 2759 | 602 | 228 | 492 | 4081 |
| a | 256 | 15107 | 16101 | STTC | 2085 | 220 | 1106 | 670 | 4081 |
| a | 256 | 15107 | 16102 | CD | 2759 | 602 | 228 | 492 | 4081 |
| a | 256 | 15107 | 16102 | STTC | 2085 | 220 | 1106 | 670 | 4081 |
| a | 256 | 15107 | 16103 | CD | 2759 | 602 | 228 | 492 | 4081 |
| a | 256 | 15107 | 16103 | STTC | 2085 | 220 | 1106 | 670 | 4081 |
| a | 256 | 15108 | 16101 | CD | 2783 | 578 | 219 | 501 | 4081 |
| a | 256 | 15108 | 16101 | STTC | 2045 | 260 | 1075 | 701 | 4081 |
| a | 256 | 15108 | 16102 | CD | 2783 | 578 | 219 | 501 | 4081 |
| a | 256 | 15108 | 16102 | STTC | 2045 | 260 | 1075 | 701 | 4081 |
| a | 256 | 15108 | 16103 | CD | 2783 | 578 | 219 | 501 | 4081 |
| a | 256 | 15108 | 16103 | STTC | 2045 | 260 | 1075 | 701 | 4081 |
| a | 256 | 15109 | 16101 | CD | 2798 | 563 | 226 | 494 | 4081 |
| a | 256 | 15109 | 16101 | STTC | 2246 | 59 | 1565 | 211 | 4081 |
| a | 256 | 15109 | 16102 | CD | 2798 | 563 | 226 | 494 | 4081 |
| a | 256 | 15109 | 16102 | STTC | 2246 | 59 | 1565 | 211 | 4081 |
| a | 256 | 15109 | 16103 | CD | 2798 | 563 | 226 | 494 | 4081 |
| a | 256 | 15109 | 16103 | STTC | 2246 | 59 | 1565 | 211 | 4081 |
| a | 256 | 15110 | 16101 | CD | 2808 | 553 | 196 | 524 | 4081 |
| a | 256 | 15110 | 16101 | STTC | 2086 | 219 | 1191 | 585 | 4081 |
| a | 256 | 15110 | 16102 | CD | 2808 | 553 | 196 | 524 | 4081 |
| a | 256 | 15110 | 16102 | STTC | 2086 | 219 | 1191 | 585 | 4081 |
| a | 256 | 15110 | 16103 | CD | 2808 | 553 | 196 | 524 | 4081 |
| a | 256 | 15110 | 16103 | STTC | 2086 | 219 | 1191 | 585 | 4081 |
| b | 64 | 15101 | 16101 | CD | 2788 | 573 | 139 | 581 | 4081 |
| b | 64 | 15101 | 16101 | STTC | 1637 | 668 | 553 | 1223 | 4081 |
| b | 64 | 15101 | 16102 | CD | 2974 | 387 | 266 | 454 | 4081 |
| b | 64 | 15101 | 16102 | STTC | 1630 | 675 | 548 | 1228 | 4081 |
| b | 64 | 15101 | 16103 | CD | 3006 | 355 | 274 | 446 | 4081 |
| b | 64 | 15101 | 16103 | STTC | 1647 | 658 | 509 | 1267 | 4081 |
| b | 64 | 15102 | 16101 | CD | 2844 | 517 | 151 | 569 | 4081 |
| b | 64 | 15102 | 16101 | STTC | 1614 | 691 | 483 | 1293 | 4081 |
| b | 64 | 15102 | 16102 | CD | 2982 | 379 | 224 | 496 | 4081 |
| b | 64 | 15102 | 16102 | STTC | 1641 | 664 | 467 | 1309 | 4081 |
| b | 64 | 15102 | 16103 | CD | 2962 | 399 | 204 | 516 | 4081 |
| b | 64 | 15102 | 16103 | STTC | 1615 | 690 | 485 | 1291 | 4081 |
| b | 64 | 15103 | 16101 | CD | 2711 | 650 | 150 | 570 | 4081 |
| b | 64 | 15103 | 16101 | STTC | 1676 | 629 | 535 | 1241 | 4081 |
| b | 64 | 15103 | 16102 | CD | 2979 | 382 | 275 | 445 | 4081 |
| b | 64 | 15103 | 16102 | STTC | 1695 | 610 | 491 | 1285 | 4081 |
| b | 64 | 15103 | 16103 | CD | 3062 | 299 | 314 | 406 | 4081 |
| b | 64 | 15103 | 16103 | STTC | 1699 | 606 | 565 | 1211 | 4081 |
| b | 64 | 15104 | 16101 | CD | 2741 | 620 | 152 | 568 | 4081 |
| b | 64 | 15104 | 16101 | STTC | 1644 | 661 | 539 | 1237 | 4081 |
| b | 64 | 15104 | 16102 | CD | 2896 | 465 | 214 | 506 | 4081 |
| b | 64 | 15104 | 16102 | STTC | 1690 | 615 | 585 | 1191 | 4081 |
| b | 64 | 15104 | 16103 | CD | 2918 | 443 | 238 | 482 | 4081 |
| b | 64 | 15104 | 16103 | STTC | 1788 | 517 | 656 | 1120 | 4081 |
| b | 64 | 15105 | 16101 | CD | 2853 | 508 | 177 | 543 | 4081 |
| b | 64 | 15105 | 16101 | STTC | 1747 | 558 | 598 | 1178 | 4081 |
| b | 64 | 15105 | 16102 | CD | 2879 | 482 | 212 | 508 | 4081 |
| b | 64 | 15105 | 16102 | STTC | 1579 | 726 | 444 | 1332 | 4081 |
| b | 64 | 15105 | 16103 | CD | 3004 | 357 | 271 | 449 | 4081 |
| b | 64 | 15105 | 16103 | STTC | 1665 | 640 | 526 | 1250 | 4081 |
| b | 64 | 15106 | 16101 | CD | 2856 | 505 | 145 | 575 | 4081 |
| b | 64 | 15106 | 16101 | STTC | 1764 | 541 | 614 | 1162 | 4081 |
| b | 64 | 15106 | 16102 | CD | 2941 | 420 | 219 | 501 | 4081 |
| b | 64 | 15106 | 16102 | STTC | 1735 | 570 | 531 | 1245 | 4081 |
| b | 64 | 15106 | 16103 | CD | 3013 | 348 | 224 | 496 | 4081 |
| b | 64 | 15106 | 16103 | STTC | 1725 | 580 | 605 | 1171 | 4081 |
| b | 64 | 15107 | 16101 | CD | 2765 | 596 | 144 | 576 | 4081 |
| b | 64 | 15107 | 16101 | STTC | 1618 | 687 | 544 | 1232 | 4081 |
| b | 64 | 15107 | 16102 | CD | 3159 | 202 | 406 | 314 | 4081 |
| b | 64 | 15107 | 16102 | STTC | 1615 | 690 | 440 | 1336 | 4081 |
| b | 64 | 15107 | 16103 | CD | 3031 | 330 | 282 | 438 | 4081 |
| b | 64 | 15107 | 16103 | STTC | 1618 | 687 | 457 | 1319 | 4081 |
| b | 64 | 15108 | 16101 | CD | 2789 | 572 | 139 | 581 | 4081 |
| b | 64 | 15108 | 16101 | STTC | 1662 | 643 | 475 | 1301 | 4081 |
| b | 64 | 15108 | 16102 | CD | 2952 | 409 | 244 | 476 | 4081 |
| b | 64 | 15108 | 16102 | STTC | 1642 | 663 | 496 | 1280 | 4081 |
| b | 64 | 15108 | 16103 | CD | 2998 | 363 | 279 | 441 | 4081 |
| b | 64 | 15108 | 16103 | STTC | 1700 | 605 | 574 | 1202 | 4081 |
| b | 64 | 15109 | 16101 | CD | 2759 | 602 | 128 | 592 | 4081 |
| b | 64 | 15109 | 16101 | STTC | 1580 | 725 | 571 | 1205 | 4081 |
| b | 64 | 15109 | 16102 | CD | 2998 | 363 | 292 | 428 | 4081 |
| b | 64 | 15109 | 16102 | STTC | 1604 | 701 | 472 | 1304 | 4081 |
| b | 64 | 15109 | 16103 | CD | 2943 | 418 | 242 | 478 | 4081 |
| b | 64 | 15109 | 16103 | STTC | 1638 | 667 | 548 | 1228 | 4081 |
| b | 64 | 15110 | 16101 | CD | 2732 | 629 | 131 | 589 | 4081 |
| b | 64 | 15110 | 16101 | STTC | 1706 | 599 | 611 | 1165 | 4081 |
| b | 64 | 15110 | 16102 | CD | 2890 | 471 | 211 | 509 | 4081 |
| b | 64 | 15110 | 16102 | STTC | 1772 | 533 | 653 | 1123 | 4081 |
| b | 64 | 15110 | 16103 | CD | 3006 | 355 | 254 | 466 | 4081 |
| b | 64 | 15110 | 16103 | STTC | 1716 | 589 | 598 | 1178 | 4081 |
| b | 128 | 15101 | 16101 | CD | 2939 | 422 | 217 | 503 | 4081 |
| b | 128 | 15101 | 16101 | STTC | 1548 | 757 | 487 | 1289 | 4081 |
| b | 128 | 15101 | 16102 | CD | 2942 | 419 | 240 | 480 | 4081 |
| b | 128 | 15101 | 16102 | STTC | 1782 | 523 | 722 | 1054 | 4081 |
| b | 128 | 15101 | 16103 | CD | 3034 | 327 | 292 | 428 | 4081 |
| b | 128 | 15101 | 16103 | STTC | 1604 | 701 | 471 | 1305 | 4081 |
| b | 128 | 15102 | 16101 | CD | 2971 | 390 | 193 | 527 | 4081 |
| b | 128 | 15102 | 16101 | STTC | 1660 | 645 | 539 | 1237 | 4081 |
| b | 128 | 15102 | 16102 | CD | 2940 | 421 | 181 | 539 | 4081 |
| b | 128 | 15102 | 16102 | STTC | 1793 | 512 | 607 | 1169 | 4081 |
| b | 128 | 15102 | 16103 | CD | 3041 | 320 | 253 | 467 | 4081 |
| b | 128 | 15102 | 16103 | STTC | 1486 | 819 | 386 | 1390 | 4081 |
| b | 128 | 15103 | 16101 | CD | 3071 | 290 | 326 | 394 | 4081 |
| b | 128 | 15103 | 16101 | STTC | 1755 | 550 | 542 | 1234 | 4081 |
| b | 128 | 15103 | 16102 | CD | 2924 | 437 | 224 | 496 | 4081 |
| b | 128 | 15103 | 16102 | STTC | 1850 | 455 | 695 | 1081 | 4081 |
| b | 128 | 15103 | 16103 | CD | 3040 | 321 | 303 | 417 | 4081 |
| b | 128 | 15103 | 16103 | STTC | 1639 | 666 | 458 | 1318 | 4081 |
| b | 128 | 15104 | 16101 | CD | 2900 | 461 | 210 | 510 | 4081 |
| b | 128 | 15104 | 16101 | STTC | 1727 | 578 | 596 | 1180 | 4081 |
| b | 128 | 15104 | 16102 | CD | 2899 | 462 | 212 | 508 | 4081 |
| b | 128 | 15104 | 16102 | STTC | 1763 | 542 | 664 | 1112 | 4081 |
| b | 128 | 15104 | 16103 | CD | 3038 | 323 | 292 | 428 | 4081 |
| b | 128 | 15104 | 16103 | STTC | 1656 | 649 | 542 | 1234 | 4081 |
| b | 128 | 15105 | 16101 | CD | 2993 | 368 | 248 | 472 | 4081 |
| b | 128 | 15105 | 16101 | STTC | 1813 | 492 | 612 | 1164 | 4081 |
| b | 128 | 15105 | 16102 | CD | 2894 | 467 | 189 | 531 | 4081 |
| b | 128 | 15105 | 16102 | STTC | 1821 | 484 | 674 | 1102 | 4081 |
| b | 128 | 15105 | 16103 | CD | 3075 | 286 | 309 | 411 | 4081 |
| b | 128 | 15105 | 16103 | STTC | 1598 | 707 | 436 | 1340 | 4081 |
| b | 128 | 15106 | 16101 | CD | 2995 | 366 | 202 | 518 | 4081 |
| b | 128 | 15106 | 16101 | STTC | 1741 | 564 | 610 | 1166 | 4081 |
| b | 128 | 15106 | 16102 | CD | 2966 | 395 | 194 | 526 | 4081 |
| b | 128 | 15106 | 16102 | STTC | 1770 | 535 | 636 | 1140 | 4081 |
| b | 128 | 15106 | 16103 | CD | 3025 | 336 | 230 | 490 | 4081 |
| b | 128 | 15106 | 16103 | STTC | 1609 | 696 | 483 | 1293 | 4081 |
| b | 128 | 15107 | 16101 | CD | 3082 | 279 | 337 | 383 | 4081 |
| b | 128 | 15107 | 16101 | STTC | 1708 | 597 | 507 | 1269 | 4081 |
| b | 128 | 15107 | 16102 | CD | 3002 | 359 | 294 | 426 | 4081 |
| b | 128 | 15107 | 16102 | STTC | 1789 | 516 | 552 | 1224 | 4081 |
| b | 128 | 15107 | 16103 | CD | 3050 | 311 | 283 | 437 | 4081 |
| b | 128 | 15107 | 16103 | STTC | 1522 | 783 | 382 | 1394 | 4081 |
| b | 128 | 15108 | 16101 | CD | 2924 | 437 | 215 | 505 | 4081 |
| b | 128 | 15108 | 16101 | STTC | 1708 | 597 | 500 | 1276 | 4081 |
| b | 128 | 15108 | 16102 | CD | 2940 | 421 | 220 | 500 | 4081 |
| b | 128 | 15108 | 16102 | STTC | 1795 | 510 | 698 | 1078 | 4081 |
| b | 128 | 15108 | 16103 | CD | 3038 | 323 | 301 | 419 | 4081 |
| b | 128 | 15108 | 16103 | STTC | 1589 | 716 | 480 | 1296 | 4081 |
| b | 128 | 15109 | 16101 | CD | 2974 | 387 | 237 | 483 | 4081 |
| b | 128 | 15109 | 16101 | STTC | 1715 | 590 | 644 | 1132 | 4081 |
| b | 128 | 15109 | 16102 | CD | 2906 | 455 | 224 | 496 | 4081 |
| b | 128 | 15109 | 16102 | STTC | 1766 | 539 | 648 | 1128 | 4081 |
| b | 128 | 15109 | 16103 | CD | 3034 | 327 | 295 | 425 | 4081 |
| b | 128 | 15109 | 16103 | STTC | 1588 | 717 | 494 | 1282 | 4081 |
| b | 128 | 15110 | 16101 | CD | 3010 | 351 | 245 | 475 | 4081 |
| b | 128 | 15110 | 16101 | STTC | 1653 | 652 | 545 | 1231 | 4081 |
| b | 128 | 15110 | 16102 | CD | 2978 | 383 | 225 | 495 | 4081 |
| b | 128 | 15110 | 16102 | STTC | 1824 | 481 | 707 | 1069 | 4081 |
| b | 128 | 15110 | 16103 | CD | 3025 | 336 | 276 | 444 | 4081 |
| b | 128 | 15110 | 16103 | STTC | 1528 | 777 | 464 | 1312 | 4081 |
| b | 256 | 15101 | 16101 | CD | 2989 | 372 | 252 | 468 | 4081 |
| b | 256 | 15101 | 16101 | STTC | 1613 | 692 | 494 | 1282 | 4081 |
| b | 256 | 15101 | 16102 | CD | 3087 | 274 | 329 | 391 | 4081 |
| b | 256 | 15101 | 16102 | STTC | 1672 | 633 | 496 | 1280 | 4081 |
| b | 256 | 15101 | 16103 | CD | 3051 | 310 | 299 | 421 | 4081 |
| b | 256 | 15101 | 16103 | STTC | 1649 | 656 | 479 | 1297 | 4081 |
| b | 256 | 15102 | 16101 | CD | 2996 | 365 | 205 | 515 | 4081 |
| b | 256 | 15102 | 16101 | STTC | 1662 | 643 | 486 | 1290 | 4081 |
| b | 256 | 15102 | 16102 | CD | 3051 | 310 | 246 | 474 | 4081 |
| b | 256 | 15102 | 16102 | STTC | 1859 | 446 | 611 | 1165 | 4081 |
| b | 256 | 15102 | 16103 | CD | 3043 | 318 | 240 | 480 | 4081 |
| b | 256 | 15102 | 16103 | STTC | 1612 | 693 | 399 | 1377 | 4081 |
| b | 256 | 15103 | 16101 | CD | 3088 | 273 | 339 | 381 | 4081 |
| b | 256 | 15103 | 16101 | STTC | 1760 | 545 | 511 | 1265 | 4081 |
| b | 256 | 15103 | 16102 | CD | 3097 | 264 | 347 | 373 | 4081 |
| b | 256 | 15103 | 16102 | STTC | 1877 | 428 | 611 | 1165 | 4081 |
| b | 256 | 15103 | 16103 | CD | 3129 | 232 | 354 | 366 | 4081 |
| b | 256 | 15103 | 16103 | STTC | 1734 | 571 | 488 | 1288 | 4081 |
| b | 256 | 15104 | 16101 | CD | 2975 | 386 | 247 | 473 | 4081 |
| b | 256 | 15104 | 16101 | STTC | 1711 | 594 | 487 | 1289 | 4081 |
| b | 256 | 15104 | 16102 | CD | 3027 | 334 | 274 | 446 | 4081 |
| b | 256 | 15104 | 16102 | STTC | 1820 | 485 | 600 | 1176 | 4081 |
| b | 256 | 15104 | 16103 | CD | 3079 | 282 | 310 | 410 | 4081 |
| b | 256 | 15104 | 16103 | STTC | 1728 | 577 | 570 | 1206 | 4081 |
| b | 256 | 15105 | 16101 | CD | 3042 | 319 | 271 | 449 | 4081 |
| b | 256 | 15105 | 16101 | STTC | 1826 | 479 | 617 | 1159 | 4081 |
| b | 256 | 15105 | 16102 | CD | 3049 | 312 | 260 | 460 | 4081 |
| b | 256 | 15105 | 16102 | STTC | 1832 | 473 | 568 | 1208 | 4081 |
| b | 256 | 15105 | 16103 | CD | 3108 | 253 | 312 | 408 | 4081 |
| b | 256 | 15105 | 16103 | STTC | 1634 | 671 | 453 | 1323 | 4081 |
| b | 256 | 15106 | 16101 | CD | 3054 | 307 | 235 | 485 | 4081 |
| b | 256 | 15106 | 16101 | STTC | 1688 | 617 | 485 | 1291 | 4081 |
| b | 256 | 15106 | 16102 | CD | 3068 | 293 | 247 | 473 | 4081 |
| b | 256 | 15106 | 16102 | STTC | 1824 | 481 | 610 | 1166 | 4081 |
| b | 256 | 15106 | 16103 | CD | 3048 | 313 | 246 | 474 | 4081 |
| b | 256 | 15106 | 16103 | STTC | 1735 | 570 | 544 | 1232 | 4081 |
| b | 256 | 15107 | 16101 | CD | 3123 | 238 | 363 | 357 | 4081 |
| b | 256 | 15107 | 16101 | STTC | 1695 | 610 | 472 | 1304 | 4081 |
| b | 256 | 15107 | 16102 | CD | 3136 | 225 | 390 | 330 | 4081 |
| b | 256 | 15107 | 16102 | STTC | 1848 | 457 | 530 | 1246 | 4081 |
| b | 256 | 15107 | 16103 | CD | 3136 | 225 | 360 | 360 | 4081 |
| b | 256 | 15107 | 16103 | STTC | 1655 | 650 | 420 | 1356 | 4081 |
| b | 256 | 15108 | 16101 | CD | 3051 | 310 | 297 | 423 | 4081 |
| b | 256 | 15108 | 16101 | STTC | 1718 | 587 | 502 | 1274 | 4081 |
| b | 256 | 15108 | 16102 | CD | 3058 | 303 | 304 | 416 | 4081 |
| b | 256 | 15108 | 16102 | STTC | 1816 | 489 | 601 | 1175 | 4081 |
| b | 256 | 15108 | 16103 | CD | 3055 | 306 | 319 | 401 | 4081 |
| b | 256 | 15108 | 16103 | STTC | 1628 | 677 | 461 | 1315 | 4081 |
| b | 256 | 15109 | 16101 | CD | 3104 | 257 | 330 | 390 | 4081 |
| b | 256 | 15109 | 16101 | STTC | 1653 | 652 | 499 | 1277 | 4081 |
| b | 256 | 15109 | 16102 | CD | 3139 | 222 | 358 | 362 | 4081 |
| b | 256 | 15109 | 16102 | STTC | 1820 | 485 | 601 | 1175 | 4081 |
| b | 256 | 15109 | 16103 | CD | 3067 | 294 | 305 | 415 | 4081 |
| b | 256 | 15109 | 16103 | STTC | 1592 | 713 | 451 | 1325 | 4081 |
| b | 256 | 15110 | 16101 | CD | 3057 | 304 | 264 | 456 | 4081 |
| b | 256 | 15110 | 16101 | STTC | 1680 | 625 | 522 | 1254 | 4081 |
| b | 256 | 15110 | 16102 | CD | 3102 | 259 | 280 | 440 | 4081 |
| b | 256 | 15110 | 16102 | STTC | 1752 | 553 | 559 | 1217 | 4081 |
| b | 256 | 15110 | 16103 | CD | 3066 | 295 | 269 | 451 | 4081 |
| b | 256 | 15110 | 16103 | STTC | 1677 | 628 | 526 | 1250 | 4081 |
| c | 64 | 15101 | 16101 | CD | 2724 | 637 | 139 | 581 | 4081 |
| c | 64 | 15101 | 16101 | STTC | 1637 | 668 | 509 | 1267 | 4081 |
| c | 64 | 15101 | 16102 | CD | 3002 | 359 | 282 | 438 | 4081 |
| c | 64 | 15101 | 16102 | STTC | 1596 | 709 | 433 | 1343 | 4081 |
| c | 64 | 15101 | 16103 | CD | 2988 | 373 | 242 | 478 | 4081 |
| c | 64 | 15101 | 16103 | STTC | 1609 | 696 | 499 | 1277 | 4081 |
| c | 64 | 15102 | 16101 | CD | 2801 | 560 | 121 | 599 | 4081 |
| c | 64 | 15102 | 16101 | STTC | 1533 | 772 | 477 | 1299 | 4081 |
| c | 64 | 15102 | 16102 | CD | 3006 | 355 | 238 | 482 | 4081 |
| c | 64 | 15102 | 16102 | STTC | 1571 | 734 | 434 | 1342 | 4081 |
| c | 64 | 15102 | 16103 | CD | 3026 | 335 | 237 | 483 | 4081 |
| c | 64 | 15102 | 16103 | STTC | 1553 | 752 | 460 | 1316 | 4081 |
| c | 64 | 15103 | 16101 | CD | 2738 | 623 | 140 | 580 | 4081 |
| c | 64 | 15103 | 16101 | STTC | 1646 | 659 | 536 | 1240 | 4081 |
| c | 64 | 15103 | 16102 | CD | 2999 | 362 | 310 | 410 | 4081 |
| c | 64 | 15103 | 16102 | STTC | 1651 | 654 | 480 | 1296 | 4081 |
| c | 64 | 15103 | 16103 | CD | 3011 | 350 | 275 | 445 | 4081 |
| c | 64 | 15103 | 16103 | STTC | 1725 | 580 | 568 | 1208 | 4081 |
| c | 64 | 15104 | 16101 | CD | 2751 | 610 | 118 | 602 | 4081 |
| c | 64 | 15104 | 16101 | STTC | 1559 | 746 | 513 | 1263 | 4081 |
| c | 64 | 15104 | 16102 | CD | 2996 | 365 | 234 | 486 | 4081 |
| c | 64 | 15104 | 16102 | STTC | 1638 | 667 | 482 | 1294 | 4081 |
| c | 64 | 15104 | 16103 | CD | 3016 | 345 | 247 | 473 | 4081 |
| c | 64 | 15104 | 16103 | STTC | 1740 | 565 | 562 | 1214 | 4081 |
| c | 64 | 15105 | 16101 | CD | 2841 | 520 | 169 | 551 | 4081 |
| c | 64 | 15105 | 16101 | STTC | 1753 | 552 | 613 | 1163 | 4081 |
| c | 64 | 15105 | 16102 | CD | 3043 | 318 | 255 | 465 | 4081 |
| c | 64 | 15105 | 16102 | STTC | 1611 | 694 | 430 | 1346 | 4081 |
| c | 64 | 15105 | 16103 | CD | 3018 | 343 | 287 | 433 | 4081 |
| c | 64 | 15105 | 16103 | STTC | 1742 | 563 | 574 | 1202 | 4081 |
| c | 64 | 15106 | 16101 | CD | 2742 | 619 | 119 | 601 | 4081 |
| c | 64 | 15106 | 16101 | STTC | 1706 | 599 | 584 | 1192 | 4081 |
| c | 64 | 15106 | 16102 | CD | 2933 | 428 | 206 | 514 | 4081 |
| c | 64 | 15106 | 16102 | STTC | 1725 | 580 | 509 | 1267 | 4081 |
| c | 64 | 15106 | 16103 | CD | 2966 | 395 | 243 | 477 | 4081 |
| c | 64 | 15106 | 16103 | STTC | 1708 | 597 | 569 | 1207 | 4081 |
| c | 64 | 15107 | 16101 | CD | 2815 | 546 | 177 | 543 | 4081 |
| c | 64 | 15107 | 16101 | STTC | 1702 | 603 | 600 | 1176 | 4081 |
| c | 64 | 15107 | 16102 | CD | 3044 | 317 | 315 | 405 | 4081 |
| c | 64 | 15107 | 16102 | STTC | 1583 | 722 | 489 | 1287 | 4081 |
| c | 64 | 15107 | 16103 | CD | 3046 | 315 | 296 | 424 | 4081 |
| c | 64 | 15107 | 16103 | STTC | 1658 | 647 | 611 | 1165 | 4081 |
| c | 64 | 15108 | 16101 | CD | 2839 | 522 | 142 | 578 | 4081 |
| c | 64 | 15108 | 16101 | STTC | 1714 | 591 | 496 | 1280 | 4081 |
| c | 64 | 15108 | 16102 | CD | 3018 | 343 | 237 | 483 | 4081 |
| c | 64 | 15108 | 16102 | STTC | 1752 | 553 | 503 | 1273 | 4081 |
| c | 64 | 15108 | 16103 | CD | 2950 | 411 | 207 | 513 | 4081 |
| c | 64 | 15108 | 16103 | STTC | 1757 | 548 | 576 | 1200 | 4081 |
| c | 64 | 15109 | 16101 | CD | 2770 | 591 | 145 | 575 | 4081 |
| c | 64 | 15109 | 16101 | STTC | 1642 | 663 | 531 | 1245 | 4081 |
| c | 64 | 15109 | 16102 | CD | 2990 | 371 | 278 | 442 | 4081 |
| c | 64 | 15109 | 16102 | STTC | 1618 | 687 | 443 | 1333 | 4081 |
| c | 64 | 15109 | 16103 | CD | 2958 | 403 | 228 | 492 | 4081 |
| c | 64 | 15109 | 16103 | STTC | 1715 | 590 | 567 | 1209 | 4081 |
| c | 64 | 15110 | 16101 | CD | 2761 | 600 | 122 | 598 | 4081 |
| c | 64 | 15110 | 16101 | STTC | 1610 | 695 | 503 | 1273 | 4081 |
| c | 64 | 15110 | 16102 | CD | 2998 | 363 | 271 | 449 | 4081 |
| c | 64 | 15110 | 16102 | STTC | 1683 | 622 | 488 | 1288 | 4081 |
| c | 64 | 15110 | 16103 | CD | 3035 | 326 | 268 | 452 | 4081 |
| c | 64 | 15110 | 16103 | STTC | 1752 | 553 | 582 | 1194 | 4081 |
| c | 128 | 15101 | 16101 | CD | 2940 | 421 | 208 | 512 | 4081 |
| c | 128 | 15101 | 16101 | STTC | 1720 | 585 | 562 | 1214 | 4081 |
| c | 128 | 15101 | 16102 | CD | 2919 | 442 | 228 | 492 | 4081 |
| c | 128 | 15101 | 16102 | STTC | 1814 | 491 | 654 | 1122 | 4081 |
| c | 128 | 15101 | 16103 | CD | 3066 | 295 | 292 | 428 | 4081 |
| c | 128 | 15101 | 16103 | STTC | 1476 | 829 | 384 | 1392 | 4081 |
| c | 128 | 15102 | 16101 | CD | 2997 | 364 | 212 | 508 | 4081 |
| c | 128 | 15102 | 16101 | STTC | 1604 | 701 | 469 | 1307 | 4081 |
| c | 128 | 15102 | 16102 | CD | 2912 | 449 | 163 | 557 | 4081 |
| c | 128 | 15102 | 16102 | STTC | 1734 | 571 | 544 | 1232 | 4081 |
| c | 128 | 15102 | 16103 | CD | 3011 | 350 | 239 | 481 | 4081 |
| c | 128 | 15102 | 16103 | STTC | 1478 | 827 | 379 | 1397 | 4081 |
| c | 128 | 15103 | 16101 | CD | 3023 | 338 | 261 | 459 | 4081 |
| c | 128 | 15103 | 16101 | STTC | 1782 | 523 | 583 | 1193 | 4081 |
| c | 128 | 15103 | 16102 | CD | 2928 | 433 | 249 | 471 | 4081 |
| c | 128 | 15103 | 16102 | STTC | 1792 | 513 | 631 | 1145 | 4081 |
| c | 128 | 15103 | 16103 | CD | 3036 | 325 | 283 | 437 | 4081 |
| c | 128 | 15103 | 16103 | STTC | 1619 | 686 | 459 | 1317 | 4081 |
| c | 128 | 15104 | 16101 | CD | 3006 | 355 | 231 | 489 | 4081 |
| c | 128 | 15104 | 16101 | STTC | 1733 | 572 | 591 | 1185 | 4081 |
| c | 128 | 15104 | 16102 | CD | 2953 | 408 | 205 | 515 | 4081 |
| c | 128 | 15104 | 16102 | STTC | 1746 | 559 | 612 | 1164 | 4081 |
| c | 128 | 15104 | 16103 | CD | 3038 | 323 | 259 | 461 | 4081 |
| c | 128 | 15104 | 16103 | STTC | 1621 | 684 | 458 | 1318 | 4081 |
| c | 128 | 15105 | 16101 | CD | 3005 | 356 | 248 | 472 | 4081 |
| c | 128 | 15105 | 16101 | STTC | 1790 | 515 | 611 | 1165 | 4081 |
| c | 128 | 15105 | 16102 | CD | 2976 | 385 | 202 | 518 | 4081 |
| c | 128 | 15105 | 16102 | STTC | 1876 | 429 | 646 | 1130 | 4081 |
| c | 128 | 15105 | 16103 | CD | 3020 | 341 | 236 | 484 | 4081 |
| c | 128 | 15105 | 16103 | STTC | 1596 | 709 | 425 | 1351 | 4081 |
| c | 128 | 15106 | 16101 | CD | 2958 | 403 | 210 | 510 | 4081 |
| c | 128 | 15106 | 16101 | STTC | 1738 | 567 | 582 | 1194 | 4081 |
| c | 128 | 15106 | 16102 | CD | 2894 | 467 | 176 | 544 | 4081 |
| c | 128 | 15106 | 16102 | STTC | 1855 | 450 | 667 | 1109 | 4081 |
| c | 128 | 15106 | 16103 | CD | 2977 | 384 | 245 | 475 | 4081 |
| c | 128 | 15106 | 16103 | STTC | 1573 | 732 | 453 | 1323 | 4081 |
| c | 128 | 15107 | 16101 | CD | 3003 | 358 | 273 | 447 | 4081 |
| c | 128 | 15107 | 16101 | STTC | 1650 | 655 | 545 | 1231 | 4081 |
| c | 128 | 15107 | 16102 | CD | 2988 | 373 | 247 | 473 | 4081 |
| c | 128 | 15107 | 16102 | STTC | 1709 | 596 | 603 | 1173 | 4081 |
| c | 128 | 15107 | 16103 | CD | 3055 | 306 | 299 | 421 | 4081 |
| c | 128 | 15107 | 16103 | STTC | 1500 | 805 | 495 | 1281 | 4081 |
| c | 128 | 15108 | 16101 | CD | 3064 | 297 | 271 | 449 | 4081 |
| c | 128 | 15108 | 16101 | STTC | 1711 | 594 | 486 | 1290 | 4081 |
| c | 128 | 15108 | 16102 | CD | 2927 | 434 | 184 | 536 | 4081 |
| c | 128 | 15108 | 16102 | STTC | 1922 | 383 | 716 | 1060 | 4081 |
| c | 128 | 15108 | 16103 | CD | 2984 | 377 | 200 | 520 | 4081 |
| c | 128 | 15108 | 16103 | STTC | 1656 | 649 | 463 | 1313 | 4081 |
| c | 128 | 15109 | 16101 | CD | 2980 | 381 | 240 | 480 | 4081 |
| c | 128 | 15109 | 16101 | STTC | 1777 | 528 | 561 | 1215 | 4081 |
| c | 128 | 15109 | 16102 | CD | 2885 | 476 | 220 | 500 | 4081 |
| c | 128 | 15109 | 16102 | STTC | 1777 | 528 | 575 | 1201 | 4081 |
| c | 128 | 15109 | 16103 | CD | 3006 | 355 | 270 | 450 | 4081 |
| c | 128 | 15109 | 16103 | STTC | 1665 | 640 | 468 | 1308 | 4081 |
| c | 128 | 15110 | 16101 | CD | 2999 | 362 | 223 | 497 | 4081 |
| c | 128 | 15110 | 16101 | STTC | 1679 | 626 | 513 | 1263 | 4081 |
| c | 128 | 15110 | 16102 | CD | 2976 | 385 | 239 | 481 | 4081 |
| c | 128 | 15110 | 16102 | STTC | 1823 | 482 | 649 | 1127 | 4081 |
| c | 128 | 15110 | 16103 | CD | 3054 | 307 | 282 | 438 | 4081 |
| c | 128 | 15110 | 16103 | STTC | 1501 | 804 | 358 | 1418 | 4081 |
| c | 256 | 15101 | 16101 | CD | 3059 | 302 | 306 | 414 | 4081 |
| c | 256 | 15101 | 16101 | STTC | 1747 | 558 | 524 | 1252 | 4081 |
| c | 256 | 15101 | 16102 | CD | 3075 | 286 | 319 | 401 | 4081 |
| c | 256 | 15101 | 16102 | STTC | 1762 | 543 | 503 | 1273 | 4081 |
| c | 256 | 15101 | 16103 | CD | 3119 | 242 | 332 | 388 | 4081 |
| c | 256 | 15101 | 16103 | STTC | 1642 | 663 | 418 | 1358 | 4081 |
| c | 256 | 15102 | 16101 | CD | 3074 | 287 | 281 | 439 | 4081 |
| c | 256 | 15102 | 16101 | STTC | 1712 | 593 | 460 | 1316 | 4081 |
| c | 256 | 15102 | 16102 | CD | 3059 | 302 | 273 | 447 | 4081 |
| c | 256 | 15102 | 16102 | STTC | 1859 | 446 | 576 | 1200 | 4081 |
| c | 256 | 15102 | 16103 | CD | 3067 | 294 | 273 | 447 | 4081 |
| c | 256 | 15102 | 16103 | STTC | 1629 | 676 | 410 | 1366 | 4081 |
| c | 256 | 15103 | 16101 | CD | 3062 | 299 | 310 | 410 | 4081 |
| c | 256 | 15103 | 16101 | STTC | 1799 | 506 | 572 | 1204 | 4081 |
| c | 256 | 15103 | 16102 | CD | 3042 | 319 | 294 | 426 | 4081 |
| c | 256 | 15103 | 16102 | STTC | 1938 | 367 | 681 | 1095 | 4081 |
| c | 256 | 15103 | 16103 | CD | 3102 | 259 | 316 | 404 | 4081 |
| c | 256 | 15103 | 16103 | STTC | 1702 | 603 | 483 | 1293 | 4081 |
| c | 256 | 15104 | 16101 | CD | 3093 | 268 | 278 | 442 | 4081 |
| c | 256 | 15104 | 16101 | STTC | 1740 | 565 | 541 | 1235 | 4081 |
| c | 256 | 15104 | 16102 | CD | 3087 | 274 | 276 | 444 | 4081 |
| c | 256 | 15104 | 16102 | STTC | 1899 | 406 | 649 | 1127 | 4081 |
| c | 256 | 15104 | 16103 | CD | 3092 | 269 | 293 | 427 | 4081 |
| c | 256 | 15104 | 16103 | STTC | 1810 | 495 | 580 | 1196 | 4081 |
| c | 256 | 15105 | 16101 | CD | 3021 | 340 | 238 | 482 | 4081 |
| c | 256 | 15105 | 16101 | STTC | 1905 | 400 | 652 | 1124 | 4081 |
| c | 256 | 15105 | 16102 | CD | 3047 | 314 | 267 | 453 | 4081 |
| c | 256 | 15105 | 16102 | STTC | 1883 | 422 | 593 | 1183 | 4081 |
| c | 256 | 15105 | 16103 | CD | 3107 | 254 | 285 | 435 | 4081 |
| c | 256 | 15105 | 16103 | STTC | 1701 | 604 | 453 | 1323 | 4081 |
| c | 256 | 15106 | 16101 | CD | 3044 | 317 | 246 | 474 | 4081 |
| c | 256 | 15106 | 16101 | STTC | 1722 | 583 | 475 | 1301 | 4081 |
| c | 256 | 15106 | 16102 | CD | 2985 | 376 | 215 | 505 | 4081 |
| c | 256 | 15106 | 16102 | STTC | 1867 | 438 | 613 | 1163 | 4081 |
| c | 256 | 15106 | 16103 | CD | 3029 | 332 | 247 | 473 | 4081 |
| c | 256 | 15106 | 16103 | STTC | 1723 | 582 | 509 | 1267 | 4081 |
| c | 256 | 15107 | 16101 | CD | 3117 | 244 | 337 | 383 | 4081 |
| c | 256 | 15107 | 16101 | STTC | 1585 | 720 | 473 | 1303 | 4081 |
| c | 256 | 15107 | 16102 | CD | 3097 | 264 | 297 | 423 | 4081 |
| c | 256 | 15107 | 16102 | STTC | 1716 | 589 | 583 | 1193 | 4081 |
| c | 256 | 15107 | 16103 | CD | 3107 | 254 | 315 | 405 | 4081 |
| c | 256 | 15107 | 16103 | STTC | 1629 | 676 | 502 | 1274 | 4081 |
| c | 256 | 15108 | 16101 | CD | 3146 | 215 | 334 | 386 | 4081 |
| c | 256 | 15108 | 16101 | STTC | 1822 | 483 | 540 | 1236 | 4081 |
| c | 256 | 15108 | 16102 | CD | 3097 | 264 | 287 | 433 | 4081 |
| c | 256 | 15108 | 16102 | STTC | 1929 | 376 | 652 | 1124 | 4081 |
| c | 256 | 15108 | 16103 | CD | 3078 | 283 | 263 | 457 | 4081 |
| c | 256 | 15108 | 16103 | STTC | 1713 | 592 | 433 | 1343 | 4081 |
| c | 256 | 15109 | 16101 | CD | 3070 | 291 | 312 | 408 | 4081 |
| c | 256 | 15109 | 16101 | STTC | 1748 | 557 | 472 | 1304 | 4081 |
| c | 256 | 15109 | 16102 | CD | 3088 | 273 | 328 | 392 | 4081 |
| c | 256 | 15109 | 16102 | STTC | 1784 | 521 | 521 | 1255 | 4081 |
| c | 256 | 15109 | 16103 | CD | 3035 | 326 | 287 | 433 | 4081 |
| c | 256 | 15109 | 16103 | STTC | 1622 | 683 | 408 | 1368 | 4081 |
| c | 256 | 15110 | 16101 | CD | 3072 | 289 | 279 | 441 | 4081 |
| c | 256 | 15110 | 16101 | STTC | 1711 | 594 | 485 | 1291 | 4081 |
| c | 256 | 15110 | 16102 | CD | 3076 | 285 | 287 | 433 | 4081 |
| c | 256 | 15110 | 16102 | STTC | 1833 | 472 | 554 | 1222 | 4081 |
| c | 256 | 15110 | 16103 | CD | 3104 | 257 | 310 | 410 | 4081 |
| c | 256 | 15110 | 16103 | STTC | 1631 | 674 | 410 | 1366 | 4081 |

## Verification and limits

- **1498 per-run rows and 220 mean rows (1718 total)** checked with exact integer/rational arithmetic. Every row sums to its study-specific N (2,158 / 2,633 / 3,860 / 4,081); class supports are constant within each study and label.
- Source rosters, method/budget/draw coverage, label identities, observation completeness and historical thresholds were checked explicitly. Experiment 4 mean counts also reconcile with the saved corrected aggregate file.
- Experiment 2 and Experiment 4 aggregate source hashes match their original verification receipts. Phase 4C and Phase 5 aggregate analysis hashes match their completed-operation receipts. Source files were rehashed after document generation and remained unchanged. No source/private hash is disclosed here.
- No historical result, protocol, evaluation receipt, source code or private file was modified. No model was loaded, no test signal/label/prediction array was read and no evaluation operation was invoked. Only existing aggregate JSON was used.
- These counts do not replace ranking metrics, registered contrasts or uncertainty estimates. They do not establish a clinically validated threshold or prospective clinical performance. No file was staged, committed, pushed or uploaded by this task.
