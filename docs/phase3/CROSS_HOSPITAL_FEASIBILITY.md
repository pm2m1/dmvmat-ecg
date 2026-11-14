# Cross-hospital zero-shot annotation-proxy feasibility

Executed inference only. These are exploratory annotation-classification results, not clinical diagnostic negatives, external clinical validation, or future-event prediction. Source: PTB-XL v1.0.3. Targets: CinC2021 v1.0.3. No training, fine-tuning, normalization fitting or calibration fitting was performed.

## Download and dataset audit

Official release files: https://physionet.org/files/challenge-2021/1.0.3/. Eight maximum concurrent single-stream downloads, checksum-aware skip/resume, at most ten total tries per curl transfer, five-second retry wait. Initial aria2 downloads used max-tries10; runtime low-speed aborts prompted removal of that floor, then the permitted curl fallback with reusable HTTP2 connections. A16-file probe passed all hashes. A premature completion check during curl startup hashing was archived, not used as final verification; see results/phase3a/STARTUP_RACE_INCIDENT.json. No model ran during that incident. PTB, PTB-XL and INCART copies deliberately excluded: source overlap and non-target design. Publisher checksum list SHA256: `f29c2609343c38734f3a3d62be4a9052ca56a9e560e7ad2efc2118563fca2473`.

| Source | Expected files | Verified files | Bytes | Mismatches | Records | Short<10s | Long>10s | Flat-lead records | Duplicates removed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chapman_shaoxing | 20505 | 20505 | 1237621878 | 0 | 10247 | 0 | 0 | 15 | 13 |
| cpsc_2018 | 13762 | 13762 | 1321038090 | 0 | 6877 | 10 | 4451 | 18 | 255 |
| cpsc_2018_extra | 6910 | 6910 | 661708575 | 0 | 3453 | 12 | 2038 | 8 | 404 |
| georgia | 20699 | 20699 | 1245971648 | 0 | 10344 | 52 | 0 | 9 | 94 |
| ningbo | 69845 | 69845 | 4215789066 | 0 | 34905 | 0 | 0 | 1480 | 74 |

Full rate, length, lead-order, unit and gain inventories: [DATASET_AUDIT.json](../../results/phase3a/DATASET_AUDIT.json). Header/MAT/finite checks failed for 0 records. Native exact duplicate totals: 840. Pair counts: `{"chapman_shaoxing -> chapman_shaoxing": 13, "chapman_shaoxing -> ningbo": 71, "cpsc_2018 -> cpsc_2018": 255, "cpsc_2018 -> cpsc_2018_extra": 342, "cpsc_2018_extra -> cpsc_2018_extra": 62, "georgia -> georgia": 94, "ningbo -> ningbo": 3}`. First lexicographic complete native-wave identity retained, independently of diagnoses; labels are never merged. Canonical lead/mV hash rounded at1e-9mV includes native rate and full length. It does not detect near-duplicates, crops or common patients.

## Frozen annotation mapping and proxy-matched reference

- **CD-proxy SNOMED**: 270492004, 445118002, 59118001, 698252002, 713426002, 713427006, 733534002. **Matched fold9 SCP keys**: 1AVB, CLBBB, CRBBB, IRBBB, IVCD, LAFB.
- **STTC-proxy SNOMED**: 164930006, 164934002, 428750005, 429622005, 59931005. **Matched fold9 SCP keys**: INVT, NDT, NST_, NT_, STD_, STE_, TAB_.

Mapping hash: `ecce890b4678b8f2a06cf0796476221adceb8156c7d476ecfbeba910a43891d6`. [LABEL_MAPPING.json](../../results/phase3a/LABEL_MAPPING.json) lists EVERY observed code, included or explicitly excluded, condition families and descriptive per-code tasks. Code164909002 is excluded because of the documented LBBB/fascicular conversion collision. AF is not a model output. Absent codes mean **not annotated**, never clinically negative. Source SCP keys are used as recorded, without inventing a likelihood cutoff. Only fold9 IDs are admitted before SCP decoding. Form-only SCP tags support the restricted ST/T proxy; this is deliberately different from the original full diagnostic superclass.

The same named-condition unions are applied on both sides. Subtype/reporting completeness remains uncertain, including unspecified RBBB and nonspecific T/ST annotations. Therefore **part of any DROP can result from label-definition/reporting mismatch rather than true signal shift**. No clinically verified negative endpoint was established.

## Frozen FEAS/RESERVED split

Fixed seed74301, one deterministic joint-pattern Hamilton20/80 allocation after native duplicate and header-only short-record exclusions. No seed search. ID-only manifests remain git-ignored. RESERVED labels were used only for stratification, then discarded from model-facing caches; no RESERVED inference or subsequent label analysis.

| Domain | FEAS | CD+ | STTC+ | RESERVED count only | Expected RESERVED CD+ | Expected RESERVED STTC+ | FEAS truncated | All short excluded | Split SHA256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CPSC | 1930 | 506 | 494 | 7719 | 2023.7 | 1975.7 | 1203 | 22 | `84bea540fef6e04d5638f7b09f1107de5b0335a0e58df1d2b93007e8bed16cf1` |
| Chapman-Ningbo | 9013 | 839 | 2011 | 36052 | 3356.0 | 8044.0 | 0 | 0 | `d325ee68a87c6e30ac5d05ff93444e4807b09852a121c69d303e49cc497fb9fb` |
| Georgia | 2040 | 374 | 899 | 8161 | 1496.2 | 3596.4 | 0 | 52 | `dd1821af54d649b2f4d8b673ca4227cfa9b653a4feb9a34eeac6ddeb1a1f8af7` |

Expected RESERVED positives are **estimates**, calculated from FEAS prevalence times RESERVED size; they are not observed RESERVED label counts.

## Frozen inference and reference

Historical no-augmentation five-view and matched flat CNN; both already-trained seeds101/202, selected previously on five-class fold9 macro AUROC. No target-based checkpoint selection. Mean seed-wise metrics are reported, not a new ensemble. Full10s [12,1000] inference uses exact historical train-only scalar normalization; rhythm and frequency transforms are unchanged. Longer records use first10s; shorter records are excluded, never padded. Anti-aliasing: scipy.resample_poly with default Kaiser5 filter and line boundary extension. Lead reordering and header gain/unit conversion precede resampling. No target normalization fit. All checkpoint and statistics hashes appear in [RUN_PROTOCOL.json](../../results/phase3a/RUN_PROTOCOL.json); per-pass runtimes in [PREDICTION_SEAL.json](../../results/phase3a/PREDICTION_SEAL.json). Five-view212261 parameters; flat206117.

## Fold9 references and FEAS metrics

Continuous scores throughout. AP random-ranking reference is the annotation-positive prevalence. Brier and mean predicted probability are diagnostic only: the model outputs full superclass probabilities rather than calibrated probabilities of these restricted proxies. Source fold9 full-class and proxy metrics share the same frozen scores. DROP uses **proxy-matched fold9 only**.

| Population/definition | Model | Label | AUROC | AP | Prevalence | Brier | Mean predicted | AUROC95% CI | AP95% CI |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| CPSC | flat_cnn | CD | 0.8690 | 0.7271 | 0.2622 | 0.1377 | 0.2909 | [0.8505, 0.8866] | [0.6861, 0.7667] |
| CPSC | flat_cnn | STTC | 0.6886 | 0.4062 | 0.2560 | 0.2097 | 0.2553 | [0.6616, 0.7162] | [0.3694, 0.4484] |
| CPSC | multiview | CD | 0.8779 | 0.7272 | 0.2622 | 0.1459 | 0.3334 | [0.8610, 0.8931] | [0.6891, 0.7644] |
| CPSC | multiview | STTC | 0.6999 | 0.4064 | 0.2560 | 0.2203 | 0.2934 | [0.6737, 0.7256] | [0.3693, 0.4484] |
| Chapman-Ningbo | flat_cnn | CD | 0.8733 | 0.5872 | 0.0931 | 0.0993 | 0.1932 | [0.8607, 0.8855] | [0.5525, 0.6204] |
| Chapman-Ningbo | flat_cnn | STTC | 0.8294 | 0.5379 | 0.2231 | 0.1418 | 0.2358 | [0.8203, 0.8387] | [0.5164, 0.5629] |
| Chapman-Ningbo | multiview | CD | 0.8714 | 0.5398 | 0.0931 | 0.1312 | 0.2377 | [0.8595, 0.8826] | [0.5060, 0.5735] |
| Chapman-Ningbo | multiview | STTC | 0.8318 | 0.5491 | 0.2231 | 0.1472 | 0.2761 | [0.8224, 0.8412] | [0.5263, 0.5729] |
| Georgia | flat_cnn | CD | 0.8571 | 0.6133 | 0.1833 | 0.1362 | 0.2660 | [0.8378, 0.8761] | [0.5649, 0.6623] |
| Georgia | flat_cnn | STTC | 0.8160 | 0.7397 | 0.4407 | 0.2218 | 0.2638 | [0.7971, 0.8345] | [0.7072, 0.7739] |
| Georgia | multiview | CD | 0.8632 | 0.5967 | 0.1833 | 0.1614 | 0.3181 | [0.8443, 0.8813] | [0.5478, 0.6477] |
| Georgia | multiview | STTC | 0.8080 | 0.7347 | 0.4407 | 0.2151 | 0.2965 | [0.7896, 0.8263] | [0.7034, 0.7677] |
| fold9 | flat_cnn | CD | 0.9255 | 0.8245 | 0.2209 | 0.0841 | 0.2029 | [0.9119, 0.9386] | [0.7951, 0.8535] |
| fold9 | flat_cnn | STTC | 0.8488 | 0.4490 | 0.1710 | 0.1261 | 0.1996 | [0.8295, 0.8663] | [0.4039, 0.5018] |
| fold9 | multiview | CD | 0.9247 | 0.8248 | 0.2209 | 0.0943 | 0.2509 | [0.9110, 0.9378] | [0.7935, 0.8554] |
| fold9 | multiview | STTC | 0.8505 | 0.4460 | 0.1710 | 0.1324 | 0.2294 | [0.8324, 0.8672] | [0.4009, 0.4942] |
| fold9_full_superclass | flat_cnn | CD | 0.9276 | 0.8514 | 0.2307 | 0.0823 | 0.2029 | [0.9144, 0.9412] | [0.8266, 0.8762] |
| fold9_full_superclass | flat_cnn | STTC | 0.9275 | 0.7798 | 0.2460 | 0.0965 | 0.1996 | [0.9160, 0.9387] | [0.7411, 0.8176] |
| fold9_full_superclass | multiview | CD | 0.9262 | 0.8542 | 0.2307 | 0.0923 | 0.2509 | [0.9124, 0.9398] | [0.8297, 0.8778] |
| fold9_full_superclass | multiview | STTC | 0.9257 | 0.7881 | 0.2460 | 0.0950 | 0.2294 | [0.9144, 0.9368] | [0.7534, 0.8207] |

| Domain | Model | Label | AUROC DROP | AP DROP | Prior-adjusted source AP |
|---|---|---|---:|---:|---:|
| CPSC | flat_cnn | CD | -0.0565 | -0.0974 | 0.8484 |
| CPSC | flat_cnn | STTC | -0.1602 | -0.0428 | 0.5737 |
| CPSC | multiview | CD | -0.0468 | -0.0976 | 0.8487 |
| CPSC | multiview | STTC | -0.1506 | -0.0396 | 0.5701 |
| Chapman-Ningbo | flat_cnn | CD | -0.0522 | -0.2372 | 0.6900 |
| Chapman-Ningbo | flat_cnn | STTC | -0.0194 | 0.0888 | 0.5300 |
| Chapman-Ningbo | multiview | CD | -0.0533 | -0.2850 | 0.6880 |
| Chapman-Ningbo | multiview | STTC | -0.0187 | 0.1032 | 0.5265 |
| Georgia | flat_cnn | CD | -0.0683 | -0.2112 | 0.7973 |
| Georgia | flat_cnn | STTC | -0.0327 | 0.2907 | 0.7521 |
| Georgia | multiview | CD | -0.0615 | -0.2281 | 0.7975 |
| Georgia | multiview | STTC | -0.0425 | 0.2887 | 0.7491 |

DROP intervals, undefined bootstrap counts and all descriptive per-code prevalence/AUROC/AP values are in [METRICS.json](../../results/phase3a/METRICS.json). Per-code AUROC uses the associated CD/STTC output, **not a dedicated per-code predictor**. Missing-class tasks have undefined AUROC and zero-positive AP is undefined, never converted into success. Pure-prior reweighting of source AP is a descriptive benchmark, not proof that annotation semantics match. Raw AP drops are prevalence-dependent.

## Descriptive per-code results

Every task is a recorded code versus its absence; absence is not a clinical negative. No dedicated subtype head is invented. The LBBB task uses unambiguous complete-LBBB code733534002 only; ambiguous164909002 is excluded.

| Population | Code/condition | Proxy score | Annotated+ | Prevalence | Five-view AUROC | Flat AUROC |
|---|---|---|---:|---:|---:|---:|
| CPSC | 164930006 ST elevation | STTC | 94 | 0.0487 | 0.4449 | 0.4252 |
| CPSC | 164934002 T-wave abnormality | STTC | 6 | 0.0031 | 0.6196 | 0.5674 |
| CPSC | 270492004 IAVB | CD | 138 | 0.0715 | 0.5987 | 0.5668 |
| CPSC | 428750005 nonspecific ST/T annotation | STTC | 233 | 0.1207 | 0.8097 | 0.8166 |
| CPSC | 429622005 ST depression | STTC | 161 | 0.0834 | 0.5973 | 0.5723 |
| CPSC | 445118002 LAFB | CD | 0 | 0.0000 | undefined | undefined |
| CPSC | 59118001 recorded unspecified RBBB | CD | 346 | 0.1793 | 0.9262 | 0.9285 |
| CPSC | 59931005 T-wave inversion | STTC | 1 | 0.0005 | 0.7035 | 0.8981 |
| CPSC | 698252002 IVCD | CD | 1 | 0.0005 | 0.8279 | 0.8196 |
| CPSC | 713426002 IRBBB | CD | 15 | 0.0078 | 0.8287 | 0.8340 |
| CPSC | 713427006 CRBBB | CD | 13 | 0.0067 | 0.8669 | 0.8721 |
| CPSC | 733534002 complete LBBB | CD | 0 | 0.0000 | undefined | undefined |
| Chapman-Ningbo | 164930006 ST elevation | STTC | 147 | 0.0163 | 0.4597 | 0.4493 |
| Chapman-Ningbo | 164934002 T-wave abnormality | STTC | 1429 | 0.1585 | 0.8149 | 0.8130 |
| Chapman-Ningbo | 270492004 IAVB | CD | 214 | 0.0237 | 0.7229 | 0.7010 |
| Chapman-Ningbo | 428750005 nonspecific ST/T annotation | STTC | 213 | 0.0236 | 0.8333 | 0.8286 |
| Chapman-Ningbo | 429622005 ST depression | STTC | 339 | 0.0376 | 0.8648 | 0.8643 |
| Chapman-Ningbo | 445118002 LAFB | CD | 81 | 0.0090 | 0.9230 | 0.9195 |
| Chapman-Ningbo | 59118001 recorded unspecified RBBB | CD | 138 | 0.0153 | 0.9604 | 0.9683 |
| Chapman-Ningbo | 59931005 T-wave inversion | STTC | 584 | 0.0648 | 0.8142 | 0.8187 |
| Chapman-Ningbo | 698252002 IVCD | CD | 144 | 0.0160 | 0.6489 | 0.6715 |
| Chapman-Ningbo | 713426002 IRBBB | CD | 51 | 0.0057 | 0.9100 | 0.9184 |
| Chapman-Ningbo | 713427006 CRBBB | CD | 233 | 0.0259 | 0.9710 | 0.9782 |
| Chapman-Ningbo | 733534002 complete LBBB | CD | 40 | 0.0044 | 0.9598 | 0.9550 |
| Georgia | 164930006 ST elevation | STTC | 173 | 0.0848 | 0.6788 | 0.6818 |
| Georgia | 164934002 T-wave abnormality | STTC | 475 | 0.2328 | 0.7741 | 0.7881 |
| Georgia | 270492004 IAVB | CD | 153 | 0.0750 | 0.7174 | 0.6964 |
| Georgia | 428750005 nonspecific ST/T annotation | STTC | 359 | 0.1760 | 0.7087 | 0.6968 |
| Georgia | 429622005 ST depression | STTC | 8 | 0.0039 | 0.6715 | 0.7391 |
| Georgia | 445118002 LAFB | CD | 41 | 0.0201 | 0.8735 | 0.8857 |
| Georgia | 59118001 recorded unspecified RBBB | CD | 115 | 0.0564 | 0.9451 | 0.9481 |
| Georgia | 59931005 T-wave inversion | STTC | 148 | 0.0725 | 0.6047 | 0.6182 |
| Georgia | 698252002 IVCD | CD | 37 | 0.0181 | 0.6854 | 0.6802 |
| Georgia | 713426002 IRBBB | CD | 78 | 0.0382 | 0.8843 | 0.8918 |
| Georgia | 713427006 CRBBB | CD | 10 | 0.0049 | 0.9103 | 0.9198 |
| Georgia | 733534002 complete LBBB | CD | 0 | 0.0000 | undefined | undefined |
| fold9 | 164930006 ST elevation | STTC | 3 | 0.0014 | 0.8444 | 0.8207 |
| fold9 | 164934002 T-wave abnormality | STTC | 205 | 0.0955 | 0.8134 | 0.8075 |
| fold9 | 270492004 IAVB | CD | 80 | 0.0373 | 0.7817 | 0.7660 |
| fold9 | 428750005 nonspecific ST/T annotation | STTC | 273 | 0.1272 | 0.8344 | 0.8291 |
| fold9 | 429622005 ST depression | STTC | 100 | 0.0466 | 0.8285 | 0.8397 |
| fold9 | 445118002 LAFB | CD | 163 | 0.0760 | 0.9062 | 0.9100 |
| fold9 | 59118001 recorded unspecified RBBB | CD | 166 | 0.0774 | 0.9157 | 0.9180 |
| fold9 | 59931005 T-wave inversion | STTC | 30 | 0.0140 | 0.7838 | 0.7996 |
| fold9 | 698252002 IVCD | CD | 78 | 0.0363 | 0.6898 | 0.6968 |
| fold9 | 713426002 IRBBB | CD | 112 | 0.0522 | 0.8658 | 0.8669 |
| fold9 | 713427006 CRBBB | CD | 55 | 0.0256 | 0.9716 | 0.9766 |
| fold9 | 733534002 complete LBBB | CD | 54 | 0.0252 | 0.9642 | 0.9589 |

Bootstrap:2000 record-level resamples, shared across labels/models/training seeds within each population; seeds do not create extra people. Conditional percentile intervals; undefined samples excluded from quantiles with exact counts, no invented negatives. No patient IDs in targets; correlated repeated patients may make record intervals too narrow. Fold9 also has repeated patients; the requested record-level reference CI shares that limitation. Source and target resamples independent. No confirmatory multiplicity or transfer advantage claim.

## Seven-module shift evidence

Primary five-view only, both source seeds. Deterministic label-free hash sampling at most256 records/group, pooled median RBF bandwidth,999 permutations, (1+extreme)/1000; Holm7 within domain/seed. Native feature scales differ, so MMD values across modules are descriptive and bandwidth-normalized, not a causal adaptation ranking.

| Domain | Seed | Module | MMD² | Permutation p | Holm7 p |
|---|---:|---|---:|---:|---:|
| CPSC | 101 | fusion_1 | 0.02134 | 0.0010 | 0.0070 |
| CPSC | 101 | fusion_2 | 0.02567 | 0.0010 | 0.0070 |
| CPSC | 101 | view_1 | 0.01046 | 0.0160 | 0.0320 |
| CPSC | 101 | view_2 | 0.03444 | 0.0010 | 0.0070 |
| CPSC | 101 | view_3 | 0.00876 | 0.0310 | 0.0320 |
| CPSC | 101 | view_4 | 0.01757 | 0.0010 | 0.0070 |
| CPSC | 101 | view_5 | 0.02834 | 0.0010 | 0.0070 |
| CPSC | 202 | fusion_1 | 0.01884 | 0.0010 | 0.0070 |
| CPSC | 202 | fusion_2 | 0.02023 | 0.0010 | 0.0070 |
| CPSC | 202 | view_1 | 0.01708 | 0.0010 | 0.0070 |
| CPSC | 202 | view_2 | 0.03243 | 0.0010 | 0.0070 |
| CPSC | 202 | view_3 | 0.00422 | 0.2160 | 0.2160 |
| CPSC | 202 | view_4 | 0.01408 | 0.0010 | 0.0070 |
| CPSC | 202 | view_5 | 0.02313 | 0.0010 | 0.0070 |
| Chapman-Ningbo | 101 | fusion_1 | 0.01094 | 0.0030 | 0.0120 |
| Chapman-Ningbo | 101 | fusion_2 | 0.01091 | 0.0070 | 0.0150 |
| Chapman-Ningbo | 101 | view_1 | 0.02231 | 0.0010 | 0.0070 |
| Chapman-Ningbo | 101 | view_2 | 0.01183 | 0.0050 | 0.0150 |
| Chapman-Ningbo | 101 | view_3 | 0.00924 | 0.0300 | 0.0300 |
| Chapman-Ningbo | 101 | view_4 | 0.01323 | 0.0020 | 0.0100 |
| Chapman-Ningbo | 101 | view_5 | 0.03586 | 0.0010 | 0.0070 |
| Chapman-Ningbo | 202 | fusion_1 | 0.01047 | 0.0040 | 0.0120 |
| Chapman-Ningbo | 202 | fusion_2 | 0.01161 | 0.0030 | 0.0120 |
| Chapman-Ningbo | 202 | view_1 | 0.01970 | 0.0010 | 0.0070 |
| Chapman-Ningbo | 202 | view_2 | 0.01382 | 0.0040 | 0.0120 |
| Chapman-Ningbo | 202 | view_3 | 0.00459 | 0.1830 | 0.1830 |
| Chapman-Ningbo | 202 | view_4 | 0.01246 | 0.0010 | 0.0070 |
| Chapman-Ningbo | 202 | view_5 | 0.02365 | 0.0010 | 0.0070 |
| Georgia | 101 | fusion_1 | 0.03004 | 0.0010 | 0.0070 |
| Georgia | 101 | fusion_2 | 0.03389 | 0.0010 | 0.0070 |
| Georgia | 101 | view_1 | 0.01083 | 0.0160 | 0.0160 |
| Georgia | 101 | view_2 | 0.04567 | 0.0010 | 0.0070 |
| Georgia | 101 | view_3 | 0.03420 | 0.0010 | 0.0070 |
| Georgia | 101 | view_4 | 0.01780 | 0.0010 | 0.0070 |
| Georgia | 101 | view_5 | 0.01859 | 0.0010 | 0.0070 |
| Georgia | 202 | fusion_1 | 0.02569 | 0.0010 | 0.0070 |
| Georgia | 202 | fusion_2 | 0.02992 | 0.0010 | 0.0070 |
| Georgia | 202 | view_1 | 0.02472 | 0.0010 | 0.0070 |
| Georgia | 202 | view_2 | 0.02461 | 0.0010 | 0.0070 |
| Georgia | 202 | view_3 | 0.02404 | 0.0010 | 0.0070 |
| Georgia | 202 | view_4 | 0.01504 | 0.0020 | 0.0070 |
| Georgia | 202 | view_5 | 0.01295 | 0.0020 | 0.0070 |

CPSC: largest descriptive median-kernel MMD values in view_2, view_5, fusion_2. Holm-significant seed counts out of2: `{"fusion_1": 2, "fusion_2": 2, "view_1": 2, "view_2": 2, "view_3": 1, "view_4": 2, "view_5": 2}`. This is representation-shift evidence, not measured adaptation benefit.

Chapman-Ningbo: largest descriptive median-kernel MMD values in view_5, view_1, view_4. Holm-significant seed counts out of2: `{"fusion_1": 2, "fusion_2": 2, "view_1": 2, "view_2": 2, "view_3": 1, "view_4": 2, "view_5": 2}`. This is representation-shift evidence, not measured adaptation benefit.

Georgia: largest descriptive median-kernel MMD values in view_2, fusion_2, view_3. Holm-significant seed counts out of2: `{"fusion_1": 2, "fusion_2": 2, "view_1": 2, "view_2": 2, "view_3": 2, "view_4": 2, "view_5": 2}`. This is representation-shift evidence, not measured adaptation benefit.


Significant module shift is selection evidence only. Different module shifts may motivate selective adaptation; uniform shift provides less differentiation. Neither proves that updating any module improves prediction.

## Pre-stated feasibility verdict

Rule frozen in RUN_PROTOCOL.json before scoring: primary five-view macro proxy AUROC DROP≤−0.05, OR both target AP95% upper bounds below source AP reweighted to target priors; AND expected RESERVED positives≥150 each; AND preprocessing/mapping checks pass. These criteria refer only to annotation proxies. No clinical-negative equivalence is claimed.

| Domain | Verdict | Macro proxy AUROC DROP | AP alternative | Expected support |
|---|---|---:|---|---|
| CPSC | SUITABLE | -0.0987 | True | True |
| Chapman-Ningbo | BORDERLINE | -0.0360 | False | True |
| Georgia | SUITABLE | -0.0520 | False | True |

Recommended primary target: CPSC. Backup: Georgia. Domain selection on FEAS makes any later project a development-selected target study, not confirmation that selection generalizes.

## Phase3B design proposal — NOT executed

Source PTB-XL; target only an eligible RESERVED cohort after a separate reviewed label/patient-independence protocol. Budget32/64/128 distinct labeled records, three fresh source seeds if permitted, repeated paired draws. Arms: source-only, head-only, full fine-tune, parameter-matched random, re-engineered MMD/FormulationA, and a conditional literature-faithful DMV-MAT arm. Prefer fixed target update schedules to avoid concealing hundreds of stopping labels; otherwise stopping≥256 must be counted separately. Every evidence, fitting, stopping and calibration label counts; Full-cost must access the same union. Keep an untouched future target evaluation, account for lack of patient IDs, overlapping code annotations, teacher/source training history and record duplicates. No future split is frozen by this report.

**Faithful-arm evidence blocker:** the local secondary review `DMV-MAT Paper and Official Code Teardown.pdf` describes Cui etal., TNNLS36(2):2917–2930 (2025), DOI10.1109/TNNLS.2024.3350085. It reports JS transfer scores and a two-component GMM selection for feature extractors, but released model/GMM definitions are missing, repository licensing unresolved, and MMD loss calls are commented out. MK-MMD pretraining is NOT verified from the available offline evidence. The requested MK-MMD-pretraining+transfer-score+GMM feature-only arm must therefore remain conditional on primary-paper/code/license verification; otherwise name it an inspired variant, not a faithful reproduction. MMD module-selection evidence and an alignment training loss are distinct. No network was used for this Part2 inquiry.

## Limitations and integrity

1. **Explicit deviation:** use actual full10s five-view/flat pipeline, replacing original2.5s-window request. No windows or test-time averaging.
2. **Explicit deviation:** no PTB-XL-vs-target waveform duplicate audit, as authorized. Excluding named PTB/PTBXL/INCART archives does not empirically prove absence of overlap or shared patients.
3. No stable target patient identities; exact native duplicates do not establish clinical independence. Multi-source domains are dataset groupings, not verified homogeneous hospitals/devices.
4. Unlisted diagnoses are unannotated, not confirmed absence. Apparent loss may arise from label mismatch, coding completeness, prevalence or truncation, as well as signal shift. Clinical external validation is out of scope.
5. PTB-XL fold9 was used for historical checkpoint selection; its reference is optimistic development evidence. Fold10, Phase2 O and all protected Phase2 artifacts were untouched.
6. RESERVED was used only for stratification and ID manifests, never inference/diagnostics; expected counts are estimates. Future patient grouping and endpoint adjudication remain conditions.
7. Bootstrap intervals condition on already-trained models, omit source-training/selection uncertainty and may understate within-person dependence. No new transfer experiment or clinical utility established.

## Files, tests and repository

New source: `src/dmvmat_ecg/phase3/`; new scripts: `scripts/phase3/`; synthetic tests: `tests/phase3/`; aggregate artifacts: `results/phase3a/`; this report. Raw data, caches, split identities, predictions and representations stay under git-ignored `data/`. No raw/patient artifacts staged, no commit/push/merge/PR. Preservation and test receipts are recorded separately in results/phase3a; old tracked files are checked against the starting hashes.
