# Phase 4C — final confirmatory protocol

**Status: author-specified protocol finalized on 2026-11-23; awaiting author commit and explicit continuation. No test access or confirmatory replay is authorized in this writing task.**

Branch: **phase4c-confirm**. This version replaces v1 at this path and adopts the author's final choices from the preserved [v2 draft](PHASE4C_CONFIRMATORY_PROTOCOL_v2_DRAFT.md). The draft remains unchanged as history. The primary is now **M5 minus M3 at 128 labels**. The former candidate-primary allocation and 95-member Holm family are superseded by the fixed sequence below. They are not additional tests or alternative analysis options.

## 1. Research question, rationale and development evidence

Does target-aligned MMD source pretraining followed by full fine-tuning improve CPSC CD/STTC macro AP over a matched plain-retrain source pipeline? If that primary succeeds, the fixed sequence asks whether within-source MMD regularization helps, then whether target-domain alignment adds benefit beyond that regularizer.

The original Experiment 4 source pipeline uses one supervised forward pass per source minibatch. The newer aligned/placebo pipeline adds a second representation forward, changing BatchNorm running-statistic updates and dropout RNG consumption. The plain retrain uses the identical newer two-forward pipeline with lambda=0: BCE-only optimization, diagnostic MMD under no-grad, the same independent PTB second batch, and no MMD gradient. It is therefore the fair pipeline-matched comparator for M4 and M5. Removing the second forward would change the control.

Direct source comparisons found different parameters and buffers in all three plain-retrain versus original checkpoints, despite matching source settings, normalization, row identities, factory and recorded public code hashes. All 75 parameter tensors and 45 buffers differed per source. There are 15 BN layers in the view encoders and Dropout(0.2) in each fusion module; the second forward updates those BN layers and invokes both dropout modules again. The independent second-stream sampler does not advance the primary source-shuffle generator or global dropout RNG; the extra model forward does consume dropout RNG. Source-cache payloads match the aligned-source caches. Checkpoint packaging/provenance also differ, so raw-file inequality alone is not evidence of changed weights. Historical per-batch RNG/optimizer traces and library-version records were not stored in the original checkpoints; no exact first historical divergence is claimed from final states alone.

The hypotheses are motivated by the already completed Phase 4B and validation-only control amendments, not a new test calculation. At 128 labels, recorded split-B means across nine episodes were:

| Method | Macro AP | Macro AUROC |
|---|---:|---:|
| M2 original full FT | 0.645089 | 0.833343 |
| M3 plain retrain + full FT | 0.651443 | 0.837085 |
| M4 placebo MMD + full FT | 0.663445 | 0.839625 |
| M5 aligned MMD + full FT | 0.666673 | 0.840658 |

| Validation contrast | Macro AP difference [pointwise 95% CI] |
|---|---|
| M5 minus M3 | +0.015231 [-0.001916, +0.032270] |
| M4 minus M3 | +0.012002 [-0.004588, +0.028010] |
| M5 minus M4 | +0.003229 [-0.010351, +0.016690] |
| M3 minus M2, pipeline effect | +0.006353 [-0.007718, +0.020098] |

All four validation intervals include zero. At 256 labels, the combined aligned-minus-retrain AP interval was positive, but the separate regularizer and alignment-component intervals included zero. These reused-validation findings motivate the combined primary and ordered component questions; they do not establish equivalence, absence of benefit or confirmatory superiority. No source seed, checkpoint criterion, lambda or target setting is changed in response.

Sources: [Phase 4B results](PHASE4B_RESULTS.md), [plain-retrain aggregates](PHASE4B_RETRAIN_AGGREGATES.json), [placebo aggregates](PHASE4B_PLACEBO_AGGREGATES.json), [plain-retrain verification](PHASE4B_RETRAIN_VALIDATION.json), and [placebo verification](PHASE4B_PLACEBO_VALIDATION.json).

**Recorded history and author declaration:** the locked CPSC test set was evaluated once previously, in Experiment 4, and has never been used for tuning. Phase 4A/B and the control amendments used adaptation-pool validation A/B; the present choices are based on that development evidence. The earlier test evaluation remains study-level exposure: this is not a globally untouched cohort, a new independent dataset or clinical validation. No earlier test signal, label, prediction or evaluation output is inspected during protocol finalization or used for replay authentication.

## 2. Fixed methods, existing checkpoints and settings

| ID | Method | Existing source/state reference |
|---|---|---|
| M1 | Source-only, original Experiment 4 sources | Three original source checkpoints and their source normalization; zero target updates |
| M2 | Full fine-tuning, original sources | Original sources; 27 recorded original full-FT A/B pairs, using all-one oracle references at budget 64 |
| M3 | Plain retrain, lambda=0, new pipeline + full FT | Three existing lambda=0 source checkpoints; 27 amendment2 A/B pairs |
| M4 | Placebo MMD, lambda=0.1, PTB-XL versus PTB-XL + full FT | Three existing source/source placebo checkpoints; 27 amendment1 A/B pairs |
| M5 | Aligned MMD, lambda=0.1, PTB-XL versus unlabelled CPSC pool + full FT | Three existing selected aligned checkpoints; 27 Phase 4B aligned-full A/B pairs |

Use **only the existing hash-verified source checkpoints**, twelve distinct source checkpoints across the four source families; M1/M2 share the original three. No source retraining, replacement, lambda search or new checkpoint selection. No new target-training experiment, tuning or setting changes. The sole permitted optimization later is the explicitly required deterministic replay of already completed target fits to authenticate their existing states.

Budgets: **64, 128, 256**. Source seeds: **15101, 15102, 15103** crossed with draw seeds **16101, 16102, 16103**, yielding the same nine episodes per budget as Experiment 4. Reuse exact F+G label-draw memberships and the same source-only normalization/preprocessing. For M2–M5: **exactly 200 updates**, batch **32 with replacement**, AdamW learning rate **3e-4 constant**, weight decay **0.01**, betas **(0.9, 0.99)**, epsilon **1e-8**, and fit seed **seed([78501, source_seed, draw_seed, budget])**. All seven modules and the prediction head are trainable. No class-weight changes, calibration, rescaling, feature selection, threshold selection, early stopping or validation-driven fit selection.

M3/M4 source pretraining used PTB-XL only. M5's unlabelled CPSC pretraining pool was the existing 2,859 adaptation records excluding A/B, with target labels unused for that pretraining. Its lambda=0.1 was selected in Phase 4B development; that selection cost and repeated validation use remain limitations. Source checkpoint epochs followed the existing fold-9 AP/BCE tie rule. Retain the existing aligned-source quality gate: mean fold-9 macro AUROC must not be more than 0.02 below the original mean; the recorded unchanged aligned/original means (0.905576/0.902868) pass. No post-hoc source replacement.

AdaBN+full is not a method in this protocol. Its historical ranking results coincided with ordinary full FT, without universal bitwise state/probability equality. Aligned source-only is also outside this final roster.

The logical roster is **5 methods × 3 budgets × 9 episodes = 135 cells**. M1 has three physical predictors with explicit aliases across draws/budgets. M2–M5 require **108 authenticated target states**, giving **111 physical predictor slots** without further proven state equivalences. M1 rows have zero target labels/updates; budget labels on their aliases are nominal references, not additional fitting.

## 3. Primary estimand and success rule

For method m, budget b, episode e and label j in {CD, STTC}, compute AP(m,b,e,j) among observed outcomes. Define episode macro AP as the equal mean of CD and STTC AP, then average equally over the nine episodes:

    Delta_primary = (1/9) sum_e [macro_AP(M5,128,e) - macro_AP(M3,128,e)].

This is the **single primary hypothesis**, aligned MMD + full FT minus matched plain retrain + full FT at 128 labels.

Use the paired 10,000-replicate group/record bootstrap specified in section 5, with the same resample for both arms and all episodes/labels. **Success requires the lower bound of the two-sided 95% CI to be strictly greater than zero**, with at least 9,500 supported finite draws. A zero lower bound is not success. An undefined estimate or insufficient bootstrap support is invalid/incomplete and cannot open the secondary gate.

Do not use AP of averaged probabilities, pool repeated records as independent cohorts, substitute another budget/metric, introduce a practical-effect margin, or switch the comparator after outcomes. Report the estimate, CI, support and failure/invalid result regardless of direction. Failure to establish benefit is not proof of equivalence or no effect.

## 4. Fixed-sequence confirmatory secondaries

The only confirmatory secondaries are **macro AP at 128 labels**, tested in this exact order:

| Order | Contrast | Question | Eligibility for confirmatory testing |
|---|---|---|---|
| Primary | M5 minus M3 | Combined aligned pretraining benefit beyond the matched lambda=0 pipeline | Always, subject to validity gates |
| S1 | M4 minus M3 | MMD regularizer alone, in the matched source/source pipeline | Only if the primary succeeded |
| S2 | M5 minus M4 | Target alignment beyond the regularizer | Only if the primary and S1 both succeeded |

Each eligible secondary uses the same mean-of-nine-episode macro-AP estimand, paired bootstrap and **two-sided 95% CI lower bound >0** rule as the primary, including the 9,500-supported-draw minimum. This is a fixed-sequence confirmatory procedure at the specified nominal 95% level. No Holm procedure or former 95-endpoint family is applied in the final protocol.

If any earlier hypothesis fails, is undefined or has inadequate support, the confirmatory sequence stops. Every later contrast is **descriptive only**, even if its displayed CI excludes zero. Do not skip S1 to confirm S2, reopen a gate with another metric/budget, or characterize gate-closed evidence as confirmatory. Descriptive estimates/CIs may still be calculated for transparent reporting; that is not an eligible confirmatory test.

Report eligibility, whether tested confirmatorily, success/not-successful/invalid status, and the reason for each closed gate. No alternative post-hoc p-value test can replace the CI rule.

S1 measures adding weighted within-source MMD under the fixed source-training/checkpoint-selection recipe, including its downstream effects on selected weights. S2 substitutes the target-domain second stream for the source-domain stream with the same MMD recipe. Domain-dependent BatchNorm effects accompany that substitution; S2 is not a causal isolation of the MMD gradient alone.

## 5. Exact Experiment 4 paired bootstrap

Use [phase3b_eval/statistics.py](../../src/dmvmat_ecg/phase3b_eval/statistics.py), specifically shared_weights, bootstrap_rankings and confidence, and the existing weighted tie-aware ranking implementation.

- **10,000 replicates; seed 78701.**
- Reuse Experiment 4's preexisting canonical membership and known duplicate-group mapping. Sort unique group IDs as in the existing implementation, multinomial-resample groups with replacement, and broadcast each group's count to its records. Canonical singleton groups reduce to record resampling.
- Use identical resampling weights across every method, label, episode, budget and contrast. Do not infer unknown patient linkage or replace the frozen scheme with a new patient bootstrap, grouping rule or duplicate search.
- Recompute observed-label AP/AUROC within each replicate using exact tie handling, then strict equal-label/equal-episode means and paired differences.
- Use percentile **2.5th/97.5th** bounds for two-sided 95% CIs. Require **at least 9,500** supported finite replicates; report unsupported counts. Never redraw, replace undefined values with zero, or silently omit an unsupported episode/label to change the estimand.

These intervals condition on the fixed fitted models, source seeds, adaptation draws and development choices. They omit source-training/model-selection uncertainty and unknown patient clustering. Nine repeated adaptation episodes on one cohort are not nine independent test populations or cross-validation folds.

## 6. Descriptive-only outputs

No confirmatory claims or additional superiority tests arise from:

- All results at budgets **64 and 256**.
- Macro AUROC at any budget.
- Per-label **CD and STTC AP/AUROC**, including at 128 labels.
- Fixed-threshold **0.5** accuracy, precision, recall/sensitivity, specificity, F1, balanced accuracy and **NPV**, per label and strict macro mean.
- **M3 minus M2**, the retraining/pipeline effect, at every budget.
- **M1**, original source-only, as a zero-target-update reference.
- Any primary/S1/S2 contrast whose confirmatory gate is closed, or other arithmetic comparisons in the complete method table.

Report all five methods by budget, with episode-mean metrics, per-episode/per-label support and confusion counts. The predeclared primary/S1/S2 comparisons may also be displayed at 64/256 labels as descriptive arithmetic/paired intervals only. Pointwise descriptive CIs are not multiplicity-adjusted or confirmatory. Do not report post-hoc inferential p-values for a larger family or select whichever descriptive outcome looks favorable. No threshold sweep, F1-maximizing cutoff or new hypothesis selection.

For probability >=0.5 classify positive. On observed outcomes:

    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    Specificity = TN / (TN + FP)
    F1 = 2 TP / (2 TP + FP + FN)
    Balanced accuracy = (recall + specificity) / 2
    NPV = TN / (TN + FN)

Zero denominators are undefined, not measured zero. Compute per-label values, macro only when both labels are defined, and strict episode means only when all required episode values are defined. Report unknowns separately. Ranking AP/AUROC require both observed classes under the frozen handling. Keep integer confusion counts per episode/label; fractional mean counts are descriptive summaries, not one confusion matrix or a larger cohort.

## 7. Test population, endpoints and validity

The future test set is the **locked CPSC test half of 3,860 records**, identical to Experiment 4 in canonical eligibility, order and known duplicate groups. No test membership/header inspection occurs in this protocol-writing task. After the future continuation gate, inspect only permitted membership/group metadata at the prescribed stage.

Preserve the original physical/canonical-duplicate eligibility rules. If the unchanged physical rule cannot process the required 3,860 records, STOP before outcomes; do not silently alter the denominator, drop short signals, pad records or claim an unevaluated record was evaluated. No new duplicate search or outcome-dependent exclusions.

Outcomes are the unchanged verified **V2 CD/STTC annotation-union proxies**, in order CD then STTC, not adjudicated clinical diagnoses. Well-formed unmapped codes are mapping-integrity failures. Missing, malformed or conflicting annotations stay unknown. Preserve the V2 validity gate: at least **95% observed per label** and both classes present. Report known/unknown and positive/negative denominators. Unknowns never become negatives; a failed gate produces an invalid/incomplete evaluation, not success.

No PTB-XL fold 10 or other reserved/unrelated sealed material is accessed. Earlier Experiment 4 test outputs and consumed receipts are not prediction/replay references.

## 8. Later execution order and isolation

### 8.1 Author commit and explicit continuation

**STOP after writing and checking this protocol.** No fitting/replay, source-checkpoint loading, test signal/label/prediction access, scoring, commit, push or merge is performed in this task.

Later, only after the author commits this final protocol and replies **protocol committed, continue**, verify the exact committed document is unchanged and perform the permitted non-test preflight. Use a fresh private destination with no prior started/consumed marker or output. A continuation phrase does not waive input integrity, source availability, dataset-use conditions or runtime isolation. Do not reset any previous operation, receipt, ledger, authorization or failed attempt.

Before genuine execution, the implementation must pass synthetic-only checks for this final roster, the primary/S1/S2 gate truth table, descriptive-only endpoints including NPV/undefined handling, immutable prediction manifests, label separation, replay rejection and single-use behavior. This is a future preflight requirement, not new implementation or test-data work in the writing task.

### 8.2 Authenticate every target state using validation only

Only existing hash-verified source checkpoints may be loaded. Preserve source normalization, source/fit seeds, data memberships, preprocessing, label mapping and numerical code/settings bindings. Verify each stored A/B reference-file hash first.

Regenerate the **108 M2–M5 target fits** by deterministic replay of the recorded 200 updates. Each must reproduce its stored **split-A AND split-B probability arrays BITWISE**, with the same shape, dtype and row order, plus final full-state hash, training history and source/normalization/recipe bindings. Bitwise means byte-for-byte; matching AP, approximate equality or tolerances are insufficient.

**Any mismatch -> STOP and report before any test signal or prediction.** Preserve the failure; do not change a setting, seed, source, mask, budget, code path or environment until something passes. Replay is reconstruction/authentication only, not a new training/tuning experiment. Hold the authenticated states in memory; no target checkpoints are required or authorized by this protocol.

The runtime record-exclusion guard and **adaptation-only bundle remain in force for all replay/training code**. Do not disable the guard globally or provide a test loader to a fit. No CPSC test signal may enter replay, source normalization, BatchNorm updating or unlabelled U.

### 8.3 Test signals only; generate and freeze all predictions

Only after all 108 replay gates pass may the separate final prediction capability read the locked test membership/group metadata and **SIGNALS ONLY**. This is the first new Phase 4C test-signal operation. It permits no outcomes or earlier test predictions/results.

Use original CPU float32/four-thread inference, batch128, eval mode/no gradients, unchanged physical calibration/lead ordering and first10-second/100-Hz/12x1000 preprocessing. Verify parameter/BN state hashes before and after inference; no optimization, normalization fitting, BatchNorm adaptation or checkpoint selection.

For mixed metadata, field-project only permitted IDs/group/physical fields; stop if separation is impossible. For WFDB, use the unbuffered physical-prefix reader (record line plus12 signal lines), not a full-header/diagnosis-comment parser or whole-header checksum before the prediction seal. Read only the declared signal variable from MAT containers; stop if outcome-bearing values cannot be separated.

Generate and save predictions for **every method x budget x episode** in the fixed order, each logical array shape (3860,2), finite within [0,1]. M1's three physical vectors may be saved once with explicit aliases covering every nominal draw/budget; all135 logical cells must be present and resolvable. Do not alias distinct adapted fits on the basis of similar metrics.

The private prediction manifest must record **SHA-256 hashes** for all saved prediction arrays and the complete alias/roster, committed protocol and code identity, source/target state bindings, normalization/preprocessing/mapping identities, and record-order/group bindings. Verify all111 physical predictor slots and135 logical cells, then finalize one immutable complete prediction seal. No partial roster can unlock labels. No prediction may be regenerated, reordered or replaced after label access.

### 8.4 Labels only after the complete verified prediction manifest

**Only then load test labels** with the frozen V2 decoder/mapping and compute exactly the preregistered analyses, in the primary -> S1 -> S2 confirmatory order with the fixed gates and descriptive outputs above.

Use one fresh Phase 4C outcome-access/joint-analysis operation. Cache outcomes once privately; never invoke an old consumed runner or reuse its predictions/receipts. Mapping/header/validity failures preserve the seal and failed completion record and stop, without switching decoding or data rules after results.

**Single evaluation. No reruns; no changes after seeing results.** A started/consumed marker and exclusive controller protect against duplicate or interrupted attempts. Preserve failures/interruption permanently; no automatic retry or reset. No changed thresholds, grouping, CIs, success rules, endpoints, budgets, model settings or method roster after outcomes. Unexpected failures are reported as incomplete/invalid, not repaired by another test evaluation.

## 9. Final reporting and present stopping point

A later permitted report must include all5 x3 method/budget descriptive metrics and denominators; primary/S1/S2 estimates, paired CIs, support, gate eligibility/status and final verdict; per-label and fixed0.5 descriptive metrics including NPV; per-episode integer confusion counts; source/replay/prediction-manifest integrity, chronology, failures and runtime; and the acknowledged single earlier Experiment 4 use of the same test cohort. Do not claim independently untouched validation, clinical deployment, or statistical absence/equivalence from a failed superiority test.

Raw signals, labels, IDs, probabilities, group maps, fitted checkpoints, private manifests and receipts stay in protected local storage. Public release is limited to separately permitted aggregates; successful hashing is not disclosure approval.

**Protocol finalization is complete; execution is not started. Stop now for the author's commit and explicit continuation. No test data was accessed to write this document.**
