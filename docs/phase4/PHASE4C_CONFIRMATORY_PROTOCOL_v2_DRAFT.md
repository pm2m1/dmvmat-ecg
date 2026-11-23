# Phase4C v2 — DRAFT, UNADOPTED; no test access

**PRIMARY_CHOICE: UNRESOLVED. AUTHOR_ADOPTION: UNRESOLVED. TEST_EXECUTION: NOT_AUTHORIZED.**

Branch `phase4c-confirm`. This is amendment2's requested draft, not a finalized preregistration or an authorization. The author will decide after the [plain-retrain validation control](PLAIN_RETRAIN_PROTOCOL.md). The earlier [protocol](PHASE4C_CONFIRMATORY_PROTOCOL.md) is preserved unchanged. No CPSC test signal, label, prediction, metric, membership inspection or sealed/reserved file is accessed in this task. No commit/push/merge.

## 1. Candidate primary; author must choose before test access

Let AP(m,b,e,j) be observed-label AP for CD or STTC in episode e. Macro AP is the equal mean of the two label APs; the estimand then averages equally across the nine fixed source/draw episodes. Never pool repeated participants or calculate AP of mean probabilities.

Exactly one candidate may become the primary after explicit author selection and a committed final protocol:

- **P-reg:** `(1/9) sum_e [macro_AP(M4,128,e) − macro_AP(M3,128,e)]`. Does weighted within-source MMD regularization help in the matched source/source two-forward pipeline?
- **P-align:** `(1/9) sum_e [macro_AP(M5,128,e) − macro_AP(M4,128,e)]`. Does substituting target-domain data add benefit beyond the matched source/source placebo? The substitution includes domain-dependent BN effects and does not isolate only an MMD gradient.

For the author-selected primary, success remains **lower two-sided95% paired-bootstrap CI bound >0**, with at least9500 supported draws. Do not choose between candidates after test access, report success for either one post hoc, or treat two candidates as two unadjusted primary opportunities. No primary has been chosen by the assistant. The unselected candidate has no extra unadjusted superiority test; it is secondary only if included in the preserved family below, otherwise descriptive.

## 2. Fixed five-method roster and development references

| ID | Method | Source/checkpoint and validation replay reference |
|---|---|---|
| M1 | Source-only, original | Three original Experiment4 source checkpoints and normalization; no target updates |
| M2 | Full FT, original | Three original sources;27 recorded original full-FT A/B pairs (B64 all-one oracle references) |
| M3 | Retrain lambda0 + full FT | Three new source-only lambda0 checkpoints;27 amendment2 A/B pairs |
| M4 | Placebo lambda0.1 + full FT | Three source/source placebo checkpoints;27 amendment1 A/B pairs |
| M5 | Aligned lambda0.1 + full FT | Three existing selected aligned checkpoints;27 Phase4B aligned-full A/B pairs |

Budgets64/128/256; source seeds15101/15102/15103 × draw seeds16101/16102/16103; identical F+G memberships and source normalization. All target fits use batch32 replacement sampling, `seed([78501,s,d,b])`, exactly200 updates, AdamW lr3e-4 constant, weight_decay0.01, betas(0.9,0.99), eps1e-8, all modules/head trainable. No additional source training, lambda search, tuning, calibration, rescaling, feature selection or threshold selection in confirmation.

The lambda0 source retains the placebo's independent PTB second forward, BN updates and dropout RNG consumption but has no MMD optimization term. Thus M3−M2 measures the combined source-pipeline/retraining effect, not an identical-source seed repeat. Source checkpoint epochs were selected by the original fold9 AP/BCE rule. M4 uses no CPSC pretraining inputs. M5's lambda was already chosen in Phase4B development; this prior selection and repeated A/B use remain limitations. AdaBN+full is removed: ranking results coincided with original full FT, but universal bitwise score/state equality was not established. Aligned source-only is absent from this requested v2 roster.

Retain the earlier aligned-source quality gate from v1: its recorded mean fold9 macro AUROC must not be more than0.02 below the original-source mean. The unchanged original/aligned values in v1 already pass that gate. The new lambda0 and placebo source qualities are reported for author review without post-hoc replacement of a seed or checkpoint rule.

The logical roster is135 cells (5×3×9). M1 has three physical predictors aliased across budgets/draws; the other four methods require108 authenticated target states, giving111 physical predictor slots. Do not collapse adapted procedures on the basis of similar AP. The original A/B splits, source identities, validated proxy mapping and adaptation-only exclusion guard remain unchanged. Unlabelled CPSC U remains2859 adaptation records excluding A/B; no test signal may enter training, normalization or BN adaptation.

## 3. Fitted-state authentication before any test signal

No adapted checkpoints were saved. Deterministic replay may reconstruct each fixed200-update state only after final author adoption/commit and explicit continuation. It is authentication, not retuning. Verify stored source/reference file hashes, membership/order/normalization/code/settings bindings first. Every one of the108 regenerated fits must reproduce its **stored split-A AND split-B probabilities byte-for-byte**, with identical shape, dtype and row order, plus final full-state hash, training history and recipe bindings. State and buffer identities matter; matching AP or numerically close arrays is insufficient.

**Any mismatch → STOP before any test signal or prediction.** Preserve the failure; no tolerance relaxation, alternate seed, checkpoint substitution, environment changes until a fit passes, or target refitting after outcomes. Original target references stay in Phase4B, placebo references in amendment1, and lambda0 references in amendment2; none are old test outputs. Hold authenticated states in memory. No new source or public target checkpoint is authorized in confirmation.

Before any future access, the author must resolve PRIMARY_CHOICE, adopt/commit the final complete protocol, and explicitly continue. Verify the committed version is unchanged. This draft, an earlier continuation phrase, or completion of validation is insufficient. Use a fresh private destination and single exclusive controller; preserve all failed/consumed historical operations. No confirmatory replay or test operation occurs in this amendment2 task.

## 4. Same test cohort and known prior use

The future cohort remains the locked3860-record CPSC test half, same canonical eligibility/order and known duplicate groups as Experiment4. Only after the future gate may membership/group-only metadata be inspected. If the original physical rule yields a different eligible count, STOP before outcomes: no silent exclusions, short-signal padding or denominator change. No PTB-XL fold10 or other reserved population enters the operation.

Use the identical V2 CD/STTC annotation-union proxy decoder/mapping, CD then STTC. These are not adjudicated diagnoses. Unmapped well-formed codes are integrity failures; absent/malformed/conflicting outcomes stay unknown. Require95% observed per label and both classes, report known/unknown denominators, and never convert unknowns to negatives.

**Prior-use acknowledgement:** Experiment4 already evaluated this same test cohort. A future single Phase4C evaluation is not the cohort's first-ever use or a globally untouched external validation. Phase4A/B and these amendments are validation-only, but do not erase earlier study-level test exposure. Final reporting must say so.

## 5. Prediction freeze before labels; single evaluation

After all108 replay gates pass, create a fresh private predictor roster binding the committed final protocol, code, source/target states, normalization, row order, preprocessing and mapping identity. Require all135 logical cells/111 physical slots, including explicit M1 aliases. Refuse any existing started/consumed destination; never reset a historical operation.

Signal-only inference uses original CPU float32/four-thread behavior, batch128, eval mode/no gradients, fixed physical calibration/lead order, first10seconds/100Hz/12×1000. Parameters/BN buffers must remain unchanged. A separate restricted capability permits exactly the locked test signal/group fields, never test labels before the seal; do not disable the development exclusion guard globally. For mixed metadata use field projection or stop. For WFDB read only the original unbuffered record+12 signal-line prefix, not diagnosis comments/full-header hash. MAT access must separate the declared signal field from outcomes.

Save every physical probability array (3860,2), finite and within[0,1], in the fixed order. Hash every array, alias, predictor/input binding and complete roster; verify and finalize a single all-method prediction seal **before any test label is read**. An incomplete roster cannot unlock outcomes. After outcome access, never regenerate, reorder, replace or revise predictions.

Only the verified complete seal permits one new Phase4C V2 label-read/joint-analysis operation. Do not use old test outputs, legacy decoding, consumed runners or receipts. Cache outcomes once privately; validity failures preserve the seal/failed receipt and stop, without changing mapping or data rules. A failed/interrupted attempt remains consumed, without automatic retry. Public reporting contains permitted aggregates only; arrays, IDs, weights, raw signals and checkpoints remain private.

## 6. Unchanged bootstrap and uncertainty rules

Use Experiment4's `phase3b_eval/statistics.py` shared_weights/bootstrap_rankings/confidence construction: **10000 replicates, seed78701**. Sort unique preexisting known-duplicate-group IDs, multinomial-resample groups with replacement, broadcast each weight to its records (singletons reduce to record resampling). Use identical weights for all arms, labels, episodes, budgets and comparisons; no new duplicate search or claim of unknown patient linkage.

Recompute observed-label, tie-aware weighted AP/AUROC for each replicate, then strict equal-label/equal-episode means and paired differences. Two-sided percentile95% CIs; at least9500 finite supported replicates, report unsupported counts and never redraw/replace them with zero. A deficient primary support count cannot yield success. These intervals condition on fixed fitted models/seeds/draws and development choices; they omit source-training/selection uncertainty and unknown clustering.

## 7. Preserve the existing95-member Holm family explicitly

To avoid silently redesigning the secondary family when the roster is renumbered, this draft preserves the existing family **literally**, translating old aligned M3→new M5, old original M2→new M2, old placebo M4→new M4:

- M5−M2 at64 labels:24 endpoints.
- M5−M2 at128 labels:23 endpoints, excluding its macro AP (the earlier protocol's primary).
- M5−M2 at256 labels:24 endpoints.
- M5−M4 at128 labels:24 endpoints.

Total95. Each full24-endpoint cell comprises macro AP/AUROC, CD AP/AUROC, STTC AP/AUROC, and accuracy/precision/recall/specificity/F1/balanced accuracy at0.5 for CD, STTC and strict macro means. Preserve undefined endpoints with p=1; no family reduction after outcomes. M3−M2 and M4−M3 are descriptive unless the author selects P-reg as the single primary or prospectively amends the family before any access.

**Author adoption note:** P-align is already one endpoint in this preserved family. If selected as primary, its unadjusted primary CI rule and its existing Holm-adjusted secondary entry must be labelled separately; they are the same evidence, not independent findings. Literal preservation leaves the earlier M5−M2 macro AP128 outside the secondary family even though it is no longer the candidate primary. These bookkeeping consequences are explicit for author review, not a hidden family substitution. This remains a draft until the author adopts the final primary and secondary allocation; do not automatically resize/redefine it to accommodate a preferred result.

For supported endpoint difference d and bootstrap differences d*: `p=(1+count(abs(d*−d)>=abs(d)))/(1+R_valid)`. This is the same two-sided null-centered approximate conditional bootstrap test, not exact randomization. Undefined point estimates or fewer than9500 valid replicates get unavailable CI and p=1. Use all95 p-values, deterministic endpoint-name ordering for ties, and Holm step-down alpha0.05: sorted adjusted p_i=min(1,max_{k<=i}((95−k+1)p_k)). Report raw/adjusted p and pointwise CIs; do not call CIs Holm-adjusted. A positive secondary needs adjusted p<=0.05 for an improvement claim; significant negative effects are deterioration. The selected primary uses its fixed CI rule; no extra unadjusted tests or post-hoc budgets/cutoffs.

## 8. Fixed0.5 metrics and missingness

Positive iff probability>=0.5. Compute integer TP/TN/FP/FN on observed outcomes, report unknowns separately. Accuracy=(TP+TN)/N; precision=TP/(TP+FP); recall=TP/(TP+FN); specificity=TN/(TN+FP); F1=2TP/(2TP+FP+FN); balanced accuracy=(recall+specificity)/2. Zero denominators are undefined. Macro requires both labels; episode mean requires all episodes, including inside bootstrap. Do not silently average only supported values. Ranking AP/AUROC require both classes. Fractional descriptive mean counts are not one confusion matrix or independent cohorts. No threshold sweep, F1-maximizing choice, test calibration or rescaling.

## 9. Reporting and current stopping point

If eventually adopted and genuinely executed, report all5×3 episode-mean macro/per-label metrics, selected primary and its support/CI/verdict, all95 secondary results/raw+adjusted p/support, fixed0.5 metrics, state/replay/prediction freeze checks, source fold9 quality, runtimes/failures and known historical test reuse. No selective reporting or clinical-deployment claim.

**Now STOP after amendment2 validation and this draft.** No primary selection, finalization, commit, confirmatory replay, test signal/label loading, test prediction or evaluation is authorized by this document.

Amendment2 completion references: [plain-retrain results](PHASE4B_RESULTS.md#plain-retrain-control), [new aggregates](PHASE4B_RETRAIN_AGGREGATES.json), and [verification](PHASE4B_RETRAIN_VALIDATION.json). All three lambda0 source checkpoints and27 validation A/B fit pairs exist; source/target repeat checks pass. These outcomes do not fill PRIMARY_CHOICE or AUTHOR_ADOPTION, alter the draft's inferential rules, or authorize a test operation.
