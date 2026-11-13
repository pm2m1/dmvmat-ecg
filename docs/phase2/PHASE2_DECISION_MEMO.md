# Phase 2 decision memo

**2026-11-12 — DRAFT FOR HUMAN REVIEW.**

## Decision

**BLOCKED for implementation of a real independent-clinical-target transfer
experiment.** The seven-module backbone is a useful engineering starting point;
patient independence, reliable negative labels and eligible support have not
been established. This is an evidence gate, not a claim that transfer is
impossible or that a different architecture will solve annotation problems.
No transfer model, source-task head, predictor smoke test or evaluation was run.

## What the completed project establishes

The project at `1f0ff9031e9e1e2adc97962922fd97705570baf2` contains a 212,261-parameter
five-view CNN, explicit seven-owner representations, two source seeds and
preserved private checkpoints. It completed 11 real predictor attempts (three
smoke/eight full), with one locked PTB-XL fold-10 access. Our read-only audit
verified the stored lock's file hashes and best/final checkpoint bytes. See
[technical/data audit](EXTERNAL_ECG_DATASET_AUDIT.md) for exact paths/IDs.

Frozen benchmark results: five-view macro AUROC/AP 0.92139/0.81405, matched flat
0.91733/0.80657, XResNet validation-best 0.92985/0.82759. The five-view-minus-flat
AUROC interval barely excludes zero; its AP interval includes zero. Five-view
underperforms XResNet on both ranking metrics. Bounded augmentation did not
establish a ranking advantage. These are existing internal benchmark findings,
not new external evidence. The source checkpoint families here are designated
for methodological controls, not chosen anew using fold-10 feedback. Exact
source fidelity, conditional intervals and limitations remain in
[the completed summary](../FINAL_RESEARCH_SUMMARY.md).

The current ECG selector is a synthetic interface, not a completed multi-label
adaptation algorithm. It lacks signed source-relative retention/bootstrap,
patient-aware evidence handling, partial-label training, target-label accounting,
epoch-zero adaptation and robust frozen-BatchNorm enforcement. Closing those gaps
requires separately approved, versioned code. Do not reset the old evaluated
ledger, modify its five-label trainer or reuse its test for policy development.

## Corrections that materially affect the research claim

1. The authors' released DMV-MAT scripts use Jensen–Shannon divergence and
   task-specific feature logic. Our MMD/permutation/bootstrap seven-module
   logical rule is a later re-engineering. The original full paper was not
   available for complete mathematical verification; source code is not a licence
   to claim exact reproduction or to copy unlicensed implementations.
2. Five diagnostic superclasses do not automatically harmonize across ECG
   databases. Normal sinus rhythm is not normal ECG, broad MI is not identical
   to a source ECG-infarction-pattern/acute-injury superclass, and unlisted
   conditions are not confirmed negatives.
3. CinC 2021 contains PTB-XL; it is not an independent target as a whole. The
   original Chapman release, its Challenge subset and combined Chapman/Ningbo
   copies are not different independent cohorts by filename.
4. Release/codebook conversion can collapse phenotypes. The combined dictionary's
   left-sided conduction abbreviations share a code; its IRBBB coverage differs
   from the Challenge count mapping. Preserve code/version semantics.
5. More stable masks or larger MMD are not evidence of predictive benefit.
   Frozen BatchNorm buffers, effective drift, total accessible labels and
   matched update cost are necessary to interpret any future comparison.

Detailed provenance/citations are in the other five documents; no unsupported
clinical equivalence is adopted here.

## Ranked candidate external populations

| Rank | Candidate | Why investigate | Current blocking evidence |
|---|---|---|---|
|1|Georgia public component, 10,344 records|Different acquisition provenance; published exact-code support for proposed IAVB/IRBBB (769/407 records)|Patient linkage, comprehensive negative status, 10-second eligible count, code certainty and cross-source overlap |
|2|Ningbo, 34,905 component records|Larger 10-second cohort, published IAVB/IRBBB support 893/246; potential additional external database|Component-to-person/release crosswalk, IRBBB dictionary/version mismatch, labels/negatives and overlap; lower rare-label support than raw size suggests |
|3|Original Chapman or its Challenge subset, **one representation only**|Original authors describe one ECG per person; clinically documented acquisition|Original/converted identity and labels differ; Challenge reports no IRBBB positives, so current two-label task unsupported; cannot invent negatives |
|Reject as independent target|Entire CinC 2021 / unseparated combined copies|Useful distribution framework and metadata|Contains source/copy material and heterogeneous labels; exclusions and component linkage mandatory |

Rank is based on provenance and task feasibility, not expected performance.
Ningbo may become first if its evidence clears the gates and Georgia's does not;
that would require a documented pre-training protocol amendment. No target is
currently certified independent at the patient level. Cross-database waveform
matching cannot rule out a shared person with different recordings.

## Recommended task and method shortlist

Do **not** directly transfer the five-superclass head. Investigate a new,
clinically reviewed two-label task: first-degree AV block (`1AVB` ↔ `270492004`)
and incomplete right bundle branch block (`IRBBB` ↔ `713426002`). These positive
concept correspondences are verified; negative labels are not. Require documented
complete phenotype ascertainment or blinded expert adjudication, observation
masks, ambiguity policy and eligible counts. See
[crosswalk](DIAGNOSTIC_LABEL_CROSSWALK.md). AFIB is another possible future task,
not an authorized replacement to obtain easier AUROC. No full waveform download
is requested by these planning documents.

Preserve the five encoders/two fusion modules and always-trainable new two-label
head. Primary method proposal A uses per-label signed class-retention evidence;
secondary B standardizes contrasts over the other label's co-occurrence strata.
Both have explicit low-support abstention, whole-patient uncertainty and
permutation-calibrated multikernel MMD corroboration. B may be infeasible because
joint positives are scarce. No automatic alignment loss, learned meta-selector,
attention redesign or architecture search is justified in this first transfer
study. [Method definitions](MULTILABEL_DMV_MAT_METHOD.md) specify which decisions
differ from the historical binary selector.

## Proposed fair experiment and scientific interpretation

[The protocol](TRANSFER_EXPERIMENT_PROTOCOL.md) proposes:

- One fixed patient/duplicate-group-disjoint target O holdout (30%), pilot P
  (10%), fitting/evidence reservoir A (45%), stopping/calibration reserves
  (7.5% each), with isolated role access and recorded labels used for auditing.
- Source-task conversion on PTB training/fold9 only; no new fold-10 selection.
  Both unaugmented MV seeds and both XResNet seeds are retained.
- Fitting tiers 128/512/2,048 with separate selector 128/256/512, fixed E512/C512;
  record actual per-label positives and all screened/observed records. Natural
  prevalence is retained. Optional exact-positive stress tests are outside budget.
- Source-only, target-only, fitting-matched head/full, cost-matched head/full,
  parameter-matched random, two evidence policies and XResNet reference.
  Primary **Selective-A versus Full-cost** at F512/G256, mean six paired episodes.
  Full-cost can train on selector-accessible G labels. Same target update count;
  measured compute reported, not falsely called equal FLOPs.
- Exact macro AP primary, label prevalence alongside AP, AUROC/calibration and
  label efficiency secondary. Shared patient-cluster 10,000-draw paired CIs;
  six secondary AP comparisons with adjusted intervals. Two source seeds and
  overlapping draws are not independent clinical populations.
- A +0.005 AP gain is a proposed practical engineering signal, not medical
  utility. Positive point estimates are insufficient. A benefit must not merely
  reflect extra evidence/stopping labels, a weak full-fine-tuning schedule or
  degenerate random controls. Negative transfer and epoch-zero retention are
  informative outcomes, even if selection fails.

This is not yet a confirmatory external study: research direction uses prior
results, source selection history is known, endpoints are unresolved and policy
settings are new. After prospective approval and an untouched external target
holdout it can provide stronger within-design external-target evidence, not
clinical deployment validation or population-wide claims.

## Compute recommendation

The [compute plan](COMPUTE_AND_FEASIBILITY.md) uses measured PTB-XL timings,
common CPU fallback and explicit unmeasured workload qualifications. Disk/weight
memory are sufficient in principle (about 710 GiB free; 48 GiB unified memory).
XResNet validation and repeated evidence/uncertainty computation may dominate
small adaptations. Do not promise a precise runtime before a new permitted pilot.

Recommend a **20-attempt ceiling pilot** (18 planned, two technical reserve),
then a separately approved **172-attempt ceiling full matrix** (166 planned,
six technical reserve). These are proposals, not executed fits or authority to
download/fine-tune. Do not use unused reserve for favorable reruns. A second
independent target would need another endpoint/overlap audit and separate
authorization; no cloud upload or compute purchase is authorized here.

## Conditions requiring evidence and approval before implementation/execution

1. **Dataset:** choose and pin an actual distribution/version, native checksums,
   applicable licence/citations and permission for any required clinical review.
2. **Independence:** provide curator-supported patient identity/bijection,
   repeat-record grouping and cross-release duplicate audit; quantify remaining
   uncertainty. If unresolved, block an independent-cohort claim rather than
   redefining filenames as people.
3. **Endpoint:** ECG expert approval of both positives/negatives, observation
   completeness, code/likelihood conflicts, adjudication and timing; verify source
   and target support. No broad five-class equivalence claim.
4. **Population:** approve the 10-second restriction, physical-signal conversion,
   grouping, patient-level canonical record and actual split feasibility. Audit
   exclusions/verification bias, not only retained AUROC potential.
5. **Method:** review A/B's validity floors, bootstrap quantiles, Holm correction,
   voting/abstention and frozen buffers; approve new two-output source conversion.
6. **Fairness/statistics:** approve F/G/E/C costs, pilot-label accounting,
   update budgets, primary middle tier, effect-size interpretation, secondary
   family and conditional bootstrap estimand. If pilot outcomes change policy,
   amend prospectively and provide its label union to the cost comparator.
7. **Engineering:** separately authorize versioned implementation and synthetic
   verification, then data preparation/download, then a bounded pilot. Require
   passing role locks, immutable artifacts, failed-attempt accounting and owner/
   gradient/buffer tests before any real fit.
8. **Execution:** separately approve the finite full matrix only after pilot
   review; freeze all methods/checkpoints/calibration before a new one-time O
   authorization. Never reopen the old PTB-XL ledger or tune on either opened test.

## File inventory and integrity boundary

This task creates only six planning documents under `docs/phase2/`:

1. [External dataset and implementation audit](EXTERNAL_ECG_DATASET_AUDIT.md).
2. [Diagnostic crosswalk](DIAGNOSTIC_LABEL_CROSSWALK.md).
3. [Multi-label method](MULTILABEL_DMV_MAT_METHOD.md).
4. [Transfer protocol](TRANSFER_EXPERIMENT_PROTOCOL.md).
5. [Compute/feasibility](COMPUTE_AND_FEASIBILITY.md).
6. This decision memo.

No source/configuration/test change, predictive execution, raw-waveform download,
checkpoint deserialization, commit, push, merge or PR. Original project hashes
and ledger are checked read-only; historical Kaggle/Phase 1 repositories remain
unchanged. Small official codebooks, public code, item API metadata and one
header were inspected in memory only. No participant labels/rows are published
in these documents. The final task response records the terminal Git/integrity
check; these documents do not pretend that review is approval.

Terminal integrity verification: all **95 tracked files** matched the initial
SHA-256 inventory; all **48 files in the prior evaluation lock** and **16
best/final checkpoint files** matched their stored digests. Read-only SQLite
inspection still shows phase `evaluated`, 11 successful attempts, zero failed,
one access receipt and consumed evaluation authorization. Local Markdown file
links and cited historical source paths exist. No test command or predictor
script was executed. Git changes are exactly the six new untracked documents;
branch `main` and HEAD `1f0ff9031e9e1e2adc97962922fd97705570baf2` are unchanged.
