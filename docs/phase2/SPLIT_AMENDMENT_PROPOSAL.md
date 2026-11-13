# Phase 2C — prospective split amendment

**DRAFT — REQUIRES HUMAN REVIEW/ADOPTION. METADATA-ONLY AUTHORITY DOES NOT
AUTHORIZE TRAINING OR EVALUATION.**

The question remains whether DMV-MAT-inspired evidence-guided seven-module
adaptation improves target classification over full fine-tuning with matched
**total** target-label access and optimizer-update budgets. No architecture,
loss, endpoint, policy threshold or clinical domain is optimized here.

## Version and prospective specification

New version `joint-pattern-hamilton-v1`, fixed master seed **44301**. The
[specification](../../configs/phase2c/joint_pattern_v1.json) was written before
any real amended candidate generation; semantic SHA-256:
`28ec8f4d748cea15ca049ca6ba48efa8040428b1d18b3c436daaebc442f84484`.
The new generator accepts no seed override. Exactly one specification/candidate
was generated; repeating the same calculation once verified determinism, without
choosing between seeds. Existing E/C hash seeds33003/33005 and target permutation
seeds4101/4102 remain fixed. No patient scores or model performance informed
this amendment. The original draft seeds33001/33002 are retained only in the
preserved old candidate/reproduction trace, not searched as alternatives.

## Fixed elements

- Same PTB-XL1.0.3, original eligible folds1–9 only, same publisher/cached metadata
  checksums and same16,415 canonical patients; folds10 permanently excluded.
- Lowest age-eligible ECG ID per actual patient, recorded canonical age18–59
  source and60–89 target. No invented site/acquisition label or demographic
  network input. Serial diagnosis differences are not OR-combined or relabeled.
- Source85/15 train/stop; target O/E-reserve/C-reserve/A30/10/10/50 intent.
  Fractions are approximate integer allocations, not exact sample-size promises.
- E/C128 via label-independent hash order within reserves. G256 first and next
  F512 from each label-independent A permutation, nested F128 prefix. No
  positive quota, G enrichment, oversampling, exclusion or seed shopping.
- Existing A/B minimum10, source anchor cap256/class, D floor0.1, bootstrap
 4,999, permutations999, Holm-seven MMD and1−.05/(7×2) upper quantile. Original
  author JS and our MMD re-engineering remain distinct. MMD is evidence, not loss.
- New source from scratch only, source-only normalization and stopping; same
  seven modules,212,066 parameters/two outputs. No old checkpoint/normalization.

## Proposed allocation algorithm

Implementation [versioned splitter](../../src/dmvmat_ecg/phase2c/splits.py):99. This is **joint-stratum Hamilton apportionment**,
not an undocumented claim to implement a published iterative-stratification
package. For this two-label, one-record-per-patient contract it is simpler than
an unconstrained iterative solver and has an explicit integer error bound.

1. Verify patient/fold/annotation identities and canonicalize label-independently.
   Check native hash linkage across **all eligible recordings**, not only retained
   canonical ones. Cross-person duplicate ambiguity fails rather than being
   silently linked/excluded. Noncanonical recordings remain excluded permanently.
2. Partition canonical patients into their fixed age domains. Within each domain
   stratify on the complete observed vector00/10/01/11; genuine unknowns have
   separate `?` strata and retain masks. Do not pool/match source and target.
3. For stratum size N and role fraction f, compute exact rational Nf, allocate
   floor(Nf), then distribute the residual to the largest fractional remainders.
   Equal remainders break by SHA-256 of `[44301,domain,status,"quota",role]`.
4. Order patients within that stratum by SHA-256 of
   `[44301,domain,status,"patient",patient_id]`, allocate contiguous quota-sized
   blocks in the declared role order; sort each resulting reservoir by ECG ID.
   Each patient is indivisible with weight1; repeated-record counts do not
   affect allocation because exactly one canonical record is used. A future
   multi-record retained design needs a new size-weighted group protocol.
5. Each joint cell/role is within one patient of its fractional quota; aggregate
   role size may differ by the sum of stratum rounding errors. Preserving the
   four cells also approximately preserves both marginals and total counts.
   Unknowns never become negatives. The real annotation task contains none;
   an unknown-containing pilot would require explicit loss/selection review.
6. Apply the unchanged E/C and G/F samplers once. Audit all capacities, marginal
   support, source capped-anchor cells and every G draw; do not repair a failed
   draw or choose an alternate seed. Return blockers for primary support/capacity
   failure. A B-only support failure is reported as secondary abstention, not
   automatic license to redesign the primary study.

## Evaluation-role replacement and research history

Original O had2,668 patients and168 joint positives. Proposed O has2,633 and176;
**790 patients overlap** (1,878 leave old O,1,843 enter new O). Both candidates
and their feasibility-audit history remain private and immutable. Old role labels
had been audited; the observed absence of joint positives in A and concentration
in stopping motivated this amendment. New acceptance criteria (fixed domains,
proportional four-cell quotas, unchanged samplers/marginal floors, no score-based
selection) and seed were recorded before generating the new candidate.

No old/new O prediction or predictive performance was inspected. The changed O
is **not an untouched independent confirmation**: the research population and
label-feasibility information informed the design, and folds1–9 were used in the
prior classification study. It remains internal exploratory evidence. The original
fold10 cannot become a new pilot test; no claim of external clinical validity
or future-event prediction is justified.

After separate authorization, the primary estimand remains mean paired-draw raw
macroAP(A−Full-cost) at F512. Patient-cluster bootstrap must resample whole
CD/STTC vectors, masks and paired predictions with shared patient multiplicities
across all methods/draws:10,000 draws, seed64001,95% primary intervals; five
secondary AP contrasts use95% and Bonferroni99%. No test-driven split, pooled
estimand substitution or seed selection. Conditional intervals omit training/
policy-selection and design-amendment uncertainty; report those limitations.

## Budget impact and primary fairness

**Unchanged33+3:** one fresh source fit plus eight trained methods×two target
seeds/draws×two F sizes =33 planned predictor fits,36 hard ceiling including
failures/interruptions/retries. No actual attempts are consumed by metadata
preparation. Keep head-F/head-cost/full-F/full-cost/random-A/selective-A/
selective-B/target-only; source-only adds inference without a predictor fit.

At F128/F512, Selective-A and Full-cost both access640/1,024 distinct target
records including G256/E128/C128. Full-cost can train on F∪G; Selective-A uses
F gradients/G evidence. Head-F/Full-F use384/768 including E/C. Source-only raw
uses zero target labels; its calibrated variant explicitly uses C128. Auditor
labels and later evaluation labels are separate study-wide costs, not free
clinical annotation. No meta-selector search is introduced.

Source stopping remains source-only. New source_train6,492 and stop1,146 replace
6,466/1,172, but batch128 still gives51 source batches/epoch and at most2,550
updates over50 epochs. Target40/160 update ceilings, epoch-zero eligibility,
E stopping/AP ties, C calibration/fallbacks and parameter-matched random policy
remain unchanged. Equal steps are not equal compute; actual selector overhead,
updates, runtime and parameter changes must be reported after authorization.
No extra source seeds/XResNet/external20/172 fits or favorable retries are added.

## Review and activation requirements

1. Human review/adoption of the two-label annotation endpoint, canonical-record
   policy/serial differences, fixed age domains and this one candidate.
2. Approve the joint-stratum algorithm/seed, O replacement/audit history and
   explicitly retained B abstention. Review the existing Phase2B schedule/fallback
   clarifications; Phase2C has not made those drafts adopted.
3. Bind the reviewed version into a **new** real execution namespace/ledger with
   a matching role-specific access adapter; test/rehearse it synthetically.
   Current metadata context denies every predictive operation; the original
   Phase2B runner deliberately rejects this contract and must not be bypassed.
4. Separately authorize fresh source training and the bounded development/calibration
   matrix only after those checks. No historical initialization or preprocessing.
5. Separately authorize one exploratory O evaluation after complete development
   locking. External clinical transfer remains independently blocked.
