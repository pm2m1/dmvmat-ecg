# Internal exploratory selective-transfer pilot

**DRAFT — SEPARATE HUMAN APPROVAL REQUIRED. NO MEMBERSHIPS CREATED OR FITS RUN.**

## 1. Preserve the question and evidence level

Question: does evidence-guided seven-module adaptation outperform conventional
full fine-tuning with **the same total target-label access and update budget**?
This is a selective-transfer pilot, not architecture search or a new attempt to
raise PTB-XL AUROC. A negative/abstaining selector is an admissible outcome.

Evidence levels must remain separate:

1. **Synthetic engineering verification:** fictional labels/features validate
   code, not transfer performance. Completed in Phase 2A.
2. **Internal exploratory selective-transfer evaluation:** proposed here, within
   newly separated PTB-XL patient groups. Requires new source training and new
   target role locks. Not executed or authorized.
3. **Independent external clinical validation:** still blocked by external label,
   linkage and overlap evidence. An internal result cannot clear that gate.

The original fold-10 evaluation and existing benchmark remain closed. Folds 1–9
have already informed previous research, so even a new patient-disjoint holdout
there is not fresh independent confirmation of a new scientific hypothesis.

## 2. Proposed population, endpoint and domains

Use **only eligible official PTB-XL 1.0.3 folds 1–9**, canonical 12-lead, 100 Hz,
10-second records. Do not open fold-10 signals, labels, predictions or metrics.
Preparation first audits patient IDs and known exact/near-duplicate groups under
separate authorization. Link all records from one patient/export component before
splitting. Unresolved identity conflicts are excluded with counts, not guessed.
Take one canonical recording per patient: lowest ECG ID among otherwise eligible
records, independently of its labels or model scores. Other records remain
excluded, never migrate to another role. Record selection/exclusion hashes.

Proposed domain covariate: recorded age, **18–59 years source; 60–89 years target**.
The threshold is fixed now, not selected by outcome or adaptation results.
Exclude missing, invalid and age-deidentification sentinel values; do not treat
an obfuscated value as a true age. The domain assignment is based on the canonical
record only. Flag duplicate components with inconsistent patient/domain identity
for resolution/exclusion. No hospitals, countries or sites are inferred from IDs.
This is age-subgroup shift in one dataset, not independent cohort transfer or
evidence that age itself causes representation differences. Age is not an
additional network input; the network continues to use ECG alone.

Proposed **two-label internal annotation task**, ordered `(CD, STTC)`, using the
same official diagnostic superclass mapping convention as the completed
benchmark, including its any-key/likelihood-zero convention. These labels are
non-exclusive conduction and repolarization annotation groups, chosen for distinct
clinical feature families and a feasible co-occurrence analysis, **not observed
transfer performance**. It is not the external IAVB/IRBBB phenotype task. Label
absence means absence of the corresponding superclass annotation in this
benchmark, not adjudicated clinical absence. This consistency within one release
permits a clearly named annotation-classification pilot; it does not manufacture
Georgia clinical negatives or harmonize external endpoints. Report uncertainty
in the original annotations and whether this endpoint choice is approved.

The two-label task allows both A and the explicitly two-label B formulation to
be tested without inventing 16 co-occurrence strata for five outputs. Do not
switch labels after viewing model performance. If label/support feasibility
fails, stop for prospective amendment rather than choose an easier endpoint.

## 3. Newly trained source model — mandatory

**No historical PTB-XL checkpoint, prediction head, fitted normalization or other
learned transform may initialize this pilot.** Previous source training covered
many folds 1–8 participants and stopping used fold 9; such weights cannot be
independent source models for newly assigned target-evaluation participants.
Renaming run folders, resetting a head or subtracting patient lists afterwards
does not remove training history.

Create a new random five-view model, two-logit head, expected 212,066 parameters;
all source pretraining confined to the new source-training patients. Fit new
scalar normalization there only; retain it for every pretrained target method.
Fit no source statistics on source stopping or any target role. Source-only,
head/full/random/selective all start from the **same** source checkpoint.
One source seed **3101** for this bounded pilot; a later multi-source-seed study
requires another protocol. Checkpoint provenance must include complete training
and selection patient-group hashes, endpoint/order and preprocessing state.

Source age-domain split: 85% training / 15% stopping, patient-disjoint grouped
stratification, seed **33002**. Proposed scratch schedule: original five-view
architecture, no augmentation, batch128, AdamW lr0.01/wd0.01/betas(0.9,0.99),
OneCycleLR max_lr0.01, pct_start0.3, div_factor25, final_div_factor10,000,
base_momentum0.85/max_momentum0.95, cosine annealing, cycle_momentum=True,
three_phase=False, Adam epsilon1e-8, maximum50 epochs. Verify these constructor
defaults against the pinned PyTorch signature synthetically before execution;
copy them explicitly into a new immutable config after review. Source checkpoint selection uses exact
two-label macro AP on source stopping (tie lower BCE, then earlier); scratch
epoch zero diagnostic only. Do not call this a reproduction of the old five-class
AUROC-stopping experiment. Require ≥200 positive/negative source-training patients
per label and ≥20 each in source stopping before source fitting. These are
feasibility guards, not power guarantees. No source hyperparameter search.

## 4. New target roles and budgets

Proposed target group split, seed **33001**: O evaluation30%, E reserve10%,
C reserve10%, A fitting/evidence reservoir50%. Grouping precedes assignment.
Use the previous draft's deterministic greedy two-label count/size stratification
algorithm with this new role list; freeze a synthetic algorithm fixture before
real preparation. Actual integer counts/prevalences are reported. Do not search
split seeds using performance. Stratification/count auditing must be isolated
from predictive workers; its label access is recorded separately.

Choose fixed E=128 and C=128 patients from their reserves by label-independent
hash order, seeds33003/33005. Require E≥10 positives/negatives per label; O≥50
of each and a prospective precision review before fitting. Unused reserve
patients remain unused. C calibration requires ≥20 positives/negatives per label
or logged identity mapping/default threshold. No patient/group crosses source
training, source stopping, target F/G/E/C/O. A must contain ≥768 eligible patients.
No counts are asserted here: no real metadata membership preparation was run.

Two target draw seeds **4101/4102**. In each A permutation, first256 form G;
next512 form the F reservoir. F=128/512 are nested prefixes, disjoint from G.
The same F/G/E/C memberships and stochastic seeds are used for all paired methods.
All counts are **patients/records**, not guaranteed positive quotas:

| Tier | F | G | E | C | Selective / full-cost target-label union |
|---|---:|---:|---:|---:|---:|
| small |128|256|128|128|640|
| **primary larger** |512|256|128|128|1,024|

Report CD/STTC positives/negatives and joint positives per role/draw, plus the
union of all distinct accessed record-label cells. Natural prevalence is retained;
do not use a positive-search oracle without charging all screened labels. A/B
geometry needs ten/class or joint cell and otherwise abstains to SHARE; a rare
co-occurrence does not authorize changing the validity floor. No records are
relabeled negative for numerical convenience.

## 5. Prespecified paired comparisons and actual compute

Seven selectable owners remain five clinical-view encoders plus fusion1/2;
head always trainable except source-only. No gating/attention/encoder search.

| Method | Gradient labels | Evidence labels | Additional target labels |
|---|---|---|---|
| Source-only |none|none|none raw; calibrated variant C explicitly charged |
| Head-F |F|none|E/C |
| Head-cost |F∪G|none|E/C |
| Full-F |F|none|E/C |
| **Full-cost** |**F∪G**|none|E/C |
| Parameter-matched random-A |F|G-derived mask budget|E/C |
| Selective A |F|G|E/C |
| Selective B |F|G|E/C |
| Target-only |F∪G, fresh random model|none|E/C; statistics fit on F∪G only |

Head-F/Full-F consume F+E+C =384/768; cost-matched methods/selectors/random
consume640/1,024. Primary contrast is **A−Full-cost**, not the supervision-unequal
A−Full-F. Report source pretraining/selection labels separately and shared target
audit/stratification annotation access separately. No selector-specific meta-label
or policy search is planned; if introduced later, baseline label access must
increase equally under a prospective amendment. Target-only is a bounded scratch
reference and may underfit; do not call its short-schedule score its optimum.

Adaptation: AdamW0.0003/wd0.01/betas(0.9,0.99), batch64, no augmentation,
maximum `20*ceil(nF/64)` updates (40/160), E looks each `ceil(nF/64)` updates,
patience5. Larger gradient pools receive the **same update budget**. Training
seed5101 for draw4101 /5102 for draw4102, shared across methods. Freeze source
normalization; freeze SHARE weights and BN buffers, enforce eval on each train
call. ADAPT BN uses fitting data only. Head always updates for pretrained
adaptation. E exact macro AP selects checkpoints, tie lower BCE then earliest;
**epoch zero eligible**. Target-only scratch epoch zero is diagnostic only.
Record effective updates, retained epoch, examples, drift/buffer changes, number
of updateable parameters, evidence CPU cost and wall time. Same updates do not
mean same FLOPs: report measured costs and flag any selector overhead.

Random control enumerates128 masks and samples the nearest parameter-cost set,
including possible equality with A. Log degeneracy/equality probability; all-SHARE
does not permit an unfairly larger random mask. MMD is selection evidence only;
masked BCE is the training loss, with no alignment loss added.

## 6. Finite proposed compute and stop points

**33 planned predictor fits:** one new source pretraining + eight trained methods
×two draws ×two tiers. Source-only has no additional fit. **Hard proposed ceiling36**,
including three shared technical failure/retry attempts, never favorable reruns.
Identical fully hashed logical/random jobs may alias immutable artifacts and
reduce actual fits; never use freed slots for new policies. Calibrator fits and
evidence extraction are logged separately. Calibration of32 adaptation/scratch
artifacts plus the one shared source-only checkpoint gives up to33 unique
two-label mappings (66 individual Platt label mappings), before fallback/identical-
artifact deduplication. Record actual deduplication rather than inventing more
predictor fits.

The single-source-seed pilot limits scientific inference. A strong XResNet
transfer extension would require its **own newly trained source model on the
same source-only patients**; old XResNet weights are equally prohibited. It and
extra source seeds are outside36 attempts, subject to separate review. The
external20/172-fit draft budgets do not authorize this distinct internal pilot.

Measured prior full five-view training was about45 optimization minutes plus
validation for50 epochs/17,084 records; this source cohort is different and
unmeasured. It cannot supply an exact new runtime. Default CPU, at most two
four-thread fit workers initially; no unverified MPS numerical equivalence.
Use a separate private Phase2-internal ledger/cache/artifact tree, atomic
reservation-before-fit, fresh initialization on interrupted retries and a new
code/config/role/checkpoint lock. Never reset the completed study or reuse its
remaining fit slots. Clinical external blockers do not authorize any internal fit.

## 7. Locked analysis specified before fitting

Primary estimand: raw continuous-score macro AP(A)−macro AP(Full-cost) at F512,
arithmetic mean of the two paired target-draw metrics. Not pooled predictions,
not best draw, not the budget with the most favorable result. Source checkpoint
is shared, so two draws are not two independent source models/cohorts.

One O authorization after all configurations, masks, E checkpoints, C mappings,
thresholds, source lineage and evaluation code are locked. Seal all registered
O predictions before inspecting aggregate metrics. Use a new generalized
two-label patient-cluster paired bootstrap,10,000 draws, seed **64001**; shared
multiplicities across methods, labels and draws. Report ordinary95% percentile
CI, undefined-label draws, and withhold conclusive CI interpretation if >1%
are undefined. Intervals condition on this split, source and fitted models;
they omit source retraining/policy selection uncertainty and external validity.

Five secondary AP contrasts: A−Head-cost, A−Random, A−Full-F, B−A,
A−raw Source-only. Report95% and Bonferroni99% intervals; small tier, per-label
metrics, target-only and all source-only-calibrated variants are exploratory.
AUROC, prevalence, Brier/log loss/ECE10 equal-width bins, calibration slope/
intercept, precision/recall/specificity/F1/balanced accuracy and confusion counts
are secondary. Use only C for Platt mappings/operating thresholds as defined in
the external draft, with support/convergence fallbacks; never fit them on O.
Strict monotonic calibration preserves per-label ranking except implementation
ties/clipping; changing thresholds cannot improve AP/AUROC.

A positive point estimate alone is not improvement evidence. Proposed +0.005 AP
is an engineering relevance signal, not a clinical margin; lower95% bound>0
is conditional positive evidence and >0.005 is stronger practical support.
Report both labels, both draws, negative-transfer frequency against source-only,
epoch-zero retention, mask stability, invalid geometry and overhead. A benefit
only over Full-F does not show matched-budget superiority. Neither mask diversity
nor a smaller confidence bound itself establishes prediction benefit.

## 8. Required approval sequence

Approve endpoint/domain/eligibility and36-attempt matrix first. Then separately
authorize **metadata-only preparation** of folds1–9 and patient/duplicate roles,
with no fits; review counts, source-history exclusions, new normalization lineage
and statistical feasibility. Complete real role-bound adapters, durable attempt
ledger and the separate lock with synthetic integration tests. Only after that
authorize new source pretraining and bounded adaptation. Locked internal O
access requires its own explicit authorization. No step was executed here.
