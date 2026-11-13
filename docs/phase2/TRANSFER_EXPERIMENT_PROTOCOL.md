# Proposed external-target transfer experiment

**DRAFT — REQUIRES REVIEW, ENDPOINT/COHORT EVIDENCE AND SEPARATE EXECUTION AUTHORIZATION.**

No part of this protocol is registered, frozen, implemented or executed by this
design sprint. Numerical budgets are proposed ceilings, not permission to fit.
The completed PTB-XL study remains closed and unchanged.

## 1. Question, population and gates

Primary question: does evidence-guided seven-module adaptation improve target
**macro AP** versus full fine-tuning with the same total accessible target-label
budget and gradient-update budget? Target: Georgia's public Challenge component,
subject to the gates below; Ningbo is a second choice requiring a prospective
replacement protocol, not a performance-based fallback. Never use the whole
Challenge as an independent target. Sources and limitations are in the
[dataset audit](EXTERNAL_ECG_DATASET_AUDIT.md) and
[crosswalk](DIAGNOSTIC_LABEL_CROSSWALK.md).

The proposed task is **two simultaneous ECG phenotypes, IAVB and IRBBB**, not the
old five superclasses, general heart disease, clinical outcome prediction or
future risk. Both positive meanings and reliable negatives must be signed off
by ECG experts/curators. A missing code is unknown. Source likelihood-zero
handling, target code conversions and conflicting diagnoses must be resolved
before generating task labels. Primary target eligibility requires a reliable
observed status for both labels; an audit mask is retained for exclusions and
possible separately reported partial-label analyses, never silently zero-filled.
Do not select only easily diagnosed cases: use complete curator review or a
label-independent sampling/adjudication frame, documenting verification bias.

Before any predictive work, require:

1. Release-specific stable patient linkage or documented one-ECG-per-person
   bijection; grouping of all repeated exports/segments and known duplicates.
2. Cross-release physical-signal/near-duplicate audit against PTB-XL, including
   historical conversions. Exclude all Challenge PTB-XL/PTB and uncertain linked
   components. Different file IDs or hospitals alone are insufficient proof.
3. Clinical crosswalk/negative coverage, eligible counts and licensing verified;
   no unresolved native scaling/lead semantics. No large download yet authorized.
4. Three independent roles can support the task: source development,
   target development and a sealed target evaluation. Cohort auditor can use
   labels for stratification/counts, not predictive development on the test.
5. The planned main pool supports ≥2,560 eligible distinct patient groups for
   disjoint maximum F/G sets, E/C reserves of ≥512 each, and outer evaluation
   ≥100 positives and ≥100 negatives **per label**. These are feasibility floors,
   not proof of sufficient statistical power. Review achievable CI precision
   prospectively after counts, before fitting. If counts fail, stop for amendment;
   do not drop an inconvenient head or shrink test after seeing predictions.

Current status: **BLOCKED by gates 1–3 and unverified eligible support**. No
claim that the current public code counts satisfy these gates.

## 2. Source initialization and task conversion

Source backbone training history: eligible original PTB-XL folds 1–8, 17,084
records, using its saved train-only normalization. New source-task development
may use those records and fold 9 (2,146) for the **new task**, with provenance
that this is a reused research population. Fold 10 is unavailable for all new
training, stopping, calibration and selection. An integrity-only duplicate auditor
may compare its waveform fingerprints without running model/prediction code.

Use both existing **unaugmented** five-view source best checkpoints, seeds
101/202, by designation, not by fold-10 score. Use both existing XResNet best
checkpoints as the strong reference. Exact IDs/hashes are in the
[audit](EXTERNAL_ECG_DATASET_AUDIT.md). The old five-output heads cannot directly
predict the new task. Proposed four source-task conversion fits:

- Clone each of two five-view and two XResNet checkpoints into a new Phase 2
  namespace; replace the head with two logits, initialize it under its seed.
- Update the complete cloned model on source folds 1–8 only with the approved
  new labels/masks; source fold
  9 selects exact macro AP, tie by lower masked BCE then earlier epoch.
- AdamW lr 0.0003, weight decay 0.01, betas (0.9,0.99), batch 64, constant LR,
  at most 20 epochs, patience 5. No augmentation; retain each family's original
  input/crop and inference semantics. Random new source head means epoch zero
  diagnostic only, not checkpoint-eligible for this conversion stage.
- Do not continue an old optimizer trajectory or overwrite source weights.
  Expected counts: five-view 212,066; XResNet 1,871,874, to verify synthetically.

This is new source-supervised task adaptation, not an exact reproduction of the
old five-class training. Source-only on the new task means these converted
checkpoints, not reused broad-CD logits. If source labels cannot support the two
heads, the proposed study stops before external prediction.

## 3. Signals and preprocessing

Retain canonical 12 leads, physical mV and the 100 Hz/full-1,000-point five-view
input. First study includes genuine 10-second recordings only: exact 5,000
samples at 500 Hz (or release-verified equivalent), all leads present, finite
header-calibrated samples. Georgia's shorter recordings are excluded with counts;
no invented beats, repeat padding or undisclosed length-model changes. This may
reduce support below the feasibility gate. A broader variable-length model would
be another protocol. Resolve native sampling/ADC/header anomalies before use.

Proposed new target resampling: CPU SciPy `resample_poly(up=1,down=5,
window=('kaiser',5.0),padtype='constant',cval=0)` on calibrated 500 Hz signals;
verify anti-aliasing, length, physical units and edge behavior synthetically.
Pin the tested version and a fixture digest before execution. See
[official API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html).
Do not resample the already official 100 Hz source twice. Recompute the unchanged
rhythm/spectral views deterministically from the normalized signal. No target
augmentation in the initial transfer comparison.

All pretrained methods retain the original source scalar normalization. No
target/test normalization fitting. Target-only fits its scalar statistics on
its own F∪G only and documents that legitimate from-scratch pipeline difference.
Hashes include statistics, units, lead order, resampling, label map and grouping.
Quality exclusions are label-independent and reported before model access.

## 4. Target partitions and protected roles

Proposed one fixed external-target split, patient/duplicate-component disjoint:

| Role | Fraction of eligible target groups | Permitted use |
|---|---:|---|
| O |30%| One final external-target evaluation after all locks |
| P |10%| Optional pilot/meta-development only; never O |
| A |45%| Reservoir for main fitting F and selector G draws |
| E-reserve |7.5%| Stopping labels; first 512 eligible groups form E |
| C-reserve |7.5%| Calibration/threshold labels; first 512 form C |

Resolve duplicate components spanning conflicting verified person IDs before
eligibility; exclude unresolved components with counts rather than treating
different people as one known patient. Analysis/sampling units must consequently
be verified patient units with their linked exports/segments, not arbitrary
filename components. If that cannot be established, the independence gate fails.

One canonical ECG per patient for this first experiment: earliest documented
acquisition, ties by release/record identity; if no time, stable record-ID order,
never diagnosis or model confidence. All duplicates/patient records remain in
the same split even when only one is used. Do not call a filename a patient ID.

Proposed deterministic grouping split seed **23001**. Sort components by rarest
positive-label pattern first, then descending group size, ties by a seeded hash.
Greedily assign each component to the role minimizing the total squared
normalized deviation from target role totals of groups, observed-positive counts
for each label, and observed-negative counts for each label (each of these five
terms equal weight; denominator `max(1,desired_count)`). Among equal minima use
role order O/P/A/E/C. Label-count targets use the eligible population prevalence;
they do not rebalance O. Final integer partitions and any prevalence deviations
are reported. Freeze implementation fixtures and deterministic hashes before
any fit; no split search using prediction performance.

Select fixed E/C subsets by label-independent seeded group permutation
(23003/23005). Unused reserved groups remain unused. Require E ≥10 positives
and ≥10 negatives per label. Calibration support checks below allow logged
identity fallbacks. O is accessible only to an isolated auditor for eligibility,
stratification and aggregate feasibility counts before lock; record these
accesses separately from predictive test access. Model workers cannot open it.

## 5. Fitting-positive support and complete label accounting

Primary design increases support at **natural target prevalence**, rather than
claiming a fixed number of positives under unknown eligible counts:

| Tier | F records/patients | G records/patients | E | C | Selective/full-cost distinct target-label access |
|---|---:|---:|---:|---:|---:|
| small |128|128|512|512|1,280|
| **primary middle** |512|256|512|512|1,792|
| large |2,048|512|512|512|3,584|

Proposed paired draw seeds **2101/2102/2103**; source seeds **101/202**. For each
draw seed, independently permute A groups; first 512 form the G reservoir,
next 2,048 the F reservoir; take prefixes for each tier. F/G never overlap at
any tier. Reuse exactly the same draws, E and C for all paired methods and both
source seeds. Draws may overlap one another; they are not independent cohorts.

Report **actual observed positive/negative counts per label**, joint positives,
unknowns and every role's group/record count. Published Georgia counts suggest
only illustrative expected F positives of approximately (IAVB,IRBBB)
(9.5,5.0), (38,20), (152,81), and G approximately (9.5,5.0), (19,10), (38,20).
These are arithmetic projections of unfiltered code prevalence, **not** achieved
budgets or reliable clinical counts. Low-support abstention is an intended result,
not a reason to oversample selector positives after evaluation.

An optional, separately approved **positive-support stress test** could sample
10/25/50 observed fitting positives per label, with selector positives
`max(20,ceil(b/2))`, fixed negative quota and explicit co-occurrence handling.
It would be outcome-conditioned sampling, not natural clinical prevalence or
pure causal identification of support. All labels screened to find those cases
must be counted. It is **not** part of the main fit budget. Never call the
small tier “128-label transfer” without its additional 1,152 evidence/stopping/
calibration labels. Evaluation adjudication and split-auditor labels are reported
as separate study-wide annotation costs, not hidden as free acquisition. This
is a retrospective label-access study: it does not measure prospective clinical
annotation cost merely because optimizer access to existing labels is limited.

## 6. Methods and matched supervision

Evidence formulations A/B and their new multiplicity rules are specified in
[the method](MULTILABEL_DMV_MAT_METHOD.md). No learned meta-selector or alignment
loss is included. All targets use masked multi-label BCE; no label weighting,
oversampling or loss search. Head is always updateable for adaptation.

| Method | Initialization | Gradient labels | Evidence labels | E/C | Purpose |
|---|---|---|---|---|---|
| Source-only MV / XResNet | Converted source | none |none|none raw; C only calibrated variant| Reference; calibrated source-only is not zero-target-label use |
| Target-only MV | Fresh random model | F∪G |none|E/C| Same target-label budget, no source weights/statistics |
| Head-F | Same converted MV | F |none|E/C| Fitting-label-matched head adaptation |
| Full-F | Same converted MV | F |none|E/C| Fitting-label-matched full adaptation |
| Random-A-F | Same converted MV | F |G to define matched cost|E/C| Uniform closest-parameter mask to A |
| Selective-A-F | Same converted MV | F |G|E/C| Primary new multi-label policy |
| **Full-cost** | Same converted MV | **F∪G** |none|E/C| **Primary total-label/gradient-budget comparator** |
| Head-cost | Same converted MV | F∪G |none|E/C| Total-label-matched simpler method |
| Selective-B-F | Same converted MV | F |G|E/C| Secondary co-occurrence-standardized policy |
| XResNet Full-cost | Converted XResNet | F∪G |none|E/C| Strong conventional transfer pipeline |

Head-F/Full-F access F+E+C (1,152/1,536/3,072), whereas selectors and all cost-matched
methods access F+G+E+C. Random-A also consumes G because its cost is derived
from the selector. Record role-set unions, not sums that double-count repeated
patients/labels. Source pretraining labels, pilot policy-development labels and
outer outcome adjudication costs are separate totals. Label access means available
supervision, not only optimizer inputs. Calibration fits are separately counted.

P must not silently train a new selector while baselines lack its labels. The
proposed A/B policies are fixed before the pilot, and P is primarily an engineering
check. If pilot performance is used to change any main policy/configuration,
record a prospective amendment and the distinct P-label union. Then the main
Full-cost gradient pool must also include those accessible P labels, with the
same update budget; its label total becomes F+G+P_used+E+C for both procedures.
If this cannot be supported, block a “same total label access” claim. No pilot
outcome can authorize using O or guarantee absence of research-selection bias.

## 7. Adaptation schedule, checkpoint and random-mask controls

All MV adaptation methods share the same converted source checkpoint per seed,
F/G draw, optimizer settings, batches and initialization RNG contract. Proposed
AdamW 0.0003, wd0.01, betas(0.9,0.99), batch64, no schedule or augmentation.
Maximum optimizer updates `20*ceil(n_F/64)` = 40/160/640, with E evaluation every
`ceil(n_F/64)` updates, at most 20 looks, patience five. Cost-matched methods
sample the larger accessible pool using the **same number of updates**, not an
extra epoch's work. Independent deterministic stream for sampler/dropout, same
key `(source_seed, draw_seed, tier)` across paired methods. Architecture-dependent
RNG consumption prevents a claim of identical stochastic trajectories.

Epoch zero on E is **eligible** for all pretrained adaptation. Maximize exact
scikit-learn macro AP on E; tie within 1e-12 by lower masked BCE then earlier
checkpoint. Target-only random epoch zero is diagnostic, not eligible. Its same
short update budget may underfit: this is a compute-matched reference, not its
best achievable performance. Save all selected epochs and whether adaptation
lost to epoch zero. XResNet uses its own original 250-sample crops/seven-window
maximum inference; the same update-count budget does not imply equal FLOPs.
Record measured runtime/exposure separately; a speed-budget sensitivity is future
work rather than an unregistered retuning exercise.

Freeze SHARE parameters **and** BatchNorm buffers, reassert eval after every
`train()` call, verify snapshots; ADAPT modules update weights/BN. Report changes
per module, head, buffers and source initialization integrity. All 128 masks are
enumerated for the parameter-matched random control; exact degeneracy/equality
is reported. Do not force a different mask at an unfair cost. No source checkpoint
may be overwritten. Interrupted fits consume attempts; restart fresh with a new
ID unless full optimizer/RNG continuation was independently verified in advance.

## 8. Calibration and operating points

Fit each label's positive-slope Platt mapping on C only:
`sigmoid(exp(a)*logit(clip(p,1e-6,1−1e-6))+b)`; bounds
`a∈[log(.01),log(100)]`, `b∈[−10,10]`, penalized mean BCE +
`1e-4*(a²+b²)`. Pin a tested deterministic CPU optimizer in future implementation.
Require ≥20 observed positives/negatives per label; otherwise identity fallback.
Convergence/nonfinite failures also yield a logged identity fallback, never O fit.

Report raw and calibrated probabilities. Strictly increasing per-label transforms
preserve per-label AP/AUROC and their macro means; clipping/finite precision can
introduce ties, so verify and report any changes rather than promise exact
invariance of the implemented mapping. Micro AP may change because cross-label
ordering changes. No recalibration is an AP-improvement claim.
Choose per-label C-F1 threshold over .05… .95 step .01; ties closest to .5 then
lower. Insufficient support defaults to .5. Report .5 and C-selected operating
points; thresholds are descriptive research settings, not clinical recommendations.
Nothing on C may select architecture, masks or training checkpoints.

## 9. Finite proposed fit budget

| Stage | Calculation | Planned predictor fits | Shared failure/compatibility reserve | Hard ceiling |
|---|---|---:|---:|---:|
| Small pilot | 2 source conversions (seed101 MV/X) + 8 trained methods ×2 draws ×1 small tier |18|2|**20**|
| Full study | 4 source conversions + 9 trained methods ×2 source seeds ×3 draws ×3 tiers |166|6|**172**|

Pilot omits formulation B; uses P only: fixed E_p=128 and S_p=128 distinct
patients, separate F_p=128/G_p=128 from the remaining P, two paired draws.
Require P≥768 groups to allow useful resampling. No pilot calibration, no O, no
patient shared between pilot roles. S_p scores are development-only, not final
external results. Pilot rare geometry may abstain. A pilot is optional and needs
its own approval; it is not a sneaky preliminary access to the main O.

Full study has 18 episodes and nine fitted controls/policies listed above,
plus source-only inference with zero additional predictor fits. Immutable fully
identical jobs may alias one completed artifact (e.g. a degenerate random mask),
with both logical slots and actual fit counts reported. Pilot/full together
have ceiling 192 if separately authorized; identical source-conversion artifacts
can reduce actual fits but never enlarge a search. No old project's attempt reserve
applies. Reserve is for predefined technical failures, not new candidates or
unfavorable scores. Calibration/evidence extraction/resampling are logged
separately; a hidden compatibility predictor fit still consumes the ceiling.

## 10. Lock, primary analysis and secondary family

New Phase 2 data access and attempt ledgers: atomic fit reservation before launch,
immutable failed/interrupted records; separate auditor/fit/E/G/C/O capabilities,
group/hash checks at the data boundary. Do not reset the existing `evaluated`
PTB-XL state or its 13-attempt policy. Save configs, dependencies/source revision,
source and target label manifests, grouping/role/preprocessing hashes, evidence,
mask, every checkpoint and calibrator/threshold. Private per-record predictions
and waveforms never enter public Git.

After all required fits and calibration: lock candidate definitions, policies,
all masks and checkpoints, C artifacts and evaluation code. One auditable O
lease for the entire registered set. Generate and seal **all** predictions before
inspecting aggregate outcomes; no tuning thereafter. Any access failure remains
recorded. Reporting corrections use the same sealed prediction artifacts.

**Primary estimand:** Selective-A-F minus Full-cost **raw macro AP**, at middle
tier (F512/G256), averaged over six paired episodes (two source seeds × three
draws). AP is scikit-learn non-interpolated AP from continuous scores; compute
each episode's two-label macro AP, then average episode metrics. Do not substitute
pooled/ensemble scores after seeing them. AP is primary because both proposed
labels are uncommon; report each label's actual O prevalence/random AP reference.
Do not compare raw AP across different target populations as if baselines matched.

Use 10,000 shared patient-cluster paired bootstrap draws (seed **24001**) of O;
same resampling weights for all models, both labels and every episode. Compute
the same within-episode metrics then paired mean contrast. Repeated ECGs, if
retained in a future amendment, resample their whole patient component; no
independent seed/prediction-row resampling. Undefined-label draws are counted
and omitted from that metric only; if >1% lack either class, flag the interval
unreliable and withhold conclusive inference. Ordinary two-sided percentile 95%
primary CI, with finite Monte Carlo limitations.

Six secondary AP comparisons at the middle tier: A−Head-cost; A−Random-A-F;
A−Full-F; A−raw Source-only MV; B−A; XResNet Full-cost−MV Full-cost. Report
ordinary 95% and Bonferroni 99.1667% two-sided intervals (six comparisons).
Other tiers/labels/metrics are exploratory stability descriptions, not substitute
primary outcomes. No clinical noninferiority margin is assumed.

Secondary metrics: macro/per-label AUROC and AP, micro AP (with observation-mask
interpretation), Brier/log loss, ECE in ten equal-width bins [0,.1)…[.9,1],
calibration intercept/slope where identifiable, precision/recall/specificity/F1/
balanced accuracy/confusion counts at fixed and C thresholds. Any outcome-based
slope/intercept fitting on O is descriptive calibration assessment **after** lock,
not a new prediction/calibration mapping. Include updateable counts, runtime,
inference time, drift, selector stability/Jaccard, valid/abstention rates,
epoch-zero frequency and negative transfer versus source-only by episode.

## 11. Success, limitations and rejection rules

Positive point estimate is not established superiority. A gain ≥0.005 macro AP is
proposed as a practical engineering signal, not clinical utility; lower primary
95% bound >0 supplies conditional evidence of improvement; lower bound >0.005
is stronger evidence exceeding that proposed magnitude. Review this margin
prospectively, never adjust to the observed result. Evidence-guided value also
needs comparison with Head-cost/random, label-cost transparency and consistent
label/draw behavior; a win only over Full-F may be explained by extra supervision.

Intervals condition on fixed fitted models, source checkpoints, policies and
split. They do not include retraining/model-selection uncertainty or certify
generalization to other hospitals. Seeds and overlapping target draws are not
independent patients. Report all six episode and fold-free target subgroup
results descriptively; optional hierarchical episode sensitivity must be specified
before O and cannot replace the primary patient-bootstrap analysis. This design
uses one target holdout, not invented three independent clinical folds.

Stop before fitting on identity/permission/label/shape/isolation failure, infeasible
support or reserve exhaustion. Stop before O if any fit, artifact/hash, label-cost
or lock check is incomplete. After O: preserve all results, no retraining,
recalibration, new target task or mask redesign. A negative or inconclusive
result is acceptable; no 0.93 AUROC requirement and no claim of deployment readiness.
