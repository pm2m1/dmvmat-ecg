# Multi-label evidence-guided module selection

**DRAFT — proposed new method; not implemented, registered or experimentally validated.**

**Phase 2A status update:** the preceding design status is historical. A separate
`src/dmvmat_ecg/phase2/` implementation now exercises A/B and the seven-module
controls on synthetic fixtures only; see
[implementation report](SELECTOR_IMPLEMENTATION_REPORT.md). No real transfer
experiment is registered or validated, and the legacy prototype remains unchanged.

## 1. Method provenance: three distinct implementations

The original paper is Cui et al., *Deep Multiview Learning With Module-Adaptive
Transfer for Cross-Subject/Session Brain Decoding*, DOI
[10.1109/TNNLS.2024.3350085](https://pubmed.ncbi.nlm.nih.gov/38252578/), online
2024, journal volume 36(2), 2025. Its application is brain decoding; “MI” in
the authors' repository denotes motor imagery, not myocardial infarction. The
abstract and authors' released code were available; the full paper's mathematical
details were not independently verified from full text in this audit.

The [authors' repository](https://github.com/YangLibuaa/DMV-MAT/tree/4693df334bf34cc9037ea38390efb31d539a5584)
was inspected at `4693df334bf34cc9037ea38390efb31d539a5584`:

- `github_model/MI/adpation.py`: `JS_divergence`, class-distance calculations,
  shallow/deep representations and ratio-based decisions. This is **Jensen–Shannon
  divergence**, not our multikernel MMD/permutation pipeline.
- `github_model/MI/d_s_train.py`: branch-specific freezing of temporal/spatial
  convolutions, BatchNorm and deep modules, not our seven ECG clinical-view modules.
- Other released scripts use five EEG frequency bands and GMM-related scores.
  Dependencies/network definitions are incomplete in the inspected repository;
  CUDA-specific tensor construction is present. No reuse licence was found in
  its tree. No authors' source was copied or run.

The public, re-engineered Phase 1 release at
`~/Desktop/cvd-selective-transfer-github-release`, commit
`79bd96c078e517824638a893919153f497978b44`, is a separate scientific adaptation:

| Component | Exact public evidence | Verified behavior |
|---|---|---|
| Modules | `src/dmv_mat_cvd/models/multiview_mlp.py`:23–60 | Five tabular clinical encoders, fusion1/2; binary head always trainable |
| Legacy score | `src/dmv_mat_cvd/transfer/scores.py`:67–113; `selector.py`:17–46 | Biased Gaussian multikernel MMD²; normalized MMD/class-distance top-k score |
| Calibrated MMD | `transfer/calibration.py`:43–79 | Pooled RMS scalar scaling; four bandwidths; 199 permutations; cap 128/domain; explicit degenerate null |
| Geometry | `transfer/calibration.py`:82–138 | Source-direction retention R and conditional class-mean bootstrap upper bound U; min five/class, source D ≥0.1 |
| Logical rule | `transfer/calibration.py`:156–179 | ADAPT if U<0.75 OR (p≤0.05 AND R<0.8 AND U<1), valid geometry required |
| Random masks | `transfer/selector.py`:54–107 | Enumerate 128 masks, match trainable parameter cost, including possible equality with reference |
| Freezing | `transfer/adapt.py`:13–61 | Clone, set requires_grad, keep head trainable, snapshot frozen parameters |

These paths are relative to the public Phase 1 release, **not** this ECG repository.
The [Phase 1 method document](https://github.com/pm2m1/cvd-selective-transfer-research/blob/79bd96c078e517824638a893919153f497978b44/docs/PHASE1_METHOD_AND_REPRODUCIBILITY.md)
describes 10 paired target draws (25 fitting positives/75 negatives), the shared
283-person/21-positive evaluation and frequent logical head-only masks. It does
not establish selective superiority. The public generic `training.py`:15–65
does not score epoch zero; it cannot by itself substantiate the private historical
executor's epoch-zero handling. The later Kaggle checkout at
`928b3994eb77bf0c217dcba3fda8a70eabe44db3`, branch
`research/v3-adaptive-multiview`, `src/cvd70k/v2/transfer.py`:57–109
explicitly implements zero-eligible adaptation; its full-cost control and label
accounting appear around lines 311 onward and 454–459. Historical private
participant data/checkpoints were not accessed.

The current ECG [interface](../../src/dmvmat_ecg/transfer/interface.py):13–82
is only a prototype: pooled-median bandwidth MMD, binary per-label D, logical
predicate and freeze utilities. It does **not** supply source-relative R/U,
multi-label observation masks, patient-aware permutations or a real adaptation
executor. It is not evidence that a full ECG transfer algorithm already exists.
Our proposed policy below is another explicitly new variant, not an exact
reproduction of either the original paper or Phase 1.

## 2. Representation and label contract

Use the five-view source encoder and seven owners documented in the
[audit](EXTERNAL_ECG_DATASET_AUDIT.md). Each source/target evidence record carries
release, record, true-patient/duplicate-group, role and preprocessing hashes.
Extract one canonical eligible ECG per patient for selector evidence, using a
label-independent, frozen record ordering. Extraction is `eval()/no_grad()`,
detached and read-only. Each of the five views yields 64 dimensions, fusion1
128, fusion2 64. Cache keys include checkpoint and code hashes. No outer target
embedding or outcome may enter selection. All seven representations are computed
from the **same unchanged source initialization**, before adaptation.

Proposed task L=2: IAVB and IRBBB, conditional on the
[clinical crosswalk gate](DIAGNOSTIC_LABEL_CROSSWALK.md). Store outcomes Y and an
observation mask M separately. Unknown is not negative. Loss is the mean over
observed record-label entries; batches without observed entries must fail/skip
under an explicit logged rule, never contribute fake zero labels. Metrics and
geometry use only the relevant observed entries. Whole patient vectors,
co-occurrence and observation masks travel together during resampling.

## 3. Label-blind module shift: MMD evidence

MMD is **selection evidence**, not an added neural alignment loss. The proposed
training objective remains masked BCE. MMD detects distributional discrepancy;
it does not show that changing a module will improve target prediction.

For each module, pool source-anchor and selector-G representations; subtract the
pooled mean and divide by the single pooled RMS Euclidean radius. Degenerate
radius ≤1e-8/nonfinite data means invalid evidence. This is scalar normalization,
not whitening. Cap at 256 distinct patients/domain using fixed hash sampling,
independent of labels. Use the biased nonnegative V-statistic

`MMD² = mean(Kss) + mean(Ktt) − 2 mean(Kst)`,

including diagonal terms, and
`K(x,z) = mean_sigma exp(−||x−z||²/(2 sigma²))`,
`sigma ∈ {0.5,1,2,4}`. Reuse the same pooled normalization and Gram matrix for
999 domain-label permutations. `p=(1+number(null≥observed))/(1000)`.
An all-identical/nonfinite null is explicitly invalid, not evidence of adaptation.
Apply Holm adjustment to the fixed family of seven module p-values, treating
invalid tests as p=1 for adjustment but retaining their invalid status; use
valid q≤0.05 for corroboration.
Seed 24007 plus module index; record cap identities, raw statistic, null summary,
p/q and degeneracy. One record/patient makes patient the permutation unit;
multiple recordings cannot be independently permuted. Ordinary exchangeability
assumptions remain necessary. This is a new cap/draw-count/multiplicity convention.

Reference: [Gretton et al., 2012](https://www.gatsby.ucl.ac.uk/~gretton/papers/GreBorRasSchetal12.pdf).
Source/target prevalence and co-occurrence can cause marginal MMD even with useful
invariant conditional representations. Report label prevalence/observation
coverage alongside it. Class-conditional, equal-mixture MMD may be a later
diagnostic with a separately defined within-class permutation null; it is not
an extra vote or covert loss in this finite proposed study.

## 4. Formulation A: per-label source-direction retention (primary proposal)

For module m, label l and domain d, using observed positives/negatives:

`delta_d,l = mean(z | Y_l=1,M_l=1) − mean(z | Y_l=0,M_l=1)`.

Let `s_d,l = 0.5*(mean_positive ||z−mu_positive|| +
mean_negative ||z−mu_negative||)`: equal-class-weighted mean Euclidean scatter,
not prevalence-weighted pooling. Define `D_d,l=||delta_d,l||/s_d,l`,
`v_s,l=delta_s,l/||delta_s,l||`, and

`R_m,l = (v_s,l · delta_t,l / s_t,l) / D_s,l
       = cos(theta_m,l) D_t,l / D_s,l`.

Negative retention remains negative; do not take its absolute value. Require
≥10 observed positive and ≥10 observed negative **patients** in each domain,
finite scatter >1e-8 and source D≥0.1. Otherwise this label abstains. Source
anchors use at most 256 positives and 256 negatives per label, sampled from source
training only; retain their union with full label vectors and sampling identities.
Target evidence uses all G (up to 512 in the proposed full experiment).

Use 4,999 paired patient bootstrap draws within each domain, shared across labels
and modules. For the class-mean uncertainty calculation keep the original source
direction and both original scatters fixed, following the conditional spirit of
the Phase 1 implementation. Recompute source/target mean contrasts per bootstrap;
divide projected target contrast by projected source contrast, with the fixed
scatter ratio. If the source projected contrast divided by its fixed source
scatter is ≤0.02, or a required class is
absent in >5% of draws, abstain. This does not estimate uncertainty in the whole
trained representation or in all nuisance quantities.

Use upper percentile `1−0.05/(7L)` (L fixed at two, thus 0.99642857), denoted U_adj,
instead of the historical U95. Shared resampling preserves co-occurrence;
independent per-label class-stratified bootstraps would not. Report finite-draw
tail instability. Bonferroni percentile bounds are an approximate conditional
uncertainty safeguard, not a proved finite-sample simultaneous guarantee.
Seed 24011, deterministically separated from permutation and evaluation RNGs.

For each valid label:

`vote_ADAPT = U_adj < 0.75 OR
              (q_MMD <= 0.05 AND R < 0.8 AND U_adj < 1.0)`.

Module ADAPT if **any valid label** votes; otherwise SHARE. All-invalid is SHARE
with an explicit abstention flag, not positive invariance evidence. A valid label
may vote despite another abstaining; record this partial coverage and possible
harm to the unsupported label. The thresholds borrow the historical rule, while
the multiplicity/aggregation/validity changes make this a distinct policy.
No thresholds are optimized against target outer outcomes.

Assumptions/failures: marginal one-vs-rest classes mix other diagnoses; source
class imbalance, incomplete labels, verification bias and changes in co-occurrence
can change R without loss of useful features. An “any-label” vote can favor
well-supported labels. Wide bounds can select only the head. None is proof of
adaptation benefit. Log D/R/U/q, valid counts, observation coverage and all seven
votes for every episode, including abstentions.

## 5. Formulation B: co-occurrence-standardized geometry (secondary proposal)

For two labels, condition label l on the other observed label c=0/1. Require
≥10 distinct observed patients in **each of the four joint cells**, in both
domains. Compute within-stratum positive-minus-negative contrasts delta_d,l,c.
Use common prespecified weights w0=w1=0.5:

`delta_std_d,l = 0.5 delta_d,l,0 + 0.5 delta_d,l,1`.

Use the corresponding equally weighted within-cell mean Euclidean scatter
(four cells weight 0.25 each), source direction, D and signed R as above.
Bootstrap whole patients with joint masks; require support/denominator validity
and use the same U_adj/decision rule. Empty/rare cells produce abstention; no
invented cell means, independence assumptions or post-hoc stratum dropping.

This separates measured other-label composition from the standardized contrast,
but assumes overlapping observed joint phenotypes. It does not identify a causal
effect or correct unrecorded diagnoses, demographic confounding or differential
verification. With rare IAVB/IRBBB co-occurrence it may almost always abstain.
For five labels there are 16 strata per label; that is not feasible by default
and is not the proposed experiment. Formulation B is a controlled secondary
variant, not a route to choosing whichever policy wins evaluation.

## 6. Masks, freezing and controls

Keep owners `view_1`…`view_5`, `fusion_1`, `fusion_2`; head always trainable.
Every `train()` call must reapply eval mode to SHARE BatchNorm/dropout modules;
snapshot and verify frozen parameters **and buffers** after fitting. ADAPT
modules may update their BatchNorm state. Record effective changed parameter
norms and buffer changes, not only requires_grad flags. Source models are cloned,
never modified. Epoch zero is an eligible target checkpoint. The adaptation
engine must be new versioned code; do not edit or run the completed trainer.

For random controls enumerate all 128 internal masks; sample uniformly among
those with the closest total updateable parameter count to policy A, including
the identical mask if necessary. Head cost is common. Log mismatch, support set
and equality probability. An all-SHARE selection may have no distinct exact-cost
random control; report this degeneracy, do not manufacture a different-size mask.
Identical fully hashed jobs can reuse an immutable artifact; distinguish logical
comparison slots from actual fits. Score-shuffled/learned mask variants are not
included in this finite design: a few episodes from one hospital do not supply
independent meta-training tasks. Selection stability is descriptive, not efficacy.

## 7. Required future synthetic verification (not run in this audit)

Verify masked loss/metrics, known signed-R rotations/scalings, constant MMD and
permutation behavior, patient bootstrap co-occurrence, rare-cell abstention,
all seven owner partitions and parameter counts, frozen BatchNorm buffers,
epoch-zero selection, no source mutation, random-mask equality/degeneracy,
role access and label accounting, RNG isolation, serialization and single-use
outer locking. Source-label and cohort gates must pass separately; synthetic
correctness cannot establish clinical validity or transfer effectiveness.
