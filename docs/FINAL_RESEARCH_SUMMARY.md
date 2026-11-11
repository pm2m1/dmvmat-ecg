# Final research summary

## Executed work

New independent PTB-XL1.0.3 project; 21,799 raw recordings, 18,869 patient IDs. 21,388 eligible diagnostic recordings, official folds1–8/9/10, patient-disjoint and no detected exact waveform leakage. All43,598 required100Hz files match publisher checksums; no500Hz data downloaded.

Completed 8 full50-epoch fits plus three training-only smoke fits. Test access consumed once after the complete artifact lock; ten model/seed prediction sets sealed before metrics. All numerical findings: [experiment results](EXPERIMENT_RESULTS.md).

## Baseline and our architecture

Architecture-faithful modern PyTorch XResNet1D101 adaptation: final-epoch macro AUROC 0.92800, macro AP 0.82241. Difference from published approximately0.928 macro AUROC: +0.00000. This is contextual, not a paired study or proof of exact historical reproduction. Validation-best variant is separately reported.

Five-view no-augmentation model: macro AUROC 0.92139, macro AP 0.81405; 212,261 parameters versus206,117 for the flat CNN and1,872,261 for XResNet. Five signal-appropriate CNN encoders, 64-dimensional embeddings, fusion320→128→64, five-label head. Rhythm is a learned temporal view; spectrum is deterministic Hann FFT, not independent information.

multiview-minus-flat_cnn: Δmacro AUROC=+0.00406, primary97.5% Bonferroni CI [+0.00009, +0.00786] (positive conditional evidence); Δmacro AP=+0.00748, exploratory95%CI [-0.00078, +0.01600].

multiview-minus-xresnet101: Δmacro AUROC=-0.00847, primary97.5% Bonferroni CI [-0.01312, -0.00400] (negative conditional evidence); Δmacro AP=-0.01354, exploratory95%CI [-0.02253, -0.00408].

multiview_aug-minus-multiview: Δmacro AUROC=-0.00071, secondary95% unadjusted CI [-0.00333, +0.00191] (inconclusive conditional evidence); Δmacro AP=+0.00293, exploratory95%CI [-0.00247, +0.00803].

A positive point estimate alone does not establish superiority. Individual diagnostic-label changes are tabulated with exploratory intervals; no blanket claim of improvement across labels. No probability ensemble across seeds and no test-fitted calibration. Macro AP is not macro AUROC; prevalence baselines accompany each AP.

## Interpretation and next research

This study tests a finite lightweight multiview design and a bounded augmentation against a matched flat CNN and strong reference pipeline. Any advantage is conditional on this benchmark and design. A negative result motivates inspecting view redundancy, model capacity and optimization with a new development-only protocol, not tuning against this opened test. Numerical proximity to0.92–0.93 AUROC does not establish clinical readiness or generalization.

Future useful work: separately preregister view ablations, compare budgets/representations using newly separated development roles (still exploratory after dataset reuse), and establish an independently harmonized external ECG cohort. Do not claim selective transfer from this classification study. Seven module interfaces and synthetic selector checks are available; patient overlap removal and five-class endpoint harmonization for CinC2021 are unresolved. See [feasibility](TRANSFER_FEASIBILITY.md).

## Reproducibility and release

See [protocol](EXPERIMENT_PROTOCOL.md), [audit](DATASET_AUDIT.md), [architecture](ECG_ARCHITECTURE.md), [reference fidelity](BASELINE_REPRODUCTION.md) and [reproduction](REPRODUCIBILITY.md). Hardware AppleM5Pro48GiB, CPU fallback following failed XResNet synthetic MPS-gradient tolerance. Existing research repositories remain read-only. No GitHub remote or push; no data/checkpoints/individual scores published.

## Direct answers and seed stability

The reference reproduction succeeded as a competitive **modern adaptation**:
the prespecified final-epoch procedure attained macro AUROC **0.92800**, close
to the published approximately 0.928; its macro AP was **0.82241**. The
validation-best procedure attained AUROC **0.92985**, AP **0.82759**. Different
dataset/software versions and checkpoint conventions prevent an exact historical
reproduction or a paired superiority claim against the published number.

Our no-augmentation five-view network attained macro AUROC **0.92139** and
macro AP **0.81405**. Relative to the matched flat CNN, both seeds improved
AUROC (seed 101: +0.00423; seed 202: +0.00388). The mean difference +0.00406
has a positive 97.5% conditional bootstrap interval, **[+0.000085, +0.007856]**,
whose lower endpoint is extremely close to zero. With only 2,000 bootstrap draws,
finite Monte Carlo resolution and omitted retraining/model-selection uncertainty
matter. This is limited conditional evidence, not established general superiority.
The AP improvement +0.00748 has an interval spanning zero.

Relative to XResNet validation-best, five-view underperformed in both seeds and
on both macro ranking metrics: AUROC difference -0.00847 (adjusted interval
[-0.01312, -0.00400]); AP difference -0.01354 (exploratory interval
[-0.02253, -0.00408]). The proposed lightweight design did not match the stronger
reference's ranking under this protocol. No equivalence margin was registered.

Exploratory label findings versus flat CNN: HYP AUROC improved by about 0.01112
and HYP AP by 0.03183; CD AP improved by 0.01220. These per-label intervals are
unadjusted and do not establish medically meaningful improvements. There is no
uniform gain over XResNet. Bounded augmentation changed macro AUROC by -0.00071
and AP by +0.00293, both intervals spanning zero: no reliable ranking benefit.

The small CNNs show substantial late overfitting in development curves; their
validation-selected checkpoints were retained. Global pooling after short local
receptive fields limits what the rhythm branch can represent. These are observed
behavior/static architectural limits, not a causal explanation of the result.
View-specific utility remains untested. The matched comparison changes clinical
grouping and deterministic representations together, so it does not isolate one
fusion mechanism or one derived view.

Achieving 0.92–0.93 macro AUROC is realistic **on this benchmark**: our actual
five-view and reference results lie in that range. Macro AP is a distinct
prevalence-dependent quantity and is lower. No external validity, future-event
prediction or selective-transfer advantage follows. A defensible next contribution
is a separately registered view/receptive-field ablation and an overlap-audited,
clinically harmonized external ECG study; do not tune again against this opened
fold 10.
