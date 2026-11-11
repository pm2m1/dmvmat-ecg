# Initial ECG study — protocol fixed before predictive fitting

This is an independent PTB-XL 1.0.3, five diagnostic-superclass, multilabel
classification study. It does not predict future events or establish clinical
utility or cross-cohort transfer. Published benchmark performance is known.
There is one authorized final fold-10 access for the entire finite candidate set.

## Data and boundaries

Use only official 100 Hz data and publisher SHA-256 checksums. Of 21,799 raw
recordings, 21,388 have at least one diagnostic superclass. Mapping uses every
diagnostic SCP key, regardless of likelihood (including zero), as in the
benchmark. Order is NORM, MI, STTC, CD, HYP; upstream alphabetic ordering differs
but class semantics do not. No five-way softmax. Folds 1–8 train (17,084), 9
validation (2,146), 10 test (2,158). Patient isolation and exact physical/raw
waveform overlap are audited across all raw records. A constant lead is retained;
no outcome-based quality exclusions. Unknown cross-patient identity remains a
limitation. Metadata-only test counts and waveform-integrity inspection are not
predictive evaluation access.

Train a scalar global mean/std across all training signal values only, matching
the reference global standardization. No filtering, resampling, imputation or
label-informed signal transforms. Physical mV signals cached float32; fitting
statistics use float64. Never fit statistics on folds9/10.

## Finite models and training

The exact machine-readable matrix is `configs/study.json`. Eight full fits:
four variants × seeds101/202, plus three training-only smoke fits (512 records,
four batches, one epoch, seed17), two shared failure repairs, hard13 attempts.
Failures count. Successful fits cannot be repeated. No optimizer/RNG continuation
is claimed; a repair begins fresh with a new attempt identity. No additional
searches, ensembles, hyperparameters or test-guided reruns.

All full fits run 50 epochs, batch128, AdamW lr0.01, betas(0.9,0.99), eps1e-8,
weight decay0.01 on all trainable parameters, BCEWithLogitsLoss, OneCycleLR
pct_start0.3, div25, final_div10000, beta1 cycle0.95→0.85→0.95. These are modern
PyTorch adaptations of FastAI1 training, not identical optimizer trajectories.
No LR-finder warmup fit. Only a training-smoke out-of-memory event could trigger
the declared common batch64 fallback before full training; this would be recorded
prospectively as a resource amendment. No performance-triggered epoch reduction.
Common CPU float32, four threads, deterministic algorithms warn-only, zero loader
workers. CPU fallback follows the failed XResNet MPS gradient parity check; do
not relax tolerances to obtain a pass. Other encoders passed the synthetic probe.

1. XResNet1D101: upstream GPL3 source commit
   `cdbf4e66d7e57d9b6a2657b6024716212b8d0afa`, redirected head helper, 1,872,261
   parameters. Random 250-point training crop; upstream-exclusive final random
   start preserved. Seven fixed 250-point held-out windows, starts0:125:750;
   aggregate **maximum sigmoid probabilities**, not logits. No extra augmentation.
2. Flat CNN: full12×1000, 206,117 parameters, no augmentation.
3. Five-view CNN: full10s raw lead groups, learned temporal rhythm alternative,
   Hann-rFFT0.5–40Hz log-magnitude spectrum, 212,261 parameters, no augmentation.
4. Same five-view CNN with training-only scale U(0.95,1.05) and Gaussian noise
   sigma0.005×training std. No held-out augmentation, dropout or beat fabrication.

Flat and multiview have matched full-record inputs, optimizer, epochs and near
parameter count. Reference comparisons are **pipeline comparisons**, since its
published crop/max-window inference differs. Extra deterministic views do not
add independent measurements. No assertion that frequency/rhythm views help
individually without further ablations. No test-time ensembling across seeds.

Train all epochs, choose earliest epoch attaining maximum validation-record
macro AUROC, no early termination. Scratch epoch0 is not checkpoint-eligible.
Keep both best and final checkpoints. Also evaluate the XResNet final-epoch
alias (no extra fit), because upstream default retained epoch50 rather than the
validation-best checkpoint. Published 0.928 is a contextual macro AUROC reference,
not an achieved value or required target; modern software, release and training
differences prevent exact numerical reproduction claims.

## Thresholds and calibration

Each retained checkpoint chooses five independent validation-F1 thresholds on
0.05..0.95 by0.01. Ties: closest0.5 then smaller threshold. Also report fixed0.5.
No probability recalibration is fitted. Raw sigmoid scores are used for AP/AUROC.
ECE uses ten equal-width probability bins, left-closed/right-open, last includes1.
Reliability plots/Brier describe probability quality; they are not ranking gains.

## Lock and estimands

All eight fits, their best/final checksums, thresholds, preprocessing, membership
hashes, source/configuration, dependency lock, evaluation scripts and this protocol
must be locked before test access. Durable SQLite access receipt is consumed
before first test read. Generate and seal all ten registered prediction sets
(four best procedures plus XResNet-final ×two seeds) **before any test metric**.
Failure preserves the receipt; no automatic fresh test lease or model repair.
Post-access reporting corrections may use the same sealed predictions and must
be labeled corrections. Never tune against fold10.

Primary metric: macro AUROC, arithmetic mean across five labels, then across
paired seeds. Secondary: per-label AUROC/AP, macro AP, micro AP, prevalence
(random-ranking AP reference), per-label Brier/ECE, precision/recall/specificity/F1,
confusion counts at validation thresholds and0.5. Continuous scores only for
ranking. Training curves' online training metrics are explicitly distinct from
fixed-model validation metrics.

Primary contrasts: five-view minus flat CNN and five-view minus validation-best
XResNet. Patient-cluster bootstrap, 2,000 valid draws, seed70501, at most20,000
attempts; redraw samples missing a class. Sample patients with replacement,
retain all their recordings and use identical multiplicities for all models and
seeds. Mean seed-specific metrics, never mean probabilities. Ordinary95% CIs;
two primary AUROC contrasts additionally97.5% Bonferroni intervals. AP and
per-class/augmentation contrasts are secondary exploratory95% intervals, no
unregistered superiority claim. An interval spanning zero is inconclusive;
absence of a difference is not equivalence. Intervals condition on fitted models
and split; they omit retraining/model-selection uncertainty and external shift.
Two seeds are stability observations, not independent clinical populations.

## Interpretation and transfer gate

Record duration, inference seconds, parameters and Conv/Linear MAC estimates
(exclude FFT/normalization/pooling; XResNet full-record uses seven crop passes).
No fixed clinical improvement margin is asserted. A negative multiview result
is retained and reported. Attention, transfer, tuning against test and extra
datasets are outside this finite classification experiment.

Seven module representation interfaces and a synthetic MMD/logical-policy
adapter may be tested. MMD is selection evidence, not a training loss. ECG is
multilabel: geometry must explicitly condition on an individual label; any
cross-label aggregation/retention calibration needs a separate approved protocol.
CinC2021 contains PTB-XL and PTB. Independent transfer is blocked pending
verified overlap removal, person separation, endpoint harmonization and permissions.

## Sources

- [PTB-XL1.0.3](https://physionet.org/content/ptb-xl/1.0.3/), CC-BY4.0,
  DOI10.13026/kfzx-aw45; publisher checksum receipt in `results/dataset_audit.json`.
- [Official benchmark](https://github.com/helme/ecg_ptbxl_benchmarking), GPL3,
  pinned source as above, `code/models/fastai_model.py` constructor/fit and
  `code/models/xresnet1d.py`.
- [FastAI1 learner defaults](https://raw.githubusercontent.com/fastai/fastai1/master/fastai/basic_train.py)
  and [optimizer defaults](https://raw.githubusercontent.com/fastai/fastai1/master/fastai/torch_core.py).
