# Reference baseline fidelity

Source: [helme/ecg_ptbxl_benchmarking](https://github.com/helme/ecg_ptbxl_benchmarking),
commit `cdbf4e66d7e57d9b6a2657b6024716212b8d0afa`, GPL3. The upstream clone is
local/ignored. Vendored `models/vendor/xresnet1d.py` retains the original code
with a redirected head-helper import and attribution. Root GPL3 license covers
this derivative project; no implication that dataset licensing is GPL.

The upstream environment specifies FastAI1.0.61 and PyTorch1.4.0, Linux-oriented
dependencies. This project uses Python3.12.14/PyTorch2.15.0 on Apple Silicon.
It is an **architecture-faithful modern PyTorch adaptation**, not exact numerical
reproduction of the historical software stack or dataset release.

## Architecture actually implemented

Upstream `code/models/xresnet1d.py` bottleneck expansion4, depths3/4/23/3.
Stage base widths are64/64/64/64, not the common ImageNet64/128/256/512 schedule.
Stem32/32/64, kernel5, first stride2, maxpool3stride2. Residual bottleneck
1×1→kernel5→1×1 with avg-pool identity downsampling. Output256 channels.
Adaptive max then average concatenation512; BN→dropout0.25→linear128→ReLU,
BN→dropout0.5→linear5. Upstream post-construction initialization reapplied:
Kaiming-normal Conv/Linear, bias0, BatchNorm gamma1/beta0.

Verified **1,872,261 parameters**. Conv/Linear MACs41,138,176 per250-point crop,
287,967,232 per seven-window recording inference, excluding pooling,
normalization and preprocessing. Parameter/MAC evidence is in the synthetic
device report; final actual model counts are also recorded in each completed fit.

## Training and signal fidelity

Upstream `code/models/fastai_model.py:160` constructor defaults: batch128,
epochs50, lr0.01, wd0.01,2.5s crops, held-out windows with50% overlap,
max prediction aggregation; default no early stopping. Training applies
`weight_init`, runs an LR-finder diagnostic then `fit_one_cycle` and retains
the final model. This project omits the LR-finder fit and explicitly pins
AdamW betas0.9/0.99 (FastAI1 AdamW wrapper's beta2), epsilon1e-8, weight decay
on all parameters, modern OneCycleLR. FastAI/PyTorch step-index and numerical
details need not be identical. We retain both epoch50 and validation-best models;
the latter is a separate, labeled procedure. No epoch-count reduction is based
on test results or used to claim exact reproduction.

Upstream global StandardScaler is reproduced as a single train-fitted scalar.
Our float32 signal cache introduces a small rounding difference from unrounded
physical values. Label construction follows diagnostic SCP statement membership,
no likelihood threshold, drop empty diagnostic targets. Output order differs
from upstream alphabetical order but mapping is explicit. Random training crops
preserve the source's exclusive upper random-start bound; validation/test use
seven fixed250-sample windows and maximum **sigmoid probability** per label.

## Device decision

MPS is built and available. Synthetic CPU/MPS check passed forward/loss but
failed a preset XResNet gradient tolerance in one of20,480 elements of one
gradient tensor (maxabsolute0.0052895 vsatol0.003, rtol0.01). This is a numerical
acceptance failure, not evidence that all MPS operations are unsupported.
Flat and multiview encoders passed the same probe. To retain a common reliable
device, all real fits use CPU float32, four threads per paired-seed process.
No claim of identical CPU/MPS fitted weights. See `results/device_probe.json`.

The published superclass XResNet101 macro AUROC is approximately0.928, as shown
in the repository benchmark table. This is neither our achieved score nor an AP
value. The final experimental-results report contains our own locked metrics
and qualified comparison after all training and the single test lease complete.
