# Five-view ECG and matched control

All models output five logits; independent sigmoid probabilities, multilabel
BCE training. Architecture source `src/dmvmat_ecg/models/networks.py`; no old
tabular checkpoint, weight matrix or patient data is reused.

| Module | Input | Encoder | Output |
|---|---|---|---|
|view_1|6 limb leads,1000samples|CNN6→32→64→64|64|
|view_2|V1,V2,V3,1000samples|CNN3→32→64→64|64|
|view_3|V4,V5,V6,1000samples|CNN3→32→64→64|64|
|view_4|leadII, across-lead RMS, mean absolute first difference;1000samples|CNN3→32→64→64|64|
|view_5|12 Hann-window FFT log-magnitude spectra;396bins|CNN12→32→64→64|64|
|fusion_1|concatenation320|Linear→ReLU→LayerNorm→Dropout0.2|128|
|fusion_2|128|Linear→ReLU→LayerNorm→Dropout0.2|64|
|prediction_head|64|Linear|5|

CNN stages use kernels7/5/3, stride2, symmetric padding, bias-free convolutions,
BatchNorm and ReLU. Each concatenates global average and max, then Linear128→64,
ReLU and LayerNorm. Full raw input `[B,12,1000]`; view5 input `[B,12,396]`.
Spectrum uses `np.hanning(1000)`, rFFT at100Hz, retain0.5–40Hz inclusive,
`log1p(abs(FFT)/sqrt(1000))`, CPU deterministic preprocessing. Rhythm is a learned
temporal alternative; no RR, heart-rate estimate or beat-detection success is
fabricated. Derivative uses a prepended first sample, giving first difference0.
Derived features use the globally train-standardized signal. They do not add
independent physiological measurements and have not been validated as clinical
rhythm summaries. No diagnosis is inferred directly from these transformations.

Five-view total **212,261 parameters**, Conv/Linear MACs20,314,816 per full
recording. Matched flat CNN **206,117 parameters** (2.9%smaller), widths64/128/208,
same kernels/stride, project416→128, fusion128→128→64→5. FlatMACs22,990,144.
Both consume full10s and share training budget. Counts/MACs exclude FFT,
pooling and normalization operations. XResNet is a stronger, larger reference
with a different crop/inference pipeline; its comparison is not capacity-matched.

## Future selective-adaptation interface

`representations` returns seven named `[B,features]` tensors; `selectable_modules`
maps their owners explicitly. The prediction head is always separate.
Synthetic-only interface tests verify MMD/permutation behavior, invalid-geometry
SHARE, original logical thresholds, module freezing and independent head updates.
MMD is module-selection evidence, never automatically added as a classification
loss. All frozen normalization buffers must remain unchanged too; applying
`model.train()` later requires restoring frozen-module eval modes.

This sprint does not reproduce a complete ECG transfer study. Source-relative
class-direction retention/bootstrap calibration, label-wise policy aggregation,
target-label accounting, epoch-zero checkpoints and cross-population roles need
a separate protocol. Five multilabel ECG outputs cannot safely be reduced to an
argmax class for old binary geometry. Neither modularity nor synthetic success
demonstrates any observed selective-transfer advantage.

## Temporal receptive-field limitation

The three stride-two convolution stages have a local receptive field of23 samples (7+4×2+2×4), or0.23seconds at100Hz, before global pooling. Pooling summarizes the whole recording but discards temporal order. This does not establish an ability to recover RR intervals or long-range rhythm variability. The frequency view summarizes the full recording, but no view-specific benefit or clinical rhythm validity is demonstrated without a separate ablation.
