# Phase 2 external ECG audit

**DRAFT — research design only; reviewed 2026-11-12. No new predictive execution.**

## 1. Frozen project and evidence boundary

Workspace `~/Desktop/dmvmat-ecg`, branch `main`, HEAD
`1f0ff9031e9e1e2adc97962922fd97705570baf2`; starting working tree clean.
The completed study is not reopened. The audit read existing aggregate reports,
source, checkpoint bytes/checksums and read-only SQLite metadata; it did not load
test prediction arrays, perform inference, run tests or reconstruct metrics.

Evidence at this commit:

- [Final summary](../FINAL_RESEARCH_SUMMARY.md) and
  [results](../EXPERIMENT_RESULTS.md): five-view AUROC 0.92139, AP 0.81405;
  flat 0.91733/0.80657; XResNet validation-best 0.92985/0.82759. The adjusted
  five-view-minus-flat AUROC interval barely excludes zero, while its AP interval
  includes zero. These findings do not establish broad architecture superiority.
- [Evaluation completion](../../results/evaluation_completion.json): one consumed
  receipt `f12a6564e6bb436b8675bcbe5223dea1`, lock digest
  `ec832433e023cbff689effd868791f7060008ff12e88a6e73e708acbadb5fbdd`.
  Ledger remains `evaluated`, 11 attempts, one access. All files in the existing
  [lock](../../results/evaluation_lock.json) matched their stored hashes on this audit.
- [Dataset audit](../../results/dataset_audit.json): PTB-XL 1.0.3,
  21,799 raw recordings/18,869 patient IDs; 21,388 eligible recordings;
  17,084/2,146/2,158 official train/validation/test records. No detected exact
  waveform or known-patient overlap. This does not exclude near-duplicates or
  unknown person linkage. Fold 10 has already been evaluated.

## 2. Exact backbone readiness

Implementation: [networks.py](../../src/dmvmat_ecg/models/networks.py), lines 19–75;
[signals.py](../../src/dmvmat_ecg/preprocessing/signals.py), lines 5–35.

| Selectable owner | Physical/derived input | Intermediate output |
|---|---|---|
| `view_1` | I, II, III, aVR, aVL, aVF; `[B,6,1000]` | `[B,64]` |
| `view_2` | V1–V3; `[B,3,1000]` | `[B,64]` |
| `view_3` | V4–V6; `[B,3,1000]` | `[B,64]` |
| `view_4` | lead II, across-lead RMS, mean absolute first difference; `[B,3,1000]` | `[B,64]` |
| `view_5` | 12 log-magnitude spectra; `[B,12,396]` | `[B,64]` |
| `fusion_1` | concatenated 320 features | `[B,128]` |
| `fusion_2` | 128 features | `[B,64]` |
| `prediction_head` (not selectable) | 64 features | `[B,5]` logits |

Each encoder uses bias-free Conv1d widths 32/64/64, kernels 7/5/3, stride 2,
BatchNorm/ReLU; average/max concatenation 128 → Linear64/ReLU/LayerNorm.
Fusion blocks are Linear/ReLU/LayerNorm/dropout 0.2. Five independent sigmoids;
BCE-with-logits, not softmax. Count: 212,261 parameters. Analytical module counts
are 32,576; 31,904; 31,904; 31,904; 33,920; 41,344; 8,384; head 325.

`representations()` returns all seven tensors, but does **not** itself detach
them or enforce roles. A new extractor must authorize the dataset role, call
`eval()`/`no_grad()`, detach, retain ordered record/group hashes, and prohibit
outer-test extraction for selector decisions. `selectable_modules()` explicitly
identifies the owners. Do not alter this completed implementation.

Known limitations: local receptive field 23 samples (0.23 s) before global
pooling; temporal order is discarded by pooling; no RR detector or verified
heart-rate representation. Spectrum is Hann-window rFFT at 100 Hz, retaining
0.5–40 Hz, `log1p(abs(FFT)/sqrt(length))`; frequency location is also summarized
by convolution/pooling. Derived views add no independent measurement. No
view-wise benefit, missing-lead robustness or transfer advantage was established.
BatchNorm buffers require freezing as well as parameters. Calling `train()`
after `apply_mask()` currently reactivates frozen BatchNorm unless corrected by
a new transfer trainer. See [interface.py](../../src/dmvmat_ecg/transfer/interface.py):67–82.

The current data/metric/training interfaces hard-code five fully observed labels,
official folds and scratch training. [store.py](../../src/dmvmat_ecg/data/store.py):14–33;
[metrics.py](../../src/dmvmat_ecg/evaluation/metrics.py):15–88;
[engine.py](../../src/dmvmat_ecg/training/engine.py):76–255.
They do not implement partial-label loss or adaptation epoch-zero selection.
The existing state machine cannot be repurposed: [state.py](../../src/dmvmat_ecg/evaluation/state.py):38–49,115–174
closes development and consumes the sole test lease. Phase 2 needs new versioned
code/configuration/ledger paths, not a phase reset or edits to old files.

## 3. Source checkpoints actually available

Read-only checksum verification passed for best/final files of all eight full
fits. Paths below are local/private; no model was deserialized or run.

| Family | Seed | Run folder under `private/runs/` | Best epoch |
|---|---:|---|---:|
| five-view |101|`541ceff8cfbd45b2b8f16db2c3fc3275`|13|
| five-view |202|`1419561ecfdd4d329559e33e1b3a322f`|18|
| augmented five-view |101|`8740f1e218804bd89052150f25279c5f`|19|
| augmented five-view |202|`fede089b62934469be7b39c217bab136`|22|
| flat CNN |101|`5defa4e903534533aaf0106451f77412`|18|
| flat CNN |202|`a33ff61f65864b30b258966cf85ade6d`|18|
| XResNet |101|`9f340af838904c878b9ee7e47b2cf558`|37|
| XResNet |202|`1cd7d36ecdb844abba9092c7960bcfb4`|37|

Each folder has `best.pt` and `final.pt`; their digests are in
[development lineage](../../results/development_lineage.json) and the lock.
Five-view best files are 884,501 bytes; XResNet best files 7,822,165 bytes.
They contain model state, not complete optimizer/RNG continuation state.
They are reusable initializations for an approved new experiment, not resumable
unfinished fits. Use both unaugmented seeds by methodological designation,
never whichever seed/family won fold 10. The proposed narrower task needs new
source-supervised heads; current CD logits are not specific IAVB/IRBBB classifiers.

Preprocessing is a single scalar mean/std fitted only on eligible PTB training
signals, in mV, then float32 model inputs. Source identity, preprocessing and
membership digests are in the existing audit/lock; preserve them. New target
resampling must not overwrite source caches or refit source normalization on
target/test data. XResNet uses 250-point training crops and seven-window maximum
probability inference; five-view uses the full 1,000 points. Comparisons with
XResNet therefore remain pipeline comparisons, not parameter-matched experiments.

## 4. Official external-source inventory

Numbers in this section describe releases, not newly audited eligible people.

| Candidate/release | Recordings and signal format | Identifiers/labels | Access and overlap assessment |
|---|---|---|---|
| PTB-XL 1.0.3 [D1] | 21,799 raw, 12 leads, 10 s; official 100/500 Hz versions | Actual `patient_id`, `strat_fold`, SCP statements | CC-BY-4.0; already available locally; source, not a new external cohort |
| Georgia public Challenge component [D2,D3] | 10,344 public of 20,672 total; 12 leads, 500 Hz, 5–10 s | WFDB/MAT, SNOMED-like diagnoses; stable person linkage not established by inspected documentation | CC-BY-4.0 Challenge release. Hidden 5,167 validation/5,161 test records are not available for our local study |
| Chapman original Figshare collection v2 [D4,D5] | 10,646 patients/records per authors; 12 leads, 500 Hz, 10 s, raw/denoised CSV | `FileName` is a record ID; rhythm and condition abbreviations; one-record-per-subject described | Official item APIs verify CC0 for raw ZIP item 8361860 v1 and diagnostic item 8360408 v2; do not generalize one item's licence to every collection item |
| Chapman Challenge component [D2] | 10,247 recordings, not the original 10,646 | Converted WFDB/MAT/SNOMED labels | CC-BY-4.0 release; overlaps original Chapman; not a second independent cohort |
| Ningbo Challenge component [D2,D6] | 34,905 recordings; 12 leads, 500 Hz, 10 s | Combined release describes patient ECGs; inspected WFDB sample lacks a patient-ID comment | CC-BY-4.0. Need component-to-person and component-to-release crosswalk |
| Combined `ecg-arrhythmia` 1.0.0 [D6] | 45,152 patient ECGs per release; 12 leads, 500 Hz, 10 s | WFDB/MAT, condition dictionary; authors describe physician review | CC-BY-4.0; contains Chapman/Ningbo material; do not count these as independent datasets from their Challenge copies |
| Entire CinC 2021 [D2,D3] | Heterogeneous multi-source collection, includes PTB and PTB-XL | Different acquisition, annotation and diagnostic-code processes | Reject as an independent target in its entirety; remove PTB-XL, and conservatively exclude PTB in this first design |

The Challenge page's six comment fields and an inspected combined-release sample
header show `Age/Sex/Dx/Rx/Hx/Sx`, not a reliable person key. Absence in one sample
is not proof about every header. Nor does a unique filename prove a unique person.
The combined release's patient count is useful provenance evidence, not an
independently verified linkage table for the Challenge subsets. Separate hospitals
and geography support distinct acquisition provenance but do not prove no shared
person with different ECGs. No independent-patient claim is currently justified.

## 5. New findings versus the initial feasibility memo

1. Challenge 1.0.3, published July 2022, is available in addition to the previously
   cited 1.0.2 landing page. The former provides the training tree and large
   checksum listing; a recursive fetch of 1.0.2's own file listing would mainly
   retrieve the licence/papers, not the named external waveforms. Pin an actual
   distribution, not only a landing-page DOI.
2. Original Chapman 10,646, Challenge Chapman 10,247, and the combined 45,152
   release must not be conflated. Conversion can change IDs, units and labels;
   counts summing to 45,152 do not constitute a file-by-file identity proof.
3. The official condition dictionary collapses left anterior/posterior bundle
   abbreviations to an LBBB code; the crosswalk document treats that as a blocker.
4. Published code counts are not verified negatives, unique-patient counts,
   complete-task labels or final adaptation support after length/overlap checks.

## 6. Mandatory future independence audit (not executed)

Keep database/release/file identity separately from person and duplicate-group ID.
Require curator documentation of stable patient linkage or a release-specific
one-ECG-per-person assertion with a reproducible crosswalk. Group repeated exports,
raw/denoised pairs, duplicates and all segments of one recording/patient together.
Never infer person identity from age/sex or assign hospitals from numeric IDs.

Exclude the complete PTB-XL Challenge component, its historical versions, and PTB
from target candidates. Compare candidate waveforms against all known PTB-XL
release fingerprints using an integrity-only auditor, not source fold-10 prediction
or performance code. Match native bytes, header-calibrated canonical physical
signals, and consistently resampled signals. Search for time shifts, crops and
scale/lead permutations. A proposed near-duplicate flag is correlation ≥0.995
on ≥10 leads over ≥8 seconds after ≤0.2 s alignment; this is a conservative
manual-review trigger, not proof of identity or an automatic clinical exclusion.
Validate it on known converted copies/synthetic transformations before use;
do not tune it using target prediction outcomes. Remove uncertain source-linked
target components or block the independent-cohort claim. Retain an exclusion log.

Do not train Chapman→Ningbo or evaluate on combined copies until a verified
component crosswalk rules out cross-release duplicate people/exports. No assertion
of full byte identity is made here: no external waveform archive was downloaded.

## 7. Source register and inspected metadata

All links are primary sources; accessed 2026-11-12. Nature full-page requests
redirected to an inaccessible identity endpoint; the Chapman paper was read via
its PMC mirror. Figshare item pages returned 403, but official item API metadata
was available. Only small codebooks, code and one header were read in memory;
no external patient table, ZIP, full waveform collection or derived data artifact
was saved.

- **D1:** [PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/).
- **D2:** [Challenge 2021 v1.0.2, Data Sources/Data Access](https://physionet.org/content/challenge-2021/1.0.2/).
- **D3:** [Challenge 2021 v1.0.3](https://physionet.org/content/challenge-2021/1.0.3/), DOI `10.13026/34va-7q14`.
- **D4:** [Original Chapman paper, participant/data-record sections](https://pmc.ncbi.nlm.nih.gov/articles/PMC7016169/), DOI `10.1038/s41597-020-0386-x`.
- **D5:** [Figshare collection v2](https://figshare.com/collections/ChapmanECG/4560497/2);
  official APIs [raw item](https://api.figshare.com/v2/articles/8361860) and
  [diagnostic item](https://api.figshare.com/v2/articles/8360408).
- **D6:** [Combined release 1.0.0](https://physionet.org/content/ecg-arrhythmia/1.0.0/), DOI `10.13026/wgex-er52`.
- **D7:** [Challenge authors' heterogeneous-label discussion](https://pmc.ncbi.nlm.nih.gov/articles/PMC9469795/), DOI `10.1088/1361-6579/ac79fd`.
- Official combined dictionary: `ConditionNames_SNOMED-CT.csv`, SHA-256
  `05d6fa23acdcbb466b378310ccfad22ea6cef4d3abfa83e8f660c77016be028d`.
  Inspected sample-header SHA-256
  `ba1482ac0a6bd7bfd9252c55c08885806be69ab27be52a393ae00d522a46999a`;
  no individual field values reproduced here.

**Decision: BLOCKED for a claim of independent clinical transfer.** See the
crosswalk and decision memo for the precise evidence gates. Engineering
implementability is a separate question from cohort/endpoint validity.

## Phase 2A evidence addendum (2026-11-12)

The Challenge authors' [Table 2 and split methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC9469795/)
give 15,738 Georgia patients across 20,672 recordings and describe original
split isolation of repeated patients. Public file-to-patient linkage remains
unresolved; the public subset's exact unique-patient count is not established.
This strengthens the need for patient keys before constructing new target roles.
See [external evidence resolution](EXTERNAL_EVIDENCE_RESOLUTION.md) for the
code-absence versus clinical-negative distinction and sources searched. **External
clinical transfer remains BLOCKED**; no full archive was downloaded.

The checkpoint inventory above describes the completed benchmark, not independent
initializations for an internal resplit. The new
[internal pilot](INTERNAL_TRANSFER_PILOT_PROPOSAL.md) must pretrain entirely new
source models and normalization on its newly assigned source patients; it may
not reuse historical weights that saw new target fitting/stopping/evaluation
participants. Synthetic Phase 2A implementation is separate from this frozen
baseline's code and results; see [implementation report](SELECTOR_IMPLEMENTATION_REPORT.md).
