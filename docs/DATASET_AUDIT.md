# PTB-XL1.0.3 audit

Official release: [PhysioNet PTB-XL1.0.3](https://physionet.org/content/ptb-xl/1.0.3/),
DOI10.13026/kfzx-aw45. Dataset license CC-BY4.0; credit Wagner et al.,
[PTB-XL paper](https://doi.org/10.1038/s41597-020-0495-6). Read the publisher's
terms and retain attribution for any future release. No raw data, per-record
metadata, predictions or model weights are included in the public-safe inventory.

## Identity and integrity

Exact100Hz release downloaded from the official public S3 mirror. No500Hz files.
All43,598 required header/data files verified against publisher SHA256SUMS.txt;
metadata and label statements separately verified. Download receipt local at
`private/download-receipt.json`; aggregate audit `results/dataset_audit.json`.

| Source | SHA-256 |
|---|---|
| Publisher checksum manifest | b7224b92b341511ec3ceb13dc6652079b2c36a06504bcb49506f157f51dc695d |
| ptbxl_database.csv | 7600de9c1b27d181d850b3c6038a35d7c3ddb6bb33b702e3a20252a6859d216b |
| scp_statements.csv | ad05b0b1fcae83bb1230755ad9cfc7c96f303feddc08a4a9ad5bdc9ca63bac8f |

Every recording decoded through WFDB, finite physical signals,100Hz,1000samples,
12 named leads in mV, verified header/data compatibility. Original stored WFDB
orientation is `[1000,12]`; canonical model orientation `[12,1000]`. Named leads
are reordered I,II,III,aVR,aVL,aVF,V1…V6; no assumption about source array order.
One constant-lead observation is retained and reported, not imputed or silently
excluded. No cross-partition or within-partition exact raw/canonical waveform
duplicates were detected. This does not establish absence of near-duplicates or
undocumented person linkage. Waveform integrity checks do not inspect model
predictions or tune on test outcomes.

## Eligibility and official folds

Raw21,799 recordings /18,869 patient identifiers. Eligible21,388 recordings /
18,617 patient identifiers,411 without any diagnostic superclass excluded.
5,144 eligible recordings have more than one superclass. SCP diagnostic mappings
use statement keys including likelihood0, matching benchmark behavior; no
invented certainty threshold. Empty diagnostic targets are excluded before
fitting; other statement types are not misassigned to the five diagnostic labels.

| Role | Folds | Recordings | Patients | NORM | MI | STTC | CD | HYP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Training |1–8|17,084|14,823|7,596|4,379|4,186|3,907|2,119|
| Validation |9|2,146|1,917|955|540|528|495|268|
| Locked test |10|2,158|1,877|963|550|521|496|262|

Patient sets disjoint before and after eligibility filtering. Exclusions by role:
334train,37validation,40test. Test label counts are descriptive audit information,
not authorization for model selection. Membership lists, patient identifiers and
individual target arrays remain ignored/local. Test prediction access is gated
by the study ledger and one-use evaluation lease.

## Preprocessing lineage

One global scalar mean/std fitted across the training physical signal values,
float64 accumulation; applied without refitting to validation/test. Signals cached
float32, approximately1.027GB payload. Cache/state membership hashes are checked
at each role load. No test-derived normalization, quality threshold, clipping,
baseline removal, beat detection or data imputation. Derived frequency and rhythm
representations are deterministic signal transformations, with settings fixed
in the protocol before training.

## Version comparison limitation

The published benchmark predates1.0.3. Publisher release notes describe duplicate
removal and label revisions across1.0.2/1.0.3. Consequently the historical0.928
macro AUROC is context, not an identical-version paired comparison. Do not
substitute the entire CinC2021 collection as an independent dataset: it contains
an older PTB-XL release and PTB records.
