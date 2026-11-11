# Local data

PTB-XL1.0.3, PhysioNet, CC-BY4.0, DOI10.13026/kfzx-aw45.
Run `uv run python scripts/download_ptbxl.py` for publisher-verified metadata and
100Hz records only. Run `uv run python scripts/audit_dataset.py` once before
training to verify checksums, labels, patient isolation, waveform identity and
training-only normalization. These scripts require network/local storage; no
raw data or patient-level assignments belong in Git. Never rerun an audit to
reset an evaluated study; preserve the independent experiment ledger.
