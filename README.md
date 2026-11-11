# DMV-MAT ECG

Independent PTB-XL1.0.3 five-superclass multilabel ECG research. This project
compares a modern PyTorch adaptation of the published XResNet1D101 benchmark,
a capacity-matched flat CNN, and a modular five-view ECG network. It does not
estimate future cardiovascular risk or establish clinical or transfer validity.

See [protocol](docs/EXPERIMENT_PROTOCOL.md). Results and the complete reproduction
instructions are generated after the finite experiments complete. Published
benchmark numbers must not be confused with our own measurements.

Python3.12; `uv sync --frozen`; `uv run pytest`; `uv run ruff check .`.
Official data are CC-BY4.0; code includes attributed GPL3 upstream architecture.
Raw waveforms, individual metadata/manifests, predictions and model checkpoints
are local and ignored. This repository has no remote publication authorization.
