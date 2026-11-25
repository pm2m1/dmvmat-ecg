"""Invented data only: source isolation and numerical parity to Phase4B."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dmvmat_ecg.phase2.contracts import MODULES
from dmvmat_ecg.phase3b.roles import Normalization
from dmvmat_ecg.phase4.safety import digest
from dmvmat_ecg.phase4b.train_alignment import TargetStream
from dmvmat_ecg.phase4c_placebo.data import (
    BUNDLE,
    OLD,
    ROOT,
    RUN,
    SOURCE,
    SPLIT,
    SourceInputs,
    access_check,
    normalization,
)


@pytest.mark.parametrize(
    "path",
    [
        BUNDLE / "labels.npy",
        BUNDLE / "signals.npy",
        OLD / "alignment_inputs/U_wave.npy",
        ROOT / "data/FORBIDDEN_SIGNAL",
        SPLIT,
    ],
)
def test_source_guard_refuses_CPSC_or_non_source_inputs_before_open(path):
    with pytest.raises(PermissionError):
        access_check("source", "open", (str(path), "rb", os.O_RDONLY))


def test_source_guard_allows_source_reads_and_only_new_output_writes():
    access_check("source", "open", (str(SOURCE / "source_train_x.npy"), "rb", os.O_RDONLY))
    access_check("source", "open", (str(RUN / "INVENTED_OUTPUT"), "wb", os.O_WRONLY | os.O_CREAT))
    with pytest.raises(PermissionError):
        access_check("source", "open", (str(SOURCE / "source_train_x.npy"), "wb", os.O_WRONLY))
    access_check("target", "open", (str(OLD / "fits/INVENTED.json"), "rb", os.O_RDONLY))
    with pytest.raises(PermissionError):
        access_check("target", "open", (str(OLD / "fits/INVENTED.json"), "wb", os.O_WRONLY))


def test_source_features_reject_any_target_role():
    features = object.__new__(SourceInputs)
    for role in ("U", "A", "B", "test", "F", "G"):
        with pytest.raises(PermissionError):
            features.batch(role, [0])


def test_memory_efficient_normalization_is_bitwise_original_algorithm():
    x = np.random.default_rng(12).normal(size=(33, 12, 40)).astype(np.float32)
    identity = "INVENTED_SOURCE_ROLE"
    source = SimpleNamespace(
        x={"source_train": x}, manifest={"roles": {"source_train": {"identity": identity}}}
    )
    data = SimpleNamespace(
        read=lambda *a: (x, None, None), role=lambda *a: SimpleNamespace(identity=identity)
    )
    assert normalization(source).__dict__ == Normalization.fit(data).__dict__


def test_second_stream_does_not_advance_first_source_shuffle_generator():
    first = torch.Generator().manual_seed(13)
    before = first.get_state().clone()
    stream = TargetStream(83, torch.Generator().manual_seed(14))
    indices = stream.take(100)
    assert torch.equal(first.get_state(), before)
    assert len(set(indices[:83])) == 83 and indices.min() >= 0 and indices.max() < 83


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(5, 8), torch.nn.BatchNorm1d(8), torch.nn.ReLU(), torch.nn.Dropout(0.2)
        )
        self.prediction_head = torch.nn.Linear(8, 2)

    def representations(self, wave):
        z = self.encoder(wave)
        return {name: z * (j + 1) / 7 for j, name in enumerate(MODULES)}

    def forward(self, **x):
        return self.prediction_head(self.representations(**x)["fusion_2"])


def factory(s):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(s)
        return Toy()


def test_source_loop_exactly_matches_frozen_aligned_loop_when_second_data_equal(
    monkeypatch, tmp_path
):
    import dmvmat_ecg.phase4b.train_alignment as original
    import dmvmat_ecg.phase4c_placebo.source as placebo

    torch.set_num_threads(2)
    rng = np.random.default_rng(15)
    x = {
        "source_train": rng.normal(size=(32, 5)).astype(np.float32),
        "source_stop": rng.normal(size=(16, 5)).astype(np.float32),
    }
    y = {
        role: np.array([[j % 2, (j // 2) % 2] for j in range(len(a))], dtype=np.float32)
        for role, a in x.items()
    }
    source = SimpleNamespace(y=y, identity="INVENTED_SOURCE")

    class Features:
        def __init__(self):
            self.source = source

        def batch(self, role, ix):
            if role == "U":
                role = "source_train"  # fictional second stream only
            return {"wave": torch.tensor(x[role][ix], dtype=torch.float32)}

    norm = Normalization(0.0, 1.0, digest(["INVENTED_SOURCE_ROLE"]), "fresh-source-only")
    monkeypatch.setattr(placebo, "fresh_model", factory)
    monkeypatch.setattr(original, "fresh_model", factory)
    monkeypatch.setattr(
        original,
        "setup",
        lambda: SimpleNamespace(identity="INVENTED_TARGET", ids=lambda _: range(32)),
    )
    monkeypatch.setattr(original, "SourceData", lambda: source)
    monkeypatch.setattr(original, "Features", Features)
    monkeypatch.setattr(original, "load_source", lambda *a: (None, norm, None, None))
    monkeypatch.setattr(original, "sha", lambda *a: "INVENTED_FEATURE_HASH")
    monkeypatch.setattr(original, "RUN", tmp_path)
    prior = original.source_fit(15101, 0.1, retain=False)
    _, p, current = placebo.train(15101, Features(), norm)
    for key in ("state_hash", "history_sha256", "fold9_predictions_sha256", "selected_epoch"):
        assert prior[key] == current[key]
    assert p.shape == (16, 2)
