"""Invented source fixtures: matching lambda0 path and mechanism checks."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dmvmat_ecg.phase2.contracts import MODULES
from dmvmat_ecg.phase3b.roles import Normalization
from dmvmat_ecg.phase4c_retrain.data import BUNDLE, OLD, PLACEBO, ROOT, RUN, SOURCE, access_check


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


@pytest.fixture
def invented(monkeypatch):
    import dmvmat_ecg.phase4c_placebo.source as placebo
    import dmvmat_ecg.phase4c_retrain.source as retrain

    torch.set_num_threads(2)
    rng = np.random.default_rng(110)
    x = {
        r: rng.normal(size=(n, 5)).astype(np.float32)
        for r, n in [("source_train", 32), ("source_stop", 16)]
    }
    y = {
        r: np.column_stack((np.arange(len(a)) % 2, (np.arange(len(a)) // 2) % 2)).astype(np.float32)
        for r, a in x.items()
    }

    class Features:
        source = SimpleNamespace(y=y)

        def batch(self, role, ix):
            return {"wave": torch.tensor(x[role][ix], dtype=torch.float32)}

    monkeypatch.setattr(placebo, "fresh_model", factory)
    monkeypatch.setattr(retrain, "fresh_model", factory)
    return Features(), Normalization(0.0, 1.0, "INVENTED", "fresh-source-only")


def test_parameterized_lambda_point_one_is_exact_frozen_placebo(invented):
    from dmvmat_ecg.phase4c_placebo.source import train as placebo
    from dmvmat_ecg.phase4c_retrain.source import train

    f, n = invented
    _, p, r = placebo(15101, f, n)
    _, q, s = train(15101, f, n, 0.1)
    assert p.tobytes() == q.tobytes()
    for key in (
        "state_hash",
        "history",
        "history_sha256",
        "selected_epoch",
        "fold9_predictions_sha256",
    ):
        assert r[key] == s[key]


def test_lambda_zero_is_BCE_only_and_ignores_diagnostic_MMD(invented, monkeypatch):
    import dmvmat_ecg.phase4c_retrain.source as module

    f, n = invented
    _, p, r = module.train(15101, f, n)
    old = module.alignment_loss
    monkeypatch.setattr(module, "alignment_loss", lambda a, b: old(a, b) + 7.0)
    _, q, s = module.train(15101, f, n)
    assert p.tobytes() == q.tobytes() and r["state_hash"] == s["state_hash"]
    assert all(e["loss"] == e["BCE"] for e in r["history"])
    assert r["lambda_value"] == 0.0 and r["MMD_gradient_used"] is False
    assert r["second_training_forward_retained"] is True


def test_extra_forward_changes_BN_and_dropout_rng_not_first_BCE_gradient():
    x = torch.arange(160, dtype=torch.float32).reshape(32, 5) / 160
    labels = torch.tensor(
        np.column_stack((np.arange(32) % 2, np.arange(32) % 3 == 0)), dtype=torch.float32
    )

    def run(second):
        m = factory(15101).train()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(15101)
            logits = m(wave=x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            if second:
                m.representations(wave=x.flip(0))
            loss.backward()
            return (
                [p.grad.clone() for p in m.parameters()],
                m.encoder[1].num_batches_tracked.item(),
                torch.get_rng_state().clone(),
            )

    g, count, state = run(False)
    h, count2, state2 = run(True)
    assert all(torch.equal(a, b) for a, b in zip(g, h, strict=True))
    assert (count, count2) == (1, 2)
    assert not torch.equal(state, state2)


def test_independent_second_sampler_does_not_consume_dropout_rng():
    from dmvmat_ecg.phase4b.train_alignment import TargetStream

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(15101)
        before = torch.get_rng_state().clone()
        stream = TargetStream(83, torch.Generator().manual_seed(19))
        stream.take(200)
        assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize(
    "path",
    [
        BUNDLE / "signals.npy",
        BUNDLE / "labels.npy",
        OLD / "fits/INVENTED.json",
        PLACEBO / "predictions/INVENTED.npz",
        ROOT / "data/INVENTED_TEST",
    ],
)
def test_source_scope_blocks_CPSC_and_old_result_reads(path):
    with pytest.raises(PermissionError):
        access_check("source", "open", (str(path), "rb", os.O_RDONLY))


def test_source_feature_reuse_is_read_only():
    p = PLACEBO / "source_inputs/source_train_wave.npy"
    access_check("source", "open", (str(p), "rb", os.O_RDONLY))
    with pytest.raises(PermissionError):
        access_check("source", "open", (str(p), "wb", os.O_WRONLY))
    access_check("source", "open", (str(SOURCE / "source_train_y.npy"), "rb", os.O_RDONLY))
    access_check("source", "open", (str(RUN / "INVENTED.json"), "wb", os.O_WRONLY | os.O_CREAT))


def test_compare_scope_permits_only_explicit_source_checkpoints():
    cp = BUNDLE / "checkpoints/INVENTED.pt"
    access_check("compare", "open", (str(cp), "rb", os.O_RDONLY), {cp})
    with pytest.raises(PermissionError):
        access_check("compare", "open", (str(cp), "rb", os.O_RDONLY))
    with pytest.raises(PermissionError):
        access_check("compare", "open", (str(BUNDLE / "labels.npy"), "rb", os.O_RDONLY), {cp})
