"""Isolated synthetic training integration; never opens the real study store."""

import json

import numpy as np
import pytest
import torch

from dmvmat_ecg.evaluation.state import Study
from dmvmat_ecg.training import engine
from dmvmat_ecg.utils.io import digest, sha, write_json


def synthetic_store(tmp_path):
    from pathlib import Path

    config = json.loads((Path(__file__).parents[1] / "configs/study.json").read_text())
    config.update(epochs=3, batch_size=8)
    write_json(tmp_path / "configs/study.json", config)
    study = Study(tmp_path)
    memberships = {}
    hashes = {}
    rng = np.random.default_rng(83)
    cache = tmp_path / "private/cache"
    cache.mkdir()
    for role, n, fold, offset in [("train", 32, 1, 0), ("validation", 16, 9, 100)]:
        ids = np.arange(n) + offset
        memberships[role] = ids.tolist()
        np.save(cache / f"{role}_x.npy", rng.normal(size=(n, 12, 1000)).astype("float32"))
        np.savez(
            cache / f"{role}_meta.npz",
            ecg_id=ids,
            patient_id=ids,
            y=np.tile(np.arange(n) % 2, (5, 1)).T.astype("float32"),
            fold=np.full(n, fold),
        )
        for name in (f"{role}_x.npy", f"{role}_meta.npz"):
            hashes[name] = sha(cache / name)
    state = dict(
        mean=0.0, std=1.0, fit_role="train", fit_membership_hash=digest(memberships["train"])
    )
    write_json(cache / "standardization.json", state)
    hashes["standardization.json"] = sha(cache / "standardization.json")
    study.set("memberships", memberships)
    study.set(
        "dataset",
        dict(
            cache_hashes=hashes,
            publisher_manifest_sha256="synthetic",
            standardized_fit_membership_hash=state["fit_membership_hash"],
        ),
    )
    study.set("phase", "development")
    return study


def test_synthetic_fit_checkpoint_direction_and_ledger(tmp_path, monkeypatch):
    study = synthetic_store(tmp_path)
    monkeypatch.setattr(
        engine.subprocess, "check_output", lambda *a, **kw: "synthetic-source-revision"
    )
    calls = []

    def prediction(model, ds, device, batch_size):
        calls.append(len(calls))
        # Epoch1 perfect, then two worse epochs; final-threshold pass is fourth.
        p = 0.1 + 0.8 * ds.y if len(calls) == 1 else 0.9 - 0.8 * ds.y
        return p, 0.001

    monkeypatch.setattr(engine, "predict", prediction)
    trial = engine.fit(study, "flat_cnn", 101, "cpu")
    record = study.runs()[0]
    assert record["status"] == "successful" and record["metadata"]["best_epoch"] == 1
    assert record["metadata"]["epochs_completed"] == 3 and len(calls) == 4
    assert record["metadata"]["validation_macro_auroc"] == 1
    folder = tmp_path / "private/runs" / trial
    best = torch.load(folder / "best.pt", weights_only=True)
    final = torch.load(folder / "final.pt", weights_only=True)
    assert any(not torch.equal(best[k], final[k]) for k in best)
    with pytest.raises(PermissionError):
        engine.dataset(study, "flat_cnn", "test")


def test_preprocessing_fit_membership_and_cache_tampering(tmp_path):
    study = synthetic_store(tmp_path)
    ds, _ = engine.dataset(study, "multiview", "train")
    assert ds[0][0]["spectrum"].shape == (12, 396)
    path = tmp_path / "private/cache/standardization.json"
    state = json.loads(path.read_text())
    state["fit_role"] = "test"
    write_json(path, state)
    with pytest.raises(PermissionError):
        engine.dataset(study, "flat_cnn", "train")


def test_complete_prediction_shape_and_seven_window_max():
    class Dummy(torch.nn.Module):
        def forward(self, wave):
            return wave[:, 0, :1].expand(-1, 5)

    x = np.zeros((2, 12, 1000), dtype="float32")
    x[0, 0] = np.linspace(-2, 2, 1000)
    x[1, 0] = np.linspace(2, -2, 1000)
    ds = engine.CropDataset(
        x, np.zeros((2, 5)), [1, 2], dict(mean=0.0, std=1.0), training=False, window=250
    )
    scores, _ = engine.predict(Dummy(), ds, "cpu", batch_size=4)
    expected = torch.sigmoid(torch.tensor([x[0, 0, 750], x[1, 0, 0]])).numpy()
    np.testing.assert_allclose(scores, np.repeat(expected[:, None], 5, axis=1))
