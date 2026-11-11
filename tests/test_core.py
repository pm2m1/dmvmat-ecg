import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import wfdb

from dmvmat_ecg.data.store import CropDataset, Store
from dmvmat_ecg.evaluation.metrics import complete, ranking, thresholds_from_validation
from dmvmat_ecg.evaluation.state import Study
from dmvmat_ecg.labels.superclasses import CLASSES, encode, label_map
from dmvmat_ecg.models.networks import build, parameters
from dmvmat_ecg.preprocessing.signals import augment, canonical_leads, normalize, views
from dmvmat_ecg.utils.io import write_json

torch.set_num_threads(2)


@pytest.fixture
def config():
    return json.loads((Path(__file__).parents[1] / "configs/study.json").read_text())


def test_wfdb_orientation_frequency_leads(tmp_path):
    x = np.random.default_rng(10).normal(0, 0.1, (1000, 12))
    names = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    wfdb.wrsamp(
        "invented",
        fs=100,
        units=["mV"] * 12,
        sig_name=names,
        p_signal=x,
        fmt=["16"] * 12,
        write_dir=str(tmp_path),
    )
    signal, fields = wfdb.rdsamp(str(tmp_path / "invented"))
    assert signal.shape == (1000, 12) and fields["fs"] == 100
    assert signal[:, canonical_leads(fields["sig_name"])].T.shape == (12, 1000)


def test_label_mapping_likelihood_zero_multilabel():
    statements = pd.DataFrame(
        {"diagnostic": [1.0, 1.0, 0.0], "diagnostic_class": ["NORM", "MI", np.nan]},
        index=["NORM", "AMI", "SR"],
    )
    mapping = label_map(statements)
    assert encode("{'NORM':100,'AMI':0,'SR':100}", mapping).tolist() == [1, 1, 0, 0, 0]
    assert encode({"SR": 100}, mapping).sum() == 0
    assert CLASSES == ("NORM", "MI", "STTC", "CD", "HYP")


def test_lead_ordering_and_rejection():
    assert canonical_leads(
        ["II", "I", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    )[:2] == [1, 0]
    with pytest.raises(ValueError):
        canonical_leads(["I"] * 12)


@pytest.mark.parametrize("length", [250, 1000])
def test_deterministic_view_construction(length):
    x = np.random.default_rng(11).normal(size=(2, 12, length)).astype("float32")
    a = views(x)
    b = views(x)
    assert a[1].shape == (2, 3, length) and a[2].shape == (2, 12, 99 if length == 250 else 396)
    for left, right in zip(a, b, strict=True):
        assert np.array_equal(left, right) and np.isfinite(left).all()


def test_training_only_normalization_and_augmentation():
    state = {"mean": 0.2, "std": 0.5}
    before = state.copy()
    x = np.zeros((12, 1000))
    assert np.isfinite(normalize(x + 1e3, state)).all() and state == before
    with pytest.raises(PermissionError):
        augment(x, np.random.default_rng(1), training=False, state=state)
    np.testing.assert_array_equal(
        augment(x, np.random.default_rng(1), training=True, state=state),
        augment(x, np.random.default_rng(1), training=True, state=state),
    )


@pytest.mark.parametrize(
    "name,count",
    [
        ("xresnet101", 1872261),
        ("flat_cnn", 206117),
        ("multiview", 212261),
        ("multiview_aug", 212261),
    ],
)
def test_model_dimensions_gradients_serialization(name, count, tmp_path):
    m = build(name)
    n = 250 if name == "xresnet101" else 1000
    x = torch.randn(4, 12, n)
    inputs = {"wave": x}
    if name.startswith("multiview"):
        _, r, f = views(x.numpy())
        inputs.update(rhythm=torch.tensor(r), spectrum=torch.tensor(f))
    out = m(**inputs)
    assert out.shape == (4, 5) and parameters(m) == count
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out, torch.zeros(4, 5))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    m.eval()
    torch.save(m.state_dict(), tmp_path / "model.pt")
    other = build(name)
    other.load_state_dict(torch.load(tmp_path / "model.pt", weights_only=True))
    other.eval()
    torch.testing.assert_close(m(**inputs), other(**inputs), rtol=0, atol=0)


def test_capacity_and_seven_module_ownership():
    m = build("multiview")
    flat = build("flat_cnn")
    assert abs(parameters(m) - parameters(flat)) / parameters(m) < 0.05
    owners = {**m.selectable_modules(), "prediction_head": m.prediction_head}
    lists = [{id(p) for p in module.parameters()} for module in owners.values()]
    assert set.union(*lists) == {id(p) for p in m.parameters()}
    assert sum(map(len, lists)) == len(set.union(*lists))


def test_fixed_seeds_and_crop_roles():
    a = build("flat_cnn", 101)
    b = build("flat_cnn", 101)
    assert all(
        torch.equal(x, y)
        for x, y in zip(a.state_dict().values(), b.state_dict().values(), strict=True)
    )
    x = np.zeros((2, 12, 1000), dtype="float32")
    y = np.zeros((2, 5))
    stats = {"mean": 0.0, "std": 1.0}
    ds = CropDataset(x, y, [1, 2], stats, training=False, window=250)
    assert len(ds) == 14
    with pytest.raises(PermissionError):
        CropDataset(x, y, [1, 2], stats, training=False, augmentation=True)
    full = CropDataset(x, y, [1, 2], stats, training=True, window=1000)
    assert len(full) == 2 and full[0][0]["wave"].shape == (12, 1000)


def test_exact_metrics_and_thresholds():
    y = np.tile([1, 0, 1, 0], (5, 1)).T
    p = np.tile([0.9, 0.8, 0.7, 0.1], (5, 1)).T
    result = ranking(y, p)
    assert result["macro_ap"] == pytest.approx(5 / 6) and result["macro_auroc"] == pytest.approx(
        0.75
    )
    complete_result = complete(y, p, [0.5] * 5)
    assert complete_result["labels"]["NORM"]["confusion_matrix"] == [[1, 1], [0, 2]]
    assert len(thresholds_from_validation(y, p)) == 5
    assert ranking(np.zeros((4, 5)), p)["macro_auroc"] is None


def fixture_study(tmp_path, config):
    write_json(tmp_path / "configs/study.json", config)
    return Study(tmp_path)


def test_phase_lock_and_attempt_budget(tmp_path, config):
    s = fixture_study(tmp_path, config)
    with pytest.raises(PermissionError):
        Store(s).load("test")
    s.set("phase", "development")
    with pytest.raises(PermissionError):
        s.reserve("made-up", 101, "full", config)
    for name in ("xresnet101", "flat_cnn", "multiview"):
        trial = s.reserve(name, 17, "smoke", config)
        s.finish(trial, "successful", {})
    for name in config["models"]:
        for seed in config["seeds"]:
            trial = s.reserve(name, seed, "full", config)
            s.finish(trial, "successful", {})
    s.lock([])
    with pytest.raises(PermissionError):
        s.reserve("flat_cnn", 101, "full", config)
    with s.evaluate_once("user-authorized-single-final-PTBXL-evaluation"):
        s.require("test")
    with pytest.raises(PermissionError):
        s.require("test")
    with pytest.raises(PermissionError):
        with s.evaluate_once("user-authorized-single-final-PTBXL-evaluation"):
            pass
    assert len(s.runs()) == 11


def test_failures_count_fresh_retry_and_immutability(tmp_path, config):
    s = fixture_study(tmp_path, config)
    s.set("phase", "development")
    first = s.reserve("xresnet101", 101, "full", config)
    s.finish(first, "failed", {"reason": "invented failure"})
    second = s.reserve("xresnet101", 101, "full", config)
    assert second != first
    s.finish(second, "successful", {})
    with pytest.raises(PermissionError):
        s.reserve("xresnet101", 101, "full", config)
    assert len(s.runs()) == 2 and s.runs()[0]["status"] == "failed"


def test_changed_source_revokes_evaluation(tmp_path, config):
    s = fixture_study(tmp_path, config)
    s.set("phase", "development")
    for name in config["models"]:
        for seed in config["seeds"]:
            s.finish(s.reserve(name, seed, "full", config), "successful", {})
    s.lock([])
    config["epochs"] = 49
    write_json(tmp_path / "configs/study.json", config)
    with pytest.raises(PermissionError):
        with s.evaluate_once("user-authorized-single-final-PTBXL-evaluation"):
            pass
