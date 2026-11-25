"""Invented arrays/metadata only: no cohort or source-checkpoint inputs."""

import copy
import itertools

import numpy as np
import pytest
from scipy.io import savemat
from sklearn.metrics import average_precision_score, roc_auc_score

from dmvmat_ecg.phase3b_eval.statistics import bootstrap_rankings, shared_weights
from dmvmat_ecg.phase4.safety import digest, new_json, sha
from dmvmat_ecg.phase4c_confirm.contract import (
    authenticate,
    consume_labels,
    gate_sequence,
    roster,
    save_array,
    verify_manifest,
)
from dmvmat_ecg.phase4c_confirm.provider import signal_only_mat
from dmvmat_ecg.phase4c_confirm.scope import MEMBERSHIPS, RAW, RUN, Scope, capability
from dmvmat_ecg.phase4c_confirm.statistics import aggregate, measures, strict_mean


def ci(success=True, valid=10000):
    return dict(estimate=0.1, CI95=dict(interval=[0.01 if success else -0.1, 0.2], valid=valid))


def test_roster():
    r = roster()
    assert len(r) == 135 and len(set(r.values())) == 111
    assert len([k for k in set(r.values()) if not k.startswith("M1/")]) == 108
    assert len({v for k, v in r.items() if k.startswith("M1/")}) == 3


@pytest.mark.parametrize("bits", list(itertools.product((False, True), repeat=3)))
def test_all_sequence_gates(bits):
    rows = gate_sequence(dict(zip(("primary", "S1", "S2"), map(ci, bits), strict=True)))
    assert [r["eligible"] for r in rows] == [True, bits[0], bits[0] and bits[1]]
    for i, r in enumerate(rows):
        assert r["status"] == (
            "descriptive only" if not r["eligible"] else "success" if bits[i] else "not successful"
        )


@pytest.mark.parametrize("change", ["support", "missing", "zero"])
def test_gate_cannot_open(change):
    first = ci()
    if change == "support":
        first["CI95"]["valid"] = 9499
    if change == "missing":
        first["CI95"]["interval"] = None
    if change == "zero":
        first["CI95"]["interval"][0] = 0
    rows = gate_sequence(dict(primary=first, S1=ci(), S2=ci()))
    assert rows[0]["status"] != "success"
    assert rows[1]["status"] == rows[2]["status"] == "descriptive only"


def auth_fixture():
    fields = (
        "source_seed",
        "draw_seed",
        "budget",
        "updates",
        "mask_index",
        "fit_seed",
        "source_sha256",
        "source_state_hash",
        "state_hash",
        "history",
        "drift",
        "final_training_loss",
    )
    r = dict.fromkeys(fields, "fixture")
    r.update(history=[1.0, 0.5], drift={"a": 0.1}, normalization={"means": [0]})
    arrays = {k: np.array([[0.0, 0.2], [0.7, 0.8]], dtype=np.float32) for k in ("A", "B")}
    return r, arrays


def test_auth_valid_and_early_norm_binding():
    r, a = auth_fixture()
    old = copy.deepcopy(r)
    del old["normalization"]
    authenticate(old, a, r, a, r["normalization"])


@pytest.mark.parametrize("role", ["A", "B"])
@pytest.mark.parametrize("kind", ["ulp", "signedzero", "dtype", "shape"])
def test_byte_auth_rejects(role, kind):
    r, a = auth_fixture()
    b = copy.deepcopy(a)
    if kind == "ulp":
        b[role][0, 1] = np.nextafter(b[role][0, 1], np.float32(1))
    if kind == "signedzero":
        b[role][0, 0] = -0.0
    if kind == "dtype":
        b[role] = b[role].astype(np.float64)
    if kind == "shape":
        b[role] = b[role].reshape(1, 4)
    with pytest.raises(RuntimeError):
        authenticate(r, a, r, b, r["normalization"])


@pytest.mark.parametrize(
    "field", ["state_hash", "history", "fit_seed", "source_sha256", "drift", "normalization"]
)
def test_auth_recipe(field):
    r, a = auth_fixture()
    b = copy.deepcopy(r)
    b[field] = "changed"
    with pytest.raises(RuntimeError):
        authenticate(r, a, b, a, r["normalization"])


def test_metrics_undefined_precision_and_npv_orientation():
    y = np.array([[0, 1], [1, 0], [0, 1], [1, 0.0]])
    p = np.array([[0.1, 0.7], [0.9, 0.2], [0.8, 0.8], [0.2, 0.4]])
    r = measures(y, p, np.ones_like(y, bool))
    assert r["per_label"][0]["confusion"] == dict(TP=1, TN=1, FP=1, FN=1)
    assert r["per_label"][0]["NPV"] == 0.5
    assert r["per_label"][1]["F1"] == 1
    assert r["per_label"][0]["AP"] == average_precision_score(y[:, 0], p[:, 0])
    none = measures(y, np.zeros_like(y), np.ones_like(y, bool))
    assert none["macro"]["precision"] is None
    assert none["macro"]["F1"] == 0
    assert none["macro"]["NPV"] == 0.5
    assert strict_mean([1, None]) is None
    assert aggregate([r, none])["macro"]["precision"] is None


def test_unknowns_never_negative():
    y = np.array([[1, 0], [0, 1], [np.nan, 0.0]])
    obs = np.isfinite(y)
    r = measures(y, np.full_like(y, 0.5), obs)
    assert r["per_label"][0]["unknown"] == 1
    assert r["per_label"][0]["observed"] == 2
    with pytest.raises(ValueError):
        measures(y, np.full_like(y, 0.5), np.ones_like(obs))


def test_bootstrap_exact_ties_and_group_broadcast():
    y = np.tile([[0, 1], [1, 0], [1, 1], [0, 0]], (3, 1)).astype(float)
    p = np.tile([[0.2, 0.8], [0.7, 0.8], [0.7, 0.2], [0.2, 0.1]], (3, 1))
    weights = shared_weights(np.repeat(np.arange(6), 2), draws=25)
    assert np.array_equal(weights[:, ::2], weights[:, 1::2])
    boot = bootstrap_rankings(y, p, np.ones_like(y, bool), weights)
    for i, w in enumerate(weights):
        for j in range(2):
            if (w @ y[:, j]) and (w @ (1 - y[:, j])):
                assert boot["AP"][i, j] == pytest.approx(
                    average_precision_score(y[:, j], p[:, j], sample_weight=w)
                )
                assert boot["AUROC"][i, j] == pytest.approx(
                    roc_auc_score(y[:, j], p[:, j], sample_weight=w)
                )


def manifest_fixture(tmp_path):
    save_array(tmp_path / "p.npy", np.array([[0.1, 0.7], [0.8, 0.2]], np.float32))
    mapping = {"logical": "physical"}
    m = dict(
        roster=mapping, predictions={"physical": dict(file="p.npy", sha256=sha(tmp_path / "p.npy"))}
    )
    new_json(tmp_path / "PREDICTION_MANIFEST.json", m)
    new_json(
        tmp_path / "PREDICTION_SEAL.json",
        dict(manifest_sha256=sha(tmp_path / "PREDICTION_MANIFEST.json"), manifest_digest=digest(m)),
    )
    return mapping


def test_manifest_and_single_use(tmp_path):
    r = manifest_fixture(tmp_path)
    assert verify_manifest(tmp_path, r, 2)["roster"] == r
    called = []
    assert consume_labels(tmp_path, lambda m: called.append(m) or 42, r, 2) == 42
    with pytest.raises(FileExistsError):
        consume_labels(tmp_path, lambda m: called.append(m), r, 2)
    assert len(called) == 1


def test_interruption_is_consumed(tmp_path):
    r = manifest_fixture(tmp_path)

    def fail(_):
        raise RuntimeError("invented interrupted annotation loader")

    with pytest.raises(RuntimeError):
        consume_labels(tmp_path, fail, r, 2)
    with pytest.raises(FileExistsError):
        consume_labels(tmp_path, fail, r, 2)


@pytest.mark.parametrize("kind", ["array", "manifest", "seal", "roster"])
def test_frozen_manifest_tampering(tmp_path, kind):
    r = manifest_fixture(tmp_path)
    file = {
        "array": "p.npy",
        "manifest": "PREDICTION_MANIFEST.json",
        "seal": "PREDICTION_SEAL.json",
    }.get(kind)
    if file:
        (tmp_path / file).write_bytes(b"tampered")
    else:
        r = {"wrong": "physical"}
    with pytest.raises((ValueError, PermissionError)):
        verify_manifest(tmp_path, r, 2)


def test_scope_no_test_until_gate():
    s = Scope()
    for p in (MEMBERSHIPS, RAW / "invented.hea", RAW / "invented.mat"):
        with pytest.raises(PermissionError):
            s.check("open", (str(p), "r", 0))
    with pytest.raises(PermissionError):
        s.predictions(107)
    s.predictions(108)
    s.check("open", (str(MEMBERSHIPS), "r", 0))
    p = RAW / "invented.hea"
    s.raw_paths = {p}
    with pytest.raises(PermissionError):
        s.check("open", (str(p), "r", 0))
    with capability("prefix"):
        s.check("open", (str(p), "r", 0))
    with capability("annotation"):
        with pytest.raises(PermissionError):
            s.check("open", (str(p), "r", 0))
    with pytest.raises(PermissionError):
        s.check("open", (str(MEMBERSHIPS), "w", 1))
    s.check("open", (str(RUN / "fixture.json"), "w", 1))


def test_signal_mat_only(tmp_path):
    p = tmp_path / "invented.mat"
    savemat(p, {"val": np.zeros((12, 1000))})
    signal_only_mat(p)
    savemat(p, {"val": np.zeros((12, 1000)), "outcome": np.ones(2)})
    with pytest.raises(PermissionError):
        signal_only_mat(p)


def test_array_exclusive_and_permissions(tmp_path):
    p = tmp_path / "p.npy"
    save_array(p, np.zeros((2, 2), np.float32))
    original = p.read_bytes()
    assert p.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        save_array(p, np.ones((2, 2)))
    assert p.read_bytes() == original


def test_synthetic_end_to_end_analysis_and_reporting(tmp_path, monkeypatch):
    from dmvmat_ecg.phase4c_confirm import report
    from dmvmat_ecg.phase4c_confirm.statistics import analyze

    rng = np.random.default_rng(86)
    y = np.tile([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]], (6, 1))
    mapping = roster()
    predictions = {
        k: rng.random(y.shape).astype(np.float32) for k in dict.fromkeys(mapping.values())
    }
    result = analyze(
        y,
        np.ones_like(y, bool),
        predictions,
        mapping,
        np.arange(len(y)),
        tmp_path,
        draws=25,
        minimum=20,
    )
    assert len(result["methods"]) == 15
    assert len(result["per_physical"]) == 111
    assert len(result["comparisons"]) == 12
    assert result["confirmatory_sequence"][0]["status"] == "invalid"
    for b in (64, 128, 256):
        assert result["methods"][f"M1/B{b}"] == result["methods"]["M1/B64"]
    (tmp_path / "docs/phase4").mkdir(parents=True)
    monkeypatch.setattr(report, "ROOT", tmp_path)
    report.write(
        result,
        dict(
            protocol=dict(commit="INVENTED_COMMIT", sha256="INVENTED_HASH"),
            replay_matches=108,
            manifest_path="INVENTED_PATH",
            manifest_sha256="INVENTED_HASH",
            manifest_timestamp="INVENTED_TIMESTAMP",
        ),
    )
    assert (tmp_path / "figures/phase4/phase4c_macro_ap_vs_label_budget.svg").exists()
    text = (tmp_path / "docs/phase4/PHASE4C_RESULTS.md").read_text()
    assert "NPV" in text and "descriptive only" in text
    with pytest.raises(FileExistsError):
        report.write(result, {})


def test_synthetic_inference_bitwise_and_bn_unchanged():
    import io

    import torch

    from dmvmat_ecg.phase2.adaptation import state_hash
    from dmvmat_ecg.phase3b.models import fresh_model
    from dmvmat_ecg.phase4c_confirm.run import predict_state

    model = fresh_model(42).eval()
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    record = dict(
        source_seed=42,
        state_hash=state_hash(model),
        normalization=dict(
            mean=0.0, std=1.0, training_identity="INVENTED", origin="fresh-source-only"
        ),
    )
    x = np.random.default_rng(81).normal(size=(4, 12, 1000)).astype(np.float32)
    a = predict_state(stream.getvalue(), record, x)
    b = predict_state(stream.getvalue(), record, x)
    assert a.shape == (4, 2) and a.tobytes() == b.tobytes()
