"""Invented arrays/paths only. No real cohort or checkpoint inputs."""

import copy
import io
import os

import numpy as np
import pytest

from dmvmat_ecg.phase3b_eval.signals import prefix_only
from dmvmat_ecg.phase4.safety import digest, new_json, sha
from dmvmat_ecg.phase4c_confirm.contract import (
    authenticate,
    consume_labels,
    save_array,
    verify_manifest,
)
from dmvmat_ecg.phase4c_confirm.scope import capability
from dmvmat_ecg.phase5_confirm.contract import RAW, ROOT, Scope, roster
from dmvmat_ecg.phase5_confirm.provider import Provider
from dmvmat_ecg.phase5_confirm.statistics import analyze, verdict


def test_roster_complete_and_only_source_only_aliases():
    r = roster()
    assert len(r) == 270 and len(set(r.values())) == 190
    assert len({v for k, v in r.items() if k.startswith("a/")}) == 10
    assert len({v for k, v in r.items() if not k.startswith("a/")}) == 180


@pytest.mark.parametrize(
    "lower,valid,bounds,expected",
    [
        (0.001, 10000, True, "success"),
        (0.0, 10000, True, "not successful"),
        (-0.01, 10000, True, "not successful"),
        (0.001, 9499, True, "invalid/incomplete"),
        (0.001, 10000, False, "invalid/incomplete"),
    ],
)
def test_fixed_primary_rule(lower, valid, bounds, expected):
    assert verdict(0.01, dict(interval=[lower, 0.02] if bounds else None, valid=valid)) == expected


def test_no_test_capability_before_all180_replays():
    s = Scope()
    for count in (0, 108, 179):
        with pytest.raises(PermissionError):
            s.predictions(count)
    p = RAW / "training/georgia/g1/INVENTED.hea"
    s.raw_paths = {p}
    with capability("prefix"):
        with pytest.raises(PermissionError):
            s.check("open", (str(p), "rb", os.O_RDONLY))
    s.predictions(180)
    with capability("prefix"):
        s.check("open", (str(p), "rb", os.O_RDONLY))
    with capability("annotation"):
        with pytest.raises(PermissionError):
            s.check("open", (str(p), "rb", os.O_RDONLY))
    with pytest.raises(PermissionError):
        s.outcomes()


@pytest.mark.parametrize("stage", ["replay", "predict", "score"])
@pytest.mark.parametrize(
    "path",
    [
        "data/raw/cinc2021_v1_0_3/training/cpsc_2018/g1/INVENTED.mat",
        "private/phase4c_confirm_run/ANALYSIS.json",
        "data/raw/ptbxl_v1_0_3/records100/INVENTED_fold10.dat",
    ],
)
def test_cpsc_test_and_fold10_never_permitted(stage, path):
    s = Scope()
    s.stage = stage
    with pytest.raises(PermissionError):
        s.check("open", (str(ROOT / path), "rb", os.O_RDONLY))


def test_provider_refuses_before_any_membership_open(monkeypatch):
    import dmvmat_ecg.phase5_confirm.provider as module

    monkeypatch.setattr(
        module, "project_json", lambda *a: pytest.fail("Premature private membership read")
    )
    with pytest.raises(PermissionError):
        Provider(Scope())


def test_physical_prefix_does_not_read_diagnosis_comments():
    raw = (
        b"INVENTED 12 500 5000\n"
        + b"".join(b"INVENTED.mat 16 1000/mV 16 0 0 0 0 I\n" for _ in range(12))
        + b"#Dx: NOT_TO_BE_READ\n"
    )
    source = io.BytesIO(raw)
    prefix = prefix_only(source)
    assert b"#Dx" not in prefix
    assert source.read() == b"#Dx: NOT_TO_BE_READ\n"


def sealed(tmp):
    save_array(tmp / "p.npy", np.array([[0.1, 0.9], [0.9, 0.1]], np.float32))
    r = {"one": "physical"}
    m = dict(roster=r, predictions={"physical": dict(file="p.npy", sha256=sha(tmp / "p.npy"))})
    new_json(tmp / "PREDICTION_MANIFEST.json", m)
    new_json(
        tmp / "PREDICTION_SEAL.json",
        dict(manifest_sha256=sha(tmp / "PREDICTION_MANIFEST.json"), manifest_digest=digest(m)),
    )
    return r


def test_labels_once_after_complete_verified_seal(tmp_path):
    r = sealed(tmp_path)
    seen = []
    consume_labels(tmp_path, lambda m: seen.append(m), r, 2)
    with pytest.raises(FileExistsError):
        consume_labels(tmp_path, lambda m: seen.append(m), r, 2)
    assert len(seen) == 1


def test_tampered_prediction_never_unlocks_labels(tmp_path):
    r = sealed(tmp_path)
    (tmp_path / "p.npy").write_bytes(b"changed")
    with pytest.raises(PermissionError):
        consume_labels(tmp_path, lambda _: pytest.fail("Labels read"), r, 2)
    assert not (tmp_path / "OUTCOMES_CONSUMED.json").exists()


def test_interrupted_outcome_attempt_stays_consumed(tmp_path):
    r = sealed(tmp_path)

    def interrupted(_):
        raise RuntimeError("INVENTED INTERRUPTION")

    with pytest.raises(RuntimeError):
        consume_labels(tmp_path, interrupted, r, 2)
    with pytest.raises(FileExistsError):
        consume_labels(tmp_path, interrupted, r, 2)


def test_missing_logical_cell_rejected(tmp_path):
    r = sealed(tmp_path)
    r["missing"] = "other"
    with pytest.raises(PermissionError):
        verify_manifest(tmp_path, r, 2)


def test_authentication_rejects_one_ulp_before_test_gate():
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
    old = dict.fromkeys(fields, "INVENTED")
    old["normalization"] = {"mean": 0, "std": 1}
    p = {k: np.full((2, 2), 0.5, np.float32) for k in ("A", "B")}
    q = copy.deepcopy(p)
    q["B"][0, 0] = np.nextafter(q["B"][0, 0], np.float32(1))
    with pytest.raises(RuntimeError):
        authenticate(old, p, old, q, old["normalization"])
    assert Scope().stage == "replay"


def test_synthetic_full_analysis_and_report(tmp_path, monkeypatch):
    from dmvmat_ecg.phase5_confirm import report

    rng = np.random.default_rng(19)
    y = np.tile([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0], [1.0, 1.0]], (6, 1))
    obs = np.ones_like(y, bool)
    r = roster()
    preds = {key: rng.random(y.shape).astype(np.float32) for key in dict.fromkeys(r.values())}
    a = analyze(y, obs, preds, r, np.arange(len(y)), tmp_path, draws=30, minimum=20)
    assert len(a["methods"]) == 9 and len(a["per_physical"]) == 190 and len(a["per_seed"]) == 30
    assert a["primary"]["verdict"] == "invalid/incomplete"
    for b in (64, 128, 256):
        assert a["methods"][f"a/B{b}"] == a["methods"]["a/B64"]
    manual = np.mean(
        [a["per_physical"][r[k]]["macro"]["AP"] for k in r if k.startswith("c/B128/")]
    ) - np.mean([a["per_physical"][r[k]]["macro"]["AP"] for k in r if k.startswith("b/B128/")])
    assert a["primary"]["estimate"] == pytest.approx(manual)
    (tmp_path / "docs/phase5").mkdir(parents=True)
    monkeypatch.setattr(report, "ROOT", tmp_path)
    report.write(
        a,
        dict(
            protocol=dict(commit="INVENTED", sha256="INVENTED"),
            manifest_path="INVENTED",
            manifest_sha256="INVENTED",
            manifest_timestamp="INVENTED",
            replay_matches=180,
        ),
    )
    text = (tmp_path / "docs/phase5/PHASE5_GEORGIA_RESULTS.md").read_text()
    assert "NPV" in text and "single Phase5" in text
    assert (tmp_path / "figures/phase5/phase5_georgia_test_per_seed_ap_differences.svg").exists()
    with pytest.raises(FileExistsError):
        report.write(a, {})


def test_validity_fails_before_metric_or_bootstrap(tmp_path):
    y = np.zeros((24, 2))
    obs = np.ones_like(y, bool)
    with pytest.raises(PermissionError):
        analyze(y, obs, {}, roster(), np.arange(24), tmp_path, draws=30, minimum=20)
    assert not (tmp_path / "ANALYSIS.json").exists()
