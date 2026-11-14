from dataclasses import replace

import numpy as np
import pytest

from dmvmat_ecg.phase2.contracts import MODULES
from dmvmat_ecg.phase2.evidence import logical_vote, permutation_mmd
from dmvmat_ecg.phase3b.evidence import RepresentationBank, analyze, configuration


def banks(*, n=64, target_shift=0, target_signal=1, constant=False, insufficient=False):
    rng = np.random.default_rng(7)
    y = np.column_stack([np.arange(n) % 2, (np.arange(n) // 2) % 2]).astype(float)
    obs = np.ones_like(y, dtype=bool)
    source, target = {}, {}
    for name in MODULES:
        noise = rng.normal(0, 0.4, (n, 3))
        signal = np.column_stack([y, y.sum(1)]) * 2
        source[name] = noise + signal
        target[name] = noise + signal * target_signal + target_shift
        if constant:
            source[name][:] = 0
            target[name][:] = 0
    s = RepresentationBank(
        source,
        y,
        obs,
        tuple(f"s{i}" for i in range(n)),
        "source_train",
        "patient",
        "known",
        "source",
        "checkpoint",
        "norm",
    )
    ty = y.copy()
    if insufficient:
        ty[:] = 0
    t = RepresentationBank(
        target,
        ty,
        obs,
        tuple(f"t{i}" for i in range(n)),
        "G",
        "record_cluster",
        "unknown",
        "G",
        "checkpoint",
        "norm",
    )
    return s, t


def test_registered_evidence_settings():
    c = configuration()
    assert c.permutations == 999 and c.bootstraps == 4999
    assert c.minimum == 10 and c.mmd_cap == c.source_class_cap == 256
    assert c.mmd_seed == 78301 and c.geometry_seed == 78401
    assert c.source_d_floor == 0.1 and c.uncertainty_cutoff == 0.75


@pytest.mark.parametrize("shift", [0, 5])
def test_full_geometry_monte_carlo_and_holm(shift):
    from dmvmat_ecg.phase2.evidence import holm

    s, t = banks(n=32, target_shift=shift)
    result = analyze(s, t).report
    assert result["configuration"]["bootstraps"] == 4999
    modules = result["modules"]
    expected = holm([modules[n]["mmd"]["p"] if modules[n]["mmd"]["valid"] else 1 for n in MODULES])
    assert [modules[n]["mmd"]["q"] for n in MODULES] == expected.tolist()
    assert all(g["valid"] for m in modules.values() for g in m["geometry"])
    if shift:
        assert min(m["mmd"]["q"] for m in modules.values()) <= 0.05
        assert set(result["mask"].values()) == {"SHARE"}


def test_full_monte_carlo_pvalue_and_identical_control():
    source, target = banks(n=24)
    e = permutation_mmd(source.values["view_1"], target.values["view_1"], config=configuration())
    assert e["valid"]
    assert e["p"] == (1 + sum(v >= e["statistic"] for v in e["null"])) / 1000
    assert e["p"] > 0.05


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(),
        dict(target_shift=5),
        dict(target_signal=-1),
        dict(constant=True),
        dict(insufficient=True),
    ],
)
def test_policy_reproducibility_and_controls(kwargs):
    s, t = banks(**kwargs)
    a, b = analyze(s, t, fixture=True), analyze(s, t, fixture=True)
    assert a.payload == b.payload
    r = a.report
    assert set(r["mask"]) == set(MODULES)
    assert r["patient_linkage"] == "unknown" and "target_patients" not in r
    for module in r["modules"].values():
        for g in module["geometry"]:
            if g["valid"]:
                assert g["quantile"] == pytest.approx(1 - 0.05 / 14)
    if kwargs.get("constant") or kwargs.get("insufficient"):
        assert set(r["mask"].values()) == {"SHARE"}
        assert all(m["abstained_labels"] == 2 for m in r["modules"].values())
    elif not kwargs:
        assert set(r["mask"].values()) == {"SHARE"}
    elif kwargs.get("target_shift"):
        assert set(r["mask"].values()) == {"SHARE"}
        # Nineteen fixture permutations cannot resolve a seven-test Holm rejection.
        # The full 999-permutation test above verifies rejection at production resolution.
        assert all(m["mmd"]["q"] >= 0.05 for m in r["modules"].values())
    elif kwargs.get("target_signal") == -1:
        assert "ADAPT" in r["mask"].values()


def test_unknown_evidence_and_wrong_lineage():
    s, t = banks()
    obs = np.zeros_like(t.observed)
    t = replace(t, y=np.full_like(t.y, np.nan), observed=obs)
    result = analyze(s, t, fixture=True).report
    assert result["target_unknown_counts"] == [64, 64]
    assert set(result["mask"].values()) == {"SHARE"}
    with pytest.raises(PermissionError):
        analyze(s, replace(t, checkpoint_hash="other"), fixture=True)
    with pytest.raises(PermissionError):
        replace(t, role="O")
    with pytest.raises(ValueError):
        replace(t, patient_linkage="known")


def test_invalid_geometry_never_votes_and_mmd_not_a_loss():
    cfg = configuration()
    assert not logical_vote(dict(valid=False), dict(valid=True, q=0), cfg)
    assert logical_vote(dict(valid=True, upper=0.7, retention=0.6), dict(valid=False), cfg)
    assert not logical_vote(dict(valid=True, upper=0.9, retention=0.6), dict(valid=False), cfg)
