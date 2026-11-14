"""Fictional headers, labels, probabilities and temporary access markers only."""

import io
from datetime import date

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from dmvmat_ecg.phase3b.jobs import atomic_new, file_hash
from dmvmat_ecg.phase3b.protocol import BUDGETS, roster
from dmvmat_ecg.phase3b_eval.access import (
    AUTH_STATEMENT,
    COMMIT,
    LOCK,
    MEMBERSHIPS,
    REAL_CODE,
    Session,
    validate_authorization,
)
from dmvmat_ecg.phase3b_eval.analysis import analyze
from dmvmat_ecg.phase3b_eval.signals import decode_labels, physical_metadata, prefix_only
from dmvmat_ecg.phase3b_eval.statistics import (
    bootstrap_rankings,
    confidence,
    contrast,
    metrics,
    shared_weights,
)
from dmvmat_ecg.phase3b_real.boundary import BASE_HASH, MAPPING_HASH


def header_prefix(samples=5000):
    leads = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
    return (
        f"fixture 12 500 {samples}\n"
        + "".join(f"fixture.mat 16 1000/mV 16 0 0 0 0 {n}\n" for n in leads)
    ).encode()


def test_header_reads_exactly_physical_prefix():
    prefix = header_prefix()
    stream = io.BytesIO(prefix + b"#Dx: THIS_MUST_NOT_BE_READ\n")
    assert prefix_only(stream) == prefix
    assert stream.tell() == len(prefix)


@pytest.mark.parametrize("samples", [4999, 5000, 7500])
def test_redacted_header_metadata(tmp_path, samples):
    original = tmp_path / "original.hea"
    redacted = tmp_path / "safe" / "fixture.hea"
    original.write_bytes(header_prefix(samples) + b"#Dx: SECRET\n#Age: SECRET\n")
    meta = physical_metadata(original, redacted)
    assert meta["fs"] == 500 and meta["samples"] == samples
    assert meta["duration"] == samples / 500 and meta["order"] == list(range(12))
    assert meta["factors_to_mV"] == [1.0] * 12
    assert b"SECRET" not in redacted.read_bytes() and "dx" not in meta


def test_labels_missing_are_not_negatives():
    mapping = {"entries": {"1": {"label": "CD"}, "2": {"label": "STTC"}, "3": {"label": None}}}
    assert decode_labels(b"#Dx: 1,2\n", mapping) == ([1, 1], [True, True])
    assert decode_labels(b"#Dx: 3\n", mapping) == ([0, 0], [True, True])
    y, obs = decode_labels(b"#No diagnosis annotation\n", mapping)
    assert np.isnan(y).all() and obs == [False, False]


def test_single_use_outcomes_and_full_seal(tmp_path):
    gate = Session(tmp_path / "eval", {"fixture": "one"}, anchor=tmp_path / "anchor")
    called = []
    with pytest.raises(PermissionError):
        gate.consume(lambda: called.append(True))
    gate.begin()
    with pytest.raises(FileExistsError):
        Session(gate.folder, gate.identity, anchor=gate.anchor).begin()
    p = tmp_path / "prediction.npy"
    atomic_new(p, b"fictional prediction bytes")
    item = dict(path=str(p), sha=file_hash(p))
    with pytest.raises(PermissionError):
        gate.seal(["a", "b"], {"a": item}, {}, "membership")
    sealed = gate.seal(["a", "b"], {"a": item, "b": item}, {"model": "frozen"}, "membership")
    assert len(sealed["seal_hash"]) == 64
    assert gate.consume(lambda: called.append(True)) is None
    assert called == [True]
    with pytest.raises(PermissionError):
        gate.consume(lambda: called.append(True))
    with pytest.raises(FileExistsError):
        Session(tmp_path / "alternate", gate.identity, anchor=gate.anchor).begin()
    assert called == [True]


def test_failure_burns_access_without_retry(tmp_path):
    gate = Session(tmp_path, {"fixture": "failure"})
    gate.begin()
    p = tmp_path / "p"
    p.write_bytes(b"fixture")
    gate.seal(["a"], {"a": dict(path=str(p), sha=file_hash(p))}, {}, "ids")

    def fail():
        raise OSError("fictional I/O failure")

    with pytest.raises(OSError):
        gate.consume(fail)
    with pytest.raises(PermissionError):
        gate.consume(lambda: None)


def test_tamper_prevents_outcome_loader(tmp_path):
    gate = Session(tmp_path, {"fixture": "tamper"})
    gate.begin()
    p = tmp_path / "p"
    p.write_bytes(b"fixture")
    gate.seal(["a"], {"a": dict(path=str(p), sha=file_hash(p))}, {}, "ids")
    p.write_bytes(b"changed")
    called = []
    with pytest.raises(PermissionError):
        gate.consume(lambda: called.append(True))
    assert not called and not (tmp_path / "O_OUTCOME_ACCESS_V1.json").exists()


@pytest.mark.parametrize("change", ["none", "lock", "extra"])
def test_human_authorization_schema(change):
    a = dict(
        protocol_owner="Fictional Reviewer",
        date=date.today().isoformat(),
        authorization=AUTH_STATEMENT,
        lock_hash=LOCK,
        memberships_hash=MEMBERSHIPS,
        frozen_base_code_identity=BASE_HASH,
        real_implementation_identity=REAL_CODE,
        authorized_commit=COMMIT,
        label_mapping_sha256=MAPPING_HASH,
    )
    if change == "lock":
        a["lock_hash"] = "different"
    if change == "extra":
        a["invented_approval"] = True
    if change == "none":
        validate_authorization(a)
    else:
        with pytest.raises(PermissionError):
            validate_authorization(a)


def arrays():
    y = np.array([[0, 1], [1, 0], [1, 1], [0, 0], [1, 0], [0, 1]], dtype=float)
    p = np.array([[0.1, 0.8], [0.8, 0.3], [0.8, 0.8], [0.3, 0.2], [0.6, 0.3], [0.3, 0.8]])
    return y, p, np.ones_like(y, dtype=bool)


def test_metrics_exact_known_arrays():
    y, p, obs = arrays()
    r = metrics(y, p, obs)
    for j in range(2):
        assert r["per_label"][j]["AP"] == average_precision_score(y[:, j], p[:, j])
        assert r["per_label"][j]["AUROC"] == roc_auc_score(y[:, j], p[:, j])
        assert r["per_label"][j]["prevalence_AP_reference"] == 0.5
        assert sum(r["per_label"][j]["confusion"].values()) == 6
    assert r["macro"]["AP"] == 1


def test_only_log_loss_clips_and_unknowns_masked():
    y = np.array([[0, 1], [1, np.nan], [0, 0], [1, 1.0]])
    p = np.array([[1, 0], [0, 1], [0, 0.3], [1, 0.8]])
    obs = np.isfinite(y)
    r = metrics(y, p, obs)
    assert r["per_label"][1]["unknown"] == 1
    assert r["per_label"][0]["Brier"] == 0.5
    assert np.isfinite(r["per_label"][0]["log_loss"])
    assert r["per_label"][0]["AP"] == average_precision_score(y[:, 0], p[:, 0])


def test_group_weights_shared_and_deterministic():
    groups = ["a", "a", "b", "c"]
    w = shared_weights(groups, draws=200, seed=78701)
    assert np.array_equal(w, shared_weights(groups, 200, 78701))
    assert np.array_equal(w[:, 0], w[:, 1])


def test_bootstrap_ties_and_masks_match_sklearn_weighted():
    y, p, obs = arrays()
    obs[0, 1] = False
    weights = shared_weights(np.arange(len(y)), draws=100)
    result = bootstrap_rankings(y, p, obs, weights, chunk=17)
    for i, w in enumerate(weights):
        for j in range(2):
            use = obs[:, j]
            yy = y[use, j]
            ww = w[use]
            if (ww @ yy) > 0 and (ww @ (1 - yy)) > 0:
                assert result["AP"][i, j] == pytest.approx(
                    average_precision_score(yy, p[use, j], sample_weight=ww), abs=1e-12
                )
                assert result["AUROC"][i, j] == pytest.approx(
                    roc_auc_score(yy, p[use, j], sample_weight=ww), abs=1e-12
                )
            else:
                assert np.isnan(result["AP"][i, j])


def test_undefined_not_redrawn_and_minimum_rule():
    y = np.array([[0.0, 0.0], [1.0, 1.0]])
    w = shared_weights(["a", "b"], 100)
    r = bootstrap_rankings(y, y, np.ones_like(y, dtype=bool), w)
    assert len(r["AP"]) == 100 and np.isnan(r["AP"]).any()
    ci = confidence(r["AP"].mean(1), minimum_valid=95)
    assert ci["interval"] is None and ci["undefined"] > 0 and ci["no_redrawing"]


def test_equal_nine_episode_contrast_not_probability_ensemble():
    names = {name: name for name in roster()}
    points = {name: 0.0 for name in names}
    for name in names:
        if name.endswith("/selective_a"):
            points[name] = 0.8
        if name.endswith("/full_cost"):
            points[name] = 0.6
    assert contrast(names, points, budget=128, comparator="full_cost") == pytest.approx(0.2)


def test_complete_synthetic_analysis_roster(tmp_path):
    import json

    y, p, obs = arrays()
    names = {name: "one" for name in roster()}
    out = tmp_path / "aggregate"
    out.mkdir()
    result = analyze(
        y, obs, {"one": p}, names, np.arange(len(y)), tmp_path, out, draws=50, minimum_valid=1
    )
    assert len(result["episode"]) == 138
    for b in BUDGETS:
        assert result["comparisons"][f"B{b}/A-minus-full_cost"]["delta_AP"] == 0
    descriptive = json.loads((out / "NEGATIVE_TRANSFER_AND_RECOVERY.json").read_text())
    assert len(descriptive["negative_transfer"]) == 12
    assert len(descriptive["target_only_relative_reference"]) == 3


def test_complete_synthetic_report(tmp_path):
    import json

    from dmvmat_ecg.phase3b_eval.report import report

    y, p, obs = arrays()
    out = tmp_path / "results/phase3b_eval"
    out.mkdir(parents=True)
    folder = tmp_path / "private/phase3b_cpsc_v1"
    folder.mkdir(parents=True)
    result = analyze(
        y,
        obs,
        {"one": p},
        {name: "one" for name in roster()},
        np.arange(len(y)),
        folder,
        out,
        draws=50,
        minimum_valid=1,
    )
    source = tmp_path / "results/phase3b"
    source.mkdir()
    point = metrics(y, p, obs)
    (source / "SOURCE_VALIDATION_RESULTS.json").write_text(
        json.dumps(
            {
                str(s): dict(
                    report=dict(
                        source_stop_per_label=[
                            dict(AP=r["AP"], AUROC=r["AUROC"]) for r in point["per_label"]
                        ]
                    )
                )
                for s in (15101, 15102, 15103)
            }
        )
    )
    (folder / "MEMBERSHIPS.json").write_text(
        json.dumps(dict(source=dict(source_stop=[dict(y=row) for row in y.tolist()])))
    )
    (tmp_path / "docs/phase3").mkdir(parents=True)
    report(
        tmp_path,
        result,
        y,
        obs,
        dict(evaluated=6, excluded_short=0, truncated_first10=0),
        "fictional-seal",
        {"one": dict(seconds=0.1)},
    )
    text = (tmp_path / "docs/phase3/PHASE3B_LOCKED_EVALUATION_RESULTS.md").read_text()
    assert "Inconclusive" in text and "fictional-seal" in text
    assert len(list((out / "figures").glob("*.pdf"))) == 2
