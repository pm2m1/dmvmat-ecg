import copy
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest
import torch

from dmvmat_ecg.phase2.adaptation import state_hash
from dmvmat_ecg.phase2.contracts import MODULES
from dmvmat_ecg.phase3b.engine import fit_target, train_source, unpack
from dmvmat_ecg.phase3b.evidence import analyze, extract
from dmvmat_ecg.phase3b.jobs import RunLog, atomic_new, file_hash
from dmvmat_ecg.phase3b.protocol import METHODS, digest
from dmvmat_ecg.phase3b.roles import RoleData, array_hash
from dmvmat_ecg.phase3b.sealing import PredictionSeal

from .fixtures import decision_fixture, factory, fixture_data, source_fixture


def target(data, log, slot, source, method, **kwargs):
    return fit_target(
        data,
        log,
        slot,
        source,
        method,
        source_seed=15101,
        draw_seed=16101,
        budget=64,
        factory=factory,
        **kwargs,
    )


def test_fixed_runner_union_aliases_and_frozen_drift(tmp_path):
    torch.set_num_threads(4)
    data, counters, _ = fixture_data()
    source = source_fixture(data)
    checkpoint = state_hash(source.model)
    log = RunLog(tmp_path / "run.sqlite", data.namespace, jobs={m: {} for m in METHODS})
    all_share = dict.fromkeys(MODULES, "SHARE")
    decision = decision_fixture(source, data, all_share)
    head, h = target(data, log, "head_cost", source, "head_cost")
    selective, a = target(data, log, "selective_a", source, "selective_a", decision=decision)
    random, r = target(data, log, "random_a", source, "random_a", decision=decision)
    assert not h["alias"] and a["alias"] and r["alias"]
    assert h["artifact"] == a["artifact"] == r["artifact"]
    assert state_hash(head) == state_hash(selective) == state_hash(random)
    assert log.summary()["used"] == 1
    assert h["selected_update"] == h["completed_updates"] == 200
    assert set(h["drift"].values()) == {0.0}
    assert h["fitting_exposures"]["F"] > 0 and h["fitting_exposures"]["G"] > 0
    assert sum(h["fitting_exposures"].values()) == 200 * 32
    assert h["label_cost"]["records"] == 64
    assert all(op not in {"stop_target", "select_target"} for _, op in data.access_log)
    assert state_hash(source.model) == checkpoint
    assert counters["O"] == dict(features=0, outcomes=0)
    with pytest.raises(PermissionError):
        target(
            data,
            log,
            "full_cost",
            source,
            "selective_a",
            decision=replace(
                decision, payload=decision.payload.replace(b'"G_identity":"', b'"G_identity":"x')
            ),
        )


def test_all_adapt_alias_and_target_only(tmp_path):
    data, _, _ = fixture_data()
    source = source_fixture(data)
    log = RunLog(tmp_path / "run.sqlite", data.namespace, jobs={m: {} for m in METHODS})
    decision = decision_fixture(source, data, dict.fromkeys(MODULES, "ADAPT"))
    full, f = target(data, log, "full_cost", source, "full_cost")
    selective, a = target(data, log, "selective_a", source, "selective_a", decision=decision)
    assert a["alias"] and state_hash(full) == state_hash(selective)
    _, t = target(data, log, "target_only_cost", source, "target_only_cost")
    assert not t["alias"] and t["normalization"]["origin"] == "fresh-target-union"
    assert f["normalization"]["origin"] == "fresh-source-only"
    assert log.summary()["used"] == 2


def test_G_labels_influence_gradient_fit(tmp_path):
    data, _, arrays = fixture_data()
    source = source_fixture(data)
    log = RunLog(tmp_path / "one.sqlite", data.namespace, jobs={"head": {}})
    model, _ = target(data, log, "head", source, "head_cost")
    arrays["G"][1][:] = 1 - arrays["G"][1]
    g = replace(data.role("G"), outcome_hash=array_hash(arrays["G"][1:]))
    other = RoleData(
        data.namespace,
        [data.role(n) if n != "G" else g for n in ("source_train", "source_stop", "F", "G", "O")],
    )
    log2 = RunLog(tmp_path / "two.sqlite", data.namespace, jobs={"head": {}})
    changed, _ = target(other, log2, "head", source, "head_cost")
    assert state_hash(model) != state_hash(changed)


def test_immutable_jobs_failures_restart_and_one_retry(tmp_path):
    namespace = "synthetic:phase3b:crash"
    path = tmp_path / "run.sqlite"
    log = RunLog(path, namespace, jobs={"job": {}, "other": {}})
    identity = dict(seed=1, data="fixture")
    r, _ = log.reserve("job", digest(identity), identity)
    assert log.summary()["used"] == 1  # Crash-before-fit still consumes an attempt.
    with pytest.raises(RuntimeError):
        RunLog(path, namespace, jobs=log.jobs).reserve("job", digest(identity), identity)
    log.fail(
        r["id"], "interrupted before fit; no trajectory resume", kind="environmental_interruption"
    )
    with pytest.raises(PermissionError):
        log.reserve("job", "different", identity)
    with pytest.raises(RuntimeError):
        log.execute("job", identity, lambda: b"complete")
    artifact, alias = log.execute("job", identity, lambda: b"complete", retry=True)
    assert not alias and log.summary()["used"] == 2
    _, alias = log.execute("other", identity, lambda: pytest.fail("alias retrained"))
    assert alias
    artifact.write_bytes(b"corrupt")
    with pytest.raises(PermissionError):
        log.execute("job", identity, lambda: b"replacement")


def test_failure_during_fit_and_interrupted_write(tmp_path, monkeypatch):
    import os

    log = RunLog(tmp_path / "run.sqlite", "synthetic:phase3b:fail", jobs={"job": {}})

    def failure():
        raise FloatingPointError("fixture nonfinite")

    with pytest.raises(FloatingPointError):
        log.execute("job", {"seed": 1}, failure)
    assert log.summary()["attempts"][0]["status"] == "failed"
    assert log.summary()["used"] == 1
    with pytest.raises(PermissionError, match="require review"):
        log.execute("job", {"seed": 1}, failure, retry=True)

    def interrupted(*args):
        raise OSError("fixture interrupted write")

    monkeypatch.setattr(os, "link", interrupted)
    with pytest.raises(OSError):
        atomic_new(tmp_path / "never-complete.pt", b"fixture")
    assert not (tmp_path / "never-complete.pt").exists()


def test_atomic_concurrent_reservation_and_hard_ceiling(tmp_path):
    jobs = {str(i): {} for i in range(160)}
    log = RunLog(tmp_path / "run.sqlite", "synthetic:phase3b:budget", jobs=jobs)

    def reserve(i):
        return log.reserve(str(i), str(i), {"seed": i})

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(reserve, range(144)))
    assert log.summary()["used"] == 144
    with pytest.raises(RuntimeError, match="ceiling"):
        reserve(144)
    with pytest.raises(PermissionError):
        RunLog(tmp_path / "real.sqlite", "real:phase3b")
    with pytest.raises(PermissionError):
        RunLog(log.path, log.namespace, jobs={"different": {}})


def test_duplicate_concurrent_jobs_reserve_once(tmp_path):
    log = RunLog(tmp_path / "run.sqlite", "synthetic:phase3b:dupe", jobs={"job": {}})

    def reserve(_):
        try:
            return log.reserve("job", "same", {})
        except RuntimeError:
            return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        result = list(pool.map(reserve, range(4)))
    assert sum(r is not None for r in result) == 1
    assert log.summary()["used"] == 1


def test_six_reserves_and_one_retry_per_job(tmp_path):
    log = RunLog(
        tmp_path / "run.sqlite", "synthetic:phase3b:retry", jobs={str(i): {} for i in range(7)}
    )
    for i in range(7):
        r, _ = log.reserve(str(i), str(i), {"i": i})
        log.fail(r["id"], "fixture interruption", kind="environmental_interruption")
        if i < 6:
            retry, _ = log.reserve(str(i), str(i), {"i": i}, retry=True)
            log.fail(retry["id"], "fixture second interruption", kind="environmental_interruption")
            with pytest.raises(RuntimeError, match="one fresh"):
                log.reserve(str(i), str(i), {"i": i}, retry=True)
        else:
            with pytest.raises(RuntimeError, match="ceiling"):
                log.reserve(str(i), str(i), {"i": i}, retry=True)
    assert log.summary()["reserves_used"] == 6


def test_default_matrix_entry_binding(tmp_path):
    log = RunLog(tmp_path / "matrix.sqlite", "synthetic:phase3b:matrix")
    log.validate_slot("15101/16101/B128/selective_a", method="selective_a", budget=128)
    with pytest.raises(PermissionError):
        log.validate_slot("15101/16101/B128/selective_a", method="full_cost")
    with pytest.raises(PermissionError):
        log.validate_slot("15101/16101/B32/selective_a", budget=32)


@pytest.fixture
def rehearsed(tmp_path):
    data, counters, _ = fixture_data()
    jobs = {"source": {}, **{m: {} for m in METHODS}}
    log = RunLog(tmp_path / "run.sqlite", data.namespace, jobs=jobs)
    source, report = train_source(data, log, "source", 15101, factory=factory)
    source_bank = extract(source, source.normalization, data, "source_train")
    target_bank = extract(source, source.normalization, data, "G")
    decision = analyze(source_bank, target_bank, fixture=True)
    models = dict(source=source.model)
    predictors = dict(
        source=dict(
            artifact=source.file,
            sha=source.sha,
            state_hash=source.checkpoint_hash,
            normalization=source.normalization.__dict__,
        )
    )
    for method in METHODS:
        model, record = target(
            data,
            log,
            method,
            source,
            method,
            decision=decision if method in {"random_a", "selective_a"} else None,
        )
        models[method] = model
        predictors[method] = record
    assert len(report["history"]) == 50 and 1 <= report["selected_epoch"] <= 50
    assert counters["O"] == dict(features=0, outcomes=0)
    return data, counters, log, models, predictors


def test_complete_end_to_end_seal_and_single_outcome_access(rehearsed):
    data, counters, log, models, predictors = rehearsed
    seal = PredictionSeal(log, data)
    with pytest.raises(PermissionError):
        seal.outcomes_once()
    with pytest.raises(PermissionError):
        seal.predict("source", models["source"])
    with pytest.raises(PermissionError):
        seal.freeze({k: v for k, v in predictors.items() if k != "selective_a"})
    seal.freeze(predictors)
    with pytest.raises(PermissionError):
        log.execute("source", {}, lambda: pytest.fail("fit after freeze"))
    seal.predict("source", models["source"])
    with pytest.raises(PermissionError):
        seal.seal()
    assert counters["O"]["outcomes"] == 0
    for name in METHODS:
        seal.predict(name, models[name])
    assert counters["O"]["outcomes"] == 0
    prediction_hash = seal.seal()
    assert len(prediction_hash) == 64
    y, obs = seal.outcomes_once()
    assert y.shape == obs.shape == (16, 2)
    assert counters["O"]["outcomes"] == 1
    with pytest.raises(PermissionError):
        PredictionSeal(log, data).outcomes_once()
    with pytest.raises(PermissionError):
        PredictionSeal(log, data).freeze(predictors)
    with pytest.raises(PermissionError):
        seal.predict("source", models["source"])
    assert all(r["status"] == "successful" for r in log.summary()["attempts"])


def test_changed_model_or_checkpoint_denied_before_feature_access(rehearsed):
    data, counters, log, models, predictors = rehearsed
    seal = PredictionSeal(log, data)
    seal.freeze(predictors)
    changed = copy.deepcopy(models["source"])
    with torch.no_grad():
        changed.prediction_head.weight.add_(1)
    with pytest.raises(PermissionError, match="Model differs"):
        seal.predict("source", changed)
    assert counters["O"] == dict(features=0, outcomes=0)
    path = predictors["source"]["artifact"]
    assert file_hash(path) == predictors["source"]["sha"]
    # Corrupt only this fictional temporary checkpoint.
    from pathlib import Path

    Path(path).write_bytes(b"fictional corruption")
    with pytest.raises(PermissionError, match="artifact change"):
        seal.predict("source", models["source"])
    assert counters["O"] == dict(features=0, outcomes=0)


def test_post_lock_code_and_sealed_prediction_tampering(rehearsed, monkeypatch):
    import dmvmat_ecg.phase3b.sealing as sealing_module

    data, counters, log, models, predictors = rehearsed
    seal = PredictionSeal(log, data)
    seal.freeze(predictors)
    actual = sealing_module.source_code_hash()
    monkeypatch.setattr(sealing_module, "source_code_hash", lambda: "changed-code")
    with pytest.raises(PermissionError, match="code/data"):
        seal.predict("source", models["source"])
    assert counters["O"]["features"] == 0
    monkeypatch.setattr(sealing_module, "source_code_hash", lambda: actual)
    for name in models:
        seal.predict(name, models[name])
    seal.seal()
    with log.connect() as db:
        predictions = json.loads(db.execute("SELECT predictions FROM evaluation").fetchone()[0])
    from pathlib import Path

    Path(predictions["source"]["path"]).write_bytes(b"tampered fictional predictions")
    with pytest.raises(PermissionError, match="prediction changed"):
        seal.outcomes_once()
    assert counters["O"]["outcomes"] == 0


def test_prediction_roster_cannot_invent_normalization(rehearsed):
    data, counters, log, _, predictors = rehearsed
    incorrect = copy.deepcopy(predictors)
    incorrect["source"]["normalization"]["mean"] += 1
    with pytest.raises(PermissionError, match="lineage"):
        PredictionSeal(log, data).freeze(incorrect)
    assert counters["O"] == dict(features=0, outcomes=0)


def test_source_file_and_namespace_integrity(tmp_path):
    data, _, _ = fixture_data()
    log = RunLog(tmp_path / "run.sqlite", data.namespace, jobs={"source": {}})
    source, _ = train_source(data, log, "source", 15101, factory=factory)
    saved = unpack(source.file)
    assert state_hash(source.model) == saved["state_hash"]
    changed_norm = replace(source.normalization, mean=source.normalization.mean + 1)
    with pytest.raises(PermissionError, match="normalization differ"):
        replace(source, normalization=changed_norm).validate(data)
    with pytest.raises(PermissionError, match="source normalization"):
        extract(source, changed_norm, data, "G")
    from pathlib import Path

    Path(source.file).write_bytes(b"bad fictional source")
    with pytest.raises(PermissionError):
        source.validate(data)
    assert json.dumps(log.summary())


def test_actual_ecg_backbone_with_fictional_waveforms(tmp_path):
    from dmvmat_ecg.phase3b.models import fresh_model

    original, counters, _ = fixture_data()
    rng = np.random.default_rng(654)
    roles = []
    for name in ("source_train", "source_stop", "F", "G", "O"):
        role = original.role(name)
        waveform = rng.normal(size=(len(role.records), 12, 1000)).astype("float32")
        roles.append(
            replace(role, features=lambda a=waveform: a.copy(), feature_hash=array_hash([waveform]))
        )
    data = RoleData(original.namespace, roles)
    log = RunLog(tmp_path / "ecg.sqlite", data.namespace, jobs={"source": {}, "head": {}})
    source, report = train_source(data, log, "source", 15101)
    model, result = fit_target(
        data, log, "head", source, "head_cost", source_seed=15101, draw_seed=16101, budget=64
    )
    assert len(report["history"]) == 50 and result["selected_update"] == 200
    assert sum(p.numel() for p in model.parameters()) == 212066
    assert set(result["drift"].values()) == {0.0}
    assert state_hash(fresh_model(15101)) != source.checkpoint_hash
    assert log.summary()["used"] == 2
    assert counters["O"] == dict(features=0, outcomes=0)
