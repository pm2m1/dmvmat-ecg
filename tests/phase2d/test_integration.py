"""Complete registered-matrix rehearsal on fictional ECGs, never real records."""

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch

from dmvmat_ecg.phase2.adaptation import APCheckpoint, SelectiveModel, macro_ap, state_hash
from dmvmat_ecg.phase2.contracts import MODULES
from dmvmat_ecg.phase2.evidence import EvidenceConfig
from dmvmat_ecg.phase2b.access import validate_source_preprocessing, verify_new_source
from dmvmat_ecg.phase2b.contract import contract as old_contract
from dmvmat_ecg.phase2b.contract import digest, file_hash
from dmvmat_ecg.phase2c.splits import build_candidate
from dmvmat_ecg.phase2d.access import AmendedAccess, seal_candidate
from dmvmat_ecg.phase2d.annotation_audit import serial_summary
from dmvmat_ecg.phase2d.contract import NAMESPACE, blank_review, contract
from dmvmat_ecg.phase2d.execution import calibrate_job, predict_locked, run_job
from dmvmat_ecg.phase2d.ledger import AmendedLedger

from .fixtures import Tiny, reader

torch.set_num_threads(2)


def fictional_rows():
    rows = []
    for age, counts in [(40, [1600, 400, 400, 400]), (70, [1200, 400, 400, 400])]:
        for y, n in zip([[0, 0], [1, 0], [0, 1], [1, 1]], counts, strict=True):
            for _ in range(n):
                i = len(rows)
                rows.append(
                    dict(
                        ecg_id=i,
                        patient_id=i,
                        age=age,
                        fold=i % 9 + 1,
                        filename=f"fictional/{i}",
                        raw_hash=f"fake{i}",
                        header_hash=f"header{i}",
                        y=y.copy(),
                        observed=[True, True],
                    )
                )
    return rows


def make(tmp_path, *, namespace="synthetic:phase2d", config=None):
    m = build_candidate(fictional_rows(), {"fictional": True})
    worker = seal_candidate(m, tmp_path / "vaults", synthetic=True)
    ledger = AmendedLedger(
        tmp_path,
        namespace=namespace,
        manifest_hash=digest(worker),
        code_hash="fictional-code",
        dependency_hash="fictional-env",
        config=config,
    )
    access = AmendedAccess(ledger, worker)
    return ledger, access


def fictional_adoption(ledger):
    review = blank_review()
    review["status"] = "ADOPTED BY HUMAN REVIEW"
    for v in review["decisions"].values():
        v.update(
            approved=True,
            reviewer="FICTIONAL TEST REVIEWER",
            reviewer_role="synthetic fixture",
            date="2000-01-01",
            notes="Fictional synthetic authorization only",
        )
    ledger.record_adoption(review)
    ledger.record_preflight([], "fictional-audit")
    ledger.authorize_development(
        {
            **ledger.identity,
            "statement": "explicitly authorize phase2d internal development",
            "adoption_hash": digest(review),
        }
    )


def test_unsigned_adoption_does_not_grant_training(tmp_path):
    ledger, access = make(tmp_path)
    with pytest.raises(PermissionError):
        ledger.record_adoption(blank_review())
    with pytest.raises(PermissionError):
        ledger.authorize_development({})
    for role, op in [
        ("source_train", "fit"),
        ("E", "stop"),
        ("C", "calibrate"),
        ("O", "evaluate_labels"),
        ("O", "evidence"),
    ]:
        with pytest.raises(PermissionError):
            access.rows(role, op)
    with pytest.raises(PermissionError):
        run_job(access, "source", synthetic_reader=reader, synthetic_factory=Tiny)
    assert ledger.snapshot()["attempts"] == ledger.snapshot()["evaluation_accesses"] == 0
    assert ledger.get("protocol_adoption") is None


def test_complete_33_fit_synthetic_matrix_rehearsal(tmp_path):
    ledger, access = make(tmp_path)
    fictional_adoption(ledger)
    # Crash before model fitting still consumes a reserved predictor attempt.
    interrupted = ledger.reserve("source")
    ledger.finish(
        interrupted, "interrupted", {"reason": "fictional crash before model initialization"}
    )
    identity = ledger.identity
    worker = access._manifest
    ledger.close()
    ledger = AmendedLedger(
        tmp_path,
        namespace=identity["namespace"],
        manifest_hash=identity["manifest_hash"],
        code_hash=identity["code_hash"],
        dependency_hash=identity["dependency_hash"],
    )
    access = AmendedAccess(ledger, worker)
    opts = dict(
        synthetic_factory=Tiny,
        synthetic_reader=reader,
        synthetic_steps=2,
        synthetic_evidence=EvidenceConfig(permutations=19, bootstraps=39),
        synthetic_dimensions=dict.fromkeys(MODULES, 4),
    )
    source = run_job(access, "source", retry_of=interrupted, **opts)
    source_sha = file_hash(ledger.artifact(source, "checkpoint.pt"))
    source_state = json.loads(ledger.artifact(source, "lineage.json").read_text())[
        "retained_state_hash"
    ]
    frozen_checks = 0
    zero_checkpoints = 0
    formulations = set()
    for job, spec in ledger.config["planned_jobs"].items():
        if job == "source":
            continue
        a = run_job(access, job, **opts)
        line = json.loads(ledger.artifact(a, "lineage.json").read_text())
        assert line["optimizer_fresh"] and line["device"] == "cpu"
        if spec["method"] != "target_only":
            assert line["initial_state_hash"] == source_state
        assert line["elapsed_seconds"] > 0 and line["trainable_parameters"] > 0
        assert line["history"][0]["eligible"] == (spec["method"] != "target_only")
        assert line["source_attempt"] == (None if spec["method"] == "target_only" else source)
        for name, mask in line["mask"].items():
            if mask == "SHARE":
                assert line["parameter_drift"][name] == 0
                frozen_checks += 1
        zero_checkpoints += line["selected_update"] == 0
        if spec["evidence_role"]:
            e = json.loads(ledger.artifact(a, "evidence.json").read_text())
            formulations.add(e["formulation"])
            assert set(e["modules"]) == set(MODULES)
            assert all("geometry" in m and "mmd" in m for m in e["modules"].values())
            assert line["selector_seconds"] > 0
        if spec["method"] == "random_a":
            assert line["random_mask_matching"] is not None
    assert (
        formulations == {"A", "B"}
        and file_hash(ledger.artifact(source, "checkpoint.pt")) == source_sha
    )
    for job, spec in ledger.config["planned_jobs"].items():
        cal = calibrate_job(access, job, synthetic_factory=Tiny, synthetic_reader=reader)
        expected = (
            128
            if job == "source"
            else spec["n_fit"] + 256
            if spec["method"] in {"head_f", "full_f"}
            else spec["n_fit"] + 512
        )
        assert cal["label_cost"]["target_records"] == expected
        assert cal["fit_role"] == "C"
    with pytest.raises(PermissionError):
        access.rows("O", "evaluate_predict")
    lock = ledger.lock()
    with pytest.raises(PermissionError):
        ledger.begin_evaluation()
    ledger.authorize_evaluation(
        {
            **ledger.identity,
            "statement": "explicitly authorize one phase2d internal evaluation",
            "lock_hash": lock,
        }
    )
    with ledger.evaluation_session() as token:
        with pytest.raises(PermissionError):
            access.rows("O", "evaluate_labels", token=token)
        fictional_predictions = []
        for job in ledger.config["planned_jobs"]:
            prediction = predict_locked(
                access, job, token, synthetic_factory=Tiny, synthetic_reader=reader
            )
            assert prediction.shape == (worker["summary"]["O"]["patients"], 2)
            assert np.isfinite(prediction).all()
            fictional_predictions.append(prediction)
        ledger.seal_predictions(token)
        out = access.rows("O", "evaluate_labels", token=token)
        assert len(out) == worker["summary"]["O"]["patients"]
        outcomes = np.array([row["y"] for row in out])
        observed = np.array([row["observed"] for row in out], bool)
        for prediction in fictional_predictions:
            assert 0 <= macro_ap(outcomes, prediction, observed) <= 1
    with pytest.raises(PermissionError):
        ledger.begin_evaluation()
    assert ledger.snapshot()["attempts"] == 34 and len(ledger.successful()) == 33
    assert ledger.snapshot()["evaluation_accesses"] == 1
    # Persist only a safe aggregate synthetic receipt; no fictional scores claimed as ECG evidence.
    (tmp_path / "rehearsal_summary.json").write_text(
        json.dumps(
            dict(
                scope="SYNTHETIC ONLY",
                successful_predictors=33,
                interrupted_attempts=1,
                attempts=34,
                calibration_artifacts=33,
                evaluation_accesses=1,
                formulations=sorted(formulations),
                frozen_module_checks=frozen_checks,
                epoch_zero_retained=zero_checkpoints,
            )
        )
    )


def dummy_complete(ledger, trial):
    job, _ = ledger.active(trial)
    h = ledger.publish(trial, "checkpoint.pt", b"fictional-placeholder-not-weights")["sha"]
    ledger.publish(
        trial,
        "lineage.json",
        json.dumps(
            {**ledger.identity, "job_id": job, "labels": ["CD", "STTC"], "checkpoint_sha256": h}
        ).encode(),
    )
    ledger.finish(trial, "successful")


def test_36_ceiling_concurrency_and_restart(tmp_path):
    ledger, _ = make(tmp_path)
    fictional_adoption(ledger)
    dummy_complete(ledger, ledger.reserve("source"))
    parents = []
    for job in list(ledger.config["planned_jobs"])[1:]:
        a = ledger.reserve(job)
        ledger.finish(a, "failed", {"reason": "fictional mid-fit crash"})
        parents.append((job, a))
    identity = ledger.identity

    def reserve(pair):
        worker_ledger = AmendedLedger(
            tmp_path,
            namespace=identity["namespace"],
            manifest_hash=identity["manifest_hash"],
            code_hash=identity["code_hash"],
            dependency_hash=identity["dependency_hash"],
        )
        try:
            return worker_ledger.reserve(pair[0], retry_of=pair[1])
        except (PermissionError, RuntimeError):
            return None
        finally:
            worker_ledger.close()

    with ThreadPoolExecutor(max_workers=4) as pool:
        r = list(pool.map(reserve, parents[:4]))
    assert sum(x is not None for x in r) == 3 and ledger.snapshot()["attempts"] == 36
    with pytest.raises(RuntimeError):
        ledger.reserve(parents[5][0], retry_of=parents[5][1])
    with pytest.raises(PermissionError):
        ledger.finish(ledger.successful()["source"], "failed")
    ledger.close()
    again = AmendedLedger(
        tmp_path,
        namespace=identity["namespace"],
        manifest_hash=identity["manifest_hash"],
        code_hash=identity["code_hash"],
        dependency_hash=identity["dependency_hash"],
    )
    assert again.snapshot()["attempts"] == 36
    with pytest.raises(RuntimeError):
        again.reserve(parents[6][0], retry_of=parents[6][1])


def test_original_manifest_endpoint_and_canonical_changes_rejected(tmp_path):
    m = build_candidate(fictional_rows(), {})
    for field, value in [
        ("split_version", "old"),
        ("labels", ["MI", "CD"]),
        ("contract_hash", "old"),
    ]:
        bad = copy.deepcopy(m)
        bad[field] = value
        with pytest.raises((PermissionError, ValueError)):
            seal_candidate(bad, tmp_path / field, synthetic=True)
    cfg = contract()
    cfg["canonical_policy"] = "choose most positive"
    with pytest.raises(ValueError):
        AmendedLedger(
            tmp_path / "real",
            namespace=NAMESPACE,
            manifest_hash="x",
            code_hash="x",
            dependency_hash="x",
            config=cfg,
        )
    with pytest.raises(PermissionError):
        seal_candidate(m, tmp_path / "real-vault")
    assert not (tmp_path / "real" / "study.sqlite").exists()


def test_source_and_normalization_history_rejected_and_O_identity_not_preloaded(
    tmp_path, monkeypatch
):
    ledger, access = make(tmp_path)
    fictional_adoption(ledger)
    assert "O" not in access._manifest["roles"]
    forbidden = {
        Path(access._manifest["role_vaults"]["O"]["path"]),
        Path(access._manifest["evaluation_membership_vault"]["path"]),
    }
    original = Path.read_text

    def read(p, *args, **kw):
        if p in forbidden:
            raise AssertionError("O preload by trainer/selector")
        return original(p, *args, **kw)

    monkeypatch.setattr(Path, "read_text", read)
    a = ledger.reserve("source")
    assert access.rows("source_train", "fit", attempt=a)
    for role, op in [("O", "evidence"), ("O", "fit"), ("C", "fit"), ("source_stop", "fit")]:
        with pytest.raises(PermissionError):
            access.rows(role, op, attempt=a)
    with pytest.raises(PermissionError):
        verify_new_source(access, "historical-checkpoint")
    for stats in [
        dict(
            namespace="old",
            fit_role="source_train",
            fresh=True,
            fit_membership_hash=access._manifest["role_hashes"]["source_train"],
        ),
        dict(
            namespace=ledger.identity["namespace"],
            fit_role="source_train",
            fresh=True,
            fit_membership_hash="old",
        ),
    ]:
        with pytest.raises(PermissionError):
            validate_source_preprocessing(stats, access)


def test_old_ledger_reuse_denied_without_modification(tmp_path):
    from dmvmat_ecg.phase2b.ledger import PilotLedger

    old = PilotLedger(
        tmp_path,
        namespace="synthetic:old",
        manifest_hash="old",
        code_hash="old",
        dependency_hash="old",
    )
    old.close()
    before = file_hash(tmp_path / "study.sqlite")
    with pytest.raises(PermissionError):
        AmendedLedger(
            tmp_path,
            namespace="synthetic:phase2d",
            manifest_hash="new",
            code_hash="new",
            dependency_hash="new",
        )
    assert file_hash(tmp_path / "study.sqlite") == before


def test_epoch_zero_and_frozen_BN_retention():
    source = Tiny().eval()
    h = state_hash(source)
    model = SelectiveModel(source, dict.fromkeys(MODULES, "SHARE"))
    picker = APCheckpoint()
    assert picker.consider(model, ap=0.8, bce=0.2, epoch=0)
    model.train()
    model(wave=torch.randn(16, 12, 1000)).sum().backward()
    model.verify_frozen()
    assert not picker.consider(model, ap=0.7, bce=0.1, epoch=2)
    model.load_state_dict(picker.state)
    assert picker.epoch == 0 and state_hash(source) == h


def test_config_only_adds_version_metadata_not_scientific_matrix():
    a, b = old_contract(), contract()
    assert all(
        b[k] == v for k, v in a.items() if k not in {"source_split_seed", "target_split_seed"}
    )
    assert b["source_split_seed"] == b["target_split_seed"] == 44301
    assert len(b["planned_jobs"]) == 33 and b["predictor_attempt_ceiling"] == 36
    assert not any(
        b[k] for k in ["protocol_approval", "training_authorization", "evaluation_authorization"]
    )


def test_serial_record_audit_keeps_exact_record_not_or():
    records = fictional_rows()[:2]
    records[1] = {**records[1], "patient_id": records[0]["patient_id"], "y": [1, 1]}
    metadata = {
        0: dict(
            recording_date="2000-01-01 00:00:00",
            validated_by_human="False",
            second_opinion="False",
            initial_autogenerated_report="True",
            validated_by="",
            likelihoods={"CD": [], "STTC": []},
        ),
        1: dict(
            recording_date="2000-01-02 00:00:00",
            validated_by_human="True",
            second_opinion="True",
            initial_autogenerated_report="False",
            validated_by="pseudonym",
            likelihoods={"CD": [100], "STTC": [0]},
        ),
    }
    summary = serial_summary(records, metadata)
    assert summary["changes"] == {"both": 1} and summary["affected_recordings"] == 2
    assert summary["timing"]["patients_different_dates"] == 1
    assert summary["canonical_exact_record_labels_verified"]
    assert summary["diagnostic_likelihoods"]["STTC"]["unknown_zero_likelihood_records"] == 1


def test_changed_live_config_checkpoint_corruption_and_pending_write(tmp_path):
    ledger, _ = make(tmp_path)
    fictional_adoption(ledger)
    ledger.config["target_optimizer"]["lr"] = 0.5
    with pytest.raises(PermissionError, match="contract changed"):
        ledger.reserve("source")
    ledger.config["target_optimizer"]["lr"] = contract()["target_optimizer"]["lr"]
    trial = ledger.reserve("source")
    path = tmp_path / "artifacts" / trial / "checkpoint.pt.pending-fictional-crash"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"partial")
    with pytest.raises(PermissionError):
        ledger.finish(trial, "successful")
    ledger.publish(trial, "checkpoint.pt", b"fictional-complete")
    with pytest.raises(FileExistsError):
        ledger.publish(trial, "checkpoint.pt", b"overwrite")
    ledger.artifact(trial, "checkpoint.pt").write_bytes(b"corrupted")
    with pytest.raises(PermissionError):
        ledger.artifact(trial, "checkpoint.pt")
    ledger.finish(trial, "interrupted", {"reason": "fictional incomplete/corrupt publication"})
    assert ledger.snapshot()["attempts"] == 1


def test_adoption_is_not_development_or_evaluation_permission(tmp_path):
    ledger, _ = make(tmp_path)
    review = blank_review()
    review["status"] = "ADOPTED BY HUMAN REVIEW"
    for value in review["decisions"].values():
        value.update(
            approved=True,
            reviewer="FICTIONAL",
            reviewer_role="synthetic only",
            date="2000-01-01",
            notes="fictional test",
        )
    ledger.record_adoption(review)
    with pytest.raises(PermissionError):
        ledger.reserve("source")
    with pytest.raises(PermissionError):
        ledger.begin_evaluation()
    with pytest.raises(PermissionError):
        ledger.record_adoption(review)
    assert ledger.snapshot()["phase"] == "draft" and ledger.snapshot()["attempts"] == 0


def test_wrong_old_schema_and_spoofed_fresh_source_rejected(tmp_path):
    import sqlite3

    foreign = tmp_path / "legacy"
    foreign.mkdir()
    db = sqlite3.connect(foreign / "study.sqlite")
    db.execute("CREATE TABLE meta(key TEXT,value TEXT)")
    db.execute("INSERT INTO meta VALUES (?,?)", ("phase", '"evaluated"'))
    db.commit()
    db.close()
    h = file_hash(foreign / "study.sqlite")
    with pytest.raises(PermissionError):
        AmendedLedger(
            foreign,
            namespace="synthetic:phase2d",
            manifest_hash="x",
            code_hash="x",
            dependency_hash="x",
        )
    assert file_hash(foreign / "study.sqlite") == h
    ledger, access = make(tmp_path / "new")
    fictional_adoption(ledger)
    trial = ledger.reserve("source")
    dummy_complete(ledger, trial)
    with pytest.raises(PermissionError):
        verify_new_source(access, trial)
