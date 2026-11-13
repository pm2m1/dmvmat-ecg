"""Durable synthetic rehearsal. No project waveform/cache/test partition reads."""

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch
from torch import nn

from dmvmat_ecg.phase2.adaptation import SelectiveModel, macro_ap, masked_bce, state_hash
from dmvmat_ecg.phase2.contracts import MODULES
from dmvmat_ecg.phase2.evidence import EvidenceConfig
from dmvmat_ecg.phase2b.access import (
    RoleAccess,
    fit_preprocessing,
    fresh_source,
    validate_source_preprocessing,
)
from dmvmat_ecg.phase2b.contract import contract, digest, file_hash
from dmvmat_ecg.phase2b.execution import calibrate_job, predict_locked, run_job
from dmvmat_ecg.phase2b.ledger import PilotLedger
from dmvmat_ecg.phase2b.preflight import build_manifest
from dmvmat_ecg.phase2b.sealing import seal_role_labels

from .test_preflight import fictional_rows

torch.set_num_threads(2)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        for n in MODULES[:5]:
            setattr(
                self,
                n,
                nn.Sequential(nn.Linear(12, 4), nn.BatchNorm1d(4), nn.ReLU(), nn.Dropout(0.2)),
            )
        self.fusion_1 = nn.Sequential(
            nn.Linear(20, 4), nn.BatchNorm1d(4), nn.ReLU(), nn.Dropout(0.2)
        )
        self.fusion_2 = nn.Sequential(
            nn.Linear(4, 4), nn.BatchNorm1d(4), nn.ReLU(), nn.Dropout(0.2)
        )
        self.prediction_head = nn.Linear(4, 2)

    def selectable_modules(self):
        return {n: getattr(self, n) for n in MODULES}

    def representations(self, wave, rhythm=None, spectrum=None):
        x = wave.mean(-1)
        r = {n: getattr(self, n)(x) for n in MODULES[:5]}
        r["fusion_1"] = self.fusion_1(torch.cat(list(r.values()), 1))
        r["fusion_2"] = self.fusion_2(r["fusion_1"])
        return r

    def forward(self, wave, rhythm=None, spectrum=None):
        return self.prediction_head(self.representations(wave, rhythm, spectrum)["fusion_2"])


def reader(row):
    rng = np.random.default_rng(row["ecg_id"] + 555)
    x = rng.normal(0, 0.1, (12, 1000)).astype("float32")
    # Fictional signal generation needs no label access (O outcomes are redacted).
    x[:3] += (row["ecg_id"] % 2) * 0.1
    x[3:6] += ((row["ecg_id"] // 2) % 2) * 0.1
    return x


def make_ledger(tmp_path, config=None, namespace="synthetic:phase2b"):
    manifest = seal_role_labels(
        build_manifest(fictional_rows(), {"synthetic": True}), tmp_path / "role_vaults"
    )
    ledger = PilotLedger(
        tmp_path,
        namespace=namespace,
        manifest_hash=digest(manifest),
        code_hash="synthetic-code",
        dependency_hash="synthetic-lock",
        config=config,
    )
    return ledger, manifest


def approve(ledger):
    ledger.authorize_development(
        {
            **ledger.identity,
            "statement": "explicitly authorize phase2b internal development",
            "adopt_draft_contract": True,
        }
    )


def dummy_complete(ledger, identifier):
    job, _ = ledger.active(identifier)
    artifact = ledger.publish(identifier, "checkpoint.pt", b"synthetic-not-model-weights")
    metadata = {
        **ledger.identity,
        "labels": ["CD", "STTC"],
        "job_id": job,
        "checkpoint_sha256": artifact["sha"],
    }
    ledger.publish(identifier, "lineage.json", json.dumps(metadata).encode())
    ledger.finish(identifier, "successful")


def test_default_deny_before_any_signal_or_source_initialization(tmp_path):
    ledger, manifest = make_ledger(tmp_path)
    access = RoleAccess(ledger, manifest)
    with pytest.raises(PermissionError):
        run_job(access, "source", synthetic_reader=reader, synthetic_factory=Tiny)
    with pytest.raises(PermissionError):
        access.load("O", "evaluate_predict", synthetic_reader=reader)
    with pytest.raises(PermissionError):
        access.rows("C", "fit", attempt="misleading-role")
    assert ledger.snapshot()["attempts"] == 0 and ledger.snapshot()["evaluation_accesses"] == 0


def test_preflight_blocker_cannot_be_silently_approved(tmp_path):
    ledger, _ = make_ledger(tmp_path)
    ledger.record_preflight(["Joint-cell failure requires prospective review"], "audit-hash")
    with pytest.raises(PermissionError):
        approve(ledger)
    with pytest.raises(PermissionError):
        ledger.record_preflight([], "different-audit")
    assert ledger.snapshot()["attempts"] == 0


def test_full_synthetic_rehearsal_and_single_use_evaluation(tmp_path):
    cfg = contract()
    names = [
        "source",
        *[
            f"d4101:n128:{m}"
            for m in [
                "head_f",
                "head_cost",
                "full_f",
                "full_cost",
                "random_a",
                "selective_a",
                "selective_b",
                "target_only",
            ]
        ],
    ]
    cfg["planned_jobs"] = {n: cfg["planned_jobs"][n] for n in names}
    ledger, manifest = make_ledger(tmp_path, config=cfg)
    access = RoleAccess(ledger, manifest)
    approve(ledger)
    opts = dict(
        synthetic_factory=Tiny,
        synthetic_reader=reader,
        synthetic_steps=2,
        synthetic_evidence=EvidenceConfig(permutations=39, bootstraps=99),
        synthetic_dimensions=dict.fromkeys(MODULES, 4),
    )
    source = run_job(access, "source", **opts)
    source_sha = file_hash(ledger.artifact(source, "checkpoint.pt"))
    for name in names[1:]:
        attempt = run_job(access, name, **opts)
        lineage = json.loads(ledger.artifact(attempt, "lineage.json").read_text())
        assert lineage["source_attempt"] == (None if name.endswith("target_only") else source)
        assert lineage["selected_update"] in [0, 2]
        assert lineage["history"][0]["eligible"] == (not name.endswith("target_only"))
        if name.endswith("target_only"):
            assert lineage["selected_update"] != 0
        assert not lineage["exact_resume_supported"]
    assert file_hash(ledger.artifact(source, "checkpoint.pt")) == source_sha
    for name in names:
        artifact = calibrate_job(access, name, synthetic_factory=Tiny, synthetic_reader=reader)
        assert artifact["fit_role"] == "C"
        expected = 128 if name == "source" else 384 if name.endswith(("head_f", "full_f")) else 640
        assert artifact["label_cost"]["target_records"] == expected
    with pytest.raises(PermissionError):
        access.rows("O", "evaluate_labels", token="false-lease")
    lock = ledger.lock()
    with pytest.raises(PermissionError):
        ledger.begin_evaluation()
    ledger.authorize_evaluation(
        {
            **ledger.identity,
            "lock_hash": lock,
            "statement": "explicitly authorize one phase2b internal evaluation",
        }
    )
    token = ledger.begin_evaluation()
    with pytest.raises(PermissionError):
        access.rows("O", "evaluate_labels", token=token)
    predictions = []
    for name in names:
        p = predict_locked(access, name, token, synthetic_factory=Tiny, synthetic_reader=reader)
        assert p.shape == (len(manifest["roles"]["O"]), 2)
        predictions.append(p)
    ledger.seal_predictions(token)
    rows = access.rows("O", "evaluate_labels", token=token)
    assert len(rows) == len(manifest["roles"]["O"])
    for p in predictions:
        y = np.array([r["y"] for r in rows])
        assert 0 <= macro_ap(y, p, np.ones_like(y, bool)) <= 1
    assert access.label_cost()["evaluation_records"] == len(rows)
    ledger.finish_evaluation(token)
    with pytest.raises(PermissionError):
        ledger.begin_evaluation()
    with pytest.raises(PermissionError):
        ledger.reserve("source")
    assert ledger.snapshot()["attempts"] == 9 and ledger.snapshot()["evaluation_accesses"] == 1
    assert ledger.snapshot()["phase"] == "evaluated"


def test_budget_failure_restart_and_exact_36_ceiling(tmp_path):
    ledger, _ = make_ledger(tmp_path)
    approve(ledger)
    source = ledger.reserve("source")
    ledger.finish(source, "interrupted", {"reason": "fictional crash before fit"})
    with pytest.raises(PermissionError):
        ledger.reserve("source")
    retry = ledger.reserve("source", retry_of=source)
    dummy_complete(ledger, retry)
    failed = []
    for job in contract()["planned_jobs"]:
        if job == "source":
            continue
        trial = ledger.reserve(job)
        ledger.finish(trial, "failed", {"reason": "fictional mid-fit crash"})
        failed.append((job, trial))
    for job, parent in failed[:2]:
        trial = ledger.reserve(job, retry_of=parent)
        ledger.finish(trial, "failed", {"reason": "fictional retry crash"})
    assert ledger.snapshot()["attempts"] == 36
    identity = ledger.identity
    ledger.close()
    restarted = PilotLedger(
        tmp_path,
        namespace=identity["namespace"],
        manifest_hash=identity["manifest_hash"],
        code_hash=identity["code_hash"],
        dependency_hash=identity["dependency_hash"],
    )
    with pytest.raises(RuntimeError, match="36-attempt"):
        restarted.reserve(failed[2][0], retry_of=failed[2][1])
    with pytest.raises(PermissionError):
        restarted.finish(retry, "failed")
    assert restarted.snapshot()["attempts"] == 36


def test_concurrent_duplicate_attempts_and_reservation_identity(tmp_path):
    ledger, manifest = make_ledger(tmp_path)
    approve(ledger)
    ledger.close()

    def reserve(_):
        one, _ = make_ledger(tmp_path)
        try:
            return one.reserve("source")
        except PermissionError:
            return None
        finally:
            one.close()

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(reserve, range(4)))
    assert sum(v is not None for v in results) == 1
    restarted, _ = make_ledger(tmp_path)
    assert restarted.snapshot()["attempts"] == 1
    with pytest.raises(PermissionError):
        PilotLedger(
            tmp_path,
            namespace="real:phase2b-internal",
            manifest_hash=digest(manifest),
            code_hash="synthetic-code",
            dependency_hash="synthetic-lock",
        )


def test_atomic_checkpoint_crash_corruption_and_foreign_files(tmp_path):
    ledger, _ = make_ledger(tmp_path)
    approve(ledger)
    trial = ledger.reserve("source")
    path = tmp_path / "artifacts" / trial / "checkpoint.pt.pending-crash"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"incomplete")
    with pytest.raises(PermissionError):
        ledger.finish(trial, "successful")
    artifact = ledger.publish(trial, "checkpoint.pt", b"complete-synthetic-fixture")
    with pytest.raises(FileExistsError):
        ledger.publish(trial, "checkpoint.pt", b"overwrite")
    assert ledger.artifact(trial, "checkpoint.pt").read_bytes() == b"complete-synthetic-fixture"
    ledger.artifact(trial, "checkpoint.pt").write_bytes(b"corrupted")
    with pytest.raises(PermissionError):
        ledger.artifact(trial, "checkpoint.pt")
    with pytest.raises(ValueError):
        ledger.publish(trial, "../historical.pt", b"no")
    assert artifact["sha"] != file_hash(path)
    ledger.finish(trial, "interrupted", {"reason": "review required after crash"})


def test_source_preprocessing_membership_and_freshness_rejection(tmp_path):
    ledger, manifest = make_ledger(tmp_path)
    approve(ledger)
    access = RoleAccess(ledger, manifest)
    trial = ledger.reserve("source")
    batch = access.load("source_train", "preprocess_fit", attempt=trial, synthetic_reader=reader)
    stats = fit_preprocessing(batch, access, trial)
    assert stats["mean"] == pytest.approx(batch["x"].astype("float64").mean())
    validate_source_preprocessing(stats, access)
    for bad in [
        {"fit_role": "train"},
        {"namespace": "historical"},
        {"fit_membership_hash": "old-hash"},
    ]:
        with pytest.raises(PermissionError):
            validate_source_preprocessing({**stats, **bad}, access)
    with pytest.raises(PermissionError):
        access.load("O", "fit", attempt=trial, synthetic_reader=reader)
    with pytest.raises(PermissionError):
        access.load("C", "preprocess_fit", attempt=trial, synthetic_reader=reader)
    fresh_source(access, trial, synthetic_factory=Tiny)
    with pytest.raises(PermissionError):
        fresh_source(access, trial, synthetic_factory=Tiny)
    with pytest.raises(PermissionError):
        access.load("source_train", "fit", attempt=trial, raw_root=tmp_path)


@pytest.mark.parametrize(
    "adapted", [[], list(MODULES), *[[n] for n in MODULES], ["view_1", "fusion_1", "fusion_2"]]
)
def test_seven_module_modes_gradients_buffers_and_source_integrity(adapted):
    torch.manual_seed(44)
    source = Tiny().eval()
    initial = state_hash(source)
    mask = {n: "ADAPT" if n in adapted else "SHARE" for n in MODULES}
    model = SelectiveModel(source, mask)
    wave = torch.randn(16, 12, 1000)
    with torch.no_grad():
        torch.testing.assert_close(source(wave), model(wave=wave), atol=0, rtol=0)
    nn.Sequential(model).train()
    optimizer = model.optimizer()
    assert {id(p) for g in optimizer.param_groups for p in g["params"]} == {
        id(p) for p in model.parameters() if p.requires_grad
    }
    loss = masked_bce(model(wave=wave), torch.ones(16, 2), torch.ones(16, 2, dtype=torch.bool))
    loss.backward()
    optimizer.step()
    model.verify_frozen()
    for n, module in model.model.selectable_modules().items():
        for m in module.modules():
            if isinstance(m, (nn.Dropout, nn.BatchNorm1d)):
                assert m.training == (n in adapted)
        for p in module.parameters():
            assert (p.grad is not None) == (n in adapted)
        bn = next(m for m in module.modules() if isinstance(m, nn.BatchNorm1d))
        assert int(bn.num_batches_tracked) == int(n in adapted)
    assert state_hash(source) == initial


def test_changed_integrity_and_historical_ledger_refused(tmp_path):
    ledger, manifest = make_ledger(tmp_path / "new")
    pin = tmp_path / "code.py"
    pin.write_text("initial")
    access = RoleAccess(ledger, manifest, integrity_files={str(pin): file_hash(pin)})
    approve(ledger)
    trial = ledger.reserve("source")
    pin.write_text("changed")
    with pytest.raises(PermissionError):
        access.rows("source_train", "fit", attempt=trial)
    import sqlite3

    old = tmp_path / "old"
    old.mkdir()
    db = sqlite3.connect(old / "study.sqlite")
    db.execute("CREATE TABLE meta(k TEXT PRIMARY KEY,v TEXT)")
    db.execute("INSERT INTO meta VALUES (?,?)", ("phase", '"evaluated"'))
    db.commit()
    db.close()
    h = file_hash(old / "study.sqlite")
    with pytest.raises(PermissionError):
        make_ledger(old)
    assert file_hash(old / "study.sqlite") == h


def test_unregistered_smoke_and_omitted_controls_cannot_expand_or_lock(tmp_path):
    ledger, _ = make_ledger(tmp_path)
    approve(ledger)
    with pytest.raises(PermissionError):
        ledger.reserve("extra_architecture_smoke")
    with pytest.raises(PermissionError):
        ledger.reserve("source", operation="compatibility_fit")
    with pytest.raises(PermissionError):
        ledger.lock()
    assert ledger.snapshot()["attempts"] == 0


def test_real_namespace_default_denies_all_predictive_access(tmp_path):
    ledger, manifest = make_ledger(tmp_path, namespace="real:phase2b-internal")
    pin = tmp_path / "code.txt"
    pin.write_text("test namespace guard, no real data")
    access = RoleAccess(ledger, manifest, integrity_files={str(pin): file_hash(pin)})
    for role, op in [
        ("source_train", "fit"),
        ("source_train", "evidence"),
        ("C", "calibrate"),
        ("O", "evaluate_predict"),
        ("O", "evaluate_labels"),
    ]:
        with pytest.raises(PermissionError):
            access.rows(role, op)
    with pytest.raises(PermissionError):
        run_job(access, "source")
    assert ledger.snapshot()["attempts"] == 0
    assert not ledger.snapshot()["development_authorized"]
    with pytest.raises(PermissionError):
        approve(ledger)


def test_worker_manifest_and_denied_access_do_not_load_sealed_outcomes(tmp_path, monkeypatch):
    from pathlib import Path

    ledger, manifest = make_ledger(tmp_path)
    assert all(
        "y" not in r and "observed" not in r for rows in manifest["roles"].values() for r in rows
    )
    forbidden = Path(manifest["role_vaults"]["O"]["path"])
    original = Path.read_text

    def read(path, *args, **kwargs):
        if path == forbidden:
            raise AssertionError("Predictive worker loaded O labels")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    access = RoleAccess(ledger, manifest)
    approve(ledger)
    trial = ledger.reserve("source")
    assert access.rows("source_train", "fit", attempt=trial)
    with pytest.raises(PermissionError):
        access.rows("O", "evidence", attempt=trial)
    with pytest.raises(PermissionError):
        access.rows("O", "evaluate_labels")


def test_concurrent_retry_reservations_cannot_exceed_36(tmp_path):
    ledger, _ = make_ledger(tmp_path)
    approve(ledger)
    dummy_complete(ledger, ledger.reserve("source"))
    parents = []
    for job in contract()["planned_jobs"]:
        if job == "source":
            continue
        trial = ledger.reserve(job)
        ledger.finish(trial, "failed")
        parents.append((job, trial))
    assert ledger.snapshot()["attempts"] == 33

    def reserve(pair):
        one, _ = make_ledger(tmp_path)
        try:
            return one.reserve(pair[0], retry_of=pair[1])
        except (PermissionError, RuntimeError):
            return None
        finally:
            one.close()

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(reserve, parents[:4]))
    assert sum(r is not None for r in results) == 3
    assert ledger.snapshot()["attempts"] == 36


def test_historical_source_id_and_foreign_preprocessing_refused(tmp_path):
    from dmvmat_ecg.phase2b.access import verify_new_source

    ledger, manifest = make_ledger(tmp_path)
    approve(ledger)
    access = RoleAccess(ledger, manifest)
    with pytest.raises(PermissionError):
        verify_new_source(access, "historical-benchmark-run")
    trial = ledger.reserve("source")
    dummy_complete(ledger, trial)
    with pytest.raises(PermissionError):
        verify_new_source(access, trial)


def test_failed_synthetic_evaluation_consumes_lease_across_restart(tmp_path):
    cfg = contract()
    cfg["planned_jobs"] = {"source": cfg["planned_jobs"]["source"]}
    ledger, manifest = make_ledger(tmp_path, config=cfg)
    approve(ledger)
    access = RoleAccess(ledger, manifest)
    run_job(access, "source", synthetic_factory=Tiny, synthetic_reader=reader, synthetic_steps=2)
    calibrate_job(access, "source", synthetic_factory=Tiny, synthetic_reader=reader)
    lock = ledger.lock()
    ledger.authorize_evaluation(
        {
            **ledger.identity,
            "lock_hash": lock,
            "statement": "explicitly authorize one phase2b internal evaluation",
        }
    )
    with pytest.raises(RuntimeError, match="fictional power loss"):
        with ledger.evaluation_session() as token:
            rows = access.rows("O", "evaluate_predict", token=token)
            assert all("y" not in row for row in rows)
            raise RuntimeError("fictional power loss")
    assert ledger.snapshot()["phase"] == "evaluation_failed"
    assert ledger.snapshot()["evaluation_accesses"] == 1
    ledger.close()
    restarted, _ = make_ledger(tmp_path, config=cfg)
    with pytest.raises(PermissionError):
        restarted.begin_evaluation()
    with pytest.raises(PermissionError):
        restarted.require_metric_access(token)
