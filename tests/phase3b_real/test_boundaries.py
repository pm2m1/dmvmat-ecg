"""Fictional IDs and temporary ledgers only; never instantiate a real data loader."""

import json

import pytest

from dmvmat_ecg.phase3b.jobs import source_code_hash
from dmvmat_ecg.phase3b_real import boundary, log
from dmvmat_ecg.phase3b_real.provider import RealData


@pytest.mark.parametrize(
    "role,record",
    [
        ("O", "sealed"),
        ("target_development", "sealed"),
        ("target_development", "old-feas"),
        ("F", "not-fitting"),
    ],
)
def test_provider_refuses_before_read(role, record):
    memberships = dict(
        O=["sealed"], FEAS=["old-feas"], target_development=["permitted"], F=["fitting"]
    )
    with pytest.raises(PermissionError):
        boundary.permit_record(record, role, memberships)


def test_permitted_development_and_frozen_identity():
    memberships = dict(O=["sealed"], FEAS=["old-feas"], target_development=["permitted"])
    boundary.permit_record("permitted", "target_development", memberships)
    assert source_code_hash() == boundary.BASE_HASH
    with pytest.raises(PermissionError):
        boundary.real_only("synthetic:phase3b:test")


def test_no_O_provider_or_role_alias():
    data = RealData.__new__(RealData)
    data.verify = lambda: None
    data._roles = {}
    with pytest.raises(PermissionError):
        data.read("O", "fit_target")
    with pytest.raises(PermissionError):
        data.role("O")
    with pytest.raises(PermissionError):
        data.read("G", "stop_source")


def test_wrong_base_identity_fails_closed(monkeypatch):
    monkeypatch.setattr(boundary, "source_code_hash", lambda: "modified")
    with pytest.raises(PermissionError):
        boundary.real_only(boundary.NAMESPACE)


def test_real_log_canonical_binding_and_development_lock(tmp_path, monkeypatch):
    # Relocate the trusted root explicitly inside this fictional fixture only.
    monkeypatch.setattr(log, "PROJECT_ROOT", tmp_path)
    folder = tmp_path / "private/phase3b_cpsc_v1"
    folder.mkdir(parents=True)
    (folder / "EXECUTION_PREFLIGHT.json").write_text(
        json.dumps(
            dict(
                implementation_hash=boundary.implementation_hash(),
                status="passed",
                O_access_authorized=False,
            )
        )
    )
    study = log.RealLog(tmp_path)
    row, alias = study.reserve("source/15101", "fictional-job", dict(seed=15101))
    assert not alias and study.summary()["used"] == 1
    study.fail(row["id"], "fictional interruption", kind="environmental_interruption")
    assert log.RealLog(tmp_path).summary()["used"] == 1
    with pytest.raises(PermissionError):
        log.RealLog(tmp_path / "different")
    with study.connect() as db:
        db.execute("INSERT INTO evaluation VALUES (1,'development_locked','{}','{}','fixture')")
    with pytest.raises(PermissionError):
        study.reserve("source/15102", "other", dict(seed=15102))
