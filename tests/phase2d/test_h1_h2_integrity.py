"""H1/H2 regressions use temporary fictional projects/DBs, never the real ledger."""

import copy
import json
import sqlite3
from types import SimpleNamespace
from uuid import UUID

import pytest

from dmvmat_ecg.phase2b.runtime import code_identity, execution_inventory
from dmvmat_ecg.phase2d import runtime
from dmvmat_ecg.phase2d.access import AmendedAccess
from dmvmat_ecg.phase2d.contract import NAMESPACE, digest
from dmvmat_ecg.phase2d.ledger import AmendedLedger, bind_existing_draft_ledger


@pytest.fixture
def fictional_real(tmp_path, monkeypatch):
    """Redirect every trusted location/dependency probe to fictional temp files."""
    root = tmp_path / "fictional-project"
    for name, content in {
        "src/fake.py": "FICTITIOUS = True\n",
        "configs/fake.json": "{}\n",
        "scripts/fake.py": "# fictional script\n",
        "uv.lock": "# fictional lock\n",
        "pyproject.toml": "# fictional package\n",
        "results/environment.json": "{}\n",
        "docs/phase2/INTERNAL_TRANSFER_PILOT_PROPOSAL.md": "fictional proposal\n",
        "docs/phase2/PILOT_EXECUTION_CONTRACT.md": "fictional contract\n",
    }.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    folder = root / "private/fictional-ledger"
    path = folder / "study.sqlite"
    monkeypatch.setattr(runtime, "REAL_PROJECT_ROOT", root)
    monkeypatch.setattr(runtime, "REAL_LEDGER_PATH", path)
    receipt = {"dependency": "fictional-only"}
    monkeypatch.setattr(runtime, "environment_receipt", lambda root: receipt)
    legacy = AmendedLedger(
        folder,
        namespace="synthetic:h1-h2-schema-only",
        manifest_hash="fictional-manifest",
        code_hash="pre-fix-code",
        dependency_hash=digest(receipt),
    )
    old = {**legacy.identity, "namespace": NAMESPACE}
    legacy._set("identity", old)
    legacy.db.commit()
    legacy.close()
    identity = bind_existing_draft_ledger(old)
    kwargs = {
        k: identity[k] for k in ("namespace", "manifest_hash", "code_hash", "dependency_hash")
    }
    return SimpleNamespace(
        root=root, folder=folder, path=path, identity=identity, kwargs=kwargs, old=old
    )


def _data_gate(ledger, pins):
    # A fictional in-memory descriptor: failure must precede all vault/signal reads.
    access = AmendedAccess.__new__(AmendedAccess)
    access.ledger = ledger
    access._manifest = {"fictional": True}
    access._hash = digest(access._manifest)
    access.integrity_files = pins
    return access


@pytest.mark.parametrize("change", ["partial", "missing", "extra", "wrong-hash", "empty"])
def test_caller_cannot_reduce_or_extend_inventory(fictional_real, change):
    f = fictional_real
    pins = execution_inventory(f.root)
    if change == "partial":
        pins = {str(f.root / "uv.lock"): pins[str(f.root / "uv.lock")]}
    elif change == "missing":
        pins.pop(str(f.root / "src/fake.py"))
    elif change == "extra":
        pins[str(f.root / "untrusted.py")] = "extra"
    elif change == "wrong-hash":
        pins[str(f.root / "src/fake.py")] = "forged"
    else:
        pins = {}
    with AmendedLedgerContext(f) as ledger:
        gate = _data_gate(ledger, pins)
        with pytest.raises(PermissionError, match="Full expected inventory"):
            gate.rows("source_train", "fit")
        assert ledger.snapshot()["attempts"] == 0


@pytest.mark.parametrize("change", ["modify", "delete", "add"])
def test_changed_source_blocks_read_reserve_and_active_before_fit(fictional_real, change):
    f = fictional_real
    with AmendedLedgerContext(f) as ledger:
        if change == "modify":
            (f.root / "src/fake.py").write_text("CHANGED = True\n")
        elif change == "delete":
            (f.root / "src/fake.py").unlink()
        else:
            (f.root / "src/unregistered.py").write_text("EXTRA = True\n")
        # Even a complete caller map recomputed after modification is insufficient.
        gate = _data_gate(ledger, execution_inventory(f.root))
        for call in (
            lambda: gate.rows("source_train", "fit"),
            lambda: ledger.reserve("source"),
            lambda: ledger.active("no-attempt"),
        ):
            with pytest.raises(PermissionError, match="full-inventory code identity"):
                call()
        assert ledger.snapshot()["attempts"] == 0
    with pytest.raises(PermissionError, match="full-inventory code identity"):
        AmendedLedger(f.folder, **f.kwargs)


def test_dependency_drift_blocks_read_and_fit(fictional_real, monkeypatch):
    f = fictional_real
    with AmendedLedgerContext(f) as ledger:
        monkeypatch.setattr(runtime, "environment_receipt", lambda root: {"changed": True})
        gate = _data_gate(ledger, None)
        for call in (gate._verify, lambda: ledger.reserve("source"), lambda: ledger.active("x")):
            with pytest.raises(PermissionError, match="dependency identity"):
                call()
        assert ledger.snapshot()["attempts"] == 0


def test_full_trusted_inventory_and_canonical_existing_ledger_open(fictional_real):
    f = fictional_real
    assert str(UUID(f.identity["ledger_id"])) == f.identity["ledger_id"]
    assert f.identity["canonical_path"] == str(f.path.resolve())
    assert f.identity["code_hash"] == code_identity(execution_inventory(f.root))
    for _ in range(2):
        with AmendedLedgerContext(f) as ledger:
            assert ledger.identity == f.identity
            _data_gate(ledger, execution_inventory(f.root))._verify()
            _data_gate(ledger, None)._verify()
            assert ledger.get("phase") == "draft"
            assert ledger.get("protocol_adoption") is None
            with pytest.raises(PermissionError):
                ledger.reserve("source")
            assert ledger.snapshot()["attempts"] == 0


@pytest.mark.parametrize("alias", [False, True])
def test_different_or_symlink_alias_real_path_rejected(fictional_real, tmp_path, alias):
    f = fictional_real
    other = tmp_path / "other-ledger"
    if alias:
        other.symlink_to(f.folder, target_is_directory=True)
    with pytest.raises(PermissionError, match="canonical real ledger"):
        AmendedLedger(other, **f.kwargs)
    assert alias or not other.exists()


@pytest.mark.parametrize("fresh", ["absent", "empty-file", "empty-db", "unbound-db"])
def test_fresh_empty_real_db_rejected(fictional_real, tmp_path, monkeypatch, fresh):
    f = fictional_real
    folder = tmp_path / "new-canonical"
    folder.mkdir()
    path = folder / "study.sqlite"
    monkeypatch.setattr(runtime, "REAL_LEDGER_PATH", path)
    if fresh == "empty-file":
        path.touch()
    elif fresh in {"empty-db", "unbound-db"}:
        db = sqlite3.connect(path)
        db.execute("CREATE TABLE meta(k TEXT PRIMARY KEY,v TEXT NOT NULL)")
        if fresh == "unbound-db":
            db.execute("INSERT INTO meta VALUES ('identity',?)", (json.dumps(f.old),))
        db.commit()
        db.close()
    before = path.read_bytes() if path.exists() else None
    with pytest.raises(PermissionError):
        AmendedLedger(folder, **f.kwargs)
    assert (path.read_bytes() if path.exists() else None) == before


def test_file_symlink_alias_rejected(fictional_real, tmp_path):
    f = fictional_real
    moved = tmp_path / "moved.sqlite"
    f.path.rename(moved)
    f.path.symlink_to(moved)
    with pytest.raises(PermissionError, match="canonical real ledger"):
        AmendedLedger(f.folder, **f.kwargs)


def test_uuid_binding_is_once_only_and_does_not_grant_permissions(fictional_real):
    f = fictional_real
    with pytest.raises(PermissionError):
        bind_existing_draft_ledger(f.old)
    with AmendedLedgerContext(f) as ledger:
        assert ledger.identity == f.identity
        assert ledger.get("ledger_id") == f.identity["ledger_id"]
        assert ledger.get("canonical_path") == str(f.path)
        snapshot = ledger.snapshot()
        assert snapshot["phase"] == "draft"
        assert snapshot["attempts"] == snapshot["evaluation_accesses"] == 0
        assert not snapshot["development_authorized"] and not snapshot["evaluation_authorized"]
        assert ledger.get("protocol_adoption") is None


@pytest.mark.parametrize("key", ["ledger_id", "canonical_path"])
def test_authorizations_require_instance_bound_identity(fictional_real, key):
    f = fictional_real
    # All mutations below are to this temporary fictional DB, never real approvals.
    with AmendedLedgerContext(f) as ledger:
        from .test_integration import fictional_adoption

        fictional_adoption(ledger)
        receipt = copy.deepcopy(ledger.get("development_authorization"))
        assert receipt[key] == ledger.identity[key]
        ledger._set("phase", "draft")
        ledger.db.commit()
        receipt.pop(key)
        with pytest.raises(PermissionError, match="development authorization"):
            ledger.authorize_development(receipt)
        ledger._set("phase", "locked")
        ledger._set("lock", {"fictional": True})
        ledger.db.commit()
        evaluation = {
            **ledger.identity,
            "statement": "explicitly authorize one phase2d internal evaluation",
            "lock_hash": digest(ledger.get("lock")),
        }
        evaluation.pop(key)
        with pytest.raises(PermissionError, match="evaluation authorization"):
            ledger.authorize_evaluation(evaluation)


class AmendedLedgerContext:
    def __init__(self, fixture):
        self.fixture = fixture

    def __enter__(self):
        self.ledger = AmendedLedger(self.fixture.folder, **self.fixture.kwargs)
        return self.ledger

    def __exit__(self, *exc):
        self.ledger.close()
