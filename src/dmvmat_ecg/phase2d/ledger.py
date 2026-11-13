"""Versioned constructor and separate adoption/training gates; inherited crash-safe ledger.

Reservation, publication, accounting and evaluation sealing reuse the immutable
Phase2B implementation. No old namespace or ledger identity is aliased.
"""

import json
import sqlite3
import uuid
from pathlib import Path

from dmvmat_ecg.phase2b.ledger import PilotLedger

from .contract import NAMESPACE, contract, digest, validate_adoption
from .runtime import real_ledger_path, verify_execution_identity


def _real_metadata(db, path):
    """An existing, explicitly bound database is required, never a fresh DB."""
    try:
        meta = {k: json.loads(v) for k, v in db.execute("SELECT k,v FROM meta")}
        ledger_id = meta["ledger_id"]
        if str(uuid.UUID(ledger_id)) != ledger_id or meta["canonical_path"] != str(path):
            raise ValueError("Invalid real ledger UUID/path")
        identity = meta["identity"]
        if identity["ledger_id"] != ledger_id or identity["canonical_path"] != str(path):
            raise ValueError("Real ledger identity is not instance-bound")
        for table in ("attempts", "artifacts", "events", "access"):
            db.execute(f"SELECT 1 FROM {table} LIMIT 0")
        return identity
    except (sqlite3.DatabaseError, KeyError, ValueError, TypeError, AttributeError) as error:
        raise PermissionError("Empty/unbound/foreign real ledger rejected") from error


def bind_existing_draft_ledger(expected_identity):
    """Explicit one-time H1/H2 metadata migration, never an authorization grant.

    Only a known, existing, unsigned draft with zero attempts/accesses may acquire
    its UUID/path and the new full-inventory hash. No constructor calls this.
    All other metadata and tables are preserved. No models/data are accessed.
    """
    from dmvmat_ecg.phase2b.runtime import code_identity, execution_inventory

    from .runtime import REAL_LEDGER_PATH, REAL_PROJECT_ROOT

    path = real_ledger_path(REAL_LEDGER_PATH.parent)
    keys = {"namespace", "manifest_hash", "code_hash", "dependency_hash", "contract_hash"}
    if (
        set(expected_identity) != keys
        or expected_identity["namespace"] != NAMESPACE
        or expected_identity["contract_hash"] != digest(contract())
    ):
        raise PermissionError("Exact pre-migration study identity required")
    updated = {
        **expected_identity,
        "code_hash": code_identity(execution_inventory(REAL_PROJECT_ROOT)),
    }
    verify_execution_identity(updated)
    db = sqlite3.connect(path.as_uri() + "?mode=rw", uri=True, timeout=30)
    try:
        db.execute("PRAGMA synchronous=FULL")
        db.execute("BEGIN IMMEDIATE")
        meta = {k: json.loads(v) for k, v in db.execute("SELECT k,v FROM meta")}
        if (
            meta.get("identity") != expected_identity
            or meta.get("phase") != "draft"
            or set(meta) - {"identity", "phase", "preflight_audit", "preflight_blockers"}
            or any(
                db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("attempts", "artifacts", "access")
            )
        ):
            raise PermissionError("Migration requires untouched unsigned zero-attempt draft")
        updated.update(ledger_id=str(uuid.uuid4()), canonical_path=str(path))
        for key in ("ledger_id", "canonical_path"):
            db.execute("INSERT INTO meta(k,v) VALUES (?,?)", (key, json.dumps(updated[key])))
        db.execute("UPDATE meta SET v=? WHERE k='identity'", (json.dumps(updated),))
        db.commit()
        return updated
    except (sqlite3.DatabaseError, ValueError, TypeError) as error:
        raise PermissionError("Cannot migrate empty/foreign real ledger") from error
    finally:
        db.close()


class AmendedLedger(PilotLedger):
    def _verify_contract(self):
        if digest(self.config) != self.identity["contract_hash"]:
            raise PermissionError("Live execution contract changed")
        if self.identity["namespace"].startswith("real:") and self.config != contract():
            raise PermissionError("Unapproved real execution contract")
        if self.identity["namespace"].startswith("real:"):
            path = real_ledger_path(self.folder)
            if _real_metadata(self.db, path) != self.identity:
                raise PermissionError("Live authoritative ledger identity changed")
            verify_execution_identity(self.identity)

    def reserve(self, job, *, retry_of=None, operation="registered_fit"):
        self._verify_contract()
        return super().reserve(job, retry_of=retry_of, operation=operation)

    def active(self, identifier):
        self._verify_contract()
        return super().active(identifier)

    def begin_evaluation(self):
        self._verify_contract()
        return super().begin_evaluation()

    def __init__(
        self, folder, *, namespace, manifest_hash, code_hash, dependency_hash, config=None
    ):
        if not (namespace == NAMESPACE or namespace.startswith("synthetic:")):
            raise ValueError("Unknown namespace")
        self.config = contract() if config is None else config
        if namespace.startswith("real:") and self.config != contract():
            raise ValueError("Real study must retain all 33 proposed slots and 36 ceiling")
        if self.config["predictor_attempt_ceiling"] != 36:
            raise ValueError("No alternative attempt ceiling")
        self.identity = dict(
            namespace=namespace,
            manifest_hash=manifest_hash,
            code_hash=code_hash,
            dependency_hash=dependency_hash,
            contract_hash=digest(self.config),
        )
        if namespace == NAMESPACE:
            existing = real_ledger_path(folder)
            self.folder = existing.parent
            self.db = sqlite3.connect(existing.as_uri() + "?mode=rw", uri=True, timeout=30)
            try:
                bound = _real_metadata(self.db, existing)
                self.identity.update(
                    ledger_id=bound["ledger_id"], canonical_path=bound["canonical_path"]
                )
                if bound != self.identity:
                    raise PermissionError(
                        "Real ledger namespace/manifest/source/dependency changed"
                    )
                self._verify_contract()
            except BaseException:
                self.db.close()
                raise
            return
        self.folder = Path(folder).resolve()
        self.folder.mkdir(parents=True, exist_ok=True)
        existing = self.folder / "study.sqlite"
        if existing.exists():
            probe = sqlite3.connect("file:" + str(existing) + "?mode=ro", uri=True)
            try:
                row = probe.execute("SELECT v FROM meta WHERE k='identity'").fetchone()
                if row is None or json.loads(row[0]) != self.identity:
                    raise PermissionError("Refuse historical, foreign or changed study ledger")
            except sqlite3.DatabaseError as error:
                raise PermissionError("Historical/foreign ledger schema rejected") from error
            finally:
                probe.close()
        self.db = sqlite3.connect(self.folder / "study.sqlite", timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY,v TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS attempts(id TEXT PRIMARY KEY,job TEXT,status TEXT,
          retry_of TEXT,identity TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS artifacts(attempt TEXT,name TEXT,path TEXT,sha TEXT,
          PRIMARY KEY(attempt,name));
        CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY,kind TEXT,detail TEXT);
        CREATE TABLE IF NOT EXISTS access(id TEXT PRIMARY KEY,detail TEXT);
        """)
        if self.get("identity") is None:
            self._set("identity", self.identity)
            self._set("phase", "draft")
            self.db.commit()
        elif self.get("identity") != self.identity:
            raise PermissionError("Namespace/manifest/source/dependency/contract changed")

    def record_adoption(self, review):
        self._verify_contract()
        validate_adoption(review)
        if self.get("phase") != "draft" or self.get("protocol_adoption"):
            raise PermissionError("Adoption is immutable and precedes development")
        self._set("protocol_adoption", review)
        self.db.commit()

    def authorize_development(self, receipt):
        self._verify_contract()
        expected = {
            **self.identity,
            "statement": "explicitly authorize phase2d internal development",
            "adoption_hash": digest(self.get("protocol_adoption")),
        }
        if (
            self.get("phase") != "draft"
            or not self.get("protocol_adoption")
            or self.get("preflight_audit") is None
            or self.get("preflight_blockers")
            or receipt != expected
        ):
            raise PermissionError("Separate human adoption AND development authorization required")
        validate_adoption(self.get("protocol_adoption"))
        self._set("development_authorization", receipt)
        self._set("phase", "development")
        self.db.commit()

    def authorize_evaluation(self, receipt):
        self._verify_contract()
        expected = {
            **self.identity,
            "statement": "explicitly authorize one phase2d internal evaluation",
            "lock_hash": digest(self.get("lock")),
        }
        if self.get("phase") != "locked" or receipt != expected:
            raise PermissionError("Separate exact one-time evaluation authorization required")
        self._set("evaluation_authorization", receipt)
        self.db.commit()
