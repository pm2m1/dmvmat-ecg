"""Independent durable 36-attempt ledger and one-time evaluation lease.

Authorization records attest an operator's separately obtained human permission;
the API cannot establish that permission by itself. Never authorize real work
merely because this module can record a receipt.
"""

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path

from .contract import contract, digest, file_hash


def atomic_bytes(path, data):
    """Durable no-overwrite publication; interrupted temp files are never artifacts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".pending-" + uuid.uuid4().hex)
    with temp.open("xb") as f:
        os.chmod(temp, 0o600)
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(temp, path)  # Atomic exclusive publication: cannot replace existing artifact.
    finally:
        temp.unlink()
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return file_hash(path)


class PilotLedger:
    def __init__(
        self, folder, *, namespace, manifest_hash, code_hash, dependency_hash, config=None
    ):
        self.folder = Path(folder).resolve()
        self.folder.mkdir(parents=True, exist_ok=True)
        if not (namespace == "real:phase2b-internal" or namespace.startswith("synthetic:")):
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
        existing = self.folder / "study.sqlite"
        if existing.exists():
            probe = sqlite3.connect("file:" + str(existing) + "?mode=ro", uri=True)
            try:
                row = probe.execute("SELECT v FROM meta WHERE k='identity'").fetchone()
                if row is None or json.loads(row[0]) != self.identity:
                    raise PermissionError("Refuse historical, foreign or changed study ledger")
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

    def close(self):
        self.db.close()

    def get(self, key):
        row = self.db.execute("SELECT v FROM meta WHERE k=?", (key,)).fetchone()
        return None if row is None else json.loads(row[0])

    def _set(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, json.dumps(value)))

    def event(self, kind, detail):
        self.db.execute(
            "INSERT INTO events(kind,detail) VALUES (?,?)",
            (kind, json.dumps(detail, sort_keys=True)),
        )
        self.db.commit()

    def authorize_development(self, receipt):
        expected = {
            **self.identity,
            "statement": "explicitly authorize phase2b internal development",
            "adopt_draft_contract": True,
        }
        if (
            receipt != expected
            or self.get("phase") != "draft"
            or self.get("preflight_blockers")
            or (
                self.identity["namespace"].startswith("real:")
                and self.get("preflight_audit") is None
            )
        ):
            raise PermissionError("Separate exact development authorization/adoption required")
        self._set("development_authorization", receipt)
        self._set("phase", "development")
        self.db.commit()

    def record_preflight(self, blockers, audit_hash):
        if self.get("phase") != "draft" or self.snapshot()["attempts"]:
            raise PermissionError("Preflight cannot alter an executing study")
        if self.get("preflight_audit") is not None:
            if (
                self.get("preflight_audit") != audit_hash
                or self.get("preflight_blockers") != blockers
            ):
                raise PermissionError(
                    "Preflight is immutable; prospective review requires a new version"
                )
            return
        self._set("preflight_audit", audit_hash)
        self._set("preflight_blockers", list(blockers))
        self.db.commit()

    def reserve(self, job, *, retry_of=None, operation="registered_fit"):
        if self.get("phase") != "development" or not self.get("development_authorization"):
            raise PermissionError("Predictor fitting DENIED")
        if job not in self.config["planned_jobs"]:
            raise PermissionError("Unregistered candidate or smoke is not a free attempt")
        if operation not in {"registered_fit", "compatibility_fit"}:
            raise PermissionError("Unknown fit operation")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            count = self.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
            if count >= 36:
                raise RuntimeError("36-attempt ceiling exhausted")
            if (
                self.identity["namespace"].startswith("real:")
                and self.db.execute(
                    "SELECT COUNT(*) FROM attempts WHERE status='attempted'"
                ).fetchone()[0]
                >= 2
            ):
                raise PermissionError("Real CPU worker ceiling is two active attempts")
            previous = self.db.execute(
                "SELECT id,status FROM attempts WHERE job=? ORDER BY rowid DESC", (job,)
            ).fetchone()
            extra = self.db.execute(
                "SELECT COUNT(*) FROM attempts WHERE retry_of IS NOT NULL"
            ).fetchone()[0]
            if previous:
                if (
                    previous[1] not in {"failed", "interrupted"}
                    or retry_of != previous[0]
                    or extra >= 3
                ):
                    raise PermissionError(
                        "Duplicate/completed/active job or retry reserve exhausted"
                    )
            elif retry_of is not None:
                raise PermissionError("Retry has no parent")
            # A real-data compatibility fit is permitted only as a technical retry
            # of the same exact registered config, never as extra candidate search.
            if operation == "compatibility_fit" and previous is None:
                raise PermissionError("Compatibility probes require reviewed retry lineage")
            identifier = uuid.uuid4().hex
            spec = self.config["planned_jobs"][job]
            if spec["kind"] == "target" and "source" not in self.successful():
                raise PermissionError("New source pretraining must finish before target fits")
            identity = {
                **self.identity,
                "job_spec": spec,
                "pid": os.getpid(),
                "operation": operation,
                "fresh_initialization": True,
            }
            self.db.execute(
                "INSERT INTO attempts VALUES (?,?,?,?,?)",
                (identifier, job, "attempted", retry_of, json.dumps(identity, sort_keys=True)),
            )
            self.db.commit()
            return identifier
        except BaseException:
            self.db.rollback()
            raise

    def active(self, identifier):
        row = self.db.execute(
            "SELECT job,status,identity FROM attempts WHERE id=?", (identifier,)
        ).fetchone()
        if row is None or row[1] != "attempted" or self.get("phase") != "development":
            raise PermissionError("No active reserved predictor attempt")
        identity = json.loads(row[2])
        if any(identity[k] != v for k, v in self.identity.items()):
            raise PermissionError("Attempt identity differs")
        return row[0], identity["job_spec"]

    def publish(self, identifier, name, data):
        self.active(identifier)
        if not name or Path(name).name != name or name.startswith(".") or ".pending" in name:
            raise ValueError("Artifact must have a simple immutable name")
        path = self.folder / "artifacts" / identifier / name
        h = atomic_bytes(path, data)
        self.db.execute(
            "INSERT INTO artifacts VALUES (?,?,?,?)",
            (identifier, name, str(path.relative_to(self.folder)), h),
        )
        self.db.commit()
        return dict(attempt=identifier, name=name, sha=h)

    def artifact(self, identifier, name):
        row = self.db.execute(
            "SELECT path,sha FROM artifacts WHERE attempt=? AND name=?", (identifier, name)
        ).fetchone()
        if row is None:
            raise PermissionError("Missing/unregistered/incomplete artifact")
        path = (self.folder / row[0]).resolve()
        if not path.is_relative_to(self.folder / "artifacts") or file_hash(path) != row[1]:
            raise PermissionError("Artifact path/hash mismatch")
        return path

    def finish(self, identifier, status, detail=None):
        if status not in {"successful", "failed", "interrupted"}:
            raise ValueError("Unknown terminal status")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.active(identifier)
            if status == "successful":
                for name in ("checkpoint.pt", "lineage.json"):
                    self.artifact(identifier, name)
                lineage = json.loads(self.artifact(identifier, "lineage.json").read_text())
                required = {
                    "namespace": self.identity["namespace"],
                    "manifest_hash": self.identity["manifest_hash"],
                    "job_id": self.active(identifier)[0],
                    "code_hash": self.identity["code_hash"],
                    "dependency_hash": self.identity["dependency_hash"],
                    "labels": self.config["labels"],
                    "checkpoint_sha256": file_hash(self.artifact(identifier, "checkpoint.pt")),
                }
                if any(lineage.get(k) != v for k, v in required.items()):
                    raise PermissionError("Completed checkpoint lineage is invalid")
            self.db.execute(
                "UPDATE attempts SET status=? WHERE id=? AND status='attempted'",
                (status, identifier),
            )
            self.db.execute(
                "INSERT INTO events(kind,detail) VALUES (?,?)",
                (status, json.dumps(dict(attempt=identifier, detail=detail))),
            )
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise

    def successful(self):
        return dict(self.db.execute("SELECT job,id FROM attempts WHERE status='successful'"))

    def record_calibration(self, job, artifact_bytes):
        if self.get("phase") != "development" or job not in self.successful():
            raise PermissionError("Calibration requires a completed registered predictor")
        meta = json.loads(artifact_bytes)
        if (
            meta.get("namespace") != self.identity["namespace"]
            or meta.get("job_id") != job
            or meta.get("fit_role") != "C"
            or meta.get("manifest_hash") != self.identity["manifest_hash"]
        ):
            raise PermissionError("Wrong calibration lineage")
        accesses = [
            json.loads(v)
            for (v,) in self.db.execute("SELECT detail FROM events WHERE kind='data_access'")
        ]
        if not any(
            a["role"] == "C" and a["operation"] == "calibrate" and a["job"] == job for a in accesses
        ):
            raise PermissionError("Calibration has no authorized C access receipt")
        path = self.folder / "calibration" / (job.replace(":", "_") + ".json")
        h = atomic_bytes(path, artifact_bytes)
        self.event("calibration", dict(job=job, path=str(path.relative_to(self.folder)), sha=h))

    def lock(self):
        if self.get("phase") != "development" or set(self.successful()) != set(
            self.config["planned_jobs"]
        ):
            raise PermissionError("All required registered fits must finish before lock")
        if self.db.execute("SELECT COUNT(*) FROM attempts WHERE status='attempted'").fetchone()[0]:
            raise PermissionError("Interrupted/active work unresolved")
        cal = [
            json.loads(v)
            for (v,) in self.db.execute("SELECT detail FROM events WHERE kind='calibration'")
        ]
        if {c["job"] for c in cal} != set(self.config["planned_jobs"]):
            raise PermissionError("Calibration or explicit fallback artifacts incomplete")
        files = {}
        for identifier in self.successful().values():
            for (name,) in self.db.execute(
                "SELECT name FROM artifacts WHERE attempt=?", (identifier,)
            ):
                p = self.artifact(identifier, name)
                files[str(p.relative_to(self.folder))] = file_hash(p)
        for c in cal:
            p = self.folder / c["path"]
            if file_hash(p) != c["sha"]:
                raise PermissionError("Calibration changed")
            files[c["path"]] = c["sha"]
        bundle = dict(
            identity=self.identity,
            files=files,
            attempts=self.snapshot()["attempts"],
            successful=self.successful(),
        )
        self._set("lock", bundle)
        self._set("phase", "locked")
        self.db.commit()
        return digest(bundle)

    def authorize_evaluation(self, receipt):
        expected = {
            **self.identity,
            "statement": "explicitly authorize one phase2b internal evaluation",
            "lock_hash": digest(self.get("lock")),
        }
        if self.get("phase") != "locked" or receipt != expected:
            raise PermissionError("Separate exact evaluation authorization required")
        self._set("evaluation_authorization", receipt)
        self.db.commit()

    def begin_evaluation(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            if (
                self.get("phase") != "locked"
                or not self.get("evaluation_authorization")
                or self.db.execute("SELECT COUNT(*) FROM access").fetchone()[0]
            ):
                raise PermissionError("One-time evaluation DENIED/consumed")
            lock = self.get("lock")
            for p, h in lock["files"].items():
                if file_hash(self.folder / p) != h:
                    raise PermissionError("Locked artifact changed")
            token = uuid.uuid4().hex
            self.db.execute(
                "INSERT INTO access VALUES (?,?)", (token, json.dumps(dict(lock=digest(lock))))
            )
            self._set("phase", "evaluating")
            self.db.commit()
            return token
        except BaseException:
            self.db.rollback()
            raise

    def require_evaluation(self, token):
        if (
            self.get("phase") != "evaluating"
            or not token
            or self.db.execute("SELECT 1 FROM access WHERE id=?", (token,)).fetchone() is None
        ):
            raise PermissionError("No active evaluation lease")

    @contextmanager
    def evaluation_session(self):
        token = self.begin_evaluation()
        try:
            yield token
            self.finish_evaluation(token)
        except BaseException:
            if self.get("phase") == "evaluating":
                self.finish_evaluation(token, successful=False)
            raise

    def publish_predictions(self, token, job, payload):
        self.require_evaluation(token)
        if job not in self.config["planned_jobs"] or self.get("prediction_seal"):
            raise PermissionError("Unregistered/post-seal prediction")
        path = self.folder / "evaluation" / token / (job.replace(":", "_") + ".npz")
        h = atomic_bytes(path, payload)
        self.event(
            "prediction", dict(token=token, job=job, path=str(path.relative_to(self.folder)), sha=h)
        )

    def seal_predictions(self, token):
        self.require_evaluation(token)
        records = [
            json.loads(v)
            for (v,) in self.db.execute("SELECT detail FROM events WHERE kind='prediction'")
        ]
        if {r["job"] for r in records} != set(self.config["planned_jobs"]) or self.get(
            "prediction_seal"
        ):
            raise PermissionError("Complete registered prediction set must be sealed once")
        for r in records:
            if r["token"] != token or file_hash(self.folder / r["path"]) != r["sha"]:
                raise PermissionError("Prediction integrity mismatch")
        path = self.folder / "evaluation" / token / "seal.json"
        h = atomic_bytes(path, json.dumps(records, sort_keys=True).encode())
        self._set("prediction_seal", dict(path=str(path.relative_to(self.folder)), sha=h))
        self.db.commit()

    def require_metric_access(self, token):
        self.require_evaluation(token)
        seal = self.get("prediction_seal")
        if seal is None or file_hash(self.folder / seal["path"]) != seal["sha"]:
            raise PermissionError("Metrics/labels denied before complete prediction seal")
        for r in json.loads((self.folder / seal["path"]).read_text()):
            if file_hash(self.folder / r["path"]) != r["sha"]:
                raise PermissionError("Sealed predictions changed")

    def finish_evaluation(self, token, *, successful=True):
        self.require_evaluation(token)
        if successful:
            self.require_metric_access(token)
        self._set("phase", "evaluated" if successful else "evaluation_failed")
        self.db.commit()

    def snapshot(self):
        return dict(
            identity=self.identity,
            phase=self.get("phase"),
            attempts=self.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0],
            statuses=dict(self.db.execute("SELECT status,COUNT(*) FROM attempts GROUP BY status")),
            evaluation_accesses=self.db.execute("SELECT COUNT(*) FROM access").fetchone()[0],
            development_authorized=bool(self.get("development_authorization")),
            evaluation_authorized=bool(self.get("evaluation_authorization")),
            preflight_blockers=self.get("preflight_blockers") or [],
        )
