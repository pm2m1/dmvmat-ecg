"""Small durable synthetic run log and exact-job cache; no real execution grants."""

import hashlib
import json
import os
import sqlite3
import tempfile
import uuid
from pathlib import Path

from .protocol import CEILING, RESERVES, adopted_settings, canonical, digest, roster
from .roles import synthetic_only


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_new(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".writing-")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # link is atomic and refuses to overwrite a completed artifact.
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


class RunLog:
    def __init__(self, path, namespace, *, jobs=None):
        synthetic_only(namespace)
        self.path = Path(path).resolve()
        self.namespace = namespace
        self.jobs = roster() if jobs is None else jobs
        identity = digest(dict(namespace=namespace, protocol=adopted_settings(), jobs=self.jobs))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS meta (identity TEXT PRIMARY KEY, namespace TEXT);
                CREATE TABLE IF NOT EXISTS attempts (
                    id TEXT PRIMARY KEY, slot TEXT, job TEXT, status TEXT, identity TEXT,
                    artifact TEXT, sha TEXT, failure TEXT, retry INTEGER);
                CREATE TABLE IF NOT EXISTS aliases (
                    slot TEXT PRIMARY KEY, job TEXT, attempt TEXT);
                CREATE TABLE IF NOT EXISTS evaluation (
                    id INTEGER PRIMARY KEY CHECK (id=1), status TEXT, frozen TEXT,
                    predictions TEXT, seal TEXT);
            """)
            row = db.execute("SELECT identity FROM meta").fetchone()
            if row and row[0] != identity:
                raise PermissionError("Run namespace/protocol/roster differs on restart")
            if not row:
                db.execute("INSERT INTO meta VALUES (?,?)", (identity, namespace))

    def connect(self):
        db = sqlite3.connect(self.path, timeout=20)
        db.execute("PRAGMA synchronous=FULL")
        db.row_factory = sqlite3.Row
        return db

    def validate_slot(self, slot, **fields):
        if slot not in self.jobs:
            raise PermissionError("Unregistered fit slot")
        spec = self.jobs[slot]
        if any(k in spec and spec[k] != v for k, v in fields.items()):
            raise PermissionError("Fit entry differs from its registered matrix slot")

    def reserve(self, slot, job, identity, *, retry=False):
        if slot not in self.jobs:
            raise PermissionError("Unregistered fit slot")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM evaluation").fetchone():
                raise PermissionError("No fits after candidate freezing")
            existing = db.execute("SELECT * FROM aliases WHERE slot=?", (slot,)).fetchone()
            if existing:
                if existing["job"] != job:
                    raise PermissionError("Slot cannot be rescheduled with different settings")
            earlier = db.execute("SELECT job FROM attempts WHERE slot=?", (slot,)).fetchall()
            if any(r["job"] != job for r in earlier):
                raise PermissionError("A failed slot cannot change predictive settings")
            rows = db.execute(
                "SELECT * FROM attempts WHERE job=? ORDER BY rowid", (job,)
            ).fetchall()
            completed = [r for r in rows if r["status"] == "successful"]
            if completed:
                r = completed[-1]
                if file_hash(r["artifact"]) != r["sha"]:
                    raise PermissionError("Completed checkpoint corrupted; do not regenerate")
                db.execute("INSERT OR IGNORE INTO aliases VALUES (?,?,?)", (slot, job, r["id"]))
                return dict(r), True
            if rows and rows[-1]["status"] == "attempted":
                raise RuntimeError("In-flight/interrupted job requires explicit failure recording")
            if rows and not retry:
                raise RuntimeError("Failed job requires an explicit fresh technical retry")
            if retry and (not rows or len(rows) >= 2):
                raise RuntimeError("At most one fresh technical retry per failed job")
            if retry and json.loads(rows[-1]["failure"])["kind"] != "environmental_interruption":
                raise PermissionError("Scientific/numerical failures require review, not a retry")
            counts = db.execute("SELECT COUNT(*), COALESCE(SUM(retry),0) FROM attempts").fetchone()
            if counts[0] >= CEILING or (retry and counts[1] >= RESERVES):
                raise RuntimeError("Hard attempt/reserve ceiling exhausted")
            if rows and rows[-1]["identity"] != digest(identity):
                raise PermissionError("Retry must retain the original predictive identity")
            attempt = uuid.uuid4().hex
            db.execute(
                "INSERT INTO attempts VALUES (?,?,?,?,?,?,?,?,?)",
                (attempt, slot, job, "attempted", digest(identity), None, None, None, int(retry)),
            )
            return dict(id=attempt, slot=slot, job=job), False

    def complete(self, attempt, payload):
        path = self.path.parent / "artifacts" / f"{attempt}.pt"
        atomic_new(path, payload)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM attempts WHERE id=?", (attempt,)).fetchone()
            if row is None or row["status"] != "attempted":
                raise RuntimeError("Only a reserved active attempt may complete")
            db.execute(
                "UPDATE attempts SET status='successful',artifact=?,sha=? WHERE id=?",
                (str(path), file_hash(path), attempt),
            )
            db.execute(
                "INSERT OR IGNORE INTO aliases VALUES (?,?,?)", (row["slot"], row["job"], attempt)
            )
        return path

    def fail(self, attempt, reason, *, kind="review_required"):
        if kind not in {"environmental_interruption", "review_required"}:
            raise ValueError("Undeclared failure category")
        with self.connect() as db:
            result = db.execute(
                "UPDATE attempts SET status='failed',failure=? WHERE id=? AND status='attempted'",
                (canonical(dict(reason=reason, kind=kind)).decode(), attempt),
            )
            if result.rowcount != 1:
                raise RuntimeError("Cannot rewrite immutable completed/failed attempt")

    def summary(self):
        with self.connect() as db:
            rows = [dict(r) for r in db.execute("SELECT * FROM attempts ORDER BY rowid")]
            aliases = [dict(r) for r in db.execute("SELECT * FROM aliases ORDER BY slot")]
        return dict(
            namespace=self.namespace,
            attempts=rows,
            used=len(rows),
            aliases=aliases,
            ceiling=CEILING,
            reserves_used=sum(r["retry"] for r in rows),
        )

    def execute(self, slot, identity, fit, *, retry=False):
        job = digest(identity)
        reservation, alias = self.reserve(slot, job, identity, retry=retry)
        if alias:
            return Path(reservation["artifact"]), True
        try:
            path = self.complete(reservation["id"], fit())
        except BaseException as exc:
            self.fail(
                reservation["id"],
                f"{type(exc).__name__}: {exc}",
                kind="environmental_interruption"
                if isinstance(exc, (OSError, KeyboardInterrupt))
                else "review_required",
            )
            raise
        return path, False


def source_code_hash():
    # Include reused numerical/backbone code; neither source identity nor Git HEAD alone suffices.
    package = Path(__file__).resolve().parents[1]
    paths = list((package / "phase3b").glob("*.py"))
    paths += [
        package / p
        for p in (
            "models/networks.py",
            "models/heads.py",
            "preprocessing/signals.py",
            "phase2/adaptation.py",
            "phase2/contracts.py",
            "phase2/evidence.py",
        )
    ]
    values = {str(p.relative_to(package)): file_hash(p) for p in sorted(paths)}
    root = package.parents[1]
    for name in (
        "pyproject.toml",
        "uv.lock",
        "configs/phase3b/adopted_v1.json",
        "configs/phase3b/fit_matrix_v1.json",
    ):
        values[name] = file_hash(root / name)
    return digest(values)


def save_json(path, value):
    atomic_new(path, canonical(value) + b"\n")


def read_json(path):
    return json.loads(Path(path).read_text())
