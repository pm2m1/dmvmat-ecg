"""Durable finite-fit ledger and single-use joint evaluation lease."""

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path

from dmvmat_ecg.utils.io import digest, sha, source_hash


class Study:
    def __init__(self, root):
        self.root = Path(root)
        self.active_evaluation = False
        self.path = self.root / "private/study.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY,v TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY,slot TEXT UNIQUE,status TEXT,metadata TEXT);
        CREATE TABLE IF NOT EXISTS access(id TEXT PRIMARY KEY,kind TEXT,detail TEXT);""")
        if self.get("phase") is None:
            self.set("phase", "data")

    def get(self, k):
        row = self.db.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
        return None if row is None else json.loads(row[0])

    def set(self, k, value):
        self.db.execute(
            "INSERT OR REPLACE INTO meta VALUES (?,?)", (k, json.dumps(value, sort_keys=True))
        )
        self.db.commit()

    def require(self, role):
        phase = self.get("phase")
        if role in ("train", "validation") and phase != "development":
            raise PermissionError("Development is closed")
        if role == "test" and not (phase == "evaluated" and self.active_evaluation):
            raise PermissionError("Test partition is locked")
        if role not in ("train", "validation", "test"):
            raise PermissionError("Unknown role")

    def reserve(self, name, seed, stage, config):
        if self.get("phase") != "development":
            raise PermissionError("Predictor fitting is closed")
        if (
            config != json.loads((self.root / "configs/study.json").read_text())
            or name not in config["models"]
        ):
            raise PermissionError("Unregistered model/configuration")
        if (stage == "full" and seed not in config["seeds"]) or stage not in ("full", "smoke"):
            raise PermissionError("Unregistered stage/seed")
        if stage == "smoke" and (seed != config["smoke"]["seed"] or name == "multiview_aug"):
            raise PermissionError("Only three architecture smoke trials are registered")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            count = self.db.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            if count >= 13:
                raise RuntimeError("Registered 13-attempt total exhausted")
            slot = f"{stage}:{name}:{seed}"
            previous = self.db.execute(
                "SELECT id,status,metadata FROM runs WHERE slot=?", (slot,)
            ).fetchone()
            retry_of = None
            if previous:
                if previous[1] != "failed" or json.loads(previous[2])["configuration"] != config:
                    raise PermissionError("Successful/active/different trial cannot be repeated")
                retry_of = previous[0]
                slot += ":retry1"
                if self.db.execute("SELECT 1 FROM runs WHERE slot=?", (slot,)).fetchone():
                    raise PermissionError("Only one fresh repair retry per failed trial")
            number = uuid.uuid4().hex
            meta = dict(
                model=name,
                seed=seed,
                stage=stage,
                configuration=config,
                source_hash=source_hash(self.root),
                pid=os.getpid(),
                retry_of=retry_of,
            )
            self.db.execute(
                "INSERT INTO runs VALUES (?,?,?,?)",
                (number, slot, "attempted", json.dumps(meta, sort_keys=True)),
            )
            self.db.commit()
            return number
        except BaseException:
            self.db.rollback()
            raise

    def finish(self, number, status, metadata):
        if status not in ("successful", "failed"):
            raise ValueError(status)
        row = self.db.execute("SELECT status,metadata FROM runs WHERE id=?", (number,)).fetchone()
        if row is None or row[0] != "attempted":
            raise PermissionError("Cannot rewrite terminal trial")
        merged = {**json.loads(row[1]), **metadata}
        self.db.execute(
            "UPDATE runs SET status=?,metadata=? WHERE id=?",
            (status, json.dumps(merged, sort_keys=True), number),
        )
        self.db.commit()

    def runs(self):
        return [
            dict(id=r[0], slot=r[1], status=r[2], metadata=json.loads(r[3]))
            for r in self.db.execute("SELECT * FROM runs ORDER BY rowid")
        ]

    def lock(self, artifacts):
        if self.get("phase") != "development":
            raise PermissionError("Already locked/accessed")
        successful = [
            r
            for r in self.runs()
            if r["status"] == "successful" and r["metadata"]["stage"] == "full"
        ]
        if len(successful) != 8:
            raise RuntimeError("All eight registered full fits are required")
        config = json.loads((self.root / "configs/study.json").read_text())
        if {(r["metadata"]["model"], r["metadata"]["seed"]) for r in successful} != {
            (m, s) for m in config["models"] for s in config["seeds"]
        }:
            raise RuntimeError("Required paired model families are missing")
        self.set(
            "lock",
            dict(
                files={str(Path(p).relative_to(self.root)): sha(p) for p in artifacts},
                source_hash=source_hash(self.root),
            ),
        )
        self.set("phase", "locked")

    @contextmanager
    def evaluate_once(self, authorization):
        if self.get("phase") != "locked" or self.get("evaluation_consumed"):
            raise PermissionError("Evaluation lease is absent/consumed")
        if authorization != "user-authorized-single-final-PTBXL-evaluation":
            raise PermissionError("Missing explicit evaluation scope")
        lock = self.get("lock")
        if source_hash(self.root) != lock["source_hash"] or any(
            sha(self.root / p) != h for p, h in lock["files"].items()
        ):
            raise PermissionError("Lock integrity changed")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            if self.get("evaluation_consumed"):
                raise PermissionError("Concurrent/repeated evaluation refused")
            self.db.execute(
                "INSERT INTO access VALUES (?,?,?)",
                (
                    uuid.uuid4().hex,
                    "joint_final_evaluation",
                    json.dumps(dict(lock_hash=digest(lock), scope=authorization)),
                ),
            )
            self.db.execute(
                "INSERT OR REPLACE INTO meta VALUES (?,?)", ("evaluation_consumed", "true")
            )
            self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", ("phase", '"evaluated"'))
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise
        self.active_evaluation = True
        try:
            yield
        finally:
            self.active_evaluation = False
