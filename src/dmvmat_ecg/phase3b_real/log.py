"""One real run log, same finite accounting; no synthetic relabeling or O permissions."""

import json
from pathlib import Path

from dmvmat_ecg.phase3b.jobs import RunLog, file_hash
from dmvmat_ecg.phase3b.protocol import adopted_settings, digest, roster

from .boundary import NAMESPACE, implementation_hash, real_only
from .prepare import location, read_json

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class RealLog(RunLog):
    def __init__(self, root):
        real_only(NAMESPACE)
        root = Path(root).resolve()
        if root != PROJECT_ROOT:
            raise PermissionError("Only the canonical real project/run path is permitted")
        self.path = location(root) / "study.sqlite"
        self.namespace = NAMESPACE
        self.jobs = roster()
        self.preflight = read_json(location(root) / "EXECUTION_PREFLIGHT.json")
        if (
            self.preflight["implementation_hash"] != implementation_hash()
            or self.preflight["status"] != "passed"
            or self.preflight["O_access_authorized"]
        ):
            raise PermissionError("Execution preflight invalid or code changed")
        identity = digest(
            dict(
                namespace=NAMESPACE,
                path=str(self.path),
                protocol=adopted_settings(),
                preflight=file_hash(location(root) / "EXECUTION_PREFLIGHT.json"),
            )
        )
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS meta (identity TEXT PRIMARY KEY, namespace TEXT);
                CREATE TABLE IF NOT EXISTS attempts (
                    id TEXT PRIMARY KEY, slot TEXT, job TEXT, status TEXT, identity TEXT,
                    artifact TEXT, sha TEXT, failure TEXT, retry INTEGER);
                CREATE TABLE IF NOT EXISTS aliases (slot TEXT PRIMARY KEY,job TEXT,attempt TEXT);
                CREATE TABLE IF NOT EXISTS evaluation (
                    id INTEGER PRIMARY KEY CHECK(id=1),status TEXT,frozen TEXT,predictions TEXT,seal TEXT);
            """)
            row = db.execute("SELECT identity FROM meta").fetchone()
            if row and row[0] != identity:
                raise PermissionError("Real ledger identity differs on restart")
            if not row:
                db.execute("INSERT INTO meta VALUES (?,?)", (identity, NAMESPACE))

    def reserve(self, slot, job, identity, *, retry=False):
        real_only(self.namespace)
        if implementation_hash() != self.preflight["implementation_hash"]:
            raise PermissionError("Real execution implementation changed")
        # Save full predictive identity before the reservation/fitting boundary.
        from dmvmat_ecg.phase3b.jobs import save_json

        destination = self.path.parent / "jobs" / (job + ".json")
        if destination.exists():
            if json.loads(destination.read_text()) != json.loads(json.dumps(identity)):
                raise PermissionError("Job identity file changed")
        else:
            save_json(destination, identity)
        return super().reserve(slot, job, identity, retry=retry)
