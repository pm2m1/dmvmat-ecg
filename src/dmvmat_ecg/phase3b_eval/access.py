"""Separate single-use evaluation gate; the development provider stays unchanged."""

import hashlib
import json
import sqlite3
import subprocess
from pathlib import Path

from dmvmat_ecg.phase3b.jobs import file_hash, save_json, source_code_hash
from dmvmat_ecg.phase3b.protocol import digest, roster
from dmvmat_ecg.phase3b_real.boundary import BASE_HASH, MAPPING_HASH, implementation_hash

LOCK = "30a0c75ab9a159773bc87e3941ddfdee91c1860482d08e20a7b87ae601d25073"
MEMBERSHIPS = "ec2db8d89cca158320e5842550be87914860f82ddd00f598c095a41452df6c5e"
REAL_CODE = "92654c006f94859b896b0c977c8e3b32c9a6d7490f342eb05ccc07e0eba9943c"
COMMIT = "7e93ed3a7a00b603d34d190843bcfbfc1753db1e"
VERSION = "phase3c-eval-v1"
ROOT = Path(__file__).resolve().parents[3]
AUTH_STATEMENT = "I authorize ONE single-use locked evaluation of the frozen Phase 3B study on CPSC role O (3,860 records), exactly as adopted, bound to:"


def validate_authorization(value):
    expected = dict(
        lock_hash=LOCK,
        memberships_hash=MEMBERSHIPS,
        frozen_base_code_identity=BASE_HASH,
        real_implementation_identity=REAL_CODE,
        authorized_commit=COMMIT,
        label_mapping_sha256=MAPPING_HASH,
        authorization=AUTH_STATEMENT,
    )
    if set(value) != set(expected) | {"protocol_owner", "date"}:
        raise PermissionError("Authorization has missing or invented fields")
    if any(value[k] != v for k, v in expected.items()) or not value["protocol_owner"]:
        raise PermissionError("Human authorization identity differs")
    from datetime import date

    date.fromisoformat(value["date"])


def read_json(path):
    return json.loads(Path(path).read_text())


def evaluator_hash():
    folder = Path(__file__).resolve().parent
    files = {p.name: file_hash(p) for p in sorted(folder.glob("*.py"))}
    files["weighted_ranking"] = file_hash(folder.parents[0] / "evaluation/bootstrap.py")
    return digest(files)


def verify(root):
    root = Path(root).resolve()
    if root != ROOT:
        raise PermissionError("Only the canonical project may open the real evaluation")
    folder = root / "private/phase3b_cpsc_v1"
    receipt = read_json(root / "results/phase3b/FINAL_VERIFICATION_RECEIPT.json")
    lock = read_json(folder / "DEVELOPMENT_LOCK.json")
    if receipt.get("lock_hash") != LOCK or lock["lock_hash"] != LOCK:
        raise PermissionError("This must be the Phase3B receipt, not another study")
    if digest(lock["bundle"]) != LOCK:
        raise PermissionError("Development lock changed")
    for name, expected in lock["bundle"]["artifacts"].items():
        if file_hash(name) != expected:
            raise PermissionError("Frozen artifact changed: " + name)
    if (
        file_hash(folder / "MEMBERSHIPS.json") != MEMBERSHIPS
        or source_code_hash() != BASE_HASH
        or implementation_hash() != REAL_CODE
        or file_hash(root / "results/phase3a/LABEL_MAPPING.json") != MAPPING_HASH
    ):
        raise PermissionError("Membership/code/mapping identity differs")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() != COMMIT:
        raise PermissionError("Authorized commit differs")
    if set(lock["bundle"]["predictors"]) != set(roster()):
        raise PermissionError("Registered prediction roster is incomplete")
    with sqlite3.connect("file:" + str(folder / "study.sqlite") + "?mode=ro", uri=True) as db:
        if db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] != 121:
            raise PermissionError("Predictor-fit count changed")
        row = db.execute("SELECT status,frozen,predictions,seal FROM evaluation").fetchone()
        if row is None or row[0] != "frozen" or json.loads(row[2]) != {} or row[3] is not None:
            raise PermissionError("Prior outer-prediction/outcome access or lock mismatch")
        if digest(json.loads(row[1])) != lock["bundle"]["prediction_framework_freeze_hash"]:
            raise PermissionError("Stored development freeze differs")
    return lock


class Session:
    """Exclusive sticky start/consume files. No retry/re-entry, even after a crash."""

    def __init__(self, folder, identity, *, anchor=None):
        self.folder = Path(folder)
        self.identity = dict(identity)
        self.anchor = Path(anchor) if anchor is not None else self.folder

    def begin(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        self.anchor.mkdir(parents=True, exist_ok=True)
        save_json(self.anchor / "O_EVAL_BEGIN_V1.json", self.identity)
        save_json(self.folder / "BEGIN.json", self.identity)

    def check(self):
        if not (self.folder / "BEGIN.json").exists():
            raise PermissionError("Evaluation has not begun")
        if (
            read_json(self.folder / "BEGIN.json") != self.identity
            or read_json(self.anchor / "O_EVAL_BEGIN_V1.json") != self.identity
        ):
            raise PermissionError("Evaluation identity changed")
        if (
            self.identity.get("evaluator_hash")
            and self.identity["evaluator_hash"] != evaluator_hash()
        ):
            raise PermissionError("Evaluation code changed after begin")
        if (self.anchor / "O_OUTCOME_ACCESS_V1.json").exists():
            raise PermissionError("O outcome access has already been consumed")

    def seal(self, roster_names, predictions, lineage, membership_hash):
        self.check()
        if set(predictions) != set(roster_names):
            raise PermissionError("The complete registered prediction set is required")
        for item in predictions.values():
            if file_hash(item["path"]) != item["sha"]:
                raise PermissionError("Prediction artifact changed")
        payload = dict(
            identity=self.identity,
            roster=list(roster_names),
            predictions=predictions,
            lineage=lineage,
            membership_hash=membership_hash,
        )
        result = dict(seal_hash=digest(payload), payload=payload)
        save_json(self.folder / "PREDICTION_SEAL.json", result)
        return result

    def consume(self, loader):
        self.check()
        sealed = read_json(self.folder / "PREDICTION_SEAL.json")
        if digest(sealed["payload"]) != sealed["seal_hash"]:
            raise PermissionError("Prediction seal differs")
        for item in sealed["payload"]["predictions"].values():
            if file_hash(item["path"]) != item["sha"]:
                raise PermissionError("Sealed prediction changed before outcome access")
        receipt = dict(
            identity=self.identity, seal_hash=sealed["seal_hash"], outcome_access_count=1
        )
        save_json(self.anchor / "O_OUTCOME_ACCESS_V1.json", receipt)
        save_json(self.folder / "OUTCOME_ACCESS.json", receipt)
        return loader()


def bytes_hash(value):
    return hashlib.sha256(value).hexdigest()
