"""One synthetic prediction seal, complete roster first, lazy outcomes last."""

import io
import json

import numpy as np
import torch

from dmvmat_ecg.phase2.adaptation import state_hash

from .engine import unpack
from .jobs import atomic_new, file_hash, source_code_hash
from .models import inputs
from .protocol import canonical, digest
from .roles import Normalization


class PredictionSeal:
    def __init__(self, log, data):
        if log.namespace != data.namespace:
            raise PermissionError("Synthetic/real run namespaces cannot mix")
        self.log, self.data = log, data

    def freeze(self, predictors):
        """predictors: complete registered logical predictions, including aliases/source-only.

        Each value binds a fitted model's state and normalization to an immutable
        checkpoint. All registered fit slots must have completed or alias records.
        """
        summary = self.log.summary()
        complete = {r["slot"] for r in summary["aliases"]}
        if complete != set(self.log.jobs):
            raise PermissionError("Missing registered fitting jobs")
        expected = set(self.log.jobs)
        if set(predictors) != expected:
            raise PermissionError("Prediction roster must cover every fit/source-only exactly")
        frozen = {}
        attempts = {r["id"]: r for r in summary["attempts"]}
        completed = {r["slot"]: attempts[r["attempt"]] for r in summary["aliases"]}
        for name, item in predictors.items():
            authoritative = completed[name]
            if item["artifact"] != authoritative["artifact"] or item["sha"] != authoritative["sha"]:
                raise PermissionError("Predictor does not belong to its completed fitting job")
            if file_hash(item["artifact"]) != item["sha"]:
                raise PermissionError("Checkpoint changed before lock")
            saved = unpack(item["artifact"])
            if (
                item["state_hash"] != saved["state_hash"]
                or item["normalization"] != saved["normalization"]
                or saved["identity"]["code"] != source_code_hash()
            ):
                raise PermissionError("Checkpoint/normalization/code lineage differs")
            frozen[name] = {k: item[k] for k in ("artifact", "sha", "state_hash", "normalization")}
        payload = canonical(
            dict(predictors=frozen, code=source_code_hash(), data_identity=self.data.identity)
        ).decode()
        with self.log.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM evaluation").fetchone():
                raise PermissionError("Evaluation is single-use; lock already exists")
            db.execute("INSERT INTO evaluation VALUES (1,'frozen',?,'{}',NULL)", (payload,))
        return digest(json.loads(payload))

    def _check(self, row):
        frozen = json.loads(row["frozen"])
        if frozen["code"] != source_code_hash() or frozen["data_identity"] != self.data.identity:
            raise PermissionError("Post-lock code/data change")
        for item in frozen["predictors"].values():
            if file_hash(item["artifact"]) != item["sha"]:
                raise PermissionError("Post-lock artifact change")
        return frozen

    def predict(self, name, model):
        # Reserve each prediction access before invoking a lazy O feature provider.
        with self.log.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM evaluation").fetchone()
            if row is None or row["status"] not in {"frozen", "predicting"}:
                raise PermissionError("Predictions require finalized, unconsumed freeze")
            frozen = self._check(row)
            old = json.loads(row["predictions"])
            if name not in frozen["predictors"] or name in old:
                raise PermissionError("Unregistered or previously attempted prediction")
            item = frozen["predictors"][name]
            if state_hash(model) != item["state_hash"]:
                raise PermissionError("Model differs from frozen checkpoint")
            old[name] = dict(status="started")
            db.execute(
                "UPDATE evaluation SET status='predicting',predictions=?",
                (canonical(old).decode(),),
            )
        x = self.data._features(self.data.role("O"))
        norm = Normalization(**item["normalization"])
        model.eval()
        with torch.no_grad():
            probabilities = torch.cat(
                [
                    model(**inputs(model, norm.transform(x[i : i + 128]))).sigmoid().cpu()
                    for i in range(0, len(x), 128)
                ]
            ).numpy()
        if probabilities.shape != (len(x), 2) or not np.isfinite(probabilities).all():
            raise ValueError("Invalid predictions; preserve started receipt and stop")
        if state_hash(model) != item["state_hash"]:
            raise PermissionError("Inference changed model state")
        buffer = io.BytesIO()
        np.save(buffer, probabilities, allow_pickle=False)
        path = self.log.path.parent / "sealed_predictions" / f"{digest(name)}.npy"
        atomic_new(path, buffer.getvalue())
        with self.log.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM evaluation").fetchone()
            old = json.loads(row["predictions"])
            old[name] = dict(status="complete", path=str(path), sha=file_hash(path))
            db.execute("UPDATE evaluation SET predictions=?", (canonical(old).decode(),))

    def seal(self):
        with self.log.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM evaluation").fetchone()
            if row is None or row["status"] != "predicting":
                raise PermissionError("Complete predictions required")
            frozen = self._check(row)
            pred = json.loads(row["predictions"])
            if set(pred) != set(frozen["predictors"]) or any(
                p["status"] != "complete" or file_hash(p["path"]) != p["sha"] for p in pred.values()
            ):
                raise PermissionError("Full registered prediction set not sealed")
            seal = digest(dict(frozen=frozen, predictions=pred))
            db.execute("UPDATE evaluation SET status='sealed',seal=?", (seal,))
        return seal

    def outcomes_once(self):
        with self.log.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM evaluation").fetchone()
            if row is None or row["status"] != "sealed":
                raise PermissionError("Outcomes denied before complete seal or after first access")
            frozen = self._check(row)
            pred = json.loads(row["predictions"])
            if any(file_hash(p["path"]) != p["sha"] for p in pred.values()):
                raise PermissionError("Sealed prediction changed")
            if digest(dict(frozen=frozen, predictions=pred)) != row["seal"]:
                raise PermissionError("Prediction seal mismatch")
            db.execute("UPDATE evaluation SET status='consumed'")
        return self.data._outcomes(self.data.role("O"))
