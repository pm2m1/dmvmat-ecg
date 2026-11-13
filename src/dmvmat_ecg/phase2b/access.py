"""Fail-closed membership-bound data access and fresh-source lineage.

No access is granted by a role string alone. All roles are resolved from the
hashed manifest and bound to an active reserved job or separate evaluation lease.
Python guards are not an OS sandbox against arbitrary filesystem reads.
"""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .contract import LABELS, digest, file_hash
from .preflight import row_hash, validate_manifest


class RoleAccess:
    def __init__(self, ledger, manifest, *, integrity_files=None):
        validate_manifest(manifest)
        if not manifest.get("sealed_role_labels"):
            raise PermissionError("Execution requires a label-free worker manifest and role vaults")
        if digest(manifest) != ledger.identity["manifest_hash"]:
            raise PermissionError("Manifest is not bound to this ledger")
        self.ledger = ledger
        self._manifest = copy.deepcopy(manifest)
        self._hash = digest(manifest)
        self.integrity_files = dict(integrity_files or {})
        if ledger.identity["namespace"].startswith("real:") and not self.integrity_files:
            raise PermissionError("Real execution requires pinned file-integrity inventory")

    def _verify(self):
        if digest(self._manifest) != self._hash:
            raise PermissionError("Manifest content changed")
        if any(file_hash(Path(p)) != h for p, h in self.integrity_files.items()):
            raise PermissionError("Prediction-affecting code/configuration/dependencies changed")
        if self.ledger.identity["namespace"].startswith("real:"):
            from .runtime import environment_receipt

            lock_paths = [Path(p) for p in self.integrity_files if Path(p).name == "uv.lock"]
            if (
                len(lock_paths) != 1
                or digest(environment_receipt(lock_paths[0].parent))
                != self.ledger.identity["dependency_hash"]
            ):
                raise PermissionError("Live dependency identity cannot be verified")

    def rows(self, role, operation, *, attempt=None, job=None, token=None):
        self._verify()
        if operation in {"evaluate_predict", "evaluate_labels"}:
            if role != "O":
                raise PermissionError("Evaluation is restricted to O")
            self.ledger.require_evaluation(token)
            if operation == "evaluate_labels":
                self.ledger.require_metric_access(token)
        elif operation == "calibrate":
            if (
                self.ledger.get("phase") != "development"
                or role != "C"
                or job not in self.ledger.successful()
            ):
                raise PermissionError("Calibration requires completed job and C only")
        else:
            _, spec = self.ledger.active(attempt)
            permitted = {
                "fit": set(spec["fit_roles"]),
                "preprocess_fit": set(spec["fit_roles"])
                if spec["kind"] == "source" or spec.get("method") == "target_only"
                else set(),
                "stop": {spec["stop_role"]},
                "evidence": {"source_train", spec.get("evidence_role")}
                if spec.get("evidence_role")
                else set(),
            }
            if role not in permitted.get(operation, set()) or role is None:
                raise PermissionError("Reserved job is not authorized for this role/operation")
        if role not in self._manifest["roles"]:
            raise PermissionError("Unknown manifest role")
        rows = self._manifest["roles"][role]
        if row_hash(rows) != self._manifest["role_hashes"][role]:
            raise PermissionError("Patient/group hash mismatch")
        self.ledger.event(
            "data_access",
            dict(
                role=role,
                operation=operation,
                attempt=attempt,
                job=job,
                token=token,
                membership_hash=row_hash(rows),
            ),
        )
        returned = copy.deepcopy(rows)
        if operation != "evaluate_predict":
            vault = self._manifest["role_vaults"][role]
            path = Path(vault["path"])
            if file_hash(path) != vault["sha256"]:
                raise PermissionError("Role-label vault changed")
            full = json.loads(path.read_text())
            if (
                row_hash(full) != vault["full_role_hash"]
                or [{k: v for k, v in r.items() if k not in {"y", "observed"}} for r in full]
                != rows
            ):
                raise PermissionError("Role labels do not match bound patient descriptors")
            returned = full
        return returned

    def load(
        self,
        role,
        operation,
        *,
        attempt=None,
        job=None,
        token=None,
        raw_root=None,
        synthetic_reader=None,
    ):
        rows = self.rows(role, operation, attempt=attempt, job=job, token=token)
        scope = self.ledger.identity["namespace"]
        if scope.startswith("real:"):
            if synthetic_reader is not None or raw_root is None:
                raise PermissionError("Real/synthetic data providers cannot be mixed")
            root = Path(raw_root).resolve()
            if root.name != "ptbxl_v1_0_3":
                raise PermissionError("Unrecognized real release root")
            import wfdb

            from dmvmat_ecg.preprocessing.signals import canonical_leads

            def reader(row):
                base = (root / row["filename"]).resolve()
                if not base.is_relative_to(root):
                    raise PermissionError("Waveform path escape")
                if file_hash(base.with_suffix(".dat")) != row["raw_hash"]:
                    raise PermissionError("Raw waveform changed")
                if file_hash(base.with_suffix(".hea")) != row["header_hash"]:
                    raise PermissionError("Header changed")
                wave, info = wfdb.rdsamp(str(base))
                if info["fs"] != 100 or any(str(u).lower() != "mv" for u in info["units"]):
                    raise ValueError("Unexpected sample rate/units")
                return wave[:, canonical_leads(info["sig_name"])].T
        else:
            if raw_root is not None or synthetic_reader is None:
                raise PermissionError("Synthetic execution needs its explicit fictional provider")
            reader = synthetic_reader
        x = np.stack([reader(r) for r in rows]).astype(np.float32)
        if x.shape != (len(rows), 12, 1000) or not np.isfinite(x).all():
            raise ValueError("Invalid signal shape/numerics")
        content = hashlib.sha256(x.tobytes()).hexdigest()
        self.ledger.event(
            "signal_materialization",
            dict(
                namespace=scope,
                role=role,
                attempt=attempt,
                membership_hash=self._manifest["role_hashes"][role],
                content_hash=content,
            ),
        )
        return dict(
            namespace=scope,
            role=role,
            membership_hash=self._manifest["role_hashes"][role],
            rows=rows,
            content_hash=content,
            x=x,
            y=np.array([r["y"] for r in rows], np.float32) if "y" in rows[0] else None,
            observed=np.array([r["observed"] for r in rows], bool)
            if "observed" in rows[0]
            else None,
        )

    def label_cost(self, attempt=None):
        job_name = attempt
        if attempt in self.ledger.config["planned_jobs"]:
            attempt = self.ledger.successful().get(attempt, attempt)
        elif attempt is not None:
            row = self.ledger.db.execute(
                "SELECT job FROM attempts WHERE id=?", (attempt,)
            ).fetchone()
            job_name = row[0] if row else None
        used = {}
        for (event,) in self.ledger.db.execute(
            "SELECT detail FROM events WHERE kind='data_access'"
        ):
            event = json.loads(event)
            if event["operation"] == "evaluate_predict":
                continue
            if attempt is None or event["attempt"] == attempt or event["job"] == job_name:
                vault = self._manifest["role_vaults"][event["role"]]
                if file_hash(vault["path"]) != vault["sha256"]:
                    raise PermissionError("Label accounting vault changed")
                used[event["role"]] = json.loads(Path(vault["path"]).read_text())
        target = [
            r
            for role, rows in used.items()
            if not role.startswith("source_") and role != "O"
            for r in rows
        ]
        records = {r["ecg_id"] for r in target}
        cells = {(r["ecg_id"], j) for r in target for j, flag in enumerate(r["observed"]) if flag}
        return dict(
            target_records=len(records),
            target_observed_labels=len(cells),
            by_role={
                role: dict(records=len(rows), observed_labels=sum(sum(r["observed"]) for r in rows))
                for role, rows in used.items()
            },
            evaluation_records=len(used.get("O", [])),
        )


def fit_preprocessing(batch, access, attempt):
    batches = batch if isinstance(batch, list) else [batch]
    _, spec = access.ledger.active(attempt)
    if {b["role"] for b in batches} != set(spec["fit_roles"]):
        raise PermissionError("Preprocessing must use exactly the declared scratch fitting pool")
    values = []
    for b in batches:
        access.rows(b["role"], "preprocess_fit", attempt=attempt)
        if (
            b["namespace"] != access.ledger.identity["namespace"]
            or b["membership_hash"] != access._manifest["role_hashes"][b["role"]]
        ):
            raise PermissionError("Preprocessing batch has wrong membership/namespace")
        content = hashlib.sha256(b["x"].tobytes()).hexdigest()
        receipts = [
            json.loads(v)
            for (v,) in access.ledger.db.execute(
                "SELECT detail FROM events WHERE kind='signal_materialization'"
            )
        ]
        if content != b["content_hash"] or not any(
            r["attempt"] == attempt
            and r["role"] == b["role"]
            and r["content_hash"] == content
            and r["membership_hash"] == b["membership_hash"]
            for r in receipts
        ):
            raise PermissionError("Preprocessing signal content has no authorized materialization")
        values.append(b["x"].astype(np.float64))
    n = sum(x.size for x in values)
    mean = sum(float(x.sum()) for x in values) / n
    variance = sum(float(np.square(x - mean).sum()) for x in values) / n
    std = float(np.sqrt(variance))
    if not np.isfinite([mean, std]).all() or std <= 0:
        raise ValueError("Degenerate source-only statistics")
    return dict(
        namespace=batches[0]["namespace"],
        mean=mean,
        std=std,
        labels=list(LABELS),
        fit_role=batches[0]["role"] if len(batches) == 1 else "fitting_union",
        fit_roles=[b["role"] for b in batches],
        fit_membership_hash=batches[0]["membership_hash"]
        if len(batches) == 1
        else digest([b["membership_hash"] for b in batches]),
        fresh=True,
        fit_attempt=attempt,
    )


def validate_source_preprocessing(stats, access):
    if (
        stats.get("namespace") != access.ledger.identity["namespace"]
        or stats.get("fit_role") != "source_train"
        or not stats.get("fresh")
        or stats.get("fit_membership_hash") != access._manifest["role_hashes"]["source_train"]
    ):
        raise PermissionError("Historical/target-derived preprocessing is prohibited")


def fresh_source(access, attempt, *, synthetic_factory=None):
    access.rows("source_train", "fit", attempt=attempt)
    job, spec = access.ledger.active(attempt)
    if job != "source" or spec["kind"] != "source":
        raise PermissionError("Fresh source needs its reserved source job")
    prior = [
        json.loads(v)
        for (v,) in access.ledger.db.execute("SELECT detail FROM events WHERE kind='fresh_source'")
    ]
    if any(v["attempt"] == attempt for v in prior):
        raise PermissionError("Attempt already initialized; restart needs a new counted attempt")
    import torch
    from torch import nn

    from dmvmat_ecg.models.networks import FiveViewECG
    from dmvmat_ecg.phase2.adaptation import state_hash

    if synthetic_factory is not None and not access.ledger.identity["namespace"].startswith(
        "synthetic:"
    ):
        raise PermissionError("Synthetic source factory forbidden in real namespace")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(spec["seed"])
        if synthetic_factory is not None:
            model = synthetic_factory()
        else:
            model = FiveViewECG()
            model.prediction_head = nn.Linear(64, 2)
    access.ledger.event(
        "fresh_source",
        dict(
            attempt=attempt,
            origin="random-scratch",
            initial_state_hash=state_hash(model),
            seed=spec["seed"],
        ),
    )
    return model


def fresh_target(access, attempt, *, synthetic_factory=None):
    _, spec = access.ledger.active(attempt)
    if spec.get("method") != "target_only":
        raise PermissionError("Only target-only control may create a new scratch target model")
    for role in spec["fit_roles"]:
        access.rows(role, "fit", attempt=attempt)
    import torch
    from torch import nn

    from dmvmat_ecg.models.networks import FiveViewECG
    from dmvmat_ecg.phase2.adaptation import state_hash

    if synthetic_factory is not None and not access.ledger.identity["namespace"].startswith(
        "synthetic:"
    ):
        raise PermissionError("Synthetic factory in real namespace")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(spec["seed"])
        model = synthetic_factory() if synthetic_factory else FiveViewECG()
        if synthetic_factory is None:
            model.prediction_head = nn.Linear(64, 2)
    access.ledger.event(
        "fresh_target",
        dict(
            attempt=attempt,
            initial_state_hash=state_hash(model),
            origin="random-scratch",
            seed=spec["seed"],
        ),
    )
    return model


def verify_new_source(access, attempt):
    ledger = access.ledger
    if ledger.successful().get("source") != attempt:
        raise PermissionError("Only newly completed source job is a source checkpoint")
    lineage = json.loads(ledger.artifact(attempt, "lineage.json").read_text())
    expected = {
        "namespace": ledger.identity["namespace"],
        "origin": "random-scratch",
        "source_training_hash": access._manifest["role_hashes"]["source_train"],
        "source_selection_hash": access._manifest["role_hashes"]["source_stop"],
        "labels": list(LABELS),
    }
    if any(lineage.get(k) != v for k, v in expected.items()):
        raise PermissionError("Historical/overlapping source checkpoint lineage rejected")
    validate_source_preprocessing(lineage["preprocessing"], access)
    if lineage["preprocessing"]["fit_attempt"] != attempt:
        raise PermissionError("Preprocessing is not from this newly trained source attempt")
    initialized = [
        json.loads(v)
        for (v,) in ledger.db.execute("SELECT detail FROM events WHERE kind='fresh_source'")
    ]
    if not any(
        v["attempt"] == attempt
        and v["origin"] == "random-scratch"
        and v["initial_state_hash"] == lineage.get("initial_state_hash")
        for v in initialized
    ):
        raise PermissionError("No fresh-initialization source attestation")
    return ledger.artifact(attempt, "checkpoint.pt"), lineage
