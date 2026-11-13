"""Amended split gates. O identities and outcomes are separate sealed vaults."""

import copy
import json
from pathlib import Path

from dmvmat_ecg.phase2b.access import RoleAccess
from dmvmat_ecg.phase2b.access import fit_preprocessing as fit_preprocessing
from dmvmat_ecg.phase2b.access import fresh_source as fresh_source
from dmvmat_ecg.phase2b.access import fresh_target as fresh_target
from dmvmat_ecg.phase2b.access import validate_source_preprocessing as validate_source_preprocessing
from dmvmat_ecg.phase2b.access import verify_new_source as verify_new_source
from dmvmat_ecg.phase2c.auditor import immutable_json
from dmvmat_ecg.phase2c.splits import validate

from .contract import CANDIDATE_HASH, contract, digest, file_hash
from .runtime import verify_execution_identity


def seal_candidate(auditor, folder, *, synthetic=False):
    validate(auditor)
    if auditor.get("labels") != ["CD", "STTC"]:
        raise PermissionError("Ordered CD/STTC endpoint cannot change")
    if not synthetic and digest(auditor) != CANDIDATE_HASH:
        raise PermissionError("Only exact seed44301 candidate may enter real execution")
    folder = Path(folder).resolve()
    worker = copy.deepcopy(auditor)
    worker["schema"] = "phase2d-worker-v1"
    worker["audit_manifest_hash"] = digest(auditor)
    worker["execution_contract_hash"] = digest(contract())
    worker["synthetic_only"] = synthetic
    worker["sealed_role_labels"] = True
    worker["role_vaults"] = {}
    for name, rows in auditor["roles"].items():
        path = folder / (name.replace(":", "_") + ".labels.json")
        immutable_json(path, rows)
        metadata = [{k: v for k, v in r.items() if k not in {"y", "observed"}} for r in rows]
        worker["role_vaults"][name] = dict(
            path=str(path), sha256=file_hash(path), full_role_hash=digest(rows)
        )
        worker["role_hashes"][name] = digest(metadata)
        if name == "O":
            meta_path = folder / "O.identities.json"
            immutable_json(meta_path, metadata)
            worker["evaluation_membership_vault"] = dict(
                path=str(meta_path), sha256=file_hash(meta_path)
            )
            del worker["roles"]["O"]
        else:
            worker["roles"][name] = metadata
    return worker


def validate_worker(worker):
    if (
        worker.get("schema") != "phase2d-worker-v1"
        or not worker.get("sealed_role_labels")
        or worker.get("execution_contract_hash") != digest(contract())
        or worker.get("labels") != ["CD", "STTC"]
        or "O" in worker.get("roles", {})
    ):
        raise PermissionError("Old/mixed/unsealed worker contract")
    if set(worker["role_hashes"]) != set(worker["role_vaults"]) or set(worker["roles"]) != set(
        worker["role_hashes"]
    ) - {"O"}:
        raise PermissionError("Role set changed")
    for role, rows in worker["roles"].items():
        if (
            any("y" in r or "observed" in r for r in rows)
            or digest(rows) != worker["role_hashes"][role]
        ):
            raise PermissionError("Worker outcomes or membership changed")
    return True


class AmendedAccess(RoleAccess):
    def __init__(self, ledger, manifest, *, integrity_files=None):
        validate_worker(manifest)
        if (
            digest(manifest) != ledger.identity["manifest_hash"]
            or ledger.config["execution_version"] != "phase2d-amended-v1"
        ):
            raise PermissionError("Amended ledger/manifest mismatch")
        if ledger.identity["namespace"].startswith("real:") and (
            manifest["synthetic_only"] or manifest["audit_manifest_hash"] != CANDIDATE_HASH
        ):
            raise PermissionError("Real data must use the exact audited candidate")
        self.ledger = ledger
        self._manifest = copy.deepcopy(manifest)
        self._hash = digest(manifest)
        self.integrity_files = None if integrity_files is None else dict(integrity_files)
        if ledger.identity["namespace"].startswith("real:"):
            self._verify()
        elif self.integrity_files is None:
            self.integrity_files = {}

    def _verify(self):
        if not self.ledger.identity["namespace"].startswith("real:"):
            return super()._verify()
        if digest(self._manifest) != self._hash:
            raise PermissionError("Manifest content changed")
        self.ledger._verify_contract()
        verify_execution_identity(self.ledger.identity, self.integrity_files)

    def rows(self, role, operation, *, attempt=None, job=None, token=None):
        if role != "O":
            return super().rows(role, operation, attempt=attempt, job=job, token=token)
        self._verify()
        if operation not in {"evaluate_predict", "evaluate_labels"}:
            raise PermissionError("O is not fitting/stopping/evidence/calibration data")
        self.ledger.require_evaluation(token)
        if operation == "evaluate_labels":
            self.ledger.require_metric_access(token)
        descriptor = self._manifest["evaluation_membership_vault"]
        if file_hash(descriptor["path"]) != descriptor["sha256"]:
            raise PermissionError("O membership vault changed")
        rows = json.loads(Path(descriptor["path"]).read_text())
        if digest(rows) != self._manifest["role_hashes"]["O"]:
            raise PermissionError("O membership hash changed")
        if operation == "evaluate_labels":
            vault = self._manifest["role_vaults"]["O"]
            if file_hash(vault["path"]) != vault["sha256"]:
                raise PermissionError("O labels changed")
            full = json.loads(Path(vault["path"]).read_text())
            if (
                digest(full) != vault["full_role_hash"]
                or [{k: v for k, v in r.items() if k not in {"y", "observed"}} for r in full]
                != rows
            ):
                raise PermissionError("O labels/record identities mismatch")
            rows = full
        self.ledger.event(
            "data_access",
            dict(
                role=role,
                operation=operation,
                attempt=attempt,
                job=job,
                token=token,
                membership_hash=self._manifest["role_hashes"]["O"],
            ),
        )
        return rows
