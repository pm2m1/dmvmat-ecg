"""Auditor-only preparation of role-specific label vaults and label-free manifests."""

import copy
import json
from pathlib import Path

from .contract import digest, file_hash
from .ledger import atomic_bytes
from .preflight import row_hash, validate_manifest


def seal_role_labels(audit_manifest, folder):
    validate_manifest(audit_manifest)
    if audit_manifest.get("sealed_role_labels"):
        raise ValueError("Auditor must supply the original full manifest")
    folder = Path(folder).resolve()
    worker = copy.deepcopy(audit_manifest)
    worker["sealed_role_labels"] = True
    worker["audit_manifest_hash"] = digest(audit_manifest)
    worker["role_vaults"] = {}
    for role, rows in audit_manifest["roles"].items():
        path = folder / (role.replace(":", "_") + ".json")
        payload = json.dumps(rows, sort_keys=True).encode()
        if path.exists():
            if path.read_bytes() != payload:
                raise PermissionError("Existing role-label vault differs")
        else:
            atomic_bytes(path, payload)
        worker["role_vaults"][role] = dict(
            path=str(path), sha256=file_hash(path), full_role_hash=row_hash(rows)
        )
        worker["roles"][role] = [
            {k: v for k, v in r.items() if k not in {"y", "observed"}} for r in rows
        ]
        worker["role_hashes"][role] = row_hash(worker["roles"][role])
    validate_manifest(worker)
    return worker
