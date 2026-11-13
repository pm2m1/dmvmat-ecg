"""Read-only dependency/source identity; no device/model fitting probe."""

import importlib.metadata
import json
import os
import sys
from pathlib import Path

from .contract import digest, file_hash


def environment_receipt(root):
    root = Path(root)
    prior = json.loads((root / "results/environment.json").read_text())
    installed = {p: importlib.metadata.version(p) for p in prior["versions"]}
    if installed != prior["versions"] or sys.version.split()[0] != "3.12.14":
        raise PermissionError("Python or resolved dependency versions changed")
    uv = file_hash(root / "uv.lock")
    if uv != prior["uv_lock_sha256"]:
        raise PermissionError("Dependency lock differs")
    return dict(
        python=sys.version.split()[0],
        versions=installed,
        uv_lock_sha256=uv,
        device="cpu",
        dtype="float32",
        threads_per_fit=4,
        max_workers=2,
    )


def execution_inventory(root):
    root = Path(root).resolve()
    paths = []
    for base in ["src", "configs", "scripts"]:
        paths.extend(
            p for p in (root / base).rglob("*") if p.is_file() and p.suffix in {".py", ".json"}
        )
    paths.extend(
        root / p
        for p in [
            "uv.lock",
            "pyproject.toml",
            "results/environment.json",
            "docs/phase2/INTERNAL_TRANSFER_PILOT_PROPOSAL.md",
            "docs/phase2/PILOT_EXECUTION_CONTRACT.md",
        ]
    )
    return {str(p): file_hash(p) for p in sorted(set(paths))}


def code_identity(inventory):
    root = Path(os.path.commonpath(list(inventory)))
    return digest({str(Path(p).relative_to(root)): h for p, h in inventory.items()})
