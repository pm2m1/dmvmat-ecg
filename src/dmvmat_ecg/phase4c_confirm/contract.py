"""Fixed protocol identities, roster, exact authentication and ordered gates."""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dmvmat_ecg.phase4.safety import digest, new_json, sha

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "private/phase4c_confirm_run"
PROTOCOL = ROOT / "docs/phase4/PHASE4C_CONFIRMATORY_PROTOCOL.md"
COMMIT = "2e3d0c740bbb38aa071d5dfd92daac7f6885e562"
SOURCES = (15101, 15102, 15103)
DRAWS = (16101, 16102, 16103)
BUDGETS = (64, 128, 256)
METHODS = ("M1", "M2", "M3", "M4", "M5")


def utc():
    return datetime.now(timezone.utc).isoformat()


def protocol_identity():
    if (
        subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
        != "phase4c-confirm"
    ):
        raise PermissionError("Wrong branch")
    frozen = subprocess.check_output(
        ["git", "show", COMMIT + ":docs/phase4/PHASE4C_CONFIRMATORY_PROTOCOL.md"], cwd=ROOT
    )
    if PROTOCOL.read_bytes() != frozen:
        raise PermissionError("Protocol differs from the committed version")
    if subprocess.check_output(
        ["git", "diff", COMMIT, "--", str(PROTOCOL.relative_to(ROOT))], cwd=ROOT
    ):
        raise PermissionError("Nonempty protocol diff")
    return dict(commit=COMMIT, sha256=hashlib.sha256(frozen).hexdigest())


def roster():
    return {
        f"{m}/B{b}/s{s}/d{d}": (f"M1/s{s}" if m == "M1" else f"{m}/B{b}/s{s}/d{d}")
        for m in METHODS
        for b in BUDGETS
        for s in SOURCES
        for d in DRAWS
    }


def authenticate(old, arrays, now, probabilities, normalization):
    for role in ("A", "B"):
        a, b = arrays[role], probabilities[role]
        if a.shape != b.shape or a.dtype != b.dtype or a.tobytes(order="C") != b.tobytes(order="C"):
            raise RuntimeError("REPLAY_BITWISE_MISMATCH:" + role)
    for field in (
        "source_seed",
        "draw_seed",
        "budget",
        "updates",
        "mask_index",
        "fit_seed",
        "source_sha256",
        "source_state_hash",
        "state_hash",
        "history",
        "drift",
        "final_training_loss",
    ):
        if old[field] != now[field]:
            raise RuntimeError("REPLAY_RECIPE_OR_STATE_MISMATCH:" + field)
    # Early Phase4B records bind normalization through the source checkpoint,
    # rather than a duplicated field in the fit JSON. Never invent that field.
    if now["normalization"] != normalization or (
        "normalization" in old and old["normalization"] != normalization
    ):
        raise RuntimeError("REPLAY_NORMALIZATION_MISMATCH")


def gate_sequence(contrasts):
    gate = True
    rows = []
    for name in ("primary", "S1", "S2"):
        c = contrasts[name]
        ci = c["CI95"]
        valid = c["estimate"] is not None and ci["interval"] is not None and ci["valid"] >= 9500
        succeeds = valid and ci["interval"][0] > 0
        status = (
            ("success" if succeeds else "not successful" if valid else "invalid")
            if gate
            else "descriptive only"
        )
        rows.append(
            dict(
                name=name,
                eligible=gate,
                status=status,
                estimate=c["estimate"],
                CI95=ci,
                gate_reason="all predecessors succeeded"
                if gate
                else "an earlier confirmatory hypothesis did not succeed",
            )
        )
        gate = gate and succeeds
    return rows


def save_array(path, value):
    import os

    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        np.save(f, value, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())


def verify_manifest(folder, expected_roster=None, expected_records=3860):
    folder = Path(folder)
    m = json.loads((folder / "PREDICTION_MANIFEST.json").read_text())
    if m["roster"] != (roster() if expected_roster is None else expected_roster):
        raise PermissionError("Incomplete or changed logical prediction roster")
    if set(m["predictions"]) != set(m["roster"].values()):
        raise PermissionError("Incomplete physical predictors")
    for item in m["predictions"].values():
        p = folder / item["file"]
        if not p.resolve().is_relative_to(folder.resolve()) or sha(p) != item["sha256"]:
            raise PermissionError("Prediction hash/path differs")
        a = np.load(p, mmap_mode="r", allow_pickle=False)
        if (
            a.shape != (expected_records, 2)
            or a.dtype != np.float32
            or not np.isfinite(a).all()
            or ((a < 0) | (a > 1)).any()
        ):
            raise PermissionError("Invalid frozen prediction array")
    seal = json.loads((folder / "PREDICTION_SEAL.json").read_text())
    if seal["manifest_sha256"] != sha(folder / "PREDICTION_MANIFEST.json") or seal[
        "manifest_digest"
    ] != digest(m):
        raise PermissionError("Prediction seal differs")
    return m


def consume_labels(folder, loader, expected_roster=None, expected_records=3860):
    folder = Path(folder)
    m = verify_manifest(folder, expected_roster, expected_records)
    # Exclusive sticky marker, before even the first annotation byte is read.
    new_json(
        folder / "OUTCOMES_CONSUMED.json",
        dict(timestamp=utc(), manifest_sha256=sha(folder / "PREDICTION_MANIFEST.json"), count=1),
    )
    return loader(m)
