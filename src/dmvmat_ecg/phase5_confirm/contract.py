"""Committed roster and one-way runtime capabilities for Georgia only."""

import os
import subprocess
import sys
from pathlib import Path

from dmvmat_ecg.phase4.safety import sha
from dmvmat_ecg.phase4c_confirm.scope import TOKEN
from dmvmat_ecg.phase5.common import BUDGETS, DRAWS, GEORGIA, PLACEBO, PLAIN, ROOT, SEEDS

COMMIT = "aa1bf69ffe48fe1fa0655804156d2fcded98a5a4"
PROTOCOL = ROOT / "docs/phase5/PHASE5_GEORGIA_PROTOCOL.md"
RUN = ROOT / "private/phase5_georgia_confirm_run"
DEV = ROOT / "private/phase5_replication_run"
ASSIGNMENT = ROOT / "private/phase5_georgia_membership/assignment.json"
RAW = ROOT / "data/raw/cinc2021_v1_0_3"
MAPPING = ROOT / "results/phase3a/LABEL_MAPPING.json"
AUDIT = ROOT / "results/phase3a/DATASET_AUDIT.json"
METHODS = ("a", "b", "c")


def protocol_identity():
    if (
        subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
        != "phase5-replication"
    ):
        raise PermissionError("Wrong branch")
    path = str(PROTOCOL.relative_to(ROOT))
    frozen = subprocess.check_output(["git", "show", COMMIT + ":" + path], cwd=ROOT)
    if PROTOCOL.read_bytes() != frozen or subprocess.check_output(
        ["git", "diff", COMMIT, "--", path], cwd=ROOT
    ):
        raise PermissionError("Protocol differs from committed version")
    return dict(commit=COMMIT, sha256=sha(PROTOCOL))


def roster():
    return {
        f"{m}/B{b}/s{s}/d{d}": (f"a/s{s}" if m == "a" else f"{m}/B{b}/s{s}/d{d}")
        for m in METHODS
        for b in BUDGETS
        for s in SEEDS
        for d in DRAWS
    }


class Scope:
    def __init__(self):
        self.stage = "replay"
        self.raw_paths = set()

    def check(self, event, args):
        if (
            event not in {"open", "os.listdir", "os.scandir"}
            or not args
            or not isinstance(args[0], (str, bytes, os.PathLike))
        ):
            return
        p = Path(os.fsdecode(args[0])).absolute()
        if not p.is_relative_to(ROOT):
            return
        if any("sealed" in part.lower() or "reserved" in part.lower() for part in p.parts):
            raise PermissionError("Unrelated sealed/reserved material forbidden")
        if not any(p.is_relative_to(ROOT / n) for n in ("private", "data", "results")):
            return
        if ".." in p.parts or p.resolve() != p:
            raise PermissionError("Unsafe data path")
        flags = args[2] if event == "open" and len(args) > 2 and isinstance(args[2], int) else 0
        if p.is_relative_to(RUN):
            return
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            raise PermissionError("Existing inputs/results are read-only")
        roots = (
            GEORGIA,
            DEV / "source_models",
            DEV / "Georgia/fits",
            DEV / "Georgia/predictions",
            PLAIN / "source_models",
            PLACEBO / "source_models",
        )
        exact = {
            DEV / "SOURCE_CODE_BINDING.json",
            DEV / "SOURCES_COMPLETE.json",
            DEV / "VALIDATION_RELEASE_BINDING.json",
            DEV / "Georgia/TARGET_CODE_BINDING.json",
            PLAIN / "SOURCE_CODE_BINDING.json",
            PLACEBO / "SOURCE_CODE_BINDING.json",
            ASSIGNMENT,
            MAPPING,
            AUDIT,
        }
        if p in exact or any(p.is_relative_to(r) for r in roots):
            return
        if self.stage != "replay" and p == RAW / "SHA256SUMS.txt":
            return
        if p in self.raw_paths:
            if self.stage == "predict" and (
                (p.suffix == ".hea" and TOKEN.get() == "prefix")
                or (p.suffix == ".mat" and TOKEN.get() == "signal")
            ):
                return
            if self.stage == "score" and p.suffix == ".hea" and TOKEN.get() == "annotation":
                return
        raise PermissionError("GEORGIA_OUTSIDE_STAGE_CAPABILITY")

    def install(self):
        sys.addaudithook(self.check)

    def predictions(self, count):
        if self.stage != "replay" or count != 180:
            raise PermissionError("All180 replay matches required")
        self.stage = "predict"

    def outcomes(self):
        if self.stage != "predict" or not (RUN / "OUTCOMES_CONSUMED.json").exists():
            raise PermissionError("Complete seal and single-use outcome marker required")
        self.stage = "score"
