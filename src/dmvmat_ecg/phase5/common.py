"""Fixed identities and stage-specific filesystem capabilities for replication."""

import json
import os
import sys
from pathlib import Path

from dmvmat_ecg.phase4.safety import new_json, sha

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "private/phase5_replication_run"
SOURCE = ROOT / "private/phase4b_source_bundle"
PLACEBO = ROOT / "private/phase4c_placebo_run"
PLAIN = ROOT / "private/phase4c_retrain_run"
BUNDLE = ROOT / "private/phase4_oracle_bundle"
SPLIT = ROOT / "private/phase4_oracle_run/split.json"
GEORGIA = ROOT / "private/phase5_georgia_bundle"
SEEDS = tuple(range(15101, 15111))
DRAWS = (16101, 16102, 16103)
BUDGETS = (64, 128, 256)
ARMS = ("plain", "placebo")


def bind(path, expected):
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise PermissionError("Frozen replication binding differs")
    else:
        new_json(path, expected)


class Scope:
    def __init__(self, mode):
        if mode not in {"source", "CPSC", "Georgia"}:
            raise ValueError("No test capability is implemented")
        self.mode = mode

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
            raise PermissionError("Sealed/reserved access denied")
        if not any(p.is_relative_to(ROOT / n) for n in ("private", "data", "results")):
            return
        if ".." in p.parts or p.resolve() != p:
            raise PermissionError("Data traversal/symlink denied")
        flags = args[2] if event == "open" and len(args) > 2 and isinstance(args[2], int) else 0
        if p.is_relative_to(RUN):
            return
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            raise PermissionError("Earlier artifacts are read-only")
        # Explicit source features and checkpoint/fit metadata only, no old test run.
        roots = [
            SOURCE,
            PLACEBO / "source_inputs",
            PLACEBO / "source_models",
            PLAIN / "source_models",
        ]
        exact = {
            PLACEBO / "SOURCE_INPUTS.json",
            PLACEBO / "SOURCE_CODE_BINDING.json",
            PLAIN / "SOURCE_CODE_BINDING.json",
        }
        if self.mode == "CPSC":
            roots += [
                BUNDLE,
                PLACEBO / "fits",
                PLACEBO / "predictions",
                PLAIN / "fits",
                PLAIN / "predictions",
            ]
            exact.add(SPLIT)
        if self.mode == "Georgia":
            roots += [GEORGIA]
        if p in exact or any(p.is_relative_to(r) for r in roots):
            return
        raise PermissionError("PHASE5_INPUT_OUTSIDE_ALLOWED_STAGE")

    def install(self):
        os.umask(0o077)
        RUN.mkdir(mode=0o700, exist_ok=True)
        if RUN.is_symlink():
            raise PermissionError("Unsafe replication destination")
        sys.addaudithook(self.check)


def old_code_check():
    for folder in (PLAIN, PLACEBO):
        for rel, expected in json.loads((folder / "SOURCE_CODE_BINDING.json").read_text()).items():
            if sha(ROOT / rel) != expected:
                raise PermissionError("Historical source numerical code changed")
