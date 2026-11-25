"""Stage-specific read capabilities; old test operations remain inaccessible."""

import contextlib
import contextvars
import os
import sys
from pathlib import Path

from .contract import ROOT, RUN

BUNDLE = ROOT / "private/phase4_oracle_bundle"
OLD = ROOT / "private/phase4b_run"
PLAIN = ROOT / "private/phase4c_retrain_run"
PLACEBO = ROOT / "private/phase4c_placebo_run"
SPLIT = ROOT / "private/phase4_oracle_run/split.json"
MAPPING = ROOT / "results/phase3a/LABEL_MAPPING.json"
MEMBERSHIPS = ROOT / "private/phase3b_cpsc_v1/MEMBERSHIPS.json"
RAW = ROOT / "data/raw/cinc2021_v1_0_3"
TOKEN = contextvars.ContextVar("phase4c_raw_read", default="")


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
        if any("sealed" in x.lower() or "reserved" in x.lower() for x in p.parts):
            raise PermissionError("Unrelated sealed/reserved material forbidden")
        restricted = any(p.is_relative_to(ROOT / n) for n in ("private", "data", "results"))
        if not restricted:
            return
        if ".." in p.parts or p.resolve() != p:
            raise PermissionError("Data path traversal/symlink forbidden")
        writing = (
            event == "open"
            and len(args) > 2
            and isinstance(args[2], int)
            and args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        )
        if p.is_relative_to(RUN):
            return
        if writing:
            raise PermissionError("Existing data/artifacts are read-only")
        if (
            any(p.is_relative_to(r) for r in (BUNDLE, OLD, PLAIN, PLACEBO))
            or p == SPLIT
            or p == MAPPING
        ):
            return
        if self.stage != "replay" and p in {MEMBERSHIPS, RAW / "SHA256SUMS.txt"}:
            return
        if p in self.raw_paths:
            if p.suffix == ".mat" and self.stage == "predict" and TOKEN.get() == "signal":
                return
            if p.suffix == ".hea" and self.stage == "predict" and TOKEN.get() == "prefix":
                return
            if p.suffix == ".hea" and self.stage == "score" and TOKEN.get() == "annotation":
                return
        raise PermissionError("PHASE4C_OUTSIDE_STAGE_CAPABILITY")

    def install(self):
        sys.addaudithook(self.check)

    def predictions(self, count):
        if self.stage != "replay" or count != 108:
            raise PermissionError("All108 replay matches required before test signals")
        self.stage = "predict"

    def outcomes(self):
        if self.stage != "predict" or not (RUN / "OUTCOMES_CONSUMED.json").exists():
            raise PermissionError("Verified seal and consumed outcome marker required")
        self.stage = "score"


@contextlib.contextmanager
def capability(name):
    t = TOKEN.set(name)
    try:
        yield
    finally:
        TOKEN.reset(t)
