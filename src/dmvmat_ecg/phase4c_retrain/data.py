"""Reuse frozen PTB-only features; isolated new outputs, no test provider."""

import json
import os
import sys
from pathlib import Path

from dmvmat_ecg.phase4.safety import new_json, sha
from dmvmat_ecg.phase4c_placebo.data import SourceInputs

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "private/phase4c_retrain_run"
SOURCE = ROOT / "private/phase4b_source_bundle"
BUNDLE = ROOT / "private/phase4_oracle_bundle"
OLD = ROOT / "private/phase4b_run"
PLACEBO = ROOT / "private/phase4c_placebo_run"
SPLIT = ROOT / "private/phase4_oracle_run/split.json"


def access_check(mode, event, args, checkpoints=()):
    if event not in {"open", "os.listdir", "os.scandir"} or not args:
        return
    if not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    p = Path(os.fsdecode(args[0])).absolute()
    if not p.is_relative_to(ROOT):
        return
    if any("sealed" in x.lower() or "reserved" in x.lower() for x in p.parts):
        raise PermissionError("Reserved/sealed access forbidden")
    if not any(p.is_relative_to(ROOT / n) for n in ("private", "data", "results")):
        return
    if ".." in p.parts or p.resolve() != p:
        raise PermissionError("Data traversal/symlink forbidden")
    roots = [RUN, SOURCE, PLACEBO / "source_inputs"]
    exact = {PLACEBO / "SOURCE_INPUTS.json"}
    if mode == "target":
        roots += [BUNDLE, OLD, PLACEBO]
        exact.add(SPLIT)
    elif mode == "compare":
        exact |= {BUNDLE / "sources.json", *checkpoints}
    elif mode != "source":
        raise ValueError("Explicit allowed mode required")
    if p not in exact and not any(p.is_relative_to(r) for r in roots):
        raise PermissionError("RETRAIN_INPUT_OUTSIDE_ALLOWED_SCOPE")
    if event == "open" and not p.is_relative_to(RUN):
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            raise PermissionError("Existing inputs/artifacts are read-only")


def setup(mode):
    if mode not in {"source", "target", "compare"}:
        raise ValueError("Explicit allowed mode required")
    RUN.mkdir(mode=0o700, exist_ok=True)
    if RUN.is_symlink():
        raise PermissionError("Unsafe new run root")
    RUN.chmod(0o700)
    checkpoints = set()
    sys.addaudithook(lambda e, a: access_check(mode, e, a, checkpoints))
    if mode == "compare":
        entries = json.loads((BUNDLE / "sources.json").read_text())
        for entry in entries.values():
            p = BUNDLE / entry["checkpoint"]
            if not p.resolve().is_relative_to(BUNDLE) or p.resolve() != p.absolute():
                raise PermissionError("Unsafe original source path")
            checkpoints.add(p)
    return checkpoints


def prepare():
    setup("source")
    SourceInputs()
    path = RUN / "SOURCE_INPUTS.json"
    binding = dict(
        schema="retrain-reuses-ptb-only-features-v1",
        input_manifest_sha256=sha(PLACEBO / "SOURCE_INPUTS.json"),
        lambda_value=0.0,
        CPSC_inputs_read=False,
    )
    if path.exists():
        if json.loads(path.read_text()) != binding:
            raise PermissionError("Source input binding differs")
    else:
        new_json(path, binding)
    print("RETRAIN_SOURCE_ONLY_INPUTS_VERIFIED", flush=True)


if __name__ == "__main__":
    prepare()
