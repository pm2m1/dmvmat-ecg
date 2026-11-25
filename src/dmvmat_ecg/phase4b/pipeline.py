"""Resume verified Phase 4B stages, without touching prior study operations."""

import argparse
import json
from pathlib import Path

from dmvmat_ecg.phase4.safety import sha

from .common import ROOT, RUN, setup
from .stage_lock import stage_lock


def verify_code():
    manifest = json.loads((RUN / "NUMERICAL_CODE_IDENTITY.json").read_text())
    for name, expected in manifest["files"].items():
        if sha(Path(__file__).with_name(name)) != expected:
            raise PermissionError("Frozen Phase4B numerical code differs:" + name)
    if sha(ROOT / "uv.lock") != manifest["original_source_lock_sha256"]:
        raise PermissionError("Original environment lock changed")
    extra = RUN / "LATER_STAGE_CODE_IDENTITY_V3.json"
    names = [
        "baselines.py",
        "aligned_targets.py",
        "statistics.py",
        "report.py",
        "pipeline.py",
        "stage_lock.py",
    ]
    identities = {name: sha(Path(__file__).with_name(name)) for name in names}
    if extra.exists():
        if json.loads(extra.read_text()) != identities:
            raise PermissionError("Later-stage implementation changed on resumption")
    else:
        raise PermissionError("Explicit reviewed V3 orchestration binding required")


def run(stage):
    setup()
    verify_code()
    if stage == "baselines":
        from .baselines import run as execute
    elif stage == "source_grid":
        from .alignment_jobs import grid as execute
    elif stage == "aligned_targets":
        from .aligned_targets import run as execute
    elif stage == "analysis":
        from .report import analyze as execute
    else:
        raise ValueError("Unknown explicit stage")
    with stage_lock(RUN, stage):
        execute()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["baselines", "source_grid", "aligned_targets", "analysis"]
    )
    run(parser.parse_args().stage)
