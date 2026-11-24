"""Validation-only state authentication; never opens a test capability."""

import json

import torch

from dmvmat_ecg.phase4.safety import new_json, sha
from dmvmat_ecg.phase4c_confirm.contract import authenticate
from dmvmat_ecg.phase4c_placebo.target import full_fit

from .common import ROOT, RUN
from .sources import descriptor
from .targets import ValidationScope, filename, read, target


def replay(hospital, arm, s, d, b):
    ValidationScope(hospital).install()
    torch.set_num_threads(4)
    t = target(hospital)
    p, stored = read(t, hospital, arm, s, d, b)
    if stored["reused_historical_validation"]:
        path = ROOT / stored["historical"]["report"]
        if sha(path) != stored["historical"]["report_sha256"]:
            raise PermissionError("Historical replay binding differs")
        stored = json.loads(path.read_text())
    model, norm, _, source = descriptor(arm, s)
    model, predictions, current = full_fit(t, s, d, b, model, norm, source["sha256"])
    authenticate(stored, p, current, predictions, norm.__dict__)
    return model, norm, current


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("hospital", choices=("CPSC", "Georgia"))
    parser.add_argument("arm", choices=("plain", "placebo"))
    parser.add_argument("--source", type=int, required=True)
    parser.add_argument("--draw", type=int, default=16101)
    parser.add_argument("--budget", type=int, choices=(64, 128, 256), default=128)
    args = parser.parse_args()
    output = (
        RUN
        / args.hospital
        / "replay_checks"
        / (filename(args.arm, args.source, args.draw, args.budget) + ".json")
    )
    if output.exists():
        raise FileExistsError("Completed validation replay check preserved")
    _, _, r = replay(args.hospital, args.arm, args.source, args.draw, args.budget)
    new_json(
        output,
        dict(
            status="PASS",
            A_bitwise_match=True,
            B_bitwise_match=True,
            state_history_recipe_match=True,
            source_seed=args.source,
            draw_seed=args.draw,
            budget=args.budget,
            arm=args.arm,
            runtime_seconds=r["runtime_seconds"],
            test_accesses=0,
        ),
    )
    print("PHASE5_VALIDATION_REPLAY_PASS", args.hospital, args.arm, flush=True)
