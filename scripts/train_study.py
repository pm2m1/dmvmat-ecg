"""Finite development runner. Never grants test access."""

import argparse
import json
from pathlib import Path

import torch

from dmvmat_ecg.evaluation.state import Study
from dmvmat_ecg.training.engine import fit

p = argparse.ArgumentParser()
p.add_argument("--stage", choices=["smoke", "full"], required=True)
args = p.parse_args()
root = Path(__file__).resolve().parents[1]
config = json.loads((root / "configs/study.json").read_text())
torch.set_num_threads(config["execution"]["threads"])
torch.use_deterministic_algorithms(True, warn_only=True)
study = Study(root)
if config["execution"]["device"] != "cpu":
    raise RuntimeError("This finite study uses the common CPU fallback")
if study.get("phase") != "development":
    raise PermissionError("Not in the authorized development phase")
names = ["xresnet101", "flat_cnn", "multiview"] if args.stage == "smoke" else list(config["models"])
seeds = [config["smoke"]["seed"]] if args.stage == "smoke" else config["seeds"]
for name in names:
    for seed in seeds:
        previous = [
            r
            for r in study.runs()
            if r["metadata"]["stage"] == args.stage
            and r["metadata"]["model"] == name
            and r["metadata"]["seed"] == seed
        ]
        if previous:
            if previous[-1]["status"] == "successful":
                continue
            raise RuntimeError(
                "Failed/active attempt requires explicit repair review; no automatic retry"
            )
        fit(study, name, seed, "cpu", smoke=args.stage == "smoke")
print(json.dumps(dict(status="stage_complete", stage=args.stage, attempts=len(study.runs()))))
