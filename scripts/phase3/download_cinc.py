"""Part1 only. No clinical/model operations or experimental authorization system."""

import argparse
import json
from pathlib import Path

from dmvmat_ecg.phase3.download import download, prepare, verify

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("mode", choices=["prepare", "run", "verify"])
args = parser.parse_args()
root = Path(__file__).resolve().parents[2] / "data/raw/cinc2021_v1_0_3"
if args.mode == "prepare":
    report = prepare(root)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in {"bad_copies_removed", "stray_wget_indices_removed"}
            },
            indent=2,
        )
    )
elif args.mode == "run":
    raise SystemExit(download(root))
else:
    report = verify(root)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["complete"] else 1)
