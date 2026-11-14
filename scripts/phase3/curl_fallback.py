"""Download-only approved curl fallback, never model/data-audit operations."""

import argparse
import json
from pathlib import Path

from dmvmat_ecg.phase3.curl_fallback import run

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--probe", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parents[2] / "data/raw/cinc2021_v1_0_3"
print(json.dumps(run(root, probe=args.probe), indent=2))
