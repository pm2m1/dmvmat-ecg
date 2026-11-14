"""Read-only historical preservation check; new aggregate receipt only."""

import json
import subprocess
from pathlib import Path

from dmvmat_ecg.phase3.download import sha
from dmvmat_ecg.phase3.prepare import write_new

root = Path(__file__).resolve().parents[2]
out = root / "results/phase3a"
start = json.loads((out / "PRESERVATION_START.json").read_text())
changed = [
    name
    for name, expected in start["tracked_hashes"].items()
    if not (root / name).is_file() or sha(root / name) != expected
]
head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip()
staged = subprocess.check_output(
    ["git", "diff", "--cached", "--name-only"], cwd=root, text=True
).splitlines()
ignore = subprocess.run(
    [
        "git",
        "check-ignore",
        "data/raw/cinc2021_v1_0_3/SHA256SUMS.txt",
        "data/manifests/phase3a_v1/example_predictions.npz",
    ],
    cwd=root,
    capture_output=True,
    text=True,
    check=False,
)
status = subprocess.check_output(["git", "status", "--short"], cwd=root, text=True)
passed = (
    not changed
    and not staged
    and head == start["HEAD"]
    and branch == start["branch"]
    and ignore.returncode == 0
)
receipt = {
    "passed": passed,
    "historical_tracked_files_checked": len(start["tracked_hashes"]),
    "changed_historical_files": changed,
    "HEAD": head,
    "branch": branch,
    "staged_files": staged,
    "data_ignored": ignore.returncode == 0,
    "git_status": status,
    "protected_namespaces_not_opened": start["protected_namespaces_not_opened"],
    "no_commits_or_publication": head == start["HEAD"],
    "scope": "Hash comparison covers historical tracked assets only; protected private Phase2 assets were neither opened nor written",
}
write_new(out / "PRESERVATION_FINAL.json", receipt)
print(json.dumps(receipt, indent=2))
raise SystemExit(0 if passed else 1)
