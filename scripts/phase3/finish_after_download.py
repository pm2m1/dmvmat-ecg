"""Observe existing Part1 job, verify, and run offline Part2 only if complete.

No downloader is launched here. This keeps download network operations separate
from the strictly local inference process. A missing checksum blocks Part2.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

from dmvmat_ecg.phase3.download import sha, verify
from dmvmat_ecg.phase3.prepare import write_new

root = Path(__file__).resolve().parents[2]
raw = root / "data/raw/cinc2021_v1_0_3"
out = root / "results/phase3a"
status_path = raw / "DOWNLOAD_STATUS.json"
while True:
    try:
        status = json.loads(status_path.read_text())
    except json.JSONDecodeError:  # tiny status-write interval, no data/fit operation
        time.sleep(1)
        continue
    if status["status"] != "running":
        break
    print(json.dumps({"stage": "Part1 download still running", "time": time.time()}), flush=True)
    time.sleep(40)
verification = verify(raw)
public_verification = {k: v for k, v in verification.items() if k != "failures"}
public_verification["failed_file_count"] = len(verification["failures"])
write_new(out / "DOWNLOAD_VERIFICATION_SUMMARY.json", public_verification)
if not verification["complete"]:
    report = root / "docs/phase3/CROSS_HOSPITAL_FEASIBILITY.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Cross-hospital feasibility — blocked by incomplete download",
        "",
        "Part2 was NOT executed. No real target audit, split generation, inference, MMD or feasibility metrics were performed. Synthetic test fixtures are not cardiovascular evidence.",
        "",
        "Official release1.0.3 publisher checksum index SHA256: `"
        + sha(raw / "SHA256SUMS.txt")
        + "`.",
        "",
        "| Source | Expected files | SHA-verified files | Verified bytes | Missing/mismatched |",
        "|---|---:|---:|---:|---:|",
    ]
    for source, value in verification["sources"].items():
        lines.append(
            f"| {source} | {value['expected_files']} | {value['downloaded_files']} | {value['bytes']} | {value['mismatches']} |"
        )
    lines += [
        "",
        "Failed/missing files are listed locally in git-ignored `data/raw/cinc2021_v1_0_3/DOWNLOAD_VERIFICATION.json`; full independent checksum log: `SHASUM_VERIFICATION.log`. First50:",
        "",
        *["- `" + name + "`" for name in verification["failures"][:50]],
        "",
        "## Authorized method refinements prepared, unexecuted",
        "",
        "The new code names exact restricted SNOMED/SCP annotation unions, treats absent codes as not annotated rather than clinically negative, computes a matching restricted source fold9 reference and descriptive per-code AUROC, and uses proxy-matched DROP for the pre-stated criterion. Full source superclasses are reported separately only if execution becomes possible. Ambiguous164909002 is excluded. Clinical endpoint equivalence remains unverified.",
        "",
        "The two explicit deviations are preserved: full10s CNN pipeline instead of2.5s windows, and no target-vs-PTBXL waveform duplicate audit. Short records are excluded by headers, longer records first10s, no padding. Native target duplicate checks remain planned but unexecuted. External clinical validation is out of scope. No Phase2 artifacts, O, source fold10 or RESERVED inference was accessed. No training/commit/push/merge/PR.",
        "",
        "New code/scripts/tests are confined to Phase3 paths. Completed verification and preservation records, when available, are in results/phase3a. No scientific feasibility conclusion can be drawn from an incomplete download.",
    ]
    with report.open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    print("BLOCKED — DOWNLOAD INCOMPLETE", flush=True)
    raise SystemExit(1)
print("All publisher hashes passed; beginning offline Part2", flush=True)
raise SystemExit(
    subprocess.run(
        [sys.executable, "-B", "scripts/phase3/offline_feasibility.py"], cwd=root, check=False
    ).returncode
)
