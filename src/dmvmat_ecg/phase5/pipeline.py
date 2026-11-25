"""Finish validation from completed sources; separate processes keep guards scoped."""

import json
import subprocess
import sys

from .common import ROOT, RUN


def main():
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    if branch != "phase5-replication":
        raise PermissionError("Phase5 branch required")
    if json.loads((RUN / "SOURCES_COMPLETE.json").read_text())["status"] != "COMPLETE":
        raise PermissionError("All paired sources must finish before this controller")
    for hospital in ("CPSC", "Georgia"):
        subprocess.run(
            [sys.executable, "-u", "-m", "dmvmat_ecg.phase5.targets", hospital],
            cwd=ROOT,
            check=True,
        )
        if not (RUN / hospital / "VALIDATION_ANALYSIS.json").exists():
            subprocess.run(
                [sys.executable, "-u", "-m", "dmvmat_ecg.phase5.statistics", hospital],
                cwd=ROOT,
                check=True,
            )
    subprocess.run([sys.executable, "-u", "-m", "dmvmat_ecg.phase5.report"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
