"""Offline execution after all publisher checksums pass; no training interfaces."""

from pathlib import Path

from dmvmat_ecg.phase3.prepare import prepare
from dmvmat_ecg.phase3.report import report
from dmvmat_ecg.phase3.run import analyze, freeze, predict, shifts

root = Path(__file__).resolve().parents[2]
prepare(root)
freeze(root)
predict(root)
analyze(root)
shifts(root)
report(root)
