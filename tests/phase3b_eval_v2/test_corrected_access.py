"""Fictional-only corrected-access and pre-metric validity guard rehearsal."""

import numpy as np
import pytest

from dmvmat_ecg.phase3b.jobs import save_json
from dmvmat_ecg.phase3b_eval_v2.corrected_run import gate_before_metrics, reserve_corrected_access


def test_corrected_marker_preserves_V1_and_refuses_reentry(tmp_path):
    first = tmp_path / "O_OUTCOME_ACCESS_V1.json"
    save_json(first, dict(reason="fictional first decoder failure", outcome_access_count=1))
    before = first.read_bytes()
    reserve_corrected_access(tmp_path, dict(reason="fictional authorized correction", count=2))
    assert first.read_bytes() == before
    with pytest.raises(FileExistsError):
        reserve_corrected_access(tmp_path, dict(count=3))


def test_no_corrected_marker_without_prior_receipt(tmp_path):
    with pytest.raises(PermissionError):
        reserve_corrected_access(tmp_path, {})


def test_invalid_gate_never_calls_metrics():
    called = []
    y = np.array([[0, 1], [1, 0]])
    with pytest.raises(ValueError):
        gate_before_metrics(y, np.zeros_like(y, bool), lambda: called.append(True))
    assert not called
    gate_before_metrics(y, np.ones_like(y, bool), lambda: called.append(True))
    assert called == [True]
