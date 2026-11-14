"""Fictional diagnostic codes, WFDB headers, memberships and labels only."""

import numpy as np
import pytest
import wfdb

from dmvmat_ecg.phase3b_eval_v2.decoder import decode_labels, validity_gate
from dmvmat_ecg.phase3b_eval_v2.validation import permitted_r, reference_coverage

MAPPING = {"entries": {"1": {"label": "CD"}, "2": {"label": "STTC"}, "3": {"label": None}}}


@pytest.mark.parametrize(
    "text", ["#Dx: 1,2", "# Dx: 1,2", "#  dx :1 , 2", "# dX : 1, 2", "#DX:\t1,2"]
)
def test_whitespace_case_and_commas(text):
    assert decode_labels((text + "\n").encode(), MAPPING) == ([1, 1], [True, True])


@pytest.mark.parametrize(
    "text",
    ["# Age: 42", "# Dx:", "# Dx: a,b", "# Dx: 1,", "# Dx: 1,,2", "# Dx: -1", "# Dx"],
)
def test_missing_or_malformed_unknown(text):
    y, obs = decode_labels(text.encode(), MAPPING)
    assert np.isnan(y).all() and obs == [False, False]


def test_exact_cinc_comment_layout_against_wfdb(tmp_path):
    leads = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
    content = "fiction 12 500 5000\n"
    content += "".join(f"fiction.mat 16 1000/mV 16 0 0 0 0 {lead}\n" for lead in leads)
    content += "# Age: 42\n# Sex: Male\n# Dx: 1,2\n# Rx: Unknown\n# Hx: Unknown\n# Sx: Unknown\n"
    path = tmp_path / "fiction.hea"
    path.write_text(content)
    comments = wfdb.rdheader(str(path.with_suffix(""))).comments
    assert "Dx: 1,2" in comments
    assert decode_labels(content.encode(), MAPPING) == ([1, 1], [True, True])


def test_unlisted_annotation_is_not_clinical_negative_and_new_code_fails():
    assert decode_labels(b"# Dx: 3\n", MAPPING) == ([0, 0], [True, True])
    with pytest.raises(ValueError):
        decode_labels(b"# Dx: 99999\n", MAPPING)


def test_duplicate_conflicting_comments_unknown():
    y, obs = decode_labels(b"# Dx: 1\n# Dx: 2\n", MAPPING)
    assert np.isnan(y).all() and obs == [False, False]
    assert decode_labels(b"# Dx: 1,2\n# DX: 2,1\n", MAPPING) == ([1, 1], [True, True])


def test_non_ascii_malformed_unknown():
    y, obs = decode_labels(b"# Dx: \xff\n", MAPPING)
    assert np.isnan(y).all() and obs == [False, False]


def test_gate_fraction_and_both_classes():
    y = np.tile([[0, 1], [1, 0]], (10, 1))
    observed = np.ones_like(y, bool)
    assert validity_gate(y, observed)["passed"]
    observed[0, 0] = False
    assert validity_gate(y, observed)["passed"]  # exactly 95%
    observed[1, 0] = False
    assert not validity_gate(y, observed)["passed"]
    assert not validity_gate(np.ones((20, 2)), np.ones((20, 2), bool))["passed"]
    assert not validity_gate(y, np.zeros_like(y, bool))["passed"]


def test_no_retroactive_frozen_reference():
    c = reference_coverage(["a", "b", "c"], {"a": {}})
    assert not c["full_coverage"] and c["missing_frozen_references"] == 2
    assert reference_coverage(["a"], {"a": {}})["full_coverage"]
    with pytest.raises(PermissionError):
        reference_coverage(["a"], {"O": {}})


def test_validation_refuses_O_and_FEAS():
    membership = dict(R=["r"], O=["o"], FEAS=["f"])
    permitted_r("r", membership)
    for name in ["o", "f", "unknown"]:
        with pytest.raises(PermissionError):
            permitted_r(name, membership)
