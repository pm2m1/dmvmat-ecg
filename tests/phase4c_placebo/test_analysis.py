"""Invented stored references: enforce family, episode and prediction identity."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from dmvmat_ecg.phase4.safety import sha
from dmvmat_ecg.phase4b.statistics import measures
from dmvmat_ecg.phase4c_placebo.analysis import read_fit


@pytest.fixture
def reference(tmp_path):
    (tmp_path / "fits").mkdir()
    (tmp_path / "predictions").mkdir()
    y = np.column_stack((np.arange(500) % 2, (np.arange(500) // 2) % 2))
    p = np.random.default_rng(5).uniform(size=(500, 2)).astype(np.float32)
    path = tmp_path / "predictions/fiction.npz"
    np.savez_compressed(path, A=p, B=p)
    report = dict(
        status="COMPLETE",
        source_seed=15101,
        draw_seed=16101,
        budget=128,
        updates=200,
        mask_index=127,
        method="full",
        aligned=None,
        source_sha256="INVENTED_SOURCE",
        predictions_sha256=sha(path),
        A_AP=measures(y, p)[0].tolist(),
    )
    target = SimpleNamespace(
        sources={"15101": {"sha256": "INVENTED_SOURCE"}}, labels=lambda role: y
    )
    return tmp_path, target, report


def write_report(root, report):
    (root / "fits/fiction.json").write_text(json.dumps(report))


def test_authentic_invented_reference_accepted(reference):
    root, target, report = reference
    write_report(root, report)
    p, r = read_fit(root, "fiction", target, 15101, 16101, 128, "full")
    assert r == report and p["A"].shape == p["B"].shape == (500, 2)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_seed", 15102),
        ("draw_seed", 16102),
        ("budget", 64),
        ("updates", 199),
        ("mask_index", 0),
        ("method", "placebo_full"),
        ("aligned", "lambda0.1"),
        ("source_sha256", "OTHER_INVENTED_SOURCE"),
        ("predictions_sha256", "OTHER_INVENTED_PREDICTIONS"),
        ("A_AP", [0.0, 0.0]),
    ],
)
def test_changed_reference_rejected(reference, field, value):
    root, target, report = reference
    report[field] = value
    write_report(root, report)
    with pytest.raises(PermissionError):
        read_fit(root, "fiction", target, 15101, 16101, 128, "full")


@pytest.mark.parametrize("kind", ["extra_role", "dtype", "shape", "nonfinite", "range"])
def test_invalid_prediction_payload_rejected_even_with_updated_file_hash(reference, kind):
    root, target, report = reference
    p = np.full((500, 2), 0.2, dtype=np.float32)
    arrays = dict(A=p.copy(), B=p.copy())
    if kind == "extra_role":
        arrays["unregistered"] = p
    elif kind == "dtype":
        arrays["B"] = p.astype(np.float64)
    elif kind == "shape":
        arrays["B"] = p[:499]
    elif kind == "nonfinite":
        arrays["B"][0, 0] = np.nan
    else:
        arrays["B"][0, 0] = 1.1
    path = root / "predictions/fiction.npz"
    np.savez_compressed(path, **arrays)
    report["predictions_sha256"] = sha(path)
    write_report(root, report)
    with pytest.raises(PermissionError):
        read_fit(root, "fiction", target, 15101, 16101, 128, "full")
