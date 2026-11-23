"""Small invented states only; byte identity is stronger than value equality."""

import numpy as np
import pytest
import torch

from dmvmat_ecg.phase4c_retrain.comparison import compare_states
from dmvmat_ecg.preprocessing.signals import normalize, views


def test_state_comparison_distinguishes_parameter_buffer_and_signed_zero():
    original = {
        "weight": torch.tensor([0.0, 1.0]),
        "running_mean": torch.tensor([0.0]),
        "num_batches_tracked": torch.tensor(2),
    }
    new = {n: x.clone() for n, x in original.items()}
    assert compare_states(original, new, {"weight"})["full_state_bitwise_identical"]
    new["weight"][0] = -0.0
    new["num_batches_tracked"] += 1
    r = compare_states(original, new, {"weight"})
    assert r["changed_parameter_tensors"] == r["changed_buffer_tensors"] == 1
    assert not r["parameters_bitwise_identical"] and not r["buffers_bitwise_identical"]


@pytest.mark.parametrize("change", ["keys", "shape", "dtype"])
def test_state_comparison_rejects_incompatible_architecture(change):
    a = {"weight": torch.ones(3)}
    b = {"weight": torch.ones(3)}
    if change == "keys":
        b = {"other": torch.ones(3)}
    elif change == "shape":
        b["weight"] = torch.ones(4)
    else:
        b["weight"] = torch.ones(3, dtype=torch.float64)
    with pytest.raises(PermissionError):
        compare_states(a, b, {"weight"})


def test_input_views_are_byte_identical_chunked_or_full():
    x = np.random.default_rng(33).normal(size=(263, 12, 1000)).astype(np.float32)
    norm = {"mean": 0.2, "std": 1.8}
    full = views(normalize(x, norm))
    chunks = [views(normalize(x[start : start + 256], norm)) for start in range(0, len(x), 256)]
    for i in range(3):
        assert full[i].tobytes() == np.concatenate([a[i] for a in chunks]).tobytes()
