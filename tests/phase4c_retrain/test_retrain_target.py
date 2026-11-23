"""Synthetic parity to the frozen full-FT loop and strict replay rejection."""

import copy

import numpy as np
import pytest
import torch

from dmvmat_ecg.phase2.contracts import MODULES
from dmvmat_ecg.phase3b.roles import Normalization
from dmvmat_ecg.phase4c_retrain.target import authenticate, full_fit


class Tiny(torch.nn.Module):
    input_schema = "fictional-vector"

    def __init__(self):
        super().__init__()
        for name in MODULES[:5]:
            setattr(
                self,
                name,
                torch.nn.Sequential(
                    torch.nn.Linear(4, 4),
                    torch.nn.BatchNorm1d(4),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.2),
                ),
            )
        self.fusion_1 = torch.nn.Sequential(
            torch.nn.Linear(20, 4), torch.nn.BatchNorm1d(4), torch.nn.ReLU(), torch.nn.Dropout(0.2)
        )
        self.fusion_2 = torch.nn.Sequential(
            torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4), torch.nn.ReLU(), torch.nn.Dropout(0.2)
        )
        self.prediction_head = torch.nn.Linear(4, 2)

    def selectable_modules(self):
        return {n: getattr(self, n) for n in MODULES}

    def forward(self, wave):
        z = torch.cat([getattr(self, n)(wave) for n in MODULES[:5]], 1)
        return self.prediction_head(self.fusion_2(self.fusion_1(z)))


def factory(s):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(s)
        return Tiny().eval()


class InventedTarget:
    x = np.random.default_rng(77).normal(size=(1064, 4)).astype(np.float32)

    def ids(self, role, *args):
        return {
            "F": list(range(32)),
            "G": list(range(32, 64)),
            "A": list(range(64, 564)),
            "B": list(range(564, 1064)),
        }[role]

    def index(self, ids):
        return np.asarray(ids)

    def labels(self, role, *args):
        n = len(self.ids(role))
        return np.column_stack((np.arange(n) % 2, (np.arange(n) // 2) % 2)).astype(np.float32)


def test_full_engine_bitwise_frozen_full_recipe(monkeypatch):
    import dmvmat_ecg.phase4b.target_fit as old

    norm = Normalization(0.0, 1.0, "INVENTED_SOURCE", "fresh-source-only")
    monkeypatch.setattr(
        old, "load_source", lambda t, s, a: (factory(s), norm, None, "INVENTED_CP_HASH")
    )
    t = InventedTarget()
    p, prior = old.fit(t, 15101, 16101, 64, "full")
    _, now, current = full_fit(t, 15101, 16101, 64, factory(15101), norm, "INVENTED_CP_HASH")
    for key in (
        "fit_seed",
        "state_hash",
        "source_state_hash",
        "history",
        "drift",
        "final_training_loss",
    ):
        assert prior[key] == current[key]
    for role in ("A", "B"):
        assert p[role].dtype == now[role].dtype and np.array_equal(p[role], now[role])


def references():
    p = {n: np.full((500, 2), 0.2, dtype=np.float32) for n in ("A", "B")}
    r = dict(
        source_seed=15101,
        draw_seed=16101,
        budget=128,
        fit_seed=33,
        updates=200,
        source_sha256="INVENTED_SOURCE_FILE",
        source_state_hash="INVENTED_SOURCE_STATE",
        normalization={"mean": 0.0, "std": 1.0},
        state_hash="INVENTED_FITTED_STATE",
        history=[0.5] * 200,
        drift={"view_1": 0.1},
    )
    return r, p


def test_replay_accepts_only_identical_arrays_and_bindings():
    r, p = references()
    authenticate(r, p, copy.deepcopy(r), {n: a.copy() for n, a in p.items()})


def test_replay_rejects_sub_ulp_tolerance_style_mismatch():
    r, p = references()
    q = {n: a.copy() for n, a in p.items()}
    q["A"][0, 0] = np.nextafter(q["A"][0, 0], np.float32(1.0))
    with pytest.raises(RuntimeError, match="PROBABILITY_MISMATCH:A"):
        authenticate(r, p, r, q)


def test_bitwise_gate_rejects_equal_numeric_signed_zero():
    r, p = references()
    p["B"][0, 0] = np.float32(0.0)
    q = {n: a.copy() for n, a in p.items()}
    q["B"][0, 0] = np.float32(-0.0)
    assert np.array_equal(p["B"], q["B"])
    with pytest.raises(RuntimeError, match="PROBABILITY_MISMATCH:B"):
        authenticate(r, p, r, q)


@pytest.mark.parametrize(
    "key", ["state_hash", "source_sha256", "normalization", "fit_seed", "history"]
)
def test_same_validation_predictions_do_not_waive_state_recipe_binding(key):
    r, p = references()
    q = copy.deepcopy(r)
    q[key] = "INVENTED_DIFFERENCE"
    with pytest.raises(RuntimeError, match="STATE_OR_RECIPE_MISMATCH"):
        authenticate(r, p, q, p)


def test_same_numeric_values_wrong_dtype_refused():
    r, p = references()
    with pytest.raises(RuntimeError, match="PROBABILITY_MISMATCH"):
        authenticate(r, p, r, {k: v.astype(np.float64) for k, v in p.items()})
