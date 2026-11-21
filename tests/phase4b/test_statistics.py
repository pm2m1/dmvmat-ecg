"""Invented weighted samples and selection candidates; no real input files."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from dmvmat_ecg.phase4b.aligned_targets import choose_lambda
from dmvmat_ecg.phase4b.statistics import bootstrap_auc, intervals
from dmvmat_ecg.phase4b.train_alignment import TargetStream


@pytest.mark.parametrize("ties", [False, True])
def test_bootstrap_AUROC_matches_duplicated_records(ties):
    y = np.array([0, 1, 0, 1, 1, 0])
    p = np.array([0.2, 0.3, 0.8, 0.7, 0.6, 0.9])
    if ties:
        p = np.round(p * 2) / 2
    weights = np.random.default_rng(4).multinomial(6, np.ones(6) / 6, size=100)
    got = bootstrap_auc(y, p, weights, chunk_size=13)
    for i, w in enumerate(weights):
        ix = np.repeat(np.arange(6), w)
        if len(set(y[ix])) < 2:
            assert np.isnan(got[i])
        else:
            assert np.isclose(got[i], roc_auc_score(y[ix], p[ix]), atol=1e-14)


def test_target_stream_is_deterministic_and_does_not_drop_cycle_tail():
    import torch

    a = TargetStream(37, torch.Generator().manual_seed(7))
    b = TargetStream(37, torch.Generator().manual_seed(7))
    first = a.take(32)
    second = a.take(32)
    assert len(set(np.r_[first, second[:5]])) == 37
    assert np.array_equal(first, b.take(32))
    assert np.array_equal(second, b.take(32))


def test_lambda_selection_uses_A_and_source_not_B():
    rows = [
        dict(
            lambda_value=0.1,
            fold9_macro_AUROC=0.9,
            per_seed_AUROC=[[0.9, 0.9]] * 3,
            A_macro_AP=0.7,
            B_macro_AP=0.1,
        ),
        dict(
            lambda_value=1.0,
            fold9_macro_AUROC=0.9,
            per_seed_AUROC=[[0.9, 0.9]] * 3,
            A_macro_AP=0.6,
            B_macro_AP=0.99,
        ),
    ]
    assert choose_lambda(rows, 0.91) == 0.1
    rows[0]["fold9_macro_AUROC"] = 0.8
    assert choose_lambda(rows, 0.91) == 1.0
    rows[1]["per_seed_AUROC"][0][0] = 0.5
    assert choose_lambda(rows, 0.91) is None


def test_lambda_tie_uses_source_AUROC_then_smaller_lambda():
    rows = [
        dict(lambda_value=a, fold9_macro_AUROC=0.9, per_seed_AUROC=[[0.9, 0.9]] * 3, A_macro_AP=0.7)
        for a in [0.1, 1.0]
    ]
    assert choose_lambda(rows, 0.9) == 0.1
    rows[1]["fold9_macro_AUROC"] = 0.91
    assert choose_lambda(rows, 0.9) == 1.0


def test_familywise_interval_is_no_narrower_and_unknown_replicates_not_zeroed():
    values = np.random.default_rng(2).normal(size=10000)
    ordinary = intervals(values)
    adjusted = intervals(values, 7)
    assert adjusted["low"] <= ordinary["low"] and adjusted["high"] >= ordinary["high"]
    values[:501] = np.nan
    with pytest.raises(ValueError):
        intervals(values)
