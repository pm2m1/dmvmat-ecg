"""Invented paired seed/record effects; no clinical data or source models."""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from dmvmat_ecg.phase3b_eval.statistics import bootstrap_rankings
from dmvmat_ecg.phase5.statistics import combine, points, weights


def test_seed_then_record_counts_reproducible_and_group_paired():
    groups = np.repeat(np.arange(12), 2)
    s, r = weights(groups, nseeds=10, draws=200)
    ss, rr = weights(groups, nseeds=10, draws=200)
    assert np.array_equal(s, ss) and np.array_equal(r, rr)
    assert s.shape == (200, 10) and r.shape == (200, 24)
    assert np.all(s.sum(1) == 10) and np.all(r[:, ::2] == r[:, 1::2])
    assert np.unique(s, axis=0).shape[0] > 1


def test_source_randomness_remains_when_record_effect_is_constant():
    s, _ = weights(np.arange(12), nseeds=3, draws=100)
    fixed = np.tile([-0.1, 0.03, 0.2], (100, 1))
    b = combine(fixed, s)
    assert np.std(b) > 0
    assert np.allclose(b, np.sum(fixed * s, axis=1) / 3)


def test_zero_weight_seed_does_not_poison_replica_but_selected_unknown_does():
    a = np.array([[np.nan, 0.2], [np.nan, 0.2]])
    s = np.array([[0, 2], [1, 1]])
    b = combine(a, s)
    assert b[0] == 0.2 and np.isnan(b[1])


def test_metric_arithmetic_and_exact_weighted_bootstrap():
    y = np.tile([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]], (3, 1))
    p = np.tile([[0.1, 0.8], [0.8, 0.2], [0.8, 0.6], [0.2, 0.2]], (3, 1))
    result = points(y, p)
    assert result.shape == (2, 2)
    assert result[0, 0] == average_precision_score(y[:, 0], p[:, 0])
    assert result[1, 1] == roc_auc_score(y[:, 1], p[:, 1])
    _, w = weights(np.arange(len(y)), nseeds=3, draws=20)
    b = bootstrap_rankings(y, p, np.ones_like(y, bool), w)
    for i in range(20):
        for j in range(2):
            if (w[i] @ y[:, j]) and (w[i] @ (1 - y[:, j])):
                assert b["AP"][i, j] == pytest.approx(
                    average_precision_score(y[:, j], p[:, j], sample_weight=w[i])
                )


def test_pairwise_equal_arms_have_zero_hierarchical_difference():
    s, _ = weights(np.arange(20), nseeds=10, draws=100)
    paired = np.random.default_rng(2).random((100, 10))
    assert np.array_equal(combine(paired - paired, s), np.zeros(100))
