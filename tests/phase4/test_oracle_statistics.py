"""Invented-score checks for selection, joint record resampling and regression."""

from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from dmvmat_ecg.phase4.analysis import best_mask, bootstrap_ap, interpretation
from dmvmat_ecg.phase4.runtime import prepare_split, split_summary
from dmvmat_ecg.phase4.safety import RecordGuard


def test_A_only_selection_uses_lowest_index_tie_rule():
    a = np.zeros(128)
    a[[3, 126]] = 1
    assert best_mask(a) == 3
    with pytest.raises(ValueError):
        best_mask(np.zeros(127))


@pytest.mark.parametrize("tied", [False, True])
def test_weighted_bootstrap_exactly_matches_expanded_records(tied):
    y = np.array([0, 1, 0, 1, 0, 1, 1, 0])
    p = np.array([0.1, 0.7, 0.3, 0.6, 0.8, 0.9, 0.2, 0.4])
    if tied:
        p = np.round(p * 2) / 2
    w = np.random.default_rng(1).multinomial(8, np.ones(8) / 8, size=100)
    actual = bootstrap_ap(y, p, w, chunk_size=7)
    for i, weights in enumerate(w):
        ix = np.repeat(np.arange(8), weights)
        if len(np.unique(y[ix])) < 2:
            assert np.isnan(actual[i])
        else:
            assert np.isclose(actual[i], average_precision_score(y[ix], p[ix]), rtol=1e-14)


def test_absent_class_replicates_remain_unknown_not_redrawn():
    y = np.array([0, 1])
    w = np.array([[2, 0], [0, 2], [1, 1]])
    actual = bootstrap_ap(y, [0.1, 0.9], w)
    assert np.isnan(actual[:2]).all()
    assert actual[2] == 1


@pytest.mark.parametrize(
    "low,high,expected",
    [
        (0.006, 0.02, "a"),
        (-0.01, 0.02, "b"),
        (0.001, 0.004, "b"),
        (-0.03, -0.01, "c"),
        (0.002, 0.01, "UNCLASSIFIED"),
    ],
)
def test_requested_interpretation_rules_and_overlap(low, high, expected):
    result = interpretation(low, high)
    assert result["specific_requested_rule"] == expected
    if expected == "c":
        assert result["matching_requested_rules"] == ["b", "c"]


def test_split_excludes_every_budget_draw_and_is_fixed(tmp_path):
    records = [f"INVENTED_ONLY_{i:04d}" for i in range(1500)]
    y = np.tile([[0, 0], [0, 1], [1, 0], [1, 1]], (375, 1)).astype(np.float32)
    episodes = {
        f"{d}/B{b}": {"F": records[: b // 2], "G": records[256 : 256 + b // 2]}
        for d in (16101, 16102, 16103)
        for b in (64, 128, 256)
    }
    lookup = {r: i for i, r in enumerate(records)}
    guard = RecordGuard(records, [])

    def indices(ids):
        guard.check(ids)
        return np.array([lookup[r] for r in ids])

    bundle = SimpleNamespace(
        identity="INVENTED_BUNDLE",
        records=records,
        y=y,
        episodes=episodes,
        guard=guard,
        indices=indices,
    )
    split = prepare_split(bundle, tmp_path)
    used = {r for ep in episodes.values() for role in ("F", "G") for r in ep[role]}
    assert len(split["A"]) == len(split["B"]) == 500
    assert not (set(split["A"]) | set(split["B"])) & used
    assert not set(split["A"]) & set(split["B"])
    assert prepare_split(bundle, tmp_path) == split
    summary = split_summary(bundle, split)
    assert summary["excluded_all_budget_records"] == 256
    assert summary["A"]["combination_counts"] == summary["B"]["combination_counts"]


def test_factorial_module_effects_equal_OLS_with_episode_fixed_effects():
    bits = np.array([[int(b) for b in f"{m:07b}"] for m in range(128)])
    true = np.arange(1, 8) / 100
    response = bits @ true
    response = response[:, None] + np.arange(9)[None, :] / 1000
    X = np.c_[np.ones(1152), np.repeat(bits, 9, axis=0), np.tile(np.eye(9)[:, 1:], (128, 1))]
    coef = np.linalg.lstsq(X, response.ravel(), rcond=None)[0]
    assert np.allclose(coef[1:8], true)
    for j in range(7):
        diff = response[bits[:, j] == 1].mean() - response[bits[:, j] == 0].mean()
        assert np.isclose(diff, coef[1 + j])
