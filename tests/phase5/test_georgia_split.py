"""Invented identities/labels; no Georgia source data access."""

import os

import numpy as np
import pytest

from dmvmat_ecg.phase5.geometry_split import allocate, episodes, validation
from dmvmat_ecg.phase5.georgia import RAW, BuildScope


def test_allocation_is_id_only_and_keeps_survey_in_pool():
    feas = [f"f{i:04}" for i in range(200)]
    remaining = [f"r{i:04}" for i in range(1801)]
    pool, test = allocate(feas, remaining)
    assert len(pool) == 1100 and len(test) == 901
    assert set(feas) <= set(pool) and not set(feas) & set(test)
    assert allocate(feas[::-1], remaining[::-1]) == (pool, test)
    assert set(pool).isdisjoint(test)


def test_known_groups_never_cross_roles():
    feas = ["f1"]
    remaining = ["f1copy", "a1", "a2", "b1", "b2", "c1"]
    groups = {
        "f1": "survey",
        "f1copy": "survey",
        "a1": "a",
        "a2": "a",
        "b1": "b",
        "b2": "b",
        "c1": "c",
    }
    p, t = allocate(feas, remaining, groups)
    assert {groups[r] for r in p}.isdisjoint({groups[r] for r in t})
    assert "f1copy" in p


def test_pool_only_strata_and_disjoint_nested_draws():
    pool = [f"invented_{i:05}" for i in range(3000)]
    y = np.array([[i % 2, (i // 2) % 2] for i in range(3000)], float)
    ep = episodes(pool)
    split = validation(pool, y, ep)
    assert len(split["A"]) == len(split["B"]) == 500
    used = {r for e in ep.values() for k in ("F", "G") for r in e[k]}
    assert not used & set(split["A"] + split["B"])
    assert not set(split["A"]) & set(split["B"])
    for d in (16101, 16102, 16103):
        for k in ("F", "G"):
            assert ep[f"{d}/B64"][k] == ep[f"{d}/B128"][k][:32]
            assert ep[f"{d}/B128"][k] == ep[f"{d}/B256"][k][:64]
    # Changing labels cannot change the training draw membership.
    assert ep == episodes(pool)
    assert validation(pool, y, ep) == split


def test_missing_pool_labels_are_separate_stratum_not_negatives():
    pool = [f"invented_{i}" for i in range(3000)]
    y = np.tile([[0.0, 1.0], [1.0, 0.0]], (1500, 1))
    y[:50] = np.nan
    split = validation(pool, y, episodes(pool))
    assert len(split["A"]) == len(split["B"]) == 500
    assert np.isnan(y[:50]).all()


def test_builder_cannot_open_nonpool_or_test_raw_files():
    scope = BuildScope()
    pool = RAW / "training/georgia/g1/INVENTED_POOL.mat"
    test = RAW / "training/georgia/g1/INVENTED_TEST.hea"
    with pytest.raises(PermissionError):
        scope.check("open", (str(pool), "rb", os.O_RDONLY))
    scope.pool_paths = {pool}
    scope.check("open", (str(pool), "rb", os.O_RDONLY))
    with pytest.raises(PermissionError):
        scope.check("open", (str(test), "rb", os.O_RDONLY))
    with pytest.raises(PermissionError):
        scope.check("open", (str(pool), "wb", os.O_WRONLY))
