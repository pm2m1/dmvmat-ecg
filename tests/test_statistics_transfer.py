import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from dmvmat_ecg.evaluation.bootstrap import interval, patient_bootstrap, weighted_ranking
from dmvmat_ecg.models.networks import build
from dmvmat_ecg.transfer.interface import (
    MODULES,
    apply_mask,
    binary_geometry,
    frozen_snapshot,
    logical_policy,
    mmd,
    permutation_mmd,
)


def test_weighted_ranking_with_ties_matches_sklearn():
    y = np.array([0, 1, 0, 1, 1, 0])
    p = np.array([0.1, 0.5, 0.5, 0.8, 0.8, 0.2])
    w = np.array([2, 0, 3, 4, 1, 2])
    a, b = weighted_ranking(y, p, w)
    assert a[0] == pytest.approx(roc_auc_score(y, p, sample_weight=w))
    assert b[0] == pytest.approx(average_precision_score(y, p, sample_weight=w))


def test_patient_bootstrap_pairs_seeds_groups_and_identity():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, (30, 5))
    p = rng.uniform(size=(2, 30, 5))
    patients = np.repeat(np.arange(15), 2)
    a, receipt = patient_bootstrap(y, {"a": p, "b": p.copy()}, patients, draws=30)
    np.testing.assert_array_equal(a["a"]["auroc"], a["b"]["auroc"])
    assert interval(a["a"]["ap"] - a["b"]["ap"]) == [0, 0]
    assert receipt["patients"] == 15 and receipt["conditional"]


def test_transfer_interface_explicit_multilabel_and_invalid_geometry():
    x = np.ones((12, 4))
    y = np.zeros((12, 5))
    assert not binary_geometry(x, y, label_index=0)["valid"]
    with pytest.raises(ValueError):
        binary_geometry(x, y.argmax(1), label_index=0)
    assert logical_policy(valid=False, uncertainty=0.2, retention=0.3, p_mmd=0.01) == "SHARE"
    assert logical_policy(valid=True, uncertainty=0.7, retention=1, p_mmd=1) == "ADAPT"
    assert logical_policy(valid=True, uncertainty=0.9, retention=0.7, p_mmd=0.01) == "ADAPT"


def test_mmd_numerical_permutation_and_freezing():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(20, 4))
    assert abs(mmd(x, x)) < 1e-12
    assert mmd(x, x + 3) > 0.1
    r = permutation_mmd(x, x + 3, permutations=19)
    assert r["p"] == 0.05
    model = build("multiview")
    mask = dict.fromkeys(MODULES, "SHARE")
    apply_mask(model, mask)
    snap = frozen_snapshot(model, mask)
    assert all(
        not p.requires_grad for m in model.selectable_modules().values() for p in m.parameters()
    )
    assert all(p.requires_grad for p in model.prediction_head.parameters())
    for n, m in model.selectable_modules().items():
        assert not m.training
        for k, v in m.state_dict().items():
            torch.testing.assert_close(v, snap[n][k], atol=0, rtol=0)
