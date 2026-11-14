from dataclasses import replace

import numpy as np
import pytest
import torch

from dmvmat_ecg.phase2.adaptation import SelectiveModel, masked_bce, state_hash, validate_owners
from dmvmat_ecg.phase2.contracts import MODULES
from dmvmat_ecg.phase3b.models import fresh_model, inputs, random_count_cost
from dmvmat_ecg.phase3b.protocol import (
    BUDGETS,
    CEILING,
    DRAW_SEEDS,
    SOURCE_SEEDS,
    adopted_settings,
    digest,
    fictional_episode,
    roster,
)
from dmvmat_ecg.phase3b.roles import Normalization, RoleData, array_hash

from .fixtures import factory, fixture_data, source_fixture


def test_adopted_matrix():
    assert BUDGETS == (64, 128, 256)
    assert SOURCE_SEEDS == (15101, 15102, 15103)
    assert DRAW_SEEDS == (16101, 16102, 16103)
    jobs = roster()
    assert sum(j["kind"] == "source" for j in jobs.values()) == 3
    assert len(jobs) == 138 and CEILING == 144
    assert all("B32/" not in j and "xresnet" not in j for j in jobs)
    settings = adopted_settings()
    assert settings["gradient_roles"] == ["F", "G"]
    assert not settings["real_training_authorized"]


def test_configuration_and_support_table_match_code():
    import json
    from pathlib import Path

    from scipy.stats import binom

    root = Path(__file__).resolve().parents[2]
    config = root / "configs/phase3b"
    assert json.loads((config / "adopted_v1.json").read_text()) == json.loads(
        json.dumps(adopted_settings())
    )
    assert json.loads((config / "fit_matrix_v1.json").read_text()) == roster()
    rows = json.loads((config / "support_probability_v1.json").read_text())["rows"]
    assert [r["G"] for r in rows] == [32, 64, 128]
    for r in rows:
        for label, p in (("CD", 506 / 1930), ("STTC", 494 / 1930)):
            assert r[label] == pytest.approx(
                binom.cdf(r["G"] - 10, r["G"], p) - binom.cdf(9, r["G"], p)
            )


def test_nested_label_blind_draws():
    ids = [f"fictional:{i}" for i in range(600)]
    a = fictional_episode(ids, 16101)
    assert a == fictional_episode(list(reversed(ids)), 16101)
    assert digest(a) == digest(fictional_episode(ids, 16101))
    assert a != fictional_episode(ids, 16102)
    for b, (f, g) in a.items():
        assert len(f) == len(g) == b // 2 and not set(f).intersection(g)
        assert set(f).issubset(a[256][0]) and set(g).issubset(a[256][1])


def test_backbone_count_shapes_gradients_initialization():
    torch.set_num_threads(4)
    a, b = fresh_model(15101), fresh_model(15101)
    assert state_hash(a) == state_hash(b)
    assert sum(p.numel() for p in a.parameters()) == 212066
    assert len(validate_owners(a)) == 7
    x = np.random.default_rng(1).normal(size=(2, 12, 1000)).astype("float32")
    xx = inputs(a, x)
    assert xx["rhythm"].shape == (2, 3, 1000)
    assert xx["spectrum"].shape == (2, 12, 396)
    reps = a.representations(**xx)
    assert [v.shape[1] for v in reps.values()] == [64] * 5 + [128, 64]
    logits = a(**xx)
    assert logits.shape == (2, 2)
    logits.square().mean().backward()
    assert all(
        any(p.grad is not None for p in m.parameters()) for m in a.selectable_modules().values()
    )


@pytest.mark.parametrize(
    "mask",
    [
        dict.fromkeys(MODULES, "SHARE"),
        dict.fromkeys(MODULES, "ADAPT"),
        *[{n: "ADAPT" if n == one else "SHARE" for n in MODULES} for one in MODULES],
        {n: "ADAPT" if i % 2 else "SHARE" for i, n in enumerate(MODULES)},
    ],
)
def test_freezing_owners_modes_buffers(mask):
    torch.set_num_threads(4)
    source = factory(1).eval()
    original = state_hash(source)
    wrapped = SelectiveModel(source, mask)
    x = torch.randn(12, 4)
    assert torch.equal(source(wave=x), wrapped(wave=x))
    wrapped.train()
    for n, module in wrapped.model.selectable_modules().items():
        assert module.training == (mask[n] == "ADAPT")
        for child in module.modules():
            if isinstance(child, (torch.nn.Dropout, torch.nn.BatchNorm1d)):
                assert child.training == (mask[n] == "ADAPT")
        assert all(p.requires_grad == (mask[n] == "ADAPT") for p in module.parameters())
    opt = wrapped.optimizer()
    assert {id(p) for g in opt.param_groups for p in g["params"]} == {
        id(p) for p in wrapped.parameters() if p.requires_grad
    }
    initial_head = wrapped.model.prediction_head.weight.detach().clone()
    for _ in range(3):
        wrapped.train()
        opt.zero_grad(set_to_none=True)
        loss = wrapped(wave=x).square().mean()
        loss.backward()
        opt.step()
        wrapped.verify_frozen()
    assert not torch.equal(initial_head, wrapped.model.prediction_head.weight)
    assert state_hash(source) == original


@pytest.mark.parametrize("count", range(8))
def test_random_matches_both_count_and_parameter_cost(count):
    model = factory(1)
    mask = {n: "ADAPT" if i < count else "SHARE" for i, n in enumerate(MODULES)}
    random, info = random_count_cost(model, mask, 43)
    assert random == random_count_cost(model, mask, 43)[0]
    assert sum(v == "ADAPT" for v in random.values()) == count
    assert info["parameter_error"] == 0
    assert info["equality_probability"] == 1 / info["candidate_count"]
    if count in {0, 7}:
        assert random == mask and info["candidate_count"] == 1


def test_unknown_labels_not_negative():
    data, _, _ = fixture_data(unknown=True)
    _, y, obs = data.read("G", "evidence")
    assert np.isnan(y[:3, 1]).all() and not obs[:3, 1].any()
    z = torch.zeros(y.shape, requires_grad=True)
    loss = masked_bce(z, torch.tensor(y), torch.tensor(obs))
    loss.backward()
    assert torch.equal(z.grad[:3, 1], torch.zeros(3))
    cost = data.label_cost()
    assert cost["records"] == 64 and cost["observed_entries"] == 125
    assert cost["distinct_patients"] is None


@pytest.mark.parametrize(
    "role,operation",
    [("O", "evidence"), ("O", "fit_target"), ("G", "stop_source"), ("F", "normalize_source")],
)
def test_denied_access_is_lazy(role, operation):
    data, counters, _ = fixture_data()
    with pytest.raises(PermissionError):
        data.read(role, operation)
    assert counters[role] == dict(features=0, outcomes=0)


def test_false_patient_and_cross_role_claims():
    data, _, _ = fixture_data()
    with pytest.raises(ValueError, match="verified patients"):
        replace(data.role("F"), group_kind="patient", patient_linkage="known")
    with pytest.raises(PermissionError):
        RoleData(
            data.namespace, [data.role("F"), replace(data.role("G"), groups=data.role("F").groups)]
        )
    with pytest.raises(PermissionError):
        RoleData("real:phase3b", [])
    with pytest.raises(ValueError):
        replace(data.role("F"), name="E")


def test_canonical_record_keeps_own_annotation():
    data, _, _ = fixture_data()
    r = data.role("source_train")
    groups = (r.groups[1], r.groups[1], *r.groups[2:])
    # Lowest numeric ID is independent of diagnosis, and 2 sorts before 10.
    records = ("10", "2", *r.records[2:])
    changed = replace(r, records=records, groups=groups)
    data = RoleData(
        data.namespace,
        [changed, data.role("source_stop"), data.role("F"), data.role("G"), data.role("O")],
    )
    ix = data.canonical_source_indices()
    assert 1 in ix and 0 not in ix
    _, y, _ = data.read("source_train", "evidence")
    assert y[ix][0].tolist() == y[1].tolist()


def test_train_only_normalization_and_lineage():
    data, counters, arrays = fixture_data()
    source = source_fixture(data)
    assert source.normalization.mean == pytest.approx(arrays["source_train"][0].mean(), abs=1e-7)
    assert counters["G"]["features"] == counters["O"]["features"] == 0
    source.validate(data)
    with pytest.raises(PermissionError):
        replace(source, origin="historical-five-class").validate(data)
    with pytest.raises(PermissionError):
        replace(source, train_identity="wrong").validate(data)
    with pytest.raises(PermissionError):
        replace(source, normalization=Normalization.fit(data, target_only=True)).validate(data)
    with torch.no_grad():
        source.model.prediction_head.weight.add_(1)
    with pytest.raises(PermissionError):
        source.validate(data)


def test_provider_hash_change_detected():
    data, _, arrays = fixture_data()
    arrays["F"][0][0, 0] += 1
    with pytest.raises(PermissionError, match="Feature identity"):
        data.read("F", "fit_target")
    assert array_hash([np.zeros(1)]) != array_hash([np.ones(1)])
