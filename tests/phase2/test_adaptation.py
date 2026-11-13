"""Synthetic in-memory training and source-lineage tests; no real dataset paths."""

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score
from torch import nn

from dmvmat_ecg.models.networks import FiveViewECG, parameters
from dmvmat_ecg.phase2.adaptation import (
    APCheckpoint,
    SelectiveModel,
    SourceLineage,
    adapt_synthetic,
    extract_synthetic,
    load_synthetic,
    macro_ap,
    masked_bce,
    random_mask,
    serialize_synthetic,
    state_hash,
    validate_owners,
)
from dmvmat_ecg.phase2.contracts import MODULES, SyntheticRecords, SyntheticRoles
from dmvmat_ecg.phase2.evidence import EvidenceConfig, analyze
from dmvmat_ecg.preprocessing.signals import views

torch.set_num_threads(2)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        for n in MODULES[:5]:
            setattr(self, n, nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.ReLU()))
        self.fusion_1 = nn.Sequential(nn.Linear(20, 4), nn.BatchNorm1d(4), nn.ReLU())
        self.fusion_2 = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.ReLU())
        self.prediction_head = nn.Linear(4, 2)

    def selectable_modules(self):
        return {n: getattr(self, n) for n in MODULES}

    def representations(self, wave):
        r = {n: getattr(self, n)(wave) for n in MODULES[:5]}
        r["fusion_1"] = self.fusion_1(torch.cat(list(r.values()), dim=1))
        r["fusion_2"] = self.fusion_2(r["fusion_1"])
        return r

    def forward(self, wave):
        return self.prediction_head(self.representations(wave)["fusion_2"])


def records(role, n=24):
    x = np.random.default_rng(13).normal(size=(n, 4)).astype("float32")
    y = np.tile([[0, 0], [0, 1], [1, 0], [1, 1]], (n // 4, 1)).astype(float)
    return SyntheticRecords(
        "synthetic:phase2-tests",
        role,
        tuple(f"invented-{role}-{i}" for i in range(n)),
        tuple(f"fictional-patient-{role}-{i}" for i in range(n)),
        dict(wave=x),
        y,
        np.ones_like(y, bool),
    )


def roles():
    return SyntheticRoles(
        [
            records(r)
            for r in [
                "source_train",
                "evidence",
                "fit",
                "stop",
                "calibration",
                "selection",
                "evaluation",
            ]
        ]
    )


def lineage(source):
    return SourceLineage(
        state_hash(source),
        records("source_train").groups,
        label_names=records("source_train").label_names,
    )


def test_owner_partition_real_architecture_counts_and_synthetic_rep_shapes():
    torch.manual_seed(101)
    model = FiveViewECG().eval()
    assert parameters(model) == 212261
    assert list(validate_owners(model)) == list(MODULES)
    model.prediction_head = nn.Linear(64, 2)
    assert parameters(model) == 212066
    expected = [32576, 31904, 31904, 31904, 33920, 41344, 8384]
    assert [parameters(v) for v in model.selectable_modules().values()] == expected
    x = np.random.default_rng(9).normal(size=(4, 12, 1000)).astype("float32")
    wave, rhythm, spectrum = [torch.tensor(v) for v in views(x)]
    with torch.no_grad():
        r = model.representations(wave, rhythm, spectrum)
    assert [v.shape for v in r.values()] == [(4, 64)] * 5 + [(4, 128), (4, 64)]


@pytest.mark.parametrize(
    "adapt", [None, "view_1", "view_4", "view_5", "fusion_1", "fusion_2", "all"]
)
def test_frozen_parameters_buffers_bn_affine_optimizer_and_gradients(adapt):
    torch.manual_seed(101)
    source = Tiny().eval()
    original = state_hash(source)
    mask = {n: "ADAPT" if adapt in (n, "all") else "SHARE" for n in MODULES}
    m = SelectiveModel(source, mask)
    wrapper = nn.Sequential(m)
    wrapper.train()  # Exercises recursive external model.train(), not manual eval only.
    opt = m.optimizer(lr=0.01)
    actual = {id(p) for group in opt.param_groups for p in group["params"]}
    expected = {id(p) for p in m.parameters() if p.requires_grad}
    assert actual == expected
    before = copy.deepcopy(m.model.state_dict())
    loss = masked_bce(
        m(wave=torch.randn(16, 4)), torch.ones(16, 2), torch.ones(16, 2, dtype=torch.bool)
    )
    loss.backward()
    opt.step()
    m.verify_frozen()
    assert not torch.equal(before["prediction_head.weight"], m.model.prediction_head.weight)
    for name, module in m.model.selectable_modules().items():
        trainable = mask[name] == "ADAPT"
        assert module.training is trainable
        for param in module.parameters():
            assert param.requires_grad is trainable
            assert (param.grad is not None) is trainable
        bn = next(child for child in module.modules() if isinstance(child, nn.BatchNorm1d))
        assert bn.weight.requires_grad is trainable and bn.bias.requires_grad is trainable
        assert int(bn.num_batches_tracked) == int(trainable)
    assert state_hash(source) == original


def test_freezing_verifier_detects_accidental_direct_buffer_update():
    m = SelectiveModel(Tiny(), dict.fromkeys(MODULES, "SHARE"))
    with torch.no_grad():
        m.model.view_1[1].running_mean.add_(1)
    with pytest.raises(RuntimeError, match="buffers"):
        m.verify_frozen()


def test_source_only_and_all_masks_preserve_initial_eval_output():
    source = Tiny().eval()
    x = torch.randn(8, 4)
    expected = source(x)
    for setting in ("SHARE", "ADAPT"):
        m = SelectiveModel(source, dict.fromkeys(MODULES, setting)).eval()
        torch.testing.assert_close(m(wave=x), expected, atol=0, rtol=0)
    m = SelectiveModel(source, dict.fromkeys(MODULES, "SHARE"), source_only=True)
    m.train()
    torch.testing.assert_close(m(wave=x), expected, atol=0, rtol=0)
    assert not any(p.requires_grad for p in m.parameters())
    with pytest.raises(ValueError):
        m.optimizer()


def test_weights_only_serialization_digest_and_corruption():
    source = Tiny()
    payload, h = serialize_synthetic(source)
    other = Tiny()
    load_synthetic(other, payload, h)
    assert state_hash(source) == state_hash(other)
    with pytest.raises(PermissionError):
        load_synthetic(other, payload[:-1], h)
    with pytest.raises(RuntimeError):
        load_synthetic(nn.Linear(3, 1), payload, h)


@pytest.mark.parametrize("name", list(MODULES) + ["all", "none"])
def test_random_exact_parameter_budget_reproducibility_and_degeneracy(name):
    model = FiveViewECG()
    reference = {n: "ADAPT" if name in (n, "all") else "SHARE" for n in MODULES}
    mask, info = random_mask(model, reference, 42)
    assert (mask, info) == random_mask(model, reference, 42)
    assert info["parameter_error"] == 0
    if name in {"all", "none"}:
        assert info["same_as_reference"] and info["candidate_count"] == 1


def test_masked_bce_never_learns_unknown_negatives():
    logits = torch.zeros((2, 2), requires_grad=True)
    target = torch.tensor([[1.0, float("nan")], [0, -999.0]])
    obs = torch.tensor([[True, False], [True, False]])
    loss = masked_bce(logits, target, obs)
    assert float(loss.detach()) == pytest.approx(np.log(2))
    loss.backward()
    torch.testing.assert_close(logits.grad[:, 1], torch.zeros(2))
    assert logits.grad[0, 0] < 0 and logits.grad[1, 0] > 0
    with pytest.raises(ValueError):
        masked_bce(logits, target, torch.zeros_like(obs))


def test_exact_ap_support_and_epoch_zero_protection():
    y = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    p = np.array([[0.9, 0.1], [0.8, 0.7], [0.7, 0.2], [0.1, 0.3]])
    assert macro_ap(y, p, np.ones_like(y, bool)) == pytest.approx(
        np.mean([average_precision_score(y[:, i], p[:, i]) for i in range(2)])
    )
    with pytest.raises(ValueError):
        macro_ap(y, p, np.zeros_like(y, bool))
    m = Tiny()
    original = state_hash(m)
    checkpoint = APCheckpoint()
    checkpoint.consider(m, ap=0.9, bce=0.3, epoch=0)
    with torch.no_grad():
        m.prediction_head.weight.add_(10)
    checkpoint.consider(m, ap=0.8, bce=0.1, epoch=1)
    assert checkpoint.epoch == 0
    m.load_state_dict(checkpoint.state)
    assert state_hash(m) == original
    checkpoint.consider(m, ap=0.9, bce=0.2, epoch=2)
    assert checkpoint.epoch == 2
    assert not checkpoint.consider(m, ap=0.9, bce=0.2, epoch=3)


def test_group_isolation_and_caller_role_relabel_rejected():
    f, g = records("fit"), records("evidence")
    with pytest.raises(ValueError, match="group"):
        SyntheticRoles([f, replace(g, groups=f.groups)])
    r = roles()
    for role, op in [
        ("evaluation", "fit"),
        ("evaluation", "evidence"),
        ("calibration", "fit_cost_matched"),
        ("stop", "fit"),
        ("evidence", "select"),
        ("fit", "stop"),
    ]:
        with pytest.raises(PermissionError):
            r.access(role, op)
    assert r.label_cost()["all"]["labels"] == 0


def test_hash_tampering_and_real_namespace_refused():
    r = roles()
    # Defensive read-only arrays can be forcibly changed by Python code; hashes detect it.
    f = r._records["fit"]
    f.y.flags.writeable = True
    f.y[0, 0] = 1
    with pytest.raises(PermissionError, match="changed"):
        r.access("fit", "fit")
    with pytest.raises(PermissionError, match="synthetic"):
        replace(records("fit"), namespace="PTBXL-real")


def test_source_training_or_selection_patient_history_leakage_refused():
    source, r = Tiny(), roles()
    for key in ("training_groups", "selection_groups"):
        bad = replace(lineage(source), **{key: records("evaluation").groups})
        with pytest.raises(PermissionError, match="overlaps"):
            adapt_synthetic(source, r, lineage=bad, method="full", updates=1)


def test_label_accounting_unique_union_and_all_roles():
    r = roles()
    for _ in range(2):
        r.access("fit", "fit_cost_matched")
        r.access("evidence", "fit_cost_matched")
        r.access("evidence", "evidence")
        r.access("stop", "stop")
        r.access("calibration", "calibrate")
        r.access("selection", "select")
    cost = r.label_cost()
    assert cost["all"] == dict(records=120, patients=120, labels=240)
    assert cost["fit_cost_matched"]["records"] == 48
    assert cost["evidence"]["records"] == 24
    assert set(cost["by_role"]) == {"fit", "evidence", "stop", "calibration", "selection"}
    assert cost["by_role"]["fit"]["negative_counts"] == [12, 12]
    b = records("fit")
    observed = b.observed.copy()
    observed[:, 1] = False
    partial = SyntheticRoles([replace(b, observed=observed)])
    partial.access("fit", "fit")
    assert partial.label_cost()["by_role"]["fit"]["negative_counts"] == [12, 0]
    assert partial.label_cost()["by_role"]["fit"]["unknown_counts"] == [0, 24]


@pytest.mark.parametrize(
    "method", ["source_only", "head_only", "full", "full_cost", "head_cost", "random", "selective"]
)
def test_comparator_execution_isolated_source_unchanged_and_label_costs(method):
    torch.manual_seed(51)
    source, r = Tiny().eval(), roles()
    original = state_hash(source)
    evidence = dict(
        mask=dict.fromkeys(MODULES, "SHARE"),
        checkpoint_hash=original,
        target_identity=records("evidence").identity,
    )
    controller, receipt = adapt_synthetic(
        source, r, lineage=lineage(source), method=method, evidence=evidence, updates=2, seed=11
    )
    controller.verify_frozen()
    assert state_hash(source) == original
    assert receipt["selected_update"] in (0, 1, 2)
    accessed = receipt["label_cost"]["all"]["records"]
    if method == "source_only":
        assert accessed == 0 and receipt["updates"] == 0
    elif method in {"head_only", "full"}:
        assert accessed == 48  # F + E
    else:
        assert accessed == 72  # F + G + E, including the selector-derived random cost.
    assert "evaluation" not in receipt["label_cost"]["by_role"]


def test_complete_extraction_selection_adaptation_flow_is_reproducible():
    torch.manual_seed(21)
    source = Tiny().eval()
    cfg = EvidenceConfig(permutations=39, bootstraps=199)

    def run():
        r = roles()
        s = extract_synthetic(
            source, r, "source_train", "synthetic-scaler", dimensions=dict.fromkeys(MODULES, 4)
        )
        g = extract_synthetic(
            source, r, "evidence", "synthetic-scaler", dimensions=dict.fromkeys(MODULES, 4)
        )
        evidence = analyze(s, g, config=cfg)
        adapted, receipt = adapt_synthetic(
            source,
            r,
            lineage=lineage(source),
            method="selective",
            evidence=evidence,
            updates=3,
            seed=101,
        )
        return state_hash(adapted), receipt

    a, b = run(), run()
    assert a == b


def test_evidence_from_another_target_or_source_rejected():
    source, r = Tiny(), roles()
    evidence = dict(
        mask=dict.fromkeys(MODULES, "SHARE"),
        checkpoint_hash=state_hash(source),
        target_identity="wrong-target-role-hash",
    )
    with pytest.raises(PermissionError):
        adapt_synthetic(source, r, lineage=lineage(source), method="selective", evidence=evidence)


@pytest.mark.parametrize("trainable", ["none", "view_2", "all"])
def test_complete_ecg_architecture_frozen_bn_backward_and_checkpoint(trainable):
    torch.manual_seed(101)
    source = FiveViewECG().eval()
    source.prediction_head = nn.Linear(64, 2)
    initial = state_hash(source)
    mask = {n: "ADAPT" if trainable in (n, "all") else "SHARE" for n in MODULES}
    controller = SelectiveModel(source, mask)
    x = np.random.default_rng(75).normal(size=(4, 12, 1000)).astype("float32")
    inputs = dict(zip(["wave", "rhythm", "spectrum"], map(torch.tensor, views(x)), strict=True))
    with torch.no_grad():
        torch.testing.assert_close(controller(**inputs), source(**inputs), atol=0, rtol=0)
    controller.train()
    optimizer = controller.optimizer()
    loss = masked_bce(controller(**inputs), torch.zeros(4, 2), torch.ones(4, 2, dtype=torch.bool))
    loss.backward()
    optimizer.step()
    controller.verify_frozen()
    assert state_hash(source) == initial
    payload, sha = serialize_synthetic(controller)
    restored = SelectiveModel(source, mask)
    load_synthetic(restored, payload, sha)
    restored.eval()
    controller.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(**inputs), controller(**inputs), atol=0, rtol=0)


def test_undeclared_source_history_and_stale_source_hash_rejected():
    source, r = Tiny(), roles()
    bad = SourceLineage(state_hash(source), ("unverified-outside-patient",))
    with pytest.raises(PermissionError, match="undeclared"):
        adapt_synthetic(source, r, lineage=bad, method="full")
    bad = replace(lineage(source), checkpoint_hash="wrong-source-hash")
    with pytest.raises(PermissionError):
        adapt_synthetic(source, r, lineage=bad, method="full")


def test_target_label_cost_excludes_source_anchors_but_reports_them_separately():
    r = roles()
    r.access("source_train", "evidence")
    r.access("evidence", "evidence")
    r.access("fit", "fit")
    result = r.label_cost()
    assert result["all"]["records"] == 72
    assert result["target_all"]["records"] == 48
    assert result["source_all"]["records"] == 24


def test_extraction_does_not_update_source_bn_or_mode_and_refuses_outer():
    source, r = Tiny().train(), roles()
    initial = state_hash(source)
    rep = extract_synthetic(
        source, r, "source_train", "fictional-scaler", dimensions=dict.fromkeys(MODULES, 4)
    )
    assert rep.role == "source_train" and source.training
    assert initial == state_hash(source)
    with pytest.raises(PermissionError):
        extract_synthetic(source, r, "evaluation", "fictional-scaler")


def test_manifest_matches_evidence_defaults_without_authorizing_real_execution():
    import json
    from dataclasses import asdict
    from pathlib import Path

    manifest = json.loads(
        (Path(__file__).parents[2] / "configs/phase2/synthetic_selector.json").read_text()
    )
    assert not manifest["real_execution_authorized"]
    assert manifest["device"] == "cpu"
    assert {k: manifest[k] for k in asdict(EvidenceConfig())} == asdict(EvidenceConfig())


def test_matching_dimensions_cannot_hide_a_reordered_label_task():
    a, b = records("fit"), records("evidence")
    with pytest.raises(ValueError, match="schema"):
        SyntheticRoles([a, replace(b, label_names=tuple(reversed(b.label_names)))])
    source, r = Tiny(), roles()
    bad = replace(lineage(source), label_names=("different", "task"))
    with pytest.raises(PermissionError, match="schema"):
        adapt_synthetic(source, r, lineage=bad, method="source_only")
