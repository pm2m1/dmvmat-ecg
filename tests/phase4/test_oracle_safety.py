"""Invented metadata/arrays only; no existing experiment or cohort input reads."""

import subprocess
import sys

import pytest

from dmvmat_ecg.phase2.adaptation import SelectiveModel
from dmvmat_ecg.phase2.contracts import MODULES
from dmvmat_ecg.phase3b.models import fresh_model
from dmvmat_ecg.phase4.runtime import mask_dict, mask_index
from dmvmat_ecg.phase4.safety import RecordGuard, exclusion_hash, project_json, safe_path


@pytest.mark.parametrize("index", [0, 1, 2, 63, 64, 126, 127])
def test_mask_roundtrip_and_module_order(index):
    mask = mask_dict(index)
    assert tuple(mask) == MODULES
    assert mask_index(mask) == index
    assert [v == "ADAPT" for v in mask.values()] == [b == "1" for b in f"{index:07b}"]


@pytest.mark.parametrize("index", [-1, 128, 1.5, True])
def test_invalid_mask_index(index):
    with pytest.raises(ValueError):
        mask_dict(index)


def test_projected_reader_never_decodes_unselected_subtree(tmp_path):
    p = tmp_path / "invented.json"
    # 1e999 is deliberately not materialized as a nonfinite float.
    p.write_text(
        '{"allowed":{"R":["fictional-a"],"O":["fictional-b"]},'
        '"unselected":{"number":1e999,"quoted":"escape\\"ok"}}'
    )
    assert project_json(p, [("allowed", "R")]) == {("allowed", "R"): ["fictional-a"]}


def test_missing_or_duplicate_fields_rejected(tmp_path):
    p = tmp_path / "invented.json"
    p.write_text('{"a":1,"a":2}')
    with pytest.raises(ValueError):
        project_json(p, [("a",)])
    p.write_text('{"b":1}')
    with pytest.raises(ValueError):
        project_json(p, [("a",)])


def test_record_guards_deny_excluded_and_unknown_before_access():
    g = RecordGuard(["fictional-adapt"], [exclusion_hash("fictional-test")])
    g.check(["fictional-adapt"])
    for value in ["fictional-test", "fictional-unknown"]:
        with pytest.raises(PermissionError):
            g.check([value])
    with pytest.raises(PermissionError):
        RecordGuard(["fictional-test"], [exclusion_hash("fictional-test")])


def test_record_guard_rejects_duplicates():
    with pytest.raises(PermissionError):
        RecordGuard(["fictional-a", "fictional-a"], [])


def test_paths_refuse_escape_symlink_and_sealed(tmp_path):
    root = tmp_path / "allowed"
    root.mkdir()
    p = root / "input.json"
    p.write_text("{}")
    assert safe_path(p, root) == p
    link = root / "link.json"
    link.symlink_to(p)
    for candidate in [link, root / "sealed" / "x.json", tmp_path / "outside.json"]:
        with pytest.raises(PermissionError):
            safe_path(candidate, root)


def test_builder_rejects_lexical_traversal_before_open(tmp_path):
    from dmvmat_ecg.phase4.build_bundle import safe_path as input_path

    root = tmp_path / "allowed"
    root.mkdir()
    with pytest.raises(PermissionError, match="traversal"):
        input_path(root / ".." / "outside.json", root)


@pytest.mark.parametrize(
    "forbidden", ["private/old_run/labels.json", "data/test.npy", "results/old.json"]
)
def test_runtime_audit_denies_old_inputs_in_isolated_process(tmp_path, forbidden):
    # Fictional paths; opening is rejected even though these files do not exist.
    code = """
from pathlib import Path
from dmvmat_ecg.phase4.safety import bundle_only_audit
root=Path(%s)
bundle_only_audit(root,root/'private/new_bundle',root/'private/new_run')
try:
    (root/%s).open('rb')
except PermissionError:
    print('BLOCKED')
else:
    raise AssertionError('Forbidden open was accepted')
""" % (repr(str(tmp_path)), repr(forbidden))
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "BLOCKED"


def test_frozen_modules_keep_bn_and_head_trainable():
    model = SelectiveModel(fresh_model(101), mask_dict(0))
    model.train()
    assert model.model.prediction_head.training
    assert all(p.requires_grad for p in model.model.prediction_head.parameters())
    for module in model.model.selectable_modules().values():
        assert not module.training
        assert all(not p.requires_grad for p in module.parameters())
    model.verify_frozen()


def test_full_mask_adapts_all_modules():
    model = SelectiveModel(fresh_model(101), mask_dict(127))
    model.train()
    assert all(m.training for m in model.model.selectable_modules().values())
    assert all(p.requires_grad for p in model.parameters())
