"""Fictional metadata only. Real metadata audit is a separate, non-predictive step."""

import copy

import pytest

from dmvmat_ecg.phase2b.contract import contract, digest, dry_run, jobs
from dmvmat_ecg.phase2b.preflight import build_manifest, scientific_review_flags, validate_manifest


def fictional_rows():
    return [
        dict(
            ecg_id=i,
            patient_id=i,
            age=30.0 if i < 600 else 70.0,
            fold=i % 9 + 1,
            filename=f"fictional/{i}",
            raw_hash=f"fake-dat-{i}",
            header_hash=f"fake-hea-{i}",
            y=[i % 2, (i // 2) % 2],
            observed=[True, True],
        )
        for i in range(2600)
    ]


@pytest.fixture
def manifest():
    return build_manifest(fictional_rows(), {"synthetic": True})


def test_complete_deterministic_roles_and_registered_matrix(manifest):
    assert not manifest["blockers"]
    assert manifest == build_manifest(fictional_rows(), {"synthetic": True})
    assert validate_manifest(manifest)
    assert len(jobs()) == 33 and contract()["predictor_attempt_ceiling"] == 36
    for draw in (4101, 4102):
        assert jobs()[f"d{draw}:n512:full_cost"]["fit_roles"] == [
            f"d{draw}:n512:F",
            f"d{draw}:n512:G",
        ]
        assert jobs()[f"d{draw}:n128:selective_a"]["updates"] == 40
        assert jobs()[f"d{draw}:n512:selective_a"]["updates"] == 160


def test_canonicalization_age_exclusion_and_no_fold10(manifest):
    rows = fictional_rows()
    rows.append({**rows[0], "ecg_id": 5000, "age": 40.0, "raw_hash": "other"})
    rows.append({**rows[1], "ecg_id": 5001, "patient_id": 5001, "age": 300.0})
    m = build_manifest(rows, {"synthetic": True})
    assert m["population"]["canonical_patients"] == 2600
    with pytest.raises(ValueError, match="development-only"):
        build_manifest([{**rows[0], "fold": 10}], {})


@pytest.mark.parametrize("kind", ["label", "hash", "patient", "duplicate", "unknown"])
def test_modified_or_leaking_manifests_rejected(manifest, kind):
    m = copy.deepcopy(manifest)
    if kind == "label":
        m["labels"] = ["IAVB", "IRBBB"]
    elif kind == "hash":
        m["roles"]["E"][0]["y"][0] = 1 - m["roles"]["E"][0]["y"][0]
    elif kind == "patient":
        m["roles"]["O"][0]["patient_id"] = m["roles"]["source_train"][0]["patient_id"]
        m["role_hashes"]["O"] = digest(m["roles"]["O"])
    elif kind == "duplicate":
        m["roles"]["O"][0]["raw_hash"] = m["roles"]["source_train"][0]["raw_hash"]
        m["role_hashes"]["O"] = digest(m["roles"]["O"])
    else:
        m["roles"]["O"][0]["observed"] = [False, True]
        m["role_hashes"]["O"] = digest(m["roles"]["O"])
    with pytest.raises(ValueError):
        validate_manifest(m)


def test_cross_person_identical_waveform_blocks_without_silent_exclusions():
    rows = fictional_rows()
    rows[-1]["raw_hash"] = rows[0]["raw_hash"]
    with pytest.raises(ValueError, match="protocol review"):
        build_manifest(rows, {})


def test_draft_config_and_structural_joint_cell_review_flags(manifest):
    import json
    from pathlib import Path

    p = Path(__file__).parents[2] / "configs/phase2b/internal_pilot_draft.json"
    assert json.loads(p.read_text()) == contract()
    assert not contract()["protocol_approval"] and not contract()["training_authorization"]
    assert not scientific_review_flags(manifest)
    broken = copy.deepcopy(manifest)
    broken["summary"]["A"]["joint"][3] = 0
    assert any("formulation B" in f for f in scientific_review_flags(broken))
    plan = dry_run(manifest)
    assert plan["planned"] == 33 and not plan["training_started"]
    assert plan["matrix"][0]["target_label_records_before_C"] == 0
    for row in plan["matrix"][1:]:
        n = row["spec"]["n_fit"]
        expected = n + 256 if row["spec"]["method"] in {"head_f", "full_f"} else n + 512
        assert row["target_label_records_with_C"] == expected
