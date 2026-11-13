"""Fictional metadata only; never opens PTB-XL or a real pilot ledger."""

import copy
import inspect
import json
from pathlib import Path

import pytest

from dmvmat_ecg.phase2b.contract import digest
from dmvmat_ecg.phase2b.ledger import PilotLedger
from dmvmat_ecg.phase2b.preflight import stratify, validate_manifest
from dmvmat_ecg.phase2c.auditor import DraftBindings, context, fit_matrix, immutable_json
from dmvmat_ecg.phase2c.diagnostics import expected_draw_support, original_objective_trace
from dmvmat_ecg.phase2c.splits import (
    PATTERNS,
    SPEC_HASH,
    build_candidate,
    canonicalize,
    joint_allocate,
    pattern,
    source_anchor_support,
    specification,
    support,
    validate,
)


def rows_for(counts, age=40, start=0):
    rows = []
    for status, n in zip(PATTERNS, counts, strict=True):
        for _ in range(n):
            i = start + len(rows)
            rows.append(
                dict(
                    ecg_id=i,
                    patient_id=i,
                    age=age,
                    fold=i % 9 + 1,
                    filename=f"fictional/{i}",
                    raw_hash=f"dat{i}",
                    header_hash=f"hea{i}",
                    y=list(status),
                    observed=[True, True],
                )
            )
    return rows


@pytest.fixture
def records():
    return rows_for([4000, 1000, 1000, 1000]) + rows_for(
        [2400, 1800, 1200, 600], age=70, start=10000
    )


@pytest.fixture
def candidate(records):
    return build_candidate(records, {"fictional": True})


def test_prespecified_single_seed_no_override(records):
    spec = specification()
    assert (
        digest(spec) == SPEC_HASH and spec["seed"] == 44301 and spec["candidate_generations"] == 1
    )
    assert "seed" not in inspect.signature(build_candidate).parameters
    with pytest.raises(TypeError):
        build_candidate(records, {}, seed=44302)


def test_deterministic_order_invariant_and_stable_hash(records, candidate):
    assert candidate == build_candidate(list(reversed(records)), {"fictional": True})
    assert digest(candidate) == digest(build_candidate(records, {"fictional": True}))
    assert validate(candidate)


def test_joint_quotas_all_patterns_and_role_sizes(records):
    allocated = joint_allocate(
        records[:7000], ["a", "b"], ["0.85", "0.15"], seed=44301, domain="source"
    )
    for k, frac in [("a", 0.85), ("b", 0.15)]:
        for actual, total in zip(
            support(allocated[k])["joint"], [4000, 1000, 1000, 1000], strict=True
        ):
            assert abs(actual - total * frac) < 1
    assert sum(map(len, allocated.values())) == 7000


def test_rare_joint_pattern_preserved_without_invention():
    rows = rows_for([900, 50, 47, 3])
    a = joint_allocate(rows, ["a", "b"], ["0.85", "0.15"], seed=44301, domain="source")
    assert sum(support(r)["joint"][3] for r in a.values()) == 3
    assert all(
        abs(support(r)["joint"][3] - 3 * f) < 1
        for r, f in zip(a.values(), [0.85, 0.15], strict=True)
    )


def test_genuine_zero_support_target_stays_zero():
    rows = rows_for([4000, 1000, 1000, 1000]) + rows_for([3000, 1500, 1500, 0], age=70, start=10000)
    m = build_candidate(rows, {})
    assert m["domains"]["target"]["joint"][3] == 0
    assert all(
        not e["B_support_pass"] and e["A_support_pass"] for e in m["evidence_support"].values()
    )
    assert not m["blockers"]  # B abstention alone does not invalidate primary A.


def test_domains_different_prevalences_not_forced_equal(candidate):
    s, t = candidate["domains"]["source"], candidate["domains"]["target"]
    assert s["prevalence"] != t["prevalence"]
    assert all(abs(v) < 0.2 for d in candidate["deviations_pp"].values() for v in d["joint_pp"])


def test_repeated_patient_conflicting_diagnoses_use_earliest_not_or(records):
    duplicate = {**records[0], "ecg_id": 99999, "age": 70, "raw_hash": "another", "y": [1, 1]}
    canonical, audit = canonicalize(records + [duplicate])
    retained = next(r for r in canonical if r["patient_id"] == records[0]["patient_id"])
    assert retained["y"] == [0, 0] and retained["age"] == 40
    assert audit["differing_observed_annotations_by_label"] == [1, 1]
    assert audit["differing_joint_status_patients"] == 1
    assert audit["eligible_cross_age_boundary_patients"] == 1
    assert audit["additional_eligible_records"] == 1


def test_unknown_preserved_and_not_counted_negative(records):
    r = {**records[0], "observed": [False, True], "y": [None, 0]}
    assert pattern(r) == "?0"
    assert support([r])["unknown"] == [1, 0] and support([r])["negatives"] == [0, 1]
    m = build_candidate([r, *records[1:]], {})
    assert any("Unknown" in b for b in m["blockers"])
    assert any(
        not x["observed"][0] for role in ["source_train", "source_stop"] for x in m["roles"][role]
    )


@pytest.mark.parametrize("bad", [[2, 0], [0, float("nan")]])
def test_invalid_observed_annotations_rejected(bad):
    with pytest.raises(ValueError):
        pattern({"y": bad, "observed": [True, True]})


def test_source_stopping_representative(candidate):
    natural = candidate["domains"]["source"]["joint"]
    train, stop = [candidate["summary"][k]["joint"] for k in ["source_train", "source_stop"]]
    assert all(abs(a - 0.15 * b) < 1 for a, b in zip(stop, natural, strict=True))
    assert train[3] > stop[3]


def test_original_objective_failure_reproduced_without_domain_change():
    rows = rows_for([5600, 900, 950, 160])
    a = stratify(rows, ["train", "stop"], [0.85, 0.15], 33002)
    trace = original_objective_trace(rows, ["train", "stop"], [0.85, 0.15], 33002)
    assert trace["role_counts"] == {k: support(v) for k, v in a.items()}
    assert trace["first_pattern"] == [1, 1]
    assert (
        trace["first_joint_objective_deltas"]["stop"]
        < trace["first_joint_objective_deltas"]["train"]
    )
    assert trace["joint_assignments"]["stop"] > trace["joint_assignments"]["train"]


def test_evidence_anchors_class_caps_and_four_cells(candidate):
    anchors = source_anchor_support(candidate["roles"]["source_train"], specification())
    for a in anchors:
        j = a["label"]
        assert a["positives"][j] == 256 and a["negatives"][j] == 256
        assert a["A_support_pass"] and a["B_support_pass"]
    assert all(e["A_support_pass"] for e in candidate["evidence_support"].values())


def test_small_natural_draw_abstention_not_enriched():
    population = rows_for([700, 200, 99, 1])
    risk = expected_draw_support(population)
    assert risk["probability_cell_below_minimum"][3] == 1
    assert risk["probability_any_cell_below_minimum_bounds"] == [1, 1]
    assert sum(risk["expected_joint_counts"]) == pytest.approx(256)


def test_full_cell_floor_exactly_ten_and_nine():
    for counts, expected in [([10, 10, 10, 10], True), ([10, 10, 10, 9], False)]:
        assert (min(support(rows_for(counts))["joint"]) >= 10) is expected


def test_impossible_primary_support_and_capacity_returns_blockers():
    m = build_candidate(rows_for([30, 3, 3, 1]) + rows_for([30, 3, 3, 1], age=70, start=1000), {})
    assert any("capacity" in b for b in m["blockers"])
    assert any("primary A" in b for b in m["blockers"])
    assert any("source_train" in b for b in m["blockers"])


def test_all_active_roles_disjoint_nested_and_covered(candidate):
    for draw in (4101, 4102):
        for n in (128, 512):
            used = set()
            for role in [
                "source_train",
                "source_stop",
                f"d{draw}:n{n}:F",
                f"d{draw}:n{n}:G",
                "E",
                "C",
                "O",
            ]:
                groups = {r["patient_id"] for r in candidate["roles"][role]}
                assert not groups & used
                used |= groups
    assert validate(candidate)


@pytest.mark.parametrize("kind", ["fold", "group", "domain", "hash", "old_version", "population"])
def test_corrupted_manifest_rejected(candidate, kind):
    m = copy.deepcopy(candidate)
    if kind == "fold":
        m["roles"]["O"][0]["fold"] = 10
    if kind == "group":
        m["roles"]["O"][0]["patient_id"] = m["roles"]["source_train"][0]["patient_id"]
    if kind == "domain":
        m["roles"]["O"][0]["age"] = 30
    if kind == "hash":
        m["role_hashes"]["O"] = "bad"
    if kind == "old_version":
        m["split_version"] = "phase2b"
    if kind == "population":
        m["population"]["canonical_patients"] += 1
    if kind in {"fold", "group", "domain"}:
        m["role_hashes"]["O"] = digest(m["roles"]["O"])
    with pytest.raises((PermissionError, ValueError)):
        validate(m)


def test_all_record_duplicate_cross_patient_rejected_before_canonicalization(records):
    d = {**records[0], "ecg_id": 99999, "raw_hash": records[-1]["raw_hash"]}
    with pytest.raises(ValueError, match="Cross-patient"):
        canonicalize(records + [d])


def test_fit_matrix_primary_label_cost_and_no_attempt_consumption(candidate):
    m = fit_matrix(candidate)
    assert (
        m["planned"] == 33
        and m["ceiling"] == 36
        and m["reserve"] == 3
        and m["attempts_reserved"] == 0
    )
    for row in m["matrix"][1:]:
        n = row["spec"]["n_fit"]
        expected = n + 256 if row["spec"]["method"] in {"head_f", "full_f"} else n + 512
        assert row["target_labels_with_C"] == expected
    a = next(r for r in m["matrix"] if r["job"] == "d4101:n512:selective_a")
    f = next(r for r in m["matrix"] if r["job"] == "d4101:n512:full_cost")
    assert a["target_labels_with_C"] == f["target_labels_with_C"] == 1024
    assert a["spec"]["updates"] == f["spec"]["updates"] == 160


def test_bindings_all_predictive_access_denied_and_worker_has_no_identities(candidate):
    binding = context(candidate, "code", "deps")
    access = DraftBindings(candidate, binding)
    for role, op in [
        ("source_train", "fit"),
        ("source_train", "preprocess_fit"),
        ("E", "stop"),
        ("C", "calibrate"),
        ("O", "evaluate_predict"),
        ("O", "evaluate_labels"),
        ("A", "evidence"),
        ("O", "rows"),
    ]:
        with pytest.raises(PermissionError):
            access.request(role, op)
    assert access.request("O", "audit_counts")["patients"] > 0
    view = json.dumps(access.development_view())
    assert "patient_id" not in view and "ecg_id" not in view and '"y"' not in view
    assert binding["predictor_attempts"] == binding["evaluation_accesses"] == 0
    assert not binding["training_authorization"] and not binding["evaluation_authorization"]


def test_mixed_versions_context_and_ledger_rejected(candidate, tmp_path):
    binding = context(candidate, "code", "deps")
    changed = copy.deepcopy(binding)
    changed["manifest_hash"] = "old"
    with pytest.raises(PermissionError):
        DraftBindings(candidate, changed)
    changed = copy.deepcopy(binding)
    changed["training_authorization"] = True
    with pytest.raises(PermissionError):
        DraftBindings(candidate, changed)
    with pytest.raises((PermissionError, ValueError)):
        validate_manifest(candidate)
    # The old executor cannot adopt the new draft namespace, even with its hash.
    with pytest.raises(ValueError, match="namespace"):
        PilotLedger(
            tmp_path,
            namespace=binding["namespace"],
            manifest_hash=binding["manifest_hash"],
            code_hash="code",
            dependency_hash="deps",
        )


def test_historical_checkpoint_normalization_and_role_hashes_rejected(candidate):
    binding = context(candidate, "code", "deps")
    access = DraftBindings(candidate, binding)
    lineage = dict(
        namespace=binding["namespace"],
        split_version=binding["split_version"],
        manifest_hash=binding["manifest_hash"],
        labels=["CD", "STTC"],
        origin="random-scratch",
        fresh=True,
        source_training_hash=candidate["role_hashes"]["source_train"],
        source_selection_hash=candidate["role_hashes"]["source_stop"],
        preprocessing_fit_role="source_train",
        preprocessing_membership_hash=candidate["role_hashes"]["source_train"],
        code_hash="code",
        dependency_hash="deps",
    )
    assert access.verify_fresh_source_lineage(lineage)
    for key, value in [
        ("namespace", "real:phase2b-internal"),
        ("origin", "historical"),
        ("fresh", False),
        ("source_training_hash", "old"),
        ("preprocessing_membership_hash", "old"),
        ("preprocessing_fit_role", "O"),
    ]:
        bad = {**lineage, key: value}
        with pytest.raises(PermissionError):
            access.verify_fresh_source_lineage(bad)


def test_immutable_manifest_reuse_no_overwrite_and_code_drift(candidate, tmp_path):
    p = tmp_path / "candidate.json"
    h = immutable_json(p, candidate)
    assert immutable_json(p, candidate) == h
    with pytest.raises(PermissionError):
        immutable_json(p, {"changed": True})
    access = DraftBindings(candidate, context(candidate, "code", "deps"), files={str(p): h})
    p.write_text("changed")
    with pytest.raises(PermissionError):
        access.request("O", "audit_counts")


def test_role_restrictions_file_inventory_is_versioned():
    root = Path(__file__).resolve().parents[2]
    assert (root / "src/dmvmat_ecg/phase2c").is_dir()
    assert specification()["status"].startswith("DRAFT")


def test_old_ledger_cannot_be_reused_with_new_membership(tmp_path, candidate):
    folder = tmp_path / "old-ledger"
    old = PilotLedger(
        folder, namespace="synthetic:old", manifest_hash="old", code_hash="c", dependency_hash="d"
    )
    assert old.snapshot()["attempts"] == 0
    old.close()
    original = (folder / "study.sqlite").read_bytes()
    with pytest.raises(PermissionError, match="foreign or changed"):
        PilotLedger(
            folder,
            namespace="synthetic:old",
            manifest_hash=digest(candidate),
            code_hash="c",
            dependency_hash="d",
        )
    assert (folder / "study.sqlite").read_bytes() == original


def test_real_entrypoint_is_metadata_only_and_requires_explicit_flag():
    import ast

    root = Path(__file__).resolve().parents[2]
    code = (root / "scripts/phase2c_preflight.py").read_text()
    assert "--metadata-only" in code and "required=True" in code
    parsed = ast.parse(code)
    calls = [node for node in ast.walk(parsed) if isinstance(node, ast.Call)]
    names = {node.func.id for node in calls if isinstance(node.func, ast.Name)}
    assert not {"run_job", "fit_preprocessing", "predict", "train", "calibrate", "analyze"} & names
