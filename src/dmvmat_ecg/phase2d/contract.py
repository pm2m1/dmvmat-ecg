"""Proposed amended execution contract. Definition is not adoption/permission."""

from dataclasses import asdict

from dmvmat_ecg.phase2.evidence import EvidenceConfig
from dmvmat_ecg.phase2b.contract import LABELS as LABELS
from dmvmat_ecg.phase2b.contract import contract as parent_contract
from dmvmat_ecg.phase2b.contract import digest as digest
from dmvmat_ecg.phase2b.contract import file_hash as file_hash
from dmvmat_ecg.phase2c.splits import SPEC_HASH, VERSION

CANDIDATE_HASH = "c809d6db691d7ad3094dcd3635546ac58941529d1c6a3d06be35129f8ca0f4c9"
NAMESPACE = "real:phase2d-joint44301-v1"
REVIEW_ITEMS = (
    "endpoint_annotation_semantics",
    "canonical_exact_record_policy",
    "age_domains_and_eligibility",
    "joint_pattern_seed44301",
    "replaced_O_and_audit_history",
    "A_B_validity_and_abstention",
    "matched_labels_and_updates",
    "33_plus_3_ceiling",
    "stopping_20_post_update_looks_plus_epoch_zero",
    "calibration_failure_identity_and_threshold_point5",
    "internal_exploratory_limits",
)


def contract():
    cfg = parent_contract()
    cfg.update(
        execution_version="phase2d-amended-v1",
        split_version=VERSION,
        split_spec_hash=SPEC_HASH,
        split_seed=44301,
        source_split_seed=44301,
        target_split_seed=44301,
        superseded_split_seeds={"source": 33002, "target": 33001},
        candidate_auditor_hash=CANDIDATE_HASH,
        canonical_policy="lowest-age-eligible-ecg-id-exact-record-annotations-v1",
        mapping="PTBXL diagnostic any-key including unknown likelihood0; CD,STTC order",
        selector_configuration=asdict(EvidenceConfig()),
        source_stopping="exact two-label macro AP; tie lower BCE then earlier; scratch zero diagnostic",
        target_stopping="20 post-update looks plus pretrained-eligible zero; patience5; AP then BCE then earlier",
        calibration="positive-slope Platt C-only; support/convergence identity fallback with threshold0.5",
        optional_P_role=None,
    )
    return cfg


def blank_review():
    return dict(
        status="UNSIGNED — NOT ADOPTED",
        execution_contract_hash=digest(contract()),
        candidate_auditor_hash=CANDIDATE_HASH,
        decisions={
            item: dict(approved=None, reviewer=None, reviewer_role=None, date=None, notes=None)
            for item in REVIEW_ITEMS
        },
        training_authorization=False,
        evaluation_authorization=False,
    )


def validate_adoption(review):
    expected = blank_review()
    if set(review) != set(expected) or review.get("status") != "ADOPTED BY HUMAN REVIEW":
        raise PermissionError("Unsigned/incomplete protocol review")
    if any(
        review.get(k) != expected[k]
        for k in [
            "execution_contract_hash",
            "candidate_auditor_hash",
            "training_authorization",
            "evaluation_authorization",
        ]
    ):
        raise PermissionError("Wrong contract/candidate or implicit training grant")
    if set(review.get("decisions", {})) != set(REVIEW_ITEMS):
        raise PermissionError("Missing decisions")
    for decision in review["decisions"].values():
        if decision.get("approved") is not True or any(
            not isinstance(decision.get(k), str) or not decision[k].strip()
            for k in ["reviewer", "reviewer_role", "date", "notes"]
        ):
            raise PermissionError("Explicit attributable human decision required")
    # This verifies a supplied decision record, not authenticity of a signature.
    # The operator must obtain real authority; synthetic fixtures use fictional reviewers.
    return True
