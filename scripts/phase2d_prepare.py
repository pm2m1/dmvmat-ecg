"""Prepare the existing amended execution namespace; no authorization or fit CLI."""

import argparse
import json
from pathlib import Path

from dmvmat_ecg.phase2b.runtime import code_identity, environment_receipt, execution_inventory
from dmvmat_ecg.phase2c.auditor import immutable_json, ledger_snapshot
from dmvmat_ecg.phase2c.splits import source_anchor_support, specification
from dmvmat_ecg.phase2d.access import AmendedAccess, seal_candidate
from dmvmat_ecg.phase2d.contract import CANDIDATE_HASH, NAMESPACE, blank_review, contract, digest
from dmvmat_ecg.phase2d.ledger import AmendedLedger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", required=True)
    parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if (
        json.loads((root / "configs/phase2d/amended_execution_draft.json").read_text())
        != contract()
    ):
        raise PermissionError("Execution configuration differs from the content-pinned contract")
    if (
        json.loads((root / "configs/phase2d/HUMAN_REVIEW_UNSIGNED.json").read_text())
        != blank_review()
    ):
        raise PermissionError(
            "Unsigned review template must remain blank; use a separate review record"
        )
    auditor = json.loads(
        (root / "private/phase2c_joint_pattern_v1/auditor_manifest.json").read_text()
    )
    if digest(auditor) != CANDIDATE_HASH:
        raise PermissionError("Candidate changed")
    c = json.loads((root / "results/phase2c/METADATA_PREFLIGHT.json").read_text())
    if (
        auditor["summary"] != c["summary"]
        or source_anchor_support(auditor["roles"]["source_train"], specification())
        != c["source_anchors"]
    ):
        raise PermissionError("Actual G/source support or lineage changed")
    folder = root / "private/phase2d_joint44301_v1"
    worker = seal_candidate(auditor, folder / "role_vaults")
    immutable_json(folder / "worker_manifest.json", worker)
    env = environment_receipt(root)
    pins = execution_inventory(root)
    ledger = AmendedLedger(
        folder,
        namespace=NAMESPACE,
        manifest_hash=digest(worker),
        code_hash=code_identity(pins),
        dependency_hash=digest(env),
    )
    ledger.record_preflight(
        auditor["blockers"],
        digest(
            dict(
                candidate=CANDIDATE_HASH,
                support=c["evidence_support"],
                serial_audit=digest(
                    json.loads(
                        (root / "results/phase2d/SERIAL_ANNOTATION_AGGREGATE.json").read_text()
                    )
                ),
            )
        ),
    )
    access = AmendedAccess(ledger, worker, integrity_files=pins)
    denied = []
    for role, op in [
        ("source_train", "fit"),
        ("source_train", "evidence"),
        ("E", "stop"),
        ("C", "calibrate"),
        ("O", "evaluate_predict"),
        ("O", "evaluate_labels"),
    ]:
        try:
            access.rows(role, op)
        except PermissionError:
            denied.append([role, op])
        else:
            raise AssertionError("Unauthorized data access")
    for name, call in [
        ("adoption", lambda: ledger.record_adoption(blank_review())),
        ("fit", lambda: ledger.reserve("source")),
        ("evaluation", ledger.begin_evaluation),
    ]:
        try:
            call()
        except PermissionError:
            denied.append(name)
        else:
            raise AssertionError("Unauthorized transition")
    snapshot = ledger.snapshot()
    assert (
        snapshot["phase"] == "draft"
        and snapshot["attempts"] == snapshot["evaluation_accesses"] == 0
    )
    assert not snapshot["development_authorized"] and not snapshot["evaluation_authorized"]
    assert ledger.get("protocol_adoption") is None
    immutable_json(folder / "integrity_inventory.json", pins)
    receipt = dict(
        scope="PREPARATION ONLY",
        candidate_hash=CANDIDATE_HASH,
        worker_hash=digest(worker),
        contract_hash=digest(contract()),
        configuration=contract(),
        support=c["evidence_support"],
        source_anchors=c["source_anchors"],
        role_hashes=worker["role_hashes"],
        snapshot=snapshot,
        denied_operations=denied,
        environment=env,
        original_pilot=ledger_snapshot(root / "private/phase2b_internal/study.sqlite"),
        review_status="UNSIGNED — NOT ADOPTED",
        real_predictor_fits=0,
        real_preprocessing_fits=0,
        real_model_predictions=0,
        real_calibration_fits=0,
        real_evaluation_accesses=0,
        real_learned_representations=0,
    )
    ledger.close()
    immutable_json(root / "results/phase2d/EXECUTION_READINESS.json", receipt)
    print(
        json.dumps(
            dict(
                candidate_hash=CANDIDATE_HASH,
                snapshot=snapshot,
                permissions="DENY",
                review="UNSIGNED",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
