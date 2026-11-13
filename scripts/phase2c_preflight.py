"""Single-specification metadata-only re-preflight. No predictive execution path."""

import argparse
import copy
import json
from pathlib import Path

from dmvmat_ecg.phase2b.contract import digest, file_hash
from dmvmat_ecg.phase2b.runtime import code_identity, environment_receipt, execution_inventory
from dmvmat_ecg.phase2c.auditor import (
    DraftBindings,
    context,
    fit_matrix,
    generate_metadata_candidate,
    immutable_json,
    ledger_snapshot,
)
from dmvmat_ecg.phase2c.diagnostics import expected_draw_support, original_objective_trace
from dmvmat_ecg.phase2c.splits import SPEC_HASH, VERSION, specification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-only", action="store_true", required=True)
    args = parser.parse_args()
    if not args.metadata_only:
        raise PermissionError("Metadata-only authority required")
    root = Path(__file__).resolve().parents[1]
    spec = specification()  # The immutable prospective specification predates generation.
    old_db = root / "private/phase2b_internal/study.sqlite"
    before = ledger_snapshot(old_db)
    old_db_hash = file_hash(old_db)
    if (
        before["phase"] != "draft"
        or before["attempts"]
        or before["evaluation_accesses"]
        or before.get("development_authorization")
        or before.get("evaluation_authorization")
    ):
        raise PermissionError("Original ledger is no longer untouched draft")
    private = root / "private/phase2c_joint_pattern_v1"
    new, comparison = generate_metadata_candidate(root)
    candidate_hash = digest(new)
    # One prospective candidate only; a changed candidate is never overwritten.
    immutable_json(
        private / "generation.json",
        dict(
            spec_hash=SPEC_HASH,
            split_version=VERSION,
            seed=spec["seed"],
            candidate_hash=candidate_hash,
            unique_candidates=1,
            determinism_checks=2,
            predictive_feedback_used=False,
        ),
    )
    immutable_json(private / "auditor_manifest.json", new)
    env = environment_receipt(root)
    inventory = execution_inventory(root)
    binding = context(new, code_identity(inventory), digest(env))
    immutable_json(private / "draft_context.json", binding)
    immutable_json(private / "integrity_inventory.json", inventory)
    access = DraftBindings(new, binding, files=inventory)
    denied = []
    for role, operation in [
        ("source_train", "fit"),
        ("source_train", "preprocess_fit"),
        ("E", "stop"),
        ("A", "evidence"),
        ("C", "calibrate"),
        ("O", "evaluate_predict"),
        ("O", "evaluate_labels"),
    ]:
        try:
            access.request(role, operation)
        except PermissionError:
            denied.append([role, operation])
        else:
            raise AssertionError("Draft unexpectedly granted predictive access")
    worker = access.development_view()
    immutable_json(private / "worker_summary.json", worker)
    combined = {k: [] for k in ["source", "target"]}
    # The new domains are exactly equal to the old domains; traces cannot alter roles.
    for k in ["source_train", "source_stop"]:
        combined["source"].extend(new["roles"][k])
    for k in ["O", "E_reserve", "C_reserve", "A"]:
        combined["target"].extend(new["roles"][k])
    trace = dict(
        source=original_objective_trace(
            combined["source"], ["source_train", "source_stop"], [0.85, 0.15], 33002
        ),
        target=original_objective_trace(
            combined["target"], ["O", "E_reserve", "C_reserve", "A"], [0.3, 0.1, 0.1, 0.5], 33001
        ),
    )
    for domain in trace.values():
        for role, counts in domain["role_counts"].items():
            if counts != comparison["original_role_counts"][role]:
                raise AssertionError(
                    "Instrumented original objective failed to reproduce original roles"
                )
    # Stratification audit labels are not model-development supervision.
    public = {k: copy.deepcopy(v) for k, v in new.items() if k != "roles"}
    public.update(
        candidate_manifest_hash=candidate_hash,
        comparison=comparison,
        original_objective_trace=trace,
        draw_support_risk=expected_draw_support(new["roles"]["A"]),
        fit_matrix=fit_matrix(new),
        context=binding,
        environment=env,
        denied_operations=denied,
        old_ledger_before=before,
        old_ledger_after=ledger_snapshot(old_db),
        real_predictor_attempts=0,
        real_preprocessing_fits=0,
        real_predictions=0,
        real_evaluations=0,
        real_waveform_reads=0,
        new_learned_representations=0,
        evaluation_replacement_reason="Correct within-domain joint-pattern allocation after metadata-only co-occurrence audit; no performance feedback",
    )
    if file_hash(old_db) != old_db_hash or public["old_ledger_after"] != before:
        raise AssertionError("Original ledger changed")
    immutable_json(root / "results/phase2c/METADATA_PREFLIGHT.json", public)
    print(
        json.dumps(
            dict(
                candidate_hash=candidate_hash,
                population=new["population"],
                domains=new["domains"],
                summary=new["summary"],
                evidence_support=new["evidence_support"],
                blockers=new["blockers"],
                attempts=0,
                training="DENY",
                evaluation="DENY",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
