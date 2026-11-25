"""Verify completed Phase 4B artifacts; never fit or load a test provider."""

import json
from collections import Counter

import numpy as np

from dmvmat_ecg.phase4.safety import digest, new_json, sha

from .common import ROOT, RUN, load_source, setup
from .pipeline import verify_code
from .train_alignment import tag


def verify():
    target = setup()
    verify_code()
    stages = {}
    for name in (
        "ORACLE64_COMPLETE",
        "BASELINES_COMPLETE",
        "ALIGNMENT_GRID_COMPLETE",
        "ALIGNED_TARGETS_COMPLETE",
    ):
        entry = json.loads((RUN / (name + ".json")).read_text())
        if entry["status"] != "COMPLETE":
            raise PermissionError("Incomplete Phase 4B stage: " + name)
        stages[name] = entry
    result = json.loads((ROOT / "docs/phase4/PHASE4B_AGGREGATES.json").read_text())
    if result["schema"] != "phase4b-validation-only-results-v1":
        raise PermissionError("Unrecognized final results")
    reports = []
    counts = Counter()
    oracle = set()
    originals = 0
    for path in sorted((RUN / "fits").glob("*.json")):
        r = json.loads(path.read_text())
        prediction = RUN / "predictions" / (path.stem + ".npz")
        source_entry = (
            target.sources[str(r["source_seed"])]
            if r["aligned"] is None
            else json.loads(
                (RUN / "aligned_sources" / f"{r['aligned']}_{r['source_seed']}.json").read_text()
            )
        )
        if (
            r["status"] != "COMPLETE"
            or r["file"] != path.stem
            or sha(prediction) != r["predictions_sha256"]
            or r["source_sha256"] != source_entry["sha256"]
            or r.get("source_state_hash", source_entry["state_hash"]) != source_entry["state_hash"]
            or r["updates"] not in (0, 200)
            or len(r["history"]) != r["updates"]
            or not np.isfinite(r["history"]).all()
            or r["U_label_accesses"] != 0
        ):
            raise PermissionError("Fit integrity/recipe verification failed")
        with np.load(prediction, allow_pickle=False) as arrays:
            if set(arrays.files) != {"A", "B"}:
                raise PermissionError("Only validation A/B arrays may be retained")
            for role in ("A", "B"):
                p = arrays[role]
                if (
                    p.shape != (500, 2)
                    or not np.isfinite(p).all()
                    or (p < 0).any()
                    or (p > 1).any()
                ):
                    raise PermissionError("Invalid validation probability array")
        if r["method"] == "oracle":
            oracle.add((r["source_seed"], r["draw_seed"], r["mask_index"]))
            if r["budget"] != 64 or r["aligned"] is not None:
                raise PermissionError("Off-protocol oracle")
        if r["method"] == "full" and r["budget"] == 128 and r["aligned"] is None:
            for reference in target.references[f"{r['source_seed']}/{r['draw_seed']}"].values():
                if set(reference["mask"].values()) == {"ADAPT"}:
                    if any(
                        r[k] != reference[k] for k in ("state_hash", "drift", "final_training_loss")
                    ):
                        raise PermissionError("Original Experiment 4 reference differs")
                    originals += 1
        counts[(r["method"], r["aligned"] or "original")] += 1
        reports.append(r)
    expected_oracle = {
        (s, d, m) for s in (15101, 15102, 15103) for d in (16101, 16102, 16103) for m in range(128)
    }
    if oracle != expected_oracle or originals != 9 or len(reports) != 1377:
        raise PermissionError("Incomplete or unexpected fit collection")
    if len(list((RUN / "predictions").glob("*.npz"))) != len(reports):
        raise PermissionError("Orphaned/extra validation arrays")
    sources = []
    for lam in (0.1, 1.0):
        for s in (15101, 15102, 15103):
            model, _, _, _ = load_source(target, s, tag(lam))
            del model
            entry = json.loads((RUN / "aligned_sources" / f"{tag(lam)}_{s}.json").read_text())
            with np.load(
                RUN / "aligned_sources" / f"{tag(lam)}_{s}_fold9.npz", allow_pickle=False
            ) as arrays:
                if (
                    arrays["p"].shape != (2146, 2)
                    or digest(arrays["p"].tolist()) != entry["fold9_predictions_sha256"]
                ):
                    raise PermissionError("Source validation probability identity differs")
            sources.append(entry)
    checkpoints = list(RUN.rglob("*.pt")) + list(RUN.rglob("*.pth")) + list(RUN.rglob("*.ckpt"))
    if len(checkpoints) != 6 or any(p.parent != RUN / "aligned_sources" for p in checkpoints):
        raise PermissionError("Unexpected non-source checkpoint")
    gates = {}
    for name in (
        "ORACLE64_REPRODUCIBILITY",
        "BASELINE_REPRODUCIBILITY",
        "ALIGNMENT_REPRODUCIBILITY",
        "ALIGNED_FULL_REPRODUCIBILITY",
        "ALIGNED_SOURCE_ONLY_REPRODUCIBILITY",
        "ALIGNED_SELECTOR_REPRODUCIBILITY",
    ):
        entry = json.loads((RUN / (name + ".json")).read_text())
        if entry["status"] != "PASS":
            raise PermissionError("Unsuccessful reproducibility gate")
        gates[name] = entry
    repeated_target_seconds = gates["ORACLE64_REPRODUCIBILITY"]["repeat_seconds"] + sum(
        row["repeat_seconds"] for row in gates["BASELINE_REPRODUCIBILITY"]["methods"]
    )
    repeated_target_seconds += sum(
        gates[name]["repeat_seconds"]
        for name in (
            "ALIGNED_FULL_REPRODUCIBILITY",
            "ALIGNED_SOURCE_ONLY_REPRODUCIBILITY",
            "ALIGNED_SELECTOR_REPRODUCIBILITY",
        )
    )
    controls = json.loads((RUN / "RECONSTRUCTED_ORIGINAL_CONTROLS.json").read_text())
    if (
        controls["status"] != "PASS"
        or not controls["all_nine_B128_selector_and_random_masks_match_retained_records"]
        or controls["B_used"]
        or Counter(r["budget"] for r in controls["rows"]) != {64: 9, 128: 9}
    ):
        raise PermissionError("Original-control replay was not validated")
    audit = dict(
        schema="phase4b-completed-artifact-verification-v1",
        status="PASS",
        retained_target_and_zero_update_evaluations=len(reports),
        retained_target_gradient_fits=sum(r["updates"] == 200 for r in reports),
        retained_zero_update_evaluations=sum(r["updates"] == 0 for r in reports),
        validation_prediction_files=len(reports),
        all_validation_predictions_hash_and_shape_verified=True,
        oracle64_unique_fits=len(oracle),
        original_full128_references_matched=originals,
        new_aligned_source_checkpoints=6,
        new_target_checkpoints=0,
        checkpoint_state_file_and_fold9_prediction_hashes_verified=True,
        planned_verified_target_or_zero_update_repeats=10,
        planned_verified_full_source_repeats=1,
        failed_numerical_reproducibility_checks=0,
        original_B128_control_replays_matched=9,
        B64_selector_and_random_control_reconstructions=9,
        original_B64_control_receipts_available=False,
        control_reconstruction_used_A_or_B=False,
        execution_incident=json.loads((RUN / "ORCHESTRATION_CORRECTION.json").read_text()),
        computation_count_note="Retained unique fits and planned verified replays exclude redundant controller computations. The execution incident records three redundant full source-candidate computations, two stopped partial source computations, and a refused duplicate baseline invocation. Its source-only replay and AdaBN computation add at least two zero-update evaluations without new retained predictions. No duplicate result was selected by score or allowed to overwrite a prior artifact.",
        fit_counts=[
            dict(method=k[0], initialization=k[1], count=v) for k, v in sorted(counts.items())
        ],
        stage_recorded_wall_seconds={
            k: v["wall_seconds"] for k, v in stages.items() if "wall_seconds" in v
        },
        sum_retained_target_evaluation_wall_seconds=sum(r["runtime_seconds"] for r in reports),
        sum_repeated_target_evaluation_wall_seconds=repeated_target_seconds,
        sum_retained_source_fit_wall_seconds=sum(r["runtime_seconds"] for r in sources),
        repeated_source_fit_wall_seconds=gates["ALIGNMENT_REPRODUCIBILITY"][
            "repeated_source_seconds"
        ],
        runtime_note="Sums are worker wall-seconds with concurrent execution, not elapsed study wall time or CPU time; selector evidence extraction is not included in fit timers.",
        reproducibility=gates,
        validation_only=True,
        no_test_provider_loaded=True,
        U_labels_used=False,
    )
    destination = ROOT / "docs/phase4/PHASE4B_EXECUTION_AUDIT.json"
    if destination.exists():
        if json.loads(destination.read_text()) != audit:
            raise PermissionError("Prior verification audit differs")
    else:
        new_json(destination, audit)
    print(json.dumps({k: v for k, v in audit.items() if k != "reproducibility"}, indent=2))
    return audit


if __name__ == "__main__":
    verify()
