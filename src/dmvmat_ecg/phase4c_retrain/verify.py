"""Verify new retrain references/checkpoints and preserve prior report bytes."""

import json

import numpy as np

from dmvmat_ecg.phase2.adaptation import state_hash
from dmvmat_ecg.phase4.safety import digest, new_json, sha
from dmvmat_ecg.phase4b.common import Target

from .data import PLACEBO, ROOT, RUN, setup
from .source import code_binding
from .target import load


def run():
    setup("target")
    t = Target()
    source_input = json.loads((RUN / "SOURCE_INPUTS.json").read_text())
    if source_input != dict(
        schema="retrain-reuses-ptb-only-features-v1",
        input_manifest_sha256=sha(PLACEBO / "SOURCE_INPUTS.json"),
        lambda_value=0.0,
        CPSC_inputs_read=False,
    ):
        raise PermissionError("Source-only input binding changed")
    assert json.loads((RUN / "SOURCES_COMPLETE.json").read_text())["status"] == "COMPLETE"
    assert json.loads((RUN / "TARGETS_COMPLETE.json").read_text())["status"] == "COMPLETE"
    assert json.loads((RUN / "SOURCE_REPRODUCIBILITY.json").read_text())["status"] == "PASS"
    assert json.loads((RUN / "TARGET_REPRODUCIBILITY.json").read_text())["status"] == "PASS"
    if json.loads((RUN / "SOURCE_CODE_BINDING.json").read_text()) != code_binding():
        raise PermissionError("Source code identity changed")
    for rel, value in json.loads((RUN / "TARGET_CODE_BINDING.json").read_text()).items():
        if sha(ROOT / rel) != value:
            raise PermissionError("Target code identity changed")
    source_rows = []
    for s in (15101, 15102, 15103):
        model, _, cp = load(s, t)
        r = json.loads((RUN / "source_models" / f"retrain_s{s}.json").read_text())
        if (
            r["CPSC_inputs_read"]
            or r["lambda_value"] != 0.0
            or r["MMD_gradient_used"]
            or not r["second_training_forward_retained"]
            or r["epochs"] != 50
            or r["second_domain"] != "PTB-XL folds1to8"
            or len(r["history"]) != 50
        ):
            raise PermissionError("Not a fixed source-only retrain source")
        if state_hash(model) != r["state_hash"] or digest(r["history"]) != r["history_sha256"]:
            raise PermissionError("Source state/history differs")
        with np.load(RUN / "source_models" / f"retrain_s{s}_fold9.npz", allow_pickle=False) as a:
            if (
                a["p"].shape != (2146, 2)
                or digest(a["p"].tolist()) != r["fold9_predictions_sha256"]
            ):
                raise PermissionError("Fold9 probabilities differ")
        source_rows.append(
            dict(
                source_seed=s,
                source_sha256=cp,
                fold9_macro_AUROC=r["fold9_macro_AUROC"],
                runtime_seconds=r["runtime_seconds"],
            )
        )
    fits = list((RUN / "fits").glob("*.json"))
    if (
        len(fits) != 27
        or len(list((RUN / "predictions").glob("*.npz"))) != 27
        or len(list((RUN / "source_models").glob("*.pt"))) != 3
    ):
        raise PermissionError("Incomplete or extra artifacts")
    expected = {
        (s, d, b)
        for s in (15101, 15102, 15103)
        for d in (16101, 16102, 16103)
        for b in (64, 128, 256)
    }
    found = set()
    runtime = 0.0
    for p in fits:
        r = json.loads(p.read_text())
        found.add((r["source_seed"], r["draw_seed"], r["budget"]))
        a = RUN / "predictions" / (r["file"] + ".npz")
        cp = next(v["source_sha256"] for v in source_rows if v["source_seed"] == r["source_seed"])
        if (
            sha(a) != r["predictions_sha256"]
            or r["source_sha256"] != cp
            or r["status"] != "COMPLETE"
            or len(r["history"]) != 200
            or r["updates"] != 200
            or r["mask_index"] != 127
        ):
            raise PermissionError("Fit recipe or identity differs")
        with np.load(a, allow_pickle=False) as values:
            if set(values.files) != {"A", "B"} or any(
                values[n].shape != (500, 2)
                or values[n].dtype != np.float32
                or not np.isfinite(values[n]).all()
                or ((values[n] < 0) | (values[n] > 1)).any()
                for n in values.files
            ):
                raise PermissionError("Validation-only arrays differ")
        runtime += r["runtime_seconds"]
    if found != expected:
        raise PermissionError("Episode/budget coverage differs")
    comparison = json.loads((RUN / "SOURCE_COMPARISON.json").read_text())
    aggregates = json.loads((ROOT / "docs/phase4/PHASE4B_RETRAIN_AGGREGATES.json").read_text())
    if (
        aggregates["source_comparison"] != comparison
        or len(comparison["sources"]) != 3
        or not all(comparison["source_cache_payloads_match_alignment"].values())
        or comparison["no_CPSC_data_read"] is not True
    ):
        raise PermissionError("Source comparison evidence differs")
    for row in comparison["sources"]:
        if row["original_source_sha256"] != t.sources[str(row["source_seed"])]["sha256"]:
            raise PermissionError("Original source comparison binding differs")
        if row["retrain_source_sha256"] != next(
            v["source_sha256"] for v in source_rows if v["source_seed"] == row["source_seed"]
        ):
            raise PermissionError("Retrained source comparison binding differs")
    checkpoints = list(RUN.rglob("*.pt")) + list(RUN.rglob("*.pth")) + list(RUN.rglob("*.ckpt"))
    if len(checkpoints) != 3 or any(p.parent != RUN / "source_models" for p in checkpoints):
        raise PermissionError("Unexpected target checkpoint")
    refs = json.loads((RUN / "FROZEN_VALIDATION_REFERENCES.json").read_text())
    if (
        refs["bundle_sha256"] != t.identity
        or refs["split_sha256"] != digest(t.split)
        or len(refs["references"]) != 27
    ):
        raise PermissionError("Future authentication references differ")
    for v in refs["references"]:
        if sha(RUN / "fits" / v["report"]) != v["report_sha256"]:
            raise PermissionError("Frozen reference report changed")
    import hashlib

    preserved = json.loads((RUN / "PRESERVATION_BEFORE.json").read_text())
    before = preserved["report"]
    for rel, value in preserved["files"].items():
        if sha(ROOT / rel) != value:
            raise PermissionError("Existing code/validation evidence changed:" + rel)
    if sha(ROOT / "docs/phase4/PLAIN_RETRAIN_PROTOCOL.md") != preserved["protocol_sha256"]:
        raise PermissionError("Prospective protocol changed")
    text = (ROOT / "docs/phase4/PHASE4B_RESULTS.md").read_bytes()
    if (
        hashlib.sha256(text[: before["bytes"]]).hexdigest() != before["sha256"]
        or text.count(b"## Plain-retrain control") != 1
    ):
        raise PermissionError("Prior results changed or retrain section duplicated")
    result = dict(
        schema="retrain-validation-integrity-v1",
        lambda_value=0.0,
        MMD_gradient_used=False,
        second_training_forward_retained=True,
        previous_code_and_validation_artifacts_unchanged=True,
        status="PASS",
        source_models=3,
        source_epochs=50,
        source_pretraining_CPSC_data_used=False,
        source_replay="PASS",
        target_fits=27,
        target_updates=200,
        target_replay="PASS",
        A_B_arrays_verified=27,
        target_checkpoints=0,
        checkpoint_hashes_states_and_fold9_predictions_verified=True,
        future_replay_references_sealed=27,
        previous_results_prefix_unchanged=True,
        test_access=False,
        confirmatory_replay_performed=False,
        source_rows=source_rows,
        target_fit_worker_wall_seconds=runtime,
        source_replay_seconds=json.loads((RUN / "SOURCE_REPRODUCIBILITY.json").read_text())[
            "repeat_seconds"
        ],
        target_replay_seconds=json.loads((RUN / "TARGET_REPRODUCIBILITY.json").read_text())[
            "repeat_seconds"
        ],
    )
    destination = ROOT / "docs/phase4/PHASE4B_RETRAIN_VALIDATION.json"
    if destination.exists():
        if json.loads(destination.read_text()) != result:
            raise PermissionError("Previous verification changed")
    else:
        new_json(destination, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
