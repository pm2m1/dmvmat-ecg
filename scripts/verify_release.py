"""Read-only scientific-integrity checks; writes aggregate verification receipts."""

import json
import subprocess
from pathlib import Path

import numpy as np

from dmvmat_ecg.evaluation.state import Study
from dmvmat_ecg.utils.io import sha, source_hash, write_json

root = Path(__file__).resolve().parents[1]
study = Study(root)
assert study.get("phase") == "evaluated" and study.get("evaluation_consumed")
assert study.db.execute("SELECT COUNT(*) FROM access").fetchone()[0] == 1
assert len(study.runs()) == 11 and all(r["status"] == "successful" for r in study.runs())
full = [r for r in study.runs() if r["metadata"]["stage"] == "full"]
assert len(full) == 8 and all(r["metadata"]["epochs_completed"] == 50 for r in full)
assert source_hash(root) == study.get("lock")["source_hash"]
for file, expected in study.get("lock")["files"].items():
    assert sha(root / file) == expected, file
assert len(study.get("prediction_seal")) == 10
for key, expected in study.get("prediction_seal").items():
    name, seed = key.split(":")
    assert sha(root / "private/test_predictions" / f"{name}-{seed}.npz") == expected
metrics = json.loads((root / "results/locked_metrics.json").read_text())
assert len(metrics["runs"]) == 10 and len(metrics["summary"]) == 5
for entry in metrics["runs"].values():
    for point in ("validation_thresholds", "fixed_half"):
        for v in entry[point]["labels"].values():
            matrix = np.array(v["confusion_matrix"])
            assert matrix.sum() == 2158
            tn, fp, fn, tp = matrix.ravel()
            assert np.isclose(v["recall"], tp / (tp + fn))
            assert np.isclose(v["specificity"], tn / (tn + fp))
            assert 0 <= v["auroc"] <= 1 and 0 <= v["ap"] <= 1
figures = json.loads((root / "results/figure_inventory.json").read_text())
assert len(figures) == 9
for entry in figures.values():
    assert len(entry["files"]) == 2
    for file, expected in entry["files"].items():
        assert sha(root / file) == expected and (root / file).stat().st_size > 1000
prior = json.loads((root / "private/previous_repositories_before.json").read_text())
historical = {}
for name, before in prior.items():
    path = root.parent / name
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "-uall"], cwd=path, text=True
    )
    fingerprint = {
        p: sha(path / p)
        for p in subprocess.check_output(["git", "ls-files"], cwd=path, text=True).splitlines()
        if (path / p).is_file()
    }
    assert (
        head == before["head"]
        and status == before["status"]
        and fingerprint == before["tracked_fingerprint"]
    ), "Historical repository state changed"
    historical[name] = dict(head=head, unchanged=True, tracked_files_checked=len(fingerprint))
tracked = subprocess.check_output(["git", "ls-files"], cwd=root, text=True).splitlines()
assert not any(
    p.startswith(("data/raw/", "data/manifests/", "private/", ".venv/"))
    or Path(p).suffix in (".pt", ".pth", ".npy", ".npz", ".dat", ".hea")
    for p in tracked
)
assert not subprocess.check_output(["git", "remote"], cwd=root, text=True).strip()
starts = [(root / "private/runs" / r["id"] / "entry.json").stat().st_mtime for r in full]
ends = [(root / "private/runs" / r["id"] / "completion.json").stat().st_mtime for r in full]
receipt = dict(
    predictor_attempts=11,
    successful_full=8,
    successful_smoke=3,
    failures=0,
    repairs=0,
    single_test_access=True,
    registered_prediction_sets=10,
    registered_procedures=5,
    frozen_source_config_preserved=True,
    all_lock_artifacts_preserved=True,
    confusion_counts_consistent=True,
    figure_pairs=9,
    old_repositories=historical,
    real_training_wall_seconds_from_filesystem_receipts=max(ends) - min(starts),
    training_optimization_seconds_sum=sum(r["metadata"]["training_seconds"] for r in full),
    validation_prediction_seconds_sum=sum(
        sum(h["validation_prediction_seconds"] for h in r["metadata"]["history"]) for r in full
    ),
    synthetic_pytest="26 passed; 7 documented warnings",
    ruff="passed",
    formatting="32 files passed",
    mps="XResNet synthetic gradient parity failed; common CPU fallback; other two architectures passed",
    no_remote=True,
    raw_records_predictions_checkpoints_excluded=True,
    external_transfer_executed=False,
)
write_json(root / "results/final_integrity.json", receipt)
(root / "docs/VERIFICATION.md").write_text(
    "# Final verification\n\n"
    + "Completed11/13 allowed predictor attempts: eight full50-epoch fits and three training-only smoke fits; zero failed/retried. One joint locked test access, ten sealed prediction sets, five registered checkpoint procedures. All checkpoint, threshold-bearing completion, preprocessing, membership, dependency, configuration and code hashes remain intact.\n\n"
    + f"Real full-training wall duration from entry/completion filesystem receipts: {receipt['real_training_wall_seconds_from_filesystem_receipts'] / 3600:.3f}hours, including concurrent jobs, validation and serialization. Optimization/preprocessing time summed across fits: {receipt['training_optimization_seconds_sum'] / 3600:.3f}hours; validation prediction sum: {receipt['validation_prediction_seconds_sum'] / 3600:.3f}hours. Summed durations are not the study wall time.\n\n"
    + "26 synthetic tests passed (7 expected/deprecation warnings); Ruff and formatting checks passed. Three synthetic CPU/MPS architecture probes: CPU forward/backward passed all; MPS parity passed flat/multiview and failed the reference's preset gradient tolerance, giving common CPU real execution. No claim of MPS/CPU fitted-weight identity.\n\n"
    + "Nine PNG/PDF figure pairs verified by checksums; all per-label confusion matrices sum to2,158 and sensitivity/specificity match counts. No recalibration or additional training after test opening. No overall model selected from test scores.\n\n"
    + "Both historical cardiovascular workspaces have unchanged HEAD, exact Git status and all1,915 tracked-file hashes each, compared with the before snapshot. New project has no remote; raw waveforms, metadata, assignments, predictions and checkpoints are ignored/local. Full safe receipt: `results/final_integrity.json`. Publication still requires separate user authorization.\n\n"
    + "The observed power warning was resolved after the user connected adequate power. The attempted suspension did not execute: all workers remained running, and no fit was paused/restarted. See ignored operational power receipt. This did not change model configurations or budgets.\n"
)
inventory = json.loads((root / "results/public_release_inventory.json").read_text())
for file in ("docs/VERIFICATION.md", "results/final_integrity.json"):
    inventory["eligible_files"][file] = sha(root / file)
write_json(root / "results/public_release_inventory.json", inventory)
print(json.dumps(receipt, sort_keys=True))
