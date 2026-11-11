"""One joint prediction lease; seal all candidates before any test metrics.

No model fitting or calibration is performed here. If inference fails, the
consumed access receipt remains and prevents automatic repeated evaluation.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from dmvmat_ecg.evaluation.bootstrap import interval, patient_bootstrap
from dmvmat_ecg.evaluation.metrics import complete
from dmvmat_ecg.evaluation.state import Study
from dmvmat_ecg.models.networks import build
from dmvmat_ecg.training.engine import dataset, predict
from dmvmat_ecg.utils.io import digest, sha, source_hash, write_json

root = Path(__file__).resolve().parents[1]
study = Study(root)
config = json.loads((root / "configs/study.json").read_text())
torch.set_num_threads(config["execution"]["threads"])
torch.use_deterministic_algorithms(True, warn_only=True)
runs = [r for r in study.runs() if r["status"] == "successful" and r["metadata"]["stage"] == "full"]
if len(runs) != 8:
    raise RuntimeError("Missing registered fits")
if any(r["status"] != "successful" for r in study.runs()):
    raise RuntimeError("Incomplete/failed attempt requires explicit review before test access")
manifest = study.get("dataset")
for name, checksum in manifest["cache_hashes"].items():
    if sha(root / "private/cache" / name) != checksum:
        raise PermissionError("Dataset/preprocessing cache lineage changed")
raw = root / "data/raw/ptbxl_v1_0_3"
for file, key in [
    ("ptbxl_database.csv", "metadata_sha256"),
    ("scp_statements.csv", "statements_sha256"),
    ("SHA256SUMS.txt", "publisher_manifest_sha256"),
]:
    if sha(raw / file) != manifest[key]:
        raise PermissionError("Official source identity changed")
environment = json.loads((root / "results/environment.json").read_text())
if sha(root / "uv.lock") != environment["uv_lock_sha256"]:
    raise PermissionError("Dependency lock changed")
artifacts = [
    root / "uv.lock",
    root / "pyproject.toml",
    root / "docs/EXPERIMENT_PROTOCOL.md",
    root / "private/cache/standardization.json",
    root / "data/manifests/memberships.json",
    root / "results/environment.json",
    root / "results/dataset_audit.json",
]
artifacts += [p for p in (root / "scripts").glob("*.py")]
for r in runs:
    folder = root / "private/runs" / r["id"]
    saved = json.loads((folder / "completion.json").read_text())
    if any(r["metadata"].get(k) != v for k, v in saved.items()):
        raise PermissionError("Completion receipt and durable ledger disagree")
    if (
        r["metadata"]["source_hash"] != source_hash(root)
        or r["metadata"]["configuration_hash"] != sha(root / "configs/study.json")
        or r["metadata"]["configuration"] != config
    ):
        raise PermissionError("Prediction-affecting code/config differs from fitted models")
    for name, key in [("best.pt", "best_hash"), ("final.pt", "final_hash")]:
        if sha(folder / name) != r["metadata"][key]:
            raise PermissionError("Checkpoint differs from completed training receipt")
        artifacts.append(folder / name)
    artifacts.extend([folder / "completion.json", folder / "history.json"])
lineage = dict(
    dataset=manifest,
    environment_sha256=sha(root / "results/environment.json"),
    configuration_sha256=sha(root / "configs/study.json"),
    source_sha256=source_hash(root),
    attempts=[
        dict(
            id=r["id"],
            status=r["status"],
            stage=r["metadata"]["stage"],
            model=r["metadata"]["model"],
            seed=r["metadata"]["seed"],
            source_sha256=r["metadata"]["source_hash"],
        )
        for r in study.runs()
    ],
    checkpoint_hashes={
        r["id"]: dict(best=r["metadata"]["best_hash"], final=r["metadata"]["final_hash"])
        for r in runs
    },
)
write_json(root / "results/development_lineage.json", lineage)
artifacts.append(root / "results/development_lineage.json")
if study.get("phase") != "development":
    raise PermissionError("Fresh lock and single evaluation required; do not rerun")
study.lock(artifacts)
write_json(root / "results/evaluation_lock.json", study.get("lock"))
output = root / "private/test_predictions"
output.mkdir(exist_ok=False)
sealed = {}
durations = {}
thresholds = {}
with study.evaluate_once("user-authorized-single-final-PTBXL-evaluation"):
    for run in runs:
        m = run["metadata"]
        ds, meta = dataset(study, m["model"], "test", m["seed"])
        kinds = ["best", "final"] if m["model"] == "xresnet101" else ["best"]
        for kind in kinds:
            name = m["model"] if kind == "best" else "xresnet101_final"
            model = build(m["model"], m["seed"]).cpu()
            checkpoint = root / "private/runs" / run["id"] / (kind + ".pt")
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            before = digest({k: sha(checkpoint) for k in ["checkpoint"]})
            scores, seconds = predict(model, ds, "cpu", config["batch_size"])
            if before != digest({k: sha(checkpoint) for k in ["checkpoint"]}):
                raise PermissionError("Checkpoint changed during inference")
            file = output / f"{name}-{m['seed']}.npz"
            np.savez(
                file, p=scores, y=meta["y"], patient_id=meta["patient_id"], ecg_id=meta["ecg_id"]
            )
            key = f"{name}:{m['seed']}"
            sealed[key] = sha(file)
            durations[key] = seconds
            thresholds[key] = m["thresholds"] if kind == "best" else m["final_thresholds"]
    write_json(
        root / "results/prediction_seal.json",
        dict(
            files=sealed,
            number=len(sealed),
            status="all_registered_predictions_sealed_before_metrics",
        ),
    )
    study.set("prediction_seal", sealed)
# No dataset access after lease. Reporting uses only immutable sealed files.
all_scores = {}
metrics = {}
for key, checksum in sealed.items():
    name, seed = key.split(":")
    file = output / f"{name}-{seed}.npz"
    if sha(file) != checksum:
        raise PermissionError("Prediction seal changed")
    item = np.load(file, allow_pickle=False)
    y, patients = item["y"], item["patient_id"]
    all_scores.setdefault(name, []).append(item["p"])
    metrics[key] = dict(
        validation_thresholds=complete(y, item["p"], thresholds[key]),
        fixed_half=complete(y, item["p"], [0.5] * 5),
        inference_seconds=durations[key],
    )
boot, receipt = patient_bootstrap(
    y,
    all_scores,
    patients,
    draws=config["evaluation"]["bootstrap_draws"],
    seed=config["evaluation"]["bootstrap_seed"],
    maximum=config["evaluation"]["maximum_draw_attempts"],
)
summary = {}
for name, scores in all_scores.items():
    entries = [metrics[f"{name}:{s}"]["validation_thresholds"] for s in config["seeds"]]
    summary[name] = {
        metric: float(np.mean([m[metric] for m in entries]))
        for metric in ("macro_auroc", "macro_ap", "micro_ap", "mean_brier", "mean_ece", "macro_f1")
    }
    summary[name].update(
        auroc_ci95=interval(boot[name]["auroc"]),
        ap_ci95=interval(boot[name]["ap"]),
        per_label_auroc=np.mean([m["per_label_auroc"] for m in entries], axis=0).tolist(),
        per_label_ap=np.mean([m["per_label_ap"] for m in entries], axis=0).tolist(),
        per_label_auroc_ci95=[interval(boot[name]["per_label_auroc"][:, j]) for j in range(5)],
        per_label_ap_ci95=[interval(boot[name]["per_label_ap"][:, j]) for j in range(5)],
    )
comparisons = {}
for a, b in config["evaluation"]["primary_comparisons"] + [["multiview_aug", "multiview"]]:
    primary = [a, b] in config["evaluation"]["primary_comparisons"]
    comparisons[f"{a}-minus-{b}"] = dict(
        delta_auroc=summary[a]["macro_auroc"] - summary[b]["macro_auroc"],
        delta_ap=summary[a]["macro_ap"] - summary[b]["macro_ap"],
        auroc_ci95=interval(boot[a]["auroc"] - boot[b]["auroc"]),
        auroc_multiplicity_ci=interval(
            boot[a]["auroc"] - boot[b]["auroc"], 0.975 if primary else 0.95
        ),
        ap_ci95=interval(boot[a]["ap"] - boot[b]["ap"]),
        primary=primary,
        per_label_auroc_ci95=[
            interval((boot[a]["per_label_auroc"] - boot[b]["per_label_auroc"])[:, j])
            for j in range(5)
        ],
        per_label_ap_ci95=[
            interval((boot[a]["per_label_ap"] - boot[b]["per_label_ap"])[:, j]) for j in range(5)
        ],
    )
write_json(
    root / "results/locked_metrics.json",
    dict(
        summary=summary,
        runs=metrics,
        comparisons=comparisons,
        bootstrap=receipt,
        prevalence=y.mean(0).tolist(),
        status="single locked test evaluation",
    ),
)
write_json(
    root / "results/evaluation_completion.json",
    dict(
        prediction_seal_hash=digest(sealed),
        attempts=len(study.runs()),
        access_receipts=[
            dict(id=r[0], kind=r[1], detail=json.loads(r[2]))
            for r in study.db.execute("SELECT * FROM access")
        ],
        completed_at=time.time(),
        recalibration=False,
        additional_training=False,
    ),
)
print(json.dumps(dict(status="EVALUATION_COMPLETE", summary=summary, comparisons=comparisons)))
