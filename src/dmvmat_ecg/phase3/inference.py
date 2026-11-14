"""Historical full-10s CNN inference only; no training or fitting interfaces."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from dmvmat_ecg.models.networks import build
from dmvmat_ecg.preprocessing.signals import normalize, views

from .download import sha
from .splits import require_feas, require_fold9


def source_metadata_identity(root):
    root = Path(root)
    audit = json.loads((root / "results/dataset_audit.json").read_text())
    if audit["release"] != "1.0.3":
        raise PermissionError("Source publisher release differs")
    expected = {
        "ptbxl_database.csv": audit["metadata_sha256"],
        "scp_statements.csv": audit["statements_sha256"],
    }
    for name, checksum in expected.items():
        if sha(root / "data/raw/ptbxl_v1_0_3" / name) != checksum:
            raise PermissionError("Source metadata changed: " + name)
    return expected


def reference_models(root):
    root = Path(root)
    metadata_hashes = source_metadata_identity(root)
    found = []
    stats_path = root / "private/cache/standardization.json"
    audit = json.loads((root / "results/dataset_audit.json").read_text())
    if sha(stats_path) != audit["cache_hashes"]["standardization.json"]:
        raise PermissionError("Historical training-only normalization hash changed")
    stats = json.loads(stats_path.read_text())
    if stats["fit_role"] != "train" or stats["fit_rows"] != 17084:
        raise PermissionError("Unexpected source normalization lineage")
    for path in sorted((root / "private/runs").glob("*/completion.json")):
        entry = json.loads(path.read_text())
        if entry["stage"] == "full" and entry["model"] in {"multiview", "flat_cnn"}:
            if entry["seed"] not in {101, 202} or entry["standardization_sha256"] != sha(
                stats_path
            ):
                raise PermissionError("Unregistered historical checkpoint/normalization")
            checkpoint = path.parent / "best.pt"
            if sha(checkpoint) != entry["best_hash"]:
                raise PermissionError("Historical validation-selected checkpoint hash differs")
            found.append(
                {
                    "model": entry["model"],
                    "seed": entry["seed"],
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": entry["best_hash"],
                    "best_epoch": entry["best_epoch"],
                    "parameters": entry["parameters"],
                    "window": 1000,
                    "aggregation": "single full10s recording",
                    "source_training_folds": list(range(1, 9)),
                    "checkpoint_selection_fold": 9,
                    "source_metadata_sha256": metadata_hashes,
                }
            )
    if len(found) != 4 or {(x["model"], x["seed"]) for x in found} != {
        (m, s) for m in ["multiview", "flat_cnn"] for s in [101, 202]
    }:
        raise PermissionError("All four historical best checkpoints are required; no substitution")
    return found, stats


def fold9(root):
    root = Path(root)
    audit = json.loads((root / "results/dataset_audit.json").read_text())
    folder = root / "private/cache"
    for name in ["validation_x.npy", "validation_meta.npz"]:
        if sha(folder / name) != audit["cache_hashes"][name]:
            raise PermissionError("Source validation cache changed")
    with np.load(folder / "validation_meta.npz", allow_pickle=False) as f:
        require_fold9(f["fold"])
        meta = {k: f[k].copy() for k in f.files}
    x = np.load(folder / "validation_x.npy", mmap_mode="r", allow_pickle=False)
    if x.shape != (2146, 12, 1000):
        raise PermissionError("Unexpected source fold9 tensor shape")
    return x, meta


def infer(model_reference, x, stats, *, ids=None, manifest=None, representations=False):
    if manifest is not None:
        if ids is None:
            raise PermissionError("Target inference requires FEAS identities")
        for name in ids:
            require_feas(name, manifest)
    torch.set_num_threads(4)
    checkpoint = Path(model_reference["checkpoint"])
    if sha(checkpoint) != model_reference["checkpoint_sha256"]:
        raise PermissionError("Frozen model changed before inference")
    model = build(model_reference["model"], model_reference["seed"]).cpu()
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    model.eval()
    predictions, extracted = [], {}
    begin = time.perf_counter()
    with torch.no_grad():
        for offset in range(0, len(x), 128):
            wave = normalize(np.asarray(x[offset : offset + 128]), stats)
            inputs = {"wave": torch.tensor(wave)}
            if model_reference["model"] == "multiview":
                _, rhythm, spectrum = views(wave)
                inputs.update(rhythm=torch.tensor(rhythm), spectrum=torch.tensor(spectrum))
            if representations:
                if model_reference["model"] != "multiview":
                    raise ValueError("Flat model does not have five clinical-view modules")
                values = model.representations(**inputs)
                logits = model.prediction_head(values["fusion_2"])
                for name, value in values.items():
                    extracted.setdefault(name, []).append(value.numpy())
            else:
                logits = model(**inputs)
            predictions.append(logits.sigmoid().numpy()[:, [3, 2]])
    p = np.concatenate(predictions)
    if p.shape != (len(x), 2) or not np.isfinite(p).all():
        raise ValueError("Invalid frozen predictions")
    if sha(checkpoint) != model_reference["checkpoint_sha256"]:
        raise PermissionError("Inference must not change historical checkpoint")
    return p, {n: np.concatenate(v) for n, v in extracted.items()}, time.perf_counter() - begin
