"""Seven additional paired source seeds, reusing the frozen numerical loops."""

import concurrent.futures
import io
import json
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np
import torch

from dmvmat_ecg.phase2.adaptation import state_hash
from dmvmat_ecg.phase3b.models import fresh_model
from dmvmat_ecg.phase3b.roles import Normalization
from dmvmat_ecg.phase4.safety import digest, new_json, sha
from dmvmat_ecg.phase4b.stage_lock import stage_lock
from dmvmat_ecg.phase4c_placebo.data import SourceInputs
from dmvmat_ecg.phase4c_placebo.source import train as placebo_train
from dmvmat_ecg.phase4c_retrain.source import train as plain_train

from .common import ARMS, PLACEBO, PLAIN, ROOT, RUN, SEEDS, SOURCE, Scope, bind, old_code_check


def binding():
    paths = [
        Path(__file__),
        Path(__file__).with_name("common.py"),
        ROOT / "docs/phase5/PHASE5_VALIDATION_PLAN.md",
        ROOT / "uv.lock",
    ]
    for folder in (PLAIN, PLACEBO):
        paths += [ROOT / p for p in json.loads((folder / "SOURCE_CODE_BINDING.json").read_text())]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))}


def inputs():
    # Reject unauthorized folds before any cached signal/label payload is opened.
    for role in ("source_train", "source_stop"):
        rows = json.loads((SOURCE / (role + "_records.json")).read_text())
        if any(f not in (range(1, 9) if role == "source_train" else (9,)) for f in rows["folds"]):
            raise PermissionError("Fold10 source read forbidden")
    old_code_check()
    return SourceInputs()


def descriptor(arm, seed):
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError("Unregistered source")
    if seed <= 15103:
        folder, stem = (PLAIN, "retrain") if arm == "plain" else (PLACEBO, "placebo")
    else:
        folder, stem = RUN, arm
    report = folder / "source_models" / f"{stem}_s{seed}.json"
    r = json.loads(report.read_text())
    cp = folder / r["checkpoint"]
    if not cp.resolve().is_relative_to(folder) or sha(cp) != r["sha256"]:
        raise PermissionError("Source checkpoint hash/path differs")
    if (
        r["source_seed"] != seed
        or r["lambda_value"] != (0.0 if arm == "plain" else 0.1)
        or r["epochs"] != 50
        or r["CPSC_inputs_read"]
        or len(r["history"]) != 50
    ):
        raise PermissionError("Source recipe differs")
    saved = torch.load(cp, map_location="cpu", weights_only=True)
    model = fresh_model(seed)
    model.load_state_dict(saved["state"], strict=True)
    model.eval()
    if state_hash(model) != r["state_hash"] or digest(r["history"]) != r["history_sha256"]:
        raise PermissionError("Source state/history differs")
    norm = Normalization(**saved["normalization"])
    return (
        model,
        norm,
        r,
        dict(
            arm=arm,
            source_seed=seed,
            checkpoint=str(cp.relative_to(ROOT)),
            sha256=r["sha256"],
            state_hash=r["state_hash"],
            fold9_macro_AUROC=r["fold9_macro_AUROC"],
            fold9_AUROC=r["fold9_AUROC"],
            selected_epoch=r["selected_epoch"],
            normalization=norm.__dict__,
        ),
    )


def one(job):
    arm, seed = job
    Scope("source").install()
    torch.set_num_threads(4)
    data = inputs()
    frozen = binding()
    bind(RUN / "SOURCE_CODE_BINDING.json", frozen)
    source_id = digest(
        dict(seed=seed, arm=arm, code=frozen, input_sha256=sha(PLACEBO / "SOURCE_INPUTS.json"))
    )
    folder = RUN / "source_models"
    folder.mkdir(mode=0o700, exist_ok=True)
    path = folder / f"{arm}_s{seed}.json"
    cp = folder / f"{arm}_s{seed}.pt"
    pred = folder / f"{arm}_s{seed}_fold9.npz"
    if path.exists():
        _, _, r, row = descriptor(arm, seed)
        if r["identity"] != source_id or sha(pred) != r["fold9_file_sha256"]:
            raise PermissionError("Source resume binding differs")
        return row
    started = folder / f"{arm}_s{seed}.started.json"
    if cp.exists() or pred.exists() or started.exists():
        raise FileExistsError("Partial source fit preserved; no silent retry")
    new_json(started, dict(seed=seed, arm=arm, identity=source_id, started_unix=time.time()))
    state, p, r = (
        plain_train(seed, data, data.norm)
        if arm == "plain"
        else placebo_train(seed, data, data.norm)
    )
    if binding() != frozen:
        raise PermissionError("Code changed during source fit")
    stream = io.BytesIO()
    torch.save(
        dict(
            state=state,
            normalization=data.norm.__dict__,
            state_hash=r["state_hash"],
            history=r["history"],
            selected_epoch=r["selected_epoch"],
            identity=source_id,
        ),
        stream,
    )
    fd = os.open(cp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as out:
        out.write(stream.getvalue())
        out.flush()
        os.fsync(out.fileno())
    fd = os.open(pred, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as out:
        np.savez_compressed(out, p=p)
        out.flush()
        os.fsync(out.fileno())
    r.update(
        identity=source_id,
        checkpoint=str(cp.relative_to(RUN)),
        sha256=sha(cp),
        fold9_file_sha256=sha(pred),
    )
    new_json(path, r)
    return descriptor(arm, seed)[3]


def run():
    Scope("source").install()
    torch.set_num_threads(4)
    inputs()
    bind(RUN / "SOURCE_CODE_BINDING.json", binding())
    with stage_lock(RUN, "source_grid"):
        for s in SEEDS[:3]:
            for arm in ARMS:
                descriptor(arm, s)
        jobs = [(arm, s) for s in SEEDS[3:] for arm in ARMS]
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=3, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            future = {pool.submit(one, j): j for j in jobs}
            for f in concurrent.futures.as_completed(future):
                row = f.result()
                print(
                    "PHASE5_SOURCE_COMPLETE",
                    row["arm"],
                    row["source_seed"],
                    row["fold9_macro_AUROC"],
                    flush=True,
                )
        rows = [descriptor(a, s)[3] for s in SEEDS for a in ARMS]
        bind(
            RUN / "SOURCES_COMPLETE.json",
            dict(
                status="COMPLETE",
                sources=rows,
                source_models=20,
                new_source_models=14,
                target_pretraining_inputs=0,
            ),
        )
        print("PHASE5_SOURCES_COMPLETE 20", flush=True)


if __name__ == "__main__":
    run()
