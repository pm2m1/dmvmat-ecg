"""Matched MK-MMD source/source control; no CPSC loads in pretraining."""

import concurrent.futures
import copy
import io
import json
import math
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from dmvmat_ecg.phase2.adaptation import masked_bce, state_hash
from dmvmat_ecg.phase3b.models import fresh_model
from dmvmat_ecg.phase3b.protocol import adopted_settings, seed
from dmvmat_ecg.phase4.safety import digest, new_json, sha
from dmvmat_ecg.phase4b.alignment import alignment_loss
from dmvmat_ecg.phase4b.stage_lock import stage_lock
from dmvmat_ecg.phase4b.train_alignment import TargetStream as IndexStream

from .data import ROOT, RUN, SourceInputs, setup


def train(s, features, norm, lambda_value=0.0):
    """Numerical loop: only second stream's data role differs from Phase4B."""
    if lambda_value not in (0.0, 0.1):
        raise ValueError("Only fixed lambda0 and synthetic parity lambda0.1 permitted")
    model = fresh_model(s).cpu().float()
    y = torch.tensor(features.source.y["source_train"], dtype=torch.float32)
    observed = torch.ones(y.shape, dtype=torch.bool)
    vy = features.source.y["source_stop"]
    vv = features.batch("source_stop", np.arange(len(vy)))
    settings = adopted_settings()
    optimizer = torch.optim.AdamW(model.parameters(), **settings["source_optimizer"])
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        epochs=50,
        steps_per_epoch=math.ceil(len(y) / 128),
        **settings["source_schedule"],
    )
    best = None
    best_state = None
    history = []
    started = time.monotonic()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(s)
        source_generator = torch.Generator().manual_seed(s)
        stream = IndexStream(len(y), torch.Generator().manual_seed(seed([20261126, s])))
        for epoch in range(1, 51):
            order = torch.randperm(len(y), generator=source_generator)
            model.train()
            losses, bces, mmds = [], [], []
            for start in range(0, len(y), 128):
                ix = order[start : start + 128]
                optimizer.zero_grad(set_to_none=True)
                source_reps = model.representations(**features.batch("source_train", ix.numpy()))
                logits = model.prediction_head(source_reps["fusion_2"])
                bce = masked_bce(logits, y[ix], observed[ix])
                second_reps = model.representations(
                    **features.batch("source_train", stream.take(32))
                )
                if lambda_value == 0.0:
                    with torch.no_grad():
                        mmd = alignment_loss(source_reps, second_reps)
                    loss = bce
                else:
                    mmd = alignment_loss(source_reps, second_reps)
                    loss = bce + lambda_value * mmd
                if not torch.isfinite(mmd):
                    raise FloatingPointError("Nonfinite diagnostic MMD")
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite retrain source loss")
                loss.backward()
                if any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in model.parameters()
                ):
                    raise FloatingPointError("Nonfinite retrain source gradient")
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in model.parameters()):
                    raise FloatingPointError("Nonfinite retrain source parameter")
                schedule.step()
                losses.append(float(loss.detach()))
                bces.append(float(bce.detach()))
                mmds.append(float(mmd.detach()))
            model.eval()
            with torch.no_grad():
                logits = model(**vv)
                probabilities = logits.sigmoid().numpy()
                val_bce = float(
                    masked_bce(
                        logits,
                        torch.tensor(vy, dtype=torch.float32),
                        torch.ones(vy.shape, dtype=torch.bool),
                    )
                )
            ap = float(average_precision_score(vy, probabilities, average=None).mean())
            if (
                best is None
                or ap > best[0] + 1e-12
                or (abs(ap - best[0]) <= 1e-12 and val_bce < best[1] - 1e-12)
            ):
                best = (ap, val_bce, epoch)
                best_state = copy.deepcopy(model.state_dict())
            history.append(
                dict(
                    epoch=epoch,
                    loss=float(np.mean(losses)),
                    BCE=float(np.mean(bces)),
                    MMD=float(np.mean(mmds)),
                    fold9_macro_AP=ap,
                    fold9_BCE=val_bce,
                )
            )
            print(
                f"RETRAIN_SOURCE seed={s} epoch={epoch}/50 seconds={time.monotonic() - started:.1f}",
                flush=True,
            )
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        retained = model(**vv).sigmoid().numpy()
    aps = average_precision_score(vy, retained, average=None)
    aucs = roc_auc_score(vy, retained, average=None)
    return (
        best_state,
        retained,
        dict(
            source_seed=s,
            lambda_value=lambda_value,
            MMD_gradient_used=lambda_value != 0.0,
            second_training_forward_retained=True,
            selected_epoch=best[2],
            fold9_AP=aps.tolist(),
            fold9_AUROC=aucs.tolist(),
            fold9_macro_AUROC=float(aucs.mean()),
            state_hash=state_hash(model),
            history=history,
            history_sha256=digest(history),
            fold9_predictions_sha256=digest(retained.tolist()),
            normalization=norm.__dict__,
            runtime_seconds=time.monotonic() - started,
            source_records=len(y),
            first_batch=128,
            second_batch=32,
            second_domain="PTB-XL folds1to8",
            epochs=50,
            CPSC_inputs_read=False,
        ),
    )


def code_binding():
    files = [
        Path(__file__),
        Path(__file__).with_name("data.py"),
        ROOT / "src/dmvmat_ecg/phase4b/alignment.py",
        ROOT / "src/dmvmat_ecg/phase4b/train_alignment.py",
        ROOT / "src/dmvmat_ecg/phase4b/stage_lock.py",
        ROOT / "src/dmvmat_ecg/phase4b/source_data.py",
        ROOT / "src/dmvmat_ecg/phase3b/models.py",
        ROOT / "src/dmvmat_ecg/phase3b/protocol.py",
        ROOT / "src/dmvmat_ecg/phase3b/roles.py",
        ROOT / "src/dmvmat_ecg/preprocessing/signals.py",
        ROOT / "src/dmvmat_ecg/phase2/adaptation.py",
        ROOT / "src/dmvmat_ecg/phase4c_placebo/data.py",
        ROOT / "src/dmvmat_ecg/phase4c_placebo/source.py",
        ROOT / "uv.lock",
    ]
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def source_fit(s, retain=True):
    torch.set_num_threads(4)
    setup("source")
    data = SourceInputs()
    norm = data.norm
    identity = digest(
        dict(
            seed=s,
            inputs=sha(RUN / "SOURCE_INPUTS.json"),
            code=code_binding(),
            kind="source-source-retrain-lambda0.0",
        )
    )
    folder = RUN / "source_models"
    report_path = folder / f"retrain_s{s}.json"
    checkpoint = folder / f"retrain_s{s}.pt"
    if retain and report_path.exists():
        r = json.loads(report_path.read_text())
        if r["identity"] != identity or sha(checkpoint) != r["sha256"]:
            raise PermissionError("Placebo source resume identity mismatch")
        return r
    if retain and checkpoint.exists():
        raise FileExistsError("Partial retrain checkpoint is preserved")
    state, p, r = train(s, data, norm)
    r["identity"] = identity
    if retain:
        folder.mkdir(mode=0o700, exist_ok=True)
        payload = io.BytesIO()
        torch.save(
            dict(
                state=state,
                normalization=norm.__dict__,
                state_hash=r["state_hash"],
                history=r["history"],
                selected_epoch=r["selected_epoch"],
                identity=identity,
            ),
            payload,
        )
        fd = os.open(checkpoint, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as out:
            out.write(payload.getvalue())
            out.flush()
            os.fsync(out.fileno())
        with (folder / f"retrain_s{s}_fold9.npz").open("xb") as out:
            np.savez_compressed(out, p=p)
        r.update(checkpoint=str(checkpoint.relative_to(RUN)), sha256=sha(checkpoint))
        new_json(report_path, r)
    return r


def run():
    setup("source")
    with stage_lock(RUN, "source_grid"):
        expected = code_binding()
        binding = RUN / "SOURCE_CODE_BINDING.json"
        if binding.exists():
            if json.loads(binding.read_text()) != expected:
                raise PermissionError("Source code changed after source fitting began")
        else:
            new_json(binding, expected)
        gate = RUN / "SOURCE_REPRODUCIBILITY.json"
        if not gate.exists():
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=2, mp_context=multiprocessing.get_context("spawn")
            ) as pool:
                a = pool.submit(source_fit, 15101, True)
                b = pool.submit(source_fit, 15101, False)
                retained, repeat = a.result(), b.result()
            for key in (
                "state_hash",
                "history_sha256",
                "fold9_predictions_sha256",
                "selected_epoch",
            ):
                if retained[key] != repeat[key]:
                    raise RuntimeError("STOP_RETRAIN_SOURCE_REPLAY_MISMATCH:" + key)
            new_json(
                gate,
                dict(
                    status="PASS",
                    state_history_and_fold9_bitwise_identical=True,
                    repeat_seconds=repeat["runtime_seconds"],
                ),
            )
            print("RETRAIN_SOURCE_REPLAY_PASS", flush=True)
        elif json.loads(gate.read_text())["status"] != "PASS":
            raise PermissionError("Prior source replay was not successful")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=2, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = [pool.submit(source_fit, s, True) for s in (15101, 15102, 15103)]
            rows = [f.result() for f in futures]
        destination = RUN / "SOURCES_COMPLETE.json"
        if not destination.exists():
            new_json(
                destination,
                dict(
                    status="COMPLETE",
                    sources=3,
                    CPSC_pretraining_inputs_used=False,
                    summaries=[
                        {
                            k: r[k]
                            for k in (
                                "source_seed",
                                "fold9_macro_AUROC",
                                "selected_epoch",
                                "sha256",
                            )
                        }
                        for r in rows
                    ],
                ),
            )
        print("THREE_RETRAIN_SOURCES_COMPLETE", flush=True)


if __name__ == "__main__":
    run()
