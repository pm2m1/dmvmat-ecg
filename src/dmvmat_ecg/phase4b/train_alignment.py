"""Fifty-epoch source-only-label MK-MMD training; only source checkpoints saved."""

import copy
import io
import json
import math
import os
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from dmvmat_ecg.phase2.adaptation import masked_bce, state_hash
from dmvmat_ecg.phase3b.models import fresh_model
from dmvmat_ecg.phase3b.protocol import adopted_settings, seed
from dmvmat_ecg.phase4.safety import digest, new_json, sha

from .alignment import alignment_loss
from .common import RUN, load_source, setup
from .features import Features
from .source_data import SourceData


def tag(lam):
    if lam not in (0.1, 1.0):
        raise ValueError("Only the prespecified lambda grid is permitted")
    return f"lambda{lam:g}"


class TargetStream:
    def __init__(self, n, generator):
        self.n = n
        self.generator = generator
        self.order = torch.randperm(n, generator=generator)
        self.pos = 0

    def take(self, size):
        values = []
        while size:
            n = min(size, self.n - self.pos)
            values.append(self.order[self.pos : self.pos + n])
            self.pos += n
            size -= n
            if self.pos == self.n:
                self.order = torch.randperm(self.n, generator=self.generator)
                self.pos = 0
        return torch.cat(values).numpy()


def source_fit(source_seed, lam, *, retain=True):
    started = time.monotonic()
    torch.set_num_threads(4)
    target = setup()
    data = SourceData()
    features = Features()
    identity = digest(
        dict(
            target=target.identity,
            source=data.identity,
            seed=source_seed,
            lam=lam,
            features=sha(RUN / "ALIGNMENT_INPUTS_MANIFEST.json"),
        )
    )
    prefix = tag(lam) + f"_{source_seed}"
    entry_path = RUN / "aligned_sources" / (prefix + ".json")
    if retain and entry_path.exists():
        entry = json.loads(entry_path.read_text())
        if entry["identity"] != identity or sha(RUN / entry["checkpoint"]) != entry["sha256"]:
            raise PermissionError("Aligned source resumption identity mismatch")
        return entry
    if retain and (RUN / "aligned_sources" / (prefix + ".pt")).exists():
        raise FileExistsError("Partial source checkpoint is not overwritten")
    _, norm, _, _ = load_source(target, source_seed)
    model = fresh_model(source_seed).cpu().float()
    xrole = "source_train"
    y = torch.tensor(data.y[xrole], dtype=torch.float32)
    observed = torch.ones(y.shape, dtype=torch.bool)
    vy = data.y["source_stop"]
    vv = features.batch("source_stop", np.arange(len(vy)))
    settings = adopted_settings()
    optimizer = torch.optim.AdamW(model.parameters(), **settings["source_optimizer"])
    steps = math.ceil(len(y) / 128)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.01, epochs=50, steps_per_epoch=steps, **settings["source_schedule"]
    )
    best = None
    best_state = None
    history = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(source_seed)
        source_generator = torch.Generator().manual_seed(source_seed)
        stream = TargetStream(
            len(target.ids("U")), torch.Generator().manual_seed(seed([20261126, source_seed]))
        )
        for epoch in range(1, 51):
            order = torch.randperm(len(y), generator=source_generator)
            model.train()
            losses = []
            bces = []
            mmds = []
            for start in range(0, len(y), 128):
                ix = order[start : start + 128]
                optimizer.zero_grad(set_to_none=True)
                source_reps = model.representations(**features.batch("source_train", ix.numpy()))
                logits = model.prediction_head(source_reps["fusion_2"])
                bce = masked_bce(logits, y[ix], observed[ix])
                target_reps = model.representations(**features.batch("U", stream.take(32)))
                mmd = alignment_loss(source_reps, target_reps)
                loss = bce + lam * mmd
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite aligned source loss")
                loss.backward()
                if any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in model.parameters()
                ):
                    raise FloatingPointError("Nonfinite aligned source gradient")
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in model.parameters()):
                    raise FloatingPointError("Nonfinite aligned source parameter")
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
            aps = average_precision_score(vy, probabilities, average=None)
            ap = float(aps.mean())
            if (
                best is None
                or ap > best[0] + 1e-12
                or (abs(ap - best[0]) <= 1e-12 and val_bce < best[1] - 1e-12)
            ):
                best = (ap, val_bce, epoch)
                best_state = copy.deepcopy(model.state_dict())
            row = dict(
                epoch=epoch,
                loss=float(np.mean(losses)),
                BCE=float(np.mean(bces)),
                MMD=float(np.mean(mmds)),
                fold9_macro_AP=ap,
                fold9_BCE=val_bce,
            )
            history.append(row)
            print(
                f"ALIGNED_SOURCE seed={source_seed} lambda={lam:g} retain={retain} epoch={epoch}/50 seconds={time.monotonic() - started:.1f}",
                flush=True,
            )
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        retained = model(**vv).sigmoid().numpy()
    aps = average_precision_score(vy, retained, average=None)
    aucs = roc_auc_score(vy, retained, average=None)
    report = dict(
        identity=identity,
        source_seed=source_seed,
        lambda_value=lam,
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
        target_unlabelled_records=len(target.ids("U")),
        U_labels_read=False,
        source_batch=128,
        target_batch=32,
        epochs=50,
    )
    if retain:
        folder = RUN / "aligned_sources"
        folder.mkdir(mode=0o700, exist_ok=True)
        checkpoint = folder / (prefix + ".pt")
        payload = dict(
            state=best_state,
            normalization=norm.__dict__,
            state_hash=report["state_hash"],
            identity=dict(seed=source_seed, lambda_value=lam, source_bundle=data.identity),
            selected_epoch=best[2],
            source_stop_per_label=[dict(AP=float(aps[j]), AUROC=float(aucs[j])) for j in range(2)],
            history=history,
        )
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        fd = os.open(checkpoint, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(buffer.getvalue())
            f.flush()
            os.fsync(f.fileno())
        report.update(checkpoint=str(checkpoint.relative_to(RUN)), sha256=sha(checkpoint))
        with (folder / (prefix + "_fold9.npz")).open("xb") as f:
            np.savez_compressed(f, p=retained)
        new_json(entry_path, report)
    return report
