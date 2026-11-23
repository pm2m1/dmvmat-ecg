"""Fixed full-FT validation engine; stores references, never target checkpoints."""

import concurrent.futures
import copy
import json
import multiprocessing
import os
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from dmvmat_ecg.phase2.adaptation import SelectiveModel, masked_bce, state_hash
from dmvmat_ecg.phase3b.models import fresh_model, inputs
from dmvmat_ecg.phase3b.protocol import OPTIMIZER, seed
from dmvmat_ecg.phase3b.roles import Normalization
from dmvmat_ecg.phase4.runtime import mask_dict
from dmvmat_ecg.phase4.safety import digest, new_json, sha
from dmvmat_ecg.phase4b.common import Target, predict
from dmvmat_ecg.phase4b.stage_lock import stage_lock

from .data import ROOT, RUN, setup


def full_fit(target, s, d, budget, model, norm, source_sha):
    """Exact Phase4B full recipe; source loading is explicitly injected."""
    started = time.monotonic()
    torch.set_num_threads(4)
    if budget not in (64, 128, 256):
        raise ValueError("Unregistered target budget")
    original = state_hash(model)
    controller = SelectiveModel(model, mask_dict(127)).cpu()
    ids = target.ids("F", d, budget) + target.ids("G", d, budget)
    xi = target.index(ids)
    y = np.concatenate([target.labels("F", d, budget), target.labels("G", d, budget)])
    yy, observed = torch.tensor(y, dtype=torch.float32), torch.ones(y.shape, dtype=torch.bool)
    xx = inputs(controller.model, norm.transform(target.x[xi]))
    optimizer = torch.optim.AdamW(
        [p for p in controller.parameters() if p.requires_grad], **OPTIMIZER
    )
    initial = copy.deepcopy(controller.model.state_dict())
    fit_seed = seed([78501, s, d, budget])
    losses = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(fit_seed)
        generator = torch.Generator().manual_seed(fit_seed)
        for _ in range(200):
            batch = torch.randint(len(y), (32,), generator=generator)
            controller.train()
            optimizer.zero_grad(set_to_none=True)
            loss = masked_bce(
                controller(**{k: v[batch] for k, v in xx.items()}), yy[batch], observed[batch]
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite retrain target loss")
            loss.backward()
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in controller.parameters()
            ):
                raise FloatingPointError("Nonfinite retrain target gradient")
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in controller.parameters()):
                raise FloatingPointError("Nonfinite retrain target parameter")
            controller.verify_frozen()
            losses.append(float(loss.detach()))
    controller.eval()
    predictions = {role: predict(controller.model, norm, target, role) for role in ("A", "B")}
    drift = {}
    for name, module in controller.model.selectable_modules().items():
        drift[name] = float(
            np.sqrt(
                sum(
                    float(torch.square(p.detach() - initial[f"{name}.{k}"]).sum())
                    for k, p in module.named_parameters()
                )
            )
        )
    report = dict(
        method="retrain_full",
        source_seed=s,
        draw_seed=d,
        budget=budget,
        updates=200,
        mask_index=127,
        fit_seed=fit_seed,
        state_hash=state_hash(controller.model),
        source_sha256=source_sha,
        source_state_hash=original,
        normalization=norm.__dict__,
        history=losses,
        final_training_loss=losses[-1],
        drift=drift,
        runtime_seconds=time.monotonic() - started,
        head_always_trainable=True,
    )
    return controller.model, predictions, report


def authenticate(stored, arrays, regenerated, predictions):
    for role in ("A", "B"):
        if (
            arrays[role].shape != predictions[role].shape
            or arrays[role].dtype != predictions[role].dtype
            or arrays[role].tobytes(order="C") != predictions[role].tobytes(order="C")
        ):
            raise RuntimeError("STOP_REPLAY_PROBABILITY_MISMATCH:" + role)
    for name in (
        "source_seed",
        "draw_seed",
        "budget",
        "fit_seed",
        "updates",
        "source_sha256",
        "source_state_hash",
        "normalization",
        "state_hash",
        "history",
        "drift",
    ):
        if stored[name] != regenerated[name]:
            raise RuntimeError("STOP_REPLAY_STATE_OR_RECIPE_MISMATCH:" + name)


def name(s, d, b):
    return f"retrain_full_s{s}_d{d}_b{b}"


def load(s, target):
    entry = json.loads((RUN / "source_models" / f"retrain_s{s}.json").read_text())
    path = RUN / entry["checkpoint"]
    if sha(path) != entry["sha256"]:
        raise PermissionError("Placebo source file hash differs")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = fresh_model(s)
    model.load_state_dict(saved["state"], strict=True)
    model.eval()
    if state_hash(model) != entry["state_hash"]:
        raise PermissionError("Placebo source state differs")
    norm = Normalization(**saved["normalization"])
    if norm.__dict__ != target.sources[str(s)]["normalization"]:
        raise PermissionError("Source-only normalization differs from original sources")
    return model, norm, entry["sha256"]


def save(s, d, b):
    setup("target")
    t = Target()
    fn = name(s, d, b)
    path = RUN / "fits" / (fn + ".json")
    pred = RUN / "predictions" / (fn + ".npz")
    model, norm, cp_sha = load(s, t)
    identity = digest(
        dict(
            bundle=t.identity,
            split=t.split,
            s=s,
            d=d,
            b=b,
            source_sha256=cp_sha,
            code=sha(ROOT / "src/dmvmat_ecg/phase4c_retrain/target.py"),
        )
    )
    if path.exists():
        r = json.loads(path.read_text())
        if (
            r["identity"] != identity
            or sha(pred) != r["predictions_sha256"]
            or r["status"] != "COMPLETE"
        ):
            raise PermissionError("Target resume identity differs")
        return r
    if pred.exists():
        raise FileExistsError("Partial target output cannot be overwritten")
    _, p, r = full_fit(t, s, d, b, model, norm, cp_sha)
    path.parent.mkdir(mode=0o700, exist_ok=True)
    pred.parent.mkdir(mode=0o700, exist_ok=True)
    fd = os.open(pred, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        np.savez_compressed(stream, **p)
        stream.flush()
        os.fsync(stream.fileno())
    a = average_precision_score(t.labels("A"), p["A"], average=None)
    r.update(
        identity=identity,
        status="COMPLETE",
        file=fn,
        predictions_sha256=sha(pred),
        A_AP=a.tolist(),
        A_macro_AP=float(a.mean()),
    )
    new_json(path, r)
    return r


def run():
    setup("target")
    if json.loads((RUN / "SOURCES_COMPLETE.json").read_text())["status"] != "COMPLETE":
        raise PermissionError("All retrain sources must finish")
    with stage_lock(RUN, "baselines"):
        binding = RUN / "TARGET_CODE_BINDING.json"
        expected = {
            str(p.relative_to(ROOT)): sha(p)
            for p in (
                ROOT / "src/dmvmat_ecg/phase4c_retrain/target.py",
                ROOT / "src/dmvmat_ecg/phase4c_retrain/data.py",
                ROOT / "src/dmvmat_ecg/phase4b/common.py",
                ROOT / "src/dmvmat_ecg/phase3b/protocol.py",
                ROOT / "src/dmvmat_ecg/phase3b/models.py",
                ROOT / "src/dmvmat_ecg/phase3b/roles.py",
                ROOT / "src/dmvmat_ecg/phase2/adaptation.py",
                ROOT / "src/dmvmat_ecg/preprocessing/signals.py",
                ROOT / "uv.lock",
            )
        }
        if binding.exists():
            if json.loads(binding.read_text()) != expected:
                raise PermissionError("Target code changed")
        else:
            new_json(binding, expected)
        gate = RUN / "TARGET_REPRODUCIBILITY.json"
        if not gate.exists():
            prior = save(15101, 16101, 128)
            t = Target()
            m, norm, cp = load(15101, t)
            _, p, r = full_fit(t, 15101, 16101, 128, m, norm, cp)
            with np.load(
                RUN / "predictions" / (prior["file"] + ".npz"), allow_pickle=False
            ) as arrays:
                authenticate(prior, arrays, r, p)
            new_json(
                gate,
                dict(
                    status="PASS",
                    A_B_bitwise_identical=True,
                    state_history_and_recipe_identical=True,
                    repeat_seconds=r["runtime_seconds"],
                ),
            )
            print("RETRAIN_TARGET_REPLAY_PASS", flush=True)
        elif json.loads(gate.read_text())["status"] != "PASS":
            raise PermissionError("Prior target replay failed")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=3, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            jobs = [
                pool.submit(save, s, d, b)
                for s in (15101, 15102, 15103)
                for d in (16101, 16102, 16103)
                for b in (64, 128, 256)
            ]
            for n, f in enumerate(concurrent.futures.as_completed(jobs), 1):
                f.result()
                print(f"RETRAIN_TARGET_PROGRESS {n}/27", flush=True)
        p = RUN / "TARGETS_COMPLETE.json"
        if not p.exists():
            new_json(
                p,
                dict(
                    status="COMPLETE",
                    fits=27,
                    updates_per_fit=200,
                    test_access=False,
                    target_checkpoints=0,
                ),
            )


if __name__ == "__main__":
    run()
