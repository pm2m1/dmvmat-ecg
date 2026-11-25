"""Exact budget sampler plus declared AdaBN, LP-FT and L2-SP extensions."""

import copy
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn

from dmvmat_ecg.phase2.adaptation import SelectiveModel, masked_bce, state_hash
from dmvmat_ecg.phase3b.models import inputs
from dmvmat_ecg.phase3b.protocol import OPTIMIZER, seed
from dmvmat_ecg.phase4.runtime import mask_dict
from dmvmat_ecg.phase4.safety import digest, new_json, sha

from .common import RUN, load_source, predict


def adabn(model, norm, target):
    """Signal-only pass. This routine has no target-label access."""
    model = copy.deepcopy(model).cpu().eval()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    old = []
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            old.append((module, module.momentum))
            module.reset_running_stats()
            module.momentum = None
            module.train()
    order = np.random.default_rng(20261125).permutation(len(target.ids("U")))
    ui = target.index(target.ids("U"))
    with torch.no_grad():
        for start in range(0, len(order), 32):
            batch = ui[order[start : start + 32]]
            model(**inputs(model, norm.transform(target.x[batch])))
    for module, momentum in old:
        module.momentum = momentum
    model.eval()
    if any(not torch.equal(before[n], p) for n, p in model.named_parameters()):
        raise RuntimeError("AdaBN changed parameters")
    if any(row["role"] == "U" for row in target.label_access):
        raise PermissionError("AdaBN read unlabelled outcomes")
    return model


def fit(target, s, d, budget, method, *, mask=127, alpha=0.0, aligned=None):
    started = time.monotonic()
    torch.set_num_threads(4)
    model, norm, saved, source_sha = load_source(target, s, aligned)
    original = state_hash(model)
    if method in ("adabn", "adabn_full"):
        model = adabn(model, norm, target)
    if method in ("source", "adabn"):
        predictions = {r: predict(model, norm, target, r) for r in ("A", "B")}
        return predictions, dict(
            method=method,
            source_seed=s,
            draw_seed=d,
            budget=0,
            nominal_budget=budget,
            updates=0,
            state_hash=state_hash(model),
            source_sha256=source_sha,
            aligned=aligned,
            mask_index=None,
            runtime_seconds=time.monotonic() - started,
            history=[],
            U_label_accesses=0,
        )
    if budget not in (64, 128, 256):
        raise ValueError("Unregistered budget")
    chosen = mask if method in ("oracle", "aligned_selector") else 0 if method == "lpft" else 127
    controller = SelectiveModel(model, mask_dict(chosen)).cpu()
    ids = target.ids("F", d, budget) + target.ids("G", d, budget)
    xi = target.index(ids)
    y = np.concatenate([target.labels("F", d, budget), target.labels("G", d, budget)])
    yy, observed = torch.tensor(y, dtype=torch.float32), torch.ones(y.shape, dtype=torch.bool)
    xx = inputs(controller.model, norm.transform(target.x[xi]))
    optimizer = torch.optim.AdamW(
        [p for p in controller.parameters() if p.requires_grad], **OPTIMIZER
    )
    source_weights = {n: p.detach().clone() for n, p in controller.model.named_parameters()}
    initial = copy.deepcopy(controller.model.state_dict())
    fit_seed = seed([78501, s, d, budget])
    losses = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(fit_seed)
        generator = torch.Generator().manual_seed(fit_seed)
        for step in range(1, 201):
            if method == "lpft" and step == 51:
                existing = {id(p) for group in optimizer.param_groups for p in group["params"]}
                for module in controller.model.selectable_modules().values():
                    module.requires_grad_(True)
                controller.mask = mask_dict(127)
                controller._frozen = {}
                optimizer.add_param_group(
                    {"params": [p for p in controller.parameters() if id(p) not in existing]}
                )
            batch = torch.randint(len(y), (32,), generator=generator)
            controller.train()
            optimizer.zero_grad(set_to_none=True)
            loss = masked_bce(
                controller(**{k: v[batch] for k, v in xx.items()}), yy[batch], observed[batch]
            )
            if method == "l2sp":
                loss = loss + alpha * sum(
                    torch.square(p - source_weights[n]).sum()
                    for n, p in controller.model.named_parameters()
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite target loss")
            loss.backward()
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in controller.parameters()
            ):
                raise FloatingPointError("Nonfinite gradient")
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in controller.parameters()):
                raise FloatingPointError("Nonfinite target parameter")
            controller.verify_frozen()
            losses.append(float(loss.detach()))
    controller.eval()
    predictions = {r: predict(controller.model, norm, target, r) for r in ("A", "B")}
    drift = {}
    for n, module in controller.model.selectable_modules().items():
        drift[n] = float(
            np.sqrt(
                sum(
                    float(torch.square(p.detach() - initial[f"{n}.{k}"]).sum())
                    for k, p in module.named_parameters()
                )
            )
        )
    report = dict(
        method=method,
        source_seed=s,
        draw_seed=d,
        budget=budget,
        updates=200,
        mask_index=mask if method in ("oracle", "aligned_selector") else chosen,
        alpha=alpha,
        aligned=aligned,
        fit_seed=fit_seed,
        state_hash=state_hash(controller.model),
        source_sha256=source_sha,
        source_state_hash=original,
        history=losses,
        final_training_loss=losses[-1],
        drift=drift,
        U_label_accesses=0,
        runtime_seconds=time.monotonic() - started,
        head_always_trainable=True,
    )
    if method in ("full", "oracle") and aligned is None and budget == 128:
        for name, reference in target.references[f"{s}/{d}"].items():
            if reference["mask"] == mask_dict(127 if method == "full" else mask):
                for key in ("state_hash", "drift", "final_training_loss"):
                    if report[key] != reference[key]:
                        raise RuntimeError(f"ORIGINAL_REFERENCE_MISMATCH:{name}:{key}")
    return predictions, report


def key(s, d, budget, method, mask=127, alpha=0.0, aligned=None):
    return f"{method}_s{s}_d{d}_b{budget}_m{mask:03d}_a{alpha:g}_aligned{aligned}"


def save_fit(target, s, d, budget, method, *, mask=127, alpha=0.0, aligned=None):
    name = key(s, d, budget, method, mask, alpha, aligned)
    report_path = RUN / "fits" / (name + ".json")
    pred_path = RUN / "predictions" / (name + ".npz")
    identity = digest(
        dict(
            bundle=target.identity,
            split=target.split,
            s=s,
            d=d,
            budget=budget,
            method=method,
            mask=mask,
            alpha=alpha,
            aligned=aligned,
        )
    )
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report["identity"] != identity or sha(pred_path) != report["predictions_sha256"]:
            raise PermissionError("Resumption input mismatch")
        return report
    if pred_path.exists():
        raise FileExistsError("Orphaned prediction file; no overwrite")
    pred, report = fit(target, s, d, budget, method, mask=mask, alpha=alpha, aligned=aligned)
    report_path.parent.mkdir(exist_ok=True)
    pred_path.parent.mkdir(exist_ok=True)
    with pred_path.open("xb") as f:
        np.savez_compressed(f, **pred)
        f.flush()
        os.fsync(f.fileno())
    ya = target.labels("A")
    score = average_precision_score(ya, pred["A"], average=None)
    report.update(
        identity=identity,
        predictions_sha256=sha(pred_path),
        A_AP=score.tolist(),
        A_macro_AP=float(score.mean()),
        status="COMPLETE",
        file=name,
    )
    new_json(report_path, report)
    return report
