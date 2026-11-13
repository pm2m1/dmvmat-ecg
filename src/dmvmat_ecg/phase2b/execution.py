"""Guarded execution primitives. No real permission is granted by importing them."""

import io
import json
import math

import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.metrics import f1_score

from dmvmat_ecg.phase2.adaptation import (
    APCheckpoint,
    SelectiveModel,
    macro_ap,
    masked_bce,
    random_mask,
    serialize_synthetic,
    state_hash,
)
from dmvmat_ecg.phase2.contracts import MODULES

from .access import fit_preprocessing, fresh_source, fresh_target, verify_new_source
from .contract import LABELS, digest
from .integration import load_new_source, model_inputs, selector_evidence


def run_job(
    access,
    job,
    *,
    synthetic_factory=None,
    synthetic_reader=None,
    raw_root=None,
    synthetic_steps=None,
    synthetic_evidence=None,
    synthetic_dimensions=None,
    retry_of=None,
):
    """Reserve durably BEFORE model initialization, signal read or fitting."""
    identifier = access.ledger.reserve(job, retry_of=retry_of)
    try:
        return execute_reserved(
            access,
            identifier,
            synthetic_factory=synthetic_factory,
            synthetic_reader=synthetic_reader,
            raw_root=raw_root,
            synthetic_steps=synthetic_steps,
            synthetic_evidence=synthetic_evidence,
            synthetic_dimensions=synthetic_dimensions,
        )
    except BaseException as error:
        access.ledger.finish(
            identifier, "failed", dict(type=type(error).__name__, reason=str(error))
        )
        raise


def execute_reserved(
    access,
    attempt,
    *,
    synthetic_factory=None,
    synthetic_reader=None,
    raw_root=None,
    synthetic_steps=None,
    synthetic_evidence=None,
    synthetic_dimensions=None,
):
    ledger = access.ledger
    job, spec = ledger.active(attempt)
    if ledger.identity["namespace"].startswith("real:"):
        torch.set_num_threads(4)
        if torch.get_default_dtype() != torch.float32:
            raise PermissionError("Real CPU float32 execution policy changed")
    if ledger.identity["namespace"].startswith("real:") and any(
        v is not None
        for v in (
            synthetic_factory,
            synthetic_reader,
            synthetic_steps,
            synthetic_evidence,
            synthetic_dimensions,
        )
    ):
        raise PermissionError("Synthetic overrides cannot weaken a real experiment")
    reader = dict(synthetic_reader=synthetic_reader, raw_root=raw_root)
    cfg = ledger.config
    source_stage = spec["kind"] == "source"
    scratch = source_stage or spec.get("method") == "target_only"
    batches = [access.load(r, "fit", attempt=attempt, **reader) for r in spec["fit_roles"]]
    stop = access.load(spec["stop_role"], "stop", attempt=attempt, **reader)
    if source_stage:
        model = fresh_source(access, attempt, synthetic_factory=synthetic_factory)
        preprocessing = fit_preprocessing(batches, access, attempt)
        source_attempt = None
    elif scratch:
        model = fresh_target(access, attempt, synthetic_factory=synthetic_factory)
        preprocessing = fit_preprocessing(batches, access, attempt)
        source_attempt = None
    else:
        source_attempt = ledger.successful()["source"]
        model, preprocessing = load_new_source(
            access, source_attempt, synthetic_factory=synthetic_factory
        )
    initial_hash = state_hash(model)
    mask = dict.fromkeys(MODULES, "ADAPT")
    if not source_stage:
        method = spec["method"]
        if method in {"head_f", "head_cost"}:
            mask = dict.fromkeys(MODULES, "SHARE")
        if method in {"selective_a", "selective_b", "random_a"}:
            evidence = selector_evidence(
                access,
                attempt,
                ledger.successful()["source"],
                formulation="B" if method == "selective_b" else "A",
                config=synthetic_evidence,
                synthetic_factory=synthetic_factory,
                dimensions=synthetic_dimensions,
                **reader,
            )
            mask = evidence["mask"]
            if method == "random_a":
                mask, matching = random_mask(model, mask, spec["seed"])
        model = SelectiveModel(model, mask)
    model = model.cpu()
    hp = cfg["source_optimizer"] if source_stage else cfg["target_optimizer"]
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=hp["lr"],
        weight_decay=hp["weight_decay"],
        betas=tuple(hp["betas"]),
        eps=hp["eps"],
    )
    inputs = {
        k: torch.cat([model_inputs(b, preprocessing)[k] for b in batches])
        for k in ("wave", "rhythm", "spectrum")
    }
    y = torch.tensor(np.concatenate([b["y"] for b in batches]), dtype=torch.float32)
    observed = torch.tensor(np.concatenate([b["observed"] for b in batches]), dtype=torch.bool)
    stopping_inputs = model_inputs(stop, preprocessing)
    checkpoint = APCheckpoint()
    history = []

    def score(index, eligible):
        model.eval()
        with torch.no_grad():
            logits = model(**stopping_inputs)
            loss = float(
                masked_bce(logits, torch.tensor(stop["y"]), torch.tensor(stop["observed"]))
            )
            ap = macro_ap(stop["y"], logits.sigmoid().numpy(), stop["observed"])
        history.append(dict(look=index, ap=ap, bce=loss, eligible=eligible))
        return checkpoint.consider(model, ap=ap, bce=loss, epoch=index) if eligible else False

    score(0, not scratch)
    batch_size = hp["batch_size"]
    steps_per_epoch = math.ceil(len(y) / batch_size)
    total = spec["epochs"] * steps_per_epoch if source_stage else spec["updates"]
    if synthetic_steps is not None:
        total = min(total, synthetic_steps)
    schedule = None
    if source_stage:
        s = {k: v for k, v in cfg["source_schedule"].items() if k != "name"}
        schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=max(total, 2), **s)
    generator = torch.Generator().manual_seed(spec["seed"])
    stale = 0
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(spec["seed"])
        for step in range(1, total + 1):
            model.train()
            if source_stage:
                if (step - 1) % steps_per_epoch == 0:
                    order = torch.randperm(len(y), generator=generator)
                ix = order[
                    ((step - 1) % steps_per_epoch) * batch_size : ((step - 1) % steps_per_epoch + 1)
                    * batch_size
                ]
            else:
                ix = torch.randperm(len(y), generator=generator)[:batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = masked_bce(model(**{k: v[ix] for k, v in inputs.items()}), y[ix], observed[ix])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, float("inf"), error_if_nonfinite=True)
            optimizer.step()
            if schedule:
                schedule.step()
            if any(not torch.isfinite(p).all() for p in parameters):
                raise FloatingPointError("Nonfinite update")
            if isinstance(model, SelectiveModel):
                model.verify_frozen()
            interval = steps_per_epoch if source_stage else spec["look_every"]
            if step % interval == 0 or step == total:
                stale = 0 if score(step, True) else stale + 1
                if not source_stage and stale >= hp["patience"]:
                    break
    model.load_state_dict(checkpoint.state, strict=True)
    model.eval()
    if isinstance(model, SelectiveModel):
        model.verify_frozen()
    bare = model.model if isinstance(model, SelectiveModel) else model
    payload, sha = serialize_synthetic(bare)
    lineage = {
        **ledger.identity,
        "job_id": job,
        "labels": list(LABELS),
        "checkpoint_sha256": sha,
        "retained_state_hash": state_hash(bare),
        "initial_state_hash": initial_hash,
        "origin": "random-scratch" if scratch else "adapted",
        "source_attempt": source_attempt,
        "preprocessing": preprocessing,
        "source_training_hash": access._manifest["role_hashes"]["source_train"]
        if source_stage
        else None,
        "source_selection_hash": access._manifest["role_hashes"]["source_stop"]
        if source_stage
        else None,
        "selected_update": checkpoint.epoch,
        "updates": step,
        "mask": mask,
        "history": history,
        "label_cost": access.label_cost(attempt),
        "exact_resume_supported": False,
        "parameters": sum(p.numel() for p in bare.parameters()),
    }
    if not source_stage and not scratch:
        unchanged, _ = load_new_source(access, source_attempt, synthetic_factory=synthetic_factory)
        if state_hash(unchanged) != initial_hash:
            raise RuntimeError("Source artifact changed")
    ledger.publish(attempt, "checkpoint.pt", payload)
    ledger.publish(attempt, "lineage.json", json.dumps(lineage, sort_keys=True).encode())
    ledger.finish(attempt, "successful")
    return attempt


def completed_model(access, job, *, synthetic_factory=None):
    identifier = access.ledger.successful()[job]
    if synthetic_factory is not None and not access.ledger.identity["namespace"].startswith(
        "synthetic:"
    ):
        raise PermissionError("Synthetic model in real namespace")
    lineage = json.loads(access.ledger.artifact(identifier, "lineage.json").read_text())
    if any(lineage.get(k) != v for k, v in {**access.ledger.identity, "job_id": job}.items()):
        raise PermissionError("Completed model lineage mismatch")
    if job == "source":
        verify_new_source(access, identifier)
    elif access.ledger.config["planned_jobs"][job].get("method") == "target_only":
        expected = digest(
            [
                access._manifest["role_hashes"][r]
                for r in access.ledger.config["planned_jobs"][job]["fit_roles"]
            ]
        )
        if (
            lineage["origin"] != "random-scratch"
            or lineage["preprocessing"]["fit_attempt"] != identifier
            or lineage["preprocessing"]["fit_membership_hash"] != expected
        ):
            raise PermissionError("Target-only preprocessing/initialization differs")
    else:
        source = access.ledger.successful()["source"]
        _, parent = verify_new_source(access, source)
        if (
            lineage["origin"] != "adapted"
            or lineage["source_attempt"] != source
            or lineage["preprocessing"] != parent["preprocessing"]
        ):
            raise PermissionError("Adaptation did not inherit the verified new source")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3101)
        if synthetic_factory:
            model = synthetic_factory()
        else:
            from dmvmat_ecg.models.networks import FiveViewECG

            model = FiveViewECG()
            model.prediction_head = torch.nn.Linear(64, 2)
    model.load_state_dict(
        torch.load(
            access.ledger.artifact(identifier, "checkpoint.pt"),
            weights_only=True,
            map_location="cpu",
        ),
        strict=True,
    )
    if state_hash(model) != lineage["retained_state_hash"]:
        raise PermissionError("Weights changed")
    return model.eval(), lineage


def calibrate_job(access, job, *, synthetic_factory=None, synthetic_reader=None, raw_root=None):
    batch = access.load(
        "C", "calibrate", job=job, synthetic_reader=synthetic_reader, raw_root=raw_root
    )
    model, lineage = completed_model(access, job, synthetic_factory=synthetic_factory)
    with torch.no_grad():
        logits = model(**model_inputs(batch, lineage["preprocessing"])).numpy()
    mappings = []
    for j in range(2):
        use = batch["observed"][:, j]
        y = batch["y"][use, j]
        z = np.clip(logits[use, j], -math.log(1e6 - 1), math.log(1e6 - 1))
        status = "identity-insufficient-support"
        a, b = 0.0, 0.0
        if min(int((y == 1).sum()), int((y == 0).sum())) >= 20:

            def objective(v):
                calibrated_logit = np.exp(v[0]) * z + v[1]
                return float(
                    np.mean(np.logaddexp(0, calibrated_logit) - y * calibrated_logit)
                    + 1e-4 * np.square(v).sum()
                )

            result = minimize(
                objective,
                [0.0, 0.0],
                method="L-BFGS-B",
                bounds=[(math.log(0.01), math.log(100)), (-10, 10)],
            )
            if result.success and np.isfinite(result.x).all():
                a, b = map(float, result.x)
                status = "fitted"
            else:
                status = "identity-convergence-failure"
        p = 1 / (1 + np.exp(-np.clip(np.exp(a) * z + b, -700, 700)))
        grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
        if status == "fitted":
            threshold = min(
                grid, key=lambda t: (-f1_score(y, p >= t, zero_division=0), abs(t - 0.5), t)
            )
        else:
            threshold = 0.5
        mappings.append(dict(a=a, b=b, status=status, threshold=float(threshold)))
    artifact = dict(
        namespace=access.ledger.identity["namespace"],
        job_id=job,
        fit_role="C",
        manifest_hash=access.ledger.identity["manifest_hash"],
        membership_hash=batch["membership_hash"],
        preprocessing_hash=digest(lineage["preprocessing"]),
        mappings=mappings,
        labels=list(LABELS),
        label_cost=access.label_cost(job),
    )
    access.ledger.record_calibration(job, json.dumps(artifact, sort_keys=True).encode())
    return artifact


def predict_locked(
    access, job, token, *, synthetic_factory=None, synthetic_reader=None, raw_root=None
):
    batch = access.load(
        "O", "evaluate_predict", token=token, synthetic_reader=synthetic_reader, raw_root=raw_root
    )
    assert batch["y"] is None
    model, lineage = completed_model(access, job, synthetic_factory=synthetic_factory)
    with torch.no_grad():
        p = model(**model_inputs(batch, lineage["preprocessing"])).sigmoid().numpy()
    calibrations = [
        json.loads(v)
        for (v,) in access.ledger.db.execute("SELECT detail FROM events WHERE kind='calibration'")
    ]
    record = next(c for c in calibrations if c["job"] == job)
    mapping = json.loads((access.ledger.folder / record["path"]).read_text())
    pc = p.copy()
    for j, m in enumerate(mapping["mappings"]):
        if m["status"] == "fitted":
            raw = np.clip(p[:, j].astype(float), 1e-6, 1 - 1e-6)
            z = np.exp(m["a"]) * np.log(raw / (1 - raw)) + m["b"]
            pc[:, j] = 1 / (1 + np.exp(-np.clip(z, -700, 700)))
    buf = io.BytesIO()
    np.savez(buf, p=p, p_calibrated=pc)
    access.ledger.publish_predictions(token, job, buf.getvalue())
    return p
