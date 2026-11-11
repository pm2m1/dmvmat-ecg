"""Config-driven BCE training; fold10 is only reachable inside a consumed lease."""

import json
import resource
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dmvmat_ecg.data.store import CropDataset, Store
from dmvmat_ecg.evaluation.metrics import ranking, thresholds_from_validation
from dmvmat_ecg.models.networks import build, parameters
from dmvmat_ecg.utils.io import sha, source_hash, write_json


def sync(device):
    if device == "mps":
        torch.mps.synchronize()


def dataset(study, name, role, seed=101):
    config = json.loads((study.root / "configs/study.json").read_text())
    x, meta = Store(study).load(role)
    stats_path = study.root / "private/cache/standardization.json"
    if sha(stats_path) != study.get("dataset")["cache_hashes"]["standardization.json"]:
        raise PermissionError("Fitted preprocessing changed")
    stats = json.loads(stats_path.read_text())
    from dmvmat_ecg.utils.io import digest

    if stats["fit_role"] != "train" or stats["fit_membership_hash"] != digest(
        study.get("memberships")["train"]
    ):
        raise PermissionError("Preprocessing was fitted outside the training membership")
    cfg = config["models"][name]
    ds = CropDataset(
        x,
        meta["y"],
        meta["ecg_id"],
        stats,
        training=role == "train",
        multiview=name.startswith("multiview"),
        augmentation=cfg["additional_augmentation"] and role == "train",
        seed=seed,
        window=cfg["window"],
    )
    return ds, meta


def predict(model, ds, device, batch_size):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    probabilities = np.full((len(ds.x), 5), -np.inf, dtype=np.float32)
    model.eval()
    sync(device)
    start = time.perf_counter()
    with torch.no_grad():
        for inputs, _, record in loader:
            logits = model(**{k: v.to(device) for k, v in inputs.items()})
            p = logits.sigmoid().cpu().numpy()
            if not np.isfinite(p).all():
                raise FloatingPointError("Nonfinite held-out probabilities")
            np.maximum.at(probabilities, record.numpy(), p)
    sync(device)
    if not np.isfinite(probabilities).all():
        raise ValueError("Incomplete record predictions")
    return probabilities, time.perf_counter() - start


def bce_prob(y, p):
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    return float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).mean())


def fit(study, name, seed, device, *, smoke=False):
    config = json.loads((study.root / "configs/study.json").read_text())
    original_source = source_hash(study.root)
    stage = "smoke" if smoke else "full"
    number = study.reserve(name, seed, stage, config)
    folder = study.root / "private/runs" / number
    folder.mkdir(parents=True, exist_ok=False)
    write_json(
        folder / "entry.json",
        dict(
            status="attempted",
            model=name,
            seed=seed,
            stage=stage,
            config=config,
            device=device,
            source_hash=original_source,
        ),
    )
    try:
        train, train_meta = dataset(study, name, "train", seed)
        if smoke:
            limit = min(config["smoke"]["records"], len(train.x))
            train.x = train.x[:limit]
            train.y = train.y[:limit]
            train.ids = train.ids[:limit]
        model = build(name, seed).to(device)
        batch = config["batch_size"]
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
            betas=tuple(config["adam_betas"]),
            eps=config["adam_epsilon"],
        )
        epochs = 1 if smoke else config["epochs"]
        steps = (len(train) + batch - 1) // batch
        schedule = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config["learning_rate"],
            epochs=epochs,
            steps_per_epoch=steps,
            **{k: v for k, v in config["schedule"].items() if k != "kind"},
        )
        val, val_meta = dataset(study, name, "validation", seed) if not smoke else (None, None)
        best = -np.inf
        best_epoch = None
        history = []
        best_p = None
        total_training = 0.0
        peak = 0
        for epoch in range(epochs):
            train.epoch = epoch
            generator = torch.Generator().manual_seed(seed + 100003 * epoch)
            loader = DataLoader(
                train,
                batch_size=batch,
                shuffle=True,
                num_workers=0,
                generator=generator,
                drop_last=False,
            )
            model.train()
            loss_sum = 0.0
            seen = 0
            ys = []
            ps = []
            sync(device)
            tick = time.perf_counter()
            for step, (inputs, target, _) in enumerate(loader):
                if smoke and step >= config["smoke"]["maximum_batches"]:
                    break
                optimizer.zero_grad(set_to_none=True)
                logits = model(**{k: v.to(device) for k, v in inputs.items()})
                target = target.to(device)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite BCE")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float("inf"), error_if_nonfinite=True
                )
                optimizer.step()
                schedule.step()
                loss_sum += float(loss) * len(target)
                seen += len(target)
                ys.append(target.detach().cpu().numpy())
                ps.append(logits.detach().sigmoid().cpu().numpy())
                if device == "mps" and step % 20 == 0:
                    peak = max(peak, torch.mps.driver_allocated_memory())
            sync(device)
            train_seconds = time.perf_counter() - tick
            total_training += train_seconds
            row = dict(
                epoch=epoch + 1,
                train_loss=loss_sum / seen,
                training_metrics_basis="online training crops/full records before each optimizer update",
                train_metrics=ranking(np.concatenate(ys), np.concatenate(ps)),
                training_seconds=train_seconds,
                samples=seen,
                examples_per_second=seen / train_seconds,
            )
            if not smoke:
                p, inference_seconds = predict(model, val, device, batch)
                score = ranking(val_meta["y"], p)
                row.update(
                    validation_loss=bce_prob(val_meta["y"], p),
                    validation_metrics=score,
                    validation_prediction_seconds=inference_seconds,
                )
                if score["macro_auroc"] is None:
                    raise ValueError("Undefined validation macro AUROC")
                if score["macro_auroc"] > best:
                    best = score["macro_auroc"]
                    best_epoch = epoch + 1
                    best_p = p.copy()
                    torch.save(
                        {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        folder / "best.pt",
                    )
            history.append(row)
            write_json(folder / "history.json", history)
            print(
                json.dumps(dict(run=number, model=name, seed=seed, stage=stage, **row)), flush=True
            )
        if any(not torch.isfinite(v).all() for v in model.parameters()):
            raise FloatingPointError("Nonfinite final parameters")
        torch.save(
            {k: v.detach().cpu() for k, v in model.state_dict().items()}, folder / "final.pt"
        )
        metadata = dict(
            model=name,
            seed=seed,
            stage=stage,
            device=device,
            parameters=parameters(model),
            best_epoch=best_epoch,
            epochs_completed=len(history),
            training_seconds=total_training,
            history=history,
            maximum_mps_driver_bytes=peak,
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            configuration_hash=sha(study.root / "configs/study.json"),
            source_hash=original_source,
            source_revision=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=study.root, text=True
            ).strip(),
            standardization_sha256=sha(study.root / "private/cache/standardization.json"),
            dataset_manifest_hash=study.get("dataset")["publisher_manifest_sha256"],
            training_membership_hash=study.get("dataset")["standardized_fit_membership_hash"],
            checkpoint_kind="fresh scratch training; no exact continuation claimed",
            final_hash=sha(folder / "final.pt"),
        )
        if not smoke:
            metadata.update(
                best_hash=sha(folder / "best.pt"),
                validation_macro_auroc=best,
                thresholds=thresholds_from_validation(val_meta["y"], best_p),
            )
            np.savez(
                folder / "validation_best.npz",
                y=val_meta["y"],
                p=best_p,
                patient_id=val_meta["patient_id"],
            )
            final_p, _ = predict(model, val, device, batch)
            metadata.update(
                final_validation_metrics=ranking(val_meta["y"], final_p),
                final_thresholds=thresholds_from_validation(val_meta["y"], final_p),
            )
            np.savez(
                folder / "validation_final.npz",
                y=val_meta["y"],
                p=final_p,
                patient_id=val_meta["patient_id"],
            )
        if source_hash(study.root) != original_source:
            raise PermissionError("Source/config changed during fit")
        write_json(folder / "completion.json", metadata)
        study.finish(number, "successful", metadata)
        write_json(
            study.root
            / "results"
            / ("baseline" if name in ("xresnet101", "flat_cnn") else "multiview")
            / (number + ".json"),
            metadata,
        )
        if device == "mps":
            torch.mps.empty_cache()
        return number
    except BaseException as error:
        info = dict(failure_type=type(error).__name__, failure_reason=str(error), device=device)
        write_json(folder / "failure.json", info)
        study.finish(number, "failed", info)
        raise
