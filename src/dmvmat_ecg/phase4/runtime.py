"""Oracle fits read only the isolated bundle; never call the old real executor."""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from dmvmat_ecg.phase2.adaptation import SelectiveModel, masked_bce, state_hash
from dmvmat_ecg.phase2.contracts import MODULES
from dmvmat_ecg.phase3b.models import fresh_model, inputs
from dmvmat_ecg.phase3b.protocol import OPTIMIZER, seed
from dmvmat_ecg.phase3b.roles import Normalization

from .safety import RecordGuard, bundle_only_audit, digest, new_json, sha

ROOT = Path(__file__).resolve().parents[3]
BUNDLE = ROOT / "private/phase4_oracle_bundle"
OUTPUT = ROOT / "private/phase4_oracle_run"


def mask_dict(index):
    if type(index) is not int or not 0 <= index < 128:
        raise ValueError("Seven-bit mask index required")
    return {n: "ADAPT" if int(b) else "SHARE" for n, b in zip(MODULES, f"{index:07b}", strict=True)}


def mask_index(mask):
    if set(mask) != set(MODULES) or set(mask.values()) - {"SHARE", "ADAPT"}:
        raise ValueError("Invalid recorded mask")
    return int("".join("1" if mask[n] == "ADAPT" else "0" for n in MODULES), 2)


class Bundle:
    def __init__(self, path=BUNDLE):
        self.path = Path(path).resolve()
        self.manifest = json.loads((self.path / "MANIFEST.json").read_text())
        if self.manifest["schema"] != "phase4-adaptation-only-bundle-v1":
            raise PermissionError("Wrong bundle")
        for name, expected in self.manifest["files"].items():
            p = self.path / name
            if p.is_symlink() or not p.resolve().is_relative_to(self.path) or sha(p) != expected:
                raise PermissionError("Bundle file identity mismatch")
        self.identity = sha(self.path / "MANIFEST.json")
        self.records = json.loads((self.path / "records.json").read_text())
        excluded = json.loads((self.path / "exclusion_hashes.json").read_text())
        self.guard = RecordGuard(self.records, excluded)
        self.index = {rid: i for i, rid in enumerate(self.records)}
        self.y = np.load(self.path / "labels.npy", allow_pickle=False)
        self.x = np.load(self.path / "signals.npy", mmap_mode="r", allow_pickle=False)
        if self.y.shape != (3859, 2) or self.x.shape != (3859, 12, 1000):
            raise ValueError("Pool shape mismatch")
        self.episodes = json.loads((self.path / "episodes.json").read_text())
        self.sources = json.loads((self.path / "sources.json").read_text())
        self.references = json.loads((self.path / "reference_fits.json").read_text())
        self.recorded_masks = json.loads((self.path / "recorded_masks.json").read_text())

    def indices(self, records):
        self.guard.check(records)
        return np.array([self.index[r] for r in records], dtype=int)


def prepare_split(bundle, output=OUTPUT):
    output = Path(output)
    path = output / "split.json"
    if path.exists():
        split = json.loads(path.read_text())
        if split["bundle_sha256"] != bundle.identity:
            raise PermissionError("Existing split belongs to another bundle")
        bundle.guard.check(split["A"] + split["B"])
        return split
    used = {r for ep in bundle.episodes.values() for role in ("F", "G") for r in ep[role]}
    eligible = [r for r in bundle.records if r not in used]
    ix = bundle.indices(eligible)
    strata = bundle.y[ix, 0].astype(int) * 2 + bundle.y[ix, 1].astype(int)
    selected, _ = train_test_split(
        np.arange(len(eligible)), train_size=1000, random_state=20261121, stratify=strata
    )
    a, b = train_test_split(
        selected, train_size=500, random_state=20261122, stratify=strata[selected]
    )
    a, b = [eligible[i] for i in a], [eligible[i] for i in b]
    if set(a) & set(b) or (set(a) | set(b)) & used or len(a) != 500 or len(b) != 500:
        raise PermissionError("Invalid validation partition")
    bundle.guard.check(a + b)
    split = dict(
        bundle_sha256=bundle.identity,
        A=a,
        B=b,
        eligible_pool_records=len(eligible),
        excluded_all_budget_records=len(used),
        selection_seed=20261121,
        split_seed=20261122,
    )
    new_json(path, split)
    return split


def split_summary(bundle, split):
    out = dict(
        eligible_pool_records=split["eligible_pool_records"],
        excluded_all_budget_records=split["excluded_all_budget_records"],
        validation_overlap_with_label_draws=0,
        validation_overlap_with_test_ids=0,
        overlap_A_B=0,
        bundle_sha256=bundle.identity,
    )
    for name in ("A", "B"):
        y = bundle.y[bundle.indices(split[name])]
        out[name] = dict(
            records=len(y),
            positive_counts=y.sum(0).astype(int).tolist(),
            prevalence=y.mean(0).tolist(),
            record_ids_sha256=digest(split[name]),
            combination_counts={
                f"{a}{b}": int(np.all(y == [a, b], axis=1).sum()) for a in (0, 1) for b in (0, 1)
            },
        )
    return out


def fit(bundle, split, source_seed, draw_seed, index):
    started = time.monotonic()
    torch.set_num_threads(4)
    source_entry = bundle.sources[str(source_seed)]
    saved = torch.load(
        bundle.path / source_entry["checkpoint"], map_location="cpu", weights_only=True
    )
    source = fresh_model(source_seed)
    source.load_state_dict(saved["state"], strict=True)
    if state_hash(source) != source_entry["state_hash"]:
        raise PermissionError("Source initialization changed")
    norm = Normalization(**source_entry["normalization"])
    ep = bundle.episodes[f"{draw_seed}/B128"]
    records = ep["F"] + ep["G"]
    ix = bundle.indices(records)
    x, y = bundle.x[ix], bundle.y[ix]
    if len(records) != 128:
        raise ValueError("Only the original 128-label records are permitted")
    controller = SelectiveModel(source, mask_dict(index)).cpu()
    xx = inputs(controller.model, norm.transform(x))
    yy, mm = torch.tensor(y, dtype=torch.float32), torch.ones(y.shape, dtype=torch.bool)
    optimizer = torch.optim.AdamW(
        [p for p in controller.parameters() if p.requires_grad], **OPTIMIZER
    )
    initial_state = copy.deepcopy(controller.model.state_dict())
    fit_seed = seed([78501, source_seed, draw_seed, 128])
    losses, exposure = [], np.zeros(len(y), dtype=int)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(fit_seed)
        generator = torch.Generator().manual_seed(fit_seed)
        for _ in range(200):
            batch = torch.randint(len(y), (32,), generator=generator)
            exposure += np.bincount(batch.numpy(), minlength=len(y))
            controller.train()
            optimizer.zero_grad(set_to_none=True)
            loss = masked_bce(
                controller(**{k: v[batch] for k, v in xx.items()}), yy[batch], mm[batch]
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            loss.backward()
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in controller.parameters()
            ):
                raise FloatingPointError("Nonfinite gradient")
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in controller.parameters()):
                raise FloatingPointError("Nonfinite parameter")
            controller.verify_frozen()
            losses.append(float(loss.detach()))
    controller.eval()
    drift = {}
    for n, owner in controller.model.selectable_modules().items():
        drift[n] = float(
            np.sqrt(
                sum(
                    float(torch.square(p.detach() - initial_state[f"{n}.{k}"]).sum())
                    for k, p in owner.named_parameters()
                )
            )
        )
    predictions = {}
    for name in ("A", "B"):
        vi = bundle.indices(split[name])
        chunks = []
        with torch.no_grad():
            for start in range(0, len(vi), 128):
                batch = vi[start : start + 128]
                chunks.append(
                    controller(**inputs(controller.model, norm.transform(bundle.x[batch])))
                    .sigmoid()
                    .numpy()
                )
        predictions[name] = np.concatenate(chunks)
    controller.verify_frozen()
    if state_hash(source) != source_entry["state_hash"]:
        raise PermissionError("Source model was changed")
    report = dict(
        source_seed=source_seed,
        draw_seed=draw_seed,
        budget=128,
        mask_index=index,
        mask_bits=f"{index:07b}",
        mask=mask_dict(index),
        fit_seed=fit_seed,
        bundle_sha256=bundle.identity,
        split_sha256=digest(split),
        source_checkpoint_sha256=source_entry["sha256"],
        updates=200,
        batch_size=32,
        optimizer=OPTIMIZER,
        state_hash=state_hash(controller.model),
        drift=drift,
        history=losses,
        final_training_loss=losses[-1],
        fitting_exposures=dict(F=int(exposure[:64].sum()), G=int(exposure[64:].sum())),
        frozen_parameters_and_buffers_unchanged=True,
        runtime_seconds=time.monotonic() - started,
    )
    return predictions, report


def compare_reference(bundle, report):
    key = f"{report['source_seed']}/{report['draw_seed']}"
    refs = bundle.references[key]
    matches = [name for name, row in refs.items() if row["mask"] == report["mask"]]
    for method in matches:
        ref = refs[method]
        for field in ("state_hash", "drift", "final_training_loss"):
            if report[field] != ref[field]:
                raise RuntimeError(f"STOP_ORIGINAL_REPRODUCTION_MISMATCH:{method}:{field}")
    return matches


def reproduce(bundle, split, output=OUTPUT):
    output = Path(output)
    if (output / "REPRODUCIBILITY.json").exists():
        raise FileExistsError("Reproducibility gate already exists; inspect rather than overwrite")
    reports = []
    previous = None
    for repeat in (1, 2):
        predictions, report = fit(bundle, split, 15101, 16101, 127)
        matched = compare_reference(bundle, report)
        if previous is not None:
            if any(not np.array_equal(previous[n], predictions[n]) for n in ("A", "B")):
                raise RuntimeError("STOP_NONDETERMINISTIC_FULL_MASK_PREDICTIONS")
            if (
                reports[0]["state_hash"] != report["state_hash"]
                or reports[0]["history"] != report["history"]
            ):
                raise RuntimeError("STOP_NONDETERMINISTIC_TRAINING")
        previous = predictions
        reports.append(report)
        print(
            f"FULL_MASK_REPRODUCTION_REPEAT={repeat}; ORIGINAL_REFERENCES_MATCHED={len(matched)}",
            flush=True,
        )
    new_json(
        output / "REPRODUCIBILITY.json",
        dict(
            status="PASS",
            predictions_bitwise_identical=True,
            state_and_loss_history_identical=True,
            original_reference_methods_matched=matched,
            repeats=reports,
        ),
    )
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--reproduce", action="store_true")
    args = parser.parse_args()
    bundle_only_audit(ROOT, BUNDLE, OUTPUT)
    bundle = Bundle()
    split = prepare_split(bundle)
    summary = split_summary(bundle, split)
    summary_path = ROOT / "docs/phase4/VALIDATION_SPLIT_SUMMARY.json"
    if not summary_path.exists():
        new_json(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)
    if args.reproduce:
        reproduce(bundle, split)
