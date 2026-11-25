"""Fixed validation-only full FT, no target checkpoints or test capability."""

import concurrent.futures
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np
import torch

from dmvmat_ecg.phase3b.protocol import seed
from dmvmat_ecg.phase4.safety import digest, new_json, sha
from dmvmat_ecg.phase4b.common import Target as CPSC
from dmvmat_ecg.phase4b.stage_lock import stage_lock
from dmvmat_ecg.phase4c_placebo.target import full_fit

from .common import ARMS, BUDGETS, DRAWS, PLACEBO, PLAIN, ROOT, RUN, SEEDS, Scope, bind
from .georgia import Target as Georgia
from .sources import descriptor


class ValidationScope(Scope):
    def check(self, event, args):
        if (
            self.mode == "CPSC"
            and event == "open"
            and args
            and isinstance(args[0], (str, bytes, os.PathLike))
        ):
            p = Path(os.fsdecode(args[0])).absolute()
            if p in {
                PLAIN / "FROZEN_VALIDATION_REFERENCES.json",
                PLACEBO / "FROZEN_VALIDATION_REFERENCES.json",
            }:
                flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
                if (
                    flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
                    or p.resolve() != p
                ):
                    raise PermissionError("Frozen validation bindings are read-only")
                return
        return super().check(event, args)


def target(hospital):
    if hospital == "CPSC":
        return CPSC()
    if hospital == "Georgia":
        return Georgia()
    raise ValueError("Only registered validation hospitals")


def binding():
    paths = [
        Path(__file__),
        Path(__file__).with_name("common.py"),
        Path(__file__).with_name("georgia.py"),
        ROOT / "src/dmvmat_ecg/phase4c_placebo/target.py",
        ROOT / "src/dmvmat_ecg/phase4b/common.py",
        ROOT / "src/dmvmat_ecg/phase2/adaptation.py",
        ROOT / "src/dmvmat_ecg/phase3b/models.py",
        ROOT / "src/dmvmat_ecg/phase3b/protocol.py",
        ROOT / "src/dmvmat_ecg/phase3b/roles.py",
        ROOT / "src/dmvmat_ecg/preprocessing/signals.py",
        ROOT / "uv.lock",
    ]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def filename(arm, s, d, b):
    return f"{arm}_s{s}_d{d}_b{b}"


def historical(t, arm, s, d, b, source_sha):
    folder, stem = (PLAIN, "retrain") if arm == "plain" else (PLACEBO, "placebo")
    name = f"{stem}_full_s{s}_d{d}_b{b}"
    path = folder / "fits" / (name + ".json")
    pred = folder / "predictions" / (name + ".npz")
    r = json.loads(path.read_text())
    frozen = json.loads((folder / "FROZEN_VALIDATION_REFERENCES.json").read_text())
    matched = [v for v in frozen["references"] if v["report"] == name + ".json"]
    if (
        len(matched) != 1
        or matched[0]["report_sha256"] != sha(path)
        or frozen["bundle_sha256"] != t.identity
        or frozen["split_sha256"] != digest(t.split)
    ):
        raise PermissionError("Original frozen validation binding differs")
    for k, v in dict(
        status="COMPLETE",
        method=stem + "_full",
        source_seed=s,
        draw_seed=d,
        budget=b,
        source_sha256=source_sha,
        updates=200,
        mask_index=127,
        fit_seed=seed([78501, s, d, b]),
    ).items():
        if r[k] != v:
            raise PermissionError("Historical CPSC fit recipe differs:" + k)
    old = ROOT / f"src/dmvmat_ecg/phase4c_{'retrain' if arm == 'plain' else 'placebo'}/target.py"
    expected = digest(
        dict(
            bundle=t.identity, split=t.split, s=s, d=d, b=b, source_sha256=source_sha, code=sha(old)
        )
    )
    if r["identity"] != expected or sha(pred) != r["predictions_sha256"]:
        raise PermissionError("Historical CPSC reference integrity differs")
    with np.load(pred, allow_pickle=False) as a:
        p = {k: a[k] for k in ("A", "B")}
    validate(p)
    return p, dict(
        report=str(path.relative_to(ROOT)),
        report_sha256=sha(path),
        predictions_sha256=sha(pred),
        state_hash=r["state_hash"],
        normalization=r["normalization"],
    )


def validate(p):
    if set(p) != {"A", "B"} or any(
        a.shape != (500, 2)
        or a.dtype != np.float32
        or not np.isfinite(a).all()
        or ((a < 0) | (a > 1)).any()
        for a in p.values()
    ):
        raise PermissionError("Invalid validation-only prediction arrays")


def fit(job):
    hospital, arm, s, d, b = job
    ValidationScope(hospital).install()
    torch.set_num_threads(4)
    if arm not in ARMS or s not in SEEDS or d not in DRAWS or b not in BUDGETS:
        raise PermissionError("Unregistered target job")
    t = target(hospital)
    model, norm, source_report, source = descriptor(arm, s)
    folder = RUN / hospital
    name = filename(arm, s, d, b)
    path = folder / "fits" / (name + ".json")
    pred = folder / "predictions" / (name + ".npz")
    expected = digest(
        dict(
            hospital=hospital,
            bundle=t.identity,
            split=t.split,
            arm=arm,
            s=s,
            d=d,
            b=b,
            source=source["sha256"],
            code=binding(),
        )
    )
    if path.exists():
        r = json.loads(path.read_text())
        if (
            r["identity"] != expected
            or sha(pred) != r["predictions_sha256"]
            or r["status"] != "COMPLETE"
        ):
            raise PermissionError("Target resume binding differs")
        return name
    started = folder / "started" / (name + ".json")
    if pred.exists() or started.exists():
        raise FileExistsError("Partial target preserved, no overwrite")
    new_json(started, dict(identity=expected))
    if hospital == "CPSC" and s <= 15103:
        p, old = historical(t, arm, s, d, b, source["sha256"])
        r = dict(
            method=arm + "_full",
            source_seed=s,
            draw_seed=d,
            budget=b,
            updates=200,
            mask_index=127,
            fit_seed=seed([78501, s, d, b]),
            source_sha256=source["sha256"],
            normalization=norm.__dict__,
            state_hash=old["state_hash"],
            reused_historical_validation=True,
            historical=old,
            runtime_seconds=0.0,
        )
    else:
        _, p, r = full_fit(t, s, d, b, model, norm, source["sha256"])
        r.update(method=arm + "_full", reused_historical_validation=False)
    validate(p)
    pred.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(pred, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as out:
        np.savez_compressed(out, **p)
        out.flush()
        os.fsync(out.fileno())
    r.update(
        identity=expected,
        status="COMPLETE",
        hospital=hospital,
        arm=arm,
        file=name,
        predictions_sha256=sha(pred),
        bundle_sha256=t.identity,
        split_sha256=digest(t.split),
        code_binding_sha256=digest(binding()),
    )
    new_json(path, r)
    return name


def read(t, hospital, arm, s, d, b):
    name = filename(arm, s, d, b)
    folder = RUN / hospital
    r = json.loads((folder / "fits" / (name + ".json")).read_text())
    ppath = folder / "predictions" / (name + ".npz")
    if (
        r["bundle_sha256"] != t.identity
        or r["split_sha256"] != digest(t.split)
        or sha(ppath) != r["predictions_sha256"]
    ):
        raise PermissionError("Validation analysis input differs")
    if r["source_sha256"] != descriptor(arm, s)[3]["sha256"]:
        raise PermissionError("Analysis source family differs")
    with np.load(ppath, allow_pickle=False) as a:
        p = {k: a[k] for k in ("A", "B")}
    validate(p)
    return p, r


def run(hospital):
    ValidationScope(hospital).install()
    folder = RUN / hospital
    folder.mkdir(mode=0o700, exist_ok=True)
    bind(folder / "TARGET_CODE_BINDING.json", binding())
    with stage_lock(folder, "baselines"):
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=3, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = [
                pool.submit(fit, (hospital, a, s, d, b))
                for s in SEEDS
                for a in ARMS
                for b in BUDGETS
                for d in DRAWS
            ]
            for i, f in enumerate(concurrent.futures.as_completed(futures), 1):
                print("PHASE5_TARGET", hospital, i, "/180", f.result(), flush=True)
        bind(
            folder / "TARGETS_COMPLETE.json",
            dict(
                status="COMPLETE",
                fits=180,
                target_checkpoints=0,
                source_seeds=list(SEEDS),
                test_accesses=0,
            ),
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("hospital", choices=("CPSC", "Georgia"))
    run(parser.parse_args().hospital)
