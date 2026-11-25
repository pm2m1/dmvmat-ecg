"""Replay only adaptation-pool fits; return fitted states through memory IPC."""

import io
import json

import numpy as np
import torch

from dmvmat_ecg.phase2.adaptation import state_hash
from dmvmat_ecg.phase4.safety import digest, sha
from dmvmat_ecg.phase4b.common import Target, load_source
from dmvmat_ecg.phase4b.target_fit import key
from dmvmat_ecg.phase4c_placebo.target import full_fit
from dmvmat_ecg.phase4c_placebo.target import load as load_placebo
from dmvmat_ecg.phase4c_placebo.target import name as placebo_name
from dmvmat_ecg.phase4c_retrain.target import load as load_plain
from dmvmat_ecg.phase4c_retrain.target import name as plain_name

from .contract import BUDGETS, DRAWS, SOURCES, authenticate
from .scope import OLD, PLACEBO, PLAIN, Scope


def source(t, method, seed):
    if method in ("M1", "M2", "M5"):
        m, n, _, h = load_source(t, seed, "lambda0.1" if method == "M5" else None)
    else:
        m, n, h = (load_plain if method == "M3" else load_placebo)(seed, t)
    if n.__dict__ != t.sources[str(seed)]["normalization"]:
        raise PermissionError("Source normalization differs")
    return m, n, h


def reference(t, method, s, d, b):
    if method == "M2":
        folder, filename = OLD, key(s, d, b, "oracle" if b == 64 else "full")
        expected_method, aligned = ("oracle" if b == 64 else "full"), None
    elif method == "M5":
        folder, filename = OLD, key(s, d, b, "full", aligned="lambda0.1")
        expected_method, aligned = "full", "lambda0.1"
    else:
        folder, filename = (
            (PLAIN, plain_name(s, d, b)) if method == "M3" else (PLACEBO, placebo_name(s, d, b))
        )
        expected_method, aligned = ("retrain_full" if method == "M3" else "placebo_full"), None
    rp, pp = folder / "fits" / (filename + ".json"), folder / "predictions" / (filename + ".npz")
    r = json.loads(rp.read_text())
    if any(
        r[k] != v
        for k, v in dict(
            status="COMPLETE",
            source_seed=s,
            draw_seed=d,
            budget=b,
            updates=200,
            mask_index=127,
            method=expected_method,
        ).items()
    ):
        raise PermissionError("Wrong frozen validation reference")
    if method in ("M2", "M5") and r["aligned"] != aligned:
        raise PermissionError("Wrong source family")
    if sha(pp) != r["predictions_sha256"]:
        raise PermissionError("Validation prediction hash differs")
    with np.load(pp, allow_pickle=False) as a:
        if set(a.files) != {"A", "B"}:
            raise PermissionError("Unexpected reference arrays")
        arrays = {k: a[k] for k in ("A", "B")}
    if any(
        a.shape != (500, 2)
        or a.dtype != np.float32
        or not np.isfinite(a).all()
        or ((a < 0) | (a > 1)).any()
        for a in arrays.values()
    ):
        raise PermissionError("Invalid validation reference arrays")
    if method in ("M2", "M5"):
        identity = digest(
            dict(
                bundle=t.identity,
                split=t.split,
                s=s,
                d=d,
                budget=b,
                method=expected_method,
                mask=127,
                alpha=0.0,
                aligned=aligned,
            )
        )
    else:
        from .contract import ROOT

        package = "phase4c_retrain" if method == "M3" else "phase4c_placebo"
        identity = digest(
            dict(
                bundle=t.identity,
                split=t.split,
                s=s,
                d=d,
                b=b,
                source_sha256=r["source_sha256"],
                code=sha(ROOT / f"src/dmvmat_ecg/{package}/target.py"),
            )
        )
    if r["identity"] != identity:
        raise PermissionError("Reference membership/code identity differs")
    return (
        r,
        arrays,
        dict(
            report_sha256=sha(rp),
            predictions_sha256=sha(pp),
            reference=str(rp.relative_to(folder)),
            origin=folder.name,
        ),
    )


def preflight():
    t = Target()
    for folder in (PLAIN, PLACEBO):
        for filename in ("SOURCE_CODE_BINDING.json", "TARGET_CODE_BINDING.json"):
            from .contract import ROOT

            for path, h in json.loads((folder / filename).read_text()).items():
                if sha(ROOT / path) != h:
                    raise PermissionError("Prior numerical implementation changed:" + path)
        frozen = json.loads((folder / "FROZEN_VALIDATION_REFERENCES.json").read_text())
        if frozen["bundle_sha256"] != t.identity or frozen["split_sha256"] != digest(t.split):
            raise PermissionError("Frozen reference split differs")
        if len(frozen["references"]) != 27:
            raise PermissionError("Incomplete frozen references")
        for r in frozen["references"]:
            if sha(folder / "fits" / r["report"]) != r["report_sha256"]:
                raise PermissionError("Frozen reference report differs")
    source_rows = {}
    for m in ("M2", "M3", "M4", "M5"):
        for s in SOURCES:
            model, n, h = source(t, m, s)
            source_rows[f"{m}/s{s}"] = dict(
                source_sha256=h, state_hash=state_hash(model), normalization=n.__dict__
            )
        for b in BUDGETS:
            for s in SOURCES:
                for d in DRAWS:
                    r, _, _ = reference(t, m, s, d, b)
                    if r["source_sha256"] != source_rows[f"{m}/s{s}"]["source_sha256"]:
                        raise PermissionError("Reference source file differs")
    return dict(
        bundle_sha256=t.identity,
        split_sha256=digest(t.split),
        sources=source_rows,
        reference_pairs=108,
    )


def fit(job):
    scope = Scope()
    scope.install()
    torch.set_num_threads(4)
    method, s, d, b = job
    t = Target()
    old, arrays, ref = reference(t, method, s, d, b)
    model, n, h = source(t, method, s)
    fitted, p, now = full_fit(t, s, d, b, model, n, h)
    authenticate(old, arrays, now, p, n.__dict__)
    stream = io.BytesIO()
    # In-memory transport only. No adapted checkpoint file is written.
    torch.save(fitted.state_dict(), stream)
    return stream.getvalue(), dict(
        method=method,
        source_seed=s,
        draw_seed=d,
        budget=b,
        A_bitwise_match=True,
        B_bitwise_match=True,
        state_hash=now["state_hash"],
        source_sha256=h,
        history_sha256=digest(now["history"]),
        normalization=n.__dict__,
        **ref,
    )


def original_states():
    t = Target()
    rows = {}
    for s in SOURCES:
        model, n, h = source(t, "M1", s)
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        rows[f"M1/s{s}"] = (
            stream.getvalue(),
            dict(
                method="M1",
                source_seed=s,
                source_sha256=h,
                state_hash=state_hash(model),
                normalization=n.__dict__,
            ),
        )
    return rows
