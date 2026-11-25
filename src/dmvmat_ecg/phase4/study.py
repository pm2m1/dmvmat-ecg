"""Resumable 128-mask × 9-episode adaptation-only study; no checkpoints."""

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from .runtime import BUNDLE, OUTPUT, ROOT, Bundle, compare_reference, fit, prepare_split
from .safety import bundle_only_audit, digest, new_json, sha

WORKER_BUNDLE = None
WORKER_SPLIT = None


def initialize():
    global WORKER_BUNDLE, WORKER_SPLIT
    bundle_only_audit(ROOT, BUNDLE, OUTPUT)
    WORKER_BUNDLE = Bundle()
    WORKER_SPLIT = prepare_split(WORKER_BUNDLE)


def paths(s, d, m):
    name = f"s{s}_d{d}_mask{m:03d}"
    return OUTPUT / "fits" / (name + ".json"), OUTPUT / "predictions" / (name + ".npz")


def check_complete(s, d, m, bundle, split):
    report_path, pred_path = paths(s, d, m)
    if not report_path.exists():
        if pred_path.exists():
            raise FileExistsError("Uncommitted prediction artifact; do not overwrite")
        return None
    r = json.loads(report_path.read_text())
    if (
        r["source_seed"] != s
        or r["draw_seed"] != d
        or r["mask_index"] != m
        or r["bundle_sha256"] != bundle.identity
        or r["split_sha256"] != digest(split)
        or sha(pred_path) != r["predictions_sha256"]
        or r["status"] != "COMPLETE"
    ):
        raise PermissionError("Existing fit is incomplete or incompatible")
    arrays = np.load(pred_path, allow_pickle=False)
    if set(arrays.files) != {"A", "B"} or any(arrays[k].shape != (500, 2) for k in ("A", "B")):
        raise PermissionError("Saved validation array schema mismatch")
    compare_reference(bundle, r)
    return r


def task(s, d, m):
    bundle, split = WORKER_BUNDLE, WORKER_SPLIT
    existing = check_complete(s, d, m, bundle, split)
    if existing is not None:
        return dict(source_seed=s, draw_seed=d, mask_index=m, skipped=True)
    predictions, report = fit(bundle, split, s, d, m)
    matched = compare_reference(bundle, report)
    report_path, pred_path = paths(s, d, m)
    for path in (report_path, pred_path):
        path.parent.mkdir(exist_ok=True)
    # np.savez writes only the two validation arrays, never weights/IDs/test data.
    with pred_path.open("xb") as f:
        np.savez_compressed(f, **predictions)
        f.flush()
        os.fsync(f.fileno())
    scores = {}
    for name in ("A", "B"):
        y = bundle.y[bundle.indices(split[name])]
        per_label = average_precision_score(y, predictions[name], average=None)
        scores[name] = dict(
            CD=float(per_label[0]), STTC=float(per_label[1]), macro=float(per_label.mean())
        )
    report.update(
        status="COMPLETE",
        predictions_sha256=sha(pred_path),
        scores=scores,
        original_reference_methods_matched=matched,
    )
    new_json(report_path, report)
    return dict(
        source_seed=s,
        draw_seed=d,
        mask_index=m,
        runtime_seconds=report["runtime_seconds"],
        reference_matches=matched,
        skipped=False,
    )


def verify_parallel_repeats(bundle, split):
    destination = OUTPUT / "PARALLEL_REPRODUCIBILITY.json"
    if destination.exists():
        v = json.loads(destination.read_text())
        if v["status"] != "PASS" or v["bundle_sha256"] != bundle.identity:
            raise PermissionError("Invalid prior parallel verification")
        return v
    jobs = [
        (s, d, m) for s in (15101, 15102, 15103) for d in (16101, 16102, 16103) for m in range(128)
    ]
    chosen = np.random.default_rng(20261123).choice(len(jobs), 3, replace=False)
    records = []
    # Original predictions came from the three-worker run; verify in one serial
    # process with the same original four-thread training setting.
    for i in chosen:
        s, d, m = jobs[int(i)]
        existing = check_complete(s, d, m, bundle, split)
        if existing is None:
            raise PermissionError("Parallel fit missing")
        pred, repeat = fit(bundle, split, s, d, m)
        compare_reference(bundle, repeat)
        with np.load(paths(s, d, m)[1], allow_pickle=False) as saved:
            identical = all(np.array_equal(pred[n], saved[n]) for n in ("A", "B"))
        if (
            not identical
            or repeat["state_hash"] != existing["state_hash"]
            or repeat["history"] != existing["history"]
            or repeat["drift"] != existing["drift"]
        ):
            raise RuntimeError("STOP_PARALLEL_DETERMINISM_MISMATCH")
        records.append(
            dict(
                source_seed=s,
                draw_seed=d,
                mask_index=m,
                runtime_seconds=repeat["runtime_seconds"],
                predictions_identical=True,
                state_hash_identical=True,
                history_identical=True,
            )
        )
    v = dict(status="PASS", bundle_sha256=bundle.identity, seed=20261123, repeats=records)
    new_json(destination, v)
    return v


def run():
    started = time.monotonic()
    bundle_only_audit(ROOT, BUNDLE, OUTPUT)
    bundle = Bundle()
    split = prepare_split(bundle)
    gate = json.loads((OUTPUT / "REPRODUCIBILITY.json").read_text())
    if gate["status"] != "PASS" or not gate["predictions_bitwise_identical"]:
        raise PermissionError("Full-mask reproducibility must pass first")
    identity = dict(
        bundle_sha256=bundle.identity,
        split_sha256=digest(split),
        workers=3,
        threads=4,
        masks=128,
        episodes=9,
        budget=128,
        code_hashes={
            n: sha(Path(__file__).with_name(n)) for n in ("runtime.py", "study.py", "safety.py")
        },
    )
    contract = OUTPUT / "RUN_IDENTITY.json"
    if contract.exists():
        if json.loads(contract.read_text()) != identity:
            raise PermissionError("Resumption code/data/config differs")
    else:
        new_json(contract, identity)
    (OUTPUT / "fits").mkdir(exist_ok=True)
    (OUTPUT / "predictions").mkdir(exist_ok=True)
    jobs = []
    completed = 0
    for s in (15101, 15102, 15103):
        for d in (16101, 16102, 16103):
            for m in range(128):
                if check_complete(s, d, m, bundle, split):
                    completed += 1
                else:
                    jobs.append((s, d, m))
    print(
        f"ORACLE_START complete={completed} remaining={len(jobs)} workers=3 threads_per_worker=4",
        flush=True,
    )
    with (OUTPUT / "progress.jsonl").open("a", buffering=1) as log:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=3, mp_context=multiprocessing.get_context("spawn"), initializer=initialize
        ) as executor:
            futures = {executor.submit(task, *job): job for job in jobs}
            for future in concurrent.futures.as_completed(futures):
                job = futures[future]
                try:
                    record = future.result()
                except BaseException as exc:
                    failure = dict(
                        status="FAILED_STOP",
                        job=job,
                        error_type=type(exc).__name__,
                        error=str(exc),
                        elapsed_seconds=time.monotonic() - started,
                    )
                    log.write(json.dumps(failure) + "\n")
                    for pending in futures:
                        pending.cancel()
                    raise
                completed += 1
                log.write(json.dumps(dict(status="COMPLETE", **record)) + "\n")
                if completed % 12 == 0 or completed == 1152:
                    print(
                        f"ORACLE_PROGRESS {completed}/1152 elapsed_seconds={time.monotonic() - started:.1f}",
                        flush=True,
                    )
    check = verify_parallel_repeats(bundle, split)
    summary = dict(
        status="COMPLETE",
        completed_fits=completed,
        planned_fits=1152,
        workers=3,
        threads_per_worker=4,
        session_elapsed_seconds=time.monotonic() - started,
        parallel_reverification=check,
    )
    if not (OUTPUT / "RUN_COMPLETE.json").exists():
        new_json(OUTPUT / "RUN_COMPLETE.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
