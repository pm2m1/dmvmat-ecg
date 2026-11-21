"""A-only selection for stronger target baselines at three fixed budgets."""

import concurrent.futures
import json
import multiprocessing
import time

import numpy as np

from dmvmat_ecg.phase4.safety import new_json

from .common import RUN, setup
from .target_fit import fit, key, save_fit

TARGET = None


def initialize():
    global TARGET
    TARGET = setup()


def job(s, d, budget, method, alpha=0.0):
    r = save_fit(TARGET, s, d, budget, method, alpha=alpha)
    return dict(
        file=r["file"],
        method=method,
        budget=budget,
        source_seed=s,
        draw_seed=d,
        runtime_seconds=r["runtime_seconds"],
    )


def gates(target):
    destination = RUN / "BASELINE_REPRODUCIBILITY.json"
    if destination.exists():
        if json.loads(destination.read_text())["status"] != "PASS":
            raise PermissionError("Prior target-method repeat failed")
        return
    rows = []
    for method, budget, alpha in [
        ("source", 0, 0.0),
        ("adabn", 0, 0.0),
        ("full", 128, 0.0),
        ("adabn_full", 128, 0.0),
        ("lpft", 128, 0.0),
        ("l2sp", 128, 0.001),
    ]:
        original = save_fit(target, 15101, 16101, budget, method, alpha=alpha)
        p, r = fit(target, 15101, 16101, budget, method, alpha=alpha)
        with np.load(
            RUN / "predictions" / (original["file"] + ".npz"), allow_pickle=False
        ) as saved:
            if any(not np.array_equal(p[n], saved[n]) for n in ("A", "B")):
                raise RuntimeError("STOP_BASELINE_REPLAY_PREDICTIONS:" + method)
        if r["state_hash"] != original["state_hash"] or r["history"] != original["history"]:
            raise RuntimeError("STOP_BASELINE_REPLAY_STATE_OR_HISTORY:" + method)
        rows.append(
            dict(
                method=method,
                status="PASS",
                repeat_seconds=r["runtime_seconds"],
                predictions_bitwise_identical=True,
                state_and_history_identical=True,
            )
        )
        print("BASELINE_REPLAY_PASS " + method, flush=True)
    new_json(destination, dict(status="PASS", methods=rows))


def run():
    start = time.monotonic()
    target = setup()
    gates(target)
    jobs = []
    for s in (15101, 15102, 15103):
        for method in ("source", "adabn"):
            jobs.append((s, 16101, 0, method, 0.0))
        for d in (16101, 16102, 16103):
            for b in (64, 128, 256):
                if b != 64:  # B64 full is exactly the existing all-one oracle fit.
                    jobs.append((s, d, b, "full", 0.0))
                for method in ("adabn_full", "lpft"):
                    jobs.append((s, d, b, method, 0.0))
                for alpha in (0.0001, 0.001, 0.01):
                    jobs.append((s, d, b, "l2sp", alpha))
    with (RUN / "baseline_progress.jsonl").open("a", buffering=1) as log:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=3, mp_context=multiprocessing.get_context("spawn"), initializer=initialize
        ) as pool:
            futures = [pool.submit(job, *args) for args in jobs]
            for n, future in enumerate(concurrent.futures.as_completed(futures), 1):
                r = future.result()
                log.write(json.dumps(r) + "\n")
                if n % 12 == 0 or n == len(jobs):
                    print(
                        f"BASELINE_PROGRESS {n}/{len(jobs)} seconds={time.monotonic() - start:.1f}",
                        flush=True,
                    )
    choices = {}
    for b in (64, 128, 256):
        means = {}
        for alpha in (0.0001, 0.001, 0.01):
            rows = [
                json.loads(
                    (RUN / "fits" / (key(s, d, b, "l2sp", alpha=alpha) + ".json")).read_text()
                )
                for s in (15101, 15102, 15103)
                for d in (16101, 16102, 16103)
            ]
            means[str(alpha)] = float(np.mean([r["A_macro_AP"] for r in rows]))
        selected = min((0.0001, 0.001, 0.01), key=lambda a: (-means[str(a)], a))
        choices[str(b)] = dict(alpha=selected, A_means=means, selection_uses_B=False)
    destination = RUN / "ALPHA_SELECTIONS.json"
    if not destination.exists():
        new_json(destination, choices)
    final = RUN / "BASELINES_COMPLETE.json"
    if not final.exists():
        new_json(
            final,
            dict(
                status="COMPLETE",
                logical_jobs=len(jobs),
                wall_seconds=time.monotonic() - start,
                B_labels_read_for_selection=False,
            ),
        )


if __name__ == "__main__":
    run()
