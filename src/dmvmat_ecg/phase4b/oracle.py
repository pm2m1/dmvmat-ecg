"""B64 128-mask grid. Fixed roles, resumable fits, no test/backend access."""

import concurrent.futures
import json
import multiprocessing
import time

import numpy as np

from dmvmat_ecg.phase4.safety import new_json, sha

from .common import RUN, setup
from .target_fit import fit, save_fit

TARGET = None


def initialize():
    global TARGET
    TARGET = setup()


def task(s, d, m):
    r = save_fit(TARGET, s, d, 64, "oracle", mask=m)
    return dict(
        source_seed=s,
        draw_seed=d,
        mask_index=m,
        file=r["file"],
        runtime_seconds=r["runtime_seconds"],
    )


def repeat_gate(target):
    p = RUN / "ORACLE64_REPRODUCIBILITY.json"
    if p.exists():
        if json.loads(p.read_text())["status"] != "PASS":
            raise PermissionError("Prior reproducibility gate failed")
        return
    original = save_fit(target, 15101, 16101, 64, "oracle", mask=127)
    pred, repeat = fit(target, 15101, 16101, 64, "oracle", mask=127)
    with np.load(RUN / "predictions" / (original["file"] + ".npz"), allow_pickle=False) as previous:
        if any(not np.array_equal(pred[name], previous[name]) for name in ("A", "B")):
            raise RuntimeError("STOP_ORACLE64_PREDICTION_REPRODUCTION")
    if repeat["state_hash"] != original["state_hash"] or repeat["history"] != original["history"]:
        raise RuntimeError("STOP_ORACLE64_TRAINING_REPRODUCTION")
    new_json(
        p,
        dict(
            status="PASS",
            repeat_seconds=repeat["runtime_seconds"],
            predictions_bitwise_identical=True,
            state_and_history_identical=True,
        ),
    )


def run():
    start = time.monotonic()
    target = setup()
    identity = dict(
        bundle_sha256=target.identity,
        source_seeds=[15101, 15102, 15103],
        draw_seeds=[16101, 16102, 16103],
        budget=64,
        workers=3,
        threads=4,
        code={
            name: sha(__import__("pathlib").Path(__file__).with_name(name))
            for name in ("common.py", "target_fit.py", "oracle.py")
        },
    )
    contract = RUN / "ORACLE64_IDENTITY.json"
    if contract.exists():
        if json.loads(contract.read_text()) != identity:
            raise PermissionError("Oracle resumption code/data identity mismatch")
    else:
        new_json(contract, identity)
    repeat_gate(target)
    completed = 0
    with (RUN / "oracle64_progress.jsonl").open("a", buffering=1) as log:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=3, mp_context=multiprocessing.get_context("spawn"), initializer=initialize
        ) as executor:
            futures = [
                executor.submit(task, s, d, m)
                for s in (15101, 15102, 15103)
                for d in (16101, 16102, 16103)
                for m in range(128)
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                except BaseException:
                    for other in futures:
                        other.cancel()
                    raise
                log.write(json.dumps(result) + "\n")
                completed += 1
                if completed % 24 == 0:
                    print(
                        f"ORACLE64_PROGRESS {completed}/1152 seconds={time.monotonic() - start:.1f}",
                        flush=True,
                    )
    final = RUN / "ORACLE64_COMPLETE.json"
    if not final.exists():
        new_json(
            final, dict(status="COMPLETE", fits=completed, wall_seconds=time.monotonic() - start)
        )


if __name__ == "__main__":
    run()
