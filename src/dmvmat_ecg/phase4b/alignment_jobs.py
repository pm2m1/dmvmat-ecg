"""Bounded six-candidate alignment grid plus one exact source-fit repeat."""

import argparse
import concurrent.futures
import json
import multiprocessing
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from dmvmat_ecg.phase4.safety import new_json

from .common import RUN, load_source, setup
from .features import Features
from .source_data import SourceData
from .train_alignment import source_fit


def baseline():
    target = setup()
    data = SourceData()
    features = Features()
    p = RUN / "ORIGINAL_FOLD9_BASELINES.json"
    if p.exists():
        return json.loads(p.read_text())
    torch.set_num_threads(4)
    vv = features.batch("source_stop", np.arange(len(data.y["source_stop"])))
    result = {}
    for s in (15101, 15102, 15103):
        model, _, saved, _ = load_source(target, s)
        with torch.no_grad():
            probabilities = model(**vv).sigmoid().numpy()
        aps = average_precision_score(data.y["source_stop"], probabilities, average=None)
        aucs = roc_auc_score(data.y["source_stop"], probabilities, average=None)
        expected = saved["source_stop_per_label"]
        if any(
            abs(float(aps[j]) - expected[j]["AP"]) > 1e-12
            or abs(float(aucs[j]) - expected[j]["AUROC"]) > 1e-12
            for j in range(2)
        ):
            raise RuntimeError("ORIGINAL_FOLD9_BASELINE_RECONCILIATION_FAILED")
        result[str(s)] = dict(AP=aps.tolist(), AUROC=aucs.tolist(), macro_AUROC=float(aucs.mean()))
    new_json(p, result)
    print("ORIGINAL_FOLD9_BASELINES_RECONCILED", flush=True)
    return result


def gate():
    setup()
    p = RUN / "ALIGNMENT_REPRODUCIBILITY.json"
    if p.exists():
        if json.loads(p.read_text())["status"] != "PASS":
            raise PermissionError("Prior source repeat did not pass")
        return
    baseline()
    # Simultaneous independent CPU processes preserve identical four-thread
    # kernels and provide a strong source-training reproducibility check.
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        a = executor.submit(source_fit, 15101, 0.1, retain=True)
        b = executor.submit(source_fit, 15101, 0.1, retain=False)
        retained, repeat = a.result(), b.result()
    for key in ("state_hash", "history_sha256", "fold9_predictions_sha256", "selected_epoch"):
        if retained[key] != repeat[key]:
            raise RuntimeError("STOP_ALIGNED_SOURCE_REPRODUCTION:" + key)
    new_json(
        p,
        dict(
            status="PASS",
            source_seed=15101,
            lambda_value=0.1,
            repeated_source_seconds=repeat["runtime_seconds"],
            source_state_and_history_bitwise_identical=True,
            fold9_predictions_bitwise_identical=True,
        ),
    )
    print("ALIGNED_SOURCE_REPEAT_GATE_PASS", flush=True)


def grid():
    setup()
    if json.loads((RUN / "ALIGNMENT_REPRODUCIBILITY.json").read_text())["status"] != "PASS":
        raise PermissionError("Full source repeat must pass first")
    jobs = [(s, lam) for lam in (0.1, 1.0) for s in (15101, 15102, 15103)]
    start = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=3, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        futures = [executor.submit(source_fit, s, lam, retain=True) for s, lam in jobs]
        rows = [f.result() for f in futures]
    p = RUN / "ALIGNMENT_GRID_COMPLETE.json"
    if not p.exists():
        new_json(
            p,
            dict(
                status="COMPLETE",
                source_candidates=6,
                wall_seconds=time.monotonic() - start,
                candidates=[
                    dict(
                        seed=r["source_seed"],
                        lambda_value=r["lambda_value"],
                        checkpoint_sha256=r["sha256"],
                        fold9_macro_AUROC=r["fold9_macro_AUROC"],
                    )
                    for r in rows
                ],
            ),
        )
    print("ALIGNED_SOURCE_GRID_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--grid", action="store_true")
    args = parser.parse_args()
    if args.gate:
        gate()
    if args.grid:
        grid()
