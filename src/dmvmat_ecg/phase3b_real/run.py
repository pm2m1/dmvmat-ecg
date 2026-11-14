"""Bounded two-worker development coordinator. No O provider, scoring or evaluation."""

import contextlib
import io
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from dmvmat_ecg.phase3b.engine import unpack
from dmvmat_ecg.phase3b.evidence import RepresentationBank, analyze, extract
from dmvmat_ecg.phase3b.jobs import atomic_new, file_hash, save_json
from dmvmat_ecg.phase3b.models import fresh_model
from dmvmat_ecg.phase3b.protocol import BUDGETS, DRAW_SEEDS, METHODS, SOURCE_SEEDS
from dmvmat_ecg.phase3b.roles import Normalization

from .boundary import implementation_hash
from .engine import Source, fit_target, train_source
from .log import RealLog
from .prepare import location, read_json
from .provider import RealData


def source_worker(root, source_seed):
    folder = location(root)
    with (folder / "logs" / f"source-{source_seed}.log").open("x") as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            data = RealData(root, source_only=True)
            log = RealLog(root)
            source, report = train_source(data, log, f"source/{source_seed}", source_seed)
            source.validate(data)
            bank = extract(source, source.normalization, data, "source_train")
            buffer = io.BytesIO()
            np.savez_compressed(
                buffer,
                **bank.values,
                y=bank.y,
                observed=bank.observed,
                groups=np.asarray(bank.groups, dtype=str),
            )
            bank_path = folder / "source_banks" / f"{source_seed}.npz"
            atomic_new(bank_path, buffer.getvalue())
            record = dict(
                source_seed=source_seed,
                artifact=source.file,
                sha=source.sha,
                state_hash=source.checkpoint_hash,
                normalization=source.normalization.__dict__,
                report=report,
                bank=dict(
                    path=str(bank_path), sha=file_hash(bank_path), data_identity=bank.data_identity
                ),
            )
            save_json(folder / "source_completions" / f"{source_seed}.json", record)
            return {k: record[k] for k in ("source_seed", "report", "sha")}


def load_source(root, source_seed, data):
    entry = read_json(location(root) / "source_completions" / f"{source_seed}.json")
    if file_hash(entry["artifact"]) != entry["sha"]:
        raise PermissionError("Completed source artifact changed")
    saved = unpack(entry["artifact"])
    model = fresh_model(source_seed)
    model.load_state_dict(saved["state"])
    model.eval()
    source = Source(
        model,
        Normalization(**entry["normalization"]),
        entry["state_hash"],
        data.role("source_train").identity,
        data.role("source_stop").identity,
        source_seed=source_seed,
        file=entry["artifact"],
        sha=entry["sha"],
    )
    source.validate(data)
    b = entry["bank"]
    if file_hash(b["path"]) != b["sha"]:
        raise PermissionError("Source evidence bank changed")
    arrays = np.load(b["path"], allow_pickle=False)
    from dmvmat_ecg.phase2.contracts import MODULES

    bank = RepresentationBank(
        {n: arrays[n] for n in MODULES},
        arrays["y"],
        arrays["observed"],
        tuple(arrays["groups"].tolist()),
        "source_train",
        "patient",
        "known",
        b["data_identity"],
        source.checkpoint_hash,
        source.normalization.identity,
    )
    return source, bank


def target_worker(root, source_seed, draw, budget):
    folder = location(root)
    name = f"{source_seed}-{draw}-B{budget}"
    with (folder / "logs" / f"episode-{name}.log").open("x") as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            start = time.monotonic()
            data = RealData(root, draw, budget)
            source, bank = load_source(root, source_seed, data)
            log = RealLog(root)
            target_bank = extract(source, source.normalization, data, "G")
            decision = analyze(bank, target_bank)
            atomic_new(folder / "decisions" / (name + ".json"), decision.payload)
            rows = []
            for method in METHODS:
                _, record = fit_target(
                    data,
                    log,
                    f"{source_seed}/{draw}/B{budget}/{method}",
                    source,
                    method,
                    source_seed=source_seed,
                    draw_seed=draw,
                    budget=budget,
                    decision=decision if method in {"random_a", "selective_a"} else None,
                )
                record.update(source_seed=source_seed, draw_seed=draw, budget=budget)
                save_json(folder / "target_completions" / name / (method + ".json"), record)
                rows.append(record)
                print(method, "alias" if record["alias"] else "fit", record["sha"], flush=True)
            return dict(
                source_seed=source_seed,
                draw_seed=draw,
                budget=budget,
                mask=decision.report["mask"],
                methods=rows,
                evidence_seconds_and_fit_seconds=time.monotonic() - start,
            )


def execute(root):
    root = Path(root).resolve()
    log = RealLog(root)
    if log.summary()["used"] != 0:
        raise PermissionError("This coordinator requires a fresh real ledger; no silent restart")
    context = multiprocessing.get_context("spawn")
    print("DEVELOPMENT_BEGIN", implementation_hash(), flush=True)
    with ProcessPoolExecutor(max_workers=2, mp_context=context) as pool:
        futures = [pool.submit(source_worker, str(root), s) for s in SOURCE_SEEDS]
        try:
            for future in as_completed(futures):
                result = future.result()
                print("SOURCE_COMPLETE", result["source_seed"], flush=True)
        except BaseException:
            for future in futures:
                future.cancel()
            for process in pool._processes.values():
                process.terminate()
            raise
    print("ALL_THREE_SOURCES_COMPLETE", flush=True)
    with ProcessPoolExecutor(max_workers=2, mp_context=context) as pool:
        futures = [
            pool.submit(target_worker, str(root), s, d, b)
            for s in SOURCE_SEEDS
            for d in DRAW_SEEDS
            for b in BUDGETS
        ]
        try:
            for future in as_completed(futures):
                result = future.result()
                print(
                    "EPISODE_COMPLETE",
                    result["source_seed"],
                    result["draw_seed"],
                    result["budget"],
                    "aliases",
                    sum(r["alias"] for r in result["methods"]),
                    flush=True,
                )
        except BaseException:
            for future in futures:
                future.cancel()
            for process in pool._processes.values():
                process.terminate()
            raise
    save_json(
        location(root) / "DEVELOPMENT_FINISHED.json",
        dict(
            implementation_hash=implementation_hash(),
            attempts=log.summary()["used"],
            O_data_accesses=0,
            finished=time.time(),
        ),
    )
    print("DEVELOPMENT_FINISHED", log.summary()["used"], flush=True)


if __name__ == "__main__":
    execute(Path(__file__).resolve().parents[3])
