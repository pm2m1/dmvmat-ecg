"""Single controller: authenticate -> signals/prediction seal -> outcomes -> report."""

import concurrent.futures
import io
import multiprocessing
import os
import time
import traceback

import numpy as np
import torch

from dmvmat_ecg.phase2.adaptation import state_hash
from dmvmat_ecg.phase3b.models import fresh_model, inputs
from dmvmat_ecg.phase3b.roles import Normalization
from dmvmat_ecg.phase4.safety import digest, new_json, sha

from .contract import (
    BUDGETS,
    DRAWS,
    ROOT,
    RUN,
    SOURCES,
    consume_labels,
    protocol_identity,
    roster,
    save_array,
    utc,
    verify_manifest,
)
from .provider import Provider
from .replay import fit, original_states, preflight
from .scope import Scope


def code_binding():
    paths = []
    for package in (
        "phase4c_confirm",
        "phase4c_placebo",
        "phase4c_retrain",
        "phase4b",
        "phase4",
        "phase3b",
        "phase3b_eval",
        "phase3b_eval_v2",
        "phase3",
        "phase2",
        "models",
        "preprocessing",
        "evaluation",
    ):
        paths.extend((ROOT / "src/dmvmat_ecg" / package).rglob("*.py"))
    paths.extend((ROOT / "pyproject.toml", ROOT / "uv.lock"))
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))}


def unchanged(binding, protocol):
    if code_binding() != binding or protocol_identity() != protocol:
        raise PermissionError("Code/protocol changed during single operation")


def predict_state(payload, record, signals):
    """CPU inference only. State never leaves RAM; no BN/statistic updates."""
    torch.set_num_threads(4)
    model = fresh_model(record["source_seed"])
    model.load_state_dict(
        torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True), strict=True
    )
    model.eval()
    before = state_hash(model)
    if before != record["state_hash"]:
        raise PermissionError("In-memory authenticated state differs")
    norm = Normalization(**record["normalization"])
    pieces = []
    with torch.no_grad():
        for start in range(0, len(signals), 128):
            batch = norm.transform(signals[start : start + 128])
            pieces.append(model(**inputs(model, batch)).sigmoid().numpy())
    p = np.concatenate(pieces)
    if state_hash(model) != before:
        raise PermissionError("Inference changed authenticated parameters/BN")
    if (
        p.shape != (len(signals), 2)
        or p.dtype != np.float32
        or not np.isfinite(p).all()
        or ((p < 0) | (p > 1)).any()
    ):
        raise ValueError("Invalid prediction probabilities")
    return p


def main():
    os.umask(0o077)
    protocol = protocol_identity()
    for path in (
        ROOT / "docs/phase4/PHASE4C_RESULTS.md",
        ROOT / "figures/phase4/phase4c_macro_ap_vs_label_budget.svg",
    ):
        if path.exists():
            raise FileExistsError("Existing result requires independent review, never overwrite")
    RUN.mkdir(mode=0o700, exist_ok=False)
    scope = Scope()
    scope.install()
    started = time.monotonic()
    stage, matched = "preflight", []
    new_json(
        RUN / "STARTED.json",
        dict(
            timestamp=utc(),
            pid=os.getpid(),
            protocol=protocol,
            purpose="author-requested committed Phase4C continuation",
            single_attempt=True,
        ),
    )
    try:
        binding = code_binding()
        new_json(RUN / "CODE_BINDING.json", binding)
        checked = preflight()
        new_json(RUN / "PREFLIGHT.json", checked)
        print("PROTOCOL_MATCH " + protocol["commit"], flush=True)
        print("NON_TEST_PREFLIGHT_PASS REFERENCES=108 SOURCES=12", flush=True)
        stage = "replay"
        states = original_states()
        jobs = [
            (m, s, d, b)
            for m in ("M2", "M3", "M4", "M5")
            for b in BUDGETS
            for s in SOURCES
            for d in DRAWS
        ]
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=3, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = {pool.submit(fit, job): job for job in jobs}
            for f in concurrent.futures.as_completed(futures):
                try:
                    payload, record = f.result()
                except BaseException:
                    for other in futures:
                        other.cancel()
                    raise
                m, s, d, b = futures[f]
                key = f"{m}/B{b}/s{s}/d{d}"
                states[key] = payload, record
                matched.append(record)
                new_json(RUN / "replay" / (key.replace("/", "_") + ".json"), record)
                print(f"REPLAY_MATCH {len(matched)}/108 {key}", flush=True)
        if len(matched) != 108 or set(states) != set(roster().values()):
            raise PermissionError("Incomplete authenticated state roster")
        unchanged(binding, protocol)
        new_json(
            RUN / "REPLAY_COMPLETE.json",
            dict(
                timestamp=utc(),
                fits=108,
                A_matches=108,
                B_matches=108,
                state_history_recipe_matches=108,
                test_access=False,
                records=matched,
            ),
        )
        print("REPLAY_AUTHENTICATION_PASS FITS=108 A=108 B=108", flush=True)
        stage = "prediction"
        scope.predictions(len(matched))
        provider = Provider(scope)
        signals = provider.signals()
        predictions, descriptors = {}, {}
        order = list(dict.fromkeys(roster().values()))
        for i, key in enumerate(order, 1):
            payload, record = states.pop(key)
            p = predict_state(payload, record, signals)
            filename = "predictions/" + key.replace("/", "_") + ".npy"
            save_array(RUN / filename, p)
            predictions[key] = p
            descriptors[key] = dict(
                file=filename,
                sha256=sha(RUN / filename),
                state_binding=record,
                shape=list(p.shape),
                dtype=str(p.dtype),
            )
            print(f"FROZEN_PREDICTION {i}/111 {key}", flush=True)
        del signals, states
        unchanged(binding, protocol)
        manifest = dict(
            schema="phase4c-frozen-predictions-v1",
            timestamp=utc(),
            protocol=protocol,
            public_code_sha256=digest(binding),
            replay_matches=108,
            test_binding=provider.binding,
            roster=roster(),
            predictions=descriptors,
            preprocessing="original physical mV, first10 seconds,100Hz,12x1000; CPUfloat32,batch128,eval,4threads",
            labels_read=False,
        )
        new_json(RUN / "PREDICTION_MANIFEST.json", manifest)
        new_json(
            RUN / "PREDICTION_SEAL.json",
            dict(
                timestamp=utc(),
                manifest_sha256=sha(RUN / "PREDICTION_MANIFEST.json"),
                manifest_digest=digest(manifest),
            ),
        )
        verify_manifest(RUN)
        manifest_hash = sha(RUN / "PREDICTION_MANIFEST.json")
        print(
            "PREDICTIONS_SEALED "
            + str(RUN / "PREDICTION_MANIFEST.json")
            + " SHA256="
            + manifest_hash,
            flush=True,
        )
        stage = "outcomes_and_analysis"
        y, observed, _ = consume_labels(RUN, provider.outcomes)
        from .report import write
        from .statistics import analyze

        result = analyze(y, observed, predictions, roster(), provider.groups, RUN)
        unchanged(binding, protocol)
        provenance = dict(
            protocol=protocol,
            replay_matches=len(matched),
            manifest_path=str((RUN / "PREDICTION_MANIFEST.json").relative_to(ROOT)),
            manifest_sha256=manifest_hash,
            manifest_timestamp=manifest["timestamp"],
        )
        write(result, provenance)
        new_json(
            RUN / "COMPLETE.json",
            dict(
                timestamp=utc(),
                status="COMPLETE",
                runtime_seconds=time.monotonic() - started,
                protocol=protocol,
                prediction_manifest_sha256=manifest_hash,
                replay_matches=108,
                analysis_sha256=sha(RUN / "ANALYSIS.json"),
                source_retraining=0,
                target_checkpoint_files=0,
            ),
        )
        print("PHASE4C_COMPLETE " + str(ROOT / "docs/phase4/PHASE4C_RESULTS.md"), flush=True)
    except BaseException as error:
        new_json(
            RUN / "FAILED_NO_RETRY.json",
            dict(
                timestamp=utc(),
                stage=stage,
                replay_matches=len(matched),
                type=type(error).__name__,
                issue=str(error),
                traceback=traceback.format_exc(),
                runtime_seconds=time.monotonic() - started,
                test_capability_opened=scope.stage != "replay",
            ),
        )
        print(f"PHASE4C_STOP STAGE={stage} REPLAY_MATCHES={len(matched)}/108", flush=True)
        raise


if __name__ == "__main__":
    main()
