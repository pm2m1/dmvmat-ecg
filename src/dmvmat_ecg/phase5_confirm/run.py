"""One controller and one outcome attempt: replay, freeze, score, report, stop."""

import concurrent.futures
import multiprocessing
import os
import time
import traceback

import torch

from dmvmat_ecg.phase4.safety import digest, new_json, sha
from dmvmat_ecg.phase4c_confirm.contract import consume_labels, save_array, utc, verify_manifest
from dmvmat_ecg.phase4c_confirm.run import code_binding as earlier_binding
from dmvmat_ecg.phase4c_confirm.run import predict_state
from dmvmat_ecg.phase5.common import ARMS, BUDGETS, DRAWS, SEEDS

from .contract import ROOT, RUN, Scope, protocol_identity, roster
from .provider import Provider
from .replay import job, preflight, source_states


def code_binding():
    result = earlier_binding()
    for folder in ("phase5", "phase5_confirm"):
        result.update(
            {
                str(p.relative_to(ROOT)): sha(p)
                for p in sorted((ROOT / "src/dmvmat_ecg" / folder).glob("*.py"))
            }
        )
    return result


def unchanged(code, protocol):
    if code_binding() != code or protocol_identity() != protocol:
        raise PermissionError("Code/protocol changed during single operation")


def main():
    os.umask(0o077)
    torch.set_num_threads(4)
    protocol = protocol_identity()
    for p in (
        ROOT / "docs/phase5/PHASE5_GEORGIA_RESULTS.md",
        ROOT / "figures/phase5/phase5_georgia_test_per_seed_ap_differences.svg",
    ):
        if p.exists():
            raise FileExistsError("Existing test report/figure preserved")
    RUN.mkdir(mode=0o700, exist_ok=False)
    scope = Scope()
    scope.install()
    started = time.monotonic()
    stage = "preflight"
    matches = []
    new_json(
        RUN / "STARTED.json",
        dict(
            timestamp=utc(),
            pid=os.getpid(),
            protocol=protocol,
            single_attempt=True,
            purpose="author-requested committed Georgia continuation",
        ),
    )
    try:
        code = code_binding()
        new_json(RUN / "CODE_BINDING.json", code)
        checked = preflight()
        new_json(RUN / "PREFLIGHT.json", checked)
        print("GEORGIA_PROTOCOL_MATCH", protocol["commit"], flush=True)
        print("GEORGIA_NON_TEST_PREFLIGHT_PASS SOURCES=20 REFERENCES=180", flush=True)
        states = source_states()
        stage = "replay"
        jobs = [(a, s, d, b) for a in ARMS for b in BUDGETS for s in SEEDS for d in DRAWS]
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=3, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = {pool.submit(job, args): args for args in jobs}
            for f in concurrent.futures.as_completed(futures):
                try:
                    payload, record = f.result()
                except BaseException:
                    for other in futures:
                        other.cancel()
                    raise
                a, s, d, b = futures[f]
                m = "b" if a == "plain" else "c"
                key = f"{m}/B{b}/s{s}/d{d}"
                states[key] = (payload, record)
                matches.append(record)
                new_json(RUN / "replay" / (key.replace("/", "_") + ".json"), record)
                print(f"GEORGIA_REPLAY_MATCH {len(matches)}/180 {key}", flush=True)
        if len(matches) != 180 or set(states) != set(roster().values()):
            raise PermissionError("Incomplete authenticated roster")
        unchanged(code, protocol)
        new_json(
            RUN / "REPLAY_COMPLETE.json",
            dict(
                timestamp=utc(),
                fits=180,
                A_matches=180,
                B_matches=180,
                state_history_recipe_matches=180,
                test_signal_accesses=0,
                test_label_accesses=0,
                records=matches,
            ),
        )
        print("GEORGIA_REPLAY_PASS A=180 B=180 STATES=180", flush=True)
        stage = "signals_and_predictions"
        scope.predictions(len(matches))
        provider = Provider(scope)
        x = provider.signals()
        predictions = {}
        descriptors = {}
        for i, key in enumerate(dict.fromkeys(roster().values()), 1):
            payload, record = states.pop(key)
            p = predict_state(payload, record, x)
            path = "predictions/" + key.replace("/", "_") + ".npy"
            save_array(RUN / path, p)
            predictions[key] = p
            descriptors[key] = dict(
                file=path,
                sha256=sha(RUN / path),
                state_binding=record,
                shape=list(p.shape),
                dtype=str(p.dtype),
            )
            print(f"GEORGIA_FROZEN_PREDICTOR {i}/190 {key}", flush=True)
        del x, states
        unchanged(code, protocol)
        manifest = dict(
            schema="phase5-georgia-frozen-predictions-v1",
            timestamp=utc(),
            protocol=protocol,
            code_binding_sha256=digest(code),
            source_validation_release_sha256=checked["release_sha256"],
            test_binding=provider.binding,
            roster=roster(),
            predictions=descriptors,
            replay_matches=180,
            labels_read=False,
            preprocessing="unchanged physical mV,canonical leads,first10s100Hz12x1000,CPUfloat32,four threads,batch128,eval,no_grad",
        )
        new_json(RUN / "PREDICTION_MANIFEST.json", manifest)
        h = sha(RUN / "PREDICTION_MANIFEST.json")
        new_json(
            RUN / "PREDICTION_SEAL.json",
            dict(timestamp=utc(), manifest_sha256=h, manifest_digest=digest(manifest)),
        )
        verify_manifest(RUN, roster(), 4081)
        print(
            "GEORGIA_PREDICTIONS_SEALED",
            str(RUN / "PREDICTION_MANIFEST.json"),
            "SHA256=" + h,
            flush=True,
        )
        stage = "outcomes_and_scoring"
        y, observed = consume_labels(RUN, provider.outcomes, roster(), 4081)
        from .report import write
        from .statistics import analyze

        result = analyze(y, observed, predictions, roster(), provider.groups, RUN)
        unchanged(code, protocol)
        write(
            result,
            dict(
                protocol=protocol,
                replay_matches=180,
                manifest_path=str((RUN / "PREDICTION_MANIFEST.json").relative_to(ROOT)),
                manifest_sha256=h,
                manifest_timestamp=manifest["timestamp"],
            ),
        )
        new_json(
            RUN / "COMPLETE.json",
            dict(
                timestamp=utc(),
                status="COMPLETE",
                protocol=protocol,
                prediction_manifest_sha256=h,
                analysis_sha256=sha(RUN / "ANALYSIS.json"),
                report_sha256=sha(ROOT / "docs/phase5/PHASE5_GEORGIA_RESULTS.md"),
                runtime_seconds=time.monotonic() - started,
                replay_matches=180,
                physical_predictions=190,
                logical_predictions=270,
                outcome_attempts=1,
                target_checkpoints=0,
                source_retraining=0,
            ),
        )
        print(
            "PHASE5_GEORGIA_COMPLETE",
            str(ROOT / "docs/phase5/PHASE5_GEORGIA_RESULTS.md"),
            flush=True,
        )
    except BaseException as e:
        new_json(
            RUN / "FAILED_NO_RETRY.json",
            dict(
                timestamp=utc(),
                stage=stage,
                type=type(e).__name__,
                issue=str(e),
                traceback=traceback.format_exc(),
                replay_matches=len(matches),
                test_capability_opened=scope.stage != "replay",
                runtime_seconds=time.monotonic() - started,
            ),
        )
        print("PHASE5_GEORGIA_STOP", stage, "REPLAY_MATCHES=" + str(len(matches)), flush=True)
        raise


if __name__ == "__main__":
    main()
