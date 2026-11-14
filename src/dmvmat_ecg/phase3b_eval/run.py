"""One locked O pass; no predictive training/calibration/selection entry points."""

import io
import time
from pathlib import Path

import numpy as np
import torch

from dmvmat_ecg.phase2.adaptation import state_hash
from dmvmat_ecg.phase3b.engine import unpack
from dmvmat_ecg.phase3b.jobs import atomic_new, file_hash, save_json
from dmvmat_ecg.phase3b.models import fresh_model, inputs
from dmvmat_ecg.phase3b.roles import Normalization

from .access import ROOT, Session, evaluator_hash, read_json, validate_authorization, verify
from .analysis import analyze
from .provider import OProvider
from .report import report


def execute(root):
    root = Path(root).resolve()
    lock = verify(root)
    study = root / "private/phase3b_cpsc_v1"
    auth = study / "PHASE3C_EVAL_AUTHORIZATION.json"
    authorization = read_json(auth)
    validate_authorization(authorization)
    if (
        authorization["lock_hash"] != lock["lock_hash"]
        or authorization["memberships_hash"] != file_hash(study / "MEMBERSHIPS.json")
        or authorization["frozen_base_code_identity"] != lock["bundle"]["base_code_hash"]
        or authorization["real_implementation_identity"] != lock["bundle"]["implementation_hash"]
    ):
        raise PermissionError("Human evaluation authorization differs")
    identity = dict(
        version="phase3c-eval-v1",
        authorization_hash=file_hash(auth),
        lock_hash=lock["lock_hash"],
        evaluator_hash=evaluator_hash(),
    )
    session = Session(study / "evaluation_v1", identity, anchor=study)
    session.begin()  # Durable fail-stop marker BEFORE the first O signal/header access.
    out = root / "results/phase3b_eval"
    out.mkdir(parents=True, exist_ok=True)
    save_json(
        out / "PRE_ACCESS_INTEGRITY.json",
        dict(
            identity=identity,
            frozen_artifacts_verified=len(lock["bundle"]["artifacts"]),
            attempts=121,
            O_outcome_accesses=0,
        ),
    )
    try:
        start = time.monotonic()
        provider = OProvider(root, session)
        wave, header_summary = provider.signals()
        save_json(out / "HEADER_ONLY_SUMMARY.json", header_summary)
        requested = read_json(study / "O_EVALUATION_REQUEST.json")["prediction_roster"]
        predictors = lock["bundle"]["predictors"]
        torch.set_num_threads(4)
        tensors = {}
        predictions = {}
        physical = {}
        names = {}
        timing = {}
        for index, name in enumerate(requested, 1):
            session.check()
            item = predictors[name]
            key = item["sha"]
            if key not in physical:
                if file_hash(item["artifact"]) != key:
                    raise PermissionError("Frozen predictor artifact differs")
                saved = unpack(item["artifact"])
                model = fresh_model(
                    int(name.split("/")[1] if name.startswith("source/") else name.split("/")[0])
                )
                model.load_state_dict(saved["state"])
                model.eval()
                if (
                    state_hash(model) != item["state_hash"]
                    or saved["normalization"] != item["normalization"]
                ):
                    raise PermissionError("Frozen model/normalization lineage differs")
                normalization = Normalization(**item["normalization"])
                nk = normalization.identity
                if nk not in tensors:
                    tensors[nk] = inputs(model, normalization.transform(wave))
                xx = tensors[nk]
                elapsed = time.monotonic()
                with torch.no_grad():
                    scores = (
                        torch.cat(
                            [
                                model(**{k: v[i : i + 128] for k, v in xx.items()}).sigmoid()
                                for i in range(0, len(wave), 128)
                            ]
                        )
                        .cpu()
                        .numpy()
                    )
                elapsed = time.monotonic() - elapsed
                if (
                    scores.shape != (len(wave), 2)
                    or not np.isfinite(scores).all()
                    or state_hash(model) != item["state_hash"]
                ):
                    raise ValueError("Inference output/state check failed")
                stream = io.BytesIO()
                np.save(stream, scores, allow_pickle=False)
                path = session.folder / "predictions" / (key + ".npy")
                atomic_new(path, stream.getvalue())
                physical[key] = dict(
                    path=str(path),
                    sha=file_hash(path),
                    normalization=item["normalization"],
                    state_hash=item["state_hash"],
                    artifact_sha=key,
                )
                predictions[key] = scores
                timing[key] = dict(
                    seconds=elapsed, records=len(wave), device="cpu", dtype="float32"
                )
            elif (
                physical[key]["normalization"] != item["normalization"]
                or physical[key]["state_hash"] != item["state_hash"]
            ):
                raise PermissionError("Artifact alias has incompatible normalization/state")
            names[name] = key
            print(f"PREDICTION {index}/{len(requested)}", flush=True)
        refs = {name: physical[names[name]] for name in requested}
        seal = session.seal(
            requested,
            refs,
            lineage=dict(
                lock_hash=lock["lock_hash"],
                base_code=lock["bundle"]["base_code_hash"],
                real_code=lock["bundle"]["implementation_hash"],
                evaluator_code=evaluator_hash(),
                mapping_hash=authorization["label_mapping_sha256"],
                physical_inference_count=len(physical),
            ),
            membership_hash=header_summary["eligible_membership_hash"],
        )
        save_json(
            out / "PREDICTION_SEAL_RECEIPT.json",
            dict(
                seal_hash=seal["seal_hash"],
                identity=identity,
                roster=requested,
                logical_predictors=len(refs),
                unique_inference_runs=len(physical),
                prediction_hashes={k: v["sha"] for k, v in refs.items()},
                outcomes_opened=False,
                header_only_summary=header_summary,
                inference_timing=timing,
                pre_outcome_elapsed_seconds=time.monotonic() - start,
            ),
        )
        print("FULL_PREDICTION_SET_SEALED", seal["seal_hash"], flush=True)
        y, observed = provider.outcomes_once()
        result = analyze(y, observed, predictions, names, provider.eligible, session.folder, out)
        report(root, result, y, observed, header_summary, seal["seal_hash"], timing)
        save_json(
            session.folder / "COMPLETE.json",
            dict(
                identity=identity,
                seal_hash=seal["seal_hash"],
                O_outcome_accesses=1,
                attempts=121,
                elapsed_seconds=time.monotonic() - start,
            ),
        )
        print("LOCKED_EVALUATION_COMPLETE", flush=True)
    except BaseException as exc:
        save_json(
            session.folder / "FAILURE.json",
            dict(
                error=type(exc).__name__,
                message=str(exc),
                outcomes_consumed=(study / "O_OUTCOME_ACCESS_V1.json").exists(),
                no_retry_authorized=True,
            ),
        )
        raise


if __name__ == "__main__":
    execute(ROOT)
