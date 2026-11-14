"""One explicitly authorized corrected outcome read; sealed predictions only."""

import io
import time
from pathlib import Path

import numpy as np

from dmvmat_ecg.phase3b.jobs import atomic_new, file_hash, save_json
from dmvmat_ecg.phase3b.protocol import digest, roster
from dmvmat_ecg.phase3b_eval.access import ROOT, bytes_hash, read_json, verify
from dmvmat_ecg.phase3b_eval.analysis import analyze

from .decoder import decode_labels, validity_gate
from .integrity import SEAL, verify_unchanged
from .report_corrected import report


def correction_code_hash():
    folder = Path(__file__).parent
    files = {p.name: file_hash(p) for p in sorted(folder.glob("*.py"))}
    for name in ("analysis.py", "statistics.py"):
        files["unchanged_v1/" + name] = file_hash(folder.parent / "phase3b_eval" / name)
    files["weighted_ranking"] = file_hash(folder.parent / "evaluation/bootstrap.py")
    return digest(files)


def sealed_check(root):
    verify(root)
    study = root / "private/phase3b_cpsc_v1"
    sealed = read_json(study / "evaluation_v1/PREDICTION_SEAL.json")
    if sealed["seal_hash"] != SEAL or digest(sealed["payload"]) != SEAL:
        raise PermissionError("Original prediction seal differs")
    if set(sealed["payload"]["predictions"]) != set(roster()):
        raise PermissionError("Original logical prediction roster differs")
    for item in sealed["payload"]["predictions"].values():
        if file_hash(item["path"]) != item["sha"]:
            raise PermissionError("Original sealed prediction hash differs")
    return sealed


def reserve_corrected_access(anchor, receipt):
    """Sticky, exclusive reservation before any outcome read, including crashes."""
    anchor = Path(anchor)
    if not (anchor / "O_OUTCOME_ACCESS_V1.json").exists():
        raise PermissionError("Preserve the original consumed access marker")
    save_json(anchor / "O_OUTCOME_ACCESS_V2.json", receipt)


def gate_before_metrics(y, observed, metric_callback):
    gate = validity_gate(y, observed)
    if not gate["passed"]:
        raise ValueError("Corrected outcome validity gate failed; no metrics permitted")
    return metric_callback()


def execute(root=ROOT):
    root = Path(root).resolve()
    if root != ROOT:
        raise PermissionError("Real correction requires the canonical workspace")
    verify_unchanged(root)  # Includes absence of the V2 access marker.
    sealed = sealed_check(root)
    study = root / "private/phase3b_cpsc_v1"
    auth_path = study / "OUTCOME_CORRECTION_AUTHORIZATION_V2.json"
    revision_path = study / "STEP3_REVISION_AUTHORIZATION_V2.json"
    auth, revision = read_json(auth_path), read_json(revision_path)
    acceptance = read_json(
        root / "results/phase3b_eval_v2/REVISED_NON_O_VALIDATION_ACCEPTANCE.json"
    )
    previous_receipt = read_json(study / "EVALUATION_CORRECTION_V2_STOP_RECEIPT.json")
    reference_path = root / "results/phase3b_eval_v2/DECODER_R_VALIDATION.json"
    validation = read_json(reference_path)
    if (
        auth["prediction_seal_hash"] != SEAL
        or file_hash(auth_path) != previous_receipt["authorization_sha256"]
        or file_hash(revision_path) != acceptance["revision_sha256"]
        or file_hash(reference_path) != acceptance["source_validation_sha256"]
        or len(auth["authorized_operations"]) != 3
        or revision["reference_sha256"] != file_hash(reference_path)
        or validation["coverage"]["frozen_reference_records"] != 711
        or validation["coverage"]["required_records"] != 3859
        or any(
            validation[key] != 0
            for key in (
                "frozen_reference_mismatched_records",
                "frozen_annotation_cache_mismatched_records",
                "legacy_live_decoder_all_R_mismatched_records",
            )
        )
    ):
        raise PermissionError("Human revision or non-O validation differs")
    contract = read_json(study / "MEMBERSHIPS.json")
    header_membership = read_json(study / "evaluation_v1/HEADER_ONLY_MEMBERSHIPS.json")
    eligible = tuple(header_membership["eligible"])
    if (
        digest(list(eligible)) != sealed["payload"]["membership_hash"]
        or not set(eligible).issubset(contract["membership"]["O"])
        or set(eligible).intersection(contract["membership"]["FEAS"])
        or len(eligible) != 3860
    ):
        raise PermissionError("Original prediction/outcome membership differs")
    predictions, names = {}, {}
    for name in sealed["payload"]["roster"]:
        item = sealed["payload"]["predictions"][name]
        key = item["artifact_sha"]
        if key not in predictions:
            p = np.load(item["path"], allow_pickle=False)
            if p.shape != (len(eligible), 2) or not np.isfinite(p).all():
                raise ValueError("Original prediction shape/values differ")
            predictions[key] = p
        names[name] = key
    raw = root / "data/raw/cinc2021_v1_0_3"
    if file_hash(raw / "SHA256SUMS.txt") != contract["target_publisher_index_hash"]:
        raise PermissionError("Publisher index changed")
    publisher = {}
    for line in (raw / "SHA256SUMS.txt").read_text().splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            publisher[parts[1].lstrip(" *").removeprefix("./")] = parts[0]
    folder = study / "evaluation_v2"
    folder.mkdir(exist_ok=False)
    out = root / "results/phase3b_eval_v2"
    blocked_report = root / "docs/phase3/PHASE3B_LOCKED_EVALUATION_RESULTS_V2.md"
    atomic_new(folder / "PRE_ACCESS_BLOCKED_REPORT.md", blocked_report.read_bytes())
    identity = dict(
        version="phase3b-corrected-outcomes-v2",
        seal_hash=SEAL,
        authorization_sha256=file_hash(auth_path),
        step3_revision_sha256=file_hash(revision_path),
        non_O_validation_sha256=file_hash(reference_path),
        correction_code_hash=correction_code_hash(),
        previous_access_reason="V1 decoder defect returned zero observed labels",
        corrected_access_reason="Explicit human-authorized decoder correction",
        outcome_access_count=2,
        corrected_access_count=1,
    )
    start = time.monotonic()
    reserve_corrected_access(study, identity)
    save_json(folder / "OUTCOME_ACCESS_V2.json", identity)
    try:
        frozen = read_json(root / "results/phase3a/LABEL_MAPPING.json")
        y, observed = [], []
        for record in eligible:
            # Exactly one header-byte read per eligible O record in this V2 access.
            payload = (raw / (record + ".hea")).read_bytes()
            if bytes_hash(payload) != publisher.get(record + ".hea"):
                raise PermissionError("Corrected O header publisher checksum differs")
            a, b = decode_labels(payload, frozen)
            y.append(a)
            observed.append(b)
        y, observed = np.asarray(y, float), np.asarray(observed, bool)
        gate = validity_gate(y, observed)  # Before any predictive metric.
        save_json(out / "CORRECTED_OUTCOME_VALIDITY_GATE.json", gate)
        if not gate["passed"]:
            raise ValueError("Corrected O validity gate failed; no metrics computed")
        stream = io.BytesIO()
        np.savez_compressed(stream, y=y, observed=observed, ids=np.asarray(eligible))
        atomic_new(folder / "O_OUTCOMES_V2.npz", stream.getvalue())
        print("CORRECTED_OUTCOME_VALIDITY_GATE_PASSED", flush=True)
        result = gate_before_metrics(
            y,
            observed,
            lambda: analyze(y, observed, predictions, names, eligible, folder, out),
        )
        prior_seal_receipt = read_json(root / "results/phase3b_eval/PREDICTION_SEAL_RECEIPT.json")
        report(
            root,
            result,
            y,
            observed,
            read_json(study / "evaluation_v1/HEADER_ONLY_SUMMARY.json"),
            SEAL,
            prior_seal_receipt["inference_timing"],
        )
        sealed_check(root)
        if identity["correction_code_hash"] != correction_code_hash():
            raise PermissionError("Correction code changed after access")
        save_json(
            folder / "COMPLETE_V2.json",
            dict(
                status="CORRECTED_EVALUATION_COMPLETE",
                identity=identity,
                gate=gate,
                logical_procedures=len(names),
                physical_prediction_arrays=len(predictions),
                new_inference_runs=0,
                new_predictor_fits=0,
                attempts=121,
                elapsed_seconds=time.monotonic() - start,
            ),
        )
        print("LOCKED_EVALUATION_COMPLETE_V2", flush=True)
    except BaseException as exc:
        save_json(folder / "FAILURE_V2.json", dict(error=type(exc).__name__, reason=str(exc)))
        raise


if __name__ == "__main__":
    execute()
