"""Non-O-only validation; missing frozen references cannot be created retroactively."""

import numpy as np

from dmvmat_ecg.phase3.signals import header
from dmvmat_ecg.phase3b.jobs import file_hash, save_json
from dmvmat_ecg.phase3b_eval.access import ROOT, read_json

from .decoder import decode_labels
from .integrity import verify_unchanged


def reference_coverage(required, frozen):
    required, available = set(required), set(frozen)
    if available - required:
        raise PermissionError("Frozen development reference contains non-R records")
    return dict(
        required_records=len(required),
        frozen_reference_records=len(available),
        missing_frozen_references=len(required - available),
        full_coverage=(required == available),
    )


def permitted_r(record, membership):
    if record not in set(membership["R"]) or record in set(membership["O"] + membership["FEAS"]):
        raise PermissionError("Only R headers may enter decoder validation")


def summary(y, observed):
    y, observed = np.asarray(y), np.asarray(observed)
    return dict(
        records=len(y),
        observed_fraction=observed.mean(0).tolist(),
        positives=np.where(observed, y, 0).sum(0).astype(int).tolist(),
    )


def validate_r(root=ROOT):
    before = verify_unchanged(root)
    study = root / "private/phase3b_cpsc_v1"
    contract = read_json(study / "MEMBERSHIPS.json")
    membership = contract["membership"]
    frozen = read_json(study / "TARGET_DEVELOPMENT_METADATA.json")
    cache = read_json(study / "CACHE_RECEIPT.json")
    coverage = reference_coverage(membership["R"], frozen)
    mapping = read_json(root / "results/phase3a/LABEL_MAPPING.json")
    raw = root / "data/raw/cinc2021_v1_0_3"
    if file_hash(raw / "SHA256SUMS.txt") != contract["target_publisher_index_hash"]:
        raise PermissionError("Publisher checksum index differs")
    publisher = {}
    for line in (raw / "SHA256SUMS.txt").read_text().splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            publisher[parts[1].lstrip(" *").removeprefix("./")] = parts[0]
    all_y, all_obs, covered_y, covered_obs, reference_y, reference_obs = [], [], [], [], [], []
    mismatches = legacy_mismatches = cache_mismatches = 0
    for record in membership["R"]:
        permitted_r(record, membership)
        path = raw / (record + ".hea")
        if file_hash(path) != publisher.get(record + ".hea"):
            raise PermissionError("Non-O header checksum differs")
        y, obs = decode_labels(path.read_bytes(), mapping)
        all_y.append(y)
        all_obs.append(obs)
        # This is an additional live parser comparison, not a frozen reference.
        legacy_codes = header(path)["dx"]
        from dmvmat_ecg.phase3.labels import targets

        legacy_y = targets(legacy_codes, mapping)
        legacy_mismatches += int(y != legacy_y or obs != [True, True])
        if record in frozen:
            ref = frozen[record]
            covered_y.append(y)
            covered_obs.append(obs)
            reference_y.append(ref["y"])
            reference_obs.append(ref["observed"])
            mismatches += int(y != ref["y"] or obs != ref["observed"])
            entry = cache["target_annotations"][record]
            if file_hash(entry["path"]) != entry["sha"]:
                raise PermissionError("Frozen annotation cache differs")
            cached = read_json(entry["path"])
            cache_mismatches += int(y != cached["y"] or obs != cached["observed"])
    after = verify_unchanged(root)
    if before != after:
        raise PermissionError("Prediction seal or frozen identity changed during validation")
    passed = coverage["full_coverage"] and mismatches == cache_mismatches == legacy_mismatches == 0
    report = dict(
        status="PASS" if passed else "BLOCKED",
        labels=["CD-proxy", "STTC-proxy"],
        coverage=coverage,
        corrected_decoder_all_R=summary(all_y, all_obs),
        corrected_decoder_frozen_reference_subset=summary(covered_y, covered_obs),
        frozen_development_reference_subset=summary(reference_y, reference_obs),
        frozen_reference_mismatched_records=mismatches,
        frozen_annotation_cache_mismatched_records=cache_mismatches,
        legacy_live_decoder_all_R_mismatched_records=legacy_mismatches,
        live_decoder_comparison_is_not_frozen_reference=True,
        exact_match_to_frozen_reference_for_every_R_record=passed,
        O_read=False,
        prediction_integrity_before=before,
        prediction_integrity_after=after,
    )
    save_json(root / "results/phase3b_eval_v2/DECODER_R_VALIDATION.json", report)
    return report
