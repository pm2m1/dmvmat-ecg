"""Read-only verification of the original complete prediction seal."""

from dmvmat_ecg.phase3b.jobs import file_hash
from dmvmat_ecg.phase3b.protocol import digest, roster
from dmvmat_ecg.phase3b_eval.access import ROOT, read_json, verify

SEAL = "b7c7e795b9641979858c5050bd6f7db9c2f1340a1f466b4828a0e4088eec5cde"


def verify_unchanged(root=ROOT):
    lock = verify(root)
    study = root / "private/phase3b_cpsc_v1"
    seal = read_json(study / "evaluation_v1/PREDICTION_SEAL.json")
    if seal["seal_hash"] != SEAL or digest(seal["payload"]) != SEAL:
        raise PermissionError("Original sealed prediction set differs")
    items = seal["payload"]["predictions"]
    if set(items) != set(roster()) or len(items) != 138:
        raise PermissionError("Registered roster differs")
    for item in items.values():
        if file_hash(item["path"]) != item["sha"]:
            raise PermissionError("Original sealed predictions differ")
    if read_json(study / "O_OUTCOME_ACCESS_V1.json")["outcome_access_count"] != 1:
        raise PermissionError("First consumed access receipt must remain present")
    if (study / "O_OUTCOME_ACCESS_V2.json").exists():
        raise PermissionError("Corrected outcome access is already consumed")
    return dict(
        lock_hash=lock["lock_hash"],
        prediction_seal_hash=SEAL,
        frozen_artifacts_verified=len(lock["bundle"]["artifacts"]),
        prediction_hashes_verified=len(items),
        O_outcome_accesses=1,
    )
