"""Explicit metadata-only audit command; no training, inference or phase grant."""

import argparse
import json
from pathlib import Path

from dmvmat_ecg.phase2b.contract import digest
from dmvmat_ecg.phase2b.ledger import atomic_bytes
from dmvmat_ecg.phase2b.preflight import build_manifest, load_metadata, scientific_review_flags


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-only", action="store_true", required=True)
    args = parser.parse_args()
    assert args.metadata_only
    root = Path(__file__).resolve().parents[1]
    rows, provenance = load_metadata(root)
    manifest = build_manifest(rows, provenance)
    if manifest != build_manifest(rows, provenance):
        raise RuntimeError("Nondeterministic manifest")
    h = digest(manifest)
    path = root / "private/phase2b_internal/preflight" / f"{h}.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise RuntimeError("Existing artifact differs")
    else:
        atomic_bytes(path, (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode())
    # Aggregate-only output; this command never grants development/evaluation.
    print(
        json.dumps(
            dict(
                manifest_hash=h,
                population=manifest["population"],
                summary=manifest["summary"],
                blockers=manifest["blockers"],
                scientific_review_flags=scientific_review_flags(manifest),
                execution_permission="DENY",
            ),
            sort_keys=True,
        )
    )
    return 2 if manifest["blockers"] or scientific_review_flags(manifest) else 0


if __name__ == "__main__":
    raise SystemExit(main())
