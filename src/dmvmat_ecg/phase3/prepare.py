"""Offline verified-data audit, then one immutable proxy/split preparation.

All-target header labels are used once for coverage and split stratification.
Only FEAS labels are retained in the model-facing cache. RESERVED membership
contains IDs only. No model is constructed by this module.
"""

import ast
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .download import SOURCES, selected_entries, sha
from .labels import mapping, targets
from .signals import header, native_identity, physical_wave, to_100hz_first10
from .splits import DOMAINS, SEED, split


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)


def source_proxy_labels(csv_path, ids, frozen):
    """Decode SCP annotations ONLY after restricting to authorized fold9 IDs.

    The CSV scan necessarily encounters the ID/fold columns of other rows;
    their SCP values are never parsed, returned or used.
    """
    wanted = {int(i) for i in ids}
    found = {}
    with Path(csv_path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            record_id = int(row["ecg_id"])
            if record_id not in wanted:
                continue
            if int(row["strat_fold"]) != 9:
                raise PermissionError("Only source fold9 SCP decoding is permitted")
            if record_id in found:
                raise ValueError("Duplicated source validation annotation ID")
            scp = set(ast.literal_eval(row["scp_codes"]))
            found[record_id] = {
                "proxy": [
                    int(bool(scp & set(frozen["proxy_scp"][label]))) for label in ["CD", "STTC"]
                ],
                "code": [
                    int(bool(scp & set(task["scp"])))
                    for task in frozen["per_code_descriptive_tasks"].values()
                ],
            }
    if set(found) != wanted:
        raise ValueError("Missing or duplicated source validation annotation")
    return (
        np.array([found[int(i)]["proxy"] for i in ids], dtype=np.int8),
        np.array([found[int(i)]["code"] for i in ids], dtype=np.int8),
    )


def prepare(root):
    root = Path(root)
    raw = root / "data/raw/cinc2021_v1_0_3"
    out = root / "results/phase3a"
    cache = root / "data/manifests/phase3a_v1"
    if cache.exists() or (out / "LABEL_MAPPING.json").exists():
        raise FileExistsError("Frozen preparation already exists; no resplitting or overwrite")
    verified = json.loads((raw / "DOWNLOAD_VERIFICATION.json").read_text())
    if not verified["complete"] or verified["shasum_returncode"] != 0:
        raise PermissionError("All five sources must pass publisher checksum verification first")
    entries = selected_entries((raw / "SHA256SUMS.txt").read_text())
    records = sorted(n[:-4] for n in entries if n.endswith(".hea"))
    summaries = {
        s: {
            "records": 0,
            "sampling_rates": Counter(),
            "sample_lengths": Counter(),
            "lead_orders": Counter(),
            "units": Counter(),
            "gain_sets": Counter(),
            "shorter_than_10s": 0,
            "longer_than_10s": 0,
            "flat_lead_records": 0,
            "flat_leads": Counter(),
            "duplicate_removed": 0,
            "duplicate_annotation_disagreements": 0,
        }
        for s in SOURCES
    }
    catalog, seen, failures, observed_codes = [], {}, [], set()
    duplicate_pairs = Counter()
    for index, record_id in enumerate(records):
        source = record_id.split("/")[1]
        summary = summaries[source]
        summary["records"] += 1
        try:
            meta = header(raw / (record_id + ".hea"))
            wave = physical_wave(raw / record_id, meta)
            fingerprint = native_identity(wave, meta["fs"])
            summary["sampling_rates"][str(meta["fs"])] += 1
            summary["sample_lengths"][str((meta["fs"], meta["samples"]))] += 1
            summary["lead_orders"][",".join(meta["lead_names"])] += 1
            summary["units"][",".join(meta["units"])] += 1
            summary["gain_sets"][str(meta["gain"])] += 1
            summary["shorter_than_10s"] += int(meta["duration"] < 10)
            summary["longer_than_10s"] += int(meta["duration"] > 10)
            flat = np.ptp(wave, axis=1) == 0
            summary["flat_lead_records"] += int(flat.any())
            for lead in np.flatnonzero(flat):
                summary["flat_leads"][str(lead)] += 1
            observed_codes.update(meta["dx"])
            if fingerprint in seen:
                previous = seen[fingerprint]
                duplicate_pairs[previous["id"].split("/")[1] + " -> " + source] += 1
                summary["duplicate_removed"] += 1
                summary["duplicate_annotation_disagreements"] += int(
                    set(meta["dx"]) != set(previous["dx"])
                )
                continue
            seen[fingerprint] = {"id": record_id, "dx": meta["dx"]}
            if meta["duration"] >= 10:
                catalog.append({"id": record_id, "source": source, "header": meta})
        except (ValueError, KeyError, OSError) as error:
            failures.append({"id": record_id, "error": str(error)})
        if index % 2000 == 0:
            print(json.dumps({"audit_records": index, "total": len(records)}), flush=True)
    audit = {
        "sources": summaries,
        "duplicates_by_source_pair": dict(duplicate_pairs),
        "failures": failures,
        "native_exact_duplicate_total": len(records) - len(seen) - len(failures),
        "canonical_duplicate_rule": "Lexicographically first full native physical-wave hash, independent of diagnosis; no label merging",
        "duplicate_resolution": "Canonical leads, mV rounded1e-9, original rate and full length; no crop/shift/near-duplicate detection",
        "ptbxl_cross_match": "Not performed, explicitly waived by human; overlap not empirically excluded",
        "publisher_index_sha256": sha(raw / "SHA256SUMS.txt"),
        "patient_identity": "No stable target patient IDs available; record-level independence unverified",
    }
    write_new(out / "DATASET_AUDIT.json", audit)
    if failures:
        raise ValueError("Audit failures require review before mapping/splitting/scoring")
    frozen = mapping(observed_codes)
    write_new(out / "LABEL_MAPPING.json", frozen)
    cache.mkdir(parents=True)
    summary = {}
    for domain, sources in DOMAINS.items():
        rows = [row for row in catalog if row["source"] in sources]
        y = np.array([targets(row["header"]["dx"], frozen) for row in rows], dtype=np.int8)
        manifest = split([row["id"] for row in rows], y, domain)
        write_new(cache / (domain + "_split.json"), manifest)
        lookup = {row["id"]: row for row in rows}
        chosen = [lookup[i] for i in manifest["FEAS"]]
        # RESERVED annotations are not written into model-facing artifacts.
        write_new(cache / (domain + "_FEAS.json"), chosen)
        fy = np.array([targets(row["header"]["dx"], frozen) for row in chosen], dtype=np.int8)
        summary[domain] = {
            "FEAS_records": len(chosen),
            "RESERVED_records": len(manifest["RESERVED"]),
            "FEAS_positive_CD_STTC": fy.sum(0).tolist(),
            "FEAS_prevalence_CD_STTC": fy.mean(0).tolist(),
            "EXPECTED_RESERVED_positive_CD_STTC": (fy.mean(0) * len(manifest["RESERVED"])).tolist(),
            "FEAS_truncated_first10": sum(row["header"]["duration"] > 10 for row in chosen),
            "all_short_excluded_before_split": sum(
                summaries[s]["shorter_than_10s"] for s in sources
            ),
            "split_sha256": sha(cache / (domain + "_split.json")),
            "FEAS_metadata_sha256": sha(cache / (domain + "_FEAS.json")),
        }
    write_new(
        out / "SPLIT_SUMMARY.json",
        {
            "seed": SEED,
            "algorithm": "joint-pattern Hamilton20percent; hashed deterministic tie breaks; one seed only",
            "mapping_sha256": sha(out / "LABEL_MAPPING.json"),
            "domains": summary,
            "reserved_labels": "Used only to stratify; not retained in model-facing cache; expected counts are estimates",
        },
    )
    return summary


def load_feas(root, domain):
    root = Path(root)
    folder = root / "data/manifests/phase3a_v1"
    summary = json.loads((root / "results/phase3a/SPLIT_SUMMARY.json").read_text())
    entry = summary["domains"][domain]
    path = folder / (domain + "_split.json")
    meta_path = folder / (domain + "_FEAS.json")
    if sha(path) != entry["split_sha256"] or sha(meta_path) != entry["FEAS_metadata_sha256"]:
        raise PermissionError("Frozen FEAS membership/metadata changed")
    manifest = json.loads(path.read_text())
    rows = json.loads(meta_path.read_text())
    if [r["id"] for r in rows] != manifest["FEAS"]:
        raise PermissionError("Only complete frozen FEAS can enter inference")
    raw = root / "data/raw/cinc2021_v1_0_3"
    waves = np.stack(
        [
            to_100hz_first10(physical_wave(raw / row["id"], row["header"]), row["header"]["fs"])
            for row in rows
        ]
    )
    return waves, rows, manifest
