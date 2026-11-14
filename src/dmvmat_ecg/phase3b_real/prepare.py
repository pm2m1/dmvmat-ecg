"""One-time ID-only split first; decode annotations only for permitted development IDs."""

import ast
import csv
import io
import json
from pathlib import Path

import numpy as np

from dmvmat_ecg.phase3.labels import targets
from dmvmat_ecg.phase3.signals import header
from dmvmat_ecg.phase3b.jobs import atomic_new, file_hash, save_json
from dmvmat_ecg.phase3b.protocol import BUDGETS, DRAW_SEEDS, digest, ordered

from .boundary import MAPPING_HASH, NAMESPACE, PARENT_HASH, permit_record, real_only


def location(root):
    return Path(root) / "private/phase3b_cpsc_v1"


def read_json(path):
    return json.loads(Path(path).read_text())


def prepare(root):
    root = Path(root).resolve()
    real_only(NAMESPACE)
    folder = location(root)
    if folder.exists():
        raise FileExistsError("One-time preparation already started; do not regenerate")
    source_raw = root / "data/raw/ptbxl_v1_0_3"
    target_raw = root / "data/raw/cinc2021_v1_0_3"
    map_path = root / "results/phase3a/LABEL_MAPPING.json"
    parent_path = root / "data/manifests/phase3a_v1/CPSC_split.json"
    assert file_hash(map_path) == MAPPING_HASH and file_hash(parent_path) == PARENT_HASH
    source_receipt = read_json(root / "results/phase3a/SOURCE_METADATA_IDENTITY.json")
    for name, expected in source_receipt["sha256"].items():
        if file_hash(source_raw / name) != expected:
            raise PermissionError("PTBXL metadata identity changed")
    parent = read_json(parent_path)
    if len(parent["RESERVED"]) != 7719 or len(set(parent["RESERVED"])) != 7719:
        raise ValueError("Expected original 7719 canonical CPSC RESERVED records")
    if set(parent["FEAS"]).intersection(parent["RESERVED"]):
        raise PermissionError("Original FEAS/RESERVED overlap")
    order = ordered(parent["RESERVED"], 77101)
    r, o = order[: len(order) // 2], order[len(order) // 2 :]
    membership = dict(R=r, O=o, FEAS=tuple(parent["FEAS"]))
    episodes, used = {}, set()
    for draw in DRAW_SEEDS:
        draw_ids = ordered(r, draw)[:256]
        fm, gm = draw_ids[:128], draw_ids[128:]
        for b in BUDGETS:
            f, g = fm[: b // 2], gm[: b // 2]
            if set(f).intersection(g) or not set(f + g).issubset(r):
                raise PermissionError("Invalid nested development allocation")
            episodes[f"{draw}/B{b}"] = dict(F=f, G=g, hashes=dict(F=digest(f), G=digest(g)))
            used.update(f + g)
    membership["target_development"] = tuple(sorted(used))
    original = read_json(root / "data/manifests/memberships.json")
    source_ids = dict(source_train=set(original["train"]), source_stop=set(original["validation"]))
    if len(source_ids["source_train"]) != 17084 or len(source_ids["source_stop"]) != 2146:
        raise ValueError("Historical source eligibility counts differ")
    if source_ids["source_train"].intersection(source_ids["source_stop"]):
        raise PermissionError("Source role overlap")
    frozen = read_json(map_path)
    source = {"source_train": [], "source_stop": []}
    # Only ID/fold membership is considered before annotation decoding. Fold10 is skipped.
    with (source_raw / "ptbxl_database.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            rid = int(row["ecg_id"])
            role = next((name for name, ids in source_ids.items() if rid in ids), None)
            if role is None:
                continue
            fold = int(row["strat_fold"])
            if fold not in (range(1, 9) if role == "source_train" else (9,)):
                raise PermissionError("Source role contains unauthorized fold")
            codes = ast.literal_eval(row["scp_codes"])
            if not isinstance(codes, dict) or not codes:
                raise ValueError("Eligible source annotation is missing/malformed")
            filename = row["filename_lr"]
            if not filename.startswith("records100/") or ".." in Path(filename).parts:
                raise PermissionError("Invalid source 100Hz signal path")
            source[role].append(
                dict(
                    record=str(rid),
                    group="ptbxl-patient:" + row["patient_id"],
                    filename=filename,
                    fold=fold,
                    y=[int(bool(set(codes) & set(frozen["proxy_scp"][n]))) for n in ("CD", "STTC")],
                    observed=[True, True],
                )
            )
    for name in source:
        lookup = {int(row["record"]): row for row in source[name]}
        if set(lookup) != source_ids[name]:
            raise ValueError("Source eligibility mismatch")
        source[name] = [
            lookup[rid] for rid in original["train" if name == "source_train" else "validation"]
        ]
    if {row["group"] for row in source["source_train"]}.intersection(
        row["group"] for row in source["source_stop"]
    ):
        raise PermissionError("Source train/selection patient overlap")
    # Freeze all memberships BEFORE first target annotation/header access.
    contract = dict(
        namespace=NAMESPACE,
        split_seed=77101,
        episodes=episodes,
        membership=membership,
        source=source,
        mapping_hash=MAPPING_HASH,
        parent_manifest_hash=PARENT_HASH,
        source_metadata_hashes=source_receipt["sha256"],
        source_membership_hash=file_hash(root / "data/manifests/memberships.json"),
        duplicate_audit_hash=file_hash(root / "results/phase3a/DATASET_AUDIT.json"),
        target_publisher_index_hash=file_hash(target_raw / "SHA256SUMS.txt"),
        source_publisher_index_hash=file_hash(source_raw / "SHA256SUMS.txt"),
        target_patient_linkage="unknown",
        O_access_authorized=False,
    )
    folder.mkdir(parents=True)
    save_json(folder / "MEMBERSHIPS.json", contract)
    development = {}
    for rid in sorted(used):
        permit_record(rid, "target_development", membership)
        meta = header(target_raw / (rid + ".hea"))
        if meta["duration"] < 10:
            raise ValueError("Previously eligible CPSC recording is shorter than10s")
        development[rid] = dict(header=meta, y=targets(meta["dx"], frozen), observed=[True, True])
    save_json(folder / "TARGET_DEVELOPMENT_METADATA.json", development)
    support = {}
    for name, ep in episodes.items():
        support[name] = {
            role: dict(
                records=len(ep[role]),
                positive=np.array([development[i]["y"] for i in ep[role]]).sum(0).tolist(),
                observed_entries=2 * len(ep[role]),
            )
            for role in ("F", "G")
        }
    receipt = dict(
        namespace=NAMESPACE,
        memberships_hash=file_hash(folder / "MEMBERSHIPS.json"),
        metadata_hash=file_hash(folder / "TARGET_DEVELOPMENT_METADATA.json"),
        R_records=len(r),
        O_records=len(o),
        unique_development_records=len(used),
        source_records={k: len(v) for k, v in source.items()},
        source_patients={k: len({row["group"] for row in v}) for k, v in source.items()},
        source_positive={
            k: np.array([row["y"] for row in v]).sum(0).tolist() for k, v in source.items()
        },
        support=support,
        O_data_accesses=0,
        known_duplicate_rule="Phase3A canonical retained record ID",
        source_target_group_overlap=0,
        FEAS_overlap=0,
    )
    save_json(folder / "PREPARATION_RECEIPT.json", receipt)
    return receipt


def npy_payload(x):
    stream = io.BytesIO()
    np.save(stream, x, allow_pickle=False)
    return stream.getvalue()


def ensure_cache(root):
    """Non-predictive development-only decoding; O and FEAS fail before file reads."""
    import wfdb

    from dmvmat_ecg.phase3.signals import lead_order, physical_wave, to_100hz_first10, unit_factor

    root = Path(root).resolve()
    real_only(NAMESPACE)
    folder = location(root)
    contract = read_json(folder / "MEMBERSHIPS.json")
    receipt = read_json(folder / "PREPARATION_RECEIPT.json")
    if file_hash(folder / "MEMBERSHIPS.json") != receipt["memberships_hash"]:
        raise PermissionError("Membership preparation changed")
    if (folder / "CACHE_RECEIPT.json").exists():
        raise FileExistsError("Development caches already prepared; do not regenerate")
    target = read_json(folder / "TARGET_DEVELOPMENT_METADATA.json")
    index = {}
    stats = {}
    publisher = {}
    for domain, raw_path in (
        ("source", root / "data/raw/ptbxl_v1_0_3"),
        ("target", root / "data/raw/cinc2021_v1_0_3"),
    ):
        publisher[domain] = {}
        for line in (raw_path / "SHA256SUMS.txt").read_text().splitlines():
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                publisher[domain][parts[1].lstrip(" *").removeprefix("./")] = parts[0]

    def verify_raw(domain, path, raw_path):
        key = str(path.relative_to(raw_path))
        if publisher[domain].get(key) != file_hash(path):
            raise PermissionError("Authorized development raw file differs from publisher: " + key)

    for role in ("source_train", "source_stop", "target_development"):
        records = (
            contract["source"][role]
            if role.startswith("source")
            else [dict(record=rid, group=rid) for rid in contract["membership"][role]]
        )
        waves = []
        truncated = 0
        for row in records:
            rid = row["record"]
            if role.startswith("source"):
                if row["fold"] not in (range(1, 9) if role == "source_train" else (9,)):
                    raise PermissionError("Fold10 source access denied")
                path = root / "data/raw/ptbxl_v1_0_3" / row["filename"]
                for suffix in (".hea", ".dat"):
                    verify_raw("source", path.with_suffix(suffix), root / "data/raw/ptbxl_v1_0_3")
                wave, fields = wfdb.rdsamp(str(path))
                if fields["fs"] != 100 or wave.shape != (1000, 12):
                    raise ValueError("Source100Hz/10s shape mismatch")
                order = lead_order(fields["sig_name"])
                x = wave[:, order] * np.array([unit_factor(fields["units"][i]) for i in order])
                x = x.T.astype("float32")
            else:
                permit_record(rid, role, contract["membership"])
                meta = target[rid]["header"]
                raw_path = root / "data/raw/cinc2021_v1_0_3"
                verify_raw("target", raw_path / (rid + ".hea"), raw_path)
                verify_raw("target", raw_path / Path(rid).parent / meta["signal_file"], raw_path)
                truncated += int(meta["duration"] > 10)
                x = to_100hz_first10(
                    physical_wave(root / "data/raw/cinc2021_v1_0_3" / rid, meta), meta["fs"]
                )
            if x.shape != (12, 1000) or not np.isfinite(x).all():
                raise ValueError("Nonfinite or malformed development waveform")
            waves.append(x)
        array = np.stack(waves)
        path = folder / "cache" / (role + ".npy")
        atomic_new(path, npy_payload(array))
        index[role] = dict(
            path=str(path), sha=file_hash(path), records=[r["record"] for r in records]
        )
        stats[role] = dict(
            records=len(array), shape=list(array.shape), truncated=truncated, dtype=str(array.dtype)
        )
    annotations = {}
    for rid, row in target.items():
        permit_record(rid, "target_development", contract["membership"])
        path = folder / "target_annotations" / (digest(rid) + ".json")
        save_json(path, {k: row[k] for k in ("y", "observed")})
        annotations[rid] = dict(path=str(path), sha=file_hash(path))
    result = dict(
        target_annotations=annotations,
        caches=index,
        shapes=stats,
        O_data_accesses=0,
        memberships_hash=receipt["memberships_hash"],
        target_metadata_hash=receipt["metadata_hash"],
    )
    save_json(folder / "CACHE_RECEIPT.json", result)
    return result
