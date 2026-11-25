"""Project PTB folds 1–9 before signal access; never decode a fold-10 row."""

import ast
import csv
import io
import json
import mmap
import shutil
from pathlib import Path

import numpy as np
import torch

from dmvmat_ecg.phase3.labels import CONDITIONS
from dmvmat_ecg.phase3b.roles import UnitRole, array_hash
from dmvmat_ecg.phase4.build_bundle import safe_path
from dmvmat_ecg.phase4.safety import digest, new_json, project_json, sha

from .common import BUNDLE, ROOT, SOURCE


def csv_spans(data):
    """Offsets only: quoted CSV newlines/commas stay in their original record."""
    start = i = 0
    quoted = False
    while i < len(data):
        if data[i] == 34:
            if quoted and i + 1 < len(data) and data[i + 1] == 34:
                i += 2
                continue
            quoted = not quoted
        elif data[i] == 10 and not quoted:
            yield start, i
            start = i + 1
        i += 1
    if quoted:
        raise ValueError("Unterminated CSV quote")
    if start < len(data):
        yield start, len(data)


def cell_spans(data, start, end):
    cells = []
    i = start
    last = start
    quoted = False
    while i < end:
        if data[i] == 34:
            if quoted and i + 1 < end and data[i + 1] == 34:
                i += 2
                continue
            quoted = not quoted
        elif data[i] == 44 and not quoted:
            cells.append((last, i))
            last = i + 1
        i += 1
    cells.append((last, end))
    return cells


def project_source_csv(path, wanted):
    """Only fold number is decoded before excluding an out-of-scope record."""
    selected = {}
    with (
        Path(path).open("rb") as stream,
        mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data,
    ):
        records = iter(csv_spans(data))
        a, b = next(records)
        names = next(csv.reader(io.StringIO(data[a:b].decode("utf-8-sig"))))
        fold_column = names.index("strat_fold")
        id_column = names.index("ecg_id")
        for start, end in records:
            if start == end:
                continue
            spans = cell_spans(data, start, end)
            if len(spans) != len(names):
                raise ValueError("Malformed source CSV row")
            a, b = spans[fold_column]
            fold = int(data[a:b].strip(b' "\r'))
            if fold not in range(1, 10):
                continue  # NO ID, annotation, report or signal field decoded.
            a, b = spans[id_column]
            rid = str(int(data[a:b].strip(b' "\r')))
            if rid not in wanted:
                continue
            row = dict(
                zip(names, next(csv.reader(io.StringIO(data[start:end].decode()))), strict=True)
            )
            if rid in selected:
                raise ValueError("Duplicate source record")
            selected[rid] = dict(
                record=rid,
                fold=fold,
                group="ptbxl-patient:" + row["patient_id"],
                codes=ast.literal_eval(row["scp_codes"]),
            )
    if set(selected) != set(wanted):
        raise PermissionError("Source membership not entirely within folds 1–9")
    return selected


def prepare():
    if SOURCE.exists():
        raise FileExistsError("Existing source bundle must not be overwritten")
    membership = project_json(
        ROOT / "data/manifests/memberships.json", [("train",), ("validation",)]
    )
    expected = {
        role: [str(r) for r in membership[key,]]
        for role, key in [("source_train", "train"), ("source_stop", "validation")]
    }
    if len(expected["source_train"]) != 17084 or len(expected["source_stop"]) != 2146:
        raise PermissionError("Original source eligibility counts mismatch")
    if set(expected["source_train"]) & set(expected["source_stop"]):
        raise PermissionError("Source roles overlap")
    metadata = project_source_csv(
        ROOT / "data/raw/ptbxl_v1_0_3/ptbxl_database.csv",
        set(expected["source_train"] + expected["source_stop"]),
    )
    # Explicit source-only descriptor fields; no CPSC member/evaluation field.
    old = ROOT / "private/phase3b_cpsc_v1"
    cache = project_json(old / "CACHE_RECEIPT.json", [("caches", r) for r in expected])
    code_sets = {
        label: {s for c in CONDITIONS.values() if c["label"] == label for s in c["scp"]}
        for label in ("CD", "STTC")
    }
    prepared = {}
    for role, ids in expected.items():
        rows = [metadata[r] for r in ids]
        if any(
            row["fold"] not in (range(1, 9) if role == "source_train" else (9,)) for row in rows
        ):
            raise PermissionError("Unauthorized source fold before cache access")
        if any(not isinstance(r["codes"], dict) or not r["codes"] for r in rows):
            raise ValueError("Original source annotations missing")
        y = np.array(
            [
                [int(bool(set(row["codes"]) & code_sets[label])) for label in ("CD", "STTC")]
                for row in rows
            ],
            dtype=np.float32,
        )
        groups = [row["group"] for row in rows]
        entry = cache["caches", role]
        if entry["records"] != ids:
            raise PermissionError("Cache record order differs from verified source folds")
        path = safe_path(entry["path"], old / "cache")
        if path.name != role + ".npy":
            raise PermissionError("Wrong explicit source cache")
        with path.open("rb") as f:
            version = np.lib.format.read_magic(f)
            shape, fortran, dtype = (
                np.lib.format.read_array_header_1_0(f)
                if version == (1, 0)
                else np.lib.format.read_array_header_2_0(f)
            )
        if shape != (len(ids), 12, 1000) or dtype != np.dtype("float32") or fortran:
            raise PermissionError("Source-only cache shape mismatch before payload access")
        if sha(path) != entry["sha"]:
            raise PermissionError("Original source cache fingerprint mismatch")
        x = np.load(path, mmap_mode="r", allow_pickle=False)
        observed = np.ones(y.shape, dtype=bool)
        unit = UnitRole(
            role,
            tuple(ids),
            tuple(groups),
            "patient",
            "known",
            lambda: x,
            lambda: (y, observed),
            array_hash([x]),
            array_hash([y, observed]),
        )
        prepared[role] = dict(
            ids=ids,
            groups=groups,
            y=y,
            folds=[r["fold"] for r in rows],
            path=path,
            sha256=entry["sha"],
            identity=unit.identity,
        )
    if set(prepared["source_train"]["groups"]) & set(prepared["source_stop"]["groups"]):
        raise PermissionError("Original source patient separation differs")
    sources = json.loads((BUNDLE / "sources.json").read_text())
    identities = [prepared[r]["identity"] for r in ("source_train", "source_stop")]
    for s, entry in sources.items():
        saved = torch.load(BUNDLE / entry["checkpoint"], map_location="cpu", weights_only=True)
        if saved["identity"]["data"] != identities or saved["normalization"][
            "training_identity"
        ] != digest([identities[0]]):
            raise PermissionError(
                "Reconstructed source roles differ from original source checkpoint"
            )
    if shutil.disk_usage(ROOT).free < 4 * 1024**3:
        raise OSError("Insufficient free space for source-only isolation")
    SOURCE.mkdir(mode=0o700)
    roles = {}
    for role, item in prepared.items():
        shutil.copyfile(item["path"], SOURCE / (role + "_x.npy"))
        np.save(SOURCE / (role + "_y.npy"), item["y"], allow_pickle=False)
        new_json(
            SOURCE / (role + "_records.json"),
            dict(ids=item["ids"], groups=item["groups"], folds=item["folds"]),
        )
        roles[role] = dict(
            records=len(item["ids"]),
            identity=item["identity"],
            fold_counts={str(f): item["folds"].count(f) for f in sorted(set(item["folds"]))},
            positive_counts=item["y"].sum(0).astype(int).tolist(),
        )
    files = {p.name: sha(p) for p in SOURCE.iterdir() if p.is_file()}
    new_json(
        SOURCE / "MANIFEST.json",
        dict(
            schema="phase4b-ptbxl-folds1to9-only-v1",
            roles=roles,
            files=files,
            source_role_identities_match_original=True,
            fold10_signal_label_id_fields_decoded=False,
            permitted_original_descriptor_fields=[
                ["caches", "source_train"],
                ["caches", "source_stop"],
            ],
        ),
    )
    print(
        json.dumps(
            dict(
                status="SOURCE_BUNDLE_READY",
                roles=roles,
                source_role_identities_match_original=True,
            ),
            indent=2,
        ),
        flush=True,
    )


class SourceData:
    def __init__(self):
        manifest = json.loads((SOURCE / "MANIFEST.json").read_text())
        for name, expected in manifest["files"].items():
            if sha(SOURCE / name) != expected:
                raise PermissionError("Source-only bundle changed")
        self.manifest = manifest
        self.identity = sha(SOURCE / "MANIFEST.json")
        self.x = {}
        self.y = {}
        self.records = {}
        for role in ("source_train", "source_stop"):
            self.records[role] = json.loads((SOURCE / (role + "_records.json")).read_text())
            folds = self.records[role]["folds"]
            if any(f not in (range(1, 9) if role == "source_train" else (9,)) for f in folds):
                raise PermissionError("Fold10 source read forbidden")
            self.x[role] = np.load(SOURCE / (role + "_x.npy"), mmap_mode="r", allow_pickle=False)
            self.y[role] = np.load(SOURCE / (role + "_y.npy"), allow_pickle=False)


if __name__ == "__main__":
    prepare()
