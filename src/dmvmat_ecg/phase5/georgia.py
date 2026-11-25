"""One-shot label-blind membership, then guarded adaptation-only preparation."""

import json
import os
import sys
from pathlib import Path

import numpy as np

from dmvmat_ecg.phase3.signals import header, physical_wave, to_100hz_first10
from dmvmat_ecg.phase3b_eval_v2.decoder import decode_labels
from dmvmat_ecg.phase3b_real.boundary import MAPPING_HASH
from dmvmat_ecg.phase4.safety import (
    RecordGuard,
    digest,
    exclusion_hash,
    new_json,
    project_json,
    sha,
)
from dmvmat_ecg.phase4c_confirm.provider import signal_only_mat

from .common import GEORGIA, ROOT
from .geometry_split import allocate, episodes, validation

MEMBERS = ROOT / "private/phase5_georgia_membership"
PARENT = ROOT / "data/manifests/phase3a_v1/Georgia_split.json"
SUMMARY = ROOT / "results/phase3a/SPLIT_SUMMARY.json"
AUDIT = ROOT / "results/phase3a/DATASET_AUDIT.json"
MAPPING = ROOT / "results/phase3a/LABEL_MAPPING.json"
RAW = ROOT / "data/raw/cinc2021_v1_0_3"


class BuildScope:
    def __init__(self):
        self.pool_paths = set()

    def check(self, event, args):
        if (
            event not in {"open", "os.listdir", "os.scandir"}
            or not args
            or not isinstance(args[0], (str, bytes, os.PathLike))
        ):
            return
        p = Path(os.fsdecode(args[0])).absolute()
        if not p.is_relative_to(ROOT):
            return
        if any("sealed" in part.lower() for part in p.parts):
            raise PermissionError("Sealed material denied")
        if not any(p.is_relative_to(ROOT / n) for n in ("private", "data", "results")):
            return
        if ".." in p.parts or p.resolve() != p:
            raise PermissionError("Unsafe data path")
        flags = args[2] if event == "open" and len(args) > 2 and isinstance(args[2], int) else 0
        if p.is_relative_to(GEORGIA) or p.is_relative_to(MEMBERS):
            return
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            raise PermissionError("Historical inputs read-only")
        if p in {PARENT, SUMMARY, AUDIT, MAPPING, RAW / "SHA256SUMS.txt"} or p in self.pool_paths:
            return
        raise PermissionError("GEORGIA_NONPOOL_ACCESS_DENIED")


def build():
    os.umask(0o077)
    if GEORGIA.exists() or MEMBERS.exists():
        raise FileExistsError("Existing or partial Georgia allocation preserved")
    scope = BuildScope()
    sys.addaudithook(scope.check)
    metadata = project_json(SUMMARY, [("domains", "Georgia"), ("mapping_sha256",)])
    if (
        sha(PARENT) != metadata["domains", "Georgia"]["split_sha256"]
        or sha(MAPPING) != MAPPING_HASH
        or metadata["mapping_sha256",] != MAPPING_HASH
    ):
        raise PermissionError("Historical Georgia membership/mapping differs")
    old_index = project_json(AUDIT, [("publisher_index_sha256",)])
    if sha(RAW / "SHA256SUMS.txt") != old_index["publisher_index_sha256",]:
        raise PermissionError("Publisher index differs from historical native-duplicate audit")
    parent = project_json(PARENT, [("FEAS",), ("RESERVED",)])
    feas, remaining = parent["FEAS",], parent["RESERVED",]
    if (
        len(feas) != metadata["domains", "Georgia"]["FEAS_records"]
        or len(remaining) != metadata["domains", "Georgia"]["RESERVED_records"]
    ):
        raise PermissionError("Historical canonical count differs")
    pool, test = allocate(feas, remaining)
    if any(not r.startswith("training/georgia/") or ".." in Path(r).parts for r in pool + test):
        raise PermissionError("Non-Georgia or unsafe record identity")
    MEMBERS.mkdir(mode=0o700)
    assignment = dict(
        schema="phase5-georgia-id-only-assignment-v1",
        pool=pool,
        test=test,
        survey=feas,
        parent_sha256=sha(PARENT),
        pool_sha256=digest(pool),
        test_sha256=digest(test),
        seed=79101,
        label_blind=True,
        labels_read_to_allocate=0,
        signals_read_to_allocate=0,
        known_groups="historical canonical native-duplicate representatives; one retained ID per known group",
    )
    new_json(MEMBERS / "assignment.json", assignment)
    GEORGIA.mkdir(mode=0o700)
    guard = RecordGuard(pool, [exclusion_hash(r) for r in test])
    new_json(GEORGIA / "records.json", pool)
    new_json(GEORGIA / "exclusion_hashes.json", sorted(guard.exclusions))
    ep = episodes(pool)
    new_json(GEORGIA / "episodes.json", ep)
    scope.pool_paths = {RAW / (r + s) for r in pool for s in (".hea", ".mat")}
    publisher = {}
    for line in (RAW / "SHA256SUMS.txt").read_text().splitlines():
        h, n = line.split(maxsplit=1)
        publisher[n.lstrip(" *").removeprefix("./")] = h
    mapping = json.loads(MAPPING.read_text())
    waves = np.lib.format.open_memmap(
        GEORGIA / "signals.npy", mode="w+", dtype=np.float32, shape=(len(pool), 12, 1000)
    )
    y = np.full((len(pool), 2), np.nan, np.float32)
    observed = np.zeros((len(pool), 2), bool)
    fingerprints = {}
    for i, r in enumerate(pool):
        guard.check([r])
        hp = RAW / (r + ".hea")
        mp = RAW / (r + ".mat")
        if sha(hp) != publisher.get(r + ".hea"):
            raise PermissionError("Pool header checksum differs")
        labels, obs = decode_labels(hp.read_bytes(), mapping)
        meta = header(hp)
        if meta["duration"] < 10 or hp.parent / meta["signal_file"] != mp:
            raise PermissionError(
                "Historical physical eligibility changed; do not silently exclude"
            )
        signal_only_mat(mp)
        if sha(mp) != publisher.get(r + ".mat"):
            raise PermissionError("Pool signal checksum differs")
        waves[i] = to_100hz_first10(physical_wave(RAW / r, meta), meta["fs"])
        y[i], observed[i] = labels, obs
        fingerprints[r] = dict(header_sha256=sha(hp), signal_sha256=sha(mp))
        if (i + 1) % 500 == 0:
            print(f"GEORGIA_POOL_ONLY_PROGRESS {i + 1}/{len(pool)}", flush=True)
    waves.flush()
    np.save(GEORGIA / "labels.npy", y, allow_pickle=False)
    np.save(GEORGIA / "observed.npy", observed, allow_pickle=False)
    split = validation(pool, y, ep)
    lookup = {r: i for i, r in enumerate(pool)}
    selected = {r for e in ep.values() for k in ("F", "G") for r in e[k]}
    if not np.isfinite(y[[lookup[r] for r in selected]]).all():
        raise PermissionError(
            "Unknown label in fixed label-blind training draw; no negative imputation or redraw"
        )
    new_json(GEORGIA / "split.json", split)
    new_json(GEORGIA / "raw_input_hashes.json", fingerprints)

    def counts(ids):
        values = y[[lookup[r] for r in ids]]
        obs = np.isfinite(values)
        return dict(
            records=len(ids),
            observed=obs.sum(0).tolist(),
            positive=np.nansum(values, axis=0).astype(int).tolist(),
            prevalence=[
                float(values[obs[:, j], j].mean()) if obs[:, j].any() else None for j in range(2)
            ],
            joint={
                f"{a}{b}": int(np.all(values == [a, b], axis=1).sum())
                for a in (0, 1)
                for b in (0, 1)
            },
        )

    report = dict(
        pool=counts(pool),
        A=counts(split["A"]),
        B=counts(split["B"]),
        survey_records=len(feas),
        test_records=len(test),
        test_prevalence="NOT_ACCESSED",
        all_draws_union=len(selected),
        A_B_overlap=0,
        validation_draw_overlap=0,
        pool_test_overlap=0,
        survey_test_overlap=0,
        known_group_overlap=0,
        pool_sha256=digest(pool),
        test_sha256=digest(test),
        A_sha256=digest(split["A"]),
        B_sha256=digest(split["B"]),
        label_blind_test_assignment=True,
        test_signal_accesses=0,
        test_label_accesses=0,
    )
    files = {p.name: sha(p) for p in GEORGIA.iterdir() if p.is_file()}
    new_json(
        GEORGIA / "MANIFEST.json",
        dict(
            schema="phase5-georgia-adaptation-only-v1",
            files=files,
            mapping_sha256=MAPPING_HASH,
            decoder_sha256=sha(ROOT / "src/dmvmat_ecg/phase3b_eval_v2/decoder.py"),
            physical_code_sha256=sha(ROOT / "src/dmvmat_ecg/phase3/signals.py"),
            assignment_sha256=sha(MEMBERS / "assignment.json"),
            summary=report,
        ),
    )
    print(json.dumps(report, indent=2), flush=True)


class Target:
    def __init__(self):
        manifest = json.loads((GEORGIA / "MANIFEST.json").read_text())
        for name, h in manifest["files"].items():
            p = GEORGIA / name
            if not p.resolve().is_relative_to(GEORGIA) or sha(p) != h:
                raise PermissionError("Georgia bundle integrity differs")
        self.identity = sha(GEORGIA / "MANIFEST.json")
        self.records = json.loads((GEORGIA / "records.json").read_text())
        self.guard = RecordGuard(
            self.records, json.loads((GEORGIA / "exclusion_hashes.json").read_text())
        )
        self.lookup = {r: i for i, r in enumerate(self.records)}
        self.split = json.loads((GEORGIA / "split.json").read_text())
        self.episodes = json.loads((GEORGIA / "episodes.json").read_text())
        self.x = np.load(GEORGIA / "signals.npy", mmap_mode="r", allow_pickle=False)
        self._labels = np.load(GEORGIA / "labels.npy", mmap_mode="r", allow_pickle=False)
        a, b = self.split["A"], self.split["B"]
        used = {r for ep in self.episodes.values() for n in ("F", "G") for r in ep[n]}
        self.guard.check(a + b + list(used))
        if len(a) != 500 or len(b) != 500 or set(a) & set(b) or used & (set(a) | set(b)):
            raise PermissionError("Georgia role overlap/size differs")

    def ids(self, role, draw=None, budget=None):
        if role in ("A", "B"):
            return self.split[role]
        if role in ("F", "G"):
            return self.episodes[f"{draw}/B{budget}"][role]
        raise PermissionError("No Georgia test/U capability")

    def index(self, ids):
        self.guard.check(ids)
        return np.asarray([self.lookup[r] for r in ids], int)

    def labels(self, role, draw=None, budget=None):
        values = np.array(self._labels[self.index(self.ids(role, draw, budget))], copy=True)
        if role in ("F", "G") and not np.isin(values, [0, 1]).all():
            raise PermissionError("Unknown training labels cannot become negatives")
        return values


if __name__ == "__main__":
    build()
