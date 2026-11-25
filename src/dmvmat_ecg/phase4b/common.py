"""Phase 4B roles: adaptation-only inputs, fixed A/B, signal-only U."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from dmvmat_ecg.phase2.adaptation import state_hash
from dmvmat_ecg.phase3b.models import fresh_model, inputs
from dmvmat_ecg.phase3b.roles import Normalization
from dmvmat_ecg.phase4.safety import RecordGuard, digest, new_json, sha

ROOT = Path(__file__).resolve().parents[3]
BUNDLE = ROOT / "private/phase4_oracle_bundle"
RUN = ROOT / "private/phase4b_run"
SOURCE = ROOT / "private/phase4b_source_bundle"
SPLIT = ROOT / "private/phase4_oracle_run/split.json"


def audit():
    """Only isolated inputs/new outputs; no old source run or evaluation reader."""
    roots = tuple(p.resolve() for p in (BUNDLE, RUN, SOURCE))

    def check(event, args):
        if event not in {"open", "os.listdir", "os.scandir"} or not args:
            return
        if not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).absolute()
        if ".." in p.parts or any(p.is_symlink() for p in (p,)):
            raise PermissionError("Traversal/symlink forbidden in Phase4B")
        if any(p.is_relative_to(r) for r in roots) or p == SPLIT:
            return
        if (
            p.is_relative_to(ROOT / "private")
            or p.is_relative_to(ROOT / "data")
            or p.is_relative_to(ROOT / "results")
        ):
            raise PermissionError("Phase4B input outside isolated bundles")
        if p.is_relative_to(ROOT) and any(
            "sealed" in part.lower() or "reserved" in part.lower() for part in p.parts
        ):
            raise PermissionError("Sealed/reserved access forbidden")

    sys.addaudithook(check)


class Target:
    def __init__(self):
        m = json.loads((BUNDLE / "MANIFEST.json").read_text())
        for name, expected in m["files"].items():
            p = BUNDLE / name
            if not p.resolve().is_relative_to(BUNDLE) or p.is_symlink() or sha(p) != expected:
                raise PermissionError("Adaptation bundle integrity failure")
        self.identity = sha(BUNDLE / "MANIFEST.json")
        self.records = json.loads((BUNDLE / "records.json").read_text())
        self.guard = RecordGuard(
            self.records, json.loads((BUNDLE / "exclusion_hashes.json").read_text())
        )
        self.lookup = {rid: i for i, rid in enumerate(self.records)}
        self.split = json.loads(SPLIT.read_text())
        if self.split["bundle_sha256"] != self.identity:
            raise PermissionError("Original validation split differs")
        self.guard.check(self.split["A"] + self.split["B"])
        self.episodes = json.loads((BUNDLE / "episodes.json").read_text())
        self.sources = json.loads((BUNDLE / "sources.json").read_text())
        self.references = json.loads((BUNDLE / "reference_fits.json").read_text())
        self.x = np.load(BUNDLE / "signals.npy", mmap_mode="r", allow_pickle=False)
        self._labels = np.load(BUNDLE / "labels.npy", mmap_mode="r", allow_pickle=False)
        used = {r for ep in self.episodes.values() for role in ("F", "G") for r in ep[role]}
        selected = set(self.split["A"] + self.split["B"])
        if set(self.split["A"]) & set(self.split["B"]) or used & selected or len(selected) != 1000:
            raise PermissionError("Invalid validation separation")
        self.u = [r for r in self.records if r not in selected]
        if len(self.u) != 2859:
            raise ValueError("Unexpected signal-only U count")
        self.label_access = []

    def ids(self, role, draw=None, budget=None):
        if role in ("A", "B"):
            return self.split[role]
        if role == "U":
            return self.u
        if role in ("F", "G") and draw in (16101, 16102, 16103) and budget in (64, 128, 256):
            return self.episodes[f"{draw}/B{budget}"][role]
        raise PermissionError("Unknown target role")

    def index(self, ids):
        self.guard.check(ids)
        return np.array([self.lookup[r] for r in ids], dtype=int)

    def signals(self, role, draw=None, budget=None):
        return self.x[self.index(self.ids(role, draw, budget))]

    def labels(self, role, draw=None, budget=None):
        if role not in {"F", "G", "A", "B"}:
            raise PermissionError("UNLABELLED_TARGET_LABEL_ACCESS_FORBIDDEN")
        ids = self.ids(role, draw, budget)
        self.label_access.append(dict(role=role, records=len(ids)))
        y = np.array(self._labels[self.index(ids)], copy=True)
        if not np.isin(y, [0, 1]).all():
            raise ValueError("Unknown labels cannot become negative")
        return y


def load_source(target, source_seed, aligned=None):
    if aligned is None:
        entry = target.sources[str(source_seed)]
        path = BUNDLE / entry["checkpoint"]
        expected = entry["sha256"]
    else:
        entry = json.loads((RUN / "aligned_sources" / f"{aligned}_{source_seed}.json").read_text())
        path = RUN / entry["checkpoint"]
        expected = entry["sha256"]
    if sha(path) != expected:
        raise PermissionError("Source checkpoint identity mismatch")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = fresh_model(source_seed)
    model.load_state_dict(saved["state"], strict=True)
    model.eval()
    if state_hash(model) != entry["state_hash"]:
        raise PermissionError("Source model state mismatch")
    norm = Normalization(**saved["normalization"])
    return model, norm, saved, expected


def predict(model, norm, target, role):
    if role not in ("A", "B"):
        raise PermissionError("Only validation A/B predictions may be exported")
    model.eval()
    records = target.ids(role)
    ix = target.index(records)
    chunks = []
    with torch.no_grad():
        for start in range(0, len(ix), 128):
            batch = ix[start : start + 128]
            chunks.append(model(**inputs(model, norm.transform(target.x[batch]))).sigmoid().numpy())
    p = np.concatenate(chunks)
    if p.shape != (500, 2) or not np.isfinite(p).all():
        raise ValueError("Invalid validation predictions")
    return p


def setup():
    RUN.mkdir(mode=0o700, parents=True, exist_ok=True)
    RUN.chmod(0o700)
    audit()
    t = Target()
    p = RUN / "SPLIT_BINDING.json"
    binding = dict(
        bundle_sha256=t.identity,
        split_sha256=digest(t.split),
        A_records=500,
        B_records=500,
        U_records=len(t.u),
        unlabelled_label_accesses=0,
    )
    if p.exists():
        if json.loads(p.read_text()) != binding:
            raise PermissionError("Existing Phase4B split binding differs")
    else:
        new_json(p, binding)
    return t
