"""PTB-only source inputs; no CPSC dataset or feature-loader dependency."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from dmvmat_ecg.phase3b.roles import Normalization
from dmvmat_ecg.phase4.safety import digest, new_json, sha
from dmvmat_ecg.phase4b.source_data import SourceData
from dmvmat_ecg.preprocessing.signals import normalize, views

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "private/phase4c_placebo_run"
SOURCE = ROOT / "private/phase4b_source_bundle"
BUNDLE = ROOT / "private/phase4_oracle_bundle"
OLD = ROOT / "private/phase4b_run"
SPLIT = ROOT / "private/phase4_oracle_run/split.json"


def access_check(mode, event, args):
    if event not in {"open", "os.listdir", "os.scandir"} or not args:
        return
    if not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    p = Path(os.fsdecode(args[0])).absolute()
    private = ROOT / "private"
    if p.is_relative_to(ROOT) and any(
        "sealed" in part.lower() or "reserved" in part.lower() for part in p.parts
    ):
        raise PermissionError("Reserved/sealed access forbidden")
    if not (
        p.is_relative_to(private)
        or p.is_relative_to(ROOT / "data")
        or p.is_relative_to(ROOT / "results")
    ):
        return
    if ".." in p.parts or p.resolve() != p:
        raise PermissionError("Data path traversal/symlink rejected")
    allowed = (SOURCE, RUN) if mode == "source" else (SOURCE, RUN, BUNDLE, OLD)
    if p != SPLIT or mode == "source":
        if not any(p.is_relative_to(r) for r in allowed):
            raise PermissionError(
                "SOURCE_ONLY_INPUT_DENIED" if mode == "source" else "OUTSIDE_ADAPTATION_INPUTS"
            )
    if event == "open" and not p.is_relative_to(RUN):
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            raise PermissionError("Existing study inputs/outputs are read-only")


def setup(mode):
    if mode not in {"source", "target"}:
        raise ValueError("Explicit source or target mode required")
    RUN.mkdir(mode=0o700, exist_ok=True)
    if RUN.is_symlink():
        raise PermissionError("Unsafe placebo output root")
    RUN.chmod(0o700)
    sys.addaudithook(lambda event, args: access_check(mode, event, args))


def normalization(source):
    """Same float64 reductions as original Normalization.fit, lower peak RAM."""
    x = source.x["source_train"].astype(float)
    mean = x.sum() / x.size
    x -= mean
    np.square(x, out=x)
    variance = x.sum() / x.size
    return Normalization(
        float(mean),
        float(np.sqrt(variance)),
        digest([source.manifest["roles"]["source_train"]["identity"]]),
        "fresh-source-only",
    )


def prepare():
    setup("source")
    # Validate fold metadata before SourceData hashes/maps any signal payload.
    for role in ("source_train", "source_stop"):
        rows = json.loads((SOURCE / (role + "_records.json")).read_text())
        scope = range(1, 9) if role == "source_train" else (9,)
        if any(f not in scope for f in rows["folds"]):
            raise PermissionError("Out-of-scope source fold")
    source = SourceData()
    manifest_path = RUN / "SOURCE_INPUTS.json"
    if manifest_path.exists():
        SourceInputs(source)
        print("SOURCE_ONLY_INPUTS_VERIFIED", flush=True)
        return
    folder = RUN / "source_inputs"
    if folder.exists():
        raise FileExistsError("Partial source inputs cannot be overwritten")
    folder.mkdir(mode=0o700)
    norm = normalization(source)
    roles = {}
    for role in ("source_train", "source_stop"):
        stores = {}
        for start in range(0, len(source.x[role]), 256):
            arrays = views(normalize(source.x[role][start : start + 256], norm.__dict__))
            for name, a in zip(("wave", "rhythm", "spectrum"), arrays, strict=True):
                if name not in stores:
                    path = folder / f"{role}_{name}.npy"
                    stores[name] = np.lib.format.open_memmap(
                        path, mode="w+", dtype="float32", shape=(len(source.x[role]), *a.shape[1:])
                    )
                stores[name][start : start + len(a)] = a
        for store in stores.values():
            store.flush()
        paths = {name: str((folder / f"{role}_{name}.npy").relative_to(RUN)) for name in stores}
        roles[role] = dict(
            records=len(source.x[role]),
            source_identity=source.manifest["roles"][role]["identity"],
            paths=paths,
            hashes={n: sha(RUN / p) for n, p in paths.items()},
        )
    new_json(
        manifest_path,
        dict(
            schema="ptb-only-placebo-inputs-v1",
            source_bundle_sha256=source.identity,
            normalization=norm.__dict__,
            roles=roles,
            CPSC_inputs_read=False,
        ),
    )
    print("SOURCE_ONLY_INPUTS_PREPARED_WITHOUT_CPSC_ACCESS", flush=True)


class SourceInputs:
    def __init__(self, source=None):
        self.source = source if source is not None else SourceData()
        self.manifest = json.loads((RUN / "SOURCE_INPUTS.json").read_text())
        if self.manifest["source_bundle_sha256"] != self.source.identity:
            raise PermissionError("Source bundle changed")
        self.norm = Normalization(**self.manifest["normalization"])
        self.arrays = {}
        for role, entry in self.manifest["roles"].items():
            if role not in {"source_train", "source_stop"}:
                raise PermissionError("Non-source feature role")
            self.arrays[role] = {}
            for name, path in entry["paths"].items():
                p = RUN / path
                if not p.resolve().is_relative_to(RUN) or sha(p) != entry["hashes"][name]:
                    raise PermissionError("Source-only feature integrity failure")
                self.arrays[role][name] = np.load(p, mmap_mode="r", allow_pickle=False)

    def batch(self, role, ix):
        if role not in {"source_train", "source_stop"}:
            raise PermissionError("Placebo pretraining rejects non-PTB roles")
        return {
            n: torch.tensor(np.array(a[ix], copy=True), dtype=torch.float32)
            for n, a in self.arrays[role].items()
        }
