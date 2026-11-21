"""Frozen source-normalized input views; U cache contains no target labels."""

import json

import numpy as np
import torch

from dmvmat_ecg.phase4.safety import digest, new_json, sha
from dmvmat_ecg.preprocessing.signals import normalize, views

from .common import RUN, SOURCE, setup
from .source_data import SourceData


def prepare():
    target = setup()
    source = SourceData()
    destination = RUN / "ALIGNMENT_INPUTS_MANIFEST.json"
    if destination.exists():
        raise FileExistsError("Feature-cache manifest already exists")
    norms = [v["normalization"] for v in target.sources.values()]
    if any(n != norms[0] for n in norms):
        raise PermissionError("Original source normalization differs by seed")
    norm = norms[0]
    roles = {}
    for role in ("source_train", "source_stop", "U"):
        if role == "U":
            records = target.ids("U")
            ix = target.index(records)
            x = target.x
            directory = RUN / "alignment_inputs"
        else:
            records = source.records[role]["ids"]
            ix = np.arange(len(records))
            x = source.x[role]
            directory = SOURCE / "features"
        directory.mkdir(mode=0o700, exist_ok=True)
        stores = {}
        paths = {}
        for start in range(0, len(ix), 256):
            batch = normalize(x[ix[start : start + 256]], norm)
            arrays = views(batch)
            for name, array in zip(("wave", "rhythm", "spectrum"), arrays, strict=True):
                if name not in stores:
                    path = directory / f"{role}_{name}.npy"
                    if path.exists():
                        raise FileExistsError("Partial feature artifact is not overwritten")
                    paths[name] = str(path.relative_to(SOURCE if role != "U" else RUN))
                    stores[name] = np.lib.format.open_memmap(
                        path, mode="w+", dtype="float32", shape=(len(ix), *array.shape[1:])
                    )
                stores[name][start : start + len(array)] = array
        for store in stores.values():
            store.flush()
        root = SOURCE if role != "U" else RUN
        roles[role] = dict(
            records=len(records),
            record_ids_sha256=digest(records),
            paths=paths,
            hashes={name: sha(root / path) for name, path in paths.items()},
        )
        print(f"ALIGNMENT_FEATURES_READY role={role} records={len(records)}", flush=True)
    new_json(
        destination,
        dict(
            schema="phase4b-alignment-features-v1",
            roles=roles,
            normalization=norm,
            source_bundle_sha256=source.identity,
            target_bundle_sha256=target.identity,
            U_labels_read=False,
            U_excludes_A_and_B=True,
        ),
    )


class Features:
    def __init__(self):
        self.manifest = json.loads((RUN / "ALIGNMENT_INPUTS_MANIFEST.json").read_text())
        self.arrays = {}
        for role, entry in self.manifest["roles"].items():
            root = RUN if role == "U" else SOURCE
            self.arrays[role] = {}
            for name, path in entry["paths"].items():
                if sha(root / path) != entry["hashes"][name]:
                    raise PermissionError("Derived input feature cache changed")
                self.arrays[role][name] = np.load(root / path, mmap_mode="r", allow_pickle=False)

    def batch(self, role, ix):
        if role not in ("source_train", "source_stop", "U"):
            raise PermissionError("Unknown alignment role")
        return {
            name: torch.tensor(np.array(array[ix], copy=True), dtype=torch.float32)
            for name, array in self.arrays[role].items()
        }


if __name__ == "__main__":
    prepare()
