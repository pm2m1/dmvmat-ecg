"""Hash-bound development-only caches; no O decoder or O descriptor exists here."""

from pathlib import Path

import numpy as np

from dmvmat_ecg.phase3b.jobs import file_hash
from dmvmat_ecg.phase3b.protocol import LABELS, digest
from dmvmat_ecg.phase3b.roles import RoleData, UnitRole, array_hash

from .boundary import NAMESPACE, implementation_hash, permit_record, real_only
from .prepare import location, read_json


class RealData(RoleData):
    """Use the pure role/mask validator, with a proper, separate real constructor."""

    def __init__(self, root, draw=16101, budget=128, *, source_only=False):
        real_only(NAMESPACE)
        self.root = Path(root).resolve()
        self.namespace = NAMESPACE
        self.folder = location(self.root)
        self.contract = read_json(self.folder / "MEMBERSHIPS.json")
        self.receipt = read_json(self.folder / "PREPARATION_RECEIPT.json")
        self.cache = read_json(self.folder / "CACHE_RECEIPT.json")
        self.verify()
        if f"{draw}/B{budget}" not in self.contract["episodes"]:
            raise PermissionError("Unregistered episode")
        self.draw, self.budget = draw, budget
        ep = self.contract["episodes"][f"{draw}/B{budget}"]
        self.target = {}
        for rid in () if source_only else (*ep["F"], *ep["G"]):
            permit_record(rid, "target_development", self.contract["membership"])
            entry = self.cache["target_annotations"][rid]
            if file_hash(entry["path"]) != entry["sha"]:
                raise PermissionError("Episode annotation artifact changed")
            self.target[rid] = read_json(entry["path"])
        roles = []
        self._cache_arrays = {}
        for name in (
            ("source_train", "source_stop")
            if source_only
            else ("source_train", "source_stop", "F", "G")
        ):
            source = name.startswith("source")
            rows = (
                self.contract["source"][name]
                if source
                else [
                    dict(
                        record=rid, group=rid, **{k: self.target[rid][k] for k in ("y", "observed")}
                    )
                    for rid in ep[name]
                ]
            )
            ids = tuple(r["record"] for r in rows)
            groups = tuple(r["group"] for r in rows)
            cache_role = name if source else "target_development"
            entry = self.cache["caches"][cache_role]
            if file_hash(entry["path"]) != entry["sha"]:
                raise PermissionError("Prepared waveform cache changed")
            if cache_role not in self._cache_arrays:
                self._cache_arrays[cache_role] = np.load(
                    entry["path"], mmap_mode="r", allow_pickle=False
                )
            index = {rid: i for i, rid in enumerate(entry["records"])}
            ix = np.array([index[rid] for rid in ids])
            y = np.array([row["y"] for row in rows], dtype="float32")
            obs = np.array([row["observed"] for row in rows], dtype=bool)

            def features(n=name, rid_list=ids, c=cache_role, indices=ix):
                self.verify()
                if n not in {"source_train", "source_stop", "F", "G"}:
                    raise PermissionError("No evaluation feature provider")
                if not n.startswith("source"):
                    for rid in rid_list:
                        permit_record(rid, "target_development", self.contract["membership"])
                return self._cache_arrays[c][indices]

            roles.append(
                UnitRole(
                    name,
                    ids,
                    groups,
                    "patient" if source else "record_cluster",
                    "known" if source else "unknown",
                    features,
                    lambda a=y, b=obs: (a.copy(), b.copy()),
                    array_hash([self._cache_arrays[cache_role][ix]]),
                    array_hash([y, obs]),
                    LABELS,
                )
            )
        seen_records, seen_groups = set(), set()
        for r in roles:
            if seen_records.intersection(r.records) or seen_groups.intersection(r.groups):
                raise PermissionError("Prohibited cross-role development group overlap")
            seen_records.update(r.records)
            seen_groups.update(r.groups)
        self._roles = {r.name: r for r in roles}
        self.access_log = []

    def verify(self):
        real_only(self.namespace)
        preflight = self.folder / "EXECUTION_PREFLIGHT.json"
        if (
            preflight.exists()
            and read_json(preflight)["implementation_hash"] != implementation_hash()
        ):
            raise PermissionError("Real provider code differs from execution preflight")
        for name, key in (
            ("MEMBERSHIPS.json", "memberships_hash"),
            ("TARGET_DEVELOPMENT_METADATA.json", "metadata_hash"),
        ):
            if file_hash(self.folder / name) != self.receipt[key]:
                raise PermissionError("Preparation identity differs")
        if self.cache["memberships_hash"] != self.receipt["memberships_hash"]:
            raise PermissionError("Cache belongs to another split")

    def role(self, name):
        if name == "O" or name not in self._roles:
            raise PermissionError("Only source training/selection and F/G roles are authorized")
        return self._roles[name]

    def read(self, name, operation):
        self.verify()
        if name == "O":
            raise PermissionError("O remains sealed; evaluation authorization is absent")
        return super().read(name, operation)

    @property
    def identity(self):
        return digest(
            dict(
                namespace=self.namespace,
                memberships=self.receipt["memberships_hash"],
                roles={n: r.identity for n, r in self._roles.items()},
            )
        )
