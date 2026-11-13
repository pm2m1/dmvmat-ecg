"""In-memory synthetic roles, observed-label semantics and label-access accounting.

These are software guards, not proof that a caller's data are synthetic. This
sprint deliberately provides no filesystem dataset loader or real-study CLI.
"""

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

MODULES = tuple([f"view_{i}" for i in range(1, 6)] + ["fusion_1", "fusion_2"])
DIMENSIONS = dict(zip(MODULES, [64] * 5 + [128, 64], strict=True))
PERMISSIONS = {
    "fit": {"fit"},
    "fit_cost_matched": {"fit", "evidence"},
    "evidence": {"source_train", "evidence"},
    "stop": {"stop"},
    "calibrate": {"calibration"},
    "select": {"selection"},
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def observed_labels(y, observed):
    y = np.asarray(y, dtype=float)
    observed = np.asarray(observed)
    if y.ndim != 2 or y.shape[1] < 1 or y.shape != observed.shape:
        raise ValueError("Aligned [record,label] targets/mask required")
    if observed.dtype != np.bool_:
        raise ValueError("Observation mask must be boolean, never inferred from missing codes")
    if not np.isin(y[observed], [0, 1]).all():
        raise ValueError("Observed labels must be finite binary values")
    # Unknown NaN/-1 placeholders cannot contaminate arithmetic or become negatives.
    return np.where(observed, y, 0.0), observed.copy()


@dataclass(frozen=True)
class SyntheticRecords:
    namespace: str
    role: str
    ids: tuple
    groups: tuple
    inputs: dict
    y: np.ndarray
    observed: np.ndarray
    label_names: tuple = ()

    def __post_init__(self):
        if not self.namespace.startswith("synthetic:"):
            raise PermissionError("Only explicit synthetic namespaces are supported")
        y, obs = observed_labels(self.y, self.observed)
        names = tuple(self.label_names) or tuple(f"label_{j}" for j in range(y.shape[1]))
        if len(names) != y.shape[1] or len(set(names)) != len(names):
            raise ValueError("Explicit unique ordered label schema required")
        n = len(y)
        if any(v is None or not str(v) for v in (*self.ids, *self.groups)):
            raise ValueError("Missing group or record identity")
        ids, groups = tuple(map(str, self.ids)), tuple(map(str, self.groups))
        if not n or len(ids) != n or len(groups) != n or len(set(ids)) != n:
            raise ValueError("Nonempty unique record identities required")
        if any(not str(v) for v in (*self.ids, *self.groups)):
            raise ValueError("Missing group or record identity")
        arrays = {}
        for key, value in self.inputs.items():
            a = np.array(value, dtype=np.float32, copy=True)
            if a.shape[0] != n or not np.isfinite(a).all():
                raise ValueError("Invalid/alignment-breaking model input")
            a.flags.writeable = False
            arrays[key] = a
        if not arrays:
            raise ValueError("Missing model inputs")
        y.flags.writeable = obs.flags.writeable = False
        object.__setattr__(self, "ids", ids)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "inputs", MappingProxyType(arrays))
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "observed", obs)
        object.__setattr__(self, "label_names", names)

    @property
    def identity(self):
        h = hashlib.sha256()
        h.update(
            digest([self.namespace, self.role, self.ids, self.groups, self.label_names]).encode()
        )
        for key, a in sorted({**self.inputs, "_y": self.y, "_observed": self.observed}.items()):
            h.update(key.encode())
            h.update(str((a.shape, a.dtype)).encode())
            h.update(a.tobytes())
        return h.hexdigest()


class SyntheticRoles:
    """Bind actual data to roles; role strings cannot relabel stored records."""

    def __init__(self, records):
        self._records = {b.role: b for b in records}
        if len(self._records) != len(records):
            raise ValueError("Duplicate role")
        if len({b.namespace for b in records}) != 1:
            raise ValueError("Mixed study namespaces")
        if len({b.label_names for b in records}) != 1:
            raise ValueError("Label order/schema differs across roles")
        self.label_names = records[0].label_names
        allowed = set().union(*PERMISSIONS.values()) | {"source_stop", "evaluation"}
        seen_ids, seen_groups = set(), set()
        for role, b in self._records.items():
            if role not in allowed:
                raise ValueError("Unknown role")
            if seen_ids.intersection(b.ids) or seen_groups.intersection(b.groups):
                raise ValueError("Patient/duplicate group or recording crosses roles")
            seen_ids.update(b.ids)
            seen_groups.update(b.groups)
        self.hashes = {r: b.identity for r, b in self._records.items()}
        self.namespace = records[0].namespace
        self.manifest_hash = digest(self.hashes)
        self.accesses = []

    def access(self, role, operation):
        if role not in PERMISSIONS.get(operation, set()):
            raise PermissionError("Operation not authorized for this data role")
        b = self._records[role]
        if b.identity != self.hashes[role]:
            raise PermissionError("Role membership, labels or inputs changed")
        self.accesses.append((operation, b))
        return b

    def label_cost(self):
        """Union records, patients and observed record-label cells, per role/op."""
        totals = {}
        for category in ("all", "target_all", "source_all", *PERMISSIONS):
            used = [b for op, b in self.accesses if category == "all" or op == category]
            if category in {"target_all", "source_all"}:
                used = [
                    b
                    for _, b in self.accesses
                    if b.role.startswith("source_") == (category == "source_all")
                ]
            rows = {(b.namespace, i) for b in used for i in b.ids}
            patients = {(b.namespace, g) for b in used for g in b.groups}
            cells = {
                (b.namespace, b.ids[i], j)
                for b in used
                for i, j in zip(*np.where(b.observed), strict=True)
            }
            totals[category] = dict(records=len(rows), patients=len(patients), labels=len(cells))
        totals["by_role"] = {
            role: dict(
                records=len(set(b.ids)),
                patients=len(set(b.groups)),
                labels=int(b.observed.sum()),
                positive_counts=((b.y == 1) & b.observed).sum(0).tolist(),
                negative_counts=((b.y == 0) & b.observed).sum(0).tolist(),
                unknown_counts=(~b.observed).sum(0).tolist(),
            )
            for role, b in self._records.items()
            if any(x.role == role for _, x in self.accesses)
        }
        return totals

    def validate_source_history(self, trained_groups):
        """Reject a source model that previously saw any target-role patient."""
        forbidden = set().union(
            *[
                set(b.groups)
                for r, b in self._records.items()
                if r not in {"source_train", "source_stop"}
            ]
        )
        if forbidden.intersection(map(str, trained_groups)):
            raise PermissionError("Source training history overlaps target participants")
        declared = set().union(
            *[
                set(b.groups)
                for r, b in self._records.items()
                if r in {"source_train", "source_stop"}
            ]
        )
        if not set(map(str, trained_groups)).issubset(declared):
            raise PermissionError("Source history contains undeclared patients")


@dataclass(frozen=True)
class Embeddings:
    features: dict
    y: np.ndarray
    observed: np.ndarray
    groups: tuple
    role: str
    checkpoint_hash: str
    preprocessing_hash: str
    data_identity: str = "synthetic-standalone-embeddings"
    label_names: tuple = ()

    def __post_init__(self):
        y, obs = observed_labels(self.y, self.observed)
        names = tuple(self.label_names) or tuple(f"label_{j}" for j in range(y.shape[1]))
        if len(names) != y.shape[1] or len(set(names)) != len(names):
            raise ValueError("Explicit unique ordered label schema required")
        groups = tuple(map(str, self.groups))
        if self.role not in {"source_train", "evidence"}:
            raise PermissionError("Only source training / target evidence may inform selector")
        if not len(y) or len(groups) != len(y) or len(set(groups)) != len(y):
            raise ValueError("Exactly one canonical recording per evidence patient required")
        if set(self.features) != set(MODULES):
            raise ValueError("All seven named module representations required")
        arrays = {}
        for name, value in self.features.items():
            z = np.array(value, dtype=float, copy=True)
            if z.ndim != 2 or z.shape[0] != len(y) or not z.shape[1] or not np.isfinite(z).all():
                raise ValueError("Finite aligned 2-D embeddings required")
            z.flags.writeable = False
            arrays[name] = z
        if not self.checkpoint_hash or not self.preprocessing_hash:
            raise ValueError("Checkpoint/preprocessing lineage required")
        y.flags.writeable = obs.flags.writeable = False
        object.__setattr__(self, "features", MappingProxyType(arrays))
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "observed", obs)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "label_names", names)


def hash_sample(groups, count, seed):
    return np.array(
        sorted(range(len(groups)), key=lambda i: digest([seed, str(groups[i])]))[:count], dtype=int
    )
