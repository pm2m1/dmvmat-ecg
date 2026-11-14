"""Typed, lazy role access for fictional fixtures. No real-data backend enabled."""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .protocol import LABELS, digest


def synthetic_only(namespace):
    if not isinstance(namespace, str) or not namespace.startswith("synthetic:phase3b:"):
        raise PermissionError("Real Phase 3B preparation/execution is not authorized")


@dataclass(frozen=True)
class UnitRole:
    name: str
    records: tuple[str, ...]
    groups: tuple[str, ...]
    group_kind: str
    patient_linkage: str
    features: Callable
    outcomes: Callable
    feature_hash: str
    outcome_hash: str
    label_names: tuple = LABELS

    def __post_init__(self):
        source = self.name in {"source_train", "source_stop"}
        if self.name not in {"source_train", "source_stop", "F", "G", "O"}:
            raise ValueError("No target stopping/calibration/selection roles in adopted v1")
        if not self.records or len(set(self.records)) != len(self.records):
            raise ValueError("Nonempty unique record IDs required")
        if len(self.groups) != len(self.records) or self.label_names != LABELS:
            raise ValueError("Aligned groups and restricted proxy schema required")
        if source:
            if (self.group_kind, self.patient_linkage) != ("patient", "known"):
                raise ValueError("Known source patient linkage required")
        elif (self.group_kind, self.patient_linkage) != ("record_cluster", "unknown"):
            raise ValueError("Target records cannot be presented as verified patients")
        elif len(set(self.groups)) != len(self.groups):
            raise ValueError("One canonical target record per known duplicate cluster")

    @property
    def identity(self):
        return digest(
            dict(
                name=self.name,
                records=self.records,
                groups=self.groups,
                group_kind=self.group_kind,
                patient_linkage=self.patient_linkage,
                features=self.feature_hash,
                outcomes=self.outcome_hash,
                labels=self.label_names,
            )
        )


def array_hash(values):
    import hashlib

    h = hashlib.sha256()
    for value in values:
        a = np.ascontiguousarray(value)
        h.update(str((a.shape, a.dtype)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


class RoleData:
    def __init__(self, namespace, roles):
        synthetic_only(namespace)  # Before invoking any provider or opening any file.
        self.namespace = namespace
        self._roles = {r.name: r for r in roles}
        if len(self._roles) != len(roles):
            raise ValueError("Role names must be unique")
        seen_records, seen_groups = set(), set()
        for role in roles:
            if seen_records.intersection(role.records) or seen_groups.intersection(role.groups):
                raise PermissionError("Cross-role record/known-group intersection")
            seen_records.update(role.records)
            seen_groups.update(role.groups)
        self.access_log = []

    @property
    def identity(self):
        return digest({n: r.identity for n, r in self._roles.items()})

    def role(self, name):
        return self._roles[name]

    def read(self, name, operation):
        allowed = {
            "fit_source": {"source_train"},
            "normalize_source": {"source_train"},
            "stop_source": {"source_stop"},
            "evidence": {"source_train", "G"},
            "fit_target": {"F", "G"},
            "normalize_target_only": {"F", "G"},
        }
        if name not in allowed.get(operation, set()):
            raise PermissionError("Role/operation denied before provider access")
        r = self._roles[name]
        x = self._features(r)
        y, obs = self._outcomes(r)
        self.access_log.append((name, operation))
        return x, y, obs

    @staticmethod
    def _features(role):
        x = np.asarray(role.features(), dtype=np.float32).copy()
        if len(x) != len(role.records) or not np.isfinite(x).all():
            raise ValueError("Invalid fictional feature provider")
        if array_hash([x]) != role.feature_hash:
            raise PermissionError("Feature identity differs")
        return x

    @staticmethod
    def _outcomes(role):
        y, obs = role.outcomes()
        y, obs = np.asarray(y).copy(), np.asarray(obs).copy()
        if y.shape != (len(role.records), 2) or obs.shape != y.shape or obs.dtype != bool:
            raise ValueError("Two-label outcomes and boolean observation mask required")
        if not np.isin(y[obs], [0, 1]).all():
            raise ValueError("Unknown annotations must remain masked")
        if array_hash([y, obs]) != role.outcome_hash:
            raise PermissionError("Outcome identity differs")
        return y, obs

    def label_cost(self, names=("F", "G")):
        records, entries = set(), set()
        for name in names:
            r = self.role(name)
            _, _, obs = self.read(name, "fit_target")
            records.update(r.records)
            entries.update(
                (rid, j)
                for rid, row in zip(r.records, obs, strict=True)
                for j in range(2)
                if row[j]
            )
        return dict(
            records=len(records),
            observed_entries=len(entries),
            distinct_patients=None,
            patient_linkage="unknown",
            fitting_roles=["F", "G"],
            evidence_roles=["G"],
            stopping_records=0,
            calibration_records=0,
            selection_records=0,
        )

    def canonical_source_indices(self):
        r = self.role("source_train")
        # Numeric ECG IDs, independently of labels; never OR serial annotations.
        selected = {}
        for i, (record, group) in enumerate(zip(r.records, r.groups, strict=True)):
            if group not in selected or int(record) < int(r.records[selected[group]]):
                selected[group] = i
        return np.array([selected[g] for g in sorted(selected)], dtype=int)


@dataclass(frozen=True)
class Normalization:
    mean: float
    std: float
    training_identity: str
    origin: str

    def __post_init__(self):
        if not np.isfinite([self.mean, self.std]).all() or self.std <= 0:
            raise ValueError("Finite positive normalization scale required")

    @property
    def identity(self):
        return digest(self.__dict__)

    @classmethod
    def fit(cls, data, *, target_only=False):
        names = ("F", "G") if target_only else ("source_train",)
        operation = "normalize_target_only" if target_only else "normalize_source"
        arrays = [data.read(n, operation)[0] for n in names]
        n = sum(x.size for x in arrays)
        mean = sum(x.astype(float).sum() for x in arrays) / n
        variance = sum(np.square(x.astype(float) - mean).sum() for x in arrays) / n
        if not np.isfinite(variance) or variance <= 0:
            raise ValueError("Degenerate normalization")
        return cls(
            float(mean),
            float(np.sqrt(variance)),
            digest([data.role(n).identity for n in names]),
            "fresh-target-union" if target_only else "fresh-source-only",
        )

    def transform(self, x):
        from dmvmat_ecg.preprocessing.signals import normalize

        return normalize(x, dict(mean=self.mean, std=self.std))
