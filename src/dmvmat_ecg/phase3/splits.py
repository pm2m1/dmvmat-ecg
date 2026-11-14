"""One fixed record-level joint-pattern allocation. No seed search or model scores."""

import hashlib
import json

import numpy as np

SEED = 74301
DOMAINS = {
    "Chapman-Ningbo": ("chapman_shaoxing", "ningbo"),
    "CPSC": ("cpsc_2018", "cpsc_2018_extra"),
    "Georgia": ("georgia",),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def split(ids, y, domain, seed=SEED):
    if seed != SEED:
        raise ValueError("No split-seed search")
    if len(set(ids)) != len(ids) or np.asarray(y).shape != (len(ids), 2):
        raise ValueError("Unique aligned canonical records required")
    groups = {}
    for name, label in zip(ids, y, strict=True):
        if not all(v in (0, 1) for v in label):
            raise ValueError("Observed two-label annotation proxy required")
        groups.setdefault(tuple(map(int, label)), []).append(name)
    sizes = {k: len(v) * 0.2 for k, v in groups.items()}
    quotas = {k: int(v) for k, v in sizes.items()}
    extra = round(len(ids) * 0.2) - sum(quotas.values())
    rank = sorted(
        groups, key=lambda k: (-(sizes[k] - quotas[k]), digest([seed, domain, k, "quota"]))
    )
    for k in rank[:extra]:
        quotas[k] += 1
    feas, reserved = [], []
    for k, values in sorted(groups.items()):
        ordered = sorted(values, key=lambda name: digest([seed, domain, list(k), name]))
        feas.extend(ordered[: quotas[k]])
        reserved.extend(ordered[quotas[k] :])
    return {"FEAS": sorted(feas), "RESERVED": sorted(reserved)}


def require_feas(record_id, manifest):
    if record_id not in set(manifest["FEAS"]):
        raise PermissionError("RESERVED/non-FEAS record cannot enter inference or diagnostics")


def require_fold9(folds):
    if not np.all(np.asarray(folds) == 9):
        raise PermissionError("Source reference is fold9 only; fold10 is forbidden")
