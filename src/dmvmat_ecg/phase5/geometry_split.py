"""Pure ID-only test allocation and pool-only validation allocation."""

import numpy as np

from dmvmat_ecg.phase3b.protocol import ordered
from dmvmat_ecg.phase4.safety import digest

from .common import BUDGETS, DRAWS


def allocate(feas, remaining, groups=None):
    feas, remaining = sorted(feas), sorted(remaining)
    if set(feas) & set(remaining) or len(set(feas + remaining)) != len(feas + remaining):
        raise ValueError("Historical canonical memberships overlap/duplicate")
    groups = groups if groups is not None else {r: r for r in feas + remaining}
    if set(groups) != set(feas + remaining):
        raise ValueError("Incomplete known-group map")
    mandatory = {groups[r] for r in feas}
    units = {
        g: [r for r in remaining if groups[r] == g]
        for g in sorted({groups[r] for r in remaining} - mandatory)
    }
    candidates = ordered(units, 79101)
    target = len(remaining) // 2
    extra = [r for r in remaining if groups[r] in mandatory]
    for g in candidates:
        # Keep groups intact, choose the nearer side of half without labels.
        if abs(len(extra) + len(units[g]) - target) <= abs(len(extra) - target):
            extra.extend(units[g])
    pool = sorted(feas + extra)
    test = sorted(set(remaining) - set(extra))
    if {groups[r] for r in pool} & {groups[r] for r in test}:
        raise PermissionError("Known group leakage")
    return pool, test


def episodes(pool):
    if len(pool) < 1256:
        raise ValueError("Insufficient pool for nested draws and A/B")
    result = {}
    for d in DRAWS:
        selected = ordered(pool, d)[:256]
        for b in BUDGETS:
            f, g = list(selected[: b // 2]), list(selected[128 : 128 + b // 2])
            result[f"{d}/B{b}"] = dict(F=f, G=g, hashes=dict(F=digest(f), G=digest(g)))
    return result


def validation(pool, y, ep):
    pool = list(pool)
    y = np.asarray(y)
    if y.shape != (len(pool), 2):
        raise ValueError("Pool-only aligned labels required")
    used = {r for e in ep.values() for name in ("F", "G") for r in e[name]}
    strata = {}
    for r, labels in zip(pool, y, strict=True):
        if r in used:
            continue
        key = tuple(int(v) if np.isfinite(v) else -1 for v in labels)
        if any(v not in (-1, 0, 1) for v in key):
            raise ValueError("Invalid pool label")
        strata.setdefault(key, []).append(r)
    if sum(map(len, strata.values())) < 1000:
        raise ValueError("Insufficient records disjoint from every draw")
    rng = np.random.default_rng(79102)
    remain = {k: [v[i] for i in rng.permutation(len(v))] for k, v in sorted(strata.items())}
    splits = {}
    for name in ("A", "B"):
        total = sum(map(len, remain.values()))
        expected = {k: len(v) * 500 / total for k, v in remain.items()}
        quotas = {k: int(n) for k, n in expected.items()}
        for k in sorted(quotas, key=lambda k: (-(expected[k] - quotas[k]), k))[
            : 500 - sum(quotas.values())
        ]:
            quotas[k] += 1
        splits[name] = sorted(r for k, v in remain.items() for r in v[: quotas[k]])
        remain = {k: v[quotas[k] :] for k, v in remain.items()}
    if used & (set(splits["A"]) | set(splits["B"])) or set(splits["A"]) & set(splits["B"]):
        raise PermissionError("Validation leakage")
    return splits
