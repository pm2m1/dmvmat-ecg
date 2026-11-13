"""Metadata allocation/support diagnostics; never scores learned predictions."""

from collections import Counter

import numpy as np
from scipy.stats import hypergeom

from dmvmat_ecg.phase2b.contract import digest

from .splits import PATTERNS, support


def original_objective_trace(records, names, fractions, seed):
    """Instrument precisely the Phase2B greedy objective, without changing roles."""
    patterns = Counter(tuple(r["y"]) for r in records)
    order = sorted(
        records, key=lambda r: (patterns[tuple(r["y"])], digest([seed, r["patient_id"]]))
    )
    values = np.array([[1, *r["y"], *[1 - v for v in r["y"]]] for r in order], float)
    desired = np.array(fractions)[:, None] * values.sum(0)
    actual = np.zeros_like(desired)
    initial_joint_gain = None
    buckets = {name: [] for name in names}
    joint_by_role = Counter()
    first_pattern = tuple(order[0]["y"]) if order else ()
    for r, v in zip(order, values, strict=True):
        before = np.square((actual - desired) / np.maximum(1, desired)).sum(1)
        after = np.square((actual + v - desired) / np.maximum(1, desired)).sum(1)
        gains = after - before
        if tuple(r["y"]) == (1, 1) and initial_joint_gain is None:
            initial_joint_gain = {name: float(g) for name, g in zip(names, gains, strict=True)}
        chosen = int(np.argmin(gains))
        actual[chosen] += v
        buckets[names[chosen]].append(r)
        if tuple(r["y"]) == (1, 1):
            joint_by_role[names[chosen]] += 1
    return dict(
        pattern_frequencies={"".join(map(str, k)): n for k, n in patterns.items()},
        first_pattern=list(first_pattern),
        first_joint_objective_deltas=initial_joint_gain,
        joint_assignments={name: joint_by_role[name] for name in names},
        role_counts={k: support(v) for k, v in buckets.items()},
    )


def expected_draw_support(reservoir, n=256, minimum=10):
    """Hypergeometric metadata support risk, not selector/predictive performance."""
    counts = support(reservoir)
    if n > len(reservoir):
        return dict(feasible=False, reason="draw exceeds reservoir")
    p = [float(hypergeom.cdf(minimum - 1, len(reservoir), k, n)) for k in counts["joint"]]
    return dict(
        feasible=True,
        sample=n,
        minimum=minimum,
        joint_order=[list(v) for v in PATTERNS],
        expected_joint_counts=[n * k / len(reservoir) for k in counts["joint"]],
        probability_cell_below_minimum=p,
        probability_any_cell_below_minimum_bounds=[max(p), min(1, sum(p))],
        interpretation="Bounds for uniform sampling without replacement; source anchor/geometry/bootstrap failures are additional, not included",
    )
