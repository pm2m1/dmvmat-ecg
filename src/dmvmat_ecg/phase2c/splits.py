"""Fixed-domain, observed-joint-pattern allocation with integer quotas.

No waveform, predictor, checkpoint or experiment-ledger imports. Each indivisible
unit is one patient represented by the prospectively specified canonical record.
"""

import copy
import json
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

from dmvmat_ecg.phase2.contracts import hash_sample
from dmvmat_ecg.phase2b.contract import contract, digest

SPEC_HASH = "28ec8f4d748cea15ca049ca6ba48efa8040428b1d18b3c436daaebc442f84484"
VERSION = "joint-pattern-hamilton-v1"
PATTERNS = ((0, 0), (1, 0), (0, 1), (1, 1))
BASE = ("source_train", "source_stop", "O", "E_reserve", "C_reserve", "A")


def specification():
    path = Path(__file__).resolve().parents[3] / "configs/phase2c/joint_pattern_v1.json"
    spec = json.loads(path.read_text())
    if digest(spec) != SPEC_HASH or spec["unchanged_parent_contract_hash"] != digest(contract()):
        raise PermissionError("Prospective specification/parent contract changed")
    return spec


def pattern(row):
    if len(row["y"]) != 2 or len(row["observed"]) != 2:
        raise ValueError("Two aligned annotation statuses required")
    result = []
    for value, observed in zip(row["y"], row["observed"], strict=True):
        if type(observed) is not bool or (observed and value not in (0, 1)):
            raise ValueError("Invalid observed annotation")
        result.append(str(int(value)) if observed else "?")
    return "".join(result)


def canonicalize(rows):
    """No label aggregation: differing serial diagnoses are retained in audit only."""
    ids = [r["ecg_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate recording identifier")
    eligible = defaultdict(list)
    for row in rows:
        pattern(row)
        if row["fold"] not in range(1, 10):
            raise PermissionError("Fold10 is excluded")
        if row.get("patient_id") is None:
            raise ValueError("Actual patient ID required")
        if 18 <= row["age"] <= 89:
            eligible[row["patient_id"]].append(row)
    # Check duplicate linkage in ALL eligible recordings, not just canonical ones.
    physical = defaultdict(set)
    for records in eligible.values():
        for r in records:
            physical[r["raw_hash"]].add(r["patient_id"])
    if any(len(group) != 1 for group in physical.values()):
        raise ValueError("Cross-patient duplicate requires review; no guessed linkage/exclusion")
    canonical = sorted(
        (min(v, key=lambda r: r["ecg_id"]) for v in eligible.values()), key=lambda r: r["ecg_id"]
    )
    disagreements = [0, 0]
    for records in eligible.values():
        for j in range(2):
            statuses = {r["y"][j] for r in records if r["observed"][j]}
            disagreements[j] += len(statuses) > 1
    audit = dict(
        development_records=len(rows),
        development_patients=len({r["patient_id"] for r in rows}),
        age_eligible_records=sum(len(v) for v in eligible.values()),
        canonical_patients=len(canonical),
        additional_eligible_records=sum(len(v) - 1 for v in eligible.values()),
        multiple_record_patients=sum(len(v) > 1 for v in eligible.values()),
        differing_observed_annotations_by_label=disagreements,
        differing_joint_status_patients=sum(
            len({pattern(r) for r in v}) > 1 for v in eligible.values()
        ),
        incomplete_annotation_records=sum("?" in pattern(r) for r in rows),
        multiple_record_patients_with_incomplete_annotations=sum(
            len(v) > 1 and any("?" in pattern(r) for r in v) for v in eligible.values()
        ),
        eligible_cross_age_boundary_patients=sum(
            len({r["age"] >= 60 for r in v}) > 1 for v in eligible.values()
        ),
        native_hash_duplicate_groups=sum(
            n > 1 for n in Counter(r["raw_hash"] for v in eligible.values() for r in v).values()
        ),
        canonical_membership_hash=digest(canonical),
        grouping_policy="one lowest eligible ECG ID per true patient; no OR or adjudication",
    )
    return canonical, audit


def joint_allocate(records, names, fractions, *, seed, domain):
    """Hamilton apportionment independently within each observed joint stratum.

    Each cell/role differs by <1 patient from its fractional quota. Aggregate
    role sizes are approximate (at most number-of-strata rounding error).
    This is a bounded exact-stratum alternative to greedy iterative balancing.
    """
    fs = list(map(Fraction, fractions))
    if len(fs) != len(names) or len(set(names)) != len(names) or sum(fs) != 1:
        raise ValueError("Invalid roles/fractions")
    if any(f <= 0 for f in fs):
        raise ValueError("Positive allocation fractions required")
    if len({r["patient_id"] for r in records}) != len(records):
        raise ValueError("Canonicalize patient groups before allocation")
    strata = defaultdict(list)
    for row in records:
        strata[pattern(row)].append(row)
    result = {name: [] for name in names}
    for status, rows in sorted(strata.items()):
        ordered = sorted(
            rows, key=lambda r: digest([seed, domain, status, "patient", r["patient_id"]])
        )
        desired = [len(rows) * f for f in fs]
        quotas = [int(d) for d in desired]
        remainder = len(rows) - sum(quotas)
        order = sorted(
            range(len(names)),
            key=lambda j: (
                -(desired[j] - quotas[j]),
                digest([seed, domain, status, "quota", names[j]]),
            ),
        )
        for j in order[:remainder]:
            quotas[j] += 1
        offset = 0
        for name, quota in zip(names, quotas, strict=True):
            result[name].extend(ordered[offset : offset + quota])
            offset += quota
        assert offset == len(rows)
    return {name: sorted(rows, key=lambda r: r["ecg_id"]) for name, rows in result.items()}


def support(rows):
    known = [
        [int(r["observed"][j] and r["y"][j] == c) for r in rows] for j in (0, 1) for c in (0, 1)
    ]
    negative = [sum(known[0]), sum(known[2])]
    positive = [sum(known[1]), sum(known[3])]
    joint = [sum(pattern(r) == f"{a}{b}" for r in rows) for a, b in PATTERNS]
    n = len(rows)
    return dict(
        patients=len({r["patient_id"] for r in rows}),
        records=n,
        positives=positive,
        negatives=negative,
        unknown=[n - positive[j] - negative[j] for j in range(2)],
        observed=[positive[j] + negative[j] for j in range(2)],
        joint=joint,
        unknown_joint=n - sum(joint),
        prevalence=[
            positive[j] / (positive[j] + negative[j]) if positive[j] + negative[j] else None
            for j in range(2)
        ],
        joint_proportions=[v / n if n else None for v in joint],
    )


def source_anchor_support(rows, spec):
    """Replicate analyze()'s exact class masks, without embeddings or inference.

    B restores other-label observation but keeps the index label's capped mask.
    Therefore report four cells separately for each label's source analysis.
    """
    result = []
    for label in (0, 1):
        selected = set()
        for outcome in (0, 1):
            candidates = [r for r in rows if r["observed"][label] and r["y"][label] == outcome]
            ix = hash_sample(
                [r["patient_id"] for r in candidates],
                spec["source_class_cap"],
                spec["geometry_seed"] + label * 2 + outcome,
            )
            selected.update(candidates[int(i)]["patient_id"] for i in ix)
        selected_rows = [r for r in rows if r["patient_id"] in selected]
        counts = support(selected_rows)
        counts["label"] = label
        counts["membership_hash"] = digest([r["patient_id"] for r in selected_rows])
        counts["A_support_pass"] = (
            min(counts["positives"][label], counts["negatives"][label]) >= spec["evidence_minimum"]
        )
        counts["B_support_pass"] = min(counts["joint"]) >= spec["evidence_minimum"]
        result.append(counts)
    return result


def validate(manifest):
    if manifest.get("split_version") != VERSION or manifest.get("spec_hash") != SPEC_HASH:
        raise PermissionError("Wrong split version/specification")
    if manifest.get("contract_hash") != SPEC_HASH:
        raise PermissionError("Wrong versioned split contract")
    if manifest.get("parent_contract_hash") != digest(contract()):
        raise PermissionError("Wrong scientific parent contract")
    if manifest.get("status") != "DRAFT" or manifest.get("authorization") != {
        "training": False,
        "evaluation": False,
    }:
        raise PermissionError("Metadata candidate is not execution authorization")
    roles = manifest["roles"]
    if set(roles) != set(manifest["role_hashes"]):
        raise PermissionError("Role set changed")
    for name, rows in roles.items():
        if digest(rows) != manifest["role_hashes"][name]:
            raise PermissionError("Role content changed")
        if len({r["patient_id"] for r in rows}) != len(rows):
            raise ValueError("Patient repeated within role")
        if any(r["fold"] not in range(1, 10) for r in rows):
            raise PermissionError("Fold10 is excluded")
        for r in rows:
            pattern(r)
            if not 18 <= r["age"] <= 89 or (r["age"] < 60) != name.startswith("source_"):
                raise PermissionError("Fixed age-domain ownership violated")
    for field in ("ecg_id", "patient_id", "raw_hash"):
        seen = set()
        for role in BASE:
            values = {r[field] for r in roles[role]}
            if seen & values:
                raise ValueError("Base-role group/record/duplicate leakage")
            seen |= values
    if sum(len(roles[k]) for k in BASE) != manifest["population"]["canonical_patients"]:
        raise ValueError("Incomplete coverage")
    canonical = sorted([r for k in BASE for r in roles[k]], key=lambda r: r["ecg_id"])
    if digest(canonical) != manifest["population"]["canonical_membership_hash"]:
        raise PermissionError("Canonical patient population changed")
    for role in ("E", "C"):
        reserve = {r["ecg_id"]: r for r in roles[role + "_reserve"]}
        if any(reserve.get(r["ecg_id"]) != r for r in roles[role]):
            raise ValueError("Active role differs from its reserve")
    for draw in (4101, 4102):
        for n in (128, 512):
            f, g = f"d{draw}:n{n}:F", f"d{draw}:n{n}:G"
            used = set()
            for role in ("source_train", "source_stop", f, g, "E", "C", "O"):
                group = {r["patient_id"] for r in roles[role]}
                if used & group:
                    raise ValueError("Active-role patient overlap")
                used |= group
            available = {r["ecg_id"]: r for r in roles["A"]}
            if any(available.get(r["ecg_id"]) != r for k in (f, g) for r in roles[k]):
                raise ValueError("F/G not from A")
        if roles[f"d{draw}:n128:G"] != roles[f"d{draw}:n512:G"]:
            raise ValueError("G not paired across tiers")
        if roles[f"d{draw}:n128:F"] != roles[f"d{draw}:n512:F"][:128]:
            raise ValueError("F not nested")
    return True


def build_candidate(rows, provenance):
    """One fixed specification; seed overrides or alternative real searches absent."""
    spec = specification()
    canonical, population = canonicalize(rows)
    source = [r for r in canonical if r["age"] < 60]
    target = [r for r in canonical if r["age"] >= 60]
    roles = joint_allocate(
        source, spec["source_roles"], spec["source_fractions"], seed=spec["seed"], domain="source"
    )
    roles.update(
        joint_allocate(
            target,
            spec["target_roles"],
            spec["target_fractions"],
            seed=spec["seed"],
            domain="target",
        )
    )
    for name in ("E", "C"):
        roles[name] = sorted(
            roles[name + "_reserve"],
            key=lambda r: digest([spec[f"{name}_seed"], str(r["patient_id"])]),
        )[:128]
    for draw in spec["draw_seeds"]:
        ordered = [
            roles["A"][int(i)] for i in np.random.default_rng(draw).permutation(len(roles["A"]))
        ]
        for n in spec["F_sizes"]:
            roles[f"d{draw}:n{n}:G"] = ordered[:256]
            roles[f"d{draw}:n{n}:F"] = ordered[256 : 256 + n]
    summary = {k: support(v) for k, v in roles.items()}
    blockers = []
    for role, minimum in spec["marginal_floors"].items():
        if min(summary[role]["positives"] + summary[role]["negatives"]) < minimum:
            blockers.append(f"{role}: marginal support below {minimum}")
    if len(roles["A"]) < 768 or len(roles["E"]) != 128 or len(roles["C"]) != 128:
        blockers.append("Insufficient F/G/E/C role capacity")
    if any(s["unknown_joint"] for s in summary.values()):
        blockers.append("Unknown annotation statuses need separate pilot policy review")
    anchors = source_anchor_support(roles["source_train"], spec)
    evidence = {}
    for draw in spec["draw_seeds"]:
        g = summary[f"d{draw}:n128:G"]
        evidence[str(draw)] = dict(
            A_support_pass=all(
                min(g["positives"][j], g["negatives"][j]) >= 10 and anchors[j]["A_support_pass"]
                for j in (0, 1)
            ),
            B_support_pass=all(a["B_support_pass"] for a in anchors) and min(g["joint"]) >= 10,
            G_summary=g,
            applies_to_F_sizes=[128, 512],
            geometry_validity="not measured: source D, scatter, bootstrap uncertainty require later authorized representations",
        )
        if not evidence[str(draw)]["A_support_pass"]:
            blockers.append(f"draw{draw}: insufficient primary A evidence support")
    deviations = {}
    for domain_rows, names in ((source, spec["source_roles"]), (target, spec["target_roles"])):
        natural = support(domain_rows)
        for name in names:
            s = summary[name]
            deviations[name] = dict(
                marginal_pp=[
                    100 * (s["prevalence"][j] - natural["prevalence"][j])
                    if s["prevalence"][j] is not None and natural["prevalence"][j] is not None
                    else None
                    for j in (0, 1)
                ],
                joint_pp=[
                    100 * (s["joint_proportions"][j] - natural["joint_proportions"][j])
                    if s["patients"]
                    else None
                    for j in range(4)
                ],
            )
    manifest = dict(
        split_version=VERSION,
        spec_hash=SPEC_HASH,
        contract_hash=SPEC_HASH,
        parent_contract_hash=digest(contract()),
        status="DRAFT",
        labels=["CD", "STTC"],
        authorization=dict(training=False, evaluation=False),
        provenance=provenance,
        population=population,
        domains={
            "source": support(source),
            "target": support(target),
            "complete_eligible": support(canonical),
        },
        roles=roles,
        role_hashes={k: digest(v) for k, v in roles.items()},
        summary=summary,
        source_anchors=anchors,
        evidence_support=evidence,
        deviations_pp=deviations,
        blockers=blockers,
    )
    validate(manifest)
    return copy.deepcopy(manifest)
