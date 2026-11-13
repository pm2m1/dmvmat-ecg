"""Metadata-only preparation. Never imports torch or opens waveform/test arrays."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .contract import LABELS, contract, digest, file_hash


def stratify(records, names, fractions, seed):
    """Draft's rare-pattern-first greedy squared normalized count deviation.

    Units are already one canonical record/patient; ties follow supplied role order.
    Each count term is squared AFTER division by max(1, desired count).
    """
    if len(names) != len(fractions) or not np.isclose(sum(fractions), 1):
        raise ValueError("Invalid split fractions")
    patterns = Counter(tuple(r["y"]) for r in records)
    order = sorted(
        records, key=lambda r: (patterns[tuple(r["y"])], digest([seed, r["patient_id"]]))
    )
    values = np.array([[1, *r["y"], *[1 - v for v in r["y"]]] for r in order], float)
    desired = np.asarray(fractions)[:, None] * values.sum(0)
    actual = np.zeros_like(desired)
    result = {n: [] for n in names}
    for row, v in zip(order, values, strict=True):
        before = np.square((actual - desired) / np.maximum(1, desired)).sum(1)
        after = np.square((actual + v - desired) / np.maximum(1, desired)).sum(1)
        chosen = int(np.argmin(after - before))
        actual[chosen] += v
        result[names[chosen]].append(row)
    return {k: sorted(v, key=lambda r: r["ecg_id"]) for k, v in result.items()}


def counts(rows):
    if not rows:
        return dict(records=0, patients=0, positives=[0, 0], negatives=[0, 0], joint=[0] * 4)
    y = np.array([r["y"] for r in rows])
    return dict(
        records=len(rows),
        patients=len({r["patient_id"] for r in rows}),
        positives=y.sum(0).astype(int).tolist(),
        negatives=(1 - y).sum(0).astype(int).tolist(),
        joint=[
            int(((y[:, 0] == a) & (y[:, 1] == b)).sum())
            for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]
        ],
    )


def row_hash(rows):
    return digest(rows)


def scientific_review_flags(manifest):
    """Diagnostics do not alter memberships, thresholds, endpoints or split seeds."""
    summary = manifest["summary"]
    total_joint = sum(summary[r]["joint"][3] for r in ["O", "E_reserve", "C_reserve", "A"])
    flags = []
    if total_joint > 0 and summary["A"]["joint"][3] == 0:
        flags.append(
            "Target A has zero joint CD/STTC positives despite positives in the target; "
            "formulation B must abstain for every planned G draw. Prospective split review required."
        )
    if summary["source_train"]["joint"][3] < summary["source_stop"]["joint"][3]:
        flags.append(
            "Source stopping contains more joint positives than the 85% source-training role; "
            "review rare-pattern-first marginal stratification before fitting."
        )
    return flags


def validate_manifest(manifest):
    if manifest["labels"] != list(LABELS) or manifest["contract_hash"] != digest(contract()):
        raise ValueError("Endpoint/contract changed")
    roles = manifest["roles"]
    sealed = manifest.get("sealed_role_labels", False)
    if sealed and set(manifest.get("role_vaults", {})) != set(roles):
        raise ValueError("Missing sealed role-label descriptor")
    for name, rows in roles.items():
        if row_hash(rows) != manifest["role_hashes"][name]:
            raise ValueError("Role content changed")
        if len({r["patient_id"] for r in rows}) != len(rows):
            raise ValueError("Repeated patient in a canonical role")
        if any(r["fold"] not in range(1, 10) for r in rows):
            raise ValueError("Fold10/invalid fold is prohibited")
        if sealed and any("y" in r or "observed" in r for r in rows):
            raise ValueError("Worker manifest must not contain per-record outcomes")
        if not sealed and any(r["y"] not in ([0, 0], [0, 1], [1, 0], [1, 1]) for r in rows):
            raise ValueError("Nonbinary annotation target")
        if not sealed and any(r["observed"] != [True, True] for r in rows):
            raise ValueError(
                "Pilot endpoint requires recorded annotation statuses, not invented negatives"
            )
    base = ["source_train", "source_stop", "O", "E_reserve", "C_reserve", "A"]
    ids, patients, fingerprints = set(), set(), set()
    for name in base:
        rows = roles[name]
        for key, seen in [("ecg_id", ids), ("patient_id", patients), ("raw_hash", fingerprints)]:
            values = {r[key] for r in rows}
            if seen & values:
                raise ValueError("Record/patient/known duplicate crosses base roles")
            seen.update(values)
    for draw in (4101, 4102):
        for n in (128, 512):
            active = [
                "source_train",
                "source_stop",
                f"d{draw}:n{n}:F",
                f"d{draw}:n{n}:G",
                "E",
                "C",
                "O",
            ]
            used = set()
            for name in active:
                group = {r["patient_id"] for r in roles[name]}
                if used & group:
                    raise ValueError("Active source/F/G/E/C/O overlap")
                used |= group
            if len(roles[active[2]]) != n or len(roles[active[3]]) != 256:
                raise ValueError("Adaptation budget infeasible")
            base_rows = {r["ecg_id"]: r for r in roles["A"]}
            if any(base_rows.get(r["ecg_id"]) != r for role in active[2:4] for r in roles[role]):
                raise ValueError("F/G is not a subset of A")
    for small, big in [("E", "E_reserve"), ("C", "C_reserve")]:
        if not {r["ecg_id"] for r in roles[small]} <= {r["ecg_id"] for r in roles[big]}:
            raise ValueError("Reserve membership mismatch")
    for draw in (4101, 4102):
        if roles[f"d{draw}:n128:G"] != roles[f"d{draw}:n512:G"]:
            raise ValueError("G changed across nested tiers")
        if roles[f"d{draw}:n128:F"] != roles[f"d{draw}:n512:F"][:128]:
            raise ValueError("F tiers are not nested")
    return True


def build_manifest(rows, provenance):
    cfg = contract()
    if len({r["ecg_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate record identifier")
    by_patient = {}
    for r in rows:
        if r["fold"] not in range(1, 10):
            raise ValueError("Input is not development-only")
        if 18 <= r["age"] <= 89:
            p = r["patient_id"]
            if p not in by_patient or r["ecg_id"] < by_patient[p]["ecg_id"]:
                by_patient[p] = r
    canonical = sorted(by_patient.values(), key=lambda r: r["ecg_id"])
    # Any cross-person raw identity is ambiguous: stop rather than invent linkage/exclusions.
    raw_ids = {}
    for r in canonical:
        if r["raw_hash"] in raw_ids and raw_ids[r["raw_hash"]] != r["patient_id"]:
            raise ValueError("Cross-patient identical waveform requires protocol review")
        raw_ids[r["raw_hash"]] = r["patient_id"]
    source = [r for r in canonical if r["age"] < 60]
    target = [r for r in canonical if r["age"] >= 60]
    roles = stratify(source, ["source_train", "source_stop"], [0.85, 0.15], 33002)
    roles.update(
        stratify(target, ["O", "E_reserve", "C_reserve", "A"], [0.3, 0.1, 0.1, 0.5], 33001)
    )
    for role, seed in [("E", 33003), ("C", 33005)]:
        roles[role] = sorted(
            roles[role + "_reserve"], key=lambda r: digest([seed, str(r["patient_id"])])
        )[:128]
    for draw in (4101, 4102):
        reservoir = list(roles["A"])
        permutation = np.random.default_rng(draw).permutation(len(reservoir))
        ordered = [reservoir[i] for i in permutation]
        for n in (128, 512):
            roles[f"d{draw}:n{n}:G"] = ordered[:256]
            roles[f"d{draw}:n{n}:F"] = ordered[256 : 256 + n]
    summary = {k: counts(v) for k, v in roles.items()}
    errors = []
    for role, minimum in [("source_train", 200), ("source_stop", 20), ("E", 10), ("O", 50)]:
        if min(summary[role]["positives"] + summary[role]["negatives"]) < minimum:
            errors.append(f"{role}: class support below {minimum}")
    if len(roles["A"]) < 768 or len(roles["E"]) != 128 or len(roles["C"]) != 128:
        errors.append("Role capacity below required F/G/E/C budgets")
    manifest = dict(
        schema=1,
        status="proposed-not-approved",
        labels=list(LABELS),
        contract_hash=digest(cfg),
        provenance=provenance,
        roles=roles,
        role_hashes={k: row_hash(v) for k, v in roles.items()},
        population=dict(
            development_records=len(rows),
            age_eligible_records=sum(18 <= r["age"] <= 89 for r in rows),
            canonical_patients=len(canonical),
            source=len(source),
            target=len(target),
        ),
        summary=summary,
        blockers=errors,
    )
    validate_manifest(manifest)
    return manifest


def load_metadata(root):
    """Labels only from immutable train/validation metadata; no test cache reads.

    The original CSV is hashed as bytes for identity; only demographic/file/fold
    columns are parsed. Its scp_codes column is NEVER selected/decoded.
    """
    root = Path(root)
    audit = json.loads((root / "results/dataset_audit.json").read_text())
    raw = root / "data/raw/ptbxl_v1_0_3"
    if audit["release"] != "1.0.3" or audit["class_order"] != ["NORM", "MI", "STTC", "CD", "HYP"]:
        raise ValueError("Wrong dataset/annotation schema")
    if file_hash(raw / "SHA256SUMS.txt") != audit["publisher_manifest_sha256"]:
        raise ValueError("Publisher manifest changed")
    sums = {
        line.split()[1].removeprefix("./"): line.split()[0]
        for line in (raw / "SHA256SUMS.txt").read_text().splitlines()
    }
    for name in [
        "ptbxl_database.csv",
        "scp_statements.csv",
        "LICENSE.txt",
        "ptbxl_v103_changelog.txt",
    ]:
        if file_hash(raw / name) != sums[name]:
            raise ValueError("Publisher metadata checksum mismatch: " + name)
    columns = ["ecg_id", "patient_id", "age", "strat_fold", "filename_lr"]
    demographic = pd.read_csv(raw / "ptbxl_database.csv", usecols=columns)
    demographic = demographic.loc[demographic.strat_fold.between(1, 9)].set_index("ecg_id")
    rows = []
    for old_role in ("train", "validation"):
        name = f"{old_role}_meta.npz"
        path = root / "private/cache" / name
        if file_hash(path) != audit["cache_hashes"][name]:
            raise ValueError("Development metadata checksum mismatch")
        with np.load(path, allow_pickle=False) as z:
            if not np.isin(z["fold"], range(1, 10)).all() or z["y"].shape[1] != 5:
                raise ValueError("Prohibited fold/schema in development cache")
            for i, ecg in enumerate(z["ecg_id"]):
                row = demographic.loc[int(ecg)]
                pid = int(z["patient_id"][i])
                if row.patient_id != pid or int(row.strat_fold) != int(z["fold"][i]):
                    raise ValueError("Identity mismatch between historical cache and demographics")
                age = float(row.age) if pd.notna(row.age) else -1.0
                if age >= 18 and age <= 89 and not age.is_integer():
                    raise ValueError("Noninteger age needs inclusion-boundary clarification")
                fn = str(row.filename_lr)
                if not fn.startswith("records100/") or ".." in Path(fn).parts:
                    raise ValueError("Unexpected waveform path")
                labels = z["y"][i, [3, 2]].astype(int).tolist()
                if not np.isin(z["y"][i], [0, 1]).all():
                    raise ValueError("Corrupted annotation target")
                rows.append(
                    dict(
                        ecg_id=int(ecg),
                        patient_id=pid,
                        age=age,
                        fold=int(z["fold"][i]),
                        filename=fn,
                        raw_hash=sums[fn + ".dat"],
                        header_hash=sums[fn + ".hea"],
                        y=labels,
                        observed=[True, True],
                    )
                )
    if len(rows) != 19230:
        raise ValueError("Historical eligible development membership mismatch")
    return rows, dict(
        release="1.0.3",
        metadata_sha256=sums["ptbxl_database.csv"],
        publisher_manifest_sha256=audit["publisher_manifest_sha256"],
        statements_sha256=sums["scp_statements.csv"],
        train_meta_sha256=audit["cache_hashes"]["train_meta.npz"],
        validation_meta_sha256=audit["cache_hashes"]["validation_meta.npz"],
        annotation_policy=audit["likelihood_policy"],
        known_original_physical_duplicate_detections=audit["within_partition_duplicate_detections"],
        near_duplicate_search="not performed by this metadata-only sprint",
    )
