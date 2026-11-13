"""Prospective CD/STTC contract; this file is not training authorization."""

import hashlib
import json

LABELS = ("CD", "STTC")
METHODS = (
    "head_f",
    "head_cost",
    "full_f",
    "full_cost",
    "random_a",
    "selective_a",
    "selective_b",
    "target_only",
)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def jobs():
    result = {
        "source": dict(
            kind="source",
            seed=3101,
            fit_roles=["source_train"],
            stop_role="source_stop",
            epochs=50,
            fresh=True,
        )
    }
    for draw in (4101, 4102):
        for size in (128, 512):
            prefix = f"d{draw}:n{size}"
            for method in METHODS:
                cost = method in {"head_cost", "full_cost", "target_only"}
                result[f"{prefix}:{method}"] = dict(
                    kind="target",
                    draw=draw,
                    n_fit=size,
                    method=method,
                    seed=5101 if draw == 4101 else 5102,
                    fit_roles=[f"{prefix}:F", f"{prefix}:G"] if cost else [f"{prefix}:F"],
                    evidence_role=f"{prefix}:G"
                    if method in {"random_a", "selective_a", "selective_b"}
                    else None,
                    stop_role="E",
                    updates=20 * ((size + 63) // 64),
                    look_every=(size + 63) // 64,
                    epoch_zero_eligible=method != "target_only",
                )
    assert len(result) == 33
    return result


def contract():
    return dict(
        status="draft",
        labels=list(LABELS),
        evidence_level="internal exploratory",
        folds=list(range(1, 10)),
        device="cpu",
        source_age=[18, 59],
        target_age=[60, 89],
        planned_jobs=jobs(),
        predictor_attempt_ceiling=36,
        reserve=3,
        protocol_approval=False,
        training_authorization=False,
        evaluation_authorization=False,
        source_model_parameters=212066,
        source_fraction=[0.85, 0.15],
        target_fraction=[0.3, 0.1, 0.1, 0.5],
        source_split_seed=33002,
        target_split_seed=33001,
        E_seed=33003,
        C_seed=33005,
        G_size=256,
        E_size=128,
        C_size=128,
        source_optimizer=dict(
            name="AdamW", lr=0.01, weight_decay=0.01, betas=[0.9, 0.99], eps=1e-8, batch_size=128
        ),
        source_schedule=dict(
            name="OneCycleLR",
            max_lr=0.01,
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=10000.0,
            base_momentum=0.85,
            max_momentum=0.95,
            anneal_strategy="cos",
            cycle_momentum=True,
            three_phase=False,
        ),
        target_optimizer=dict(
            name="AdamW",
            lr=0.0003,
            weight_decay=0.01,
            betas=[0.9, 0.99],
            eps=1e-8,
            batch_size=64,
            patience=5,
        ),
        primary="mean_draw_macro_AP(selective_a - full_cost), n_fit=512",
        observation_semantics="recorded superclass annotations; not clinical absence",
    )


def dry_run(manifest):
    """No ledger reservation, signal loading, learned preprocessing or prediction."""
    from .preflight import validate_manifest

    validate_manifest(manifest)
    matrix = []
    for name, spec in jobs().items():
        fit_roles = spec["fit_roles"]
        accessed = set(fit_roles + [spec["stop_role"]])
        if spec.get("evidence_role"):
            accessed.add(spec["evidence_role"])
        target = {
            r["patient_id"]
            for role in accessed
            if not role.startswith("source_")
            for r in manifest["roles"][role]
        }
        target_after_cal = target | {r["patient_id"] for r in manifest["roles"]["C"]}
        matrix.append(
            dict(
                job=name,
                spec=spec,
                role_hashes={r: manifest["role_hashes"][r] for r in accessed},
                fit_records=sum(len(manifest["roles"][r]) for r in fit_roles),
                target_label_records_before_C=len(target),
                target_label_records_with_C=len(target_after_cal),
            )
        )
    return dict(planned=len(matrix), ceiling=36, reserve=3, matrix=matrix, training_started=False)
