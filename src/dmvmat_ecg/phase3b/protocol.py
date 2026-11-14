"""Finite adopted design. No filesystem discovery or real-cohort preparation."""

import hashlib
import itertools
import json

BUDGETS = (64, 128, 256)
SOURCE_SEEDS = (15101, 15102, 15103)
DRAW_SEEDS = (16101, 16102, 16103)
METHODS = ("head_cost", "full_cost", "random_a", "selective_a", "target_only_cost")
LABELS = ("CD-proxy-v1", "STTC-proxy-v1")
VERSION = "phase3b-adopted-v1"
UPDATES = 200
BATCH_SIZE = 32
RESERVES = 6
CEILING = 144
SOURCE_EPOCHS = 50
OPTIMIZER = dict(lr=0.0003, weight_decay=0.01, betas=(0.9, 0.99), eps=1e-8)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def seed(value):
    return int.from_bytes(hashlib.sha256(canonical(value)).digest()[:8], "big") % 2**32


def ordered(groups, rng_seed):
    if len(groups) != len(set(groups)):
        raise ValueError("Duplicate known groups")
    return tuple(sorted(groups, key=lambda g: (digest([rng_seed, g]), g)))


def fictional_episode(groups, draw_seed):
    """Pure ID-only helper. Caller must not supply real RESERVED memberships this sprint."""
    if draw_seed not in DRAW_SEEDS or len(groups) < 256:
        raise ValueError("Registered draw seed and at least 256 fictional units required")
    ids = ordered(groups, draw_seed)[:256]
    return {b: (ids[: b // 2], ids[128 : 128 + b // 2]) for b in BUDGETS}


def roster():
    jobs = {f"source/{s}": dict(kind="source", source_seed=s) for s in SOURCE_SEEDS}
    for s, d, b, method in itertools.product(SOURCE_SEEDS, DRAW_SEEDS, BUDGETS, METHODS):
        jobs[f"{s}/{d}/B{b}/{method}"] = dict(
            kind="target", source_seed=s, draw_seed=d, budget=b, method=method
        )
    return jobs


def adopted_settings():
    return dict(
        version=VERSION,
        adopted_date="2026-11-14",
        protocol_owner="Parth Mishra",
        source_seeds=SOURCE_SEEDS,
        draw_seeds=DRAW_SEEDS,
        budgets=BUDGETS,
        primary_budget=128,
        methods=METHODS,
        labels=LABELS,
        gradient_roles=["F", "G"],
        target_updates=UPDATES,
        batch_size=BATCH_SIZE,
        optimizer=OPTIMIZER,
        device="cpu",
        dtype="float32",
        torch_threads=4,
        source_epochs=SOURCE_EPOCHS,
        source_batch_size=128,
        source_optimizer=dict(lr=0.01, weight_decay=0.01, betas=(0.9, 0.99), eps=1e-8),
        source_schedule=dict(
            pct_start=0.3,
            div_factor=25,
            final_div_factor=10000,
            base_momentum=0.85,
            max_momentum=0.95,
        ),
        planned_fits=138,
        technical_reserves=RESERVES,
        attempt_ceiling=CEILING,
        target_checkpoint="fixed-update-200",
        target_stopping=False,
        calibration=False,
        faithful_arm="deferred-phase3c",
        xresnet=False,
        real_data_preparation_authorized=False,
        real_training_authorized=False,
        real_evaluation_authorized=False,
    )
