"""Guarded selector/normalization bindings. No real fitting or evaluation CLI."""

import copy
import json

import numpy as np
import torch
from torch import nn

from dmvmat_ecg.models.networks import FiveViewECG
from dmvmat_ecg.phase2.adaptation import state_hash
from dmvmat_ecg.phase2.contracts import DIMENSIONS, Embeddings
from dmvmat_ecg.phase2.evidence import analyze
from dmvmat_ecg.preprocessing.signals import views

from .access import validate_source_preprocessing, verify_new_source
from .contract import LABELS, digest


def model_inputs(batch, preprocessing):
    x = ((batch["x"].astype(np.float64) - preprocessing["mean"]) / preprocessing["std"]).astype(
        np.float32
    )
    wave, rhythm, spectrum = views(x)
    return {
        k: torch.tensor(v.copy())
        for k, v in zip(("wave", "rhythm", "spectrum"), (wave, rhythm, spectrum), strict=True)
    }


def load_new_source(access, source_attempt, *, synthetic_factory=None):
    if access.ledger.get("phase") != "development":
        raise PermissionError("Source loading requires separate development authorization")
    path, lineage = verify_new_source(access, source_attempt)
    if synthetic_factory is not None and not access.ledger.identity["namespace"].startswith(
        "synthetic:"
    ):
        raise PermissionError("Synthetic source type in real namespace")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3101)
        if synthetic_factory is None:
            model = FiveViewECG()
            model.prediction_head = nn.Linear(64, 2)
        else:
            model = synthetic_factory()
    model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"), strict=True)
    if state_hash(model) != lineage["retained_state_hash"]:
        raise PermissionError("Loaded checkpoint content differs")
    return model.eval(), lineage["preprocessing"]


def selector_evidence(
    access,
    target_attempt,
    source_attempt,
    *,
    formulation="A",
    config=None,
    synthetic_factory=None,
    synthetic_reader=None,
    raw_root=None,
    dimensions=None,
):
    _, spec = access.ledger.active(target_attempt)
    if not spec.get("evidence_role"):
        raise PermissionError("This registered comparator has no selector role")
    model, stats = load_new_source(access, source_attempt, synthetic_factory=synthetic_factory)
    validate_source_preprocessing(stats, access)
    if dimensions is not None and not access.ledger.identity["namespace"].startswith("synthetic:"):
        raise PermissionError("Real module dimensions cannot be substituted")
    dimensions = dimensions or DIMENSIONS
    reps = []
    for role in ("source_train", spec["evidence_role"]):
        batch = access.load(
            role,
            "evidence",
            attempt=target_attempt,
            raw_root=raw_root,
            synthetic_reader=synthetic_reader,
        )
        with torch.no_grad():
            r = copy.deepcopy(model).eval().representations(**model_inputs(batch, stats))
        if set(r) != set(dimensions) or any(
            tuple(r[n].shape) != (len(batch["rows"]), dimensions[n]) for n in r
        ):
            raise ValueError("Module representation shape mismatch")
        reps.append(
            Embeddings(
                {n: v.cpu().numpy() for n, v in r.items()},
                batch["y"],
                batch["observed"],
                tuple(str(x["patient_id"]) for x in batch["rows"]),
                "source_train" if role == "source_train" else "evidence",
                state_hash(model),
                digest(stats),
                batch["membership_hash"],
                LABELS,
            )
        )
    result = analyze(*reps, formulation=formulation, config=config)
    result["execution_provenance"] = dict(
        namespace=access.ledger.identity["namespace"],
        target_attempt=target_attempt,
        source_attempt=source_attempt,
        manifest_hash=access.ledger.identity["manifest_hash"],
        preprocessing_hash=digest(stats),
        code_hash=access.ledger.identity["code_hash"],
    )
    access.ledger.publish(
        target_attempt, "evidence.json", json.dumps(result, sort_keys=True).encode()
    )
    return result
