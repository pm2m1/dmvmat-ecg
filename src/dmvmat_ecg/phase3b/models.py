"""Unchanged signal backbone with a fresh, explicitly two-proxy head."""

import itertools

import numpy as np
import torch
from torch import nn

from dmvmat_ecg.models.networks import FiveViewECG
from dmvmat_ecg.phase2.adaptation import validate_owners
from dmvmat_ecg.phase2.contracts import MODULES


def fresh_model(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = FiveViewECG()
        model.prediction_head = nn.Linear(64, 2)
    model.input_schema = "ecg-12x1000"
    validate_owners(model)
    return model.cpu().float()


def inputs(model, x):
    if getattr(model, "input_schema", None) == "ecg-12x1000":
        from dmvmat_ecg.preprocessing.signals import views

        if x.shape[1:] != (12, 1000):
            raise ValueError("Exact 10-second, 100 Hz, 12-lead inputs required")
        wave, rhythm, spectrum = views(x)
        return {
            k: torch.tensor(v, dtype=torch.float32)
            for k, v in zip(("wave", "rhythm", "spectrum"), (wave, rhythm, spectrum), strict=True)
        }
    if getattr(model, "input_schema", None) == "fictional-vector":
        return dict(wave=torch.tensor(x, dtype=torch.float32))
    raise ValueError("Undeclared input schema")


def random_count_cost(model, reference, seed):
    owners = validate_owners(model)
    if set(reference) != set(MODULES) or set(reference.values()) - {"SHARE", "ADAPT"}:
        raise ValueError("Invalid seven-module reference")
    counts = [sum(p.numel() for p in owners[n].parameters()) for n in MODULES]
    bits = tuple(reference[n] == "ADAPT" for n in MODULES)
    cost = sum(c * b for c, b in zip(counts, bits, strict=True))
    eligible = [
        b
        for b in itertools.product((False, True), repeat=7)
        if sum(b) == sum(bits) and sum(c * v for c, v in zip(counts, b, strict=True)) == cost
    ]
    chosen = eligible[int(np.random.default_rng(seed).integers(len(eligible)))]
    mask = {n: "ADAPT" if b else "SHARE" for n, b in zip(MODULES, chosen, strict=True)}
    return mask, dict(
        candidate_count=len(eligible),
        equality_probability=1 / len(eligible),
        same_as_reference=mask == reference,
        adapted_count=sum(chosen),
        module_parameters=cost,
        parameter_error=0,
        head_parameters=sum(p.numel() for p in model.prediction_head.parameters()),
    )
