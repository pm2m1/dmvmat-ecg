"""Pinned official benchmark superdiagnostic key-membership semantics."""

import ast

import numpy as np

CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")


def label_map(statements):
    diagnostic = statements.loc[statements.diagnostic == 1]
    return {
        str(code): row.diagnostic_class
        for code, row in diagnostic.iterrows()
        if str(row.diagnostic_class) != "nan"
    }


def encode(codes, mapping):
    codes = ast.literal_eval(codes) if isinstance(codes, str) else codes
    if not isinstance(codes, dict):
        raise ValueError("scp_codes must be a dictionary")
    # The published code uses keys, NOT likelihood thresholds, even likelihood0.
    labels = {mapping[k] for k in codes if k in mapping}
    if not labels <= set(CLASSES):
        raise ValueError("Unexpected diagnostic superclass")
    return np.array([name in labels for name in CLASSES], dtype=np.float32)
