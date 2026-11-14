"""Decode WFDB comments using the same comment parsing as wfdb.rdheader."""

import numpy as np
from wfdb.io.header import parse_header_content

from dmvmat_ecg.phase3.labels import targets


def unknown():
    return [float("nan"), float("nan")], [False, False]


def decode_labels(payload, frozen):
    """Only missing/malformed Dx is unknown; unlisted codes are not negatives.

    WFDB rdheader calls parse_header_content, then strips ' \t#' from each
    comment. Reuse that behavior without opening any signals or other files.
    Unlike the old raw-line decoder, spaces after # and around : are accepted.
    Conflicting duplicate Dx comments are ambiguous, never silently overwritten.
    """
    try:
        content = payload.decode("ascii")
    except UnicodeDecodeError:
        return unknown()
    _, comments = parse_header_content(content)
    lists = []
    for raw in comments:
        comment = raw.strip(" \t#")
        key, separator, value = comment.partition(":")
        if key.strip().casefold() != "dx":
            continue
        if not separator:
            return unknown()
        codes = [s.strip() for s in value.split(",")]
        if not codes or any(not code or not code.isascii() or not code.isdigit() for code in codes):
            return unknown()
        lists.append(tuple(sorted(set(codes))))
    if not lists or any(codes != lists[0] for codes in lists[1:]):
        return unknown()
    # A well-formed numeric code absent from the frozen mapping is a mapping
    # integrity error, not a reason to infer a negative or hide it as unknown.
    return targets(lists[0], frozen), [True, True]


def validity_gate(y, observed):
    """Predeclared post-decoding gate; call before any predictive metric."""
    y, observed = np.asarray(y), np.asarray(observed)
    if y.ndim != 2 or y.shape[1] != 2 or y.shape != observed.shape or observed.dtype != bool:
        raise ValueError("Aligned two-label outcomes and boolean observation masks required")
    rows = []
    for j in range(2):
        yy = y[observed[:, j], j]
        if not np.isin(yy, [0, 1]).all():
            raise ValueError("Observed outcomes must be binary")
        fraction = float(observed[:, j].mean()) if len(y) else 0.0
        rows.append(
            dict(
                label=("CD-proxy", "STTC-proxy")[j],
                observed_fraction=fraction,
                positive=int((yy == 1).sum()),
                negative=int((yy == 0).sum()),
                passed=fraction >= 0.95 and bool((yy == 1).any() and (yy == 0).any()),
            )
        )
    return dict(passed=all(r["passed"] for r in rows), minimum_observed_fraction=0.95, labels=rows)
