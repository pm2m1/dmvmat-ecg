"""Read only the WFDB record/signal lines before sealing; never read Dx comments."""

from pathlib import Path

import numpy as np
import wfdb

from dmvmat_ecg.phase3.signals import lead_order, unit_factor
from dmvmat_ecg.phase3b.jobs import atomic_new

from .access import bytes_hash


def prefix_only(stream):
    """Stream must be unbuffered. Stop exactly after one record +12 signal lines."""
    lines = []
    for _ in range(13):
        line = stream.readline()
        if not line or line.startswith(b"#"):
            raise ValueError("Malformed/short physical WFDB header prefix")
        lines.append(line)
    fields = lines[0].split()
    if len(fields) < 4 or fields[1] != b"12":
        raise ValueError("Exactly twelve leads required")
    return b"".join(lines)


def physical_metadata(path, redacted_path):
    path, redacted_path = Path(path), Path(redacted_path)
    with path.open("rb", buffering=0) as stream:
        prefix = prefix_only(stream)
    atomic_new(redacted_path, prefix)
    h = wfdb.rdheader(str(redacted_path.with_suffix("")))
    files = set(h.file_name)
    if len(files) != 1:
        raise ValueError("One shared MAT file expected")
    signal_file = next(iter(files))
    if Path(signal_file).name != signal_file or Path(signal_file).suffix.casefold() != ".mat":
        raise ValueError("Local header-declared MAT basename required")
    fs = float(h.fs)
    gain, baseline = np.asarray(h.adc_gain, float), np.asarray(h.baseline, float)
    if (
        not np.isfinite(fs)
        or fs <= 0
        or fs != int(fs)
        or gain.shape != (12,)
        or baseline.shape != (12,)
        or not np.isfinite([gain, baseline]).all()
        or (gain == 0).any()
    ):
        raise ValueError("Invalid physical sampling/gain metadata")
    return dict(
        fs=int(fs),
        samples=int(h.sig_len),
        duration=h.sig_len / fs,
        lead_names=list(h.sig_name),
        order=lead_order(h.sig_name),
        units=list(h.units),
        gain=gain.tolist(),
        baseline=baseline.tolist(),
        factors_to_mV=[unit_factor(u) for u in h.units],
        signal_file=signal_file,
        physical_prefix_hash=bytes_hash(prefix),
    )


def decode_labels(payload, frozen):
    """Called only inside the single post-seal outcome loader."""
    from dmvmat_ecg.phase3.labels import targets

    dx = None
    for line in payload.decode().splitlines():
        if line.strip().casefold().startswith("#dx:"):
            dx = [s.strip() for s in line.split(":", 1)[1].split(",") if s.strip()]
    if not dx or not all(s.isdigit() for s in dx):
        return [float("nan")] * 2, [False] * 2
    return targets(dx, frozen), [True] * 2
