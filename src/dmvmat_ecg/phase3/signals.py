"""Native-resolution physical-signal audit and deterministic inference adapters."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import wfdb
from scipy.io import loadmat
from scipy.signal import resample_poly

LEADS = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")


def lead_order(names):
    names = tuple(str(n).upper() for n in names)
    if len(names) != 12 or set(names) != set(LEADS):
        raise ValueError("Exactly twelve unique canonical leads required")
    return [names.index(n) for n in LEADS]


def unit_factor(unit):
    key = str(unit).replace("µ", "u").replace("μ", "u").casefold()
    factors = {"mv": 1.0, "uv": 0.001, "v": 1000.0}
    if key not in factors:
        raise ValueError("Unsupported physical signal unit: " + str(unit))
    return factors[key]


def header(path):
    path = Path(path)
    h = wfdb.rdheader(str(path.with_suffix("")))
    signal_files = set(h.file_name)
    if len(signal_files) != 1:
        raise ValueError("One shared MAT signal file required")
    signal_file = next(iter(signal_files))
    if Path(signal_file).name != signal_file or Path(signal_file).suffix.casefold() != ".mat":
        raise ValueError("A local header-declared MAT basename is required")
    order = lead_order(h.sig_name)
    fs = float(h.fs)
    if not np.isfinite(fs) or fs <= 0 or fs != int(fs):
        raise ValueError("Positive integer sample frequency required")
    gain = np.asarray(h.adc_gain, float)
    baseline = np.asarray(h.baseline, float)
    if (
        gain.shape != (12,)
        or baseline.shape != (12,)
        or not np.isfinite([gain, baseline]).all()
        or (gain == 0).any()
    ):
        raise ValueError("Invalid per-lead gain/baseline")
    factors = np.array([unit_factor(u) for u in h.units])
    dx = []
    for comment in h.comments:
        if comment.strip().casefold().startswith("dx:"):
            dx = [v.strip() for v in comment.split(":", 1)[1].split(",") if v.strip()]
    if not dx or not all(v.isdigit() for v in dx):
        raise ValueError("Missing/invalid recorded diagnosis-code list")
    return {
        "fs": int(fs),
        "samples": int(h.sig_len),
        "duration": h.sig_len / fs,
        "lead_names": list(h.sig_name),
        "order": order,
        "units": list(h.units),
        "gain": gain.tolist(),
        "baseline": baseline.tolist(),
        "factors_to_mV": factors.tolist(),
        "dx": dx,
        "signal_file": signal_file,
    }


def physical_wave(path, metadata):
    val = loadmat(Path(path).parent / metadata["signal_file"], variable_names=["val"])["val"]
    if val.shape != (12, metadata["samples"]) or not np.issubdtype(val.dtype, np.number):
        raise ValueError("Header/MAT shape disagreement")
    if not np.isfinite(val).all():
        raise ValueError("Nonfinite waveform")
    wave = (
        (val.astype(np.float64) - np.array(metadata["baseline"])[:, None])
        / np.array(metadata["gain"])[:, None]
    ) * np.array(metadata["factors_to_mV"])[:, None]
    return np.ascontiguousarray(wave[metadata["order"]])


def native_identity(wave, fs):
    """Physical mV, canonical leads, complete native-rate recording; no resampling.

    Rounding at1e-9mV only standardizes floating arithmetic far below ADC precision.
    Not a near-duplicate/crop/shift detector and not proof of person identity.
    """
    values = np.round(np.asarray(wave, dtype="<f8"), 9)
    values[values == 0] = 0  # canonicalize negative zero
    h = hashlib.sha256(json.dumps([int(fs), list(values.shape), "mV-native-v1"]).encode())
    h.update(values.tobytes())
    return h.hexdigest()


def to_100hz_first10(wave, fs):
    if wave.ndim != 2 or wave.shape[0] != 12 or wave.shape[1] < 10 * fs:
        raise ValueError("Short/malformed ECGs excluded, never padded")
    if fs != int(fs) or fs <= 0:
        raise ValueError("Positive integer frequency required")
    fs = int(fs)
    x = np.asarray(wave[:, : 10 * fs], float)
    divisor = math.gcd(fs, 100)
    result = (
        x if fs == 100 else resample_poly(x, 100 // divisor, fs // divisor, axis=-1, padtype="line")
    )
    if result.shape != (12, 1000) or not np.isfinite(result).all():
        raise ValueError("Failed anti-aliased100Hz conversion")
    return result.astype(np.float32)
