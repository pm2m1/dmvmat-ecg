"""Deterministic CPU signal views and bounded training-only augmentation."""

import numpy as np

LEADS = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")
FS = 100
WINDOW = 250
STARTS = tuple(range(0, 751, 125))


def canonical_leads(names):
    names = [n.upper() for n in names]
    if len(names) != 12 or set(names) != set(LEADS):
        raise ValueError("Expected exactly twelve named leads")
    return [names.index(n) for n in LEADS]


def normalize(x, state):
    return ((np.asarray(x, dtype=np.float64) - state["mean"]) / state["std"]).astype(np.float32)


def views(wave):
    x = np.asarray(wave, dtype=np.float32)
    if x.shape[-2] != 12 or x.shape[-1] not in (250, 1000) or not np.isfinite(x).all():
        raise ValueError("Invalid ECG window")
    rms = np.sqrt(np.mean(x * x, axis=-2))
    derivative = np.mean(np.abs(np.diff(x, axis=-1, prepend=x[..., :1])), axis=-2)
    rhythm = np.stack([x[..., 1, :], rms, derivative], axis=-2).astype(np.float32)
    # Deliberately CPU preprocessing, not an implicit MPS operator fallback.
    length = x.shape[-1]
    frequencies = np.fft.rfftfreq(length, 1 / FS)
    mask = (frequencies >= 0.5) & (frequencies <= 40)
    spectrum = np.log1p(
        np.abs(np.fft.rfft(x * np.hanning(length), axis=-1))[..., mask] / np.sqrt(length)
    ).astype(np.float32)
    return x, rhythm, spectrum


def augment(x, rng, *, training, state):
    if not training:
        raise PermissionError("Augmentation is training-only")
    return (x * rng.uniform(0.95, 1.05) + rng.normal(0, 0.005 * state["std"], x.shape)).astype(
        np.float32
    )
