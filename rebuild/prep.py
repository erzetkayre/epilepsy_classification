"""
Data preparation for the FORTEI-ICEE 2026 revision.

Builds EEG arrays from the raw University of Bonn text files with every
preprocessing decision exposed as an explicit switch, so that each one can be
toggled independently in the ablation study.

Sets: Z,O = healthy scalp EEG; N,F = interictal intracranial; S = ictal intracranial.

Key correctness property: every sample carries a `group` id identifying the
source recording segment it came from. Windows of the same segment, and
augmented copies of the same segment, share a group id. Cross-validation must
split on groups, otherwise near-duplicate signals land on both sides of the
split and the reported accuracy is inflated.
"""

import os
import numpy as np
from scipy.signal import butter, filtfilt, decimate

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "ukb_raw")
SETS = ["Z", "O", "N", "F", "S"]
FS_RAW = 173.61

# Bonn signals are bandpass-limited to 0.53-40 Hz at acquisition (Andrzejak et al. 2001)
BAND_LOW = 0.53
BAND_HIGH = 40.0


def load_raw():
    """Load all 500 segments. Returns (signals, set_index) with signals (500, 4097)."""
    signals, set_idx = [], []
    for si, s in enumerate(SETS):
        folder = os.path.join(RAW_DIR, s)
        for i in range(1, 101):
            fn = "{}{:03d}.txt".format(s, i)
            signals.append(np.loadtxt(os.path.join(folder, fn)))
            set_idx.append(si)
    return np.asarray(signals, dtype=np.float64), np.asarray(set_idx)


def bandpass(x, fs=FS_RAW, low=BAND_LOW, high=BAND_HIGH, order=4):
    """Zero-phase Butterworth bandpass matching the acquisition passband."""
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x, axis=-1)


def augment(x, rng, shift=50, noise=0.05, scale=(0.9, 1.1)):
    """Time-shift + noise + amplitude scaling. Noise is relative to each signal's
    own std, so it survives normalization (the original used an absolute sigma of
    0.1 uV, which is ~0.05% of the ictal signal range and therefore a no-op)."""
    out = np.empty_like(x)
    for i, sig in enumerate(x):
        s = int(rng.integers(-shift, shift + 1))
        y = np.roll(sig, s)
        if s > 0:
            y[:s] = sig[0]
        elif s < 0:
            y[s:] = sig[-1]
        y = y + rng.normal(0, noise * (sig.std() + 1e-9), size=y.shape)
        out[i] = y * rng.uniform(*scale)
    return out


def window(x, groups, labels, win, hop=None):
    """Cut each segment into windows, propagating the source-segment group id."""
    hop = hop or win
    xs, gs, ys = [], [], []
    n = x.shape[1]
    for i in range(x.shape[0]):
        for st in range(0, n - win + 1, hop):
            xs.append(x[i, st:st + win])
            gs.append(groups[i])
            ys.append(labels[i])
    return np.asarray(xs), np.asarray(gs), np.asarray(ys)


def normalize(x):
    """Per-sample z-score. Computed within each sample only, so it cannot leak
    statistics between train and test."""
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True) + 1e-9
    return (x - mu) / sd


def build(task="binary", filt=True, decim=2, win=None, norm=True, seed=0):
    """Build a dataset.

    task  : 'binary' (healthy vs epileptic) or 'ternary' (healthy/interictal/ictal)
    filt  : apply the 0.53-40 Hz bandpass
    decim : integer decimation factor (2 -> 86.8 Hz, still above Nyquist for 40 Hz)
    win   : window length in samples after decimation, or None for whole segments
    norm  : per-sample z-score

    Returns dict with x (n, t, 1), y (n,), groups (n,), and the class names.
    Augmentation is NOT applied here -- it belongs inside the training fold only.
    """
    sig, set_idx = load_raw()
    groups = np.arange(len(sig))  # one group per source recording segment

    if task == "binary":
        y = (set_idx >= 2).astype(np.int64)  # Z,O -> 0 healthy ; N,F,S -> 1 epileptic
        names = ["healthy", "epileptic"]
    elif task == "ternary":
        y = np.zeros(len(set_idx), dtype=np.int64)
        y[(set_idx == 2) | (set_idx == 3)] = 1  # N,F interictal
        y[set_idx == 4] = 2                     # S ictal
        names = ["healthy", "interictal", "ictal"]
    else:
        raise ValueError("task must be 'binary' or 'ternary'")

    if filt:
        sig = bandpass(sig)
    if decim and decim > 1:
        sig = decimate(sig, decim, axis=-1, zero_phase=True)
    if win:
        sig, groups, y = window(sig, groups, y, win)
    if norm:
        sig = normalize(sig)

    return {
        "x": sig[..., None].astype(np.float32),
        "y": y,
        "groups": groups,
        "names": names,
        "fs": FS_RAW / (decim or 1),
    }


if __name__ == "__main__":
    for task in ("binary", "ternary"):
        d = build(task=task)
        print(task, "x", d["x"].shape, "y", np.bincount(d["y"]),
              "groups", len(np.unique(d["groups"])), "fs %.2f Hz" % d["fs"])
