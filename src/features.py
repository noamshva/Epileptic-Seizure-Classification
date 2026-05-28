"""
Feature extraction shared by every classifier in this branch.

Pipeline summary:

    raw EEG (256 Hz)                                   <- step 1.signal2spikes
        |
        v   SCTN resonators (per channel, per band)
    binary spike trains on disk:
        dataset/spikes_raw/<subject>/<trial>/<minute_label>/<minute>/<ch>/<clk>/<f0>.npz
        |
        v   2-second sliding windows (50% overlap)
    per-window burstiness vector  (per (channel, band), per resonator)
        |
        v
    classifier (SVM / RF / ... — picked by the script importing this module)

Why burstiness, not spike counts:

The resonator output is a delta-sigma stream; its mean rate is ~50% regardless
of input amplitude, so a total-count readout is blind to seizures. The seizure
signal lives in the *local-density variance*: during ictal activity the output
is bunched into bursts, during calm it is smooth. We bin each resonator's
output into BURST_BIN_MS slots inside the 2-second window and take the std of
that local density — that is the per-resonator burstiness used below.
"""
from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path
from typing import Iterator

import numpy as np


# ---------------------------------------------------------------------------
# Paths + window contract — pulled from 1.signal2spikes_chbmit.py so the
# encoder remains the single source of truth.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_S2S_PATH = _SCRIPT_DIR / "1.signal2spikes_chbmit.py"
_s2s_spec = _ilu.spec_from_file_location("signal2spikes_chbmit", _S2S_PATH)
_s2s = _ilu.module_from_spec(_s2s_spec)
_s2s_spec.loader.exec_module(_s2s)

enumerate_windows = _s2s.enumerate_windows
SPIKES_ROOT = _s2s.SPIKES_ROOT
WINDOW_SECONDS = _s2s.WINDOW_SECONDS
WINDOW_OVERLAP = _s2s.WINDOW_OVERLAP


# ---------------------------------------------------------------------------
# Active channels + bands. Gamma is encoded on disk for symmetry but
# intentionally excluded from the active feature set.
# ---------------------------------------------------------------------------
CHANNELS = [
    "FT9-FT10", "T7-P7", "T8-P8", "P7-O1",
    "F7-T7", "P8-O2", "CZ-PZ", "P3-O1",
    "FP1-F7", "C4-P4",
]

BANDS = ("Delta", "Theta", "Alpha", "Beta")

# Frequencies < 10 Hz are encoded at clk=15 360 Hz; the rest at clk=153 600 Hz.
BAND_RESONATORS = {
    "Delta": [1.1, 1.3, 1.6, 1.9, 2.2, 2.5, 2.88, 3.05, 3.39, 3.7],
    "Theta": [4.12, 4.62, 5.09, 5.45, 5.87, 6.36, 6.8, 7.6],
    "Alpha": [8.6, 10.5, 11.5, 12.8],
    "Beta":  [16.6, 19.4, 22.0, 24.8, 28.4, 30.5],
}


def _clk_of(f0: float) -> int:
    return 15_360 if f0 < 10 else 153_600


# ---------------------------------------------------------------------------
# Burstiness configuration.
# ---------------------------------------------------------------------------
BURST_BIN_MS = 10.0          # local-density bin width for burstiness


def _resonator_burstiness(spike_arr: np.ndarray, clk: int) -> float:
    """std of the BURST_BIN_MS-binned local spike density inside a window."""
    bin_size = max(1, int(clk * BURST_BIN_MS / 1000.0))
    n_bins = len(spike_arr) // bin_size
    if n_bins < 2:
        return 0.0
    binned = spike_arr[: n_bins * bin_size].reshape(n_bins, bin_size).sum(axis=1)
    return float(binned.std())


# ---------------------------------------------------------------------------
# Minute-level cache: opening ~280 .npz files per minute is fine on the first
# window of that minute but wasteful for the next 58 windows of the same
# minute, hence the small LRU.
# ---------------------------------------------------------------------------
class MinuteCache:
    """Caches per-minute spike arrays for a small number of recent minutes.

    A cached entry is keyed by (subject, trial, minute_label, minute) and holds
        dict[(channel, band)] -> list[(clk_freq, np.ndarray)]
    where each ndarray is the full 60-second binary spike train for ONE
    resonator at its native clock rate.
    """

    def __init__(self, capacity: int = 8):
        self.capacity = capacity
        self._store: dict = {}

    def get(self, subject: str, trial: str, minute_label: str, minute: int):
        key = (subject, trial, minute_label, minute)
        if key in self._store:
            return self._store[key]
        minute_dir = Path(SPIKES_ROOT) / subject / trial / minute_label / str(minute)
        per_ch_band: dict[tuple[str, str], list[tuple[int, np.ndarray]]] = {}
        for ch in CHANNELS:
            ch_dir = minute_dir / ch
            if not ch_dir.is_dir():
                continue                              # channel missing — skip
            for band, freqs in BAND_RESONATORS.items():
                arrays: list[tuple[int, np.ndarray]] = []
                for f0 in freqs:
                    clk = _clk_of(f0)
                    f_path = ch_dir / str(clk) / f"{f0}.npz"
                    if not f_path.is_file():
                        continue
                    arr = np.load(f_path)["spikes"].astype(np.int8)
                    arrays.append((clk, arr))
                if arrays:
                    per_ch_band[(ch, band)] = arrays
        if len(self._store) >= self.capacity:
            self._store.pop(next(iter(self._store)))
        self._store[key] = per_ch_band
        return per_ch_band


# Backwards-compatible alias for any external scripts that still imported the
# old private name.
_MinuteCache = MinuteCache


def window_burstiness(win_desc, cache: MinuteCache) -> dict[tuple[str, str], np.ndarray]:
    """Return dict[(ch, band)] -> np.ndarray[n_res] of per-resonator burstiness.

    `win_desc` is one item from `enumerate_windows(...)`.
    """
    minute_data = cache.get(
        win_desc["subject"], win_desc["trial"],
        win_desc["minute_label"], win_desc["minute"],
    )
    sec_offset = win_desc["sec_offset_in_minute"]
    out: dict[tuple[str, str], np.ndarray] = {}
    for (ch, band), per_res in minute_data.items():
        burst = []
        for clk, arr in per_res:
            start = int(round(sec_offset * clk))
            end = start + int(round(WINDOW_SECONDS * clk))
            burst.append(_resonator_burstiness(arr[start:end], clk))
        if burst:
            out[(ch, band)] = np.asarray(burst, dtype=np.float64)
    return out


# Backwards-compatible alias.
_window_burstiness = window_burstiness


def iter_window_features(
    subjects, *, feature_level: str = "band", cache: MinuteCache | None = None,
) -> Iterator[tuple[np.ndarray, int, dict]]:
    """Yield (feature_vector, label, win_desc) for every window of `subjects`.

    feature_level:
        "band"      -> mean burstiness per (channel, band)        ->  40 dims
        "resonator" -> burstiness of every (channel, resonator)   -> 280 dims
                       (fixed-length: pad with 0 if missing)
    """
    if feature_level not in ("band", "resonator"):
        raise ValueError(f"feature_level must be 'band' or 'resonator', got {feature_level!r}")
    cache = cache or MinuteCache(capacity=8)
    for win in enumerate_windows(subjects, WINDOW_SECONDS, WINDOW_OVERLAP):
        try:
            bw = window_burstiness(win, cache)
        except FileNotFoundError:
            continue
        feats = []
        for ch in CHANNELS:
            for band in BANDS:
                v = bw.get((ch, band))
                if feature_level == "resonator":
                    arr = np.asarray(v) if v is not None else np.array([])
                    n_expected = len(BAND_RESONATORS[band])
                    for i in range(n_expected):
                        feats.append(float(arr[i]) if i < len(arr) else 0.0)
                else:                                     # band level
                    feats.append(float(np.mean(v)) if v is not None and len(v) else 0.0)
        yield np.asarray(feats, dtype=np.float64), (1 if win["window_label"] == "seizure" else 0), win


def feature_names(feature_level: str = "band") -> list[str]:
    """The column names matching iter_window_features's output order."""
    names: list[str] = []
    for ch in CHANNELS:
        for band in BANDS:
            if feature_level == "resonator":
                for i in range(len(BAND_RESONATORS[band])):
                    names.append(f"{ch}-{band}-r{i}")
            else:
                names.append(f"{ch}-{band}")
    return names


def build_dataset(subjects, *, feature_level: str = "band") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Eagerly materialize the full feature matrix for `subjects`.

    Returns (X[n, d], y[n], feature_names).
    """
    X, y = [], []
    for f, label, _ in iter_window_features(subjects, feature_level=feature_level):
        X.append(f)
        y.append(label)
    if not X:
        raise RuntimeError(f"No windows found for subjects={subjects}; is dataset/spikes_raw/ populated?")
    return np.stack(X), np.asarray(y), feature_names(feature_level)
