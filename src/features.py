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

# Gamma (32-62 Hz) is encoded on disk but excluded from the active set by default.
GAMMA_RESONATORS = [34.7, 37.2, 40.2, 43.2, 47.7, 52.6, 57.2]


# Caches follow the spike folder: spikes_raw -> features_cache, spikes_raw_x -> features_cache_x.
_SFX = Path(SPIKES_ROOT).name.removeprefix("spikes_raw")
CACHE_ROOT = Path(SPIKES_ROOT).parent / f"features_cache{_SFX}"


def cache_path(subject: str) -> Path:
    return CACHE_ROOT / f"{subject}.npz"


def load_cached(subject: str, feature_level: str = "band"):
    """(X, y, groups) from the cache written by cache_features.py."""
    p = cache_path(subject)
    if not p.is_file():
        raise FileNotFoundError(f"{p} missing; run src/cache_features.py {subject}")
    d = np.load(p)
    X = d["X_band"] if feature_level == "band" else d["X_res"]
    return X.astype(np.float64), d["y"].astype(int), d["groups"]


BINS_ROOT = Path(SPIKES_ROOT).parent / f"bins_cache{_SFX}"
FFT_ROOT = Path(SPIKES_ROOT).parent / f"fft_cache{_SFX}"


def load_bins(subject: str):
    """(X int16 [n, N_BINS, 280], y, groups) from cache_features.py --bins."""
    p = BINS_ROOT / f"{subject}.npz"
    if not p.is_file():
        raise FileNotFoundError(f"{p} missing; run src/cache_features.py --bins {subject}")
    d = np.load(p)
    return d["X_bins"], d["y"].astype(int), d["groups"]


def _bands_and_resonators(include_gamma: bool):
    """(bands tuple, {band: [f0]}) with Gamma optionally appended."""
    if not include_gamma:
        return BANDS, BAND_RESONATORS
    return BANDS + ("Gamma",), {**BAND_RESONATORS, "Gamma": GAMMA_RESONATORS}


def _clk_of(f0: float) -> int:
    return 15_360 if f0 < 10 else 153_600


# ---------------------------------------------------------------------------
# Burstiness configuration.
# ---------------------------------------------------------------------------
BURST_BIN_MS = 10.0          # local-density bin width for burstiness
N_BINS = int(WINDOW_SECONDS * 1000 / BURST_BIN_MS)


def _binned_density(spike_arr: np.ndarray, clk: int) -> np.ndarray:
    """BURST_BIN_MS-binned spike counts inside a window (empty if < 2 bins)."""
    bin_size = max(1, int(clk * BURST_BIN_MS / 1000.0))
    n_bins = len(spike_arr) // bin_size
    if n_bins < 2:
        return np.empty(0)
    return spike_arr[: n_bins * bin_size].reshape(n_bins, bin_size).sum(axis=1)


def _resonator_burstiness(spike_arr: np.ndarray, clk: int) -> float:
    """std of the BURST_BIN_MS-binned local spike density inside a window."""
    binned = _binned_density(spike_arr, clk)
    return float(binned.std()) if binned.size else 0.0


def resonator_fano(spike_arr: np.ndarray, clk: int) -> float:
    """Fano factor (var/mean) of the BURST_BIN_MS-binned density. ~1 = Poisson,
    >>1 = bursty. Scale-explicit cousin of burstiness for the temporal analysis."""
    binned = _binned_density(spike_arr, clk)
    if not binned.size:
        return 0.0
    m = binned.mean()
    return float(binned.var() / m) if m > 0 else 0.0


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

    def __init__(self, capacity: int = 8, band_resonators: dict | None = None):
        self.capacity = capacity
        self.band_resonators = band_resonators or BAND_RESONATORS
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
            for band, freqs in self.band_resonators.items():
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


def window_bins(win_desc, cache: MinuteCache) -> dict[tuple[str, str], list[np.ndarray]]:
    """dict[(ch, band)] -> per-resonator BURST_BIN_MS spike counts over the window."""
    minute_data = cache.get(
        win_desc["subject"], win_desc["trial"],
        win_desc["minute_label"], win_desc["minute"],
    )
    sec_offset = win_desc["sec_offset_in_minute"]
    out: dict[tuple[str, str], list[np.ndarray]] = {}
    for (ch, band), per_res in minute_data.items():
        rows = []
        for clk, arr in per_res:
            start = int(round(sec_offset * clk))
            rows.append(_binned_density(arr[start:start + int(round(WINDOW_SECONDS * clk))], clk))
        out[(ch, band)] = rows
    return out


def iter_window_features(
    subjects, *, feature_level: str = "band", cache: MinuteCache | None = None,
    include_gamma: bool = False,
) -> Iterator[tuple[np.ndarray, int, dict]]:
    """Yield (feature_vector, label, win_desc) for every window of `subjects`.

    feature_level:
        "band"      -> mean burstiness per (channel, band)        ->  40 dims
        "resonator" -> burstiness of every (channel, resonator)   -> 280 dims
                       (fixed-length: pad with 0 if missing)
    include_gamma:  add the Gamma band (-> 50 / 350 dims).
    """
    if feature_level not in ("band", "resonator"):
        raise ValueError(f"feature_level must be 'band' or 'resonator', got {feature_level!r}")
    bands, band_res = _bands_and_resonators(include_gamma)
    cache = cache or MinuteCache(capacity=8, band_resonators=band_res)
    for win in enumerate_windows(subjects, WINDOW_SECONDS, WINDOW_OVERLAP):
        try:
            bw = window_burstiness(win, cache)
        except FileNotFoundError:
            continue
        feats = []
        for ch in CHANNELS:
            for band in bands:
                v = bw.get((ch, band))
                if feature_level == "resonator":
                    arr = np.asarray(v) if v is not None else np.array([])
                    n_expected = len(band_res[band])
                    for i in range(n_expected):
                        feats.append(float(arr[i]) if i < len(arr) else 0.0)
                else:                                     # band level
                    feats.append(float(np.mean(v)) if v is not None and len(v) else 0.0)
        yield np.asarray(feats, dtype=np.float64), (1 if win["window_label"] == "seizure" else 0), win


def feature_names(feature_level: str = "band", include_gamma: bool = False) -> list[str]:
    """The column names matching iter_window_features's output order."""
    bands, band_res = _bands_and_resonators(include_gamma)
    names: list[str] = []
    for ch in CHANNELS:
        for band in bands:
            if feature_level == "resonator":
                for i in range(len(band_res[band])):
                    names.append(f"{ch}-{band}-r{i}")
            else:
                names.append(f"{ch}-{band}")
    return names


def build_dataset(subjects, *, feature_level: str = "band",
                  include_gamma: bool = False) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Eagerly materialize the full feature matrix for `subjects`.

    Returns (X[n, d], y[n], feature_names).
    """
    X, y = [], []
    for f, label, _ in iter_window_features(subjects, feature_level=feature_level,
                                            include_gamma=include_gamma):
        X.append(f)
        y.append(label)
    if not X:
        raise RuntimeError(f"No windows found for subjects={subjects}; is dataset/spikes_raw/ populated?")
    return np.stack(X), np.asarray(y), feature_names(feature_level, include_gamma)


def build_dataset_groups(subjects, *, feature_level: str = "band",
                         include_gamma: bool = False):
    """Like build_dataset but also returns the per-window recording (trial) group,
    for leave-one-recording-out CV. Returns (X, y, groups, feature_names)."""
    X, y, g = [], [], []
    for f, label, win in iter_window_features(subjects, feature_level=feature_level,
                                              include_gamma=include_gamma):
        X.append(f); y.append(label); g.append(win["trial"])
    if not X:
        raise RuntimeError(f"No windows found for subjects={subjects}; is dataset/spikes_raw/ populated?")
    return np.stack(X), np.asarray(y), np.asarray(g), feature_names(feature_level, include_gamma)


# ---------------------------------------------------------------------------
# FFT baseline features (classical band power + fine per-bin PSD) on the SAME
# windows enumerate_windows yields. Single source of truth for the FFT baseline
# used by the analysis; mirrors the encoder's 0.5-62 Hz Butterworth pre-filter.
# ---------------------------------------------------------------------------
_FFT_FS = 256
_FFT_BANDS = {"Delta": (0.5, 4.0), "Theta": (4.0, 8.0),
              "Alpha": (8.0, 14.0), "Beta": (14.0, 32.0), "Gamma": (32.0, 62.0)}


def _fft_bandpass(data, f_lo, f_hi, fs):
    from scipy.signal import butter, filtfilt
    ny = fs / 2.0
    b1, a1 = butter(6, f_hi / ny, btype="lowpass")
    b2, a2 = butter(6, f_lo / ny, btype="highpass")
    return filtfilt(b2, a2, filtfilt(b1, a1, data, axis=0), axis=0)


def _load_eeg_per_trial(subjects, only=None):
    """{trial_stem: {channel: bandpassed 1-D signal}} for encoded trials only."""
    import mne
    from pathlib import Path as _P
    _s2s_root = _P(SPIKES_ROOT).parent / "chbmit"
    out: dict = {}
    for subj in subjects:
        subj_dir = _s2s_root / subj
        if not subj_dir.is_dir():
            continue
        encoded = {p.name for p in (Path(SPIKES_ROOT) / subj).glob("*")}
        for edf in sorted(subj_dir.glob("*.edf")):
            if edf.stem not in encoded or (only is not None and edf.stem not in only):
                continue
            raw = mne.io.read_raw_edf(str(edf), preload=True, verbose="ERROR")
            avail = [c.strip() for c in raw.ch_names]
            picks, mapping = [], {}
            for desired in CHANNELS:
                d = desired.upper().replace(" ", "")
                for a in avail:
                    an = a.upper().replace(" ", "")
                    if (an == d or an.startswith(d)) and a not in picks:
                        picks.append(a); mapping[a] = desired; break
            if not picks:
                continue
            data = _fft_bandpass(raw.get_data(picks=picks).T, 0.5, 62.0, _FFT_FS)
            out[edf.stem] = {mapping[p]: data[:, i] for i, p in enumerate(picks)}
    return out


def _fft_window(sig, lo, hi, mode):
    """mode='band' -> summed power in [lo,hi); mode='fine' -> per-bin PSD in [lo,hi)."""
    from scipy.signal import spectrogram
    nper = min(_FFT_FS, len(sig))
    f, _, Sxx = spectrogram(sig, fs=_FFT_FS, nperseg=nper, noverlap=nper // 2,
                            scaling="density", mode="psd")
    psd = Sxx.mean(axis=1)
    m = (f >= lo) & (f < hi)
    return np.array([psd[m].sum()]) if mode == "band" else psd[m]


def _build_fft(subjects, *, fine: bool, include_gamma: bool = False):
    eeg = _load_eeg_per_trial(subjects)
    bands = list(_FFT_BANDS)[: (5 if include_gamma else 4)]
    hi = 62.0 if include_gamma else 32.0
    X, y, g, nbins = [], [], [], None
    names: list[str] = []
    for win in enumerate_windows(subjects, WINDOW_SECONDS, WINDOW_OVERLAP):
        trial_eeg = eeg.get(win["trial"])
        if trial_eeg is None:
            continue
        start = int(round(win["abs_start_s"] * _FFT_FS))
        end = start + int(round(WINDOW_SECONDS * _FFT_FS))
        feats = []
        for ch in CHANNELS:
            sig = trial_eeg.get(ch)
            if fine:
                if sig is None or end > len(sig):
                    feats.append(None)
                else:
                    feats.append(_fft_window(sig[start:end], 0.5, hi, "fine"))
            else:
                for band in bands:
                    lo, bhi = _FFT_BANDS[band]
                    feats.append(0.0 if sig is None or end > len(sig)
                                 else float(_fft_window(sig[start:end], lo, bhi, "band")[0]))
        if fine:
            good = [v for v in feats if v is not None]
            if not good:
                continue
            if nbins is None:
                nbins = len(good[0])
            vec = np.concatenate([v if v is not None else np.zeros(nbins) for v in feats])
        else:
            vec = np.asarray(feats, dtype=np.float64)
        X.append(vec)
        y.append(1 if win["window_label"] == "seizure" else 0)
        g.append(win["trial"])
    if not X:
        raise RuntimeError(f"No FFT windows for subjects={subjects}")
    if fine:
        names = [f"{ch}-fbin{i}" for ch in CHANNELS for i in range(nbins)]
    else:
        names = [f"{ch}-{b}" for ch in CHANNELS for b in bands]
    return np.stack(X), np.asarray(y), np.asarray(g), names


def build_fft_bandpower(subjects, *, include_gamma: bool = False):
    """Classical FFT band-power baseline: 40-D (or 50-D w/ gamma). (X, y, groups, names)."""
    return _build_fft(subjects, fine=False, include_gamma=include_gamma)


def build_fft_fine(subjects, *, include_gamma: bool = False):
    """Fine per-frequency-bin PSD baseline (0.5-32 or 0.5-62 Hz). (X, y, groups, names)."""
    return _build_fft(subjects, fine=True, include_gamma=include_gamma)
