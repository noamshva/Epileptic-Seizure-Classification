"""
EEG / spike visualization for CHB-MIT, mirroring the Situational-Awareness paper's
Figures 4, 6, and 7 — adapted to seizure (positive) vs non-seizure (negative)
windows instead of focus/unfocus/drowsy.

Reads:
    - raw EEG:   dataset/chbmit/<subject>/<edf>           (256 Hz)
    - spikes:    dataset/spikes_raw/<subject>/<edf_stem>/<minute_label>/<minute>/<ch>/<clk>/<f0>.npz

Writes:
    analysis/figures/
        01_raw_signal_time.png                 raw EEG slice (time domain)
        02_fft_spectrogram_one_channel.png     reference FFT spectrogram + band bins
        03_sctn_one_resonator.png              one resonator's spike train + 1-s bins
        04_fft_vs_sctn_bands.png               paper Figure 4 (side-by-side)
        05_per_channel_fft_grid.png            paper-style per-channel FFT
        06_per_channel_sctn_grid.png           paper-style per-channel SCTN
        07_combined_per_channel_grid.png       per-channel FFT + SCTN side-by-side
        08_topography_fft_vs_sctn.png          paper Figure 6 (FFT row + SCTN row per band)
        09_topography_seizure_vs_non_seizure.png   paper Figure 7 style (state per row)

    analysis/stats/
        run_config.json            inputs that produced this run
        per_channel_band.csv       seizure vs non-seizure summary per (channel, band)
        channel_ranking.csv        channels sorted by SCTN seizure↔non_seizure discriminability
        run.log                    human-readable log of the run

No training, no STDP — analysis only.
"""
from __future__ import annotations

import os
import sys
import json
import math
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mne
import scipy
from scipy.signal import butter, filtfilt, spectrogram

# Force UTF-8 stdout to avoid cp1255 crashes on the Hebrew-locale Windows console.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# ============================================================
#                     CONFIG
# ============================================================
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
# Data + outputs live under PIPELINE_DATA_ROOT if set (e.g. a Drive folder in
# Colab), else under the repo.
_DATA_ROOT = Path(os.environ.get("PIPELINE_DATA_ROOT", str(_PROJECT_ROOT)))

CHBMIT_ROOT = _DATA_ROOT / "dataset" / "chbmit"
SPIKES_ROOT = _DATA_ROOT / "dataset" / "spikes_raw"
FIG_DIR = _DATA_ROOT / "analysis" / "figures"
STATS_DIR = _DATA_ROOT / "analysis" / "stats"
FIG_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR.mkdir(parents=True, exist_ok=True)


def _setup_logger() -> logging.Logger:
    """Tee log records to console AND analysis/stats/run.log."""
    logger = logging.getLogger("analyze_chbmit")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(STATS_DIR / "run.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


log = _setup_logger()

# What to analyze (env-overridable so the pipeline notebook can retarget it)
SUBJECT = os.environ.get("PIPELINE_SUBJECT", "chb01")
EDF_FILE = os.environ.get("PIPELINE_EDF", "chb01_03.edf")
EDF_STEM = EDF_FILE[:-4] if EDF_FILE.endswith(".edf") else EDF_FILE
SEIZURE_MINUTE = int(os.environ.get("PIPELINE_SEIZURE_MIN", "49"))       # minute containing the seizure
NON_SEIZURE_MINUTE = int(os.environ.get("PIPELINE_NONSEIZURE_MIN", "4")) # interictal minute (>= 5 min away)

# Bipolar channels — must match the 10 channels in 1.signal2spikes_chbmit.py
CHANNELS = [
    "FT9-FT10", "T7-P7", "T8-P8", "P7-O1",
    "F7-T7", "P8-O2", "CZ-PZ", "P3-O1",
    "FP1-F7", "C4-P4",
]

BANDS = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 14),
    "Beta":  (14, 32),
}

# Resonator banks (must match 1.signal2spikes_chbmit.py — Gamma is on disk but
# we do NOT include it in the active analysis).
CLK_RESONATORS = {
    15360: [
        1.1, 1.3, 1.6, 1.9, 2.2, 2.5, 2.88, 3.05, 3.39, 3.7,
        4.12, 4.62, 5.09, 5.45, 5.87, 6.36, 6.8, 7.6,
        8.6,
    ],
    153600: [
        10.5, 11.5, 12.8,
        16.6, 19.4, 22.0, 24.8, 28.4, 30.5,
    ],
}
MAIN_CLK = max(CLK_RESONATORS.keys())
FULL_RESONATOR_FREQS = sum(CLK_RESONATORS.values(), [])

FS = 256                       # EEG sampling rate (CHB-MIT)
FIG_DPI = 150


# ============================================================
#                10-20 ELECTRODE COORDINATES
#  Used to compute bipolar-pair midpoints for the topography plots.
# ============================================================
ELECTRODE_XY = {
    "FP1": (-0.27,  0.95), "FP2": ( 0.27,  0.95),
    "F7":  (-0.81,  0.59), "F3":  (-0.55,  0.55), "FZ": (0.0,  0.55),
    "F4":  ( 0.55,  0.55), "F8":  ( 0.81,  0.59),
    "T7":  (-0.95,  0.00), "C3":  (-0.50,  0.00), "CZ": (0.0,  0.00),
    "C4":  ( 0.50,  0.00), "T8":  ( 0.95,  0.00),
    "P7":  (-0.81, -0.59), "P3":  (-0.55, -0.55), "PZ": (0.0, -0.55),
    "P4":  ( 0.55, -0.55), "P8":  ( 0.81, -0.59),
    "O1":  (-0.27, -0.95), "O2":  ( 0.27, -0.95),
    # FT9/FT10 — approximation (lateral, slightly anterior of T7/T8)
    "FT9":  (-0.88,  0.20), "FT10": ( 0.88,  0.20),
}


def channel_xy(channel: str) -> Tuple[float, float]:
    """Midpoint of the two electrodes in a bipolar pair, in normalized [-1, 1] coords."""
    a, b = channel.split("-")
    ax, ay = ELECTRODE_XY[a]
    bx, by = ELECTRODE_XY[b]
    return (ax + bx) / 2, (ay + by) / 2


# ============================================================
#                   DATA LOADERS
# ============================================================
def bp_filter(data: np.ndarray, f_lo: float, f_hi: float, fs: int) -> np.ndarray:
    """6th-order Butterworth bandpass via two filtfilt passes (same as SA)."""
    f_ny = fs / 2.0
    b_lo = f_lo / f_ny
    b_hi = f_hi / f_ny
    b1, a1 = butter(6, b_hi, btype="lowpass")
    b2, a2 = butter(6, b_lo, btype="highpass")
    out = filtfilt(b1, a1, data, axis=0)
    out = filtfilt(b2, a2, out, axis=0)
    return out


def load_raw_edf(edf_path: Path, channels: List[str]) -> pd.DataFrame:
    """Read the EDF, extract the named bipolar channels, bandpass 0.5-62 Hz."""
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    available = [ch.strip() for ch in raw.ch_names]
    # Lenient matching (handles "T8-P8-0" → "T8-P8")
    matched: list[str] = []
    mapping: dict[str, str] = {}
    for desired in channels:
        d = desired.upper().replace(" ", "")
        for avail in available:
            a = avail.upper().replace(" ", "")
            if a == d or a.startswith(d) or d in a or a in d:
                matched.append(avail)
                mapping[avail] = desired
                break
    if not matched:
        raise ValueError(f"No requested channels found in {edf_path}")
    data = raw.get_data(picks=matched).T          # (n_samples, n_channels)
    data = bp_filter(data, 0.5, 62.0, FS)
    return pd.DataFrame(data, columns=[mapping[m] for m in matched])


def load_spike_train(subject: str, edf_stem: str, minute_label: str,
                     minute: int, channel: str, clk: int, f0: float) -> np.ndarray:
    """Load one 60-sec binary spike train for one resonator from spikes_raw."""
    path = SPIKES_ROOT / subject / edf_stem / minute_label / str(minute) / channel / str(clk) / f"{f0}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)["spikes"].astype(np.int8)


# ============================================================
#         SCTN / spike-domain helpers (ported from SA notebook)
# ============================================================
def spikes_to_events(spikes: np.ndarray) -> np.ndarray:
    """0/1 spike train → array of tick indices where a spike occurred."""
    return np.where(spikes == 1)[0]


def spikes_event_spectrogram(clk_freq: int, events: np.ndarray, window_ms: int,
                             duration_s: float | None = None) -> np.ndarray:
    """Bin spike events into fixed-time-width bins. Output: spike count per bin."""
    window = clk_freq / 1000 * window_ms
    if duration_s is None:
        duration_s = events[-1] / clk_freq + 1 if len(events) else 0
    n_ticks = duration_s * clk_freq
    n_bins = math.ceil(n_ticks / window)
    bins = np.zeros(n_bins, dtype=int)
    if len(events) == 0:
        return bins
    idx, counts = np.unique((events // window).astype(int), return_counts=True)
    mask = idx < n_bins
    bins[idx[mask]] = counts[mask]
    return bins


def events_to_max_spectrogram(resonators_by_clk: Dict[int, List[np.ndarray]],
                              duration_s: int, *, bin_ms: int = 100) -> np.ndarray:
    """SA-paper Figure 4 helper: bin each resonator's events at 10 ms, then
    downsample to `bin_ms` bins via max-pool. Returns (n_resonators, n_bins).

    `bin_ms=100` (default) reproduces the paper figure; `bin_ms=1000` matches
    the FFT plot's 1-s column width for apples-to-apples side-by-side viewing.
    """
    spikes_array = sum(resonators_by_clk.values(), start=[])
    n_resonators = len(spikes_array)
    n_bins = max(1, (duration_s * 1000) // bin_ms)
    out = np.zeros((n_resonators, n_bins))
    i = 0
    for clk_freq, events_list in resonators_by_clk.items():
        for events in events_list:
            sg = spikes_event_spectrogram(clk_freq, events, window_ms=10,
                                          duration_s=duration_s)
            # Max-pool over groups of (n_total / n_bins) consecutive 10-ms bins
            group = len(sg) // n_bins
            if group < 1:
                group = 1
            usable = group * n_bins
            out[i] = sg[:usable].reshape(n_bins, group).max(axis=1)
            out[i] *= MAIN_CLK / clk_freq                    # normalize across clocks
            out[i] -= np.mean(out[i])
            out[i][out[i] < 0] = 0
            i += 1
    return out


def spikes_to_bands(spikes_spectrogram: np.ndarray, freqs: List[float],
                    normalize_band: bool = True) -> np.ndarray:
    """Aggregate per-resonator binned spikes into per-band binned spikes.

    Returns shape (len(BANDS), n_bins).
    """
    freqs_arr = np.array(freqs)
    bands_spec = np.zeros((len(BANDS), spikes_spectrogram.shape[1]))
    for i, (lf, hf) in enumerate(BANDS.values()):
        mask = (freqs_arr >= lf) & (freqs_arr < hf)
        bands_spec[i] = np.sum(spikes_spectrogram[mask], axis=0)
        if normalize_band:
            denom = float(np.sum(freqs_arr[mask]))
            if denom > 0:
                bands_spec[i] /= denom
    return bands_spec


# ============================================================
#                     PLOT HELPERS
# ============================================================
def plot_spectrogram(data: np.ndarray, channel_name: str, fs: int,
                     fmin: float, fmax: float, show: bool = True,
                     ax=None) -> np.ndarray:
    """STFT spectrogram, same configuration as the SA notebook."""
    f, t, Sxx = spectrogram(data, fs, noverlap=0, nperseg=fs)
    Sxx = Sxx[:, :-1]
    if show:
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 5))
        ax.set_ylim(fmin, fmax)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"ch={channel_name} fs={fs} band=({fmin}-{fmax})Hz")
        im = ax.imshow(
            Sxx,
            aspect="auto",
            cmap="jet",
            origin="lower",
            extent=[t[0], t[-1], f[0], f[-1]],
        )
    return Sxx


def fft_spectrogram_to_bins(Sxx: np.ndarray, window: int = 1,
                            normalize_band: bool = True) -> np.ndarray:
    """Aggregate FFT spectrogram rows (1-Hz bins) into the 4 EEG sub-bands."""
    bins_count = max(1, Sxx.shape[1] // window)
    out = np.zeros((len(BANDS), bins_count))
    for ch_i, (lf, hf) in enumerate(BANDS.values()):
        lf_i, hf_i = int(np.floor(lf)), int(np.ceil(hf))
        denom = (hf_i - lf_i) if normalize_band else 1
        for i in range(bins_count):
            out[ch_i, i] = Sxx[lf_i:hf_i, i * window:(i + 1) * window].sum() / denom
    return out


def plot_bins(Sxx: np.ndarray, duration: float, labels: List[str],
              ax=None, fig=None, vmin=None, vmax=None,
              colorbar: bool = True, gridlines: bool = True,
              fontsize: int = 10) -> None:
    """SA-style 'banded heat strip' plot — rows = bands, x = time."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(
        Sxx,
        aspect="auto",
        cmap="jet",
        origin="lower",
        extent=[0, duration, 0, len(Sxx)],
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_yticks(np.arange(0.5, len(Sxx) + 0.5), labels=labels, fontsize=fontsize)
    if gridlines:
        ax.set_yticks(np.arange(0, len(labels)), minor=True)
        ax.grid(which="minor", color="grey", linestyle="-", linewidth=1)
    if colorbar and fig is not None:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_ticks([])
    if fig is not None:
        fig.tight_layout()


def topography_map(per_channel_value: Dict[str, float],
                   fig=None, ax=None,
                   vmin: float | None = None,
                   vmax: float | None = None,
                   title: str = "") -> None:
    """Head-shape contour topo (SA Figure 6/7 style), adapted for bipolar pairs
    (each channel placed at its electrode-pair midpoint)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))

    # Per-channel (x, y, value) records
    records = []
    for ch, val in per_channel_value.items():
        xy = channel_xy(ch)
        records.append({"channel": ch, "x": xy[0], "y": xy[1], "value": val})
    df = pd.DataFrame(records)

    xx = df["x"].to_numpy()
    yy = df["y"].to_numpy()
    z = df["value"].to_numpy()

    # Interpolate onto a 300×300 grid
    N = 300
    xi = np.linspace(-1.05, 1.05, N)
    yi = np.linspace(-1.05, 1.05, N)
    zi = scipy.interpolate.griddata((xx, yy), z, (xi[None, :], yi[:, None]),
                                    method="cubic")

    # Mask everything outside the head circle
    radius = 1.0
    for i in range(N):
        for j in range(N):
            r = np.hypot(xi[i], yi[j])
            if r > radius:
                zi[j, i] = np.nan

    if vmin is None:
        vmin = float(np.nanmin(zi)) if np.any(~np.isnan(zi)) else 0.0
    if vmax is None:
        vmax = float(np.nanmax(zi)) if np.any(~np.isnan(zi)) else 1.0

    levels = np.linspace(vmin, vmax, 60)
    cs = ax.contourf(xi, yi, zi, levels=levels, cmap=plt.cm.jet, zorder=1)
    ax.contour(xi, yi, zi, levels=np.linspace(vmin, vmax, 12), colors="grey",
               linewidths=0.5, zorder=2)

    # Plot the electrode-pair midpoints
    ax.scatter(xx, yy, marker="o", c="k", s=15, zorder=3)
    for x, y, ch in zip(xx, yy, df["channel"]):
        ax.annotate(ch, (x, y), fontsize=6, ha="center", va="bottom",
                    color="black", xytext=(0, 4), textcoords="offset points")

    # Head outline + ears + nose
    head = plt.Circle((0, 0), radius, edgecolor="k", facecolor="none",
                      linewidth=1.5, zorder=4)
    ax.add_patch(head)
    ax.add_patch(mpatches.Ellipse(xy=(-radius, 0), width=0.1, height=0.2,
                                  edgecolor="k", facecolor="w", zorder=0))
    ax.add_patch(mpatches.Ellipse(xy=( radius, 0), width=0.1, height=0.2,
                                  edgecolor="k", facecolor="w", zorder=0))
    ax.add_patch(mpatches.Polygon(xy=[(-0.08, radius), (0.08, radius), (0, radius + 0.12)],
                                  edgecolor="k", facecolor="w", zorder=0))

    ax.set_aspect("equal")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=12)
    return cs


# ============================================================
#         BAND-LEVEL SPECTROGRAMS PER CHANNEL
# ============================================================
def fft_bin_spectrograms_per_channel(df: pd.DataFrame, t_start_s: int, t_end_s: int,
                                     fft_bin_window: int = 1) -> Dict[str, np.ndarray]:
    """Compute the per-band FFT spectrogram for every channel in df, within [t_start_s, t_end_s]."""
    out: Dict[str, np.ndarray] = {}
    for ch in df.columns:
        sliced = df[ch].values[t_start_s * FS: t_end_s * FS]
        Sxx = plot_spectrogram(sliced, channel_name=ch, fs=FS, fmin=0.5, fmax=64, show=False)
        out[ch] = fft_spectrogram_to_bins(Sxx, window=fft_bin_window, normalize_band=False)
    return out


def sctn_bin_spectrograms_per_channel(subject: str, edf_stem: str,
                                       minute_label: str, minute: int) -> Dict[str, np.ndarray]:
    """Compute the per-band SCTN spikegram for every channel for one minute."""
    out: Dict[str, np.ndarray] = {}
    for ch in CHANNELS:
        resonators_by_clk: Dict[int, List[np.ndarray]] = {}
        for clk, freqs in CLK_RESONATORS.items():
            arrs = []
            for f0 in freqs:
                try:
                    spikes = load_spike_train(subject, edf_stem, minute_label,
                                              minute, ch, clk, f0)
                except FileNotFoundError:
                    continue
                arrs.append(spikes_to_events(spikes))
            if arrs:
                resonators_by_clk[clk] = arrs
        if not resonators_by_clk:
            continue
        max_spec = events_to_max_spectrogram(resonators_by_clk, duration_s=60)
        bands = spikes_to_bands(max_spec, FULL_RESONATOR_FREQS, normalize_band=False)
        out[ch] = bands
    return out


# ============================================================
#                  PLOTS
# ============================================================
def plot_raw_signal_time(df: pd.DataFrame, channel: str,
                         t_start_s: int, t_end_s: int,
                         title_suffix: str = "") -> None:
    """SA notebook's 'Signal on Time domain' + plt.specgram in one figure."""
    sliced = df[channel].values[t_start_s * FS: t_end_s * FS]
    duration = t_end_s - t_start_s
    fig, axs = plt.subplots(2, 1, figsize=(14, 6))
    axs[0].plot(np.linspace(0, duration, len(sliced)), sliced)
    axs[0].set_title(f"Signal on time domain ({channel}{title_suffix})")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Amplitude")
    axs[1].set_title(f"Spectrogram of {channel}{title_suffix}")
    axs[1].specgram(sliced, Fs=FS)
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Frequency (Hz)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_raw_signal_time.png", dpi=FIG_DPI)
    plt.close(fig)


def plot_fft_one_channel(df: pd.DataFrame, channel: str,
                         t_start_s: int, t_end_s: int) -> None:
    """Full FFT spectrogram + the 4-band bin plot for ONE channel."""
    sliced = df[channel].values[t_start_s * FS: t_end_s * FS]
    duration = t_end_s - t_start_s

    fig, axs = plt.subplots(2, 1, figsize=(14, 8))
    Sxx = plot_spectrogram(sliced, channel_name=channel, fs=FS,
                           fmin=0.5, fmax=64, show=True, ax=axs[0])
    fft_bins = fft_spectrogram_to_bins(Sxx, window=1)
    labels = [f"{lf}-{hf} ({band})" for band, (lf, hf) in BANDS.items()]
    plot_bins(fft_bins, duration, labels, ax=axs[1], fig=fig, colorbar=True)
    axs[1].set_title("(a) FFT band bins")
    axs[1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_fft_spectrogram_one_channel.png", dpi=FIG_DPI)
    plt.close(fig)


def plot_sctn_one_resonator(subject: str, edf_stem: str, minute_label: str,
                            minute: int, channel: str, f0: float = 4.0) -> None:
    """One resonator's spike train + its 1-s binned counts (SA notebook's 'spikes in bins of 1s')."""
    clk = 15360 if f0 < 10 else 153600
    try:
        spikes = load_spike_train(subject, edf_stem, minute_label, minute,
                                  channel, clk, f0)
    except FileNotFoundError as e:
        print(f"  warning: spike file missing for {channel} f0={f0} ({e}); skipping fig 03")
        return

    events = spikes_to_events(spikes)
    duration = 60
    y_density = np.convolve(spikes.astype(np.float32),
                            np.ones(500, dtype=np.float32), "valid")

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, axs = plt.subplots(2, 1, figsize=(14, 6))
    axs[0].plot(np.linspace(0, duration, len(y_density)), y_density)
    axs[0].set_xlim(0, duration)              # match imshow's edge-to-edge extent below
    axs[0].set_title(f"Spike density (500-tick convolution) — {channel} f0={f0} Hz, {minute_label} minute {minute}")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Spikes / 500-tick window")

    binned = spikes_event_spectrogram(clk, events, window_ms=1000, duration_s=duration)
    # Disable plot_bins' own colorbar (it shrinks the axis) and attach the
    # colorbar via make_axes_locatable so the upper and lower panels keep
    # identical plot widths.
    plot_bins(binned[None, :], duration, labels=[f"{f0:.2f} Hz"],
              ax=axs[1], fig=fig, colorbar=False)
    axs[1].set_title("Spikes in 1-s bins")
    axs[1].set_xlabel("Time (s)")

    # Same right-side cutout on the top panel — invisible spacer — so both
    # subplots end up with matching x-axis extents.
    top_div = make_axes_locatable(axs[0])
    spacer = top_div.append_axes("right", size="2%", pad=0.1)
    spacer.set_visible(False)

    bot_div = make_axes_locatable(axs[1])
    cax = bot_div.append_axes("right", size="2%", pad=0.1)
    fig.colorbar(axs[1].images[0], cax=cax).set_ticks([])

    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_sctn_one_resonator.png", dpi=FIG_DPI)
    plt.close(fig)


def plot_fft_vs_sctn_bands(df: pd.DataFrame, channel: str,
                           subject: str, edf_stem: str,
                           minute_label: str, minute: int,
                           *,
                           sctn_bin_ms: int = 100,
                           out_name: str = "04_fft_vs_sctn_bands.png") -> None:
    """Paper Figure 4: top = FFT band bins, bottom = SCTN band spikegram, same minute.

    `sctn_bin_ms` controls the SCTN time-bin width.
        * 100 ms (default) reproduces the SA-paper figure — the SCTN side
          shows finer time structure than the 1-s FFT side.
        * 1000 ms forces both panels to the same column width for a strict
          apples-to-apples comparison.
    """
    t_start_s = minute * 60
    t_end_s = (minute + 1) * 60
    duration = t_end_s - t_start_s
    sliced = df[channel].values[t_start_s * FS: t_end_s * FS]
    Sxx = plot_spectrogram(sliced, channel_name=channel, fs=FS,
                           fmin=0.5, fmax=64, show=False)
    fft_bins = fft_spectrogram_to_bins(Sxx, window=1, normalize_band=False)

    # SCTN spikegram for the same minute
    resonators_by_clk: Dict[int, List[np.ndarray]] = {}
    for clk, freqs in CLK_RESONATORS.items():
        arrs = []
        for f0 in freqs:
            try:
                spikes = load_spike_train(subject, edf_stem, minute_label,
                                          minute, channel, clk, f0)
            except FileNotFoundError:
                continue
            arrs.append(spikes_to_events(spikes))
        if arrs:
            resonators_by_clk[clk] = arrs
    max_spec = events_to_max_spectrogram(resonators_by_clk, duration_s=duration,
                                         bin_ms=sctn_bin_ms)
    sctn_bands = spikes_to_bands(max_spec, FULL_RESONATOR_FREQS, normalize_band=False)

    labels = [f"{lf}-{hf} ({band})" for band, (lf, hf) in BANDS.items()]

    fig, axs = plt.subplots(2, 1, figsize=(14, 10))
    plot_bins(fft_bins, duration, labels, ax=axs[0], fig=fig, colorbar=True)
    axs[0].set_title(f"(a) FFT band spectrogram (1-s bins)", fontsize=18)
    axs[0].set_ylabel("Band")
    plot_bins(sctn_bands, duration, labels, ax=axs[1], fig=fig, colorbar=True)
    axs[1].set_title(f"(b) SCTN spikegram ({sctn_bin_ms}-ms bins)", fontsize=18)
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Band")
    fig.suptitle(f"{channel} — {minute_label} minute {minute}", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / out_name, dpi=FIG_DPI)
    plt.close(fig)


def plot_selected_3x2_fft_vs_sctn(
    df: pd.DataFrame, subject: str, edf_stem: str,
    minute_label: str, minute: int,
    channels: List[str],
    *,
    sctn_bin_ms: int = 1000,
    out_name: str = "04c_fft_vs_sctn_3channels.png",
) -> None:
    """3 (channels) × 2 (FFT, SCTN) grid of band-binned heatmaps for one minute.

    Each row = one channel; left column is the FFT band spectrogram, right
    column is the SCTN band spikegram. Both columns use the same time-bin
    width (`sctn_bin_ms`, default 1 000 ms) so the comparison is column-aligned.

    A separate colour scale per column keeps each pipeline's dynamic range
    interpretable (FFT band-power and SCTN spike-count are in different units).
    """
    t_start_s = minute * 60
    t_end_s = (minute + 1) * 60
    duration = t_end_s - t_start_s

    # --- FFT side: per-channel band bins at 1-s columns ---------------------
    fft_per_ch: Dict[str, np.ndarray] = {}
    for ch in channels:
        if ch not in df.columns:
            continue
        sliced = df[ch].values[t_start_s * FS: t_end_s * FS]
        Sxx = plot_spectrogram(sliced, channel_name=ch, fs=FS,
                               fmin=0.5, fmax=64, show=False)
        fft_per_ch[ch] = fft_spectrogram_to_bins(Sxx, window=1, normalize_band=False)

    # --- SCTN side: per-channel band spikegram at the requested bin width ---
    sctn_per_ch: Dict[str, np.ndarray] = {}
    for ch in channels:
        resonators_by_clk: Dict[int, List[np.ndarray]] = {}
        for clk, freqs in CLK_RESONATORS.items():
            arrs = []
            for f0 in freqs:
                try:
                    spikes = load_spike_train(subject, edf_stem, minute_label,
                                              minute, ch, clk, f0)
                except FileNotFoundError:
                    continue
                arrs.append(spikes_to_events(spikes))
            if arrs:
                resonators_by_clk[clk] = arrs
        if not resonators_by_clk:
            continue
        max_spec = events_to_max_spectrogram(resonators_by_clk, duration_s=duration,
                                             bin_ms=sctn_bin_ms)
        sctn_per_ch[ch] = spikes_to_bands(max_spec, FULL_RESONATOR_FREQS,
                                          normalize_band=False)

    # --- Layout ------------------------------------------------------------
    fft_vmax  = max((s.max() for s in fft_per_ch.values()),  default=1.0)
    sctn_vmax = max((s.max() for s in sctn_per_ch.values()), default=1.0)
    labels = [f"{lf}-{hf} ({band})" for band, (lf, hf) in BANDS.items()]

    nrows, ncols = len(channels), 2
    fig, axs = plt.subplots(nrows, ncols, figsize=(14, 2.9 * nrows))
    if nrows == 1:
        axs = np.array([axs])

    for r, ch in enumerate(channels):
        ax_fft = axs[r, 0]
        ax_sctn = axs[r, 1]

        if ch in fft_per_ch:
            plot_bins(fft_per_ch[ch], duration, labels,
                      ax=ax_fft, fig=None, colorbar=False, gridlines=True,
                      vmin=0, vmax=fft_vmax, fontsize=8)
        else:
            ax_fft.set_visible(False)

        if ch in sctn_per_ch:
            plot_bins(sctn_per_ch[ch], duration, [""] * len(BANDS),
                      ax=ax_sctn, fig=None, colorbar=False, gridlines=True,
                      vmin=0, vmax=sctn_vmax, fontsize=8)
        else:
            ax_sctn.set_visible(False)

        # Row label = channel name, on the left of the FFT panel
        ax_fft.set_ylabel(ch, fontsize=11, fontweight="bold", rotation=90,
                          labelpad=10)
        # Only bottom row gets an x-axis label
        if r == nrows - 1:
            ax_fft.set_xlabel("Time (s)")
            ax_sctn.set_xlabel("Time (s)")

    # Column headers
    axs[0, 0].set_title("FFT band spectrogram", fontsize=14, pad=8)
    axs[0, 1].set_title("SCTN spikegram", fontsize=14, pad=8)

    state_label = {"seizure": "Seizure",
                   "non_seizure": "Non-seizure"}.get(minute_label, minute_label)
    fig.suptitle(f"FFT vs SCTN — {state_label}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG_DIR / out_name, dpi=FIG_DPI)
    plt.close(fig)


def _per_channel_grid(per_ch: Dict[str, np.ndarray], duration: float,
                      title: str, fname: str) -> None:
    """Plot a 5x2 grid of band-binned spectrograms (one per channel)."""
    n = len(CHANNELS)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(14, 2.6 * nrows))
    labels = [f"{lf}-{hf} ({band})" for band, (lf, hf) in BANDS.items()]
    vmax = max((spec.max() for spec in per_ch.values()), default=1.0)

    for i, ch in enumerate(CHANNELS):
        ax = axs[i // ncols][i % ncols]
        if ch not in per_ch:
            ax.set_visible(False)
            continue
        plot_bins(per_ch[ch], duration, labels if i % ncols == 0 else [""] * len(BANDS),
                  ax=ax, fig=None, colorbar=False, gridlines=True,
                  vmin=0, vmax=vmax)
        ax.set_title(ch, fontsize=10)
    for j in range(n, nrows * ncols):
        axs[j // ncols][j % ncols].set_visible(False)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=FIG_DPI)
    plt.close(fig)


def plot_per_channel_fft(df: pd.DataFrame, minute: int, minute_label: str,
                         fft_per_ch: Dict[str, np.ndarray] | None = None) -> None:
    t_start_s = minute * 60
    t_end_s = (minute + 1) * 60
    if fft_per_ch is None:
        fft_per_ch = fft_bin_spectrograms_per_channel(df, t_start_s, t_end_s, fft_bin_window=1)
    _per_channel_grid(fft_per_ch, duration=t_end_s - t_start_s,
                      title=f"FFT Spectrogram ({minute_label} minute {minute})",
                      fname="05_per_channel_fft_grid.png")


def plot_per_channel_sctn(subject: str, edf_stem: str,
                          minute_label: str, minute: int,
                          sctn_per_ch: Dict[str, np.ndarray] | None = None) -> None:
    if sctn_per_ch is None:
        sctn_per_ch = sctn_bin_spectrograms_per_channel(subject, edf_stem, minute_label, minute)
    _per_channel_grid(sctn_per_ch, duration=60,
                      title=f"SCTN Spikegram ({minute_label} minute {minute})",
                      fname="06_per_channel_sctn_grid.png")


def plot_combined_per_channel(df: pd.DataFrame, subject: str, edf_stem: str,
                              minute_label: str, minute: int,
                              fft_per_ch: Dict[str, np.ndarray] | None = None,
                              sctn_per_ch: Dict[str, np.ndarray] | None = None) -> None:
    """Paper Figure 4 grid: per-channel SCTN spikegram + FFT spectrogram side-by-side."""
    t_start_s = minute * 60
    t_end_s = (minute + 1) * 60
    duration = t_end_s - t_start_s
    if fft_per_ch is None:
        fft_per_ch = fft_bin_spectrograms_per_channel(df, t_start_s, t_end_s, fft_bin_window=1)
    if sctn_per_ch is None:
        sctn_per_ch = sctn_bin_spectrograms_per_channel(subject, edf_stem, minute_label, minute)

    n = len(CHANNELS)
    nrows = (n + 1) // 2          # two channels per row
    fig, axs = plt.subplots(nrows, 4, figsize=(22, 2.6 * nrows))
    labels = [f"{lf}-{hf} ({band})" for band, (lf, hf) in BANDS.items()]

    for i, ch in enumerate(CHANNELS):
        row = i // 2
        # column 0 (or 2): SCTN; column 1 (or 3): FFT
        c_sctn = 0 if i % 2 == 0 else 2
        c_fft = 1 if i % 2 == 0 else 3
        if ch in sctn_per_ch:
            plot_bins(sctn_per_ch[ch], duration,
                      labels if i % 2 == 0 else [""] * len(BANDS),
                      ax=axs[row][c_sctn], fig=None, colorbar=False)
            axs[row][c_sctn].set_title(f"SCTN — {ch}", fontsize=10)
        else:
            axs[row][c_sctn].set_visible(False)
        if ch in fft_per_ch:
            plot_bins(fft_per_ch[ch], duration,
                      [""] * len(BANDS),
                      ax=axs[row][c_fft], fig=None, colorbar=False)
            axs[row][c_fft].set_title(f"FFT — {ch}", fontsize=10)
        else:
            axs[row][c_fft].set_visible(False)

    fig.suptitle(f"FFT vs SCTN per channel ({minute_label} minute {minute})", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_combined_per_channel_grid.png", dpi=FIG_DPI)
    plt.close(fig)


def _channel_value_by_band(per_ch_specs: Dict[str, np.ndarray],
                           feature: str = "std") -> Dict[str, Dict[str, float]]:
    """Reduce each (channel, band) spectrogram row into one scalar via the feature
    function, normalize to [0, 1] across channels per-band.

    Returns: dict[band] -> dict[channel] -> value
    """
    func = {
        "min":    np.min,    "max":    np.max,
        "std":    np.std,    "mean":   np.mean,
        "median": np.median, "var":    np.var,
    }[feature]
    out: Dict[str, Dict[str, float]] = {b: {} for b in BANDS}
    for band_i, band in enumerate(BANDS):
        for ch, spec in per_ch_specs.items():
            out[band][ch] = float(func(spec[band_i]))
        # Normalize this band to [0, 1] across channels
        vals = np.array(list(out[band].values()))
        if vals.max() > vals.min():
            for ch in out[band]:
                out[band][ch] = (out[band][ch] - vals.min()) / (vals.max() - vals.min())
        else:
            for ch in out[band]:
                out[band][ch] = 0.0
    return out


def plot_topography_fft_vs_sctn(df: pd.DataFrame,
                                subject: str, edf_stem: str,
                                minute_label: str, minute: int,
                                feature: str = "std",
                                fft_per_ch: Dict[str, np.ndarray] | None = None,
                                sctn_per_ch: Dict[str, np.ndarray] | None = None) -> None:
    """Paper Figure 6 (FFT row + SCTN row, columns = bands), for one minute."""
    t_start_s = minute * 60
    t_end_s = (minute + 1) * 60
    if fft_per_ch is None:
        fft_per_ch = fft_bin_spectrograms_per_channel(df, t_start_s, t_end_s, fft_bin_window=1)
    if sctn_per_ch is None:
        sctn_per_ch = sctn_bin_spectrograms_per_channel(subject, edf_stem, minute_label, minute)

    fft_vals = _channel_value_by_band(fft_per_ch, feature=feature)
    sctn_vals = _channel_value_by_band(sctn_per_ch, feature=feature)

    nbands = len(BANDS)
    fig, axs = plt.subplots(2, nbands, figsize=(4.5 * nbands, 9))
    for i, band in enumerate(BANDS):
        topography_map(fft_vals[band], fig=fig, ax=axs[0, i],
                       vmin=0, vmax=1, title=band)
        topography_map(sctn_vals[band], fig=fig, ax=axs[1, i],
                       vmin=0, vmax=1, title=band)
    axs[0, 0].set_ylabel("FFT", fontsize=18)
    axs[1, 0].set_ylabel("SCTN", fontsize=18)
    fig.suptitle(f"Topography map ({minute_label} minute {minute}, feature={feature})",
                 fontsize=16)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_topography_fft_vs_sctn.png", dpi=FIG_DPI)
    plt.close(fig)


def plot_topography_seizure_vs_non_seizure(subject: str, edf_stem: str,
                                            seizure_minute: int,
                                            non_seizure_minute: int,
                                            feature: str = "std",
                                            sctn_seiz: Dict[str, np.ndarray] | None = None,
                                            sctn_non: Dict[str, np.ndarray] | None = None) -> None:
    """Paper Figure 7 style — rows = (seizure, non-seizure), columns = bands.
    Uses SCTN topography only (FFT-vs-SCTN comparison already in fig 08)."""
    if sctn_seiz is None:
        sctn_seiz = sctn_bin_spectrograms_per_channel(subject, edf_stem, "seizure", seizure_minute)
    if sctn_non is None:
        sctn_non = sctn_bin_spectrograms_per_channel(subject, edf_stem, "non_seizure", non_seizure_minute)

    seiz_vals = _channel_value_by_band(sctn_seiz, feature=feature)
    non_vals  = _channel_value_by_band(sctn_non,  feature=feature)

    nbands = len(BANDS)
    fig, axs = plt.subplots(2, nbands, figsize=(4.5 * nbands, 9))
    for i, band in enumerate(BANDS):
        topography_map(seiz_vals[band], fig=fig, ax=axs[0, i],
                       vmin=0, vmax=1, title=band)
        topography_map(non_vals[band], fig=fig, ax=axs[1, i],
                       vmin=0, vmax=1, title=band)
    axs[0, 0].set_ylabel(f"Seizure\n(minute {seizure_minute})", fontsize=14)
    axs[1, 0].set_ylabel(f"Non-seizure\n(minute {non_seizure_minute})", fontsize=14)
    fig.suptitle(f"SCTN topography map — seizure vs non-seizure (feature={feature})",
                 fontsize=16)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_topography_seizure_vs_non_seizure.png", dpi=FIG_DPI)
    plt.close(fig)


# ============================================================
#                  STATISTICS
# ============================================================
def compute_per_channel_band_stats(
    fft_seiz: Dict[str, np.ndarray],
    fft_non:  Dict[str, np.ndarray],
    sctn_seiz: Dict[str, np.ndarray],
    sctn_non:  Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Long-form table: one row per (channel, band, state) with FFT + SCTN summaries."""
    rows: list[dict] = []
    band_names = list(BANDS.keys())
    for ch in CHANNELS:
        for band_i, band in enumerate(band_names):
            for state, fft_d, sctn_d in (
                ("seizure",     fft_seiz, sctn_seiz),
                ("non_seizure", fft_non,  sctn_non),
            ):
                fft_row = fft_d.get(ch)
                sctn_row = sctn_d.get(ch)
                rows.append({
                    "channel": ch,
                    "band": band,
                    "state": state,
                    "fft_mean":  float(fft_row[band_i].mean()) if fft_row is not None else np.nan,
                    "fft_std":   float(fft_row[band_i].std())  if fft_row is not None else np.nan,
                    "fft_max":   float(fft_row[band_i].max())  if fft_row is not None else np.nan,
                    "sctn_mean": float(sctn_row[band_i].mean()) if sctn_row is not None else np.nan,
                    "sctn_std":  float(sctn_row[band_i].std())  if sctn_row is not None else np.nan,
                    "sctn_max":  float(sctn_row[band_i].max())  if sctn_row is not None else np.nan,
                    "sctn_spike_density": (
                        float(sctn_row[band_i].sum() / sctn_row[band_i].size)
                        if sctn_row is not None else np.nan
                    ),
                })
    return pd.DataFrame(rows)


def rank_channels_by_discriminability(stats_df: pd.DataFrame) -> pd.DataFrame:
    """For each channel, summarize how different the seizure-state SCTN activity
    is from the non-seizure-state SCTN activity, summed across bands.

    Returns a DataFrame sorted by `sctn_log_ratio_sum` (larger = more discriminative).
    """
    band_names = list(BANDS.keys())
    rows: list[dict] = []
    for ch in CHANNELS:
        seiz = stats_df[(stats_df.channel == ch) & (stats_df.state == "seizure")]
        non  = stats_df[(stats_df.channel == ch) & (stats_df.state == "non_seizure")]
        row = {"channel": ch}
        log_ratio_sum_sctn = 0.0
        log_ratio_sum_fft  = 0.0
        for band in band_names:
            s = seiz[seiz.band == band]
            n = non [non .band == band]
            if not len(s) or not len(n):
                continue
            sctn_s = float(s["sctn_std"].iloc[0]); sctn_n = float(n["sctn_std"].iloc[0])
            fft_s  = float(s["fft_std"].iloc[0]);  fft_n  = float(n["fft_std"].iloc[0])
            row[f"sctn_std_seiz_{band}"] = sctn_s
            row[f"sctn_std_non_{band}"]  = sctn_n
            # log( (seiz+eps) / (non+eps) ); positive = seizure > non-seizure
            eps = 1e-9
            row[f"sctn_log_ratio_{band}"] = float(np.log((sctn_s + eps) / (sctn_n + eps)))
            row[f"fft_log_ratio_{band}"]  = float(np.log((fft_s  + eps) / (fft_n  + eps)))
            log_ratio_sum_sctn += abs(row[f"sctn_log_ratio_{band}"])
            log_ratio_sum_fft  += abs(row[f"fft_log_ratio_{band}"])
        row["sctn_log_ratio_sum_abs"] = log_ratio_sum_sctn
        row["fft_log_ratio_sum_abs"]  = log_ratio_sum_fft
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values("sctn_log_ratio_sum_abs", ascending=False).reset_index(drop=True)


def save_run_config(extra: dict | None = None) -> None:
    """Dump the inputs that produced this run so the CSVs are reproducible."""
    cfg = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "subject": SUBJECT,
        "edf_file": EDF_FILE,
        "seizure_minute": SEIZURE_MINUTE,
        "non_seizure_minute": NON_SEIZURE_MINUTE,
        "channels": CHANNELS,
        "bands": {k: list(v) for k, v in BANDS.items()},
        "clk_resonators": {str(k): v for k, v in CLK_RESONATORS.items()},
        "fs": FS,
    }
    if extra:
        cfg.update(extra)
    with open(STATS_DIR / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


# ============================================================
#                      MAIN
# ============================================================
def main():
    log.info("=" * 60)
    log.info(f"Analyze CHB-MIT  subject={SUBJECT}  edf={EDF_FILE}")
    log.info(f"  seizure minute = {SEIZURE_MINUTE}, non-seizure minute = {NON_SEIZURE_MINUTE}")
    log.info(f"  channels ({len(CHANNELS)}) = {CHANNELS}")
    log.info(f"  bands     = {list(BANDS.keys())}")
    log.info(f"  resonators (total {len(FULL_RESONATOR_FREQS)}): "
             f"slow-clk={CLK_RESONATORS[15360]} | fast-clk={CLK_RESONATORS[153600]}")
    log.info("=" * 60)

    log.info(f"Reading EDF: {CHBMIT_ROOT / SUBJECT / EDF_FILE}")
    df = load_raw_edf(CHBMIT_ROOT / SUBJECT / EDF_FILE, CHANNELS)
    log.info(f"Loaded {df.shape[1]} channels, {df.shape[0]} samples ({df.shape[0]/FS:.0f} s)")

    primary_channel = "T7-P7" if "T7-P7" in df.columns else df.columns[0]
    log.info(f"Primary channel for single-channel plots: {primary_channel}")

    seizure_t_start     = SEIZURE_MINUTE * 60
    seizure_t_end       = (SEIZURE_MINUTE + 1) * 60
    non_seizure_t_start = NON_SEIZURE_MINUTE * 60
    non_seizure_t_end   = (NON_SEIZURE_MINUTE + 1) * 60

    # === Compute the 4 per-channel specs ONCE (reuse for plots + stats) ====
    log.info("Computing per-channel FFT spectrograms (seizure minute)...")
    fft_seiz = fft_bin_spectrograms_per_channel(df, seizure_t_start, seizure_t_end)
    log.info("Computing per-channel FFT spectrograms (non-seizure minute)...")
    fft_non  = fft_bin_spectrograms_per_channel(df, non_seizure_t_start, non_seizure_t_end)
    log.info("Computing per-channel SCTN spikegrams (seizure minute)...")
    sctn_seiz = sctn_bin_spectrograms_per_channel(SUBJECT, EDF_STEM, "seizure", SEIZURE_MINUTE)
    log.info("Computing per-channel SCTN spikegrams (non-seizure minute)...")
    sctn_non  = sctn_bin_spectrograms_per_channel(SUBJECT, EDF_STEM, "non_seizure", NON_SEIZURE_MINUTE)

    # === Figures ============================================================
    log.info("[1/9] raw signal time domain (+ plt.specgram)")
    plot_raw_signal_time(df, primary_channel, seizure_t_start, seizure_t_end,
                         title_suffix=" — seizure minute")

    log.info("[2/9] FFT spectrogram, single channel")
    plot_fft_one_channel(df, primary_channel, seizure_t_start, seizure_t_end)

    log.info("[3/9] SCTN single resonator (spike density + 1-s bins)")
    plot_sctn_one_resonator(SUBJECT, EDF_STEM, "seizure", SEIZURE_MINUTE,
                            primary_channel, f0=4.12)

    log.info("[4/9] FFT vs SCTN band spectrograms (paper Figure 4)")
    plot_fft_vs_sctn_bands(df, primary_channel, SUBJECT, EDF_STEM,
                           "seizure", SEIZURE_MINUTE)

    log.info("[4b/9] FFT vs SCTN band spectrograms — T8-P8, 1-s bins")
    plot_fft_vs_sctn_bands(df, "T8-P8", SUBJECT, EDF_STEM,
                           "seizure", SEIZURE_MINUTE,
                           sctn_bin_ms=1000,
                           out_name="04b_fft_vs_sctn_bands_1s.png")

    log.info("[4c/9] FFT vs SCTN — 3-channel 3x2 grid (seizure minute)")
    plot_selected_3x2_fft_vs_sctn(
        df, SUBJECT, EDF_STEM, "seizure", SEIZURE_MINUTE,
        channels=["FT9-FT10", "T8-P8", "F7-T7"],
        sctn_bin_ms=1000,
        out_name="04c_fft_vs_sctn_3channels_seizure.png",
    )

    log.info("[4d/9] FFT vs SCTN — 3-channel 3x2 grid (non-seizure minute)")
    plot_selected_3x2_fft_vs_sctn(
        df, SUBJECT, EDF_STEM, "non_seizure", NON_SEIZURE_MINUTE,
        channels=["FT9-FT10", "T8-P8", "F7-T7"],
        sctn_bin_ms=1000,
        out_name="04d_fft_vs_sctn_3channels_nonseizure.png",
    )

    log.info("[5/9] per-channel FFT grid")
    plot_per_channel_fft(df, SEIZURE_MINUTE, "seizure", fft_per_ch=fft_seiz)
    log.info("[6/9] per-channel SCTN grid")
    plot_per_channel_sctn(SUBJECT, EDF_STEM, "seizure", SEIZURE_MINUTE, sctn_per_ch=sctn_seiz)

    log.info("[7/9] combined per-channel FFT + SCTN grid")
    plot_combined_per_channel(df, SUBJECT, EDF_STEM, "seizure", SEIZURE_MINUTE,
                              fft_per_ch=fft_seiz, sctn_per_ch=sctn_seiz)

    log.info("[8/9] topography map FFT vs SCTN (seizure minute)")
    plot_topography_fft_vs_sctn(df, SUBJECT, EDF_STEM, "seizure", SEIZURE_MINUTE,
                                fft_per_ch=fft_seiz, sctn_per_ch=sctn_seiz)

    log.info("[9/9] topography map: seizure vs non-seizure")
    plot_topography_seizure_vs_non_seizure(SUBJECT, EDF_STEM,
                                           seizure_minute=SEIZURE_MINUTE,
                                           non_seizure_minute=NON_SEIZURE_MINUTE,
                                           sctn_seiz=sctn_seiz, sctn_non=sctn_non)

    # === Statistics =========================================================
    log.info("Computing statistics tables...")
    stats_df = compute_per_channel_band_stats(fft_seiz, fft_non, sctn_seiz, sctn_non)
    ranking_df = rank_channels_by_discriminability(stats_df)

    per_band_csv = STATS_DIR / "per_channel_band.csv"
    ranking_csv  = STATS_DIR / "channel_ranking.csv"
    stats_df.to_csv(per_band_csv, index=False)
    ranking_df.to_csv(ranking_csv, index=False)

    save_run_config()

    log.info("=" * 60)
    log.info("Done.")
    log.info(f"  Figures: {FIG_DIR}")
    for p in sorted(FIG_DIR.glob("*.png")):
        log.info(f"    {p.name}")
    log.info(f"  Stats:   {STATS_DIR}")
    log.info(f"    per_channel_band.csv  ({len(stats_df)} rows)")
    log.info(f"    channel_ranking.csv   ({len(ranking_df)} rows)")
    log.info(f"    run_config.json")
    log.info(f"    run.log")

    log.info("")
    log.info("Channel ranking (by SCTN |seizure vs non-seizure| log-ratio sum, "
             "summed across bands):")
    short = ranking_df[["channel", "sctn_log_ratio_sum_abs", "fft_log_ratio_sum_abs"]]
    for line in short.to_string(index=False).splitlines():
        log.info("    " + line)


if __name__ == "__main__":
    main()
