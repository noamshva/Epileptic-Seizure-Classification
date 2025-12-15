#!/usr/bin/env python3
"""
Pre-Training Analysis for CHB-MIT Epileptic Seizure Data

Methods adapted from SCTN example notebooks:
- situational_awereness_detection.ipynb
- time_frequency_analasys.ipynb

Key features:
1. FFT Spectrogram visualization
2. Spike event spectrogram (spikegram)
3. Spike density via convolution (500-sample window)
4. Band-grouped spectrograms (Delta, Theta, Alpha, Beta, Gamma)
5. Topography map visualization
6. FFT vs Spikegram comparison plots
7. events_to_max_spectrogram for spike normalization

Key insight from SCTN: The "alternating pattern" (0,1,0,1...) is NORMAL for PDM encoding!
What matters is spike DENSITY over time windows, not raw spike values.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib
import scipy
from scipy.interpolate import griddata
matplotlib.use("Agg")
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from scipy import signal as scipy_signal
from scipy.signal import spectrogram, butter, filtfilt
import mne

# ============================================================
# Paths Configuration
# ============================================================

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent

DATA_ROOT = _project_root / "dataset" / "spikes_windowed"
RAW_SPIKES_ROOT = _project_root / "dataset" / "spikes_raw"
RAW_EDF_PATH = _project_root / "dataset" / "chbmit" / "chb01" / "chb01_03.edf"
OUTPUT_DIR = _project_root / "analysis" / "monopolar_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Configuration (from SCTN notebooks)
# ============================================================

# EEG sampling frequency
FS = 256

# Frequency bands (from SA notebook)
BANDS = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 14),
    'Beta': (14, 32),
    'Gamma': (32, 62),
}

# Resonator frequencies and clock frequencies
# MUST MATCH the configuration in 1.signal2spikes_chbmit.py!
# Note: 6.8 and 8.6 Hz were excluded due to parameter errors
# Note: Only Delta, Theta, and partial Alpha bands have resonators
CLK_RESONATORS = {
    15360: [1.1, 1.3, 1.6, 1.9, 2.2, 2.5, 2.88, 3.05, 3.39, 3.7, 4.12, 4.62, 5.09, 5.45, 5.87, 6.36, 7.6],  # 17 freqs, up to ~7.6 Hz (Delta + Theta)
    153600: [10.5, 11.5, 12.8, 15.8],  # 4 freqs, 10.5-15.8 Hz (Alpha only)
}
MAIN_CLK = max(CLK_RESONATORS.keys())
FULL_RESONATOR_FREQS = sum(CLK_RESONATORS.values(), [])

# Which bands actually have resonator coverage
BANDS_WITH_COVERAGE = {
    'Delta': True,   # 0.5-4 Hz -> has 1.1, 1.3, 1.6, 1.9, 2.2, 2.5, 2.88, 3.05, 3.39, 3.7
    'Theta': True,   # 4-8 Hz -> has 4.12, 4.62, 5.09, 5.45, 5.87, 6.36, 7.6
    'Alpha': True,   # 8-14 Hz -> has 10.5, 11.5, 12.8 (partial, missing 8-10 Hz)
    'Beta': False,   # 14-32 Hz -> only has 15.8 Hz (almost none!)
    'Gamma': False,  # 32-62 Hz -> NO resonators at all!
}

# Monopolar electrodes
MONOPOLAR_ELECTRODES = ['C4', 'CZ', 'F7', 'FP1', 'FT10', 'FT9', 'O1', 'O2',
                        'P3', 'P4', 'P7', 'P8', 'PZ', 'T7', 'T8']

# Spike density parameters (from time_frequency_analasys notebook)
SPIKE_DENSITY_WINDOW = 500  # Convolution window size
SPARSE_STEPS = 10  # Downsampling factor

# Visualization
FIG_DPI = 150

# Time ranges for CHB-MIT chb01_03
SEIZURE_SECOND_RANGE = (2996, 3036)
NON_SEIZURE_SECOND_RANGES = [(2940, 2995), (3037, 3120)]

# ============================================================
# Topography Map Configuration (from SA notebook)
# ============================================================

# 10-20 system electrode positions for topography (polar coordinates converted)
# Format: channel -> (angle_deg, radius)
TOPO_POSITIONS_POLAR = {
    'FP1': (-18, 0.4), 'FP2': (18, 0.4),
    'F7': (-54, 0.43), 'F3': (-39, 0.28), 'FZ': (0, 0.25), 'F4': (39, 0.28), 'F8': (54, 0.43),
    'FT9': (-72, 0.45), 'FT10': (72, 0.45),
    'T7': (-90, 0.43), 'C3': (-45, 0.0), 'CZ': (0, 0.0), 'C4': (45, 0.0), 'T8': (90, 0.43),
    'P7': (-126, 0.43), 'P3': (-141, 0.28), 'PZ': (180, 0.25), 'P4': (141, 0.28), 'P8': (126, 0.43),
    'O1': (-162, 0.43), 'O2': (162, 0.43),
}


def pol2cart(angle_deg, radius):
    """Convert polar to cartesian coordinates (from SA notebook)."""
    angle_rad = np.radians(angle_deg)
    x = radius * np.cos(angle_rad)
    y = radius * np.sin(angle_rad)
    return x, y


# ============================================================
# Band-pass Filter (from SA notebook)
# ============================================================

def bp_filter(data, f_lo, f_hi, fs):
    """Digital band pass filter (6th order Butterworth) - from SA notebook."""
    f_ny = fs / 2.0
    b_lo = f_lo / f_ny
    b_hi = f_hi / f_ny

    p_lp = {"N": 6, "Wn": b_hi, "btype": "lowpass", "analog": False, "output": "ba"}
    p_hp = {"N": 6, "Wn": b_lo, "btype": "highpass", "analog": False, "output": "ba"}

    bp_b1, bp_a1 = butter(**p_lp)
    bp_b2, bp_a2 = butter(**p_hp)

    data_filter = filtfilt(bp_b1, bp_a1, data, axis=0)
    data_filter = filtfilt(bp_b2, bp_a2, data_filter, axis=0)
    return data_filter


# ============================================================
# Spike Utility Functions (from SA notebook)
# ============================================================

def spikes_to_events(spikes):
    """Convert binary spikes array to event indices (from SA notebook)."""
    return np.where(spikes == 1)[0]


def spikes_event_spectrogram(clk_freq: int, events: np.ndarray, window_ms: int,
                              duration_s: float = None) -> np.ndarray:
    """
    Convert spike events to time-binned spectrogram (from SA notebook).

    Args:
        clk_freq: Clock frequency used by resonator
        events: Array of spike event indices
        window_ms: Time window in milliseconds
        duration_s: Total duration in seconds (optional)

    Returns:
        Spike counts per time bin
    """
    if len(events) == 0:
        return np.array([0])

    window = clk_freq / 1000 * window_ms

    if duration_s is None:
        duration_samples = events[-1] + 1
    else:
        duration_samples = int(duration_s * clk_freq)

    N = duration_samples / window
    if N != int(N):
        N = int(N) + 1
    else:
        N = int(N)

    bins = np.zeros(N, dtype=int)
    unique_indices, counts = np.unique((events // window).astype(int), return_counts=True)
    valid_mask = unique_indices < N
    bins[unique_indices[valid_mask]] = counts[valid_mask]

    return bins


def events_to_max_spectrogram(resonators_by_clk: Dict[int, List[np.ndarray]],
                               duration: float) -> np.ndarray:
    """
    Convert spike events to max spectrogram with normalization (from SA notebook).

    This is a KEY function that:
    1. Bins spikes into 10ms windows
    2. Takes max over 100ms bins (10x downsampling)
    3. Normalizes by clock frequency ratio
    4. Subtracts mean and clips negative values

    Args:
        resonators_by_clk: Dict mapping clk_freq -> list of spike event arrays
        duration: Duration in seconds

    Returns:
        2D array of shape (n_resonators, n_time_bins)
    """
    spikes_array = sum(resonators_by_clk.values(), start=[])
    n_resonators = len(spikes_array)
    n_bins = (duration * 1000) // 100  # 100ms bins

    max_spikes_spectrogram = np.zeros((n_resonators, int(n_bins)))

    i = 0
    for clk_freq, spikes_arrays in resonators_by_clk.items():
        for events in spikes_arrays:
            # Get 10ms binned spectrogram
            spikes_spec = spikes_event_spectrogram(clk_freq, events, 10, duration)

            # Reshape and take max over 10 bins (10ms -> 100ms)
            target_len = len(max_spikes_spectrogram[i])
            if len(spikes_spec) >= target_len * 10:
                reshaped = spikes_spec[:target_len * 10].reshape(target_len, 10)
                max_spikes_spectrogram[i] = reshaped.max(axis=1)
            else:
                # Pad if needed
                padded = np.zeros(target_len * 10)
                padded[:len(spikes_spec)] = spikes_spec
                max_spikes_spectrogram[i] = padded.reshape(target_len, 10).max(axis=1)

            # Normalize by clock frequency ratio
            max_spikes_spectrogram[i] *= MAIN_CLK / clk_freq

            # Subtract mean and clip negatives
            max_spikes_spectrogram[i] -= np.mean(max_spikes_spectrogram[i])
            max_spikes_spectrogram[i][max_spikes_spectrogram[i] < 0] = 0

            i += 1

    return max_spikes_spectrogram


def spikes_to_bands(spikes_spectrogram: np.ndarray, freqs: List[float],
                    normalize_band: bool = True) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Group spike spectrogram by frequency bands (from SA notebook).

    Args:
        spikes_spectrogram: 2D array of shape (n_resonators, n_time_bins)
        freqs: List of resonator frequencies
        normalize_band: Whether to normalize by sum of frequencies in band

    Returns:
        bands_spectrogram: 2D array of shape (n_bands, n_time_bins)
        freq_counts: Dict mapping band name to number of resonators in that band
    """
    freqs = np.array(freqs)
    n_bands = len(BANDS)
    n_time_bins = spikes_spectrogram.shape[1]

    bands_spectrogram = np.zeros((n_bands, n_time_bins))
    freq_counts = {}

    # Use BANDS definition directly (matches 1.signal2spikes_chbmit.py)
    for i, (band_name, (f_min, f_max)) in enumerate(BANDS.items()):
        mask = (freqs >= f_min) & (freqs < f_max)
        freq_counts[band_name] = int(np.sum(mask))

        if mask.any():
            bands_spectrogram[i] = np.sum(spikes_spectrogram[mask], axis=0)

            if normalize_band:
                freq_sum = np.sum(freqs[mask])
                if freq_sum > 0:
                    bands_spectrogram[i] /= freq_sum
        else:
            # No resonators for this band - set to NaN to indicate missing data
            bands_spectrogram[i] = np.nan

    return bands_spectrogram, freq_counts


# ============================================================
# FFT Spectrogram Functions (from SA notebook)
# ============================================================

def plot_spectrogram(data: np.ndarray, channel_name: str, fs: int,
                     fmin: float, fmax: float, show: bool = True, ax=None):
    """
    Plot FFT spectrogram (from SA notebook).

    Args:
        data: 1D signal array
        channel_name: Name for title
        fs: Sampling frequency
        fmin, fmax: Frequency range to display
        show: Whether to show the plot
        ax: Matplotlib axis (optional)

    Returns:
        Sxx: Spectrogram power values
    """
    f, t, Sxx = spectrogram(data, fs, noverlap=0, nperseg=fs)
    Sxx = Sxx[:, :-1] if Sxx.shape[1] > 1 else Sxx

    if ax is None and show:
        fig, ax = plt.subplots(figsize=(14, 5))

    if ax is not None:
        ax.set_ylim(fmin, fmax)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(f'ch={channel_name} fs={fs} band=({fmin} - {fmax})Hz')
        im = ax.imshow(Sxx, aspect='auto', cmap='jet', origin='lower',
                       extent=[t[0], t[-1], f[0], f[-1]])
        if show:
            plt.colorbar(im, ax=ax)

    return Sxx


def fft_spectrogram_to_bins(Sxx: np.ndarray, window: int,
                             normalize_band: bool = True) -> np.ndarray:
    """
    Convert FFT spectrogram to band-binned representation (from SA notebook).

    Args:
        Sxx: FFT spectrogram array
        window: Time window for binning
        normalize_band: Whether to normalize by band width

    Returns:
        2D array of shape (n_bands, n_time_bins)
    """
    bins_count = len(Sxx[0]) // window + 1
    bin_sxx = np.zeros((len(BANDS), bins_count))

    for ch_i, (band_name, (f_min, f_max)) in enumerate(BANDS.items()):
        f_min_idx = int(f_min)
        f_max_idx = min(int(f_max), len(Sxx))

        alpha = 1
        if normalize_band:
            alpha = (f_max - f_min)

        for i in range(bins_count - 1):
            bin_sxx[ch_i][i] = np.sum(Sxx[f_min_idx:f_max_idx, i * window:(i + 1) * window]) / alpha
        bin_sxx[ch_i][-1] = np.sum(Sxx[f_min_idx:f_max_idx, (bins_count - 1) * window:]) / alpha

    return bin_sxx


# ============================================================
# Spike Density Functions (from time_frequency_analasys notebook)
# ============================================================

def plot_emitted_spikes(spikes: np.ndarray, x_stop: float,
                        label: str = None, window_size: int = 500,
                        sparse_steps: int = 10, ax=None):
    """
    Plot spike density over time (from time_frequency_analasys notebook).

    This is the KEY visualization: convolve spikes with window to get density.

    Args:
        spikes: Binary spike array (or spike events converted to binary)
        x_stop: X-axis maximum value
        label: Plot label
        window_size: Convolution window size
        sparse_steps: Downsampling factor
        ax: Matplotlib axis
    """
    if len(spikes) < window_size:
        return None, None

    y_spikes = np.convolve(spikes.astype(float), np.ones(window_size), 'valid')
    y_spikes = y_spikes[::sparse_steps]
    x = np.linspace(0, x_stop, len(y_spikes))

    if ax is None:
        plt.plot(x, y_spikes, label=label)
    else:
        ax.plot(x, y_spikes, label=label)

    return x, y_spikes


def compute_spike_density(spikes: np.ndarray, window_size: int = SPIKE_DENSITY_WINDOW) -> np.ndarray:
    """
    Compute spike density using convolution (from time_frequency_analasys notebook).

    This is the KEY insight: Raw spikes are often 0,1,0,1,0,1... (PDM encoding).
    We analyze spike RATE in windows via convolution.
    """
    if len(spikes) < window_size:
        return spikes.astype(float)

    density = np.convolve(spikes.astype(float), np.ones(window_size), 'valid')
    return density


# ============================================================
# Plot Bins Heatmap (from SA notebook)
# ============================================================

def plot_bins(Sxx: np.ndarray, duration: float, labels: List[str],
              annotate: bool = True, rotate_annotate: bool = False,
              show: bool = True, colorbar: bool = True, gridlines: bool = False,
              ax=None, fig=None, vmin: float = None, vmax: float = None,
              fontsize: int = None, cmap: str = 'jet'):
    """
    Plot binned spectrogram as heatmap (from SA notebook).

    Args:
        Sxx: 2D array to plot
        duration: X-axis duration
        labels: Y-axis labels
        annotate: Whether to add value annotations
        rotate_annotate: Rotate annotations 90 degrees
        show: Whether to show the plot
        colorbar: Whether to add colorbar
        gridlines: Whether to add gridlines
        ax: Matplotlib axis
        fig: Matplotlib figure
        vmin, vmax: Color scale limits
        fontsize: Font size for labels
        cmap: Colormap name

    Returns:
        ax: The matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 5))

    im = ax.imshow(Sxx, aspect='auto', cmap=cmap, origin='lower',
                   extent=[0, duration, 0, len(Sxx)],
                   vmin=vmin, vmax=vmax)

    ax.set_yticks(np.arange(0.5, len(Sxx) + 0.5))
    ax.set_yticklabels(labels, fontsize=fontsize)

    if annotate and Sxx.size < 200:  # Only annotate small arrays
        bin_size = duration / len(Sxx[0]) if len(Sxx[0]) > 0 else 1
        rotation_degree = 90 if rotate_annotate else 0
        for i in range(len(Sxx)):
            for j in range(len(Sxx[i])):
                ax.text(j * bin_size + bin_size * 0.5, i + 0.5, f'{Sxx[i, j]:.2f}',
                        rotation=rotation_degree, ha="center", va="center",
                        color="white", fontsize=6)

    if gridlines:
        ax.set_yticks(np.arange(0, len(labels)), minor=True)
        ax.grid(which='minor', color='grey', linestyle='-', linewidth=0.5)

    if fig is not None:
        fig.tight_layout()

    if colorbar and fig is not None:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_ticks([])

    if show:
        plt.show()

    return ax


# ============================================================
# Topography Map (from SA notebook)
# ============================================================

def topography_map(data_by_channel: Dict[str, np.ndarray], fig=None, ax=None,
                   feature: str = 'std', vmin: float = None, vmax: float = None,
                   title: str = None):
    """
    Plot EEG topography map with interpolation (from SA notebook).

    Args:
        data_by_channel: Dict mapping channel names to data arrays
        fig: Matplotlib figure
        ax: Matplotlib axis
        feature: Feature to compute ('std', 'mean', 'max', 'min', 'var', 'median')
        vmin, vmax: Color scale limits
        title: Plot title

    Returns:
        fig: Matplotlib figure
    """
    if fig is None:
        fig = plt.figure(figsize=(6, 6))
    if ax is None:
        ax = fig.add_subplot(111, aspect=1)

    # Compute features for each channel
    feature_func = {
        'std': np.std,
        'mean': np.mean,
        'max': np.max,
        'min': np.min,
        'var': np.var,
        'median': np.median,
    }.get(feature.lower(), np.std)

    channel_values = {}
    for ch, data in data_by_channel.items():
        if len(data) > 0:
            channel_values[ch] = feature_func(data)

    if not channel_values:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center',
                transform=ax.transAxes)
        return fig

    # Get cartesian positions for available channels
    xx, yy, zz, ch_names = [], [], [], []
    for ch, value in channel_values.items():
        ch_upper = ch.upper()
        if ch_upper in TOPO_POSITIONS_POLAR:
            angle, radius = TOPO_POSITIONS_POLAR[ch_upper]
            x, y = pol2cart(angle, radius)
            xx.append(x)
            yy.append(y)
            zz.append(value)
            ch_names.append(ch)

    if len(xx) < 3:
        ax.text(0.5, 0.5, f"Not enough channels ({len(xx)})", ha='center',
                va='center', transform=ax.transAxes)
        return fig

    xx, yy, zz = np.array(xx), np.array(yy), np.array(zz)

    # Create interpolation grid
    N = 100
    xi = np.linspace(-0.5, 0.5, N)
    yi = np.linspace(-0.5, 0.5, N)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolate
    try:
        zi = griddata((xx, yy), zz, (Xi, Yi), method='cubic')
    except:
        zi = griddata((xx, yy), zz, (Xi, Yi), method='linear')

    # Mask outside the head circle
    radius = 0.45
    for i in range(N):
        for j in range(N):
            r = np.sqrt(xi[i]**2 + yi[j]**2)
            if r > radius:
                zi[j, i] = np.nan

    # Set color limits
    if vmin is None:
        vmin = np.nanmin(zi)
    if vmax is None:
        vmax = np.nanmax(zi)

    # Plot contour
    levels = np.linspace(vmin, vmax, 30)
    CS = ax.contourf(xi, yi, zi, levels=levels, cmap='jet', extend='both')
    ax.contour(xi, yi, zi, levels=10, colors='grey', linewidths=0.5)

    # Add colorbar
    cbar = fig.colorbar(CS, ax=ax, shrink=0.8)
    cbar.ax.tick_params(labelsize=8)

    # Plot electrode positions
    ax.scatter(xx, yy, marker='o', c='black', s=30, zorder=3)
    for i, ch in enumerate(ch_names):
        ax.annotate(ch, (xx[i], yy[i]), fontsize=7, ha='center', va='bottom')

    # Draw head outline
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(0.45 * np.cos(theta), 0.45 * np.sin(theta), 'k-', linewidth=2)

    # Draw nose
    ax.plot([0, 0.05, 0], [0.45, 0.52, 0.45], 'k-', linewidth=2)

    # Draw ears
    ax.plot([-0.45, -0.5, -0.5, -0.45], [0.05, 0.05, -0.05, -0.05], 'k-', linewidth=2)
    ax.plot([0.45, 0.5, 0.5, 0.45], [0.05, 0.05, -0.05, -0.05], 'k-', linewidth=2)

    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(-0.55, 0.6)
    ax.set_aspect('equal')
    ax.axis('off')

    if title:
        ax.set_title(title, fontsize=12)

    return fig


# ============================================================
# Data Loading Functions
# ============================================================

def load_raw_eeg_segment(start_second: int, duration_s: float = 10.0) -> Dict[str, np.ndarray]:
    """Load raw EEG data segment from EDF file."""
    if not RAW_EDF_PATH.exists():
        print(f"  Raw EDF file not found: {RAW_EDF_PATH}")
        return {}

    raw = mne.io.read_raw_edf(str(RAW_EDF_PATH), preload=True, verbose="ERROR")
    start_sample = int(start_second * FS)
    end_sample = int(start_sample + duration_s * FS)

    # Get channel names
    ch_names = [ch.strip() for ch in raw.ch_names]

    # Extract data
    data = raw.get_data(start=start_sample, stop=end_sample)

    return {ch: data[i, :] for i, ch in enumerate(ch_names)}


def load_spike_file(spike_path: Path) -> np.ndarray:
    """Load spike data from .npz file."""
    try:
        data = np.load(spike_path)
        return data['spikes']
    except Exception as e:
        print(f"  Error loading {spike_path}: {e}")
        return np.array([])


def load_windowed_sample(npz_path: Path) -> Dict[str, torch.Tensor]:
    """Load windowed spike data (pickled dict)."""
    with open(npz_path, "rb") as f:
        d = pickle.load(f)
    out = {}
    for k, t in d.items():
        if isinstance(t, torch.Tensor):
            out[k] = t.detach().cpu().contiguous()
        else:
            out[k] = torch.as_tensor(t).detach().cpu().contiguous()
    return out


def collect_spike_samples(label: str) -> List[Dict[str, torch.Tensor]]:
    """Collect all spike windows for a given label (seizure/non_seizure)."""
    samples = []
    subject_dir = DATA_ROOT / "chb01" / "chb01_03" / label

    if not subject_dir.exists():
        print(f"  Directory not found: {subject_dir}")
        return samples

    for fp in sorted(subject_dir.glob("*.npz")):
        try:
            data = load_windowed_sample(fp)
            samples.append(data)
        except Exception as e:
            print(f"  Failed to load {fp.name}: {e}")

    return samples


def collect_raw_spikes_by_second(second: int, channel: str) -> Dict[int, Dict[float, np.ndarray]]:
    """
    Collect raw spike data for a specific second and channel.

    Returns:
        Dict mapping clk_freq -> Dict mapping resonator_freq -> spike_events
    """
    result = {15360: {}, 153600: {}}

    base_path = RAW_SPIKES_ROOT / "chb01" / "chb01_03" / str(second) / channel

    if not base_path.exists():
        return result

    for clk_str in os.listdir(base_path):
        try:
            clk_freq = int(clk_str)
        except ValueError:
            continue

        clk_path = base_path / clk_str
        for res_file in clk_path.glob("*.npz"):
            try:
                res_freq = float(res_file.stem)
                spikes = load_spike_file(res_file)
                events = spikes_to_events(spikes)
                result[clk_freq][res_freq] = events
            except Exception as e:
                continue

    return result


# ============================================================
# Analysis Functions
# ============================================================

def analyze_single_channel_second(second: int, channel: str, duration: float = 1.0):
    """
    Analyze a single second of data for one channel.

    Returns:
        spike_spectrogram: Max spike spectrogram
        band_spectrogram: Band-grouped spectrogram
        freq_counts: Dict mapping band name to resonator count
    """
    raw_spikes = collect_raw_spikes_by_second(second, channel)

    # Convert to list format for events_to_max_spectrogram
    resonators_by_clk = {
        clk_freq: [events for freq, events in sorted(freq_dict.items())]
        for clk_freq, freq_dict in raw_spikes.items() if freq_dict
    }

    if not any(resonators_by_clk.values()):
        return None, None, {}

    spike_spec = events_to_max_spectrogram(resonators_by_clk, duration)

    # Get corresponding frequencies
    freqs = []
    for clk_freq in [15360, 153600]:
        if clk_freq in raw_spikes:
            freqs.extend(sorted(raw_spikes[clk_freq].keys()))

    if len(freqs) != spike_spec.shape[0]:
        freqs = FULL_RESONATOR_FREQS[:spike_spec.shape[0]]

    band_spec, freq_counts = spikes_to_bands(spike_spec, freqs)

    return spike_spec, band_spec, freq_counts


def diagnose_spike_data(samples: List[Dict], label: str):
    """
    Diagnostic function to inspect spike data quality.

    Prints information about the spike data to help debug issues.
    """
    print(f"\n  === DIAGNOSTIC: {label} ===")

    if not samples:
        print("    No samples found!")
        return

    # Inspect first sample
    sample = samples[0]
    print(f"    Number of samples: {len(samples)}")
    print(f"    Keys in first sample: {list(sample.keys())[:5]}...")

    # Check data for each band
    band_stats = {band: {'count': 0, 'mean_std': [], 'mean_spike_rate': [], 'shapes': []} for band in BANDS.keys()}

    for sample in samples[:5]:  # Check first 5 samples
        for key, spikes in sample.items():
            parts = key.split("-")
            band = parts[-1] if len(parts) > 1 else "Unknown"

            if band not in band_stats:
                continue

            spikes_np = spikes.numpy() if isinstance(spikes, torch.Tensor) else spikes

            band_stats[band]['count'] += 1
            band_stats[band]['shapes'].append(spikes_np.shape)

            for res_idx in range(spikes_np.shape[0]):
                res_spikes = spikes_np[res_idx, :]
                band_stats[band]['mean_std'].append(float(np.std(res_spikes)))
                band_stats[band]['mean_spike_rate'].append(float(np.mean(res_spikes)))

    print(f"\n    Band statistics (from first 5 samples):")
    for band, stats in band_stats.items():
        has_coverage = BANDS_WITH_COVERAGE.get(band, False)
        coverage_str = "HAS COVERAGE" if has_coverage else "NO COVERAGE"

        if stats['count'] > 0:
            avg_std = np.mean(stats['mean_std']) if stats['mean_std'] else 0
            avg_rate = np.mean(stats['mean_spike_rate']) if stats['mean_spike_rate'] else 0
            shape = stats['shapes'][0] if stats['shapes'] else "N/A"
            print(f"      {band}: count={stats['count']}, shape={shape}, avg_std={avg_std:.4f}, avg_spike_rate={avg_rate:.4f} [{coverage_str}]")
        else:
            print(f"      {band}: NO DATA [{coverage_str}]")


def diagnose_raw_npz_files(spikes_root: Path, label: str, max_files: int = 5):
    """
    Diagnose raw .npz spike files directly (before windowing).

    This helps identify issues with the spike generation itself.
    """
    print(f"\n  === RAW NPZ DIAGNOSTIC: {label} ===")

    # Find npz files
    search_path = spikes_root / "chb01" / "chb01_03" / label
    if not search_path.exists():
        print(f"    Path not found: {search_path}")
        return

    # Get a few second directories
    second_dirs = sorted([d for d in search_path.iterdir() if d.is_dir()])[:3]

    for sec_dir in second_dirs:
        print(f"\n    Second {sec_dir.name}:")

        # Get channels
        ch_dirs = [d for d in sec_dir.iterdir() if d.is_dir()][:2]

        for ch_dir in ch_dirs:
            print(f"      Channel {ch_dir.name}:")

            # Get clock frequencies
            clk_dirs = [d for d in ch_dir.iterdir() if d.is_dir()]

            for clk_dir in clk_dirs:
                clk_freq = int(clk_dir.name)
                npz_files = list(clk_dir.glob("*.npz"))[:3]

                for npz_file in npz_files:
                    try:
                        data = np.load(npz_file)
                        spikes = data['spikes']

                        # Key metrics
                        total_spikes = np.sum(spikes)
                        spike_rate = total_spikes / len(spikes) if len(spikes) > 0 else 0
                        expected_len = clk_freq * 1  # 1 second

                        # Check for alternating pattern
                        if len(spikes) > 10:
                            diffs = np.diff(spikes[:100])
                            alternating_ratio = np.sum(diffs != 0) / len(diffs) if len(diffs) > 0 else 0
                        else:
                            alternating_ratio = 0

                        f0 = float(npz_file.stem)
                        print(f"        clk={clk_freq}, f0={f0:.2f}: len={len(spikes)}, "
                              f"expected={expected_len}, spikes={total_spikes}, "
                              f"rate={spike_rate:.4f}, alt_ratio={alternating_ratio:.2f}")

                    except Exception as e:
                        print(f"        ERROR loading {npz_file.name}: {e}")


def compute_power_by_band(samples: List[Dict]) -> Dict[str, List[float]]:
    """
    Compute power (std) for each band from windowed samples.

    Only includes bands that have actual resonator coverage.
    """
    power_on_band = {band: [] for band in BANDS.keys()}

    for sample in samples:
        for key, spikes in sample.items():
            # Parse band from key (format: "electrode-band")
            parts = key.split("-")
            band = parts[-1]

            if band not in power_on_band:
                continue

            # Skip bands without resonator coverage
            if not BANDS_WITH_COVERAGE.get(band, False):
                continue

            spikes_np = spikes.numpy() if isinstance(spikes, torch.Tensor) else spikes

            # Skip if data is empty or all zeros
            if spikes_np.size == 0 or spikes_np.shape[0] == 0:
                continue

            # Compute power (std) for each resonator
            for res_idx in range(spikes_np.shape[0]):
                power = float(np.std(spikes_np[res_idx, :]))
                power_on_band[band].append(power)

    return power_on_band


def compute_power_by_channel(samples: List[Dict]) -> Dict[str, List[float]]:
    """Compute power (std) for each channel from windowed samples."""
    power_on_channel = {ch: [] for ch in MONOPOLAR_ELECTRODES}

    for sample in samples:
        for key, spikes in sample.items():
            # Parse channel from key (format: "electrode-band")
            parts = key.split("-")
            channel = parts[0]

            if channel not in power_on_channel:
                continue

            spikes_np = spikes.numpy() if isinstance(spikes, torch.Tensor) else spikes

            # Compute power (std) for each resonator
            for res_idx in range(spikes_np.shape[0]):
                power = float(np.std(spikes_np[res_idx, :]))
                power_on_channel[channel].append(power)

    return power_on_channel


# ============================================================
# Visualization Functions
# ============================================================

def plot_fft_vs_spikegram_comparison(eeg_data: np.ndarray, spike_sample: Dict,
                                      channel: str, duration: float,
                                      output_path: Path = None):
    """
    Create side-by-side FFT spectrogram vs Spikegram comparison (from SA notebook).
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # FFT Spectrogram
    ax_fft = axes[0]
    Sxx = plot_spectrogram(eeg_data, channel, FS, 0.5, 64, show=False, ax=ax_fft)
    fft_bins = fft_spectrogram_to_bins(Sxx, 1)

    # Create band labels
    labels = [f'{lo}-{hi} ({band})' for band, (lo, hi) in BANDS.items()]

    # Plot FFT bins
    ax_fft.clear()
    plot_bins(fft_bins, duration, labels, annotate=False, show=False,
              colorbar=True, gridlines=True, ax=ax_fft, fig=fig)
    ax_fft.set_title('(a) FFT Spectrogram', fontsize=14)
    ax_fft.set_ylabel('Frequency Band', fontsize=12)

    # Spikegram
    ax_spike = axes[1]

    # Collect spikes for this channel
    channel_spikes = {}
    for key, spikes in spike_sample.items():
        if key.startswith(channel + "-"):
            band = key.split("-")[-1]
            spikes_np = spikes.numpy() if isinstance(spikes, torch.Tensor) else spikes
            # Average across resonators
            channel_spikes[band] = spikes_np.mean(axis=0)

    if channel_spikes:
        # Create band-ordered array
        band_data = []
        for band in BANDS.keys():
            if band in channel_spikes:
                density = compute_spike_density(channel_spikes[band])
                # Bin to match FFT bins
                bin_size = max(1, len(density) // int(duration))
                binned = density[:int(duration) * bin_size].reshape(-1, bin_size).mean(axis=1)
                band_data.append(binned)
            else:
                band_data.append(np.zeros(int(duration)))

        # Ensure all same length
        min_len = min(len(b) for b in band_data)
        band_data = np.array([b[:min_len] for b in band_data])

        plot_bins(band_data, min_len, labels, annotate=False, show=False,
                  colorbar=True, gridlines=True, ax=ax_spike, fig=fig)

    ax_spike.set_title('(b) Spikegram', fontsize=14)
    ax_spike.set_xlabel('Time (s)', fontsize=12)
    ax_spike.set_ylabel('Frequency Band', fontsize=12)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=FIG_DPI)
        plt.close()
        print(f"  Saved: {output_path.name}")
    else:
        plt.show()


def plot_topography_by_band(power_by_channel: Dict[str, List[float]],
                            title_prefix: str = "", output_path: Path = None):
    """Plot topography maps for each band (only bands with resonator coverage)."""
    # Only plot bands with coverage
    bands_to_plot = [(band, bounds) for band, bounds in BANDS.items()
                     if BANDS_WITH_COVERAGE.get(band, False)]
    n_bands = len(bands_to_plot)

    if n_bands == 0:
        print("  WARNING: No bands with resonator coverage to plot!")
        return

    fig, axes = plt.subplots(1, n_bands, figsize=(5 * n_bands, 5))
    if n_bands == 1:
        axes = [axes]

    for i, (band, (f_lo, f_hi)) in enumerate(bands_to_plot):
        ax = axes[i]

        # Create channel data for this band
        channel_data = {ch: np.array(vals) for ch, vals in power_by_channel.items() if vals}

        topography_map(channel_data, fig=fig, ax=ax, feature='std',
                       title=f'{band}\n({f_lo}-{f_hi} Hz)')

    # Note about missing bands
    missing_bands = [b for b in BANDS.keys() if not BANDS_WITH_COVERAGE.get(b, False)]
    if missing_bands:
        plt.suptitle(f'{title_prefix} Topography by Band\n(Beta, Gamma excluded: no resonator coverage)',
                     fontsize=14)
    else:
        plt.suptitle(f'{title_prefix} Topography by Band', fontsize=14)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=FIG_DPI)
        plt.close()
        print(f"  Saved: {output_path.name}")
    else:
        plt.show()


def plot_spike_density_comparison(seizure_samples: List[Dict],
                                   non_seizure_samples: List[Dict],
                                   output_path: Path = None):
    """Plot spike density over time comparing seizure vs non-seizure."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    for idx, (samples, label, color) in enumerate([
        (seizure_samples, 'Seizure', '#d24b4b'),
        (non_seizure_samples, 'Non-Seizure', '#4b8fd2')
    ]):
        ax = axes[idx]

        if not samples:
            ax.text(0.5, 0.5, f"No {label} samples", ha='center', va='center',
                    transform=ax.transAxes)
            continue

        # Collect densities from first few samples
        for i, sample in enumerate(samples[:3]):
            all_densities = []
            for key, spikes in sample.items():
                spikes_np = spikes.numpy() if isinstance(spikes, torch.Tensor) else spikes
                avg_spikes = spikes_np.mean(axis=0)
                density = compute_spike_density(avg_spikes)
                all_densities.append(density)

            if all_densities:
                min_len = min(len(d) for d in all_densities)
                avg_density = np.mean([d[:min_len] for d in all_densities], axis=0)
                avg_density = avg_density[::SPARSE_STEPS]

                ax.plot(avg_density, alpha=0.7, linewidth=0.8,
                        label=f'Sample {i}', color=color)

        ax.set_xlabel('Time (downsampled)')
        ax.set_ylabel('Spike Density')
        ax.set_title(f'{label} - Spike Density over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=FIG_DPI)
        plt.close()
        print(f"  Saved: {output_path.name}")
    else:
        plt.show()


def plot_power_comparison(seizure_power: Dict[str, List[float]],
                          non_seizure_power: Dict[str, List[float]],
                          output_path: Path = None):
    """Plot power (std) comparison by band - only for bands with resonator coverage."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Only use bands with actual resonator coverage
    band_names = [b for b in BANDS.keys() if BANDS_WITH_COVERAGE.get(b, False)]
    x = np.arange(len(band_names))
    width = 0.35

    # Left: Bar comparison
    ax1 = axes[0]
    seiz_means = [np.mean(seizure_power.get(b, [0])) for b in band_names]
    seiz_stds = [np.std(seizure_power.get(b, [0])) for b in band_names]
    non_seiz_means = [np.mean(non_seizure_power.get(b, [0])) for b in band_names]
    non_seiz_stds = [np.std(non_seizure_power.get(b, [0])) for b in band_names]

    ax1.bar(x - width/2, seiz_means, width, yerr=seiz_stds,
            label='Seizure', color='#d24b4b', alpha=0.8, capsize=3)
    ax1.bar(x + width/2, non_seiz_means, width, yerr=non_seiz_stds,
            label='Non-Seizure', color='#4b8fd2', alpha=0.8, capsize=3)

    ax1.set_xlabel('Frequency Band')
    ax1.set_ylabel('Power (std)')
    ax1.set_title('Power by Frequency Band (Bands with Coverage Only)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(band_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Right: Ratio
    ax2 = axes[1]
    ratios = []
    for i in range(len(band_names)):
        if non_seiz_means[i] > 0:
            ratios.append(seiz_means[i] / non_seiz_means[i])
        else:
            ratios.append(1.0)

    colors = ['#2ca02c' if r > 1.1 else '#d62728' if r < 0.9 else '#1f77b4' for r in ratios]
    ax2.bar(x, ratios, color=colors, alpha=0.8)
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Frequency Band')
    ax2.set_ylabel('Seizure / Non-Seizure Ratio')
    ax2.set_title('Power Ratio by Band (Bands with Coverage Only)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(band_names)
    ax2.grid(True, alpha=0.3, axis='y')

    for i, r in enumerate(ratios):
        ax2.annotate(f'{r:.2f}x', (i, r), ha='center', va='bottom', fontsize=9)

    # Add note about missing bands
    missing_bands = [b for b in BANDS.keys() if not BANDS_WITH_COVERAGE.get(b, False)]
    if missing_bands:
        fig.text(0.5, 0.02, f"Note: {', '.join(missing_bands)} bands excluded (no resonator coverage)",
                 ha='center', fontsize=9, style='italic', color='red')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    if output_path:
        plt.savefig(output_path, dpi=FIG_DPI)
        plt.close()
        print(f"  Saved: {output_path.name}")
    else:
        plt.show()


# ============================================================
# Main Analysis
# ============================================================

def main():
    print("=" * 70)
    print("CHBMIT SPIKE ANALYSIS")
    print("Methods from SCTN example notebooks")
    print("=" * 70)
    print(f"\nData directory: {DATA_ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Show frequency coverage information
    print("RESONATOR FREQUENCY COVERAGE:")
    print("-" * 50)
    print(f"  15360 Hz clock: {len(CLK_RESONATORS[15360])} resonators")
    print(f"    Frequencies: {CLK_RESONATORS[15360]}")
    print(f"  153600 Hz clock: {len(CLK_RESONATORS[153600])} resonators")
    print(f"    Frequencies: {CLK_RESONATORS[153600]}")
    print()

    # Show band coverage
    print("BAND COVERAGE:")
    print("-" * 50)
    freqs = np.array(FULL_RESONATOR_FREQS)
    for band_name, (f_min, f_max) in BANDS.items():
        mask = (freqs >= f_min) & (freqs < f_max)
        count = int(np.sum(mask))
        has_coverage = BANDS_WITH_COVERAGE.get(band_name, False)
        status = "OK" if has_coverage else "WARNING: Limited/No coverage!"
        matching_freqs = freqs[mask].tolist() if mask.any() else []
        print(f"  {band_name} ({f_min}-{f_max} Hz): {count} resonators - {status}")
        if matching_freqs:
            print(f"    -> {matching_freqs}")
    print()

    print("Key methods:")
    print("  - Spike density via convolution (window=500)")
    print("  - events_to_max_spectrogram with clk normalization")
    print("  - spikes_to_bands for frequency grouping")
    print("  - Power = np.std metric")
    print("  - Topography map with interpolation")
    print()

    # 1. Load windowed spike data
    print("[1/7] Loading windowed spike samples...")
    seizure_samples = collect_spike_samples("seizure")
    non_seizure_samples = collect_spike_samples("non_seizure")

    print(f"  Loaded {len(seizure_samples)} seizure windows")
    print(f"  Loaded {len(non_seizure_samples)} non-seizure windows")

    if len(seizure_samples) == 0 and len(non_seizure_samples) == 0:
        print("\n  ERROR: No windowed data found!")
        print("  Please run 1.signal2spikes_chbmit.py first.")
        return

    # Run diagnostics on loaded data
    diagnose_spike_data(seizure_samples, "SEIZURE DATA")
    diagnose_spike_data(non_seizure_samples, "NON-SEIZURE DATA")

    # Also diagnose raw npz files
    print("\n[1b/7] Diagnosing raw NPZ files (before windowing)...")
    diagnose_raw_npz_files(RAW_SPIKES_ROOT, "seizure")
    diagnose_raw_npz_files(RAW_SPIKES_ROOT, "non_seizure")

    # 2. Compute power statistics
    print("\n[2/7] Computing power statistics (std method)...")
    seizure_power_band = compute_power_by_band(seizure_samples) if seizure_samples else {}
    non_seizure_power_band = compute_power_by_band(non_seizure_samples) if non_seizure_samples else {}

    seizure_power_channel = compute_power_by_channel(seizure_samples) if seizure_samples else {}
    non_seizure_power_channel = compute_power_by_channel(non_seizure_samples) if non_seizure_samples else {}

    # 3. Plot power comparison
    print("\n[3/7] Creating power comparison plots...")
    if seizure_power_band and non_seizure_power_band:
        plot_power_comparison(seizure_power_band, non_seizure_power_band,
                              OUTPUT_DIR / "power_comparison_by_band.png")

    # 4. Plot spike density comparison
    print("\n[4/7] Creating spike density comparison plots...")
    if seizure_samples or non_seizure_samples:
        plot_spike_density_comparison(seizure_samples, non_seizure_samples,
                                       OUTPUT_DIR / "spike_density_comparison.png")

    # 5. Plot topography maps
    print("\n[5/7] Creating topography maps...")
    if seizure_power_channel:
        plot_topography_by_band(seizure_power_channel, "Seizure",
                                OUTPUT_DIR / "topography_seizure.png")
    if non_seizure_power_channel:
        plot_topography_by_band(non_seizure_power_channel, "Non-Seizure",
                                OUTPUT_DIR / "topography_non_seizure.png")

    # 6. FFT vs Spikegram comparison (if raw EEG available)
    print("\n[6/7] Creating FFT vs Spikegram comparison...")
    if RAW_EDF_PATH.exists() and seizure_samples:
        # Load a segment of raw EEG
        seizure_mid = (SEIZURE_SECOND_RANGE[0] + SEIZURE_SECOND_RANGE[1]) // 2
        eeg_data = load_raw_eeg_segment(seizure_mid, duration_s=10.0)

        if eeg_data:
            # Use first available channel
            for ch_name, ch_data in eeg_data.items():
                channel_base = ch_name.split("-")[0]
                if channel_base in MONOPOLAR_ELECTRODES:
                    plot_fft_vs_spikegram_comparison(
                        ch_data, seizure_samples[0], channel_base, 10.0,
                        OUTPUT_DIR / f"fft_vs_spikegram_{channel_base}.png"
                    )
                    break
    else:
        print("  Skipping (raw EDF not found or no samples)")

    # 7. Summary report
    print("\n[7/7] Generating summary...")

    print("\n" + "=" * 70)
    print("POWER (STD) BY FREQUENCY BAND")
    print("=" * 70)
    print(f"{'Band':<10} {'Seizure':<12} {'Non-Seizure':<12} {'Ratio':<10} {'Coverage':<20}")
    print("-" * 70)

    freqs = np.array(FULL_RESONATOR_FREQS)
    for band, (f_min, f_max) in BANDS.items():
        seiz_power = np.mean(seizure_power_band.get(band, [0]))
        non_seiz_power = np.mean(non_seizure_power_band.get(band, [0]))
        ratio = seiz_power / non_seiz_power if non_seiz_power > 0 else 0

        # Count resonators in this band
        mask = (freqs >= f_min) & (freqs < f_max)
        n_resonators = int(np.sum(mask))
        has_coverage = BANDS_WITH_COVERAGE.get(band, False)

        if has_coverage:
            coverage_str = f"{n_resonators} resonators"
        else:
            coverage_str = f"NO COVERAGE ({n_resonators})"

        # Mark bands without coverage
        if not has_coverage:
            print(f"{band:<10} {'N/A':<12} {'N/A':<12} {'N/A':<10} {coverage_str:<20}")
        else:
            print(f"{band:<10} {seiz_power:<12.4f} {non_seiz_power:<12.4f} {ratio:<10.2f}x {coverage_str:<20}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - power_comparison_by_band.png")
    print("  - spike_density_comparison.png")
    print("  - topography_seizure.png / topography_non_seizure.png")
    print("  - fft_vs_spikegram_*.png")
    print()


if __name__ == "__main__":
    main()
