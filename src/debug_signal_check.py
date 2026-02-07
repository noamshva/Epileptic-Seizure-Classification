#!/usr/bin/env python3
"""
Debug script to compare raw signal characteristics between seizure and non-seizure.
This helps diagnose why spike rates are identical.

Also analyzes the SPIKE spectrogram patterns (what SA code actually uses).
"""

import numpy as np
import mne
from pathlib import Path
from scipy.signal import butter, filtfilt, welch
import matplotlib.pyplot as plt
import os

# Paths
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
EDF_PATH = _project_root / "dataset" / "chbmit" / "chb01" / "chb01_03.edf"
SPIKES_ROOT = _project_root / "dataset" / "spikes_raw"

fs = 256  # Hz

# Seizure: 2996-3036 seconds
SEIZURE_RANGE = (2996, 3036)
# Non-seizure: 20 seconds before seizure
NON_SEIZURE_RANGE = (2976, 2996)

# Bipolar channel to analyze
TARGET_CHANNEL = "FP1-F7"


def bp_filter(data, f_lo, f_hi, fs_):
    f_ny = fs_ / 2.0
    b_lo = f_lo / f_ny
    b_hi = f_hi / f_ny
    b1, a1 = butter(6, b_hi, btype="lowpass")
    b2, a2 = butter(6, b_lo, btype="highpass")
    out = filtfilt(b1, a1, data, axis=0)
    out = filtfilt(b2, a2, out, axis=0)
    return out


def analyze_segment(data, fs, label):
    """Analyze signal characteristics of a segment."""
    print(f"\n=== {label} ===")
    print(f"  Shape: {data.shape}")
    print(f"  Duration: {len(data)/fs:.1f} seconds")
    print(f"  Min: {np.min(data):.6f}")
    print(f"  Max: {np.max(data):.6f}")
    print(f"  Mean: {np.mean(data):.6f}")
    print(f"  Std: {np.std(data):.6f}")
    print(f"  RMS: {np.sqrt(np.mean(data**2)):.6f}")

    # Compute power spectral density
    freqs, psd = welch(data, fs=fs, nperseg=min(256, len(data)))

    # Band powers
    bands = {
        'Delta (0.5-4)': (0.5, 4),
        'Theta (4-8)': (4, 8),
        'Alpha (8-14)': (8, 14),
        'Beta (14-32)': (14, 32),
        'Gamma (32-62)': (32, 62),
    }

    print(f"\n  Band Power (PSD):")
    for band_name, (f_lo, f_hi) in bands.items():
        mask = (freqs >= f_lo) & (freqs < f_hi)
        power = np.mean(psd[mask]) if mask.any() else 0
        print(f"    {band_name}: {power:.6f}")

    return freqs, psd


def main():
    print("=" * 70)
    print("DEBUG: Signal Characteristics Comparison")
    print("=" * 70)

    # Load EDF
    raw = mne.io.read_raw_edf(str(EDF_PATH), preload=True, verbose="ERROR")
    ch_names = [ch.strip() for ch in raw.ch_names]

    # Find target channel
    target_idx = None
    for i, ch in enumerate(ch_names):
        if TARGET_CHANNEL.upper() in ch.upper():
            target_idx = i
            print(f"Found channel: {ch} at index {i}")
            break

    if target_idx is None:
        print(f"Channel {TARGET_CHANNEL} not found!")
        print(f"Available: {ch_names[:10]}...")
        return

    # Get full data for this channel
    full_data = raw.get_data()[target_idx, :]

    # Extract seizure and non-seizure segments
    seiz_start = SEIZURE_RANGE[0] * fs
    seiz_end = SEIZURE_RANGE[1] * fs
    non_seiz_start = NON_SEIZURE_RANGE[0] * fs
    non_seiz_end = NON_SEIZURE_RANGE[1] * fs

    seizure_raw = full_data[seiz_start:seiz_end]
    non_seizure_raw = full_data[non_seiz_start:non_seiz_end]

    print("\n" + "=" * 70)
    print("RAW DATA (before filtering)")
    print("=" * 70)
    analyze_segment(seizure_raw, fs, "SEIZURE (raw)")
    analyze_segment(non_seizure_raw, fs, "NON-SEIZURE (raw)")

    # Apply bandpass filter
    seizure_filt = bp_filter(seizure_raw, 0.5, 62.0, fs)
    non_seizure_filt = bp_filter(non_seizure_raw, 0.5, 62.0, fs)

    print("\n" + "=" * 70)
    print("FILTERED DATA (0.5-62 Hz)")
    print("=" * 70)
    freqs_seiz, psd_seiz = analyze_segment(seizure_filt, fs, "SEIZURE (filtered)")
    freqs_non, psd_non = analyze_segment(non_seizure_filt, fs, "NON-SEIZURE (filtered)")

    # Normalize globally (what was happening before)
    global_peak = max(np.max(np.abs(seizure_filt)), np.max(np.abs(non_seizure_filt)))
    seizure_global_norm = seizure_filt / global_peak
    non_seizure_global_norm = non_seizure_filt / global_peak

    print("\n" + "=" * 70)
    print("GLOBAL NORMALIZATION (the problematic approach)")
    print("=" * 70)
    analyze_segment(seizure_global_norm, fs, "SEIZURE (global norm)")
    analyze_segment(non_seizure_global_norm, fs, "NON-SEIZURE (global norm)")

    # Normalize per-segment
    seizure_local_norm = seizure_filt / np.max(np.abs(seizure_filt))
    non_seizure_local_norm = non_seizure_filt / np.max(np.abs(non_seizure_filt))

    print("\n" + "=" * 70)
    print("PER-SEGMENT NORMALIZATION")
    print("=" * 70)
    analyze_segment(seizure_local_norm, fs, "SEIZURE (local norm)")
    analyze_segment(non_seizure_local_norm, fs, "NON-SEIZURE (local norm)")

    # Plot comparison
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))

    # Row 1: Raw signals
    t = np.arange(len(seizure_raw)) / fs
    axes[0, 0].plot(t[:fs*5], seizure_raw[:fs*5], 'r-', alpha=0.7)
    axes[0, 0].set_title('Seizure - Raw (first 5 sec)')
    axes[0, 0].set_ylabel('Amplitude')

    t = np.arange(len(non_seizure_raw)) / fs
    axes[0, 1].plot(t[:fs*5], non_seizure_raw[:fs*5], 'b-', alpha=0.7)
    axes[0, 1].set_title('Non-Seizure - Raw (first 5 sec)')

    # Row 2: Filtered signals
    axes[1, 0].plot(t[:fs*5], seizure_filt[:fs*5], 'r-', alpha=0.7)
    axes[1, 0].set_title('Seizure - Filtered (first 5 sec)')
    axes[1, 0].set_ylabel('Amplitude')

    axes[1, 1].plot(t[:fs*5], non_seizure_filt[:fs*5], 'b-', alpha=0.7)
    axes[1, 1].set_title('Non-Seizure - Filtered (first 5 sec)')

    # Row 3: Normalized signals
    axes[2, 0].plot(t[:fs*5], seizure_local_norm[:fs*5], 'r-', alpha=0.7)
    axes[2, 0].set_title('Seizure - Per-segment Normalized')
    axes[2, 0].set_ylabel('Normalized')

    axes[2, 1].plot(t[:fs*5], non_seizure_local_norm[:fs*5], 'b-', alpha=0.7)
    axes[2, 1].set_title('Non-Seizure - Per-segment Normalized')

    # Row 4: PSD comparison
    axes[3, 0].semilogy(freqs_seiz, psd_seiz, 'r-', label='Seizure')
    axes[3, 0].semilogy(freqs_non, psd_non, 'b-', label='Non-Seizure')
    axes[3, 0].set_xlabel('Frequency (Hz)')
    axes[3, 0].set_ylabel('PSD')
    axes[3, 0].set_title('Power Spectral Density Comparison')
    axes[3, 0].legend()
    axes[3, 0].set_xlim(0, 65)

    # PSD ratio
    ratio = psd_seiz / (psd_non + 1e-10)
    axes[3, 1].plot(freqs_seiz, ratio, 'g-')
    axes[3, 1].axhline(y=1.0, color='k', linestyle='--')
    axes[3, 1].set_xlabel('Frequency (Hz)')
    axes[3, 1].set_ylabel('Ratio (Seizure/Non-Seizure)')
    axes[3, 1].set_title('PSD Ratio')
    axes[3, 1].set_xlim(0, 65)
    axes[3, 1].set_ylim(0, 5)

    plt.tight_layout()
    output_path = _project_root / "analysis" / "bipolar_validation" / "debug_signal_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")
    plt.close()

    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("""
If seizure and non-seizure have DIFFERENT power spectral densities (PSDs),
but produce IDENTICAL spike rates, then:

1. The resonators might not be responding to frequency differences
2. The PDM encoding might be saturating (always ~50% spike rate)
3. The signal amplitude/frequency content might not be reaching the resonators properly

Look at the PSD comparison plot to see if there ARE frequency differences.
""")


if __name__ == "__main__":
    main()
