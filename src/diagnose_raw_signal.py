"""
Compare the RAW EEG of a seizure window vs a non-seizure window — BEFORE and
AFTER the per-minute [-1,1] normalization that signal2spikes applies — to see
whether the discriminative signal exists in the raw data and whether the
normalization erases it.

Fast (reads the EDF directly, no spike encoding). Run locally or on the remote:
    python -u src/diagnose_raw_signal.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import mne
from scipy.signal import butter, filtfilt, welch

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
EDF = _PROJECT_ROOT / "dataset" / "chbmit" / "chb01" / "chb01_03.edf"
FS = 256

CHANNELS = [
    "FT9-FT10", "T7-P7", "T8-P8", "P7-O1", "F7-T7",
    "P8-O2", "CZ-PZ", "P3-O1", "FP1-F7", "C4-P4",
]
BANDS = {"Delta": (0.5, 4), "Theta": (4, 8), "Alpha": (8, 14), "Beta": (14, 32)}

# The two windows the SNN diagnostic compared
SEIZURE_WIN = (3016.0, 3018.0)     # deep in the seizure (2996-3036)
NONSEIZ_WIN = (2220.0, 2222.0)     # interictal (minute 37)
SEIZURE_MINUTE = 50                # minute containing 3016
NONSEIZ_MINUTE = 37                # minute containing 2220


def bp(data, lo, hi):
    # data shape is (n_samples, n_channels); filter along time (axis 0)
    ny = FS / 2
    b1, a1 = butter(6, hi / ny, btype="lowpass")
    b2, a2 = butter(6, lo / ny, btype="highpass")
    return filtfilt(b2, a2, filtfilt(b1, a1, data, axis=0), axis=0)


def band_powers(sig):
    """Welch PSD integrated over each band."""
    f, pxx = welch(sig, fs=FS, nperseg=min(256, len(sig)))
    out = {}
    for band, (lo, hi) in BANDS.items():
        mask = (f >= lo) & (f < hi)
        out[band] = float(np.trapz(pxx[mask], f[mask])) if mask.any() else 0.0
    return out


def load_channels():
    raw = mne.io.read_raw_edf(str(EDF), preload=True, verbose="ERROR")
    avail = [c.strip() for c in raw.ch_names]
    picks, names = [], []
    for want in CHANNELS:
        wn = want.upper().replace(" ", "")
        for a in avail:
            an = a.upper().replace(" ", "")
            if an == wn or an.startswith(wn) or wn in an:
                picks.append(a); names.append(want); break
    data = raw.get_data(picks=picks).T          # (n_samples, n_ch)
    data = bp(data, 0.5, 62.0)
    return {name: data[:, i] for i, name in enumerate(names)}


def window(sig, t0, t1):
    return sig[int(t0 * FS):int(t1 * FS)]


def minute_norm(sig, minute):
    """Replicate signal2spikes per-minute peak normalization for one minute."""
    seg = sig[minute * 60 * FS:(minute + 1) * 60 * FS]
    peak = np.max(np.abs(seg))
    return seg / peak if peak > 0 else seg


def main():
    print(f"EDF: {EDF}")
    chans = load_channels()
    print(f"Loaded {len(chans)} channels\n")

    print("=" * 78)
    print("RAW signal (no normalization) — RMS amplitude per window")
    print("=" * 78)
    print(f"{'channel':10s} {'seiz RMS':>12s} {'non RMS':>12s} {'ratio s/n':>10s}")
    raw_ratios = []
    for ch, sig in chans.items():
        s = window(sig, *SEIZURE_WIN)
        n = window(sig, *NONSEIZ_WIN)
        rs, rn = float(np.sqrt((s ** 2).mean())), float(np.sqrt((n ** 2).mean()))
        ratio = rs / rn if rn > 0 else float("inf")
        raw_ratios.append(ratio)
        print(f"{ch:10s} {rs:12.4g} {rn:12.4g} {ratio:10.2f}")
    print(f"{'MEAN':10s} {'':>12s} {'':>12s} {np.mean(raw_ratios):10.2f}")

    print("\n" + "=" * 78)
    print("AFTER per-minute [-1,1] normalization — RMS amplitude per window")
    print("(this is what the encoder actually sees)")
    print("=" * 78)
    print(f"{'channel':10s} {'seiz RMS':>12s} {'non RMS':>12s} {'ratio s/n':>10s}")
    norm_ratios = []
    for ch, sig in chans.items():
        s_min = minute_norm(sig, SEIZURE_MINUTE)
        n_min = minute_norm(sig, NONSEIZ_MINUTE)
        # offsets within each minute
        s = s_min[int((SEIZURE_WIN[0] - SEIZURE_MINUTE * 60) * FS):
                  int((SEIZURE_WIN[1] - SEIZURE_MINUTE * 60) * FS)]
        n = n_min[int((NONSEIZ_WIN[0] - NONSEIZ_MINUTE * 60) * FS):
                  int((NONSEIZ_WIN[1] - NONSEIZ_MINUTE * 60) * FS)]
        rs, rn = float(np.sqrt((s ** 2).mean())), float(np.sqrt((n ** 2).mean()))
        ratio = rs / rn if rn > 0 else float("inf")
        norm_ratios.append(ratio)
        print(f"{ch:10s} {rs:12.4g} {rn:12.4g} {ratio:10.2f}")
    print(f"{'MEAN':10s} {'':>12s} {'':>12s} {np.mean(norm_ratios):10.2f}")

    print("\n" + "=" * 78)
    print("RAW band power ratio (seizure / non-seizure), per channel & band")
    print("=" * 78)
    hdr = f"{'channel':10s}" + "".join(f"{b:>10s}" for b in BANDS)
    print(hdr)
    for ch, sig in chans.items():
        s_bp = band_powers(window(sig, *SEIZURE_WIN))
        n_bp = band_powers(window(sig, *NONSEIZ_WIN))
        row = f"{ch:10s}"
        for b in BANDS:
            r = s_bp[b] / n_bp[b] if n_bp[b] > 0 else float("inf")
            row += f"{r:10.2f}"
        print(row)

    print("\nInterpretation:")
    print("  - RAW RMS ratio >> 1  → seizure window is higher amplitude (signal exists)")
    print("  - NORM RMS ratio ~ 1  → per-minute normalization erased that amplitude cue")
    print("  - band-power ratios far from 1 → frequency content differs (survives if encoded well)")


if __name__ == "__main__":
    main()
