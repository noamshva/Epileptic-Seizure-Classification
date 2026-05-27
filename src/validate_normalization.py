"""
Validate the normalization fix BEFORE committing to a full re-encode.

Runs a handful of SCTN resonators on the exact seizure vs non-seizure windows
under two normalizations:
  - OLD: per-minute peak  (current signal2spikes behavior)  -> expect ~equal rates
  - NEW: per-channel whole-recording scale                   -> expect seizure > non

If the NEW normalization makes the resonator output spike count clearly larger
for the seizure window, the rate-based readout will have signal and a full
re-encode is justified.

Fast: only a few (channel, resonator) pairs on 2-second windows.
    python -u src/validate_normalization.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import mne
from scipy.signal import butter, filtfilt, resample_poly, resample
from sctn.resonator import resonator as spikes_resonator

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
EDF = _PROJECT_ROOT / "dataset" / "chbmit" / "chb01" / "chb01_03.edf"
FS = 256

# (channel, f0, clk) test set — spread across bands and discriminating channels
TESTS = [
    ("FP1-F7", 2.88, 15360),    # Delta, strong discriminator in raw band power
    ("FP1-F7", 22.0, 153600),   # Beta, 65x raw power ratio
    ("T7-P7",  4.62, 15360),    # Theta, mid channel
    ("T8-P8",  22.0, 153600),   # Beta, 136x raw power ratio
    ("FT9-FT10", 2.88, 15360),  # Delta
]

SEIZURE_WIN = (3016.0, 3018.0)
NONSEIZ_WIN = (2220.0, 2222.0)
SEIZURE_MINUTE = 50
NONSEIZ_MINUTE = 37

GLOBAL_PCTL = 99.9   # percentile of |signal| used for the per-channel global scale


def bp(data, lo, hi):
    ny = FS / 2
    b1, a1 = butter(6, hi / ny, btype="lowpass")
    b2, a2 = butter(6, lo / ny, btype="highpass")
    return filtfilt(b2, a2, filtfilt(b1, a1, data, axis=0), axis=0)


def resample_signal(f_new, f_src, data):
    ratio = f_new / f_src
    if abs(round(ratio) - ratio) < 1e-12:
        return resample_poly(data, up=int(round(ratio)), down=1)
    return resample(data, int(len(data) * ratio))


def run_resonator(sig_256, f0, clk):
    """Resample a 256 Hz window to clk and count output spikes from the resonator."""
    rs = resample_signal(clk, FS, sig_256)
    res = spikes_resonator(freq0=f0, clk_freq=clk)
    res.log_out_spikes(-1)
    out = res.layers_neurons[-1].neurons[-1]
    # warmup with zeros to settle the resonator
    res.input_full_data(np.zeros(int(clk * 0.5)))
    out.forget_logs()
    res.input_full_data(rs)
    sp = out.out_spikes(False, len(rs)).astype(np.int8)
    # Total count (DC, ~50% — uninformative) and the std of the 10 ms-binned
    # local density (the candidate discriminative feature).
    total = int(sp.sum())
    bin_size = max(1, int(clk * 0.010))          # 10 ms bins
    n_bins = len(sp) // bin_size
    binned = sp[: n_bins * bin_size].reshape(n_bins, bin_size).sum(axis=1)
    density_std = float(binned.std())
    return total, density_std


def main():
    raw = mne.io.read_raw_edf(str(EDF), preload=True, verbose="ERROR")
    avail = [c.strip() for c in raw.ch_names]

    def pick(name):
        wn = name.upper().replace(" ", "")
        for a in avail:
            an = a.upper().replace(" ", "")
            if an == wn or an.startswith(wn) or wn in an:
                return a
        raise KeyError(name)

    needed = sorted({t[0] for t in TESTS})
    picks = [pick(n) for n in needed]
    data = raw.get_data(picks=picks).T
    data = bp(data, 0.5, 62.0)
    chan = {n: data[:, i] for i, n in enumerate(needed)}

    # Per-channel global scale (whole recording)
    global_scale = {n: np.percentile(np.abs(sig), GLOBAL_PCTL) for n, sig in chan.items()}

    def win(sig, t0, t1):
        return sig[int(t0 * FS):int(t1 * FS)].copy()

    def per_minute_norm(sig, minute, t0, t1):
        seg = sig[minute * 60 * FS:(minute + 1) * 60 * FS]
        peak = np.max(np.abs(seg))
        seg = seg / peak if peak > 0 else seg
        return seg[int((t0 - minute * 60) * FS):int((t1 - minute * 60) * FS)].copy()

    def global_norm(sig, name, t0, t1):
        w = win(sig, t0, t1)
        s = global_scale[name]
        return np.clip(w / s, -1.0, 1.0) if s > 0 else w

    print(f"Per-channel global scale (|signal| {GLOBAL_PCTL} pctl): "
          + ", ".join(f"{n}={global_scale[n]:.2g}" for n in needed))
    print("\nUsing the CURRENT (per-minute) normalization — same as the encoded spikes_raw.")
    print("Two readouts per window:")
    print("  count = total output spikes (the SNN's current readout, expected ~equal)")
    print("  dstd  = std of 10 ms-binned output density (the figures' feature)\n")
    print(f"{'channel':10s}{'f0':>7s}{'clk':>8s}  | "
          f"{'cnt seiz':>9s}{'cnt non':>9s}{'cnt r':>6s} | "
          f"{'dstd seiz':>10s}{'dstd non':>10s}{'dstd r':>7s}")
    print("-" * 92)

    for ch, f0, clk in TESTS:
        sig = chan[ch]
        cnt_s, std_s = run_resonator(per_minute_norm(sig, SEIZURE_MINUTE, *SEIZURE_WIN), f0, clk)
        cnt_n, std_n = run_resonator(per_minute_norm(sig, NONSEIZ_MINUTE, *NONSEIZ_WIN), f0, clk)
        cnt_r = cnt_s / cnt_n if cnt_n else float("inf")
        std_r = std_s / std_n if std_n else float("inf")
        print(f"{ch:10s}{f0:7.2f}{clk:8d}  | "
              f"{cnt_s:9d}{cnt_n:9d}{cnt_r:6.2f} | "
              f"{std_s:10.2f}{std_n:10.2f}{std_r:7.2f}")

    print("\nInterpretation:")
    print("  cnt r ~ 1    → total count is blind (DC ~50%), regardless of class")
    print("  dstd r >> 1  → local-density variance discriminates at the 2-s window scale")
    print("                 → extract THIS feature (no re-encode); feed to the classifier")


if __name__ == "__main__":
    main()
