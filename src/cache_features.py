"""Extract burstiness features once per patient and cache them to disk.

Reading the per-resonator spike files takes minutes per patient; classifiers
that train for many epochs load the cache instead.

Writes <data root>/dataset/features_cache/<subject>.npz with:
    X_res   [n, 280]  per-resonator burstiness
    X_band  [n, 40]   mean per (channel, band)
    y       [n]       1 = seizure
    groups  [n]       recording (trial) of each window, for LORO
    start_s [n]       window start within the recording

With --bins it instead writes <data root>/dataset/bins_cache/<subject>.npz with:
    X_bins  [n, 200, 280] int16  10-ms spike counts per resonator
    y, groups, start_s           as above

With --fft it writes <data root>/dataset/fft_cache/<subject>.npz with the FFT
band-power baseline (X_band [n, 40]) on the same windows.

Usage:
    python -u src/cache_features.py                # all encoded patients
    python -u src/cache_features.py chb01 chb03
    python -u src/cache_features.py --bins
    python -u src/cache_features.py --fft
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F                                                    # noqa: E402


def cache_subject(subj: str, force: bool = False) -> tuple[str, int, float]:
    out = F.cache_path(subj)
    if out.is_file() and not force:
        return subj, -1, 0.0
    t0 = time.time()
    cache = F.MinuteCache(capacity=1)
    X_res, X_band, y, g, start = [], [], [], [], []
    for win in F.enumerate_windows([subj], F.WINDOW_SECONDS, F.WINDOW_OVERLAP):
        try:
            bw = F.window_burstiness(win, cache)
        except FileNotFoundError:
            continue
        res, band = [], []
        for ch in F.CHANNELS:
            for b in F.BANDS:
                v = bw.get((ch, b))
                n = len(F.BAND_RESONATORS[b])
                vals = [float(v[i]) if v is not None and i < len(v) else 0.0 for i in range(n)]
                res.extend(vals)
                band.append(float(np.mean(v)) if v is not None and len(v) else 0.0)
        X_res.append(res); X_band.append(band)
        y.append(1 if win["window_label"] == "seizure" else 0)
        g.append(win["trial"]); start.append(win["abs_start_s"])
    if not y:
        return subj, 0, time.time() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        X_res=np.asarray(X_res, np.float32), X_band=np.asarray(X_band, np.float32),
        y=np.asarray(y, np.int8), groups=np.asarray(g), start_s=np.asarray(start, np.float32),
    )
    return subj, len(y), time.time() - t0


def cache_subject_bins(subj: str, force: bool = False) -> tuple[str, int, float]:
    out = F.BINS_ROOT / f"{subj}.npz"
    if out.is_file() and not force:
        return subj, -1, 0.0
    t0 = time.time()
    cache = F.MinuteCache(capacity=1)
    n_res = sum(len(F.BAND_RESONATORS[b]) for b in F.BANDS) * len(F.CHANNELS)
    X, y, g, start = [], [], [], []
    for win in F.enumerate_windows([subj], F.WINDOW_SECONDS, F.WINDOW_OVERLAP):
        try:
            bw = F.window_bins(win, cache)
        except FileNotFoundError:
            continue
        x = np.zeros((F.N_BINS, n_res), np.int16)
        col = 0
        for ch in F.CHANNELS:
            for b in F.BANDS:
                v = bw.get((ch, b), [])
                for i in range(len(F.BAND_RESONATORS[b])):
                    if i < len(v):
                        x[:len(v[i]), col] = v[i][:F.N_BINS]
                    col += 1
        X.append(x)
        y.append(1 if win["window_label"] == "seizure" else 0)
        g.append(win["trial"]); start.append(win["abs_start_s"])
    if not y:
        return subj, 0, time.time() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X_bins=np.stack(X), y=np.asarray(y, np.int8),
                        groups=np.asarray(g), start_s=np.asarray(start, np.float32))
    return subj, len(y), time.time() - t0


def cache_subject_fft(subj: str, force: bool = False) -> tuple[str, int, float]:
    out = F.FFT_ROOT / f"{subj}.npz"
    if out.is_file() and not force:
        return subj, -1, 0.0
    t0 = time.time()
    bands = list(F._FFT_BANDS)[:4]
    n_win = int(round(F.WINDOW_SECONDS * F._FFT_FS))
    X, y, g, start = [], [], [], []
    trial, eeg = None, None
    for win in F.enumerate_windows([subj], F.WINDOW_SECONDS, F.WINDOW_OVERLAP):
        if win["trial"] != trial:                      # one recording in memory at a time
            trial = win["trial"]
            eeg = F._load_eeg_per_trial([subj], only={trial}).get(trial)
        if eeg is None:
            continue
        a = int(round(win["abs_start_s"] * F._FFT_FS)); b = a + n_win
        feats = []
        for ch in F.CHANNELS:
            sig = eeg.get(ch)
            for band in bands:
                lo, hi = F._FFT_BANDS[band]
                feats.append(0.0 if sig is None or b > len(sig)
                             else float(F._fft_window(sig[a:b], lo, hi, "band")[0]))
        X.append(feats)
        y.append(1 if win["window_label"] == "seizure" else 0)
        g.append(win["trial"]); start.append(win["abs_start_s"])
    if not y:
        return subj, 0, time.time() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X_band=np.asarray(X, np.float64), y=np.asarray(y, np.int8),
                        groups=np.asarray(g), start_s=np.asarray(start, np.float32))
    return subj, len(y), time.time() - t0


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    force = "--force" in sys.argv
    subjects = args or sorted(p.name for p in Path(F.SPIKES_ROOT).iterdir() if p.is_dir())
    bins = "--bins" in sys.argv
    fft = "--fft" in sys.argv
    fn = cache_subject_fft if fft else cache_subject_bins if bins else cache_subject
    print(f"Caching {len(subjects)} patients -> "
          f"{F.FFT_ROOT if fft else F.BINS_ROOT if bins else F.CACHE_ROOT}", flush=True)
    workers = min(int(os.environ.get("WORKERS", 3)), len(subjects))
    failed = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, s, force): s for s in subjects}
        for f in as_completed(futs):
            try:
                subj, n, dt = f.result()
            except Exception as e:
                failed.append(futs[f])
                print(f"  {futs[f]}: FAILED ({type(e).__name__}: {e})", flush=True)
                continue
            msg = "cached already" if n < 0 else f"{n} windows in {dt:.0f}s"
            print(f"  {subj}: {msg}", flush=True)
    if failed:
        sys.exit(f"Failed: {failed}")


if __name__ == "__main__":
    main()
