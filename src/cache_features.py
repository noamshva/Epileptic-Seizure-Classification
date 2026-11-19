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

With --gamma it writes <data root>/dataset/gamma_cache/<subject>.npz with the
Gamma-band burstiness (X_res [n, 70], X_band [n, 10]) on the same windows.

--peri (with SPIKES_ROOT = the encoder's PERI=1 folder) caches the non-seizure
windows near seizures instead: guard-band minutes plus the seizure-free part of
the seizure minutes in dataset/spikes_raw.

Usage:
    python -u src/cache_features.py                # all encoded patients
    python -u src/cache_features.py chb01 chb03
    python -u src/cache_features.py --bins
    python -u src/cache_features.py --fft
    python -u src/cache_features.py --gamma
    SPIKES_ROOT=dataset/spikes_raw_peri python -u src/cache_features.py --peri [--fft]
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


def peri_windows(subj: str):
    """Non-seizure windows near seizures, same fields as enumerate_windows plus 'root'."""
    S = F._s2s
    seizures = S._load_seizures_for_subjects([subj])
    main = Path(F.SPIKES_ROOT).parent / "spikes_raw"
    step = F.WINDOW_SECONDS * (1 - F.WINDOW_OVERLAP)
    for trial in sorted(p.name for p in (Path(F.SPIKES_ROOT) / subj).iterdir() if p.is_dir()):
        iv = seizures.get(trial, [])
        for root, lab in ((Path(F.SPIKES_ROOT), "peri"), (main, "seizure")):
            d = root / subj / trial / lab
            if not d.is_dir():
                continue
            for minute in sorted(int(m.name) for m in d.iterdir()):
                for off in np.arange(0, 60 - F.WINDOW_SECONDS + 1e-9, step):
                    a = minute * 60 + float(off)
                    if S._overlaps_any_seizure(a, a + F.WINDOW_SECONDS, iv):
                        continue
                    yield {"subject": subj, "trial": trial, "minute_label": lab, "minute": minute,
                           "sec_offset_in_minute": float(off), "abs_start_s": a,
                           "abs_end_s": a + F.WINDOW_SECONDS, "window_label": "non_seizure",
                           "root": root}


def _windows(subj: str, peri: bool):
    return peri_windows(subj) if peri else F.enumerate_windows([subj], F.WINDOW_SECONDS, F.WINDOW_OVERLAP)


def _minute_cache(caches: dict, win: dict, band_res=None):
    root = win.get("root")
    if root not in caches:
        caches[root] = F.MinuteCache(capacity=1, band_resonators=band_res, root=root)
    return caches[root]


def cache_subject(subj: str, force: bool = False, peri: bool = False,
                  gamma: bool = False) -> tuple[str, int, float]:
    out = F.GAMMA_ROOT / f"{subj}.npz" if gamma else F.cache_path(subj)
    if out.is_file() and not force:
        return subj, -1, 0.0
    t0 = time.time()
    bands = ("Gamma",) if gamma else F.BANDS
    band_res = {"Gamma": F.GAMMA_RESONATORS} if gamma else F.BAND_RESONATORS
    caches: dict = {}
    X_res, X_band, y, g, start = [], [], [], [], []
    for win in _windows(subj, peri):
        try:
            bw = F.window_burstiness(win, _minute_cache(caches, win, band_res))
        except FileNotFoundError:
            continue
        res, band = [], []
        for ch in F.CHANNELS:
            for b in bands:
                v = bw.get((ch, b))
                n = len(band_res[b])
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


def cache_subject_bins(subj: str, force: bool = False, peri: bool = False) -> tuple[str, int, float]:
    out = F.BINS_ROOT / f"{subj}.npz"
    if out.is_file() and not force:
        return subj, -1, 0.0
    t0 = time.time()
    caches: dict = {}
    n_res = sum(len(F.BAND_RESONATORS[b]) for b in F.BANDS) * len(F.CHANNELS)
    X, y, g, start = [], [], [], []
    for win in _windows(subj, peri):
        try:
            bw = F.window_bins(win, _minute_cache(caches, win))
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


def cache_subject_fft(subj: str, force: bool = False, peri: bool = False) -> tuple[str, int, float]:
    out = F.FFT_ROOT / f"{subj}.npz"
    if out.is_file() and not force:
        return subj, -1, 0.0
    t0 = time.time()
    bands = list(F._FFT_BANDS)[:4]
    n_win = int(round(F.WINDOW_SECONDS * F._FFT_FS))
    X, y, g, start = [], [], [], []
    trial, eeg = None, None
    for win in _windows(subj, peri):
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
    gamma = "--gamma" in sys.argv
    peri = "--peri" in sys.argv
    fn = cache_subject_fft if fft else cache_subject_bins if bins else cache_subject
    extra = (peri, True) if gamma else (peri,)
    print(f"Caching {len(subjects)} patients -> "
          f"{F.FFT_ROOT if fft else F.BINS_ROOT if bins else F.GAMMA_ROOT if gamma else F.CACHE_ROOT}", flush=True)
    workers = min(int(os.environ.get("WORKERS", 3)), len(subjects))
    failed = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, s, force, *extra): s for s in subjects}
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
