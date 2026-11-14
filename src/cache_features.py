"""Extract burstiness features once per patient and cache them to disk.

Reading the per-resonator spike files takes minutes per patient; classifiers
that train for many epochs load the cache instead.

Writes <data root>/dataset/features_cache/<subject>.npz with:
    X_res   [n, 280]  per-resonator burstiness
    X_band  [n, 40]   mean per (channel, band)
    y       [n]       1 = seizure
    groups  [n]       recording (trial) of each window, for LORO
    start_s [n]       window start within the recording

Usage:
    python -u src/cache_features.py                # all encoded patients
    python -u src/cache_features.py chb01 chb03
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


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    force = "--force" in sys.argv
    subjects = args or sorted(p.name for p in Path(F.SPIKES_ROOT).iterdir() if p.is_dir())
    print(f"Caching {len(subjects)} patients -> {F.CACHE_ROOT}", flush=True)
    workers = min(int(os.environ.get("WORKERS", 3)), len(subjects))
    failed = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(cache_subject, s, force): s for s in subjects}
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
