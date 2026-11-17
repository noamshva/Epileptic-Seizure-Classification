"""False alarms on seizure-free recordings, plus sensitivity at the same rule.

Per patient:
  1. LORO on the seizure recordings -> window and seizure-event sensitivity.
  2. One model trained on all seizure recordings -> applied to the seizure-free
     recordings (never used for training or tuning) -> false alarms per hour.
Default decision threshold. Alarm = ALARM_RUN consecutive flagged windows.

Needs caches for SPIKES_ROOT and for <SPIKES_ROOT>_free (encoder SEIZURE_FREE=1).
Per-window scores go to analysis/stats/fa_scores_<model>/ for threshold curves.
tune_mask() fixes which seizure-free recordings may be used for tuning; the
rest are held out.

Usage:
    python -u src/eval_false_alarms.py svm
    python -u src/eval_false_alarms.py fft      # FFT band power + SVM (cache_features.py --fft)
    python -u src/eval_false_alarms.py snn
    INPUT=bins READOUT=membrane LR=0.002 CLIP=1 EPOCHS=60 python -u src/eval_false_alarms.py snn
"""
from __future__ import annotations

import importlib.util
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))
import features as F                                                    # noqa: E402

ALARM_RUN = 3
MODEL = next((a for a in sys.argv[1:] if a in ("svm", "snn", "fft")), "svm")
BINS = MODEL == "snn" and os.environ.get("INPUT", "features") == "bins"
NAME = "snn_bins" if BINS else MODEL
_OUT = Path(os.environ.get("PIPELINE_DATA_ROOT", str(_SRC.parent))) / "analysis" / "stats"


def tune_mask(subj, groups):
    """True for windows of the tuning half of a patient's seizure-free recordings."""
    recs = sorted(set(groups.tolist()))
    perm = np.random.default_rng(zlib.crc32(subj.encode())).permutation(len(recs))
    tune = {recs[i] for i in perm[: (len(recs) + 1) // 2]}
    return np.array([r in tune for r in groups.tolist()])


def _load(subj, free):
    root = F.BINS_ROOT if BINS else F.FFT_ROOT if MODEL == "fft" else F.CACHE_ROOT
    if free:
        root = root.parent / (root.name + "_free")
    d = np.load(root / f"{subj}.npz")
    X = d["X_bins"] if BINS else d["X_band"].astype(np.float64)
    return X, d["y"].astype(int), d["groups"], d["start_s"].astype(float)


def _predict(X, y, tr, te):
    """(label, score) for te from a model trained on tr."""
    if MODEL in ("svm", "fft"):
        m = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale",
                                                class_weight="balanced", random_state=0))
        sc = m.fit(X[tr], y[tr]).decision_function(X[te])
        return (sc > 0).astype(int), sc
    spec = importlib.util.spec_from_file_location("snn3c", _SRC / "3c.run_classify_snn_chbmit.py")
    snn = importlib.util.module_from_spec(spec); spec.loader.exec_module(snn)
    prob, pred = snn.fit_predict(X, y, tr, te)[:2]
    return pred, prob


def _segments(g, start):
    """Segment id per window: same recording, consecutive in time."""
    order = np.lexsort((start, g))
    brk = np.ones(len(g), bool)
    brk[1:] = (g[order][1:] != g[order][:-1]) | (np.diff(start[order]) > 2.5)
    seg = np.empty(len(g), int); seg[order] = np.cumsum(brk)
    return seg, order


def _runs(flag, seg, order):
    """(segment, length) of every run of flagged windows."""
    out = []; n = 0; f = flag[order]; s = seg[order]
    for i in range(len(f)):
        if f[i] and (n == 0 or s[i] == s[i - 1]):
            n += 1
        else:
            if n:
                out.append((s[i - 1], n))
            n = 1 if f[i] else 0
    if n:
        out.append((s[-1], n))
    return out


def run_patient(subj):
    X, y, g, st = _load(subj, False)
    Xf, yf, gf, sf = _load(subj, True)

    pred = np.zeros(len(y), int); score = np.full(len(y), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, groups=g):
        if len(set(y[tr].tolist())) == 2:
            pred[te], score[te] = _predict(X, y, tr, te)

    # seizure events = segments of seizure-labelled windows
    sz = np.flatnonzero(y == 1)
    seg, order = _segments(g[sz], st[sz])
    hit = {s for s, n in _runs(pred[sz] == 1, seg, order) if n >= ALARM_RUN}

    Xa = np.concatenate([X, Xf])
    predf, scoref = _predict(Xa, np.concatenate([y, yf]), np.arange(len(y)), np.arange(len(y), len(Xa)))
    sdir = _OUT / f"fa_scores_{NAME}"; sdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(sdir / f"{subj}.npz", y=y, g=g, st=st, score=score,
                        gf=gf, sf=sf, scoref=scoref, tune=tune_mask(subj, gf))
    segf, orderf = _segments(gf, sf)
    alarms = sum(n >= ALARM_RUN for _, n in _runs(predf == 1, segf, orderf))
    hours = len(yf) / 59 / 60

    return {"patient": subj, "seizures": int(seg.max()), "seizures_detected": len(hit),
            "win_sensitivity": float(pred[y == 1].mean()),
            "win_fpr_loro": float(pred[y == 0].mean()),
            "free_recordings": len(set(gf.tolist())), "free_hours": hours,
            "win_fpr_free": float(predf.mean()), "false_alarms": int(alarms),
            "fa_per_hour": alarms / hours}


def main():
    patients = [a for a in sys.argv[1:] if a.startswith("chb")] or [f"chb{n:02d}" for n in range(1, 25)]
    workers = min(int(os.environ.get("WORKERS", 8)), len(patients))
    print(f"{NAME}: {len(patients)} patients, alarm = {ALARM_RUN} consecutive windows", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(run_patient, s) for s in patients]):
            r = fut.result(); rows.append(r)
            print(f"  {r['patient']}: seizures {r['seizures_detected']}/{r['seizures']}"
                  f" | win sens {r['win_sensitivity']:.3f} | free FPR {r['win_fpr_free']:.3f}"
                  f" | FA/h {r['fa_per_hour']:.1f} ({r['free_hours']:.2f} h)", flush=True)
    df = pd.DataFrame(rows).sort_values("patient")
    _OUT.mkdir(parents=True, exist_ok=True)
    out = _OUT / f"false_alarms_{NAME}.csv"
    df.to_csv(out, index=False)
    h = df.free_hours.sum()
    print(f"\nSeizures detected: {df.seizures_detected.sum()}/{df.seizures.sum()}"
          f" | window sensitivity (mean) {df.win_sensitivity.mean():.3f}"
          f" | LORO window FPR (mean) {df.win_fpr_loro.mean():.3f}")
    print(f"Seizure-free: {h:.1f} h, {df.free_recordings.sum()} recordings"
          f" | window FPR pooled {np.average(df.win_fpr_free, weights=df.free_hours):.4f}"
          f" | false alarms {df.false_alarms.sum()} = {df.false_alarms.sum() / h:.2f}/h"
          f" | median patient {df.fa_per_hour.median():.2f}/h")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
