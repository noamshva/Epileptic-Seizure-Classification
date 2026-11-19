"""False alarms on seizure-free recordings the model never saw.

Control for the final test, where two other minutes of every test recording are
in the training data. Per patient, the seizure-free recordings of
<SPIKES_ROOT>_free are split into FOLDS groups: train on all seizure recordings
plus the other groups, score the held-out group. The final test data
(<SPIKES_ROOT>_free2) is not used.

Writes analysis/stats/free_unseen_<model>_k<K>.csv.

Usage:
    ALARM_RUN=5 python -u src/eval_free_unseen.py svm
    ALARM_RUN=5 python -u src/eval_free_unseen.py fft
    ALARM_RUN=5 SEED=0 python -u src/eval_free_unseen.py snn
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))
import eval_false_alarms as E                                           # noqa: E402

K = E.ALARM_RUN
FOLDS = int(os.environ.get("FOLDS", 5))


def run_patient(subj):
    X, y, _, _ = E._load(subj)
    Xf, yf, gf, sf = E._load(subj, "_free")
    Xa = np.concatenate([X, Xf]); ya = np.concatenate([y, yf])
    main = np.arange(len(y)); pred = np.zeros(len(yf), int)
    for tr, te in GroupKFold(min(FOLDS, len(set(gf.tolist())))).split(Xf, yf, groups=gf):
        pred[te] = E._predict(Xa, ya, np.concatenate([main, len(y) + tr]), len(y) + te)[0]
    seg, order = E._segments(gf, sf)
    alarms = sum(n >= K for _, n in E._runs(pred == 1, seg, order))
    hours = len(yf) / 59 / 60
    return {"patient": subj, "free_recordings": len(set(gf.tolist())), "free_hours": hours,
            "win_fpr": float(pred.mean()), "false_alarms": int(alarms), "fa_per_hour": alarms / hours}


def main():
    patients = [a for a in sys.argv[1:] if a.startswith("chb")] or [f"chb{n:02d}" for n in range(1, 25)]
    workers = min(int(os.environ.get("WORKERS", 6)), len(patients))
    print(f"{E.NAME}: unseen seizure-free recordings, {len(patients)} patients, "
          f"alarm = {K} consecutive windows", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(run_patient, s) for s in patients]):
            r = fut.result(); rows.append(r)
            print(f"  {r['patient']}: {r['free_hours']:.2f} h | alarms {r['false_alarms']}", flush=True)
    df = pd.DataFrame(rows).sort_values("patient")
    out = E._OUT / f"free_unseen_{E.NAME}_k{K}.csv"
    df.to_csv(out, index=False)
    h, a = df.free_hours.sum(), df.false_alarms.sum()
    print(f"{h:.1f} h | window FPR {np.average(df.win_fpr, weights=df.free_hours):.4f} | "
          f"alarms {a} = {a / h:.2f}/h")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
