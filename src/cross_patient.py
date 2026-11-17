"""Cross-patient test: train on 23 patients, test on the one left out.

Models: svm (SCTN burstiness), fft (FFT band power, SVM), snn (SCTN burstiness).
PATIENT_SCALE=1 z-scores each patient's features with that patient's own
statistics first (unlabelled calibration); default scales with the training
patients only.

Usage:
    python -u src/cross_patient.py svm
    PATIENT_SCALE=1 python -u src/cross_patient.py snn
"""
from __future__ import annotations

import importlib.util
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))
import features as F                                                    # noqa: E402

MODEL = next((a for a in sys.argv[1:] if a in ("svm", "snn", "fft")), "svm")
PATIENT_SCALE = os.environ.get("PATIENT_SCALE", "0") == "1"
SUBJECTS = [f"chb{n:02d}" for n in range(1, 25)]
_OUT = Path(os.environ.get("PIPELINE_DATA_ROOT", str(_SRC.parent))) / "analysis" / "stats"


def _data():
    root = F.FFT_ROOT if MODEL == "fft" else F.CACHE_ROOT
    X, y, p = [], [], []
    for s in SUBJECTS:
        d = np.load(root / f"{s}.npz")
        x = d["X_band"].astype(np.float64)
        X.append(StandardScaler().fit_transform(x) if PATIENT_SCALE else x)
        y.append(d["y"].astype(int)); p.append(np.full(len(x), s))
    return np.concatenate(X), np.concatenate(y), np.concatenate(p)


def run(test):
    X, y, p = _data()
    tr, te = np.flatnonzero(p != test), np.flatnonzero(p == test)
    if MODEL == "snn":
        spec = importlib.util.spec_from_file_location("snn3c", _SRC / "3c.run_classify_snn_chbmit.py")
        snn = importlib.util.module_from_spec(spec); spec.loader.exec_module(snn)
        score = snn.fit_predict(X, y, tr, te)[0]
    else:
        m = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale",
                                                class_weight="balanced", random_state=0))
        score = m.fit(X[tr], y[tr]).decision_function(X[te])
    return test, float(roc_auc_score(y[te], score))


def main():
    name = MODEL + ("_patient_scale" if PATIENT_SCALE else "")
    print(f"cross-patient {name}", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=int(os.environ.get("WORKERS", 8))) as ex:
        for fut in as_completed([ex.submit(run, s) for s in SUBJECTS]):
            s, auc = fut.result(); rows.append({"patient": s, "auc": auc})
            print(f"  {s}: AUC={auc:.3f}", flush=True)
    df = pd.DataFrame(rows).sort_values("patient")
    out = _OUT / f"cross_patient_{name}.csv"
    df.to_csv(out, index=False)
    print(f"mean {df.auc.mean():.3f}  median {df.auc.median():.3f}  min {df.auc.min():.3f}"
          f"  patients >= 0.90: {(df.auc >= 0.9).sum()}/24\nSaved: {out}")


if __name__ == "__main__":
    main()
