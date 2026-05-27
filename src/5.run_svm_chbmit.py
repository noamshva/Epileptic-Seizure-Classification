"""
Seizure vs non-seizure classification with an SVM on SCTN burstiness features
— the SA paper's main-results method (Table 1), applied to CHB-MIT.

Per 2-second window we extract a feature vector of resonator BURSTINESS
(std of the 10 ms-binned output density) — the quantity we validated as
discriminative (FP1-F7 ~2.6x seizure/non). These features go to a standard
SVM. No SNN simulation, no GPU, reads the existing dataset/spikes_raw/.

Usage:
    python -u src/5.run_svm_chbmit.py
    FEATURE_LEVEL=resonator python -u src/5.run_svm_chbmit.py   # 280 features
"""
import sys
import types
import os

# bindsnet shim (the train module we import pulls in bindsnet via utils)
if "torch._six" not in sys.modules:
    _shim = types.ModuleType("torch._six")
    _shim.container_abcs = ()
    _shim.string_classes = (str,)
    _shim.int_classes = (int,)
    sys.modules["torch._six"] = _shim

import importlib.util as ilu
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
)

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
OUT_DIR = _PROJECT_ROOT / "output_data" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Reuse window enumeration + burstiness computation from the train module.
_spec = ilu.spec_from_file_location("train_chbmit", _SCRIPT_DIR / "3.run_train_chbmit.py")
_t = ilu.module_from_spec(_spec)
_spec.loader.exec_module(_t)

CHANNELS = _t.CHANNELS
BANDS = _t.BANDS
enumerate_windows = _t.enumerate_windows
WINDOW_SECONDS = _t.WINDOW_SECONDS
WINDOW_OVERLAP = _t.WINDOW_OVERLAP

SUBJECTS = ["chb01"]
FEATURE_LEVEL = os.environ.get("FEATURE_LEVEL", "band")   # "band" (40) or "resonator" (280)


def window_feature(win, cache):
    """Return (feature_vector, feature_names) for one window.

    band level     : mean burstiness of the resonators in each (channel, band) -> 40
    resonator level: burstiness of every (channel, resonator)                  -> 280
    """
    bw = _t._window_burstiness(win, cache)   # dict[(ch,band)] -> array[n_res]
    feats, names = [], []
    for ch in CHANNELS:
        for band in BANDS:
            v = bw.get((ch, band))
            if FEATURE_LEVEL == "resonator":
                # fixed-length per band: pad/truncate to the configured count
                arr = np.asarray(v) if v is not None else np.array([])
                for i in range(len(_t.BAND_RESONATORS[band])):
                    feats.append(float(arr[i]) if i < len(arr) else 0.0)
                    names.append(f"{ch}-{band}-r{i}")
            else:
                feats.append(float(np.mean(v)) if v is not None and len(v) else 0.0)
                names.append(f"{ch}-{band}")
    return np.asarray(feats, dtype=np.float64), names


def build_dataset():
    windows = list(enumerate_windows(SUBJECTS, WINDOW_SECONDS, WINDOW_OVERLAP))
    cache = _t._MinuteCache(capacity=8)
    X, y, names = [], [], None
    for win in windows:
        try:
            f, names = window_feature(win, cache)
        except FileNotFoundError as e:
            print(f"  skip {win['abs_start_s']:.1f}s: {e}")
            continue
        X.append(f)
        y.append(1 if win["window_label"] == "seizure" else 0)
    return np.stack(X), np.asarray(y), names, windows


def main():
    print(f"Feature level: {FEATURE_LEVEL}")
    X, y, names, windows = build_dataset()
    n_seiz = int(y.sum())
    print(f"Dataset: {X.shape[0]} windows x {X.shape[1]} features "
          f"({n_seiz} seizure, {len(y) - n_seiz} non-seizure)")

    clf = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0, gamma="scale",
            class_weight="balanced", probability=True),
    )

    # Stratified 5-fold CV. NOTE: windows overlap 50% in time, so adjacent
    # windows are correlated — random CV slightly leaks and inflates accuracy.
    # This is a single-EDF proof-of-concept; the honest evaluation is
    # leave-one-EDF-out / leave-one-patient-out once more data is encoded.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    y_pred = cross_val_predict(clf, X, y, cv=cv)
    y_score = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]

    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    report = classification_report(y, y_pred, target_names=["non_seizure", "seizure"],
                                   output_dict=True, zero_division=0)
    try:
        auc = roc_auc_score(y, y_score)
    except ValueError:
        auc = float("nan")

    print("\n=== 5-fold CV results (SVM-RBF on burstiness features) ===")
    print(pd.DataFrame(report).T.round(3).to_string())
    print(f"\nROC AUC: {auc:.3f}")
    print(f"Confusion matrix [rows=true, cols=pred]  (0=non,1=seiz):\n{cm}")
    print("\nCaveat: 50%-overlap windows from one EDF → random CV leaks somewhat.")
    print("Treat as a 'does the feature carry signal' check, not a final score.")

    # Top discriminative features (mean |t-like| difference between classes)
    seiz_mean = X[y == 1].mean(axis=0)
    non_mean = X[y == 0].mean(axis=0)
    pooled_std = X.std(axis=0) + 1e-9
    effect = np.abs(seiz_mean - non_mean) / pooled_std
    order = np.argsort(effect)[::-1][:15]
    print("\nTop 15 discriminative features (standardized seizure-non difference):")
    for i in order:
        print(f"  {names[i]:20s} effect={effect[i]:.3f}  "
              f"seiz={seiz_mean[i]:.3f} non={non_mean[i]:.3f}")

    # Save confusion matrix figure + a CSV of the report
    pd.DataFrame(report).T.to_csv(OUT_DIR / "svm_report.csv")
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Seizure", "Seizure"],
                yticklabels=["Non-Seizure", "Seizure"])
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(f"SVM on burstiness ({FEATURE_LEVEL})\nAUC={auc:.3f}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "svm_confusion_matrix.png", dpi=150)
    plt.close()
    print(f"\nSaved: {OUT_DIR / 'svm_report.csv'}, {OUT_DIR / 'svm_confusion_matrix.png'}")


if __name__ == "__main__":
    main()
