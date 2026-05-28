"""
Seizure vs non-seizure classification on SCTN burstiness features.

This is the *classical-classifiers* branch entry point: skips the SNN
classifier entirely and feeds the burstiness vectors from `src/features.py`
into a scikit-learn pipeline. First classifier wired up is an SVM (RBF);
other classifiers (RF, XGB, ...) can drop into the same harness later.

Usage (from project root, .venv3_11 active):

    python -u src/3.run_classify_chbmit.py

Env vars:
    FEATURE_LEVEL = band (40 features, default)  |  resonator (280 features)
    CLASSIFIER    = svm (default)                |  (more later)
    KERNEL        = rbf (default)                |  linear  |  poly
    C, GAMMA      = SVC hyperparameters (defaults C=1, gamma='scale')
    CV_FOLDS      = 5 (stratified k-fold; random shuffle, seed 0)
    BALANCE       = 1 (use class_weight='balanced', default)  |  0

Outputs go under output_data/results/.

Caveat: with 50% overlap on a single EDF, random k-fold leaks adjacent
windows across folds — treat the score as "do the features carry signal?"
not as a final benchmark. Honest evaluation = leave-one-EDF-out (and later
leave-one-patient-out) once more data is encoded.
"""
from __future__ import annotations

import os
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

import features as F


# ---------------------------------------------------------------------------
# Configuration (env-overridable)
# ---------------------------------------------------------------------------
SUBJECTS = ["chb01"]
FEATURE_LEVEL = os.environ.get("FEATURE_LEVEL", "band")
CLASSIFIER    = os.environ.get("CLASSIFIER", "svm").lower()
KERNEL        = os.environ.get("KERNEL", "rbf")
C_VAL         = float(os.environ.get("C", "1.0"))
GAMMA         = os.environ.get("GAMMA", "scale")
CV_FOLDS      = int(os.environ.get("CV_FOLDS", "5"))
BALANCE       = os.environ.get("BALANCE", "1") == "1"

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
OUT_DIR = _PROJECT_ROOT / "output_data" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Classifier factory — keep the rest of the script classifier-agnostic so we
# can add CLASSIFIER=rf / xgb later without rewriting the harness.
# ---------------------------------------------------------------------------
def make_classifier(name: str):
    if name == "svm":
        # Try to parse GAMMA as float; otherwise pass the sklearn keyword as-is.
        try:
            gamma = float(GAMMA)
        except ValueError:
            gamma = GAMMA
        clf = SVC(
            kernel=KERNEL, C=C_VAL, gamma=gamma,
            class_weight=("balanced" if BALANCE else None),
            probability=True,
        )
        return make_pipeline(StandardScaler(), clf), f"SVM-{KERNEL} (C={C_VAL}, gamma={gamma})"
    raise ValueError(f"Unknown CLASSIFIER={name!r}. Known: svm.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Subjects     : {SUBJECTS}")
    print(f"Feature level: {FEATURE_LEVEL}")
    print(f"Classifier   : {CLASSIFIER}")
    print(f"CV folds     : {CV_FOLDS}    balanced={BALANCE}")

    X, y, names = F.build_dataset(SUBJECTS, feature_level=FEATURE_LEVEL)
    n_seiz = int(y.sum())
    print(f"Dataset      : {X.shape[0]} windows x {X.shape[1]} features "
          f"({n_seiz} seizure, {len(y) - n_seiz} non-seizure)")
    if n_seiz == 0 or n_seiz == len(y):
        raise RuntimeError("Dataset has only one class — check encoder output / channel selection.")

    pipe, label = make_classifier(CLASSIFIER)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=0)
    y_pred = cross_val_predict(pipe, X, y, cv=cv)
    y_score = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]

    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    report = classification_report(
        y, y_pred, target_names=["non_seizure", "seizure"],
        output_dict=True, zero_division=0,
    )
    try:
        auc = roc_auc_score(y, y_score)
    except ValueError:
        auc = float("nan")

    print(f"\n=== {CV_FOLDS}-fold CV results ({label}) ===")
    print(pd.DataFrame(report).T.round(3).to_string())
    print(f"\nROC AUC: {auc:.3f}")
    print(f"Confusion matrix [rows=true, cols=pred]  (0=non,1=seiz):\n{cm}")
    print("\nCaveat: 50%-overlap windows from one EDF -> random CV leaks somewhat.")
    print("Treat as a 'does the feature carry signal' check, not a final score.")

    # Top discriminative features (standardized seizure-vs-non difference).
    seiz_mean = X[y == 1].mean(axis=0)
    non_mean = X[y == 0].mean(axis=0)
    pooled_std = X.std(axis=0) + 1e-9
    effect = np.abs(seiz_mean - non_mean) / pooled_std
    order = np.argsort(effect)[::-1][:15]
    print("\nTop 15 discriminative features (standardized seizure-non difference):")
    for i in order:
        print(f"  {names[i]:24s} effect={effect[i]:.3f}  "
              f"seiz={seiz_mean[i]:.3f} non={non_mean[i]:.3f}")

    # Persist results.
    report_csv = OUT_DIR / f"{CLASSIFIER}_report.csv"
    cm_png = OUT_DIR / f"{CLASSIFIER}_confusion_matrix.png"
    pd.DataFrame(report).T.to_csv(report_csv)

    plt.figure(figsize=(4, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Non-Seizure", "Seizure"],
        yticklabels=["Non-Seizure", "Seizure"],
    )
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(f"{label}\n[{FEATURE_LEVEL} features]   AUC={auc:.3f}")
    plt.tight_layout()
    plt.savefig(cm_png, dpi=150)
    plt.close()

    print(f"\nSaved: {report_csv}")
    print(f"Saved: {cm_png}")


if __name__ == "__main__":
    main()
