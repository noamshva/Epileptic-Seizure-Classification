"""Seizure vs non-seizure classifier on SCTN burstiness features (SVM).

The third stage of the pipeline: encode (1) -> analyse (2) -> classify (3).
Feeds the burstiness vectors from features.py into a scikit-learn classifier and
reports performance per patient with BOTH:
  - leave-one-recording-out (LORO)  -- the honest, leakage-free score
  - stratified 5-fold               -- reference only (optimistic: 50%-overlap
                                       windows leak near-duplicates across folds)

The comprehensive cross-patient comparison / figures live in src/2.analyze; this
file is just the classifier and its per-patient scorecard + confusion matrix.

Usage:
    python -u src/3.run_classify_chbmit.py                 # SVM, chb01
    python -u src/3.run_classify_chbmit.py chb01 chb03     # specific patients
Env:
    FEATURE_LEVEL = band (default) | resonator
    CLASSIFIER    = svm (default) | rf | gb
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, confusion_matrix, classification_report,
)

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
import features as F                                                    # noqa: E402

FEATURE_LEVEL = os.environ.get("FEATURE_LEVEL", "band")
SEED = 0
OUT_DIR = _SCRIPT_DIR.parent / "output_data" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_classifier(name: str):
    """Standardised pipeline for the requested classical classifier."""
    name = name.lower()
    if name == "svm":
        clf = SVC(kernel="rbf", C=1.0, gamma="scale",
                  class_weight="balanced", probability=True, random_state=SEED)
    elif name == "rf":
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     random_state=SEED, n_jobs=-1)
    elif name == "gb":
        clf = GradientBoostingClassifier(random_state=SEED)
    else:
        raise ValueError(f"Unknown CLASSIFIER={name!r} (svm|rf|gb)")
    return make_pipeline(StandardScaler(), clf), name


def _loro_pooled(pipe, X, y, g):
    """Pooled leave-one-recording-out predictions + probabilities."""
    logo = LeaveOneGroupOut()
    yp = np.zeros(len(y), int)
    ps = np.full(len(y), np.nan)
    for tr, te in logo.split(X, y, groups=g):
        m = pipe.fit(X[tr], y[tr])
        yp[te] = m.predict(X[te])
        ps[te] = m.predict_proba(X[te])[:, 1]
    return yp, ps


def main(classifier: str | None = None):
    patients = [a for a in sys.argv[1:] if not a.startswith("-")] or ["chb01"]
    name = (classifier or os.environ.get("CLASSIFIER", "svm")).lower()
    print(f"Classifier: {name}   feature_level: {FEATURE_LEVEL}   patients: {patients}")

    rows = []
    pooled_y, pooled_pred = [], []
    for subj in patients:
        try:
            X, y, g, names = F.build_dataset_groups([subj], feature_level=FEATURE_LEVEL)
        except Exception as e:
            print(f"  {subj}: skip ({e})"); continue
        n_rec = len(set(g.tolist()))
        if int(y.sum()) == 0 or int(y.sum()) == len(y):
            print(f"  {subj}: single-class, skip"); continue

        pipe, _ = make_classifier(name)
        # honest LORO
        if n_rec >= 2:
            yp, ps = _loro_pooled(pipe, X, y, g)
            loro_auc = roc_auc_score(y, ps) if len(set(y.tolist())) > 1 else float("nan")
            loro_f1 = f1_score(y, yp, zero_division=0)
            loro_acc = accuracy_score(y, yp)
            pooled_y.append(y); pooled_pred.append(yp)
        else:
            loro_auc = loro_f1 = loro_acc = float("nan"); yp = None

        # stratified reference
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        pipe_s, _ = make_classifier(name)
        sp = cross_val_predict(pipe_s, X, y, cv=cv, method="predict_proba")[:, 1]
        strat_auc = roc_auc_score(y, sp)

        rows.append({"patient": subj, "n_rec": n_rec, "n_win": len(y),
                     "seizure": int(y.sum()), "loro_auc": loro_auc, "loro_f1": loro_f1,
                     "loro_acc": loro_acc, "stratified_auc": strat_auc})
        print(f"  {subj}: LORO AUC={loro_auc:.3f} F1={loro_f1:.3f} | "
              f"stratified AUC={strat_auc:.3f}")

    if not rows:
        print("No patients classified."); return
    df = pd.DataFrame(rows)
    rep_csv = OUT_DIR / f"{name}_classifier_report.csv"
    df.to_csv(rep_csv, index=False)
    print(f"\nMean LORO AUC (patients with >=2 recordings): "
          f"{df['loro_auc'].mean(skipna=True):.3f}")

    # pooled confusion matrix (honest LORO)
    if pooled_y:
        yy = np.concatenate(pooled_y); pp = np.concatenate(pooled_pred)
        cm = confusion_matrix(yy, pp, labels=[0, 1])
        print("\nPooled LORO classification report:")
        print(classification_report(yy, pp, target_names=["non_seizure", "seizure"],
                                    zero_division=0))
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black", fontsize=13)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Non-seizure", "Seizure"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Non-seizure", "Seizure"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{name.upper()} — pooled LORO confusion")
        fig.tight_layout()
        cm_png = OUT_DIR / f"{name}_confusion_matrix.png"
        fig.savefig(cm_png, dpi=150); plt.close(fig)
        print(f"Saved: {rep_csv}\n       {cm_png}")


if __name__ == "__main__":
    main()
