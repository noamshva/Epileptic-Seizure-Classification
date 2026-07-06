"""Leakage-free FFT-vs-SCTN comparison: leave-one-recording-out CV.

2b uses 5-fold *stratified* CV, which is optimistic here because 2-s windows
overlap 50% -- near-duplicate windows can land in both train and test. This
script groups by recording and holds entire recordings out (LeaveOneGroupOut),
so the score reflects generalisation to an unseen recording. Prints the
stratified number too for side-by-side context.

Reuses 2b's exact FFT/SCTN feature extraction so the only difference vs 2b is
the CV scheme.

Usage:  python -u src/2c.groupkfold_fft_vs_sctn.py
Outputs:
    output_data/results/fft_vs_sctn_groupcv.csv
    output_data/results/fft_vs_sctn_groupcv_bar.png
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    StratifiedKFold, LeaveOneGroupOut, cross_val_predict,
)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR))
import features as F                                                    # noqa: E402

# Load 2b to reuse its FFT extraction (module name starts with a digit).
_spec = importlib.util.spec_from_file_location(
    "cmp2b", _DIR / "2b.compare_fft_vs_sctn.py")
cmp2b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmp2b)

SUBJECTS = ["chb01"]
SEED = 0
OUT_DIR = _DIR.parent / "output_data" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _build_sctn():
    X, y, g = [], [], []
    for feat, label, win in F.iter_window_features(SUBJECTS, feature_level="band"):
        X.append(feat)
        y.append(label)
        g.append(win["trial"])
    return np.stack(X), np.asarray(y), np.asarray(g)


def _build_fft():
    eeg = cmp2b._load_eeg_per_trial()
    X, y, g = [], [], []
    for win in F.enumerate_windows(SUBJECTS, F.WINDOW_SECONDS, F.WINDOW_OVERLAP):
        trial_eeg = eeg.get(win["trial"])
        if trial_eeg is None:
            continue
        start = int(round(win["abs_start_s"] * cmp2b.FS))
        end = start + int(round(F.WINDOW_SECONDS * cmp2b.FS))
        feats = []
        for ch in F.CHANNELS:
            sig = trial_eeg.get(ch)
            if sig is None or end > len(sig):
                feats.extend([0.0] * len(F.BANDS))
                continue
            feats.extend(cmp2b._window_fft_features(sig[start:end]).tolist())
        X.append(np.asarray(feats, dtype=np.float64))
        y.append(1 if win["window_label"] == "seizure" else 0)
        g.append(win["trial"])
    return np.stack(X), np.asarray(y), np.asarray(g)


def _clf():
    return make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0, gamma="scale",
            class_weight="balanced", probability=True, random_state=SEED),
    )


def _eval(X, y, cv, groups=None):
    yp = cross_val_predict(_clf(), X, y, cv=cv, groups=groups)
    ys = cross_val_predict(_clf(), X, y, cv=cv, groups=groups,
                           method="predict_proba")[:, 1]
    return {
        "accuracy":   accuracy_score(y, yp),
        "f1_seizure": f1_score(y, yp, pos_label=1, zero_division=0),
        "roc_auc":    roc_auc_score(y, ys),
    }


def main() -> None:
    print("Building SCTN burstiness features...")
    Xs, ys, gs = _build_sctn()
    print(f"  SCTN: {Xs.shape}  pos={int(ys.sum())} neg={int((ys==0).sum())}")

    print("Building FFT band-power features...")
    Xf, yf, gf = _build_fft()
    print(f"  FFT:  {Xf.shape}")

    assert (ys == yf).all(), "FFT/SCTN labels must align"
    assert (gs == gf).all(), "FFT/SCTN groups must align"

    recordings = sorted(set(gs.tolist()))
    print(f"\nGroups (recordings): {len(recordings)} -> {recordings}")

    strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    logo = LeaveOneGroupOut()

    rows = []
    for name, X, y in [("FFT band power", Xf, yf), ("SCTN burstiness", Xs, ys)]:
        s = _eval(X, y, strat)
        l = _eval(X, y, logo, groups=gs)
        rows.append({"Features": name,
                     "acc_stratified": s["accuracy"], "f1_stratified": s["f1_seizure"], "auc_stratified": s["roc_auc"],
                     "acc_LORO": l["accuracy"], "f1_LORO": l["f1_seizure"], "auc_LORO": l["roc_auc"]})

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "fft_vs_sctn_groupcv.csv"
    df.to_csv(csv_path, index=False)

    print("\n=== Stratified 5-fold (optimistic) vs Leave-One-Recording-Out (leakage-free) ===")
    print(df.round(4).to_string(index=False))

    # ----- Grouped bar: LORO accuracy, both feature sets -----
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    fft_loro = df.loc[df.Features == "FFT band power", "acc_LORO"].iloc[0]
    sctn_loro = df.loc[df.Features == "SCTN burstiness", "acc_LORO"].iloc[0]
    bars = ax.bar(["FFT", "SCTN"], [fft_loro, sctn_loro], width=0.55,
                  color=["#7f7f7f", "#1f77b4"], edgecolor="black", linewidth=0.6)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Accuracy (leave-one-recording-out)")
    ax.set_title("FFT vs SCTN — leakage-free CV", fontsize=12, pad=10)
    ax.grid(axis="y", alpha=0.3)
    for bar, name in zip(bars, ("FFT band power", "SCTN burstiness")):
        r = df.loc[df.Features == name].iloc[0]
        x = bar.get_x() + bar.get_width() / 2
        ax.annotate(f"{r['acc_LORO']:.3f}", (x, r["acc_LORO"]),
                    ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax.annotate(f"F1 = {r['f1_LORO']:.3f}\nAUC = {r['auc_LORO']:.3f}",
                    (x, 0.04), ha="center", va="bottom", fontsize=9, color="#444")
    fig.tight_layout()
    png_path = OUT_DIR / "fft_vs_sctn_groupcv_bar.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    print(f"\nSaved:\n  {csv_path}\n  {png_path}")


if __name__ == "__main__":
    main()
