"""
Side-by-side check that the SCTN burstiness features land at the same SVM
accuracy as classical FFT band-power features on the *exact* same EEG windows.

Pipeline per window:
    raw EEG (256 Hz, 10 bipolar ch)
        |
        +---- FFT path  ---- STFT → mean band power per (ch, band) → 40-D
        |
        +---- SCTN path ---- burstiness per (ch, band)              → 40-D
        |
        v
    StandardScaler  →  SVM-RBF (C=1.0, gamma='scale', class_weight='balanced')
        |
        v
    5-fold stratified CV → accuracy / F1 / AUC

This script is intentionally narrow: ONE classifier (SVM-RBF), ONE comparison.
For broader benchmarks (more classifiers, harder CV), see future work.

Usage (from project root):
    python -u src/2b.compare_fft_vs_sctn.py

Outputs:
    output_data/results/fft_vs_sctn_svm.csv          accuracy / F1 / AUC table
    output_data/results/fft_vs_sctn_svm_bar.png      paired bar chart
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
from scipy.signal import butter, filtfilt, spectrogram

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
import features as F                                                    # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SUBJECTS = ["chb01"]
CV_FOLDS = 5
SEED = 0
STFT_MS = 1000.0          # 1-s STFT frames → Δf=1 Hz, resolves Delta cleanly
FS = 256                  # CHB-MIT sampling rate

_PROJECT_ROOT = _SCRIPT_DIR.parent
CHBMIT_ROOT = _PROJECT_ROOT / "dataset" / "chbmit"
OUT_DIR = _PROJECT_ROOT / "output_data" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Band edges — kept in sync with features.BANDS / 2.analyze_chbmit.py.
BAND_EDGES = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 14.0),
    "Beta":  (14.0, 32.0),
}


# ---------------------------------------------------------------------------
# FFT side — self-contained band-power extractor on the same windows that
# F.enumerate_windows yields. Mirrors the encoder's 0.5-62 Hz Butterworth.
# ---------------------------------------------------------------------------
def _bp_filter(data: np.ndarray, f_lo: float, f_hi: float, fs: int) -> np.ndarray:
    f_ny = fs / 2.0
    b1, a1 = butter(6, f_hi / f_ny, btype="lowpass")
    b2, a2 = butter(6, f_lo / f_ny, btype="highpass")
    out = filtfilt(b1, a1, data, axis=0)
    return filtfilt(b2, a2, out, axis=0)


def _load_eeg_per_trial() -> dict[str, dict[str, np.ndarray]]:
    """Read each subject's EDFs, return {trial_stem: {channel: bandpassed_signal}}."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for subj in SUBJECTS:
        subj_dir = CHBMIT_ROOT / subj
        if not subj_dir.is_dir():
            continue
        spike_trials = {p.name for p in (Path(F.SPIKES_ROOT) / subj).glob("*")}
        for edf in sorted(subj_dir.glob("*.edf")):
            if edf.stem not in spike_trials:
                continue                                # skip EDFs we haven't encoded
            raw = mne.io.read_raw_edf(str(edf), preload=True, verbose="ERROR")
            avail = [c.strip() for c in raw.ch_names]
            picks, mapping = [], {}
            for desired in F.CHANNELS:
                d = desired.upper().replace(" ", "")
                for a in avail:
                    an = a.upper().replace(" ", "")
                    if an == d or an.startswith(d) or d in an or an in d:
                        picks.append(a); mapping[a] = desired; break
            if not picks:
                continue
            data = raw.get_data(picks=picks).T
            data = _bp_filter(data, 0.5, 62.0, FS)
            out[edf.stem] = {mapping[p]: data[:, i] for i, p in enumerate(picks)}
    return out


def _window_fft_features(signal: np.ndarray) -> np.ndarray:
    """Mean STFT power per (Delta/Theta/Alpha/Beta) for a single-channel 2-s window."""
    nperseg = max(64, int(FS * STFT_MS / 1000.0))
    nperseg = min(nperseg, len(signal))
    f, _, Sxx = spectrogram(signal, fs=FS, nperseg=nperseg, noverlap=nperseg // 2,
                            scaling="density", mode="psd")
    psd = Sxx.mean(axis=1)
    feats = np.zeros(len(F.BANDS), dtype=np.float64)
    for i, band in enumerate(F.BANDS):
        lo, hi = BAND_EDGES[band]
        feats[i] = float(psd[(f >= lo) & (f < hi)].sum())
    return feats


def build_dataset_fft() -> tuple[np.ndarray, np.ndarray]:
    eeg = _load_eeg_per_trial()
    X, y = [], []
    for win in F.enumerate_windows(SUBJECTS, F.WINDOW_SECONDS, F.WINDOW_OVERLAP):
        trial_eeg = eeg.get(win["trial"])
        if trial_eeg is None:
            continue
        start = int(round(win["abs_start_s"] * FS))
        end = start + int(round(F.WINDOW_SECONDS * FS))
        feats = []
        for ch in F.CHANNELS:
            sig = trial_eeg.get(ch)
            if sig is None or end > len(sig):
                feats.extend([0.0] * len(F.BANDS))
                continue
            feats.extend(_window_fft_features(sig[start:end]).tolist())
        X.append(np.asarray(feats, dtype=np.float64))
        y.append(1 if win["window_label"] == "seizure" else 0)
    return np.stack(X), np.asarray(y)


# ---------------------------------------------------------------------------
# Eval helper
# ---------------------------------------------------------------------------
def _eval(X: np.ndarray, y: np.ndarray) -> dict:
    clf = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0, gamma="scale",
            class_weight="balanced", probability=True, random_state=SEED),
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    y_pred = cross_val_predict(clf, X, y, cv=cv)
    y_score = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    return {
        "accuracy":   accuracy_score(y, y_pred),
        "f1_seizure": f1_score(y, y_pred, pos_label=1, zero_division=0),
        "roc_auc":    roc_auc_score(y, y_score),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Subjects: {SUBJECTS}    Channels: {len(F.CHANNELS)}    "
          f"Bands: {list(F.BANDS)}    CV: {CV_FOLDS}-fold stratified, seed {SEED}")

    print("Building SCTN burstiness features (40-D per window)...")
    X_sctn, y_sctn, _ = F.build_dataset(SUBJECTS, feature_level="band")
    print(f"  SCTN: {X_sctn.shape}   pos={int(y_sctn.sum())}  neg={int((y_sctn==0).sum())}")

    print(f"Building FFT band-power features (40-D per window, STFT {STFT_MS:g} ms)...")
    X_fft, y_fft = build_dataset_fft()
    assert (y_fft == y_sctn).all(), "FFT/SCTN windows must align"
    print(f"  FFT:  {X_fft.shape}")

    print("\nEvaluating SVM-RBF on both feature sets...")
    fft_r  = _eval(X_fft,  y_fft)
    sctn_r = _eval(X_sctn, y_sctn)

    df = pd.DataFrame([
        {"Features": "FFT band power",  **fft_r},
        {"Features": "SCTN burstiness", **sctn_r},
    ])
    df["Δ accuracy (SCTN − FFT)"] = [
        np.nan,
        sctn_r["accuracy"] - fft_r["accuracy"],
    ]

    csv_path = OUT_DIR / "fft_vs_sctn_svm.csv"
    df.to_csv(csv_path, index=False)
    print("\n=== SVM-RBF accuracy (5-fold stratified CV) ===")
    print(df.round(4).to_string(index=False))

    # ----- Paired bar chart -----
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    bars = ax.bar(
        ["FFT", "SCTN"],
        [fft_r["accuracy"], sctn_r["accuracy"]],
        width=0.55,
        color=["#7f7f7f", "#1f77b4"],
        edgecolor="black", linewidth=0.6,
    )
    ax.set_ylim(0.9, 1.005)
    ax.set_ylabel("Accuracy (5-fold CV)")
    ax.set_title("FFT vs SCTN — SVM accuracy", fontsize=12, pad=10)
    ax.grid(axis="y", alpha=0.3)
    for bar, r in zip(bars, (fft_r, sctn_r)):
        x = bar.get_x() + bar.get_width() / 2
        ax.annotate(f"{r['accuracy']:.3f}", (x, r["accuracy"]),
                    ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax.annotate(f"F1 = {r['f1_seizure']:.3f}\nAUC = {r['roc_auc']:.3f}",
                    (x, 0.905), ha="center", va="bottom",
                    fontsize=9, color="#444")
    fig.tight_layout()
    png_path = OUT_DIR / "fft_vs_sctn_svm_bar.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    print(f"\nSaved:\n  {csv_path}\n  {png_path}")


if __name__ == "__main__":
    main()
