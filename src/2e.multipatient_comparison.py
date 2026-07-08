"""Multi-patient FFT-vs-SCTN comparison (patient-specific, leakage-free, with CIs).

For every encoded patient, run leave-one-recording-out (LORO) and report AUC
(threshold-free) with recording-level bootstrap confidence intervals -- the
exact design the item-1 audit said is needed to make a defensible claim.

Feature sets (all classified by the same SVM-RBF):
  FFT-fine   : per-frequency-bin PSD 0.5-62 Hz  (fair fine-grained FFT)
  SCTN-350   : per-resonator burstiness, Delta..Gamma (uses the new gamma band)
  SCTN-280   : per-resonator burstiness, Delta..Beta  (gamma dropped)
Each is also evaluated after PCA->40 to control for feature-count (capacity).

Outputs:
  output_data/results/multipatient_fft_vs_sctn.csv
  output_data/results/multipatient_gap_forest.png   (per-patient AUC gap + 95% CI)

Usage:
  python -u src/2e.multipatient_comparison.py            # all encoded patients
  python -u src/2e.multipatient_comparison.py chb01 chb03  # subset (for testing)
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
from scipy.signal import spectrogram

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))
import features as F  # noqa: E402
_spec = importlib.util.spec_from_file_location("cmp2c", _SRC / "2c.groupkfold_fft_vs_sctn.py")
c2 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(c2)
cmp2b = c2.cmp2b

_ROOT = _SRC.parent
SPIKES_ROOT = _ROOT / "dataset" / "spikes_raw"
OUT_DIR = _ROOT / "output_data" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 256
SEED = 0
FFT_LO, FFT_HI = 0.5, 62.0
PCA_DIM = 40
N_BOOT = 2000
MIN_RECORDINGS = 2          # LORO needs >=2; results flagged unreliable if <3
GAMMA_RES = [34.7, 37.2, 40.2, 43.2, 47.7, 52.6, 57.2]   # clk 153600

rng = np.random.default_rng(SEED)

# preserve features.py band config so we can toggle gamma per build
_ORIG_BANDS = F.BANDS
_ORIG_RES = dict(F.BAND_RESONATORS)


# --------------------------------------------------------------------------
# Feature builders
# --------------------------------------------------------------------------
def build_sctn_350(subj):
    """Resonator-level burstiness incl. gamma -> (X[n,350], y, groups, names)."""
    F.BANDS = ("Delta", "Theta", "Alpha", "Beta", "Gamma")
    F.BAND_RESONATORS = {**_ORIG_RES, "Gamma": GAMMA_RES}
    try:
        X, y, g = [], [], []
        for feat, label, win in F.iter_window_features([subj], feature_level="resonator"):
            X.append(feat); y.append(label); g.append(win["trial"])
        names = F.feature_names("resonator")
    finally:
        F.BANDS = _ORIG_BANDS
        F.BAND_RESONATORS = dict(_ORIG_RES)
    if not X:
        return None
    return np.stack(X), np.asarray(y), np.asarray(g), names


def build_fft_fine(subj):
    c2.SUBJECTS = [subj]; cmp2b.SUBJECTS = [subj]
    eeg = cmp2b._load_eeg_per_trial()
    X, y, g = [], [], []
    nb = None
    for win in F.enumerate_windows([subj], F.WINDOW_SECONDS, F.WINDOW_OVERLAP):
        te = eeg.get(win["trial"])
        if te is None:
            continue
        s = int(round(win["abs_start_s"] * FS)); e = s + int(round(F.WINDOW_SECONDS * FS))
        feats = []
        for ch in F.CHANNELS:
            sig = te.get(ch)
            if sig is None or e > len(sig):
                feats.append(None); continue
            nper = min(FS, len(sig[s:e]))
            f, _, Sxx = spectrogram(sig[s:e], fs=FS, nperseg=nper, noverlap=nper // 2,
                                    scaling="density", mode="psd")
            feats.append(Sxx.mean(axis=1)[(f >= FFT_LO) & (f < FFT_HI)])
        good = [v for v in feats if v is not None]
        if not good:
            continue
        if nb is None:
            nb = len(good[0])
        X.append(np.concatenate([v if v is not None else np.zeros(nb) for v in feats]))
        y.append(1 if win["window_label"] == "seizure" else 0); g.append(win["trial"])
    if not X:
        return None
    return np.stack(X), np.asarray(y), np.asarray(g)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def _clf(pca=None):
    steps = [StandardScaler()]
    if pca:
        steps.append(PCA(n_components=pca, random_state=SEED))
    steps.append(SVC(kernel="rbf", C=1.0, gamma="scale",
                     class_weight="balanced", probability=True, random_state=SEED))
    return make_pipeline(*steps)


def loro_probs(X, y, g, pca=None):
    """Return pooled held-out probabilities + per-fold AUCs (nan if degenerate)."""
    logo = LeaveOneGroupOut()
    p = np.full(len(y), np.nan)
    perfold = {}
    for tr, te in logo.split(X, y, groups=g):
        n_comp = pca
        if pca is not None:
            n_comp = min(pca, X[tr].shape[1], max(2, len(tr) - 1))
        pipe = _clf(n_comp).fit(X[tr], y[tr])
        p[te] = pipe.predict_proba(X[te])[:, 1]
        rec = g[te][0]
        perfold[rec] = (roc_auc_score(y[te], p[te]) if len(set(y[te].tolist())) > 1 else np.nan)
    return p, perfold


def pooled_auc(y, p):
    return roc_auc_score(y, p) if len(set(y.tolist())) > 1 else np.nan


def cluster_bootstrap_diff(y, g, p_a, p_b, n=N_BOOT):
    """Recording-level bootstrap CI of pooled AUC(a) - AUC(b)."""
    recs = np.unique(g)
    idx = {r: np.where(g == r)[0] for r in recs}
    diffs = []
    for _ in range(n):
        samp = rng.choice(recs, size=len(recs), replace=True)
        ii = np.concatenate([idx[r] for r in samp])
        yy = y[ii]
        if len(set(yy.tolist())) < 2:
            continue
        diffs.append(roc_auc_score(yy, p_a[ii]) - roc_auc_score(yy, p_b[ii]))
    if not diffs:
        return (np.nan, np.nan, np.nan, np.nan)
    lo, med, hi = np.percentile(diffs, [2.5, 50, 97.5])
    return (med, lo, hi, float(np.mean(np.array(diffs) > 0)))


# --------------------------------------------------------------------------
# Per-patient driver
# --------------------------------------------------------------------------
def run_patient(subj):
    sctn = build_sctn_350(subj)
    fft = build_fft_fine(subj)
    if sctn is None or fft is None:
        print(f"  {subj}: no windows, skipping"); return None
    X350, y, g, names = sctn
    Xf, yf, gf = fft
    if not ((y == yf).all() and (g == gf).all()):
        print(f"  {subj}: FFT/SCTN misaligned, skipping"); return None
    n_rec = len(set(g.tolist()))
    if n_rec < MIN_RECORDINGS:
        print(f"  {subj}: only {n_rec} recording(s), skipping"); return None

    # SCTN-280 = drop gamma columns
    keep280 = np.array([("-Gamma-" not in nm) for nm in names])
    X280 = X350[:, keep280]

    row = {"patient": subj, "n_rec": n_rec, "n_win": len(y),
           "seizure": int(y.sum()), "reliable": n_rec >= 3}
    probs = {}
    for tag, X in [("FFT", Xf), ("SCTN350", X350), ("SCTN280", X280)]:
        for pca, ptag in [(None, ""), (PCA_DIM, "_pca")]:
            p, perfold = loro_probs(X, y, g, pca=pca)
            probs[tag + ptag] = p
            row[f"auc_{tag}{ptag}"] = pooled_auc(y, p)
            if pca is None:
                vals = [v for v in perfold.values() if not np.isnan(v)]
                row[f"degenerate_folds_{tag}"] = int(sum(np.isnan(v) for v in perfold.values()))
    # gaps FFT - SCTN (main comparisons), raw and capacity-matched
    for stag in ["SCTN350", "SCTN280"]:
        for ptag in ["", "_pca"]:
            med, lo, hi, pgt = cluster_bootstrap_diff(y, g, probs["FFT" + ptag], probs[stag + ptag])
            key = f"gap_FFT_{stag}{ptag}"
            row[key] = med; row[key + "_lo"] = lo; row[key + "_hi"] = hi; row[key + "_pFFTgt"] = pgt
    print(f"  {subj}: n_rec={n_rec} win={len(y)}  "
          f"AUC FFT={row['auc_FFT']:.3f} SCTN350={row['auc_SCTN350']:.3f} SCTN280={row['auc_SCTN280']:.3f}  "
          f"gap(FFT-SCTN350,pca40)={row['gap_FFT_SCTN350_pca']:+.3f} "
          f"[{row['gap_FFT_SCTN350_pca_lo']:+.3f},{row['gap_FFT_SCTN350_pca_hi']:+.3f}]", flush=True)
    return row


def main():
    requested = sys.argv[1:]
    patients = requested or sorted(p.name for p in SPIKES_ROOT.iterdir() if p.is_dir())
    print(f"Patients: {patients}\n")
    rows = []
    for subj in patients:
        try:
            r = run_patient(subj)
        except Exception as ex:
            print(f"  {subj}: ERROR {type(ex).__name__}: {ex}")
            r = None
        if r:
            rows.append(r)
    if not rows:
        print("No patients evaluated."); return
    df = pd.DataFrame(rows)
    csv = OUT_DIR / "multipatient_fft_vs_sctn.csv"
    df.to_csv(csv, index=False)

    # ---- cross-patient summary (reliable patients: n_rec>=3) ----
    rel = df[df.reliable].copy()
    print("\n================ CROSS-PATIENT SUMMARY ================")
    print(f"patients evaluated: {len(df)}   reliable (>=3 recordings): {len(rel)}")
    for stag in ["SCTN350", "SCTN280"]:
        for ptag, plabel in [("", "raw"), ("_pca", "PCA40")]:
            gaps = rel[f"gap_FFT_{stag}{ptag}"].dropna().values
            auc_fft = rel["auc_FFT" + ptag].mean()
            auc_sctn = rel[f"auc_{stag}{ptag}"].mean()
            if len(gaps) == 0:
                continue
            # patient-level bootstrap of the mean gap
            bmeans = [rng.choice(gaps, len(gaps), replace=True).mean() for _ in range(N_BOOT)]
            lo, hi = np.percentile(bmeans, [2.5, 97.5])
            sctn_wins = int((gaps < 0).sum())
            print(f"FFT vs {stag} ({plabel}): mean AUC {auc_fft:.3f} vs {auc_sctn:.3f} | "
                  f"mean gap {gaps.mean():+.3f}  95%CI [{lo:+.3f},{hi:+.3f}] | "
                  f"SCTN>=FFT in {sctn_wins}/{len(gaps)} patients")

    # ---- forest plot: per-patient gap FFT - SCTN350 (PCA40, matched capacity) ----
    fp = df.sort_values("gap_FFT_SCTN350_pca")
    gap = fp["gap_FFT_SCTN350_pca"].to_numpy(dtype=float)
    glo = fp["gap_FFT_SCTN350_pca_lo"].to_numpy(dtype=float)
    ghi = fp["gap_FFT_SCTN350_pca_hi"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(fp) + 1)))
    yy = np.arange(len(fp))
    ax.errorbar(gap, yy, xerr=[gap - glo, ghi - gap],
                fmt="o", color="#1f77b4", ecolor="#888", capsize=3)
    ax.axvline(0, color="k", lw=1, ls="--")
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{p} (n={n})" for p, n in zip(fp["patient"], fp["n_rec"])], fontsize=8)
    ax.set_xlabel("LORO AUC gap  (FFT − SCTN, PCA-40)   ← SCTN better | FFT better →")
    ax.set_title("Per-patient FFT vs SCTN (resonator+gamma), capacity-matched")
    fig.tight_layout()
    png = OUT_DIR / "multipatient_gap_forest.png"
    fig.savefig(png, dpi=170); plt.close(fig)

    print(f"\nSaved:\n  {csv}\n  {png}")


if __name__ == "__main__":
    main()
