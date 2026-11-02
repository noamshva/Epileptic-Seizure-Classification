"""Reproducible FFT-vs-SCTN feature comparison (regenerates the report's Table 1).

Per patient, leave-one-recording-out (LORO) AUC for the grid
    {SCTN-band, FFT-band-power, FFT-fine(PCA->40)} x {SVM, RF}
using the SAME SVM/RF pipeline as 3.run_classify_chbmit.py. The fine PSD baseline
is reduced to 40 PCA components (fit per fold, leakage-free) so all three feature
sets are compared at matched capacity (40 dims).

Outputs (under PIPELINE_DATA_ROOT or the repo, in analysis/stats/):
    fft_vs_sctn_per_patient.csv  -- one row per patient, all 6 AUCs + counts
    fft_vs_sctn_summary.json     -- means / medians / best-on / Wilcoxon
                                    (all patients and the n_rec>=5 subset)

Usage:
    python -u src/compare_fft_vs_sctn.py               # all 24 patients
    python -u src/compare_fft_vs_sctn.py chb01 chb03   # a subset
"""
from __future__ import annotations

import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))
import features as F                                                    # noqa: E402

SEED = 0
PCA_DIM = 40
_OUT = Path(os.environ.get("PIPELINE_DATA_ROOT", str(_SRC.parent))) / "analysis" / "stats"
_OUT.mkdir(parents=True, exist_ok=True)


def _pipe(name: str, pca: bool = False):
    if name == "svm":
        clf = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
                  probability=True, random_state=SEED)
    else:
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     random_state=SEED, n_jobs=-1)
    steps = [StandardScaler()]
    if pca:
        steps.append(PCA(n_components=PCA_DIM, random_state=SEED))
    steps.append(clf)
    return make_pipeline(*steps)


def _loro_auc(name: str, X, y, g, pca: bool = False) -> float:
    """Pooled leave-one-recording-out AUC (matches 3.run_classify_chbmit.py)."""
    if len(set(g.tolist())) < 2 or len(set(y.tolist())) < 2:
        return float("nan")
    ps = np.full(len(y), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, groups=g):
        if len(set(y[tr].tolist())) < 2:
            continue
        m = _pipe(name, pca).fit(X[tr], y[tr])
        ps[te] = m.predict_proba(X[te])[:, 1]
    ok = ~np.isnan(ps)
    if ok.sum() == 0 or len(set(y[ok].tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y[ok], ps[ok]))


def _cell(rec, prefix, loader, subj, pca=False):
    """Fill rec[f'{prefix}_svm'] and rec[f'{prefix}_rf'] from one feature load."""
    try:
        X, y, g, _ = loader([subj])
    except Exception as e:
        print(f"  {subj}: {prefix} load skip ({e})", flush=True)
        return
    if prefix == "sctn":
        rec["n_rec"] = len(set(g.tolist()))
        rec["n_win"] = int(len(y))
        rec["seizure"] = int(y.sum())
    rec[f"{prefix}_svm"] = _loro_auc("svm", X, y, g, pca)
    rec[f"{prefix}_rf"] = _loro_auc("rf", X, y, g, pca)


def _summ(df, a, b, min_rec=None):
    d = df.dropna(subset=[a, b])
    if min_rec is not None:
        d = d[d["n_rec"] >= min_rec]
    av, bv = d[a].to_numpy(), d[b].to_numpy()
    out = {"n": int(len(d)),
           "mean_sctn": float(np.mean(av)), "mean_fft": float(np.mean(bv)),
           "median_sctn": float(np.median(av)), "median_fft": float(np.median(bv)),
           "sctn_best": int((av > bv).sum()), "fft_best": int((bv > av).sum()),
           "median_delta_fft_minus_sctn": float(np.median(bv - av))}
    try:
        out["wilcoxon_p"] = (float(wilcoxon(av, bv).pvalue)
                             if len(d) >= 2 and np.any(av != bv) else float("nan"))
    except Exception:
        out["wilcoxon_p"] = float("nan")
    return out


def main():
    patients = [a for a in sys.argv[1:] if not a.startswith("-")] \
        or [f"chb{n:02d}" for n in range(1, 25)]
    print(f"Comparison over {len(patients)} patients: {patients}", flush=True)

    rows = []
    for subj in patients:
        rec = {"patient": subj}
        _cell(rec, "sctn", F.build_dataset_groups, subj)
        _cell(rec, "fftband", F.build_fft_bandpower, subj)
        _cell(rec, "fftfine", F.build_fft_fine, subj, pca=True)
        rows.append(rec)
        got = "  ".join(f"{k}={rec[k]:.3f}" for k in rec
                        if (k.endswith("_svm") or k.endswith("_rf"))
                        and isinstance(rec[k], float) and not np.isnan(rec[k]))
        print(f"{subj} (rec={rec.get('n_rec','?')}): {got}", flush=True)

    df = pd.DataFrame(rows)
    csv = _OUT / "fft_vs_sctn_per_patient.csv"
    df.to_csv(csv, index=False)

    summary = {}
    for clf in ("svm", "rf"):
        summary[clf] = {
            "sctn_vs_fftband_all": _summ(df, f"sctn_{clf}", f"fftband_{clf}"),
            "sctn_vs_fftband_ge5rec": _summ(df, f"sctn_{clf}", f"fftband_{clf}", 5),
            "sctn_vs_fftfine_all": _summ(df, f"sctn_{clf}", f"fftfine_{clf}"),
            "sctn_vs_fftfine_ge5rec": _summ(df, f"sctn_{clf}", f"fftfine_{clf}", 5),
        }
    js = _OUT / "fft_vs_sctn_summary.json"
    js.write_text(json.dumps(summary, indent=2))

    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nSaved: {csv}\n       {js}", flush=True)


if __name__ == "__main__":
    main()
