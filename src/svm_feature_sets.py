"""SVM LORO AUC per patient for the SCTN feature sets, with and without Gamma.

    band 40-D, resonator 280-D, band+Gamma 50-D, resonator+Gamma 350-D, Gamma only 10-D,
    FFT band power 40-D and with Gamma 50-D.

Needs: cache_features.py, cache_features.py --gamma, cache_features.py --fft.
Writes analysis/stats/svm_feature_sets.csv.

Usage: python -u src/svm_feature_sets.py [chb01 ...]
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))
import features as F                                                    # noqa: E402
import compare_fft_vs_sctn as C                                         # noqa: E402

PAIRS = (("band_gamma", "band"), ("res_gamma", "res"), ("res", "band"),
         ("fft_gamma", "fft"), ("band", "fft"), ("band_gamma", "fft_gamma"))


def run(subj):
    d = np.load(F.cache_path(subj)); gm = np.load(F.GAMMA_ROOT / f"{subj}.npz")
    y, g = d["y"].astype(int), d["groups"]
    assert (gm["y"] == d["y"]).all() and (gm["groups"] == g).all() and (gm["start_s"] == d["start_s"]).all()
    band, res = d["X_band"].astype(float), d["X_res"].astype(float)
    sets = {"band": band, "res": res,
            "band_gamma": np.hstack([band, gm["X_band"]]), "res_gamma": np.hstack([res, gm["X_res"]]),
            "gamma": gm["X_band"].astype(float)}
    row = {"patient": subj, **{k: C._loro_auc("svm", X, y, g) for k, X in sets.items()}}
    f = np.load(F.FFT_ROOT / f"{subj}.npz")
    row["fft"] = C._loro_auc("svm", f["X_band"], f["y"].astype(int), f["groups"])
    Xg, yg, gg, _ = F.build_fft_bandpower([subj], include_gamma=True)
    row["fft_gamma"] = C._loro_auc("svm", Xg, yg, gg)
    print(f"  {subj}: " + "  ".join(f"{k}={v:.3f}" for k, v in row.items() if k != "patient"), flush=True)
    return row


def main():
    patients = [a for a in sys.argv[1:] if a.startswith("chb")] or [f"chb{n:02d}" for n in range(1, 25)]
    with ProcessPoolExecutor(min(int(os.environ.get("WORKERS", 3)), len(patients))) as ex:
        df = pd.DataFrame(list(ex.map(run, patients)))
    out = C._OUT / "svm_feature_sets.csv"
    df.to_csv(out, index=False)
    print(df.drop(columns="patient").mean().round(4).to_string())
    for a, b in PAIRS:
        d = df[a] - df[b]
        p = wilcoxon(df[a], df[b]).pvalue if np.any(d != 0) else float("nan")
        print(f"{a} vs {b}: higher on {int((d > 0).sum())}/{len(df)}, mean diff {d.mean():+.4f}, p {p:.4g}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
