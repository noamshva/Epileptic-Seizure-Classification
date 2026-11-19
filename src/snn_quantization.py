"""LORO AUC of the SNN after rounding its trained weights to N bits (no retraining).

Same training and env knobs as 3c.run_classify_snn_chbmit.py (INPUT=features|bins).
Every parameter tensor (encoder gain and bias, both weight matrices and their
biases) is rounded to N-bit signed levels, scaled per tensor. Leak and threshold
are unchanged.

Writes analysis/stats/snn_quantization_<features|bins>.csv.

Usage:
    python -u src/snn_quantization.py
    INPUT=bins READOUT=membrane LR=0.002 CLIP=1 EPOCHS=60 python -u src/snn_quantization.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))
import features as F                                                    # noqa: E402

_spec = importlib.util.spec_from_file_location("snn3c", _SRC / "3c.run_classify_snn_chbmit.py")
snn = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(snn)

BITS = (8, 6, 5, 4, 3, 2)
_OUT = Path(os.environ.get("PIPELINE_DATA_ROOT", str(_SRC.parent))) / "analysis" / "stats"


def run(subj):
    X, y, g = F.load_bins(subj) if snn.INPUT == "bins" else F.load_cached(subj, snn.FEATURE_LEVEL)
    probs = {k: np.full(len(y), np.nan) for k in ("float",) + BITS}
    for tr, te in LeaveOneGroupOut().split(X, y, groups=g):
        if len(set(y[tr].tolist())) < 2:
            continue
        p, _, _, _, q = snn.fit_predict(X, y, tr, te, quant_bits=BITS)
        probs["float"][te] = p
        for b in BITS:
            probs[b][te] = q[b]
    ok = ~np.isnan(probs["float"])
    return {"patient": subj, **{k if k == "float" else f"bits{k}": roc_auc_score(y[ok], v[ok])
                                for k, v in probs.items()}}


def main():
    patients = [a for a in sys.argv[1:] if a.startswith("chb")] or [f"chb{n:02d}" for n in range(1, 25)]
    print(f"SNN quantization, input={snn.INPUT}, seed={snn.SEED}, {len(patients)} patients", flush=True)
    rows = []
    with ProcessPoolExecutor(min(int(os.environ.get("WORKERS", 4)), len(patients))) as ex:
        for fut in as_completed([ex.submit(run, s) for s in patients]):
            r = fut.result(); rows.append(r)
            print(f"  {r['patient']}: " + "  ".join(f"{k}={v:.3f}" for k, v in r.items() if k != "patient"),
                  flush=True)
    df = pd.DataFrame(rows).sort_values("patient")
    out = _OUT / f"snn_quantization_{snn.INPUT}.csv"
    df.to_csv(out, index=False)
    print(df.drop(columns="patient").mean().round(4).to_string())
    for b in BITS:
        d = df[f"bits{b}"] - df["float"]
        p = wilcoxon(df[f"bits{b}"], df["float"]).pvalue if np.any(d != 0) else float("nan")
        print(f"{b} bits vs float: mean diff {d.mean():+.4f}, worst patient {d.min():+.4f}, p {p:.4g}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
