"""Sensitivity vs false alarms/hour from the scores saved by eval_false_alarms.py.

Per patient the threshold is the Q-quantile of the LORO scores of that patient's
non-seizure windows (needs no seizure-free data). Alarm = K consecutive windows
above threshold. False alarms are counted on the tuning half of the seizure-free
recordings by default (HALF=tune | held | all).

Usage:
    python -u src/fa_tradeoff.py svm fft snn
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))
from eval_false_alarms import _runs, _segments                          # noqa: E402

HALF = os.environ.get("HALF", "tune")
QS = [0.90, 0.95, 0.98, 0.99, 0.995, 0.999]
KS = [3, 5, 10]
_OUT = Path(os.environ.get("PIPELINE_DATA_ROOT", str(_SRC.parent))) / "analysis" / "stats"


def main():
    models = [a for a in sys.argv[1:] if not a.startswith("-")] or ["svm"]
    rows = []
    for model in models:
        files = sorted((_OUT / f"fa_scores_{model}").glob("chb*.npz"))
        for q in QS:
            for k in KS:
                det = tot = alarms = 0; hours = 0.0
                for f in files:
                    d = np.load(f)
                    y, g, st, sc = d["y"], d["g"], d["st"], d["score"]
                    thr = np.quantile(sc[(y == 0) & ~np.isnan(sc)], q)
                    sz = np.flatnonzero(y == 1)
                    seg, order = _segments(g[sz], st[sz])
                    det += len({s for s, n in _runs(sc[sz] > thr, seg, order) if n >= k})
                    tot += int(seg.max())
                    m = d["tune"] if HALF == "tune" else ~d["tune"] if HALF == "held" else np.ones(len(d["gf"]), bool)
                    gf, sf, scf = d["gf"][m], d["sf"][m], d["scoref"][m]
                    segf, orderf = _segments(gf, sf)
                    alarms += sum(n >= k for _, n in _runs(scf > thr, segf, orderf))
                    hours += len(gf) / 59 / 60
                rows.append({"model": model, "q": q, "k": k, "seizures": tot, "detected": det,
                             "sensitivity": det / tot, "hours": hours, "false_alarms": alarms,
                             "fa_per_hour": alarms / hours})
    df = pd.DataFrame(rows)
    out = _OUT / f"fa_tradeoff_{HALF}.csv"
    df.to_csv(out, index=False)
    for k in KS:
        print(f"\nalarm = {k} consecutive windows ({HALF} half): detected / FA per hour")
        t = df[df.k == k].assign(cell=lambda x: x.detected.astype(str) + " / " + x.fa_per_hour.map("{:.1f}".format))
        print(t.pivot(index="q", columns="model", values="cell").to_string())
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
