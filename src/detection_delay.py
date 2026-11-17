"""Detection delay from the per-window scores saved by eval_false_alarms.py.

Delay = end of the window that completes the first alarm (ALARM_RUN consecutive
flagged windows, default threshold) minus the annotated seizure onset.

Usage:
    ALARM_RUN=5 python -u src/detection_delay.py svm_aug snn_aug fft_aug
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))
import features as F                                                    # noqa: E402
from eval_false_alarms import _OUT, _segments                           # noqa: E402

K = int(os.environ.get("ALARM_RUN", 3))


def delays(name: str) -> pd.DataFrame:
    thr = 0.5 if name.startswith("snn") else 0.0
    rows = []
    for f in sorted((_OUT / f"fa_scores_{name}").glob("chb*.npz")):
        d = np.load(f)
        onsets = F._s2s._load_seizures_for_subjects([f.stem])
        sz = np.flatnonzero(d["y"] == 1)
        g, st, flag = d["g"][sz], d["st"][sz], d["score"][sz] > thr
        seg, order = _segments(g, st)
        for s in np.unique(seg):
            i = order[seg[order] == s]
            t0 = st[i[0]]
            cand = [a for a, b in onsets.get(Path(str(g[i[0]])).stem, []) if a <= t0 + F.WINDOW_SECONDS]
            onset = max(cand) if cand else t0
            run = 0; alarm = np.nan
            for j in i:
                run = run + 1 if flag[j] else 0
                if run >= K:
                    alarm = st[j] + F.WINDOW_SECONDS; break
            rows.append({"model": name, "patient": f.stem, "recording": str(g[i[0]]), "onset_s": onset,
                         "duration_s": st[i[-1]] + F.WINDOW_SECONDS - onset, "delay_s": alarm - onset})
    return pd.DataFrame(rows)


def main():
    names = [a for a in sys.argv[1:] if not a.startswith("-")]
    df = pd.concat([delays(n) for n in names])
    out = _OUT / f"detection_delay_k{K}.csv"
    df.to_csv(out, index=False)
    for n, d in df.groupby("model"):
        ok = d.delay_s.dropna()
        print(f"{n}: detected {len(ok)}/{len(d)} | delay median {ok.median():.1f} s, mean {ok.mean():.1f} s,"
              f" 90th pct {ok.quantile(0.9):.1f} s | within 10 s: {(ok <= 10).sum()}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
