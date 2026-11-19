"""False alarms near seizures: the guard-band minutes and the seizure-free part of
the seizure minutes. No model was trained or tuned on these windows.

Per patient, leave-one-recording-out with the frozen detector protocol: train on
the other seizure recordings plus all of <SPIKES_ROOT>_free, then score the
held-out recording's windows near its seizures. Alarm = ALARM_RUN consecutive
flagged windows, placed at the window that completes it.

Needs: SPIKES_ROOT=dataset/spikes_raw_peri python src/cache_features.py --peri [--fft]
Writes analysis/stats/peri_<model>_k<K>.csv (per patient),
peri_profile_<model>_k<K>.csv (by distance from the seizure) and per-window
scores in fa_scores_peri_<model>/.

Usage:
    ALARM_RUN=5 python -u src/eval_peri.py svm
    ALARM_RUN=5 python -u src/eval_peri.py fft
    ALARM_RUN=5 SEED=0 python -u src/eval_peri.py snn
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))
import features as F                                                    # noqa: E402
import eval_false_alarms as E                                           # noqa: E402

K = E.ALARM_RUN
EDGES = (0, 60, 120, 180, 240, 300, np.inf)        # seconds from the seizure
WIN_PER_HOUR = 59 * 60


def _rel(subj, g, st):
    """Signed seconds to the nearest seizure: negative = before onset, positive = after end."""
    seizures = F._s2s._load_seizures_for_subjects([subj])
    rel = np.empty(len(g))
    for i, (trial, a) in enumerate(zip(g.tolist(), st.tolist())):
        best = np.inf
        for s, e in seizures.get(trial, []):
            d = s - (a + F.WINDOW_SECONDS) if a < s else a - e
            if abs(d) < abs(best):
                best = (-1 if a < s else 1) * (max(d, 0.0) + 1e-6)
        rel[i] = best
    return rel


def _alarm_idx(flag, seg, order):
    """Index of the window that completes each alarm (one per run of >= K flagged windows)."""
    out = []; n = 0
    f, s = flag[order], seg[order]
    for i in range(len(f)):
        n = (n + 1 if n and s[i] == s[i - 1] else 1) if f[i] else 0
        if n == K:
            out.append(order[i])
    return np.asarray(out, int)


def run_patient(subj):
    X, y, g, _ = E._load(subj)
    Xf, yf, _, _ = E._load(subj, "_free")
    Xp, yp, gp, sp = E._load(subj, "_peri")
    Xa = np.concatenate([X, Xf, Xp]); ya = np.concatenate([y, yf, yp])
    extra = len(y) + np.arange(len(yf)); off = len(y) + len(yf)

    pred = np.zeros(len(yp), int); score = np.full(len(yp), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, groups=g):
        idx = np.flatnonzero(gp == g[te][0])
        if len(idx) and len(set(y[tr].tolist())) == 2:
            pred[idx], score[idx] = E._predict(Xa, ya, np.concatenate([tr, extra]), off + idx)

    ok = ~np.isnan(score)
    gp, sp, pred, score = gp[ok], sp[ok], pred[ok], score[ok]
    rel = _rel(subj, gp, sp)
    seg, order = E._segments(gp, sp)
    alarm = np.zeros(len(gp), bool); alarm[_alarm_idx(pred == 1, seg, order)] = True

    sdir = E._OUT / f"fa_scores_peri_{E.NAME}"; sdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(sdir / f"{subj}.npz", g=gp, st=sp, score=score, pred=pred, rel=rel)

    rows = []
    for zone, sign in (("pre", -1), ("post", 1)):
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            m = (np.sign(rel) == sign) & (np.abs(rel) >= lo) & (np.abs(rel) < hi)
            rows.append({"patient": subj, "zone": zone, "from_s": lo, "windows": int(m.sum()),
                         "flagged": int(pred[m].sum()), "alarms": int(alarm[m].sum())})
    return rows


def main():
    patients = [a for a in sys.argv[1:] if a.startswith("chb")] or [f"chb{n:02d}" for n in range(1, 25)]
    workers = min(int(os.environ.get("WORKERS", 6)), len(patients))
    print(f"{E.NAME}: windows near seizures, {len(patients)} patients, alarm = {K} consecutive windows",
          flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(run_patient, s) for s in patients]):
            r = fut.result(); rows += r
            n = sum(x["windows"] for x in r); a = sum(x["alarms"] for x in r)
            print(f"  {r[0]['patient']}: {n / WIN_PER_HOUR:.2f} h | alarms {a}", flush=True)
    df = pd.DataFrame(rows).sort_values(["patient", "zone", "from_s"])
    df["far"] = df.from_s >= 60

    def rate(d):
        h = d.windows.sum() / WIN_PER_HOUR
        return pd.Series({"hours": h, "win_fpr": d.flagged.sum() / max(d.windows.sum(), 1),
                          "alarms": d.alarms.sum(), "alarms_per_h": d.alarms.sum() / h if h else np.nan})

    prof = df.groupby(["zone", "from_s"]).apply(rate).reset_index()
    pat = pd.concat({"all": df.groupby("patient").apply(rate),
                     "far": df[df.far].groupby("patient").apply(rate)}, axis=1)
    pat.columns = [f"{b}_{a}" for a, b in pat.columns]
    tag = f"{E.NAME}_k{K}"
    pat.reset_index().to_csv(E._OUT / f"peri_{tag}.csv", index=False)
    prof.to_csv(E._OUT / f"peri_profile_{tag}.csv", index=False)

    print(prof.round(3).to_string(index=False))
    for name, d in (("all windows near seizures", df), ("more than 1 min from a seizure", df[df.far]),
                    ("before onset, more than 1 min", df[df.far & (df.zone == "pre")]),
                    ("after end, more than 1 min", df[df.far & (df.zone == "post")])):
        r = rate(d)
        print(f"{name}: {r.hours:.1f} h | window FPR {r.win_fpr:.4f} | alarms {int(r.alarms)}"
              f" = {r.alarms_per_h:.2f}/h")
    print(f"Saved: peri_{tag}.csv, peri_profile_{tag}.csv")


if __name__ == "__main__":
    main()
