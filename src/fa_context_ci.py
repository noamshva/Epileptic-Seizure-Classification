"""False-alarm rate by context, with 95% patient-bootstrap intervals.

Contexts: final test (seizure-free recordings with two other minutes in training),
unseen seizure-free recordings (eval_free_unseen.py), and from eval_peri.py the
seizure recordings at least 5 min from a seizure (bg) and the zones around seizures.

Also prints each zone against the count expected from the same patients' final
and unseen rates (patient mix matched, Wilcoxon on per-patient rates), and how
many seizures have an alarm in the minute before onset and after the end.

Writes analysis/stats/fa_context_ci_k<K>.csv.

Usage: ALARM_RUN=5 python src/fa_context_ci.py svm fft snn_s0
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F                                                    # noqa: E402
import eval_false_alarms as E                                           # noqa: E402
from eval_peri import _alarm_idx, WIN_PER_HOUR, K                       # noqa: E402

STATS = E._OUT
B = 10000
PATS = [f"chb{n:02d}" for n in range(1, 25)]
ZONES = {"bg": lambda r, bg: bg,
         "pre_1to5": lambda r, bg: ~bg & (r <= -60),
         "pre_0to1": lambda r, bg: ~bg & (r < 0) & (r > -60),
         "post_0to1": lambda r, bg: ~bg & (r > 0) & (r < 60),
         "post_1to5": lambda r, bg: ~bg & (r >= 60)}
ORDER = ["final", "unseen_free", "bg", "pre_1to5", "pre_0to1", "post_0to1", "post_1to5"]


def _final_name(m):
    return m.replace("_s", "_final_s") if "_s" in m else f"{m}_final"


def load(m):
    """({context: (alarms[24], hours[24])}, seizure counts)"""
    out = {}
    for ctx, f in (("final", f"false_alarms_{_final_name(m)}_k{K}.csv"), ("unseen_free", f"free_unseen_{m}_k{K}.csv")):
        t = pd.read_csv(STATS / f).set_index("patient").loc[PATS]
        out[ctx] = (t.false_alarms.to_numpy(float), t.free_hours.to_numpy(float))
    z = {k: ([], []) for k in ZONES}
    sz = {"seizures": 0, "alarm_before": 0, "alarm_after": 0}
    for p in PATS:
        d = np.load(STATS / f"fa_scores_peri_{m}" / f"{p}.npz")
        g, st, near = d["g"], d["st"], ~d["bg"]
        seg, order = E._segments(g, st)
        alarm = np.zeros(len(g), bool); alarm[_alarm_idx(d["pred"] == 1, seg, order)] = True
        for k, f in ZONES.items():
            mk = f(d["rel"], d["bg"])
            z[k][0].append(alarm[mk].sum()); z[k][1].append(mk.sum() / WIN_PER_HOUR)
        iv = F._s2s._load_seizures_for_subjects([p])
        for trial in set(g.tolist()):
            a = alarm & near & (g == trial)
            for s, e in iv.get(trial, []):
                end = st + F.WINDOW_SECONDS
                sz["seizures"] += 1
                sz["alarm_before"] += bool((a & (end <= s) & (end > s - 60)).any())
                sz["alarm_after"] += bool((a & (st >= e) & (st < e + 60)).any())
    out.update({k: (np.array(a, float), np.array(h, float)) for k, (a, h) in z.items()})
    return out, sz


def main():
    models = sys.argv[1:] or ["svm", "fft"]
    idx = np.random.default_rng(0).integers(0, len(PATS), (B, len(PATS)))
    rows, boot, data, counts = [], {}, {}, {}
    for m in models:
        data[m], counts[m] = load(m)
        for ctx, (a, h) in data[m].items():
            b = boot[m, ctx] = a[idx].sum(1) / h[idx].sum(1)
            rows.append({"model": m, "context": ctx, "hours": h.sum(), "alarms": int(a.sum()),
                         "per_h": a.sum() / h.sum(), "lo": np.percentile(b, 2.5), "hi": np.percentile(b, 97.5),
                         "patients_with_alarms": int((a > 0).sum()), "top3_share": np.sort(a)[-3:].sum() / max(a.sum(), 1)})
    df = pd.DataFrame(rows)
    df.to_csv(STATS / f"fa_context_ci_k{K}.csv", index=False)
    print(df.round(2).to_string(index=False))

    def diff(x, y, label):
        d = boot[x] - boot[y]
        print(f"{label}: {d.mean():+.2f}/h, 95% [{np.percentile(d, 2.5):+.2f}, {np.percentile(d, 97.5):+.2f}]")

    for m in models:
        for a, b in zip(ORDER[1:], ORDER[:-1]):
            if (a, b) != ("post_1to5", "post_0to1") and (a, b) != ("post_0to1", "pre_0to1"):
                diff((m, a), (m, b), f"{m}: {a} - {b}")
        diff((m, "bg"), (m, "final"), f"{m}: bg - final")
        diff((m, "post_1to5"), (m, "bg"), f"{m}: post_1to5 - bg")
    for ctx in ORDER:
        for i, a in enumerate(models):
            for b in models[i + 1:]:
                diff((a, ctx), (b, ctx), f"{ctx}: {a} - {b}")

    print("Patient mix matched:")
    for m in models:
        for ref in ORDER[:2]:
            r = data[m][ref][0] / data[m][ref][1]
            for z in ORDER[2:]:
                a, h = data[m][z]
                rz = np.divide(a, h, out=np.zeros_like(a), where=h > 0)
                print(f"  {m}: {z} {int(a.sum())} alarms, {(r * h).sum():.1f} expected at the {ref} rates, "
                      f"higher in {int((rz > r).sum())} and lower in {int((rz < r).sum())} patients, "
                      f"p {wilcoxon(rz, r).pvalue:.3g}")
    for m in models:
        c = counts[m]
        print(f"{m}: {c['seizures']} seizures, alarm in the minute before onset {c['alarm_before']}, "
              f"in the minute after the end {c['alarm_after']}")


if __name__ == "__main__":
    main()
