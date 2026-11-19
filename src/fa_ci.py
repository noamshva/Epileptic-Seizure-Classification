"""95% intervals for the final-test false-alarm rate and seizure sensitivity.

Poisson / Clopper-Pearson treat alarms and seizures as independent. The patient
bootstrap resamples patients, so it also covers alarms clustering in a few patients.

Usage: python src/fa_ci.py svm_final fft_final snn_final_s0 ...   (reads false_alarms_<name>_k5.csv)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, chi2

STATS = Path(__file__).resolve().parent.parent / "analysis" / "stats"
B = 10000


def poisson_ci(k, t):
    lo = chi2.ppf(0.025, 2 * k) / 2 if k else 0.0
    return lo / t, chi2.ppf(0.975, 2 * k + 2) / 2 / t


def cp_ci(k, n):
    return (beta.ppf(0.025, k, n - k + 1) if k else 0.0,
            beta.ppf(0.975, k + 1, n - k) if k < n else 1.0)


def main():
    names = sys.argv[1:]
    d = {n: pd.read_csv(STATS / f"false_alarms_{n}_k5.csv").set_index("patient") for n in names}
    pats = d[names[0]].index
    idx = np.random.default_rng(0).integers(0, len(pats), (B, len(pats)))
    rows, boot = [], {}
    for n, t in d.items():
        t = t.loc[pats]
        fa, h = t.false_alarms.to_numpy(), t.free_hours.to_numpy()
        sz, det = t.seizures.to_numpy(), t.seizures_detected.to_numpy()
        boot[n] = fa[idx].sum(1) / h[idx].sum(1)
        bs = det[idx].sum(1) / sz[idx].sum(1)
        p = poisson_ci(fa.sum(), h.sum())
        c = cp_ci(det.sum(), sz.sum())
        rows.append({"model": n, "false_alarms": int(fa.sum()), "hours": round(h.sum(), 2),
                     "fa_per_h": fa.sum() / h.sum(),
                     "fa_poisson_lo": p[0], "fa_poisson_hi": p[1],
                     "fa_boot_lo": np.percentile(boot[n], 2.5), "fa_boot_hi": np.percentile(boot[n], 97.5),
                     "detected": int(det.sum()), "seizures": int(sz.sum()), "sens": det.sum() / sz.sum(),
                     "sens_cp_lo": c[0], "sens_cp_hi": c[1],
                     "sens_boot_lo": np.percentile(bs, 2.5), "sens_boot_hi": np.percentile(bs, 97.5)})
    df = pd.DataFrame(rows)
    df.to_csv(STATS / "fa_ci_final.csv", index=False)
    print(df.round(3).to_string(index=False))
    for a in names:
        for b in names:
            if a < b:
                diff = boot[a] - boot[b]
                print(f"{a} - {b}: {diff.mean():+.2f}/h, 95% [{np.percentile(diff, 2.5):+.2f}, "
                      f"{np.percentile(diff, 97.5):+.2f}], P(a<b) {np.mean(diff < 0):.3f}")


if __name__ == "__main__":
    main()
