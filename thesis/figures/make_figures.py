"""Thesis result figures from analysis/stats. Writes PNG (300 dpi) and PDF next to this file.

Usage: python thesis/figures/make_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
STATS = OUT.parents[1] / "analysis" / "stats"

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# model -> (label, colour, marker); fixed per model in every figure
M = {"svm": ("SVM, SCTN", "#2a78d6", "o"),
     "fft": ("SVM, FFT", "#eb6834", "s"),
     "snn": ("SNN, SCTN", "#1baf7a", "^")}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.bbox": "tight", "pdf.fonttype": 42})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    plt.close(fig)


def fig_auc():
    d = pd.read_csv(STATS / "fft_vs_sctn_per_patient_fs100.csv").sort_values("sctn_svm", ascending=False)
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.vlines(x, d.fftband_svm, d.sctn_svm, color=AXIS, lw=1.2, zorder=1)
    for k, col in (("fft", "fftband_svm"), ("svm", "sctn_svm")):
        lab, c, mk = M[k]
        ax.scatter(x, d[col], s=30, color=c, marker=mk, edgecolor="white", lw=0.8, zorder=3,
                   label=f"{lab} (mean {d[col].mean():.3f})")
    ax.set_xticks(x, d.patient.str.replace("chb", ""))
    ax.set_xlabel("Patient (chb), ordered by SCTN AUC"); ax.set_ylabel("AUC, leave one recording out")
    ax.set_ylim(0.70, 1.005); ax.set_xlim(-0.6, len(d) - 0.4); ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left")
    save(fig, "fig_auc_per_patient")


def fig_fa_patient():
    t = {k: pd.read_csv(STATS / f"false_alarms_{f}_k5.csv").set_index("patient")
         for k, f in (("svm", "svm_final"), ("snn", "snn_final_s0"), ("fft", "fft_final"))}
    order = t["svm"].index[::-1]
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 4.6), sharex=True, sharey=True)
    for ax, k in zip(axes, ("svm", "snn", "fft")):
        lab, c, _ = M[k]
        d = t[k].loc[order]
        ax.barh(y, d.fa_per_hour, height=0.62, color=c)
        ax.set_title(f"{lab}{' (seed 0)' if k == 'snn' else ''}\n{d.false_alarms.sum() / d.free_hours.sum():.2f} per hour overall",
                     fontsize=9, loc="left", color=INK)
        ax.grid(axis="y", visible=False); ax.set_ylim(-0.7, len(order) - 0.3)
    axes[0].set_yticks(y, order)
    axes[1].set_xlabel("False alarms per hour (final test, about 1.5 h per patient)")
    save(fig, "fig_false_alarms_per_patient")


def fig_peri():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True, gridspec_kw={"wspace": 0.06})
    for k, f in (("svm", "svm"), ("snn", "snn_s0"), ("fft", "fft")):
        lab, c, mk = M[k]
        d = pd.read_csv(STATS / f"peri_profile_{f}_k5.csv")
        for ax, zone, sign in zip(axes, ("pre", "post"), (-1, 1)):
            z = d[(d.zone == zone) & (d.from_s < 300)].sort_values("from_s")
            ax.plot(sign * (z.from_s / 60 + 0.5), z.alarms_per_h, color=c, marker=mk, ms=5, lw=1.6,
                    markeredgecolor="white", markeredgewidth=0.8,
                    label=f"{lab}{' (seed 0)' if k == 'snn' else ''}")
    for ax, title, ticks in zip(axes, ("Before onset", "After the end"), (-np.arange(5, -1, -1), np.arange(0, 6))):
        ax.axhspan(0.80, 1.32, color=GRID, zorder=0)
        ax.set_title(title, fontsize=9, loc="left", color=INK); ax.set_xticks(ticks, [str(abs(v)) for v in ticks])
        ax.set_ylim(0, 40)
    axes[0].set_xlim(-5.2, 0); axes[1].set_xlim(0, 5.2)
    axes[0].set_xlabel("Minutes before seizure onset"); axes[1].set_xlabel("Minutes after seizure end")
    axes[0].set_ylabel("Alarms per hour")
    axes[1].spines["left"].set_visible(False); axes[1].tick_params(axis="y", length=0)
    axes[1].text(0.08, 1.9, "final test rate (0.80 to 1.32)", fontsize=7.5, color=INK2, va="bottom")
    axes[0].legend(loc="upper left")
    save(fig, "fig_alarms_near_seizures")


def fig_quant():
    cols = ["float", "bits8", "bits6", "bits5", "bits4", "bits3", "bits2"]
    names = ["Full", "8", "6", "5", "4", "3", "2"]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for f, lab, c, mk in (("features", "SNN, 40 SCTN features", "#1baf7a", "^"),
                          ("bins", "SNN, binned spike trains", "#6250d6", "D")):
        m = pd.read_csv(STATS / f"snn_quantization_{f}.csv")[cols].mean()
        ax.plot(range(len(cols)), m, color=c, marker=mk, ms=5, lw=1.6, markeredgecolor="white",
                markeredgewidth=0.8, label=lab)
    ax.set_xticks(range(len(cols)), names); ax.set_xlabel("Weight precision (bits)")
    ax.set_ylabel("Mean AUC, 24 patients"); ax.set_ylim(0.65, 1.0); ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left")
    save(fig, "fig_weight_precision")


if __name__ == "__main__":
    fig_auc(); fig_fa_patient(); fig_peri(); fig_quant()
    print("wrote", sorted(p.name for p in OUT.glob("fig_*.png")))
