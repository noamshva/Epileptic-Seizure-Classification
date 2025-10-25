#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CHB-MIT spike-based analysis (PNG only, Matplotlib only).

Reads windows produced by your signal2spikes -> create_datasets() pipeline:
  ../datasets/EEG_Data/EEG_dataset_output/<subject>/<trial>/<label>/<timestamp>.npz

NOTE: Those ".npz" are actually pickled dicts:
  { "<channel>-<band>": torch.Tensor [n_resonators, time] }

For each sample, we generate:
  1) Spikeogram grids (per band: one panel per channel; each panel is [resonator × time-bins])
  2) Band spike-rate heatmaps (channels × bands for this sample)
  3) Summary heatmap of mean spike-rate (channels × bands)

Output directory structure mirrors the dataset tree:
  ./output_analysis/<subject>/<trial>/<label>/<timestamp>_*.png
"""

import os
import sys
import math
import pickle
from pathlib import Path
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


# ----------------------------
# Config (edit as you like)
# ----------------------------
DATA_ROOT = Path("../dataset/spikes_windowed")
OUT_ROOT  = Path("../analysis/figures")

# How we bin along time for spikeograms
# (we don't know sample_fs explicitly; we just split the saved time axis into this many bins)
TIME_BINS_PER_WINDOW = 25          # e.g., 25 bins across the window
MIN_RESONATORS_TO_PLOT = 2         # skip panels with fewer than this
FIG_DPI = 140

# If you want to limit the number of files for a dry run, set MAX_FILES > 0
MAX_FILES = 1   # 0 = no limit


def safe_makedirs(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_sample_pickle(npz_path: Path) -> Dict[str, torch.Tensor]:
    """Load your saved 'npz' (actually pickled) dict."""
    with open(npz_path, "rb") as f:
        d = pickle.load(f)
    # d: key "<channel>-<band>" -> torch.Tensor [n_resonators, time]
    # Ensure tensors are on CPU and in contiguous memory
    out = {}
    for k, t in d.items():
        if isinstance(t, torch.Tensor):
            out[k] = t.detach().cpu().contiguous()
        else:
            # just in case someone stored numpy
            out[k] = torch.as_tensor(t).detach().cpu().contiguous()
    return out


def split_key(key: str) -> Tuple[str, str]:
    """
    Keys look like "<channel>-<band>" and channels can have hyphens (e.g., 'FP2-F4').
    We split from the RIGHT to get band reliably.
    """
    parts = key.split("-")
    band = parts[-1]
    channel = "-".join(parts[:-1])
    return channel, band


def bin_spikes_time(spike_tensor: torch.Tensor, n_bins: int) -> np.ndarray:
    """
    spike_tensor: [n_resonators, time] with 0/1 spikes.
    Returns: [n_resonators, n_bins] with counts per bin.
    """
    n_res, T = spike_tensor.shape
    if T < n_bins:
        # pad with zeros on time axis to reach at least n_bins
        pad = n_bins - T
        spike_tensor = torch.nn.functional.pad(spike_tensor, (0, pad), value=0)
        T = n_bins

    bin_len = T // n_bins
    if bin_len == 0:
        # degenerate case: squish everything into a single bin
        return spike_tensor.sum(dim=1, keepdim=True).numpy()

    T_used = bin_len * n_bins
    x = spike_tensor[:, :T_used]  # drop remainder
    x = x.view(n_res, n_bins, bin_len).sum(dim=2)  # [n_res, n_bins]
    return x.numpy()


def aggregate_mean_rate(spike_tensor: torch.Tensor) -> float:
    """
    Mean spike rate proxy for a window: average spikes per time-step across resonators.
    """
    n_res, T = spike_tensor.shape
    if T == 0 or n_res == 0:
        return 0.0
    return float(spike_tensor.sum().item()) / (n_res * T)


def collect_channels_and_bands(sample: Dict[str, torch.Tensor]) -> Tuple[List[str], List[str]]:
    chans = set()
    bands = set()
    for k in sample.keys():
        ch, b = split_key(k)
        chans.add(ch)
        bands.add(b)
    return sorted(chans), sorted(bands)


def plot_spikeogram_grid(sample: Dict[str, torch.Tensor],
                         subject: str, trial: str, label: str, stem: str,
                         out_dir: Path):
    """
    For each band, create a grid of (channels) with panels showing [resonator × time-bins].
    """
    channels, bands = collect_channels_and_bands(sample)

    for band in bands:
        # gather panels for this band
        panels: List[Tuple[str, np.ndarray]] = []   # (channel, mat[resonator, bins])
        for ch in channels:
            key = f"{ch}-{band}"
            if key not in sample:
                continue
            X = sample[key]  # [R, T]
            if X.ndim != 2:
                continue
            if X.shape[0] < MIN_RESONATORS_TO_PLOT:
                continue
            M = bin_spikes_time(X, TIME_BINS_PER_WINDOW)  # [R, bins]
            panels.append((ch, M))

        if not panels:
            continue

        # Figure size heuristics
        n = len(panels)
        cols = min(4, n)
        rows = int(math.ceil(n / cols))
        fig_w = 4 * cols
        fig_h = 2.6 * rows
        fig, axs = plt.subplots(rows, cols, figsize=(fig_w, fig_h), dpi=FIG_DPI)
        axs = np.atleast_2d(axs)

        vmin = min(p[1].min() for p in panels)
        vmax = max(p[1].max() for p in panels)
        vmax = vmax if vmax > vmin else (vmin + 1.0)

        for i, (ch, mat) in enumerate(panels):
            r, c = divmod(i, cols)
            ax = axs[r, c]
            im = ax.imshow(mat, aspect="auto", origin="lower",
                           vmin=vmin, vmax=vmax, cmap="jet")
            ax.set_title(ch, fontsize=9)
            ax.set_ylabel("Resonator", fontsize=8)
            ax.set_xlabel("Time bins", fontsize=8)
            ax.tick_params(labelsize=7)

        # turn off unused axes
        for j in range(len(panels), rows * cols):
            r, c = divmod(j, cols)
            axs[r, c].axis("off")

        fig.suptitle(f"Spikeogram — subject={subject}, trial={trial}, label={label}, band={band}", fontsize=12)
        fig.tight_layout(rect=[0, 0, 0.92, 0.95])

        # colorbar on the right
        cax = fig.add_axes([0.94, 0.12, 0.02, 0.76])
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        cb = matplotlib.colorbar.ColorbarBase(cax, norm=norm, cmap="jet")
        cb.set_label("Spike count / bin", fontsize=8)

        safe_makedirs(out_dir)
        outfile = out_dir / f"{stem}_spikeogram_{band}.png"
        fig.savefig(outfile)
        plt.close(fig)


def plot_bandrate_heatmaps(sample: Dict[str, torch.Tensor],
                           subject: str, trial: str, label: str, stem: str,
                           out_dir: Path):
    """
    1) Heatmap Channels × Bands of mean spike-rate (one figure)
    2) Per-band heatmap of channels (one figure per band; optional)
    """
    channels, bands = collect_channels_and_bands(sample)
    if not channels or not bands:
        return

    # Build matrix: rows=channels, cols=bands
    mat = np.zeros((len(channels), len(bands)), dtype=float)
    for i, ch in enumerate(channels):
        for j, b in enumerate(bands):
            key = f"{ch}-{b}"
            if key in sample:
                mat[i, j] = aggregate_mean_rate(sample[key])

    # Summary heatmap
    fig_w = max(6, 0.35 * len(bands))
    fig_h = max(6, 0.35 * len(channels))
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=FIG_DPI)
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap="jet")
    ax.set_yticks(np.arange(len(channels)))
    ax.set_yticklabels(channels, fontsize=8)
    ax.set_xticks(np.arange(len(bands)))
    ax.set_xticklabels(bands, rotation=45, ha="right", fontsize=9)
    ax.set_title(f"Mean spike-rate — subject={subject}, trial={trial}, label={label}", fontsize=12)
    ax.set_xlabel("Band")
    ax.set_ylabel("Channel")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean spikes / (resonator×timestep)")

    safe_makedirs(out_dir)
    outfile = out_dir / f"{stem}_bandrate_summary.png"
    fig.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)

    # Optional: per-band channel heatmap (kept simple & compact)
    # Comment out this block if you want fewer files.
    for j, b in enumerate(bands):
        col = mat[:, j:j+1]  # (len(channels), 1)
        fig2, ax2 = plt.subplots(1, 1, figsize=(4, max(3, 0.35 * len(channels))), dpi=FIG_DPI)
        im2 = ax2.imshow(col, aspect="auto", origin="lower", cmap="jet")
        ax2.set_yticks(np.arange(len(channels)))
        ax2.set_yticklabels(channels, fontsize=8)
        ax2.set_xticks([0])
        ax2.set_xticklabels([b], fontsize=9)
        ax2.set_title(f"Mean spike-rate per channel — {b}", fontsize=11)
        fig2.colorbar(im2, ax=ax2)
        fig2.tight_layout()
        outfile2 = out_dir / f"{stem}_bandrate_{b}.png"
        fig2.savefig(outfile2)
        plt.close(fig2)

    # --- Save numeric summary for later analysis ---
    df = pd.DataFrame(mat, index=channels, columns=bands)
    csv_path = out_dir / f"{stem}_bandrate_summary.csv"
    df.to_csv(csv_path)


def walk_dataset(data_root: Path):
    """
    Yields tuples: (subject, trial, label, file_path)
    """
    if not data_root.exists():
        print(f"[ERROR] DATA_ROOT not found: {data_root.resolve()}", file=sys.stderr)
        return
    for subject_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        subject = subject_dir.name
        for trial_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
            trial = trial_dir.name
            for label in ["seizure", "non_seizure"]:
                label_dir = trial_dir / label
                if not label_dir.exists():
                    continue
                for fp in sorted(label_dir.rglob("*.npz")):
                    yield subject, trial, label, fp
                    print(f"[DEBUG] found file: {fp}")
                    sys.stdout.flush()



def main():
    print(f"[INFO] Scanning: {DATA_ROOT.resolve()}")
    processed = 0

    for subject, trial, label, fp in walk_dataset(DATA_ROOT):
        stem = fp.stem  # timestamp string
        out_dir = OUT_ROOT / subject / trial / label
        try:
            sample = load_sample_pickle(fp)
        except Exception as e:
            print(f"[WARN] Failed to load {fp}: {e}")
            continue

        # 1) Spikeogram grids per band (panels = channels)
        try:
            plot_spikeogram_grid(sample, subject, trial, label, stem, out_dir)
        except Exception as e:
            print(f"[WARN] spikeogram failed for {fp}: {e}")

        # 2) Band spike-rate heatmaps
        try:
            plot_bandrate_heatmaps(sample, subject, trial, label, stem, out_dir)
        except Exception as e:
            print(f"[WARN] bandrate heatmaps failed for {fp}: {e}")

        processed += 1
        if MAX_FILES and processed >= MAX_FILES:
            print(f"[INFO] Reached MAX_FILES={MAX_FILES}. Stopping.")
            break


    if processed == 0:
        print("[INFO] No samples processed. Check paths & data format.")
    else:
        print(f"[INFO] Done. Processed {processed} sample(s). Output under: {OUT_ROOT.resolve()}")

        # ==========================================================
        # Summary across all samples (mean spike-rate per band)
        # ==========================================================
        print("[INFO] Creating overall summary across all windows...")
        import pandas as pd

        all_csvs = list(OUT_ROOT.rglob("*_bandrate_summary.csv"))
        if not all_csvs:
            print("[WARN] No CSV summaries found. Skipping overall summary.")
            return

        band_stats = {}  # {label: [DataFrames]}
        for csv_path in all_csvs:
            label = "seizure" if "seizure" in str(csv_path).lower() else "non_seizure"
            try:
                df = pd.read_csv(csv_path, index_col=0)
                band_stats.setdefault(label, []).append(df.mean(axis=0))
            except Exception as e:
                print(f"[WARN] Failed to read {csv_path}: {e}")

        if not band_stats:
            print("[WARN] No valid summaries aggregated.")
            return

        # Compute averages
        summary_df = pd.DataFrame({
            label: pd.concat(vals, axis=1).mean(axis=1)
            for label, vals in band_stats.items()
        })

        # --- Save CSV summary ---
        summary_csv = OUT_ROOT / "summary_mean_bandrates.csv"
        summary_df.to_csv(summary_csv)
        print(f"[INFO] Saved summary CSV: {summary_csv}")

        # --- Plot summary ---
        fig, ax = plt.subplots(figsize=(6, 4), dpi=FIG_DPI)
        x = np.arange(len(summary_df.index))
        width = 0.35

        for i, label in enumerate(summary_df.columns):
            ax.bar(x + i * width, summary_df[label].values, width, label=label.capitalize())

        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(summary_df.index, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Mean Spike Rate")
        ax.set_title("Mean Spike Rate per Band (Seizure vs Non-Seizure)")
        ax.legend()
        fig.tight_layout()

        summary_fig = OUT_ROOT / "summary_mean_bandrates.png"
        fig.savefig(summary_fig)
        plt.close(fig)
        print(f"[INFO] Saved summary figure: {summary_fig}")



if __name__ == "__main__":
    main()
