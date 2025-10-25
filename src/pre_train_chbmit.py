import os
import pickle
from pathlib import Path
import torch
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt

import sys, types
if "torch._six" not in sys.modules:
    m = types.ModuleType("torch._six")
    m.container_abcs = ()
    m.string_classes = (str,)
    m.int_classes = (int,)
    sys.modules["torch._six"] = m
    
from utils import Topographies2SNN, Bands2TopographyLayer, Resonators2BandsLayer


# ---------------------------------------------------------------------
# 5) Load windowed samples from the new dataset location
#    The signal2spikes file saved pickles with .npz extension.
#    Directory shape:
#      spikes_windowed/<subject>/<trial>/<label>/<timestamp>.npz
# ---------------------------------------------------------------------
DATA_ROOT = Path("../dataset/spikes_windowed")
OUTPUT_STATS_DIR = Path("../analysis/pretrain_stats")

# ---------------------------------------------------------------------
# 1) Resonator set – only the ones you actually generated (<10 Hz, clk=15360)
#    If later you also generate the >=10 Hz set (clk=153600), restore the full list.
# ---------------------------------------------------------------------
full_resonator_array = torch.tensor([
    1.10, 1.30, 1.60, 1.90, 2.20, 2.50, 2.88, 3.05, 3.39,
    3.70, 4.12, 4.62, 5.09, 5.45, 5.87, 6.36, 6.80, 7.60, 8.60
], dtype=torch.float32)

# full_resonator_array = torch.tensor([
#     1.1, 1.3, 1.6, 1.9, 2.2, 2.5, 2.88, 3.05, 3.39, 3.7, 4.12, 4.62, 5.09, 5.45, 5.87, 6.36, 6.8, 7.6, 8.6,
#     10.5, 11.5, 12.8, 15.8, 16.6, 19.4, 22.0, 24.8, 28.4, 30.5, 34.7, 37.2, 40.2, 43.2, 47.7, 52.6, 57.2
# ], dtype=torch.float32)

bands = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 14),
    "Beta":  (14, 32),
    "Gamma": (32, 62),
}

# ---------------------------------------------------------------------
# 2) Channels – the CHB-MIT bipolar channels you processed
#    (Your example indices [4,16,13,5,12,9,10,14,22] map to:)
#       4:P7-O1, 16:P8-O2, 13:FP2-F8, 5:FP1-F3, 12:P4-O2,
#       9:FP2-F4, 10:F4-C4, 14:F8-T8, 22:FT10-T8
#    Order here must match the names used in your spikes/windows.
# ---------------------------------------------------------------------
channels = [
    "P7-O1",
    "P8-O2",
    "FP2-F8",
    "FP1-F3",
    "P4-O2",
    "FP2-F4",
    "F4-C4",
    "F8-T8",
    "FT10-T8",
]

# ---------------------------------------------------------------------
# 3) Simple topography – place 9 channels on a 3x3 grid (works fine)
# ---------------------------------------------------------------------
N = 9                     # grid width/height for topo (9x9 inputs)
g = np.linspace(0.15, 0.85, 3)  # three rows/cols inside the canvas
grid_xy = [(x, y) for y in g for x in g]   # 3x3

# Map channels -> (row, col) in an N x N image
# We’ll spread the 3x3 logical positions across an N x N canvas.
def to_int_pos(xy, N):
    x, y = xy
    return (int(round(y*(N-1))), int(round(x*(N-1))))  # (row, col)

ch_pos = {ch: grid_xy[i] for i, ch in enumerate(channels)}
ch_pos_N = {k: to_int_pos(v, N) for k, v in ch_pos.items()}

# ---------------------------------------------------------------------
# 4) Build SNN layers that expect "<channel>-<band>-resonators" inputs
# ---------------------------------------------------------------------
bands2topographies = {}
for band, (lf, hf) in bands.items():
    resonators2band = {}
    for ch in channels:
        # only the resonators present in your dataset (all of them are < 10 Hz)
        freqs = full_resonator_array[(full_resonator_array >= lf) & (full_resonator_array < hf)]
        if len(freqs) == 0:
            # no resonators in this band at the low set; skip band for this ch
            continue
        clk_freqs = torch.where(freqs < 10.0,
                    torch.tensor(15360,  dtype=torch.int64),
                    torch.tensor(153600, dtype=torch.int64))
        resonators2band[ch] = Resonators2BandsLayer(ch, band, clk_freqs)

    if len(resonators2band) > 0:
        bands2topographies[band] = Bands2TopographyLayer(
            band, resonators2band, shape=(N, N), ch_pos=ch_pos_N
        )

# 2-class SNN head (change hidden sizes as you like)
num_classes = 2
snn = Topographies2SNN(bands2topographies, N * N, int(N*np.sqrt(N)), num_classes)

# Device/threads
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
print("Running on Device =", device)
snn.to(device)


labels_map = {"seizure": 0, "non_seizure": 1}   # you can flip if you prefer

def load_one_npz_pickle(fp: Path):
    with open(fp, "rb") as f:
        # dict: "<channel>-<band>" -> torch.Tensor [n_resonators, time]
        d = pickle.load(f)
    # match the SNN naming convention expected by utils
    out = {
        f"{k}-resonators": t.T.reshape(t.shape[1], 1, t.shape[0]).to(device)
        for k, t in d.items()
    }
    return out

# Collect a small sample set to sanity-check run()
samples = []   # list of (inputs_dict, label_tensor)
# Pick one subject/trial as an example; extend to loop all subjects later.
for subject_dir in sorted(DATA_ROOT.iterdir()):
    if not subject_dir.is_dir():
        continue
    for trial_dir in sorted(subject_dir.iterdir()):
        # two labels: seizure, non_seizure
        for label_name, label_id in labels_map.items():
            label_dir = trial_dir / label_name
            if not label_dir.is_dir():
                continue
            for fp in sorted(label_dir.glob("*.npz")):
                try:
                    inputs_dict = load_one_npz_pickle(fp)
                except Exception as e:
                    print(f"Failed to load {fp}: {e}")
                    continue
                samples.append((inputs_dict, torch.tensor([label_id], device=device)))
    # break  # uncomment to only take the first subject

print(f"Loaded {len(samples)} window(s).")

# ---------------------------------------------------------------------
# 6) Run a quick forward pass on one sample (get time dynamically)
# ---------------------------------------------------------------------
if len(samples):
    inputs, y = samples[0]
    # infer time from any input tensor: shape is [n_resonators, 1, time]
    some_key = next(iter(inputs))
    time = inputs[some_key].shape[-1]
    print("Inferred time window =", time)
    snn.run(inputs=inputs, time=time)

# ---------------------------------------------------------------------
# 7) (Optional) compute simple power features to mirror your old printout
# ---------------------------------------------------------------------
power = np.std
power_on_band = {lbl: {band: [] for band in bands.keys()} for lbl in labels_map.keys()}
power_on_channel = {lbl: {ch: [] for ch in channels} for lbl in labels_map.keys()}

for lbl_name in labels_map.keys():
    # take first window of that label if available
    any_sample = next((inp for inp, lab in samples if lab.item() == labels_map[lbl_name]), None)
    if any_sample is None:
        continue

    for channel_band_res, signals in any_sample.items():
        # key format: "<channel>-<band>-resonators"
        try:
            channel, band, _ = channel_band_res.split("-")
        except ValueError:
            # if your key has extra '-' in channel names like 'FP2-F8', split safely:
            parts = channel_band_res.split("-")
            channel = "-".join(parts[:-2])  # all but last two tokens
            band = parts[-2]
        # signals: [n_resonators, 1, time]
        for j in range(signals.shape[0]):
            v = signals[j, 0].detach().cpu().numpy()
            if band in power_on_band[lbl_name]:
                power_on_band[lbl_name][band].append(power(v))
            if channel in power_on_channel[lbl_name]:
                power_on_channel[lbl_name][channel].append(power(v))

print("Power features (sampled):")
for lbl_name in labels_map:
    for band in bands:
        vals = power_on_band[lbl_name][band]
        if vals:
            print(f"{lbl_name:>12} | {band:>5}: mean={np.mean(vals):.4f} (n={len(vals)})")


# ---------------------------------------------------------------------
# 7.5) Guard clause – no data loaded
# ---------------------------------------------------------------------
if len(samples) == 0:
    print("\n⚠️ No samples were loaded from", DATA_ROOT)
    print("   → Please check that the dataset directory exists and contains .npz files.")
    print("   → Example path structure:")
    print("       dataset/spikes_windowed/chb01/chb01_03/seizure/3061.0.npz")
    print("       dataset/spikes_windowed/chb01/chb01_03/non_seizure/2941.0.npz")
    print("\nSkipping plots and stats export.\n")
    sys.exit(0)


# --- Compute mean stats ---
band_means = {
    lbl: {band: float(np.mean(vals)) if len(vals) else None
          for band, vals in power_on_band[lbl].items()}
    for lbl in power_on_band
}
channel_means = {
    lbl: {ch: float(np.mean(vals)) if len(vals) else None
          for ch, vals in power_on_channel[lbl].items()}
    for lbl in power_on_channel
}

# --- Save CSVs ---
band_df = pd.DataFrame(band_means)
band_csv = OUTPUT_STATS_DIR / "power_per_band.csv"
band_df.to_csv(band_csv, index_label="Band")

channel_df = pd.DataFrame(channel_means)
channel_csv = OUTPUT_STATS_DIR / "power_per_channel.csv"
channel_df.to_csv(channel_csv, index_label="Channel")

# --- Save JSON (optional) ---
json_path = OUTPUT_STATS_DIR / "power_full_stats.json"
with open(json_path, "w") as f:
    json.dump({"band": band_means, "channel": channel_means}, f, indent=2)

print(f"\n✅ Saved power statistics to {OUTPUT_STATS_DIR}")
print(f"   - {band_csv.name}")
print(f"   - {channel_csv.name}")
print(f"   - {json_path.name}")

# --- Plot: Mean Band Power (Seizure vs. Non-Seizure) ---
plt.figure(figsize=(8, 5))
bands_list = list(band_df.index)
x = np.arange(len(bands_list))
width = 0.35

plt.bar(x - width/2, band_df["seizure"], width, label="Seizure", color="#d24b4b")
plt.bar(x + width/2, band_df["non_seizure"], width, label="Non-Seizure", color="#4b8fd2")

plt.xticks(x, bands_list, rotation=45)
plt.ylabel("Mean Power (std)")
plt.title("Mean Band Power: Seizure vs Non-Seizure")
plt.legend()
plt.tight_layout()

plot_path = OUTPUT_STATS_DIR / "power_bands.png"
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"📊 Saved plot: {plot_path}")