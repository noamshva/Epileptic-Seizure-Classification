"""
Train the SA-style SNN on CHB-MIT seizure / non-seizure windows.

Pure BindsNet supervised STDP: PostPre learning is already attached to the
`topography → fc1 → fc2 → output` connections in `utils.Topographies2SNN`.
During training we drive the network with the per-window spike trains and
**clamp** the correct output neuron to spike, which is the standard BindsNet
recipe for supervised STDP. No PyTorch optimizer / backprop.
"""
import sys
import types

# bindsnet 0.3.x imports `torch._six` which was removed from PyTorch >= 1.8.
# Provide a shim before importing bindsnet.
if "torch._six" not in sys.modules:
    _shim = types.ModuleType("torch._six")
    _shim.container_abcs = ()
    _shim.string_classes = (str,)
    _shim.int_classes = (int,)
    sys.modules["torch._six"] = _shim

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from utils import (
    Bands2TopographyLayer,
    Resonators2BandsLayer,
    Topographies2SNN,
)


# ==========================================================
# 1. Configuration
# ==========================================================
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

OUTPUT_MODEL_DIR = _PROJECT_ROOT / "output_data" / "trained_models"
OUTPUT_LOG_DIR = _PROJECT_ROOT / "output_data" / "train_logs"
OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_LOG_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("Running on CPU (CUDA not available)")

# Resonator set across Delta/Theta/Alpha/Beta (Gamma dropped) — kept in sync
# with `clk_resonators` in 1.signal2spikes_chbmit.py.
full_resonator_array = torch.tensor(
    [
        # Delta + Theta (slow clock)
        1.1, 1.3, 1.6, 1.9, 2.2, 2.5, 2.88, 3.05, 3.39, 3.7,
        4.12, 4.62, 5.09, 5.45, 5.87, 6.36, 6.8, 7.6,
        # Alpha (split across clocks)
        8.6, 10.5, 11.5, 12.8,
        # Beta
        16.6, 19.4, 22.0, 24.8, 28.4, 30.5,
    ],
    dtype=torch.float32,
)

BANDS = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 14),
    "Beta":  (14, 32),
}

# ==========================================================
# 2. CHB-MIT 18-channel bipolar layout on the 11×11 topography
# ==========================================================
# Standard 10-20 normalized coordinates (x = lateral, y = anterior, both in [-1, 1]).
ELECTRODE_XY = {
    "FP1": (-0.27,  0.95), "FP2": ( 0.27,  0.95),
    "F7":  (-0.81,  0.59), "F3":  (-0.55,  0.55), "FZ": (0.0,  0.55),
    "F4":  ( 0.55,  0.55), "F8":  ( 0.81,  0.59),
    "T7":  (-0.95,  0.00), "C3":  (-0.50,  0.00), "CZ": (0.0,  0.00),
    "C4":  ( 0.50,  0.00), "T8":  ( 0.95,  0.00),
    "P7":  (-0.81, -0.59), "P3":  (-0.55, -0.55), "PZ": (0.0, -0.55),
    "P4":  ( 0.55, -0.55), "P8":  ( 0.81, -0.59),
    "O1":  (-0.27, -0.95), "O2":  ( 0.27, -0.95),
}

CHANNELS = [
    "FT9-FT10", "T7-P7", "T8-P8", "P7-O1",
    "F7-T7", "P8-O2", "CZ-PZ", "P3-O1",
    "FP1-F7", "C4-P4",
]

# FT9-FT10 isn't in the standard 10-20 list — approximate it by the midpoint of
# the FT9/FT10 placements (slightly anterior of T7/T8, and slightly inferior).
# These coordinates are normalized like the 10-20 dict above.
ELECTRODE_XY["FT9"]  = (-0.88,  0.20)
ELECTRODE_XY["FT10"] = ( 0.88,  0.20)


def _midpoint(pair: str):
    a, b = pair.split("-")
    ax, ay = ELECTRODE_XY[a]
    bx, by = ELECTRODE_XY[b]
    return ((ax + bx) / 2, (ay + by) / 2)


N = 11
# Map each bipolar pair to a grid cell on the 11×11 topography.
# x (lateral) → column, y (anterior, larger = front) → row, where row 0 is the front.
ch_pos = {}
for ch in CHANNELS:
    xc, yc = _midpoint(ch)
    col = int(round((xc + 1) / 2 * (N - 1)))
    row = int(round((1 - (yc + 1) / 2) * (N - 1)))
    ch_pos[ch] = (row, col)


# ==========================================================
# 3. Build the SNN
# ==========================================================
def build_snn():
    bands2topographies = {}
    for band, (lo, hi) in BANDS.items():
        resonators2band = {}
        for ch in ch_pos.keys():
            freqs = full_resonator_array[
                (full_resonator_array >= lo) & (full_resonator_array < hi)
            ]
            if len(freqs) == 0:
                continue
            clk_freqs = torch.where(freqs < 10, 15360, 153600)
            resonators2band[ch] = Resonators2BandsLayer(ch, band, clk_freqs)
        if not resonators2band:
            continue
        bands2topographies[band] = Bands2TopographyLayer(
            band, resonators2band, shape=(N, N), ch_pos=ch_pos
        )

    # Paper-matching classifier: input (121) → hidden1 (64) → hidden2 (32) → output (2)
    snn = Topographies2SNN(
        topography_maps=bands2topographies,
        input_neurons=N * N,    # 121
        hidden1_neurons=64,
        hidden2_neurons=32,
        output_neurons=2,       # seizure vs non-seizure
    )
    return snn


# ==========================================================
# 4. Window loader (option B — lazy, read directly from spikes_raw/)
# ==========================================================
import os
import importlib.util as _ilu

# Load 1.signal2spikes_chbmit.py to reuse enumerate_windows + path constants.
_S2S_PATH = _SCRIPT_DIR / "1.signal2spikes_chbmit.py"
_s2s_spec = _ilu.spec_from_file_location("signal2spikes_chbmit", _S2S_PATH)
_s2s = _ilu.module_from_spec(_s2s_spec)
_s2s_spec.loader.exec_module(_s2s)

enumerate_windows = _s2s.enumerate_windows
SPIKES_ROOT = _s2s.SPIKES_ROOT
WINDOW_SECONDS = _s2s.WINDOW_SECONDS
WINDOW_OVERLAP = _s2s.WINDOW_OVERLAP

# Which resonator frequencies belong to each ACTIVE band (Gamma intentionally excluded).
# Frequencies < 10 Hz are encoded at clk=15360 Hz; the rest at clk=153600 Hz.
BAND_RESONATORS = {
    "Delta": [1.1, 1.3, 1.6, 1.9, 2.2, 2.5, 2.88, 3.05, 3.39, 3.7],
    "Theta": [4.12, 4.62, 5.09, 5.45, 5.87, 6.36, 6.8, 7.6],
    "Alpha": [8.6, 10.5, 11.5, 12.8],
    "Beta":  [16.6, 19.4, 22.0, 24.8, 28.4, 30.5],
}
_SAMPLE_FS = 153_600        # encoded-spike rate (resonator high clock)

# Temporal binning for the SNN. The encoded spike trains on disk stay at the
# full 153.6 kHz rate; this is just how many of those ticks we collapse into a
# single SNN simulation step. BIN_FACTOR=1 means no binning (paper fidelity).
# Each output bin holds the SUM of spikes in that window (0..BIN_FACTOR per
# resonator per bin). Larger BIN_FACTOR -> fewer SNN ticks per window ->
# proportionally faster training, at the cost of timing precision finer than
# BIN_FACTOR / _SAMPLE_FS seconds.
BIN_FACTOR = int(os.environ.get("BIN_FACTOR", "1"))


def _clk_of(f0: float) -> int:
    return 15_360 if f0 < 10 else 153_600


class _MinuteCache:
    """Caches the per-minute spike arrays for a small number of recent minutes.

    A cached entry is keyed by (subject, trial, minute_label, minute) and holds:
        dict[(channel, band)] -> list[(clk_freq, np.ndarray)]
    where each ndarray is the full 60-second binary spike train for ONE resonator
    at its native clock rate. ~280 small .npz reads per (channel, minute) miss,
    but with 58 windows per minute the cost amortizes to near-zero.
    """

    def __init__(self, capacity: int = 4):
        self.capacity = capacity
        self._store = {}

    def get(self, subject: str, trial: str, minute_label: str, minute: int):
        key = (subject, trial, minute_label, minute)
        if key in self._store:
            return self._store[key]
        minute_dir = Path(SPIKES_ROOT) / subject / trial / minute_label / str(minute)
        per_ch_band: dict[tuple[str, str], list[tuple[int, "np.ndarray"]]] = {}
        for ch in CHANNELS:
            ch_dir = minute_dir / ch
            if not ch_dir.is_dir():
                # Channel missing — leave it out; the SNN forward will see no input
                # for these keys, which is harmless (zero current into that layer).
                continue
            for band, freqs in BAND_RESONATORS.items():
                arrays: list[tuple[int, "np.ndarray"]] = []
                for f0 in freqs:
                    clk = _clk_of(f0)
                    f_path = ch_dir / str(clk) / f"{f0}.npz"
                    if not f_path.is_file():
                        continue
                    arr = np.load(f_path)["spikes"].astype(np.int8)
                    arrays.append((clk, arr))
                if arrays:
                    per_ch_band[(ch, band)] = arrays
        if len(self._store) >= self.capacity:
            self._store.pop(next(iter(self._store)))
        self._store[key] = per_ch_band
        return per_ch_band


def _build_window_inputs(win_desc, cache: _MinuteCache):
    """Slice a 2-sec window out of the cached per-minute spikes and return the
    BindsNet input dict (keyed by '<ch>-<band>-resonators') plus the int label.

    Slow-clock spikes are upsampled to _SAMPLE_FS by zero-insertion, so every
    input tensor has shape [_SAMPLE_FS * WINDOW_SECONDS, n_resonators].
    """
    minute_data = cache.get(
        win_desc["subject"], win_desc["trial"],
        win_desc["minute_label"], win_desc["minute"],
    )
    sec_offset = win_desc["sec_offset_in_minute"]
    target_len = int(_SAMPLE_FS * WINDOW_SECONDS)

    inputs: dict[str, torch.Tensor] = {}
    for (ch, band), per_res in minute_data.items():
        stacks = []
        for clk, arr in per_res:
            start = int(round(sec_offset * clk))
            end = start + int(round(WINDOW_SECONDS * clk))
            slice_arr = arr[start:end]
            if clk != _SAMPLE_FS:
                stride = _SAMPLE_FS // clk
                up = np.zeros(target_len, dtype=np.int8)
                fill = slice_arr[: up[::stride].size]
                up[::stride][: fill.size] = fill
                slice_arr = up
            else:
                if slice_arr.size != target_len:
                    pad = np.zeros(target_len, dtype=np.int8)
                    pad[: slice_arr.size] = slice_arr[:target_len]
                    slice_arr = pad
            stacks.append(slice_arr)
        if not stacks:
            continue
        # [n_res, T_full] — sum-bin along the time axis if BIN_FACTOR > 1
        stacked = np.stack(stacks)
        if BIN_FACTOR > 1:
            T_full = stacked.shape[1]
            T_binned = T_full // BIN_FACTOR
            # Trim to a multiple of BIN_FACTOR, then sum within each bin
            stacked = stacked[:, : T_binned * BIN_FACTOR]
            stacked = stacked.reshape(stacked.shape[0], T_binned, BIN_FACTOR).sum(axis=2)
        # [n_res, T] → [T, n_res], float32 for BindsNet
        tensor = torch.from_numpy(stacked).T.contiguous().float()
        inputs[f"{ch}-{band}-resonators"] = tensor

    label = 1 if win_desc["window_label"] == "seizure" else 0
    return inputs, label


# ==========================================================
# 5. Training
# ==========================================================
def _make_clamp(label: int, T: int):
    """Boolean mask of shape [T, 2] forcing the correct output neuron to spike."""
    mask = torch.zeros(T, 2, dtype=torch.bool)
    mask[:, label] = True
    return mask


def train_snn(snn, subjects, epochs=EPOCHS, shuffle: bool = True, max_windows=None):
    """Train the SNN via supervised PostPre STDP, reading windows lazily from
    `dataset/spikes_raw/` (no precomputed windowed cache).

    Args:
        snn: BindsNet network built by `build_snn`.
        subjects: list[str] of subject IDs to train on (e.g. ['chb01']).
        epochs: number of passes over the data.
        shuffle: shuffle the window order each epoch.
        max_windows: optional cap on windows per epoch (for smoke tests).
    """
    snn.to(DEVICE)
    snn.train(mode=True)

    windows = list(enumerate_windows(subjects, WINDOW_SECONDS, WINDOW_OVERLAP))
    n_seiz = sum(1 for w in windows if w["window_label"] == "seizure")
    n_non = len(windows) - n_seiz
    print(f"enumerate_windows -> {len(windows)} windows ({n_seiz} seizure, {n_non} non-seizure)")
    if max_windows is not None:
        windows = windows[:max_windows]
        print(f"(capped to {len(windows)} windows for this run)")

    cache = _MinuteCache(capacity=4)

    epoch_losses, epoch_accs = [], []
    for epoch in range(epochs):
        if shuffle:
            np.random.default_rng(epoch).shuffle(windows)
        correct, seen = 0, 0
        for win in tqdm(windows, desc=f"epoch {epoch + 1}/{epochs}"):
            try:
                inputs, label = _build_window_inputs(win, cache)
            except FileNotFoundError as e:
                print(f"  skip window {win['abs_start_s']:.2f}s: missing file {e}")
                continue
            if not inputs:
                continue

            T = next(iter(inputs.values())).shape[0]
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            clamp = {"output_layer": _make_clamp(label, T).to(DEVICE)}

            snn.run(inputs=inputs, time=T, clamp=clamp)

            # Read output spikes for accuracy bookkeeping.
            out_layer = snn.layers["output_layer"]
            if out_layer.s is not None:
                pred = int(out_layer.s.long().sum(dim=0).argmax().item())
                correct += int(pred == label)
                seen += 1

            snn.reset_state_variables()

        acc = correct / seen if seen else float("nan")
        epoch_accs.append(acc)
        epoch_losses.append(float("nan"))
        print(f"epoch {epoch + 1}/{epochs} | seen={seen} | train_acc={acc:.4f}")

    # Save connection weights (the only learned state).
    weights = {f"{s}__{t}": conn.w.detach().cpu()
               for (s, t), conn in snn.connections.items()}
    weights_path = OUTPUT_MODEL_DIR / "snn_weights.pt"
    torch.save(weights, weights_path)

    log_df = pd.DataFrame(
        {"epoch": np.arange(1, epochs + 1),
         "accuracy": epoch_accs,
         "loss": epoch_losses}
    )
    log_csv = OUTPUT_LOG_DIR / "training_log.csv"
    log_df.to_csv(log_csv, index=False)

    print(f"\nTraining complete.\n  - weights: {weights_path}\n  - log: {log_csv}")


# ==========================================================
# 6. Main
# ==========================================================
SUBJECTS = ["chb01"]   # which subjects to read from dataset/spikes_raw/

def main():
    print(f"Device = {DEVICE}")
    sim_T = (int(_SAMPLE_FS * WINDOW_SECONDS)
             // (BIN_FACTOR if BIN_FACTOR > 0 else 1))
    print(f"Config: EPOCHS={EPOCHS}  BIN_FACTOR={BIN_FACTOR}  "
          f"SNN ticks/window={sim_T:,}  "
          f"(sample_fs={_SAMPLE_FS} Hz, window={WINDOW_SECONDS}s)")
    snn = build_snn()
    print(f"Built SNN: {len(snn.layers)} layers, {len(snn.connections)} connections, "
          f"output_neurons={snn.layers['output_layer'].n}")
    # MAX_WINDOWS env var caps the per-epoch window count for smoke testing.
    max_windows = int(os.environ["MAX_WINDOWS"]) if "MAX_WINDOWS" in os.environ else None
    train_snn(snn, SUBJECTS, epochs=EPOCHS, max_windows=max_windows)


if __name__ == "__main__":
    main()
