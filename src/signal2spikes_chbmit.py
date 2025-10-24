import os
import re
import torch
import pickle
from zipfile import BadZipFile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import mne
from scipy.signal import butter, filtfilt, resample

from sctn.resonator import resonator as spikes_resonator

# ============================================================
#                       CONFIG
# ============================================================

# Root folder that contains subject folders: datasets/EEG_Data/chbmit/chb01/...
# CHBMIT_ROOT = "../datasets/EEG_Data/chbmit"
CHBMIT_ROOT = r"C:\Users\talsh\Desktop\Noam\Msc\Thesa\chbmit"

# Where to write spikes (keeps your original directory shape)
SPIKES_ROOT = "../datasets/EEG_Data/EEG_spikes"   # same structure as before

# Sampling rate for CHB-MIT
fs = 256  # Hz

# Resonators (unchanged)
clk_resonators = {
    15360: [1.1, 1.3, 1.6, 1.9, 2.2, 2.5, 2.88, 3.05, 3.39, 3.7, 4.12, 4.62, 5.09, 5.45, 5.87, 6.36, 6.8, 7.6, 8.6],
    153600: [10.5, 11.5, 12.8, 15.8, 16.6, 19.4, 22.0, 24.8, 28.4, 30.5, 34.7, 37.2, 40.2, 43.2, 47.7, 52.6, 57.2],
}

# Limit what you process for a first run
SUBJECTS = ["chb01"]              # folder names inside CHBMIT_ROOT
EDF_WHITELIST = ["chb01_03.edf"]               # e.g., ["chb01_03.edf"] to restrict; or None for all
ONLY_MINUTES = [49,50,51]                # e.g., set to [0,1,2] to shorten; or None for all in file

# ============================================================
#                   UTILS
# ============================================================

def bp_filter(data, f_lo, f_hi, fs_):
    """6th-order Butterworth bandpass on time axis 0."""
    f_ny = fs_ / 2.0
    b_lo = f_lo / f_ny
    b_hi = f_hi / f_ny
    b1, a1 = butter(6, b_hi, btype="lowpass")
    b2, a2 = butter(6, b_lo, btype="highpass")
    out = filtfilt(b1, a1, data, axis=0)
    out = filtfilt(b2, a2, out, axis=0)
    return out


def resample_signal(f_new, f_source, data):
    n_samples_orig = data.shape[0]
    n_samples_new = int(n_samples_orig * f_new / f_source)
    if n_samples_new <= 0:
        return np.array([], dtype=data.dtype)
    return resample(data, n_samples_new)


def is_file_exist(path: str, clk_freq: int):
    """Check if spike file exists and is valid; normalize timestamps format if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        return False
    try:
        loaded_spikes = np.load(path)["spikes"]
        if len(loaded_spikes) == 0:
            return False
        if np.sum(loaded_spikes > 1) > 0:
            # timestamps -> spikes encoding
            spikes_encode = np.zeros(60 * clk_freq, dtype=np.int8)
            if loaded_spikes[-1] >= 60 * clk_freq:
                raise ValueError(
                    f"{path} seems not to be spikes nor timestamps."
                )
            spikes_encode[loaded_spikes] = 1
            np.savez_compressed(file=path, spikes=spikes_encode)
        return True
    except BadZipFile as e:
        print(path)
        raise e


def save_output(resonator, spikes_output_path, clk_freq: int):
    output_neuron = resonator.layers_neurons[-1].neurons[-1]
    np.savez_compressed(
        file=spikes_output_path,
        spikes=output_neuron.out_spikes(False, clk_freq * 60).astype("int8"),
    )


# ============================================================
#            CHB-MIT: SUMMARY & EDF PARSING
# ============================================================

SUMMARY_CH_LINE = re.compile(r"^\s*Channel\s+\d+:\s*(.+?)\s*$", re.IGNORECASE)
SEIZURE_BLOCK = re.compile(
    r"^File Name:\s*(?P<fname>\S+)\s*$"
    r"|^Number of Seizures in File:\s*(?P<n>\d+)\s*$"
    r"|^Seizure\s+\d+\s+Start Time:\s*(?P<start>\d+)\s*seconds\s*$"
    r"|^Seizure\s+\d+\s+End Time:\s*(?P<end>\d+)\s*seconds\s*$",
    re.IGNORECASE
)

def parse_summary(summary_path: str):
    """
    Returns:
        sampling_rate: int
        channels_order: list[str] (bipolar e.g. 'FP1-F7', exclude '-' entries)
        seizures_by_file: dict[str, list[tuple(start_sec, end_sec)]]
    """
    sampling_rate = 256
    channels_order = []
    seizures_by_file = {}

    with open(summary_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.readlines()]

    # sampling rate (optional, default to 256)
    for ln in lines:
        if "Data Sampling Rate" in ln and "Hz" in ln:
            try:
                sampling_rate = int(re.findall(r"(\d+)\s*Hz", ln)[0])
            except Exception:
                sampling_rate = 256
            break

    # channels
    for ln in lines:
        m = SUMMARY_CH_LINE.match(ln)
        if m:
            label = m.group(1).strip()
            if label != "-" and len(label) > 0:
                channels_order.append(label)

    # seizures per file
    current_file = None
    starts = []
    ends = []
    for ln in lines:
        m = SEIZURE_BLOCK.match(ln)
        if not m:
            continue
        gd = m.groupdict()
        if gd.get("fname"):
            # save previous
            if current_file is not None and len(starts) == len(ends):
                seizures_by_file[current_file] = list(zip(starts, ends))
            # reset
            current_file = gd["fname"].strip()
            starts, ends = [], []
        elif gd.get("start"):
            starts.append(int(gd["start"]))
        elif gd.get("end"):
            ends.append(int(gd["end"]))
        # 'n' (Number of Seizures) not strictly needed to build intervals
    # last block
    if current_file is not None and len(starts) == len(ends):
        seizures_by_file[current_file] = list(zip(starts, ends))

    return sampling_rate, channels_order, seizures_by_file


def read_edf_channels(edf_path: str, desired_labels: list[str]) -> pd.DataFrame:
    """
    Load EDF and return a DataFrame with columns ordered as desired_labels,
    keeping only labels that exist in the file.
    Applies bandpass (.5–62 Hz) and a common average reference (CAR).
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    # EDF channel names in CHB-MIT are already the bipolar labels (e.g., 'FP1-F7')
    available = {name.upper(): i for i, name in enumerate(raw.ch_names)}

    keep_labels = [lab for lab in desired_labels if lab.upper() in available]
    if len(keep_labels) == 0:
        raise ValueError(f"No desired bipolar channels found in {edf_path}")

    data = raw.get_data(picks=keep_labels).T  # shape (n_samples, n_ch)
    # band-pass
    data = bp_filter(data, 0.5, 62.0, fs)
    # common average reference across channels (gives everyone the same baseline)
    data = data - np.mean(data, axis=1, keepdims=True)

    # normalize per channel to [-1,1] range to be consistent with your old preprocessing
    peak = np.max(np.abs(data), axis=0)
    peak[peak == 0] = 1.0
    data = data / peak

    return pd.DataFrame(data, columns=keep_labels)


def minutes_labels(n_samples: int, fs_: int, seizure_intervals_sec: list[tuple[int,int]], only_minutes=None):
    """
    Return a dict: {'seizure':[m_indices], 'non_seizure':[m_indices]}
    Minute m corresponds to seconds [60*m, 60*(m+1)).
    """
    total_minutes = n_samples // (60 * fs_)
    if only_minutes is not None:
        total_minutes = max([m for m in only_minutes if m is not None] + [0]) + 1

    seizure_mins = set()
    for (s, e) in seizure_intervals_sec:
        m0 = int(s // 60)
        m1 = int((e - 1) // 60)  # inclusive end minute if any overlap
        for m in range(m0, m1 + 1):
            seizure_mins.add(m)

    all_mins = range(total_minutes) if only_minutes is None else only_minutes
    non_mins = [m for m in all_mins if m not in seizure_mins and m * 60 * fs_ < n_samples]

    return {
        "seizure": sorted([m for m in seizure_mins if m * 60 * fs_ < n_samples]),
        "non_seizure": sorted(non_mins),
    }


# ============================================================
#                SPIKE GENERATION PIPELINE
# ============================================================

def resonator_fname_to_freq(name: str) -> float:
    return float(Path(name).stem)  # e.g., '5.45.npz' -> 5.45


def process_file(subject: str, edf_file: str, channels_order: list[str], seizures_by_file: dict[str, list[tuple[int,int]]]):
    """
    Generate spikes per minute, per channel, per resonator for a single EDF file.
    Output folder: SPIKES_ROOT/<subject>/<label>/<minute>/<channel>/<clk>/<f0>.npz
    """
    edf_path = os.path.join(CHBMIT_ROOT, subject, edf_file)
    df = read_edf_channels(edf_path, channels_order)
    n_samples = df.shape[0]

    # minutes labeled from summary
    seizure_intervals = seizures_by_file.get(edf_file, [])
    mins = minutes_labels(n_samples, fs, seizure_intervals, only_minutes=ONLY_MINUTES)

    # iterate channels
    for ch in df.columns:
        print(f"  Processing {edf_file}, channel {ch}...")
        ch_data = df[ch].values  # normalized already

        for clk_freq, resonators in clk_resonators.items():
            print(f"    Clock freq: {clk_freq} Hz")
            for f0 in resonators:
                print(f"      Resonator freq: {f0} Hz")
                resonator = spikes_resonator(freq0=f0, clk_freq=clk_freq)

                # warm-up
                resonator.log_out_spikes(-1)
                output_neuron = resonator.layers_neurons[-1].neurons[-1]
                resonator.input_full_data(np.zeros(clk_freq * 5))
                output_neuron.forget_logs()

                # per minute
                for label, minute_list in mins.items():
                    print(f"        Label: {label}, minutes: {minute_list}")
                    for m in minute_list:
                        print(f"          Minute {m}...")
                        start = fs * m * 60
                        stop = min(fs * (m + 1) * 60, n_samples)
                        if stop <= start:
                            continue

                        # resample this minute to clk_freq
                        segment = ch_data[start:stop]
                        data_resampled = resample_signal(clk_freq, fs, segment)
                        if data_resampled.size == 0:
                            continue

                        # output path
                        # use "trial" = <subject>/<edf_basename_without_ext>
                        edf_base = Path(edf_file).stem
                        output_folder = os.path.join(
                            SPIKES_ROOT,
                            f"{subject}/{edf_base}",   # trial id
                            label,
                            str(m),
                            ch,
                            str(clk_freq),
                            f"{f0}.npz",
                        )

                        if not is_file_exist(output_folder, clk_freq):
                            resonator.input_full_data(data_resampled)
                            save_output(resonator, output_folder, clk_freq)
                            output_neuron.forget_logs()


def create_datasets(time_of_sample_s, overlap, subjects, output_path, skip_beginning_s=5):
    """
    Build dataset windows from the generated per-minute spikes (same as your original logic).
    Keeps path format compatible with your training code:
      <trial>/<label>/<minute>/<channel>/<clk>/<f0>.npz  -> windows *.npz under output_path
    """
    spikes_root = SPIKES_ROOT
    spikes_in_sample = int(153600 * time_of_sample_s)  # default sample_fs is 153600 in your pipeline
    step = int(spikes_in_sample * (1 - overlap))
    bands = {  # used only for selection by frequency ranges when aggregating
        "Delta": (0.5, 4),
        "Theta": (4, 8),
        "Alpha": (8, 14),
        "Beta": (14, 32),
        "Gamma": (32, 62),
    }

    trials = []
    for subj in subjects:
        subj_dir = os.path.join(spikes_root, subj)
        if not os.path.isdir(subj_dir):
            continue
        for trial in os.listdir(subj_dir):
            trials.append(f"{subj}/{trial}")

    total_items = 0
    # rough upper bound (safe progress bar); exact counting would be expensive
    for trial in trials:
        for label in ["seizure", "non_seizure"]:
            trial_label_dir = os.path.join(spikes_root, trial, label)
            if not os.path.isdir(trial_label_dir):
                continue
            for minute in os.listdir(trial_label_dir):
                minute_dir = os.path.join(trial_label_dir, minute)
                for ch in os.listdir(minute_dir):
                    for clk in os.listdir(os.path.join(minute_dir, ch)):
                        clk_dir = os.path.join(minute_dir, ch, clk)
                        total_items += len(os.listdir(clk_dir))

    with tqdm(total=total_items, desc="Create dataset windows") as pbar:
        for trial in trials:
            trial_dir = os.path.join(spikes_root, trial)
            for label in ["seizure", "non_seizure"]:
                trial_label_dir = os.path.join(trial_dir, label)
                if not os.path.isdir(trial_label_dir):
                    continue
                for minute in os.listdir(trial_label_dir):
                    minute = int(minute)
                    full_minute_spikes_input = {}

                    minute_dir = os.path.join(trial_label_dir, str(minute))
                    # build inputs by bands/channels
                    for band_name, (lf, hf) in bands.items():
                        for ch_name in os.listdir(minute_dir):
                            ch_dir = os.path.join(minute_dir, ch_name)
                            full_minute_spikes_input[f"{ch_name}-{band_name}"] = {}
                            for clk_freq_str in os.listdir(ch_dir):
                                clk_freq = int(clk_freq_str)
                                freq_dir = os.path.join(ch_dir, clk_freq_str)
                                resonator_freqs = sorted(os.listdir(freq_dir), key=resonator_fname_to_freq)
                                for f0_str in resonator_freqs:
                                    f0 = float(Path(f0_str).stem)
                                    if not (lf < f0 <= hf):
                                        continue
                                    loaded_spikes = np.load(os.path.join(freq_dir, f0_str))["spikes"].astype(np.int8)
                                    # drop first few seconds to avoid warm-up
                                    loaded_spikes = loaded_spikes[skip_beginning_s * clk_freq:]
                                    # ensure minute length 60 - skip_beginning_s seconds
                                    if clk_freq != 153600:
                                        extended = np.zeros(153600 * (60 - skip_beginning_s), dtype=np.int8)
                                        step_stride = int(153600 / clk_freq)
                                        extended[::step_stride] = loaded_spikes[:extended[::step_stride].shape[0]]
                                        loaded_spikes = extended
                                    full_minute_spikes_input[f"{ch_name}-{band_name}"][f0] = loaded_spikes
                                    pbar.update(1)

                    # window over the minute
                    sample_fs = 153600
                    for sample_start_index in range(
                        sample_fs,
                        sample_fs * (60 - skip_beginning_s) - spikes_in_sample,
                        step,
                    ):
                        sample_spikes_input = {
                            input_name: torch.from_numpy(
                                np.stack([
                                    s[sample_start_index: sample_start_index + spikes_in_sample]
                                    for s in resonators_arrays.values()
                                ])
                            )
                            for input_name, resonators_arrays in full_minute_spikes_input.items()
                        }
                        file_name = f"{(sample_start_index / sample_fs) + minute * 60}"
                        out_path = Path(output_path) / trial / label / f"{file_name}.npz"
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        if not out_path.is_file():
                            with open(out_path, "wb") as f:
                                pickle.dump(sample_spikes_input, f, protocol=pickle.HIGHEST_PROTOCOL)


# ============================================================
#                    MAIN DRIVER
# ============================================================

def main():
    subjects = [s for s in SUBJECTS if os.path.isdir(os.path.join(CHBMIT_ROOT, s))]
    if not subjects:
        raise RuntimeError(f"No subject folders found under {CHBMIT_ROOT}")

    # Pass 1: Generate spikes per minute per channel (same as your original minute pipeline)
    for subject in subjects:
        subj_dir = os.path.join(CHBMIT_ROOT, subject)
        summary_path = os.path.join(subj_dir, f"{subject}-summary.txt")
        if not os.path.isfile(summary_path):
            raise FileNotFoundError(f"Missing summary: {summary_path}")

        sampling_rate, ch_order, seizures_by_file = parse_summary(summary_path)
        if sampling_rate != fs:
            print(f"Warning: summary says fs={sampling_rate} but using fs={fs}")

        edf_files = sorted([f for f in os.listdir(subj_dir) if f.lower().endswith(".edf")])
        if EDF_WHITELIST:
            edf_files = [f for f in edf_files if f in EDF_WHITELIST]
        if not edf_files:
            print(f"No EDF files for {subject}, skipping.")
            continue

        total = len(edf_files) * (len(ch_order) or 1) * sum(len(v) for v in clk_resonators.values())
        with tqdm(total=total, desc=f"Subject {subject} (spikes)") as pbar:
            for edf_file in edf_files:
                process_file(subject, edf_file, ch_order, seizures_by_file)
                pbar.update(1)

    # Pass 2: Build dataset windows (unchanged logic)
    create_datasets(
        time_of_sample_s=0.25,
        overlap=0,
        subjects=subjects,
        output_path="../datasets/EEG_Data/",  # mirrors your original output root
        skip_beginning_s=5,
    )


if __name__ == "__main__":
    main()
