import re
import torch
import pickle
from zipfile import BadZipFile
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import mne
import time
from scipy.signal import butter, filtfilt, resample, resample_poly
from concurrent.futures import ProcessPoolExecutor, as_completed
from sctn.resonator import resonator as spikes_resonator

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ============================================================
#                       CONFIG
# ============================================================

CHBMIT_ROOT = "../dataset/chbmit"
SPIKES_ROOT = "../dataset/spikes_raw"
OUTPUT_PATH = "../dataset/spikes_windowed"

fs = 256  # Hz

clk_resonators = {
    15360: [1.1, 1.3, 1.6, 1.9, 2.2, 2.5, 2.88, 3.05, 3.39, 3.7, 4.12, 4.62, 5.09, 5.45, 5.87, 6.36, 6.8, 7.6, 8.6],
    # 153600: [10.5, 11.5, 12.8, 15.8, 16.6, 19.4, 22.0, 24.8, 28.4, 30.5, 34.7, 37.2, 40.2, 43.2, 47.7, 52.6, 57.2],
}

SUBJECTS = ["chb01"]
EDF_WHITELIST = ["chb01_03.edf"]
ONLY_MINUTES = [49, 50, 51]
SELECT_BIPOLAR_INDEXES = [4, 16, 13, 5, 12, 9, 10, 14, 22]   # set to None to use all available channels

# ============================================================
#                   UTILS
# ============================================================

def bp_filter(data, f_lo, f_hi, fs_):
    f_ny = fs_ / 2.0
    b_lo = f_lo / f_ny
    b_hi = f_hi / f_ny
    b1, a1 = butter(6, b_hi, btype="lowpass")
    b2, a2 = butter(6, b_lo, btype="highpass")
    out = filtfilt(b1, a1, data, axis=0)
    out = filtfilt(b2, a2, out, axis=0)
    return out


def resample_signal(f_new, f_source, data):
    ratio = f_new / f_source
    if abs(round(ratio) - ratio) < 1e-12:
        up = int(round(ratio))
        return resample_poly(data, up=up, down=1)
    else:
        n_samples_new = int(data.shape[0] * ratio)
        if n_samples_new <= 0:
            return np.array([], dtype=data.dtype)
        return resample(data, n_samples_new)


def is_file_exist(path: str, clk_freq: int):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        return False
    try:
        loaded_spikes = np.load(path)["spikes"]
        if len(loaded_spikes) == 0:
            return False
        if np.sum(loaded_spikes > 1) > 0:
            spikes_encode = np.zeros(60 * clk_freq, dtype=np.int8)
            if loaded_spikes[-1] >= 60 * clk_freq:
                raise ValueError(f"{path} seems not to be spikes nor timestamps.")
            spikes_encode[loaded_spikes] = 1
            np.savez_compressed(file=path, spikes=spikes_encode)
        return True
    except BadZipFile as e:
        print(path)
        raise e


def save_output(resonator, spikes_output_path, clk_freq: int):
    output_neuron = resonator.layers_neurons[-1].neurons[-1]
    sp = output_neuron.out_spikes(False, clk_freq * 60).astype(np.int8)
    np.savez_compressed(file=spikes_output_path, spikes=sp)


# --- worker that processes ONE (clk_freq, f0) over all minutes for a single channel ---
def _run_one_resonator(args):
    (subject, edf_file, ch_name, clk_freq, f0,
     minutes_by_label, resampled_by_minute, spikes_root) = args

    start_t = time.time()
    edf_base = Path(edf_file).stem
    resonator = spikes_resonator(freq0=f0, clk_freq=clk_freq)

    # warmup
    resonator.log_out_spikes(-1)
    out_neuron = resonator.layers_neurons[-1].neurons[-1]
    resonator.input_full_data(np.zeros(clk_freq * 5))
    out_neuron.forget_logs()

    wrote = 0
    for label, minute_list in minutes_by_label.items():
        for m in minute_list:
            rs = resampled_by_minute.get(m)
            if rs is None or rs.size == 0:
                continue

            out_path = Path(spikes_root) / f"{subject}/{edf_base}" / label / str(m) / ch_name / str(clk_freq) / f"{f0}.npz"

            if not out_path.exists():
                out_path.parent.mkdir(parents=True, exist_ok=True)
                resonator.input_full_data(rs)
                sp = out_neuron.out_spikes(False, clk_freq * 60).astype(np.int8)
                np.savez_compressed(out_path, spikes=sp)
                out_neuron.forget_logs()
                wrote += 1

    dur = time.time() - start_t
    return (clk_freq, f0, ch_name, wrote, dur)

# ============================================================
#            CHB-MIT: SUMMARY & EDF PARSING
# ============================================================

SUMMARY_CH_LINE = re.compile(r"^\s*Channel\s+(\d+):\s*(.+?)\s*$", re.IGNORECASE)
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
        channels_order: list[str]                     # labels excluding '-'
        seizures_by_file: dict[str, list[(s,e)]]
        channels_by_index: dict[int, Optional[str]]   # keeps original index mapping; '-' -> None
    """
    sampling_rate = 256
    channels_order = []
    channels_by_index = {}
    seizures_by_file = {}

    with open(summary_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.readlines()]

    for ln in lines:
        if "Data Sampling Rate" in ln and "Hz" in ln:
            try:
                sampling_rate = int(re.findall(r"(\d+)\s*Hz", ln)[0])
            except Exception:
                sampling_rate = 256
            break

    # channels (preserve index)
    for ln in lines:
        m = SUMMARY_CH_LINE.match(ln)
        if m:
            idx = int(m.group(1))
            label = m.group(2).strip()
            if label == "-" or label == "":
                channels_by_index[idx] = None
            else:
                channels_by_index[idx] = label
                channels_order.append(label)

    # seizures per file
    current_file = None
    starts, ends = [], []
    for ln in lines:
        m = SEIZURE_BLOCK.match(ln)
        if not m:
            continue
        gd = m.groupdict()
        if gd.get("fname"):
            if current_file is not None and len(starts) == len(ends):
                seizures_by_file[current_file] = list(zip(starts, ends))
            current_file = gd["fname"].strip()
            starts, ends = [], []
        elif gd.get("start"):
            starts.append(int(gd["start"]))
        elif gd.get("end"):
            ends.append(int(gd["end"]))
    if current_file is not None and len(starts) == len(ends):
        seizures_by_file[current_file] = list(zip(starts, ends))

    return sampling_rate, channels_order, seizures_by_file, channels_by_index


def read_edf_channels(edf_path: str, desired_labels: list[str]) -> pd.DataFrame:
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    available = {name.upper(): i for i, name in enumerate(raw.ch_names)}
    keep_labels = [lab for lab in desired_labels if lab and lab.upper() in available]
    if len(keep_labels) == 0:
        raise ValueError(f"No desired bipolar channels found in {edf_path}")

    data = raw.get_data(picks=keep_labels).T  # (n_samples, n_ch)
    data = bp_filter(data, 0.5, 62.0, fs)
    data = data - np.mean(data, axis=1, keepdims=True)

    peak = np.max(np.abs(data), axis=0)
    peak[peak == 0] = 1.0
    data = data / peak
    return pd.DataFrame(data, columns=keep_labels)


def minutes_labels(n_samples: int, fs_: int, seizure_intervals_sec, only_minutes=None):
    total_minutes = n_samples // (60 * fs_)
    if only_minutes is not None:
        total_minutes = max([m for m in only_minutes if m is not None] + [0]) + 1

    seizure_mins = set()
    for (s, e) in seizure_intervals_sec:
        m0 = int(s // 60)
        m1 = int((e - 1) // 60)
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
    return float(Path(name).stem)

def process_file(subject: str,
                 edf_file: str,
                 desired_labels: list[str],
                 seizures_by_file: dict[str, list[tuple[int,int]]],
                 log_every:int = 5):

    """
    Generate spikes per minute, per channel, per resonator for a single EDF file.
    Parallelizes across f0 for each (channel, clk_freq).
    Prints indicative progress for (channel, clk) with completion rate across f0's.
    """
  
    edf_path = os.path.join(CHBMIT_ROOT, subject, edf_file)
    df = read_edf_channels(edf_path, desired_labels)
    n_samples = df.shape[0]

    seizure_intervals = seizures_by_file.get(edf_file, [])
    mins = minutes_labels(n_samples, fs, seizure_intervals, only_minutes=ONLY_MINUTES)

    n_seiz = len(mins.get("seizure", []))
    n_non  = len(mins.get("non_seizure", []))
    print(f"\n>>> {subject}/{edf_file} | samples={n_samples:,} | minutes: seizure={n_seiz} non_seizure={n_non}")

    for ch in df.columns:
        ch_data = df[ch].values
        print(f"  [Channel] {ch}")

        for clk_freq, resonators in clk_resonators.items():
            resampled_by_minute = {}
            for label, minute_list in mins.items():
                for m in minute_list:
                    start = fs * m * 60
                    stop  = min(fs * (m + 1) * 60, n_samples)
                    if stop <= start:
                        continue
                    seg = ch_data[start:stop]
                    rs = resample_signal(clk_freq, fs, seg)
                    if rs.size > 0:
                        resampled_by_minute[m] = rs

            n_minutes = len(resampled_by_minute)
            if n_minutes == 0:
                print(f"    clk={clk_freq:>6} Hz | no usable minutes after resample; skipping.")
                continue

            total_f0 = len(resonators)
            print(f"    clk={clk_freq:>6} Hz | minutes_resampled={n_minutes} | f0_count={total_f0}")

            maxw = max(1, (os.cpu_count() or 2) - 1)
            done = 0
            wrote_total = 0
            t0 = time.time()

            with ProcessPoolExecutor(max_workers=maxw) as ex:
                futures = []
                for f0 in resonators:
                    args = (
                        subject, edf_file, ch, clk_freq, f0,
                        mins,
                        resampled_by_minute,
                        SPIKES_ROOT,
                    )
                    futures.append(ex.submit(_run_one_resonator, args))

                for fut in as_completed(futures):
                    clk_done, f0_done, ch_done, wrote, dur = fut.result()
                    done += 1
                    wrote_total += wrote
                    if (done % log_every == 0) or (done == total_f0):
                        elapsed = time.time() - t0
                        rate = done / elapsed if elapsed > 0 else 0.0
                        print(f"      [clk={clk_freq:>6} Hz] f0 {done:>2}/{total_f0} "
                              f"(last={f0_done:>5.2f} Hz, wrote {wrote} files) | "
                              f"elapsed {elapsed:5.1f}s | {rate:4.2f} f0/s")

            print(f"    ✔ clk={clk_freq:>6} Hz done | wrote_files={wrote_total}\n")


def create_datasets(time_of_sample_s, overlap, subjects, output_path, skip_beginning_s=5):
    """
    Build dataset windows from the generated per-minute spikes.

    Robust to missing bands (e.g., when only 15_360 Hz clk is used),
    and uses a single, consistent sample_fs derived from the clk dirs found.
    """
    spikes_root = SPIKES_ROOT

    # Discover which clk_freqs actually exist on disk for the first trial,
    # and pick the highest as the windowing sample rate.
    def discover_sample_fs():
        for subj in subjects:
            subj_dir = os.path.join(spikes_root, subj)
            if not os.path.isdir(subj_dir): 
                continue
            for trial in os.listdir(subj_dir):
                trial_dir = os.path.join(subj_dir, trial)
                # find any minute dir
                for label in ["seizure", "non_seizure"]:
                    lab_dir = os.path.join(trial_dir, label)
                    if not os.path.isdir(lab_dir): 
                        continue
                    for minute in os.listdir(lab_dir):
                        minute_dir = os.path.join(lab_dir, minute)
                        # find any channel dir
                        for ch in os.listdir(minute_dir):
                            ch_dir = os.path.join(minute_dir, ch)
                            clk_list = [int(x) for x in os.listdir(ch_dir) if x.isdigit()]
                            if clk_list:
                                return max(clk_list)
        # fallback to 15360
        return 15360

    sample_fs = discover_sample_fs()
    spikes_in_sample = int(sample_fs * time_of_sample_s)
    step = int(spikes_in_sample * (1 - overlap))

    bands = {
        "Delta": (0.5, 4),
        "Theta": (4, 8),
        "Alpha": (8, 14),
        "Beta": (14, 32),
        "Gamma": (32, 62),
    }

    # gather all trials
    trials = []
    for subj in subjects:
        subj_dir = os.path.join(spikes_root, subj)
        if not os.path.isdir(subj_dir):
            continue
        for trial in os.listdir(subj_dir):
            trials.append(f"{subj}/{trial}")

    # rough count for the progress bar
    total_items = 0
    for trial in trials:
        for label in ["seizure", "non_seizure"]:
            trial_label_dir = os.path.join(spikes_root, trial, label)
            if not os.path.isdir(trial_label_dir):
                continue
            for minute in os.listdir(trial_label_dir):
                minute_dir = os.path.join(trial_label_dir, minute)
                for ch in os.listdir(minute_dir):
                    ch_dir = os.path.join(minute_dir, ch)
                    if not os.path.isdir(ch_dir): 
                        continue
                    for clk in os.listdir(ch_dir):
                        freq_dir = os.path.join(ch_dir, clk)
                        if os.path.isdir(freq_dir):
                            total_items += len(os.listdir(freq_dir))

    with tqdm(total=total_items, desc="Create dataset windows") as pbar:
        for trial in trials:
            trial_dir = os.path.join(spikes_root, trial)
            for label in ["seizure", "non_seizure"]:
                trial_label_dir = os.path.join(trial_dir, label)
                if not os.path.isdir(trial_label_dir):
                    continue

                for minute in os.listdir(trial_label_dir):
                    minute_i = int(minute)
                    minute_dir = os.path.join(trial_label_dir, minute)
                    full_minute_spikes_input = {}

                    # build inputs by bands/channels; SKIP empty bands
                    for band_name, (lf, hf) in bands.items():
                        for ch_name in os.listdir(minute_dir):
                            ch_dir = os.path.join(minute_dir, ch_name)
                            if not os.path.isdir(ch_dir): 
                                continue

                            # collect spikes across all clk dirs, but resample/expand only if != sample_fs
                            band_dict = {}
                            for clk_freq_str in os.listdir(ch_dir):
                                if not clk_freq_str.isdigit():
                                    continue
                                clk_freq = int(clk_freq_str)
                                freq_dir = os.path.join(ch_dir, clk_freq_str)
                                if not os.path.isdir(freq_dir):
                                    continue

                                # filter f0 files that fall into the band
                                f0_files = sorted(os.listdir(freq_dir))
                                for f0_str in f0_files:
                                    try:
                                        f0 = float(Path(f0_str).stem)
                                    except Exception:
                                        continue
                                    if not (lf < f0 <= hf):
                                        continue

                                    arr = np.load(os.path.join(freq_dir, f0_str))["spikes"].astype(np.int8)

                                    # drop warm-up seconds
                                    cut = skip_beginning_s * clk_freq
                                    if cut >= arr.size:
                                        continue
                                    arr = arr[cut:]

                                    # upsample/expand to sample_fs if needed
                                    if clk_freq != sample_fs:
                                        # integer stride required here (we only expect sample_fs to be multiple of clk_freq)
                                        stride = sample_fs // clk_freq
                                        if stride * clk_freq != sample_fs:
                                            # safety: fallback nearest-upsample with zeros at nearest indices
                                            up_len = (60 - skip_beginning_s) * sample_fs
                                            up = np.zeros(up_len, dtype=np.int8)
                                            up_idx = np.arange(0, up_len, max(1, int(round(sample_fs/clk_freq))))[:arr.size]
                                            up[up_idx] = arr[:up_idx.size]
                                            arr = up
                                        else:
                                            up_len = (60 - skip_beginning_s) * sample_fs
                                            up = np.zeros(up_len, dtype=np.int8)
                                            up[::stride] = arr[:up[::stride].shape[0]]
                                            arr = up

                                    band_dict[f0] = arr
                                    pbar.update(1)

                            # if no f0 ended up in this band for this channel, SKIP this input_name
                            if band_dict:
                                full_minute_spikes_input[f"{ch_name}-{band_name}"] = band_dict

                    # If nothing to stack for this minute, skip quietly
                    if not full_minute_spikes_input:
                        continue

                    # window over the minute (consistent sample_fs)
                    for sample_start_index in range(
                        sample_fs,  # skip first second as before
                        sample_fs * (60 - skip_beginning_s) - spikes_in_sample,
                        step,
                    ):
                        # build tensor per input_name; SKIP any input that has no arrays (safety check)
                        sample_spikes_input = {}
                        for input_name, resonators_arrays in full_minute_spikes_input.items():
                            if not resonators_arrays:
                                continue
                            stacks = [
                                s[sample_start_index: sample_start_index + spikes_in_sample]
                                for s in resonators_arrays.values()
                            ]
                            # if no stacks (shouldn't happen due to band_dict check), skip this input
                            if len(stacks) == 0:
                                continue
                            sample_spikes_input[input_name] = torch.from_numpy(np.stack(stacks))

                        # if everything was empty for this window, move on
                        if not sample_spikes_input:
                            continue

                        file_name = f"{(sample_start_index / sample_fs) + minute_i * 60}"
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

    for subject in subjects:
        subj_dir = os.path.join(CHBMIT_ROOT, subject)
        summary_path = os.path.join(subj_dir, f"{subject}-summary.txt")
        if not os.path.isfile(summary_path):
            raise FileNotFoundError(f"Missing summary: {summary_path}")

        sampling_rate, ch_order, seizures_by_file, ch_by_idx = parse_summary(summary_path)
        if sampling_rate != fs:
            print(f"Warning: summary says fs={sampling_rate} but using fs={fs}")

        # ⬇️ NEW: convert desired indexes → labels
        if SELECT_BIPOLAR_INDEXES is not None:
            desired_labels = [ch_by_idx.get(i) for i in SELECT_BIPOLAR_INDEXES]
            desired_labels = [lab for lab in desired_labels if lab]  # drop None / missing
            if not desired_labels:
                raise ValueError("None of the requested channel indexes exist for this subject.")
        else:
            desired_labels = ch_order[:]  # all available

        edf_files = sorted([f for f in os.listdir(subj_dir) if f.lower().endswith(".edf")])
        if EDF_WHITELIST:
            edf_files = [f for f in edf_files if f in EDF_WHITELIST]
        if not edf_files:
            print(f"No EDF files for {subject}, skipping.")
            continue

        total = len(edf_files) * (len(desired_labels) or 1) * sum(len(v) for v in clk_resonators.values())
        with tqdm(total=total, desc=f"Subject {subject} (spikes)") as pbar:
            for edf_file in edf_files:
                process_file(subject, edf_file, desired_labels, seizures_by_file)
                pbar.update(1)

    create_datasets(
        time_of_sample_s=0.25,
        overlap=0,
        subjects=subjects,
        output_path=OUTPUT_PATH,
        skip_beginning_s=5,
    )

if __name__ == "__main__":
    main()