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

# Paths - handle running from either src/ or root directory
import os as _os
from pathlib import Path as _Path

_script_dir = _Path(__file__).parent
_project_root = _script_dir.parent

CHBMIT_ROOT = str(_project_root / "dataset" / "chbmit")
SPIKES_ROOT = str(_project_root / "dataset" / "spikes_raw")
OUTPUT_PATH = str(_project_root / "dataset" / "spikes_windowed")

fs = 256  # Hz

# ======== STAGED FREQUENCY CONFIGURATION ========
# CRITICAL: Clock frequencies chosen to match SCTN parameter files
# Formula: configured_freq = freq0 * 1_536_000 / clk_freq
# Example: 2.88 Hz * 1_536_000 / 15_360 = 288 → uses f_288.json
# Example: 10.5 Hz * 1_536_000 / 153_600 = 105 → uses f_105.json

# Complete frequency coverage for final results (run overnight)
clk_resonators = {
    15360: [1.1, 1.3, 1.6, 1.9, 2.2, 2.5, 2.88, 3.05, 3.39, 3.7,
            4.12, 4.62, 5.09, 5.45, 5.87, 6.36, 7.6],  # 17 frequencies (excluded 6.8, 8.6 due to large parameter errors)
    153600: [10.5, 11.5, 12.8, 15.8],  # 4 frequencies: Alpha band
}

SUBJECTS = ["chb01"]              # [] → process all subjects in CHBMIT_ROOT
EDF_WHITELIST = ["chb01_03.edf"]  # [] → include all EDF files under each subject

# ======== MINUTE-LEVEL PROCESSING (SA approach) ========
# SA paper uses 60-second (1 minute) windows for resonator processing.
# This allows proper frequency characterization (Delta needs multiple cycles).
#
# We use the exact seizure times from the summary file (e.g., 2996-3036 seconds)
# and determine which 60-second windows to process based on those times.
#
# Processing range: How many minutes before/after seizure to include
MINUTES_BEFORE_SEIZURE = 1  # Process 1 minute before seizure starts
MINUTES_AFTER_SEIZURE = 1   # Process 1 minute after seizure ends

# ======== BIPOLAR CHANNEL SELECTION ========
# Use bipolar channels directly from the EDF file (no reconstruction needed)
# These are the important bipolar pairs identified from feature analysis
SELECT_BIPOLAR_CHANNELS = [
    "FT9-FT10", "P8-O2", "T7-P7", "CZ-PZ", "T8-P8",
    "P3-O1", "P7-O1", "FP1-F7", "F7-T7", "C4-P4"
]


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


# --- worker that processes ONE (clk_freq, f0) over all MINUTES for a single channel ---
def _run_one_resonator(args):
    (subject, edf_file, ch_name, clk_freq, f0,
     resampled_by_minute, spikes_root) = args

    start_t = time.time()
    edf_base = Path(edf_file).stem
    resonator = spikes_resonator(freq0=f0, clk_freq=clk_freq)

    # Warmup with 5 seconds of zeros (SA approach)
    resonator.log_out_spikes(-1)
    out_neuron = resonator.layers_neurons[-1].neurons[-1]
    resonator.input_full_data(np.zeros(clk_freq * 5))
    out_neuron.forget_logs()

    wrote = 0
    for (label, minute), rs in resampled_by_minute.items():
        if rs is None or rs.size == 0:
            continue

        # Store by MINUTE (SA approach: 60-second windows)
        out_path = Path(spikes_root) / f"{subject}/{edf_base}" / label / str(minute) / ch_name / str(clk_freq) / f"{f0}.npz"

        if not out_path.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
            resonator.input_full_data(rs)
            # Output spikes for full 60 seconds (SA approach)
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
    r"^\s*File\s+Name:\s*(?P<fname>\S+)"
    r"|^\s*Number\s+of\s+Seizures\s+in\s+File:\s*(?P<n>\d+)"
    r"|^\s*Seizure\s+\d+\s+Start\s+Time:\s*(?P<start>\d+)\s*seconds?"
    r"|^\s*Seizure\s+\d+\s+End\s+Time:\s*(?P<end>\d+)\s*seconds?",
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

    # seizures per file - simple string-based parsing (more robust than regex)
    current_file = None
    current_start = None
    for ln in lines:
        # File name
        if 'File Name:' in ln and '.edf' in ln.lower():
            current_file = ln.split(':', 1)[1].strip()

        # Start time
        if 'Start Time:' in ln and 'seconds' in ln.lower() and current_file:
            try:
                # Extract number before 'seconds'
                current_start = int(ln.split('seconds')[0].split(':')[-1].strip())
            except (ValueError, IndexError):
                pass

        # End time
        if 'End Time:' in ln and 'seconds' in ln.lower() and current_file and current_start is not None:
            try:
                end = int(ln.split('seconds')[0].split(':')[-1].strip())
                if current_file not in seizures_by_file:
                    seizures_by_file[current_file] = []
                seizures_by_file[current_file].append((current_start, end))
                current_start = None
            except (ValueError, IndexError):
                pass

    return sampling_rate, channels_order, seizures_by_file, channels_by_index


def read_edf_channels_bipolar(edf_path: str, desired_bipolar_channels: list[str]) -> pd.DataFrame:
    """
    Read EDF file and extract bipolar channels directly.

    This is the simple, direct approach - no reconstruction needed.
    Bipolar montage is actually preferred for seizure detection because it:
    - Localizes activity to specific brain regions
    - Reduces common-mode noise and artifacts
    - Preserves seizure-specific signal characteristics

    Args:
        edf_path: Path to EDF file
        desired_bipolar_channels: List of bipolar channel names (e.g., ['FP1-F7', 'T7-P7'])

    Returns:
        DataFrame with bipolar channel data, normalized to [-1, 1]
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")

    # Get available channels from the EDF file
    available_channels = [ch.strip() for ch in raw.ch_names]

    # Match desired channels to available channels (case-insensitive, flexible matching)
    matched_channels = []
    channel_mapping = {}  # Maps actual name in file to desired name

    for desired in desired_bipolar_channels:
        # Normalize: uppercase, remove spaces, remove trailing numbers/suffixes
        desired_normalized = desired.upper().replace(" ", "").strip()
        found = False
        for avail in available_channels:
            # Normalize available channel name similarly
            avail_normalized = avail.upper().replace(" ", "").strip()
            # Check exact match first
            if desired_normalized == avail_normalized:
                matched_channels.append(avail)
                channel_mapping[avail] = desired
                found = True
                break
            # Check if available channel starts with desired (handles "T8-P8-0" matching "T8-P8")
            if avail_normalized.startswith(desired_normalized):
                matched_channels.append(avail)
                channel_mapping[avail] = desired
                found = True
                break
            # Check if desired is contained in available (handles variations)
            if desired_normalized in avail_normalized or avail_normalized in desired_normalized:
                matched_channels.append(avail)
                channel_mapping[avail] = desired
                found = True
                break
        if not found:
            print(f"  Warning: Channel '{desired}' not found in EDF file.")
            print(f"           Available channels: {available_channels[:5]}...")

    if len(matched_channels) == 0:
        raise ValueError(f"No matching bipolar channels found in {edf_path}. "
                        f"Available: {available_channels[:10]}...")

    print(f"  Matched {len(matched_channels)}/{len(desired_bipolar_channels)} channels")

    # Read the matched bipolar channels directly
    data = raw.get_data(picks=matched_channels).T  # (n_samples, n_channels)

    # Apply band-pass filter (0.5-62 Hz, same as SA code)
    data = bp_filter(data, 0.5, 62.0, fs)

    # NOTE: Do NOT normalize globally here!
    # Normalization will be done per-second in process_file() to preserve
    # amplitude differences between seizure and non-seizure segments.
    # Global normalization erases these differences and causes identical spike rates.

    # Use the original desired channel names for consistency
    column_names = [channel_mapping[ch] for ch in matched_channels]

    return pd.DataFrame(data, columns=column_names)


def get_minutes_with_labels(n_samples: int, fs_: int, seizure_intervals_sec):
    """
    Get minute-level labeling based on seizure intervals from summary file.

    Uses the SA paper approach: process full 60-second windows.
    A minute is labeled 'seizure' if it contains ANY seizure activity.

    Args:
        n_samples: Total samples in EDF
        fs_: Sampling rate
        seizure_intervals_sec: List of (start, end) seizure intervals from summary
                               e.g., [(2996, 3036)] means seizure from second 2996 to 3036

    Returns:
        Dict with 'seizure' and 'non_seizure' lists of minute indices
    """
    max_minutes = n_samples // (fs_ * 60)

    # Determine which minutes contain seizure activity based on intervals from summary
    seizure_minutes = set()
    for (seiz_start, seiz_end) in seizure_intervals_sec:
        # Find which minutes contain this seizure
        # seiz_start and seiz_end are in SECONDS from the summary file
        start_minute = seiz_start // 60
        end_minute = (seiz_end - 1) // 60 + 1  # Inclusive end
        for m in range(start_minute, min(end_minute, max_minutes)):
            seizure_minutes.add(m)

    # Determine non-seizure minutes: minutes before/after the seizure minutes
    non_seizure_minutes = set()
    if seizure_minutes:
        min_seiz_minute = min(seizure_minutes)
        max_seiz_minute = max(seizure_minutes)

        # Add minutes BEFORE the first seizure minute
        for m in range(max(0, min_seiz_minute - MINUTES_BEFORE_SEIZURE), min_seiz_minute):
            if m not in seizure_minutes and m < max_minutes:
                non_seizure_minutes.add(m)

        # Add minutes AFTER the last seizure minute
        for m in range(max_seiz_minute + 1, min(max_seiz_minute + 1 + MINUTES_AFTER_SEIZURE, max_minutes)):
            if m not in seizure_minutes:
                non_seizure_minutes.add(m)

    result = {
        "seizure": sorted(list(seizure_minutes)),
        "non_seizure": sorted(list(non_seizure_minutes)),
    }

    # Print info about seizure intervals (in seconds) and resulting minutes
    if seizure_intervals_sec:
        print(f"  Seizure intervals (seconds): {seizure_intervals_sec}")
    print(f"  Seizure minutes: {result['seizure']}")
    print(f"  Non-seizure minutes: {result['non_seizure']}")

    return result

# ============================================================
#                SPIKE GENERATION PIPELINE
# ============================================================

def resonator_fname_to_freq(name: str) -> float:
    return float(Path(name).stem)

def process_file(subject: str,
                 edf_file: str,
                 desired_bipolar_channels: list[str],
                 seizures_by_file: dict[str, list[tuple[int,int]]],
                 log_every:int = 5):
    """
    Generate spikes per MINUTE, per bipolar channel, per resonator for a single EDF file.

    Uses the SA paper approach:
    - Process full 60-second windows (not 1-second)
    - Normalize per-minute segment to [-1, 1]
    - Store spikes for entire minute continuously
    """

    edf_path = os.path.join(CHBMIT_ROOT, subject, edf_file)
    df = read_edf_channels_bipolar(edf_path, desired_bipolar_channels)
    n_samples = df.shape[0]

    seizure_intervals = seizures_by_file.get(edf_file, [])
    minute_labels = get_minutes_with_labels(n_samples, fs, seizure_intervals)

    n_seiz = len(minute_labels.get("seizure", []))
    n_non = len(minute_labels.get("non_seizure", []))
    print(f"\n>>> {subject}/{edf_file} | samples={n_samples:,} | minutes: seizure={n_seiz} non_seizure={n_non}")

    for ch in df.columns:
        ch_data = df[ch].values
        print(f"  [Channel] {ch}")

        # Prepare resampled data for ALL clock frequencies, per MINUTE
        resampled_by_clk = {}
        for clk_freq in clk_resonators.keys():
            resampled_by_minute = {}
            for label, minute_list in minute_labels.items():
                for minute in minute_list:
                    # Extract 60-second segment for this minute
                    start_sample = fs * 60 * minute
                    stop_sample = min(fs * 60 * (minute + 1), n_samples)
                    if stop_sample <= start_sample:
                        continue
                    seg = ch_data[start_sample:stop_sample]

                    # SA NORMALIZATION: Normalize per-minute to [-1, 1]
                    # This preserves relative frequency content within the minute
                    seg_peak = np.max(np.abs(seg))
                    if seg_peak > 0:
                        seg = seg / seg_peak

                    # Resample to clock frequency
                    rs = resample_signal(clk_freq, fs, seg)
                    if rs.size > 0:
                        resampled_by_minute[(label, minute)] = rs

            if len(resampled_by_minute) > 0:
                resampled_by_clk[clk_freq] = resampled_by_minute

        # Process ALL clock frequencies in parallel
        maxw = max(1, (os.cpu_count() or 2) - 1)

        with ProcessPoolExecutor(max_workers=maxw) as ex:
            all_futures = []
            clk_info = {}  # Track metadata for each clk_freq

            # Submit ALL f0 resonators across ALL clock frequencies
            for clk_freq, resonators in clk_resonators.items():
                if clk_freq not in resampled_by_clk:
                    print(f"    clk={clk_freq:>6} Hz | no usable minutes after resample; skipping.")
                    continue

                resampled_by_minute = resampled_by_clk[clk_freq]
                n_minutes = len(resampled_by_minute)
                total_f0 = len(resonators)

                clk_info[clk_freq] = {
                    'total_f0': total_f0,
                    'done': 0,
                    'wrote_total': 0,
                    't0': time.time()
                }

                print(f"    clk={clk_freq:>6} Hz | minutes_resampled={n_minutes} | f0_count={total_f0}")

                for f0 in resonators:
                    args = (
                        subject, edf_file, ch, clk_freq, f0,
                        resampled_by_minute,
                        SPIKES_ROOT,
                    )
                    fut = ex.submit(_run_one_resonator, args)
                    all_futures.append((fut, clk_freq))

            # Process results as they complete
            for fut, clk_freq in all_futures:
                clk_done, f0_done, ch_done, wrote, dur = fut.result()

                info = clk_info[clk_freq]
                info['done'] += 1
                info['wrote_total'] += wrote

                # Log progress for this clock frequency
                if (info['done'] % log_every == 0) or (info['done'] == info['total_f0']):
                    elapsed = time.time() - info['t0']
                    rate = info['done'] / elapsed if elapsed > 0 else 0.0
                    print(f"      [clk={clk_freq:>6} Hz] f0 {info['done']:>2}/{info['total_f0']} "
                          f"(last={f0_done:>5.2f} Hz, wrote {wrote} files) | "
                          f"elapsed {elapsed:5.1f}s | {rate:4.2f} f0/s")

        # Print completion summary for all clock frequencies
        for clk_freq in clk_info:
            print(f"    ✔ clk={clk_freq:>6} Hz done | wrote_files={clk_info[clk_freq]['wrote_total']}")
        print()


def create_datasets(time_of_sample_s, overlap, subjects, output_path, skip_beginning_s=0):
    """
    Build dataset windows from the generated per-MINUTE spikes (SA approach).

    Now works with minute-level storage: each minute has 60 seconds of spike data.
    Windows are extracted from the full 60-second spike arrays.

    Args:
        time_of_sample_s: Window size in seconds (e.g., 0.25)
        overlap: Overlap fraction (e.g., 0.5 for 50% overlap)
        subjects: List of subjects to process
        output_path: Output directory for windowed data
        skip_beginning_s: Skip first N seconds of each minute (warmup)
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
                        if not os.path.isdir(minute_dir):
                            continue
                        # find any channel dir
                        for ch in os.listdir(minute_dir):
                            ch_dir = os.path.join(minute_dir, ch)
                            if not os.path.isdir(ch_dir):
                                continue
                            clk_list = [int(x) for x in os.listdir(ch_dir) if x.isdigit()]
                            if clk_list:
                                return max(clk_list)
        # fallback to 153600 (highest freq now)
        return 153600

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
                if not os.path.isdir(minute_dir):
                    continue
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

                                    # drop warm-up (skip_beginning_s seconds)
                                    cut = skip_beginning_s * clk_freq
                                    if cut >= arr.size:
                                        continue
                                    arr = arr[cut:]

                                    # upsample/expand to sample_fs if needed
                                    if clk_freq != sample_fs:
                                        # integer stride required here (we only expect sample_fs to be multiple of clk_freq)
                                        stride = sample_fs // clk_freq
                                        target_len = (60 - skip_beginning_s) * sample_fs
                                        if stride * clk_freq != sample_fs:
                                            # safety: fallback nearest-upsample with zeros at nearest indices
                                            up = np.zeros(target_len, dtype=np.int8)
                                            up_idx = np.arange(0, target_len, max(1, int(round(sample_fs/clk_freq))))[:arr.size]
                                            up[up_idx] = arr[:up_idx.size]
                                            arr = up
                                        else:
                                            up = np.zeros(target_len, dtype=np.int8)
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

                    # Window over the full minute (60 seconds at sample_fs)
                    # Extract windows with overlap support
                    usable_samples = (60 - skip_beginning_s) * sample_fs
                    for sample_start_index in range(
                        0,  # start from beginning of minute (after warmup skip)
                        usable_samples - spikes_in_sample + 1,
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

                        # File name = absolute time in seconds
                        # minute_i * 60 gives the start of the minute in seconds
                        # sample_start_index / sample_fs gives offset within minute
                        absolute_time_s = minute_i * 60 + skip_beginning_s + (sample_start_index / sample_fs)
                        file_name = f"{absolute_time_s:.3f}"
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

        # Use bipolar channels directly (no reconstruction needed)
        desired_bipolar_channels = SELECT_BIPOLAR_CHANNELS
        if not desired_bipolar_channels:
            raise ValueError("No bipolar channels selected. Please set SELECT_BIPOLAR_CHANNELS.")

        print(f"Selected bipolar channels: {desired_bipolar_channels}")

        edf_files = sorted([f for f in os.listdir(subj_dir) if f.lower().endswith(".edf")])
        if EDF_WHITELIST:
            edf_files = [f for f in edf_files if f in EDF_WHITELIST]
        if not edf_files:
            print(f"No EDF files for {subject}, skipping.")
            continue

        total = len(edf_files) * len(desired_bipolar_channels) * sum(len(v) for v in clk_resonators.values())
        with tqdm(total=total, desc=f"Subject {subject} (spikes)") as pbar:
            for edf_file in edf_files:
                process_file(subject, edf_file, desired_bipolar_channels, seizures_by_file)
                pbar.update(1)

    create_datasets(
        time_of_sample_s=0.25,  # 250ms windows
        overlap=0.5,  # 50% overlap (125ms step) for better temporal resolution
        subjects=subjects,
        output_path=OUTPUT_PATH,
        skip_beginning_s=0,  # No skip needed for second-level processing
    )

if __name__ == "__main__":
    main()