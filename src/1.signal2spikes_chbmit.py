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

# ======== SECOND-LEVEL LABELING ========
# Define exact second ranges for processing (more precise than minute-level)
# For chb01_03.edf: Seizure is 2996-3036 seconds (40 seconds)

# STAGE 1: Quick validation (~15-20 min)
PROCESS_SECOND_RANGES = [
    (2996-20, 3036+20),  # 20s before, seizure (40s), 20s after = 80 seconds total
]

# ======== MONOPOLAR CHANNEL SELECTION ========
# Select important bipolar pairs (from feature analysis), then extract monopolar electrodes
# These will be average-referenced to create proper monopolar recordings
IMPORTANT_BIPOLAR_PAIRS = [
    "FT9-FT10", "P8-O2", "T7-P7", "CZ-PZ", "T8-P8",
    "P3-O1", "P7-O1", "FP1-F7", "F7-T7", "C4-P4"
]

# Automatically extract unique monopolar electrodes from the bipolar pairs
def extract_monopolar_electrodes(bipolar_pairs):
    """Extract unique electrode names from bipolar channel pairs."""
    electrodes = set()
    for pair in bipolar_pairs:
        parts = pair.split("-")
        if len(parts) == 2:
            electrodes.add(parts[0])
            electrodes.add(parts[1])
    return sorted(electrodes)

SELECT_MONOPOLAR_ELECTRODES = extract_monopolar_electrodes(IMPORTANT_BIPOLAR_PAIRS)
# Results in: ['C4', 'CZ', 'F7', 'FP1', 'FT10', 'FT9', 'O1', 'O2', 'P3', 'P4', 'P7', 'P8', 'PZ', 'T7', 'T8']


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
     seconds_by_label, resampled_by_second, spikes_root) = args

    start_t = time.time()
    edf_base = Path(edf_file).stem
    resonator = spikes_resonator(freq0=f0, clk_freq=clk_freq)

    # warmup
    resonator.log_out_spikes(-1)
    out_neuron = resonator.layers_neurons[-1].neurons[-1]
    resonator.input_full_data(np.zeros(clk_freq * 5))
    out_neuron.forget_logs()

    wrote = 0
    for label, second_list in seconds_by_label.items():
        for (sec_start, sec_end) in second_list:
            rs = resampled_by_second.get(sec_start)
            if rs is None or rs.size == 0:
                continue

            # Store by second instead of minute
            out_path = Path(spikes_root) / f"{subject}/{edf_base}" / label / str(sec_start) / ch_name / str(clk_freq) / f"{f0}.npz"

            if not out_path.exists():
                out_path.parent.mkdir(parents=True, exist_ok=True)
                resonator.input_full_data(rs)
                # Output spikes for 1 second (not 60 seconds)
                sp = out_neuron.out_spikes(False, clk_freq * 1).astype(np.int8)
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


def read_edf_channels_monopolar(edf_path: str, desired_monopolar_electrodes: list[str]) -> pd.DataFrame:
    """
    Read EDF file and create average-referenced monopolar channels.

    Args:
        edf_path: Path to EDF file
        desired_monopolar_electrodes: List of individual electrode names (e.g., ['P7', 'O1', 'FP1'])

    Returns:
        DataFrame with average-referenced monopolar channels
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")

    # Map electrode names to available channels (case-insensitive)
    # The CHB-MIT dataset has bipolar channels, so we need to extract individual electrodes
    all_bipolar_channels = [ch.strip() for ch in raw.ch_names]

    # Build a mapping of which bipolar channels contain each monopolar electrode
    electrode_to_bipolar = {}
    for electrode in desired_monopolar_electrodes:
        electrode_to_bipolar[electrode] = []
        for ch_name in all_bipolar_channels:
            # Match electrode in bipolar pair (e.g., "P7" in "P7-O1" or "FP1-P7")
            parts = ch_name.split("-")
            if len(parts) == 2:
                if parts[0].upper() == electrode.upper() or parts[1].upper() == electrode.upper():
                    electrode_to_bipolar[electrode].append(ch_name)

    # Get all unique bipolar channels needed
    needed_bipolar = set()
    for channels in electrode_to_bipolar.values():
        needed_bipolar.update(channels)
    needed_bipolar = sorted(needed_bipolar)

    if len(needed_bipolar) == 0:
        raise ValueError(f"No bipolar channels found containing desired electrodes in {edf_path}")

    # Read all needed bipolar channels
    bipolar_data = raw.get_data(picks=needed_bipolar).T  # (n_samples, n_bipolar_ch)
    bipolar_data = bp_filter(bipolar_data, 0.5, 62.0, fs)

    # Reconstruct monopolar electrode signals from bipolar pairs
    # For a bipolar channel A-B, we have: V(A-B) = V(A) - V(B)
    # We'll use a least-squares approach to estimate monopolar voltages
    n_samples = bipolar_data.shape[0]
    all_electrodes_in_bipolar = set()
    for ch in needed_bipolar:
        parts = ch.split("-")
        if len(parts) == 2:
            all_electrodes_in_bipolar.add(parts[0].upper())
            all_electrodes_in_bipolar.add(parts[1].upper())

    all_electrodes_in_bipolar = sorted(all_electrodes_in_bipolar)
    n_electrodes = len(all_electrodes_in_bipolar)
    electrode_idx = {e: i for i, e in enumerate(all_electrodes_in_bipolar)}

    # Build system of equations: bipolar = electrode1 - electrode2
    # A * x = b, where x are monopolar voltages
    A = np.zeros((len(needed_bipolar), n_electrodes))
    for i, ch in enumerate(needed_bipolar):
        parts = ch.split("-")
        if len(parts) == 2:
            e1, e2 = parts[0].upper(), parts[1].upper()
            if e1 in electrode_idx and e2 in electrode_idx:
                A[i, electrode_idx[e1]] = 1
                A[i, electrode_idx[e2]] = -1

    # Solve for monopolar voltages at each time point using least squares
    # Add constraint that average voltage = 0 (average reference)
    A_with_avg = np.vstack([A, np.ones((1, n_electrodes))])  # Add average constraint

    monopolar_signals = np.zeros((n_samples, n_electrodes))
    for t in range(n_samples):
        b = bipolar_data[t, :]
        b_with_avg = np.append(b, 0)  # Average should be 0
        x, _, _, _ = np.linalg.lstsq(A_with_avg, b_with_avg, rcond=None)
        monopolar_signals[t, :] = x

    # Extract only the desired monopolar electrodes
    keep_electrodes = []
    keep_indices = []
    for electrode in desired_monopolar_electrodes:
        if electrode.upper() in electrode_idx:
            keep_electrodes.append(electrode)
            keep_indices.append(electrode_idx[electrode.upper()])

    if len(keep_electrodes) == 0:
        raise ValueError(f"None of the desired electrodes could be reconstructed from {edf_path}")

    data = monopolar_signals[:, keep_indices]

    # Normalize each channel
    peak = np.max(np.abs(data), axis=0)
    peak[peak == 0] = 1.0
    data = data / peak

    return pd.DataFrame(data, columns=keep_electrodes)


def get_second_ranges_with_labels(n_samples: int, fs_: int, seizure_intervals_sec,
                                   process_ranges: list[tuple[int, int]]):
    """
    Create second-level ranges with seizure/non-seizure labels.

    Args:
        n_samples: Total samples in EDF
        fs_: Sampling rate
        seizure_intervals_sec: List of (start, end) seizure intervals from summary
        process_ranges: List of (start, end) second ranges to process

    Returns:
        Dict with 'seizure' and 'non_seizure' lists of (start_sec, end_sec) tuples
    """
    max_seconds = n_samples // fs_

    seizure_ranges = []
    non_seizure_ranges = []

    for (range_start, range_end) in process_ranges:
        # Clip to available data
        range_start = max(0, range_start)
        range_end = min(range_end, max_seconds)

        if range_start >= range_end:
            continue

        # Check each second in this range
        for sec in range(range_start, range_end):
            # Check if this second overlaps with any seizure interval
            is_seizure = False
            for (seiz_start, seiz_end) in seizure_intervals_sec:
                # A second is labeled seizure if it has ANY overlap with seizure interval
                if sec < seiz_end and (sec + 1) > seiz_start:
                    is_seizure = True
                    break

            if is_seizure:
                seizure_ranges.append((sec, sec + 1))
            else:
                non_seizure_ranges.append((sec, sec + 1))

    return {
        "seizure": seizure_ranges,
        "non_seizure": non_seizure_ranges,
    }

# ============================================================
#                SPIKE GENERATION PIPELINE
# ============================================================

def resonator_fname_to_freq(name: str) -> float:
    return float(Path(name).stem)

def process_file(subject: str,
                 edf_file: str,
                 desired_monopolar_electrodes: list[str],
                 seizures_by_file: dict[str, list[tuple[int,int]]],
                 log_every:int = 5):

    """
    Generate spikes per second, per monopolar channel, per resonator for a single EDF file.
    Now works with second-level precision instead of minute-level.
    Parallelizes across f0 for each (channel, clk_freq).
    """

    edf_path = os.path.join(CHBMIT_ROOT, subject, edf_file)
    df = read_edf_channels_monopolar(edf_path, desired_monopolar_electrodes)
    n_samples = df.shape[0]

    seizure_intervals = seizures_by_file.get(edf_file, [])
    second_ranges = get_second_ranges_with_labels(n_samples, fs, seizure_intervals, PROCESS_SECOND_RANGES)

    n_seiz = len(second_ranges.get("seizure", []))
    n_non  = len(second_ranges.get("non_seizure", []))
    print(f"\n>>> {subject}/{edf_file} | samples={n_samples:,} | seconds: seizure={n_seiz} non_seizure={n_non}")

    for ch in df.columns:
        ch_data = df[ch].values
        print(f"  [Channel] {ch}")

        # OPTIMIZATION: Prepare resampled data for ALL clock frequencies first
        resampled_by_clk = {}
        for clk_freq in clk_resonators.keys():
            resampled_by_second = {}
            for label, second_list in second_ranges.items():
                for (sec_start, sec_end) in second_list:
                    start_sample = fs * sec_start
                    stop_sample = min(fs * sec_end, n_samples)
                    if stop_sample <= start_sample:
                        continue
                    seg = ch_data[start_sample:stop_sample]
                    rs = resample_signal(clk_freq, fs, seg)
                    if rs.size > 0:
                        resampled_by_second[sec_start] = rs

            if len(resampled_by_second) > 0:
                resampled_by_clk[clk_freq] = resampled_by_second

        # OPTIMIZATION: Process ALL clock frequencies in parallel (not sequential)
        maxw = max(1, (os.cpu_count() or 2) - 1)

        with ProcessPoolExecutor(max_workers=maxw) as ex:
            all_futures = []
            clk_info = {}  # Track metadata for each clk_freq

            # Submit ALL f0 resonators across ALL clock frequencies
            for clk_freq, resonators in clk_resonators.items():
                if clk_freq not in resampled_by_clk:
                    print(f"    clk={clk_freq:>6} Hz | no usable seconds after resample; skipping.")
                    continue

                resampled_by_second = resampled_by_clk[clk_freq]
                n_seconds = len(resampled_by_second)
                total_f0 = len(resonators)

                clk_info[clk_freq] = {
                    'total_f0': total_f0,
                    'done': 0,
                    'wrote_total': 0,
                    't0': time.time()
                }

                print(f"    clk={clk_freq:>6} Hz | seconds_resampled={n_seconds} | f0_count={total_f0}")

                for f0 in resonators:
                    args = (
                        subject, edf_file, ch, clk_freq, f0,
                        second_ranges,
                        resampled_by_second,
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
    Build dataset windows from the generated per-second spikes.

    Now works with second-level precision and labels each window individually
    based on whether it overlaps with seizure intervals.

    Args:
        time_of_sample_s: Window size in seconds (e.g., 0.25)
        overlap: Overlap fraction (e.g., 0.5 for 50% overlap)
        subjects: List of subjects to process
        output_path: Output directory for windowed data
        skip_beginning_s: Skip first N seconds of each second (usually 0 for second-level)
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
                # find any second dir
                for label in ["seizure", "non_seizure"]:
                    lab_dir = os.path.join(trial_dir, label)
                    if not os.path.isdir(lab_dir):
                        continue
                    for second in os.listdir(lab_dir):
                        second_dir = os.path.join(lab_dir, second)
                        if not os.path.isdir(second_dir):
                            continue
                        # find any channel dir
                        for ch in os.listdir(second_dir):
                            ch_dir = os.path.join(second_dir, ch)
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
            for second in os.listdir(trial_label_dir):
                second_dir = os.path.join(trial_label_dir, second)
                if not os.path.isdir(second_dir):
                    continue
                for ch in os.listdir(second_dir):
                    ch_dir = os.path.join(second_dir, ch)
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

                for second in os.listdir(trial_label_dir):
                    second_i = int(second)
                    second_dir = os.path.join(trial_label_dir, second)
                    full_second_spikes_input = {}

                    # build inputs by bands/channels; SKIP empty bands
                    for band_name, (lf, hf) in bands.items():
                        for ch_name in os.listdir(second_dir):
                            ch_dir = os.path.join(second_dir, ch_name)
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

                                    # drop warm-up (usually 0 for second-level)
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
                                            up_len = (1 - skip_beginning_s) * sample_fs
                                            up = np.zeros(up_len, dtype=np.int8)
                                            up_idx = np.arange(0, up_len, max(1, int(round(sample_fs/clk_freq))))[:arr.size]
                                            up[up_idx] = arr[:up_idx.size]
                                            arr = up
                                        else:
                                            up_len = (1 - skip_beginning_s) * sample_fs
                                            up = np.zeros(up_len, dtype=np.int8)
                                            up[::stride] = arr[:up[::stride].shape[0]]
                                            arr = up

                                    band_dict[f0] = arr
                                    pbar.update(1)

                            # if no f0 ended up in this band for this channel, SKIP this input_name
                            if band_dict:
                                full_second_spikes_input[f"{ch_name}-{band_name}"] = band_dict

                    # If nothing to stack for this second, skip quietly
                    if not full_second_spikes_input:
                        continue

                    # window over the second (consistent sample_fs)
                    # Now with overlap support
                    for sample_start_index in range(
                        0,  # start from beginning of second
                        sample_fs * (1 - skip_beginning_s) - spikes_in_sample + 1,
                        step,
                    ):
                        # build tensor per input_name; SKIP any input that has no arrays (safety check)
                        sample_spikes_input = {}
                        for input_name, resonators_arrays in full_second_spikes_input.items():
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

                        # File name = absolute time in seconds (second + offset within second)
                        file_name = f"{second_i + (sample_start_index / sample_fs)}"
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

        # Use monopolar electrodes instead of bipolar channel indexes
        desired_monopolar_electrodes = SELECT_MONOPOLAR_ELECTRODES
        if not desired_monopolar_electrodes:
            raise ValueError("No monopolar electrodes selected. Please set SELECT_MONOPOLAR_ELECTRODES.")

        print(f"Selected monopolar electrodes: {desired_monopolar_electrodes}")

        edf_files = sorted([f for f in os.listdir(subj_dir) if f.lower().endswith(".edf")])
        if EDF_WHITELIST:
            edf_files = [f for f in edf_files if f in EDF_WHITELIST]
        if not edf_files:
            print(f"No EDF files for {subject}, skipping.")
            continue

        total = len(edf_files) * (len(desired_monopolar_electrodes) or 1) * sum(len(v) for v in clk_resonators.values())
        with tqdm(total=total, desc=f"Subject {subject} (spikes)") as pbar:
            for edf_file in edf_files:
                process_file(subject, edf_file, desired_monopolar_electrodes, seizures_by_file)
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