import sys

# Force UTF-8 on stdout/stderr so we don't crash on the Hebrew-locale Windows
# console (cp1255) when printing unusual characters.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

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

# Numba fast path (~1000x faster than the pure-Python resonator, bit-identical).
try:
    import sctn_fast as _sctn_fast
    USE_FAST_KERNEL = True
except Exception:
    _sctn_fast = None
    USE_FAST_KERNEL = False

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
# Data lives under PIPELINE_DATA_ROOT if set (e.g. a Google Drive folder in Colab),
# else under the repo. Must contain dataset/chbmit and dataset/spikes_raw.
_data_root = _Path(_os.environ.get("PIPELINE_DATA_ROOT", str(_project_root)))

CHBMIT_ROOT = str(_data_root / "dataset" / "chbmit")
# SPIKES_ROOT env: alternate spike folder (e.g. a re-encoding with another input scaling).
SPIKES_ROOT = _os.environ.get("SPIKES_ROOT") or str(_data_root / "dataset" / "spikes_raw")
# Input scaling before the resonators: fixed full-scale in microvolts with
# clipping, like an ADC input range. 0 = per-minute peak (the original behaviour,
# which removes the amplitude difference between seizure and non-seizure minutes).
INPUT_FULL_SCALE_UV = float(_os.environ.get("INPUT_FULL_SCALE_UV", 100))
OUTPUT_PATH = str(_data_root / "dataset" / "spikes_windowed")

fs = 256  # Hz

# ======== STAGED FREQUENCY CONFIGURATION ========
# CRITICAL: Clock frequencies chosen to match SCTN parameter files
# Formula: configured_freq = freq0 * 1_536_000 / clk_freq
# Example: 2.88 Hz * 1_536_000 / 15_360 = 288 → uses f_288.json
# Example: 10.5 Hz * 1_536_000 / 153_600 = 105 → uses f_105.json

# Full SA-paper frequency set across Delta/Theta/Alpha/Beta/Gamma.
# This drives the ENCODING step only — every channel gets the full 35-resonator
# bank on disk so any band can be re-enabled later without re-encoding.
# The WINDOWING / TRAINING step filters bands via the `bands` dict in
# create_datasets and BANDS in 3.run_train_chbmit.py (Gamma currently disabled).
# Below 10 Hz → slow clock (15.36 kHz); 10 Hz and above → fast clock (153.6 kHz).
# SCTN's resonator() auto-selects the closest pre-trained parameter file for each f0.
clk_resonators = {
    15360: [
        # Delta (0.5-4 Hz): 10 resonators
        1.1, 1.3, 1.6, 1.9, 2.2, 2.5, 2.88, 3.05, 3.39, 3.7,
        # Theta (4-8 Hz): 8 resonators
        4.12, 4.62, 5.09, 5.45, 5.87, 6.36, 6.8, 7.6,
        # Alpha low: 8.6 Hz
        8.6,
    ],
    153600: [
        # Alpha high (10-14): 3 resonators
        10.5, 11.5, 12.8,
        # Beta (14-32): 6 resonators
        16.6, 19.4, 22.0, 24.8, 28.4, 30.5,
        # Gamma (32-62): 7 resonators.
        34.7, 37.2, 40.2, 43.2, 47.7, 52.6, 57.2,
    ],
}

SUBJECTS = [f"chb{n:02d}" for n in range(1, 25)]   # all CHB-MIT patients
EDF_WHITELIST = []                # [] → all EDFs (further filtered by SEIZURE_ONLY)
SEIZURE_ONLY = True               # when EDF_WHITELIST empty, keep only seizure recordings
SEIZURE_FREE = os.environ.get("SEIZURE_FREE", "0") == "1"   # instead: only seizure-free recordings
FREE_MINUTES = int(os.environ.get("FREE_MINUTES", 2))       # minutes taken from a seizure-free recording
# FREE_SEED set: per-recording random minutes, never the ones the default draw takes.
FREE_SEED = os.environ.get("FREE_SEED")
# PERI=1: encode only the guard-band minutes around seizures (label "peri"), for testing near seizures.
PERI = os.environ.get("PERI", "0") == "1"
PERI_CHUNK = 3                    # guard minutes encoded per pass (bounds memory)

# Notebook/CLI override: PIPELINE_SUBJECTS="chb01,chb02" scopes the run.
if os.environ.get("PIPELINE_SUBJECTS"):
    SUBJECTS = [s.strip() for s in os.environ["PIPELINE_SUBJECTS"].split(",") if s.strip()]

# ======== SEIZURE-AWARE MINUTE SELECTION ========
# - Encode every minute that contains any seizure activity ("seizure" pool).
# - Encode a subsampled set of minutes that are >= GUARD_SECONDS from every
#   seizure boundary ("non_seizure" pool).
# - Subsample non-seizure minutes to match the seizure-minute count per EDF.
# Pre/post-ictal minutes (inside the guard band but not in a seizure) are excluded.
GUARD_SECONDS = 5 * 60            # 5-minute guard band around each seizure
SAMPLING_SEED = 42                # determinism for non-seizure subsampling

# ======== WINDOWING ========
# Shoeb 2010 convention: 2-second windows with 50% overlap (step = 1 s).
# Labels are assigned per window based on overlap with seizure intervals.
WINDOW_SECONDS = 2.0
WINDOW_OVERLAP = 0.5

# Channels encoded together per process pool. >1 overlaps channels to saturate
# all cores; capped to keep peak RAM safe on a 16 GB machine.
CHANNELS_PER_BATCH = 2

# Fast-path (Numba) parallelism: encode this many recordings concurrently, each
# in its own process. Recordings are independent, so this is a clean fan-out.
FAST_MAX_WORKERS = int(os.environ.get("ENCODE_WORKERS", 0)) or min(8, max(1, (os.cpu_count() or 2) - 2))

# ======== BIPOLAR CHANNEL SELECTION ========
# 10-channel set used for verification on the sa-aligned branch.
SELECT_BIPOLAR_CHANNELS = [
    "FT9-FT10", "T7-P7", "T8-P8", "P7-O1",
    "F7-T7", "P8-O2", "CZ-PZ", "P3-O1",
    "FP1-F7", "C4-P4",
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


def _resample_channel(ch_data, minute_labels, n_samples):
    """{clk_freq: {(label, minute): resampled_signal}} for one channel.

    Scaled to [-1, 1] (per-minute peak, or fixed full-scale if INPUT_FULL_SCALE_UV)
    then resampled to each clock rate.
    """
    resampled_by_clk = {}
    for clk_freq in clk_resonators.keys():
        resampled_by_minute = {}
        for label, minute_list in minute_labels.items():
            for minute in minute_list:
                start_sample = fs * 60 * minute
                stop_sample = min(fs * 60 * (minute + 1), n_samples)
                if stop_sample <= start_sample:
                    continue
                seg = ch_data[start_sample:stop_sample]
                if INPUT_FULL_SCALE_UV > 0:
                    seg = np.clip(seg / (INPUT_FULL_SCALE_UV * 1e-6), -1.0, 1.0)
                else:
                    seg_peak = np.max(np.abs(seg))
                    if seg_peak > 0:
                        seg = seg / seg_peak
                rs = resample_signal(clk_freq, fs, seg)
                if rs.size > 0:
                    resampled_by_minute[(label, minute)] = rs
        if resampled_by_minute:
            resampled_by_clk[clk_freq] = resampled_by_minute
    return resampled_by_clk

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
            # Starts-with handles the "T8-P8-0"/"T8-P8-1" suffixed duplicates.
            # (A looser substring match is intentionally NOT used: it maps bipolar
            # names onto single-electrode referential montages, e.g. chb12's
            # "FP1"/"F7", producing garbage or duplicate columns.)
            if avail_normalized.startswith(desired_normalized) and avail not in matched_channels:
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

    df = pd.DataFrame(data, columns=column_names)
    if df.columns.duplicated().any():          # safety: never let df[ch] be 2-D
        df = df.loc[:, ~df.columns.duplicated()]
    return df


def get_minutes_with_labels(n_samples: int, fs_: int, seizure_intervals_sec, *, rng=None, name=""):
    """
    Select which minutes to encode and label each as seizure / non_seizure.

    Scheme (Shoeb 2010-style):
      - "seizure"     : every minute overlapping any seizure interval.
      - "non_seizure" : interictal minutes (>= GUARD_SECONDS from every seizure
                        boundary), subsampled to match the seizure-minute count.
      - Excluded     : minutes inside the guard band (pre/post-ictal).

    Args:
        n_samples: total samples in the EDF
        fs_: EEG sampling rate (Hz)
        seizure_intervals_sec: list of (start, end) seizure intervals, in seconds
        rng: optional numpy Generator (for reproducible subsampling)

    Returns:
        Dict with 'seizure' and 'non_seizure' lists of minute indices.
    """
    max_minutes = n_samples // (fs_ * 60)
    if rng is None:
        rng = np.random.default_rng(SAMPLING_SEED)

    seizure_minutes = []
    interictal_minutes = []
    guard_minutes = []
    for m in range(max_minutes):
        m_start, m_end = m * 60, (m + 1) * 60
        in_seizure = any(
            (m_end > s) and (m_start < e) for (s, e) in seizure_intervals_sec
        )
        if in_seizure:
            seizure_minutes.append(m)
            continue
        in_guard = any(
            (m_end > s - GUARD_SECONDS) and (m_start < e + GUARD_SECONDS)
            for (s, e) in seizure_intervals_sec
        )
        if in_guard:
            guard_minutes.append(m)   # pre/post-ictal — excluded unless PERI
            continue
        interictal_minutes.append(m)

    # Per-EDF class balance: subsample non-seizure minutes to match seizure count.
    if seizure_minutes and interictal_minutes:
        k = min(len(seizure_minutes), len(interictal_minutes))
        chosen = rng.choice(interictal_minutes, size=k, replace=False).tolist()
        chosen_non_seizure = sorted(int(m) for m in chosen)
    elif not seizure_minutes:
        # No seizure in this EDF: keep a small interictal sample for negative-only training
        k = min(2, len(interictal_minutes))
        chosen = rng.choice(interictal_minutes, size=k, replace=False).tolist() if k else []
        if FREE_SEED is None:
            k = min(FREE_MINUTES, len(interictal_minutes))
            if k != len(chosen):
                chosen = np.random.default_rng(SAMPLING_SEED).choice(
                    interictal_minutes, size=k, replace=False).tolist() if k else []
        else:
            import zlib
            pool = [m for m in interictal_minutes if m not in set(chosen)]
            k = min(FREE_MINUTES, len(pool))
            r2 = np.random.default_rng([int(FREE_SEED), zlib.crc32(name.encode())])
            chosen = r2.choice(pool, size=k, replace=False).tolist() if k else []
        chosen_non_seizure = sorted(int(m) for m in chosen)
    else:
        chosen_non_seizure = []

    result = {
        "seizure": seizure_minutes,
        "non_seizure": chosen_non_seizure,
    }

    if PERI:
        print(f"  Guard-band minutes ({len(guard_minutes)}): {guard_minutes}")
        return {"peri": guard_minutes}

    if seizure_intervals_sec:
        print(f"  Seizure intervals (seconds): {seizure_intervals_sec}")
    print(f"  Seizure minutes ({len(result['seizure'])}): {result['seizure']}")
    print(f"  Non-seizure minutes ({len(result['non_seizure'])}, "
          f"sampled from {len(interictal_minutes)} interictal): {result['non_seizure']}")

    return result

# ============================================================
#                SPIKE GENERATION PIPELINE
# ============================================================

def resonator_fname_to_freq(name: str) -> float:
    return float(Path(name).stem)


def _process_file_fast(subject, edf_file, df, minute_labels, n_samples, log_every=5):
    """Numba fast path: encode all (channel, clk, f0) resonators serially via
    sctn_fast (bit-identical to the pure-Python resonator, ~1000x faster, so no
    process pool needed). Warms up then runs a channel's minutes in order,
    carrying state — same contract as _run_one_resonator. Writes only missing
    minute files."""
    edf_base = Path(edf_file).stem
    channels = list(df.columns)
    total = len(channels) * sum(len(v) for v in clk_resonators.values())
    done = wrote = 0
    t0 = time.time()
    order_all = [(lab, mn) for lab, lst in minute_labels.items() for mn in lst]
    for ch in channels:
        # Fast resume: skip resampling if every resonator file for this channel exists.
        ch_paths = [Path(SPIKES_ROOT) / f"{subject}/{edf_base}" / lab / str(mn)
                    / ch / str(clk_freq) / f"{f0}.npz"
                    for clk_freq, res in clk_resonators.items() for f0 in res
                    for (lab, mn) in order_all]
        if ch_paths and all(p.exists() for p in ch_paths):
            done += sum(len(v) for v in clk_resonators.values())
            continue
        rbclk = _resample_channel(df[ch].values, minute_labels, n_samples)
        for clk_freq, resonators in clk_resonators.items():
            rbm = rbclk.get(clk_freq)
            if not rbm:
                done += len(resonators)
                continue
            order = list(rbm.keys())                       # (label, minute) order
            signals = [rbm[k] for k in order]
            for f0 in resonators:
                done += 1
                paths = [Path(SPIKES_ROOT) / f"{subject}/{edf_base}" / lab / str(mn)
                         / ch / str(clk_freq) / f"{f0}.npz" for (lab, mn) in order]
                if all(p.exists() for p in paths):
                    continue
                outs = _sctn_fast.resonate_minutes(f0, clk_freq, signals, warmup_seconds=5)
                for p, out in zip(paths, outs):
                    if not p.exists():
                        p.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(p, spikes=out)
                        wrote += 1
                if done % (log_every * 5) == 0:
                    print(f"    {done:>4}/{total} resonators | wrote {wrote} | "
                          f"{time.time() - t0:6.1f}s")
    print(f"    [fast] {done}/{total} resonators | wrote {wrote} | "
          f"{time.time() - t0:.1f}s")


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
    minute_labels = get_minutes_with_labels(n_samples, fs, seizure_intervals, name=edf_file)

    n_seiz = len(minute_labels.get("seizure", []))
    n_non = len(minute_labels.get("non_seizure", []))
    print(f"\n>>> {subject}/{edf_file} | samples={n_samples:,} | minutes: seizure={n_seiz} non_seizure={n_non}")

    if USE_FAST_KERNEL:
        if PERI:
            mins = minute_labels["peri"]
            for i in range(0, len(mins), PERI_CHUNK):
                _process_file_fast(subject, edf_file, df, {"peri": mins[i:i + PERI_CHUNK]},
                                   n_samples, log_every)
        else:
            _process_file_fast(subject, edf_file, df, minute_labels, n_samples, log_every)
        return

    # Encode channels in batches so the process pool stays saturated: each
    # channel yields only ~9-19 fast-clock jobs, fewer than the core count, so
    # running one channel at a time leaves cores idle. Jobs stay per
    # (channel, clk, f0) — one resonator processes all its minutes in order,
    # carrying membrane state across minutes exactly as the serial code did.
    maxw = max(1, (os.cpu_count() or 2) - 1)
    channels = list(df.columns)

    for b in range(0, len(channels), CHANNELS_PER_BATCH):
        batch = channels[b:b + CHANNELS_PER_BATCH]
        print(f"  [Channels] {batch}")

        resampled = {
            ch: _resample_channel(df[ch].values, minute_labels, n_samples)
            for ch in batch
        }

        with ProcessPoolExecutor(max_workers=maxw) as ex:
            futs = {}
            for ch in batch:
                for clk_freq, resonators in clk_resonators.items():
                    rbm = resampled[ch].get(clk_freq)
                    if rbm is None:
                        continue
                    for f0 in resonators:
                        fut = ex.submit(
                            _run_one_resonator,
                            (subject, edf_file, ch, clk_freq, f0, rbm, SPIKES_ROOT),
                        )
                        futs[fut] = (ch, clk_freq, f0)

            total = len(futs)
            done = wrote_total = 0
            t0 = time.time()
            for fut in as_completed(futs):
                _, _, _, wrote, _ = fut.result()
                done += 1
                wrote_total += wrote
                if done % log_every == 0 or done == total:
                    el = time.time() - t0
                    print(f"    {done:>3}/{total} resonators | wrote {wrote_total} | "
                          f"{el:6.1f}s elapsed")

        del resampled
        print()


def _load_seizures_for_subjects(subjects):
    """Re-parse each subject's summary to map edf_stem (e.g. 'chb01_03') -> [(start_s, end_s), ...].

    Keys are normalized to the file stem (no .edf extension) to match the
    directory naming used by `_run_one_resonator` and `create_datasets`.
    """
    seizures_by_edf = {}
    for subj in subjects:
        summary_path = os.path.join(CHBMIT_ROOT, subj, f"{subj}-summary.txt")
        if not os.path.isfile(summary_path):
            continue
        _, _, seizures_by_file, _ = parse_summary(summary_path)
        for edf_name, intervals in seizures_by_file.items():
            seizures_by_edf[Path(edf_name).stem] = list(intervals)
    return seizures_by_edf


def _overlaps_any_seizure(start_s: float, end_s: float, intervals):
    return any((end_s > s) and (start_s < e) for (s, e) in intervals)


def enumerate_windows(
    subjects,
    time_of_sample_s: float = WINDOW_SECONDS,
    overlap: float = WINDOW_OVERLAP,
    skip_beginning_s: float = 0.0,
    sample_fs: int = 153_600,
):
    """Yield one descriptor per training window without writing anything to disk.

    Identical labelling logic to `create_datasets`, but returns metadata so the
    training loop can lazy-load each window from `spikes_raw/` on demand.

    Each yielded dict has:
        subject, trial            — folder names under spikes_raw
        minute_label              — "seizure" or "non_seizure" (the parent minute pool)
        minute                    — int, minute index in the EDF
        sec_offset_in_minute      — float seconds, start of the window within the minute
        abs_start_s, abs_end_s    — absolute seconds in the EDF
        window_label              — "seizure" or "non_seizure" (per-window relabel)
    """
    spikes_root = SPIKES_ROOT
    seizures_by_edf = _load_seizures_for_subjects(subjects)

    spikes_in_sample = int(sample_fs * time_of_sample_s)
    step = int(spikes_in_sample * (1 - overlap))
    usable_samples = int((60 - skip_beginning_s) * sample_fs)

    for subj in subjects:
        subj_dir = os.path.join(spikes_root, subj)
        if not os.path.isdir(subj_dir):
            continue
        for trial in sorted(os.listdir(subj_dir)):
            trial_dir = os.path.join(subj_dir, trial)
            if not os.path.isdir(trial_dir):
                continue
            seizure_intervals = seizures_by_edf.get(Path(trial).stem, [])
            for minute_label in ("seizure", "non_seizure"):
                trial_label_dir = os.path.join(trial_dir, minute_label)
                if not os.path.isdir(trial_label_dir):
                    continue
                for minute_str in sorted(os.listdir(trial_label_dir), key=int):
                    minute = int(minute_str)
                    for sample_start in range(
                        0, usable_samples - spikes_in_sample + 1, step
                    ):
                        sec_offset = skip_beginning_s + sample_start / sample_fs
                        abs_start = minute * 60 + sec_offset
                        abs_end = abs_start + time_of_sample_s
                        if _overlaps_any_seizure(abs_start, abs_end, seizure_intervals):
                            window_label = "seizure"
                        elif minute_label == "non_seizure":
                            window_label = "non_seizure"
                        else:
                            continue
                        yield {
                            "subject": subj,
                            "trial": trial,
                            "minute_label": minute_label,
                            "minute": minute,
                            "sec_offset_in_minute": sec_offset,
                            "abs_start_s": abs_start,
                            "abs_end_s": abs_end,
                            "window_label": window_label,
                        }


def create_datasets(time_of_sample_s, overlap, subjects, output_path, skip_beginning_s=0):
    """
    Build classifier-ready windows from per-minute spike trains.

    For each fixed-length window:
      - "seizure"     if the window's absolute time range overlaps any seizure
                      interval in the EDF.
      - "non_seizure" if the window lives inside a minute that was put in the
                      `non_seizure` pool by `get_minutes_with_labels` (i.e.
                      interictal, >= GUARD_SECONDS from any seizure).
      - otherwise the window is **discarded** (e.g. the non-seizure part of a
        minute that straddles the seizure boundary). This keeps pre/post-ictal
        signal out of the negative class.

    Args:
        time_of_sample_s: Window size in seconds (e.g., 2.0).
        overlap: Overlap fraction (e.g., 0.5 for 50% overlap).
        subjects: List of subjects whose spike trees should be windowed.
        output_path: Output directory for windowed data.
        skip_beginning_s: Skip first N seconds of each minute (warmup).
    """
    spikes_root = SPIKES_ROOT
    seizures_by_edf = _load_seizures_for_subjects(subjects)

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
            # trial looks like "chb01/chb01_03.edf"; pull the EDF basename for seizure lookup
            edf_name = Path(trial).name
            seizure_intervals = seizures_by_edf.get(edf_name, [])
            for minute_label in ["seizure", "non_seizure"]:
                trial_label_dir = os.path.join(trial_dir, minute_label)
                if not os.path.isdir(trial_label_dir):
                    continue

                for minute in os.listdir(trial_label_dir):
                    minute_i = int(minute)
                    minute_dir = os.path.join(trial_label_dir, minute)
                    full_minute_spikes_input = {}

                    # Restrict windowing to the configured channel subset so that
                    # any extra channels left over on disk from earlier runs are
                    # ignored (otherwise we'd accidentally include them).
                    allowed_channels = set(SELECT_BIPOLAR_CHANNELS)

                    # build inputs by bands/channels; SKIP empty bands
                    for band_name, (lf, hf) in bands.items():
                        for ch_name in os.listdir(minute_dir):
                            if ch_name not in allowed_channels:
                                continue
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

                        # File name = absolute time in seconds (start of the window)
                        absolute_start_s = minute_i * 60 + skip_beginning_s + (sample_start_index / sample_fs)
                        absolute_end_s = absolute_start_s + time_of_sample_s

                        # Per-window relabel:
                        #   "seizure"     if the window overlaps any seizure interval
                        #   "non_seizure" if the parent minute was selected as interictal
                        #   else          discard (pre/post-ictal contamination)
                        overlaps_seizure = _overlaps_any_seizure(
                            absolute_start_s, absolute_end_s, seizure_intervals
                        )
                        if overlaps_seizure:
                            window_label = "seizure"
                        elif minute_label == "non_seizure":
                            window_label = "non_seizure"
                        else:
                            continue  # seizure-minute window that's not actually ictal

                        file_name = f"{absolute_start_s:.3f}"
                        out_path = Path(output_path) / trial / window_label / f"{file_name}.npz"
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        if not out_path.is_file():
                            with open(out_path, "wb") as f:
                                pickle.dump(sample_spikes_input, f, protocol=pickle.HIGHEST_PROTOCOL)

# ============================================================
#                    MAIN DRIVER
# ============================================================

def _encode_one_recording(args):
    """Top-level worker (picklable). Returns (subject, edf, error_or_None) so a
    single bad recording (e.g. an incompatible montage) is skipped, not fatal."""
    subject, edf_file, channels, seizures_by_file = args
    try:
        process_file(subject, edf_file, channels, seizures_by_file)
        return (subject, edf_file, None)
    except Exception as ex:
        return (subject, edf_file, f"{type(ex).__name__}: {ex}")


def _prewarm_numba():
    """Trigger @njit compilation once in the parent so workers load from cache."""
    if not USE_FAST_KERNEL:
        return
    for clk, f0 in [(15360, 2.5), (153600, 22.0)]:
        _sctn_fast.resonate_minutes(f0, clk, [np.zeros(clk, dtype=np.float64)], warmup_seconds=0)


def main():
    subjects = [s for s in SUBJECTS if os.path.isdir(os.path.join(CHBMIT_ROOT, s))]
    if not subjects:
        raise RuntimeError(f"No subject folders found under {CHBMIT_ROOT}")

    print(f"Selected bipolar channels: {SELECT_BIPOLAR_CHANNELS}")

    # Build the recording work-list across all subjects.
    tasks = []
    for subject in subjects:
        subj_dir = os.path.join(CHBMIT_ROOT, subject)
        summary_path = os.path.join(subj_dir, f"{subject}-summary.txt")
        if not os.path.isfile(summary_path):
            print(f"Missing summary for {subject}, skipping.")
            continue
        _, _, seizures_by_file, _ = parse_summary(summary_path)
        edf_files = sorted(f for f in os.listdir(subj_dir) if f.lower().endswith(".edf"))
        if EDF_WHITELIST:
            edf_files = [f for f in edf_files if f in EDF_WHITELIST]
        elif SEIZURE_FREE:
            edf_files = [f for f in edf_files if not seizures_by_file.get(f)]
        elif SEIZURE_ONLY:
            edf_files = [f for f in edf_files if seizures_by_file.get(f)]
        for edf_file in edf_files:
            tasks.append((subject, edf_file, SELECT_BIPOLAR_CHANNELS, seizures_by_file))

    print(f"Recordings to encode: {len(tasks)} across {len(subjects)} subjects")
    if not tasks:
        print("Nothing to encode.")
        return

    skipped = []
    if USE_FAST_KERNEL and len(tasks) > 1:
        _prewarm_numba()
        print(f"Fast path: {FAST_MAX_WORKERS} parallel workers")
        with ProcessPoolExecutor(max_workers=FAST_MAX_WORKERS) as ex:
            futs = {ex.submit(_encode_one_recording, t): t for t in tasks}
            for i, fut in enumerate(as_completed(futs), 1):
                subj, edf, err = fut.result()
                if err:
                    skipped.append((subj, edf, err))
                    print(f"[{i}/{len(tasks)}] SKIPPED {subj}/{edf}: {err}", flush=True)
                else:
                    print(f"[{i}/{len(tasks)}] finished {subj}/{edf}", flush=True)
    else:
        for i, t in enumerate(tasks, 1):
            subj, edf, err = _encode_one_recording(t)
            if err:
                skipped.append((subj, edf, err))
                print(f"[{i}/{len(tasks)}] SKIPPED {subj}/{edf}: {err}", flush=True)

    if skipped:
        print(f"\n{len(skipped)} recording(s) skipped (incompatible montage / read error):")
        for subj, edf, err in skipped:
            print(f"  {subj}/{edf}: {err}")

    # Option B: training reads spikes_raw/ lazily; no need to pre-build a windowed
    # cache. Set BUILD_WINDOWED_CACHE=1 to keep the legacy behaviour (writes the
    # ~13 GB-per-EDF cache to OUTPUT_PATH).
    if os.environ.get("BUILD_WINDOWED_CACHE", "0") == "1":
        create_datasets(
            time_of_sample_s=WINDOW_SECONDS,
            overlap=WINDOW_OVERLAP,
            subjects=subjects,
            output_path=OUTPUT_PATH,
            skip_beginning_s=0,
        )
    else:
        n_windows = sum(1 for _ in enumerate_windows(
            subjects, WINDOW_SECONDS, WINDOW_OVERLAP
        ))
        print(f"Skipping windowed-cache build (set BUILD_WINDOWED_CACHE=1 to enable).")
        print(f"enumerate_windows() reports {n_windows} windows available for training.")

if __name__ == "__main__":
    main()