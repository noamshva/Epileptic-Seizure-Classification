# Epileptic-Seizure-Classification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/noamshva/Epileptic-Seizure-Classification/blob/main/src/run_pipeline.ipynb)

Seizure vs. non-seizure classification from CHB-MIT scalp EEG using a
**neuromorphic SCTN resonator front-end** (Spike Continuous-Time Neurons)
instead of classical FFT band-power features, classified by standard machine
learning (SVM / Random Forest). The resonator encoding is adapted from the
situational-awareness work of Hadad et al. (2024).

The project asks: *can a spike-based, multiplier-free front-end match classical
FFT features for seizure detection?* — and answers it with a leakage-free,
multi-patient comparison.

## Pipeline

```
EEG raw ──▶ SCTN resonator bank ──▶ burstiness features ──▶ classifier
 (256 Hz)    (per channel, per       (std of 10-ms binned     (SVM / RF)
             band; spike trains)      spike density → 40-D)
   1.signal2spikes        features.py            2.analyze / 3.run_classify
```

1. **Encode** — each channel is band-pass filtered and run through a bank of SCTN
   resonators tuned across EEG sub-bands; their spike trains are written to disk.
2. **Feature** — per 2-s window, the discriminative feature is *burstiness*
   (std of the 10-ms-binned spike density) per (channel, band) → a 40-D vector.
3. **Analyze / Classify** — visualize SCTN vs. FFT (paper Figs 4/6/7), and
   classify with an SVM / Random Forest under honest leave-one-recording-out CV.

## Repository layout

| File | Role |
|---|---|
| [src/1.signal2spikes_chbmit.py](src/1.signal2spikes_chbmit.py) | **Encode** EEG → per-resonator spike trains (uses the Numba fast path) |
| [src/features.py](src/features.py) | Shared feature extraction: SCTN burstiness (40-D band / 280-D resonator, ±Gamma) + FFT baselines (band-power / fine PSD) |
| [src/2.analyze_chbmit.py](src/2.analyze_chbmit.py) | **Analyze** — SCTN-vs-FFT spectrograms, per-channel grids, topography (mirrors the SA paper's Figs 4/6/7) for one example recording |
| [src/3.run_classify_chbmit.py](src/3.run_classify_chbmit.py) | **Classify** — SVM on burstiness features; per-patient LORO + stratified, confusion matrix |
| [src/3b.run_classify_rf_chbmit.py](src/3b.run_classify_rf_chbmit.py) | Same harness with a Random Forest |
| [src/sctn_fast.py](src/sctn_fast.py) | Numba `@njit` reimplementation of the resonator forward pass (~1000× faster, bit-identical) |
| [src/run_pipeline.ipynb](src/run_pipeline.ipynb) | One notebook driving encode → analyze → classify (local or Colab) |

*(An SNN classifier lives on a separate branch; `main` is the classical-classifier pipeline.)*

## Setup

```bash
python -m venv .venv && . .venv/Scripts/activate      # Windows; or bin/activate
pip install -r requirements.txt                       # sctn pulls numba/torch/numpy/... ; + mne, scikit-learn
```

The `sctn` package (installed via pip) provides the resonator neuron model.
`numba` (a transitive dependency of `sctn`) powers the fast encoder — no manual
setup required.

## Dataset

Download CHB-MIT from PhysioNet (<https://physionet.org/content/chbmit/1.0.0/>)
into `dataset/chbmit/`. The dataset and all encoded spikes are **git-ignored**,
so a fresh clone contains code only — you provide the data.

By default the scripts read/write under the repo. Set **`PIPELINE_DATA_ROOT`** to
put data + outputs elsewhere (e.g. a Google Drive folder in Colab); the folder
must contain `dataset/chbmit/` and `dataset/spikes_raw/`.

## Running

**Notebook (recommended, local or Colab):** open [src/run_pipeline.ipynb](src/run_pipeline.ipynb) —
or click the **Open in Colab** badge above (opens straight from GitHub; run the
Setup cell to clone + mount Drive). Set `DATA_ROOT` in the Config cell, run all
cells. It auto-discovers encoded patients: analysis illustrates one subject;
encoding + classification run on all.

**Command line:**

```bash
python src/1.signal2spikes_chbmit.py                        # encode (env PIPELINE_SUBJECTS scopes it)
python src/2.analyze_chbmit.py                              # SCTN-vs-FFT figures → analysis/
python src/3.run_classify_chbmit.py  chb01 chb02 ...        # SVM, per patient → output_data/results/
python src/3b.run_classify_rf_chbmit.py chb01 chb02 ...     # Random Forest
```

The numbered/dotted filenames run as scripts (`python src/1.signal2spikes_chbmit.py`);
cross-file imports use `importlib`, so nothing needs to be importable as a module.

## Configuration

| Parameter | Value |
|---|---|
| EEG sampling rate | 256 Hz (CHB-MIT) |
| Band-pass | 0.5–62 Hz, Butterworth order 6 |
| Channels | 10 CHB-MIT bipolar pairs |
| Bands | Delta, Theta, Alpha, Beta (Gamma encoded on disk; excluded from the active feature set) |
| Resonator clocks | < 10 Hz @ 15.36 kHz, ≥ 10 Hz @ 153.6 kHz (matches SCTN pre-trained params) |
| Window | 2 s, 50 % overlap (Shoeb 2010) |
| Feature | burstiness = std of 10-ms-binned spike density, mean per (channel, band) → 40-D |
| Guard band | 5 min around each seizure; interictal pool subsampled to match seizure-minute count per EDF |
| Classifiers | SVM-RBF (C=1, γ=scale, class-balanced) · Random Forest (300 trees) |
| Evaluation | leave-one-recording-out (honest) + stratified 5-fold (leaky, for reference) |

## Performance note

The `sctn` library ships with its Numba path disabled, so the reference
resonator runs as pure Python (~400 s per resonator-minute). `sctn_fast.py`
reimplements the exact forward pass as a compiled `@njit` kernel — **~1000×
faster and bit-identical** to the on-disk spikes — which makes encoding the full
24-patient dataset practical (hours instead of weeks).

## Results (summary)

On 24 CHB-MIT patients (138 seizure recordings encoded), leave-one-recording-out:

- **Within a recording**, SCTN burstiness matches classical FFT band-power.
- **Across unseen recordings**, a band-matched fine-FFT holds a *small* edge
  (median ΔAUC ≈ 0.01–0.03) that is **not robustly significant** — it is driven
  by a few low-recording patients and vanishes on well-powered ones; SCTN wins on
  a subset of patients.
- The **Gamma band is neutral** (adding it does not change performance).

Net: a multiplier-free neuromorphic front-end lands within ~1–3 AUC points of the
best classical FFT features. (See `analysis/stats/` for per-patient tables.)

## References

Hadad, Y., Bensimon, M., Ben-Shimol, Y., Greenberg, S. (2024). *Situational
Awareness Classification Based on EEG Signals and Spiking Neural Network*.
Applied Sciences 14(19), 8911. <https://doi.org/10.3390/app14198911>

Shoeb, A.H. (2009). *Application of Machine Learning to Epileptic Seizure Onset
Detection and Treatment*. PhD thesis, MIT.
