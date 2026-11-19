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
| Input scaling | fixed full scale of 100 µV with clipping (`INPUT_FULL_SCALE_UV`); `0` restores the old per-minute peak scaling |
| Bands | Delta, Theta, Alpha, Beta (Gamma encoded on disk; excluded from the active feature set) |
| Resonator clocks | < 10 Hz @ 15.36 kHz, ≥ 10 Hz @ 153.6 kHz (matches SCTN pre-trained params) |
| Window | 2 s, 50 % overlap (Shoeb 2010) |
| Feature | burstiness = std of 10-ms-binned spike density, mean per (channel, band) → 40-D |
| Guard band | 5 min around each seizure; interictal pool subsampled to match seizure-minute count per EDF |
| Classifiers | SVM-RBF (C=1, γ=scale, class-balanced) · Random Forest (300 trees) |
| Evaluation | leave-one-recording-out (honest) + stratified 5-fold (leaky, for reference) |

## Frozen detector protocol (fixed 2026-11-17, before the final test data was scored)

- Per-patient models. Training data: the patient's seizure recordings plus the
  two minutes already encoded from each of that patient's seizure-free
  recordings (`dataset/spikes_raw_free`), as extra non-seizure examples.
- Models as reported: SVM on 40 SCTN features, SVM on 40 FFT band powers, SNN on
  40 SCTN features (seeds 0, 1, 2), SNN on binned spikes (`INPUT=bins
  READOUT=membrane LR=0.002 CLIP=1 EPOCHS=60`, seed 0).
- Decision: each model's default threshold. Alarm: 5 consecutive flagged windows.
- Sensitivity and detection delay: leave-one-recording-out over the seizure
  recordings, with the same seizure-free training data in every fold.
- False alarms: four new random minutes from every seizure-free recording
  (`dataset/spikes_raw_free2`, encoder `SEIZURE_FREE=1 FREE_SEED=1
  FREE_MINUTES=4`), scored once: `FINAL=1 ALARM_RUN=5 python src/eval_false_alarms.py <model>`.

## Performance note

The `sctn` library ships with its Numba path disabled, so the reference
resonator runs as pure Python (~400 s per resonator-minute). `sctn_fast.py`
reimplements the exact forward pass as a compiled `@njit` kernel — **~1000×
faster and bit-identical** to the on-disk spikes — which makes encoding the full
24-patient dataset practical (hours instead of weeks).

## Results (summary)

24 CHB-MIT patients, 138 seizure recordings (185 seizures), fixed 100 µV input scaling.

**Window classification** (per patient, leave-one-recording-out, pooled AUC):

| Features + classifier | Mean AUC |
|---|---|
| SCTN 40-D + SVM | 0.962 |
| SCTN 280-D + SVM | 0.967 |
| SCTN 40-D + SNN (3 seeds) | 0.957 |
| SCTN binned spikes + SNN | 0.943 |
| FFT band power + SVM | 0.932 |
| FFT band power, clipped at 100 µV, + SVM | 0.949 |

SCTN beats FFT band power on 24 of 24 patients (Wilcoxon p < 0.0001). At a 200 µV
full scale SCTN drops to 0.953 and still beats FFT clipped at 200 µV (0.940,
20 of 24 patients, p = 0.0006) (`analysis/stats/scale_robustness_200uv.csv`). The
earlier result (FFT slightly ahead) came from per-minute peak scaling in the
encoder, which removed the amplitude difference between classes.

**Final test** (frozen protocol above, 36.3 h of unseen seizure-free EEG from
545 recordings, alarm after 5 consecutive windows, scored once):

| Model | Seizures detected | False alarms / h | Median delay |
|---|---|---|---|
| SVM, SCTN | 175/185 | 0.80 | 6 s |
| SNN, SCTN (seeds 0 / 1 / 2) | 178 / 179 / 178 | 1.16 / 2.04 / 0.96 | 7 s |
| SNN, binned spikes | 171/185 | 1.07 | 8 s |
| SVM, FFT | 173/185 | 1.32 | 7 s |

95% patient-bootstrap intervals (`src/fa_ci.py`): SVM SCTN 0.34 to 1.38/h, SVM FFT
0.63 to 2.30/h. The SCTN vs FFT detector difference is not significant on 36 h.

Most misses come from chb16 (4 to 10 of its 10 seizures, all models). False
alarms concentrate in a few patients (chb07, chb12, chb14, chb24); FFT has 14 of
its 48 in chb20. Cross-patient (leave one patient out) AUC is 0.83 for all
models, so the detectors stay patient-specific.

**Additional tests** (run after the final test, none of them uses the final test data):

*Gamma band* (`src/svm_feature_sets.py`). Adding the 7 Gamma resonators per
channel does not help: 50-D 0.962 against 0.962 for 40-D (p = 0.90), 350-D 0.965
against 0.967 for 280-D (p = 0.68). Gamma alone gives 0.920. SCTN stays ahead of
FFT on 24 of 24 patients with Gamma included.

*Weight precision* (`src/snn_quantization.py`). Trained SNN weights rounded to N
bits with no retraining, seed 0, mean AUC:

| Weights | Full | 8 bits | 6 bits | 5 bits | 4 bits | 3 bits | 2 bits |
|---|---|---|---|---|---|---|---|
| SNN, 40 SCTN features | 0.958 | 0.958 | 0.958 | 0.957 | 0.955 | 0.938 | 0.694 |
| SNN, binned spikes | 0.941 | 0.941 | 0.941 | 0.940 | 0.935 | 0.913 | 0.755 |

Both models keep their accuracy down to 5 bits. At 4 bits the mean loss is not
significant, but one patient (chb12, binned spikes) loses 0.11. The loss at 3
bits is significant (p = 0.003 and p = 0.0004). Leak and threshold were not rounded.

*False alarms near seizures* (`src/eval_peri.py`, `src/eval_free_unseen.py`,
`src/fa_context_ci.py`). The 5 minutes on each side of every seizure are excluded
from training and from the final test. They were encoded separately (30.2 h with
the seizure-free part of the seizure minutes) and scored leave-one-recording-out
with the frozen detector. False alarms per hour:

| EEG scored | Hours | SVM, SCTN | SNN, SCTN (seed 0) | SVM, FFT |
|---|---|---|---|---|
| Final test (seizure-free recordings) | 36.3 | 0.80 | 1.16 | 1.32 |
| Seizure-free recordings not seen in training | 18.2 | 1.21 | 1.98 | 2.64 |
| Seizure recordings, at least 5 min from a seizure | 5.9 | 3.05 | 4.75 | 4.41 |
| 1 to 5 min before onset | 12.4 | 4.45 | 2.67 | 2.43 |
| Last minute before onset | 3.1 | 11.4 | 10.7 | 9.8 |
| First minute after the end | 3.0 | 35.2 | 37.5 | 25.0 |
| 1 to 5 min after the end | 11.7 | 11.6 | 10.2 | 9.2 |

All three detectors keep alarming after a seizure ends. With the SVM, 67 of 185
seizures have an alarm in the minute after the annotated end, and the rate stays
raised for about 3 minutes. Before onset only the last minute is raised (27 of
185 seizures). The final-test rate is therefore an interictal rate. On
seizure-free recordings not seen in training the rate is 1.5 to 2 times the
final-test rate (not significant for any model). The higher rate in seizure
recordings away from seizures rests on 5.9 h and is not significant once the
patient mix is matched (p = 0.51 SVM, 0.08 SNN, 0.09 FFT). SCTN with the SVM has
fewer false alarms than FFT on unseen seizure-free recordings (difference
-1.42/h, 95% interval -3.08 to -0.12) and more in the first minute after a
seizure (+10.1/h, 3.7 to 15.8).

**Cost:** the SCTN front-end needs about 8.4e7 neuron updates per second against
about 8.7e4 multiplications for the FFT (`src/cost_estimate.py`). The advantage
for SCTN is architectural, not operation count: it needs no multipliers and is
event-driven, so it suits neuromorphic hardware. The power saving is a hypothesis
here. Measuring it on a neuromorphic processor (e.g. NEMO, Technion) is future work.

Per-patient tables are in `analysis/stats/`.

## References

Hadad, Y., Bensimon, M., Ben-Shimol, Y., Greenberg, S. (2024). *Situational
Awareness Classification Based on EEG Signals and Spiking Neural Network*.
Applied Sciences 14(19), 8911. <https://doi.org/10.3390/app14198911>

Shoeb, A.H. (2009). *Application of Machine Learning to Epileptic Seizure Onset
Detection and Treatment*. PhD thesis, MIT.
