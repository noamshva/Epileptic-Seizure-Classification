# Epileptic-Seizure-Classification

An SCTN-based Spiking Neural Network (SNN) for binary seizure / non-seizure
classification from CHB-MIT EEG, adapted from the situational-awareness SNN of
Hadad et al. (2024).

The pipeline mirrors the paper's four stages:

```
EEG raw  →  DSM encoding  →  Resonator bank per band  →  11×11 topography per band  →  Classification SNN
            (SCTN with        (Feature Extraction       (Feature Mapping              (input → hidden 64
             IDENTITY           – pre-trained .json       – fixed Gaussian              → hidden 32 → output 2)
             activation)        params from SCTN)         weights)                      PostPre STDP
```

## Setup

```powershell
# 1. Virtual environment
.venv3_11\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. SCTN resonator parameters
# Copy the .json parameter files into the installed sctn package:
#   .venv3_11\Lib\site-packages\sctn\resonators_params\parameters\
# In sctn/resonator.py, inside simple_resonator(), ensure:
#   network.clk_freq = clk_freq
```

## Dataset

Download CHB-MIT from PhysioNet
(<https://physionet.org/content/chbmit/1.0.0/>) and place it under
`dataset/chbmit/`. Path constants in [src/1.signal2spikes_chbmit.py](src/1.signal2spikes_chbmit.py)
default to that location.

## Scripts

| Script | Purpose | Inputs | Outputs |
|---|---|---|---|
| [src/1.signal2spikes_chbmit.py](src/1.signal2spikes_chbmit.py) | Encode raw EEG into per-channel / per-resonator spike trains; select which minutes to encode (seizure minutes + interictal minutes >= 5 min away) | `dataset/chbmit/<subject>/<edf>.edf` | `dataset/spikes_raw/<subject>/<edf_stem>/<seizure\|non_seizure>/<minute>/<channel>/<clk>/<f0>.npz` |
| [src/2.analyze_chbmit.py](src/2.analyze_chbmit.py) | Produce paper Figure 4/6/7-style plots and per-channel/band statistics. No training. | `dataset/spikes_raw/` + raw EDF | `analysis/figures/*.png`, `analysis/stats/*.csv` |
| [src/3.run_train_chbmit.py](src/3.run_train_chbmit.py) | Build the SNN, slice 2-second windows directly from `spikes_raw/`, train via supervised PostPre STDP (clamping the correct output neuron) | `dataset/spikes_raw/` | `output_data/trained_models/snn_weights.pt`, `output_data/train_logs/training_log.csv` |
| [src/4.run_eval_chbmit.py](src/4.run_eval_chbmit.py) | Load weights, run inference on each window, classify by output-neuron spike-count argmax | `dataset/spikes_raw/`, `output_data/trained_models/snn_weights.pt` | `output_data/results/eval_summary.csv` + confusion matrix / ROC / PR plots |
| [src/utils.py](src/utils.py) | SNN building blocks: `Resonators2BandsLayer`, `Bands2TopographyLayer`, `Topographies2SNN` | — | — |

## Pipeline configuration

| Parameter | Value | Notes |
|---|---|---|
| EEG sampling rate | 256 Hz | CHB-MIT fixed |
| Bandpass | 0.5–62 Hz, Butterworth order 6 | |
| Active channels | 10 CHB-MIT bipolar pairs | see `SELECT_BIPOLAR_CHANNELS` |
| Bands (active) | Delta, Theta, Alpha, Beta | Gamma is encoded on disk but excluded from windowing |
| Resonators / clock | Δ/θ/α(<10 Hz) @ 15 360 Hz, α(≥10)/β/γ @ 153 600 Hz | matches SCTN pre-trained `f_*.json` |
| Window length | 2 seconds | Shoeb 2010 convention |
| Window overlap | 50 % (1 s step) | |
| Guard band around seizures | 5 minutes | non-seizure pool only draws from interictal periods this far away |
| Class balance | per-EDF, subsample interictal minutes to match seizure-minute count | |
| Classifier | 121 → 64 → 32 → 2 LIF SCTNs | hidden sizes match Hadad et al. 2024 |
| Learning rule | BindsNet `PostPre` with supervised `clamp` | |

## Typical run order

1. `python src/1.signal2spikes_chbmit.py` — encode the configured EDFs (slow; ~1 hour/channel/EDF on a multi-core CPU)
2. `python src/2.analyze_chbmit.py` — sanity-check the encoded data with paper-style plots and per-channel discriminability stats
3. `python src/3.run_train_chbmit.py` — train the SNN
4. `python src/4.run_eval_chbmit.py` — evaluate

Window data is **not** pre-cached on disk: the training script slices windows
directly from `spikes_raw/` at training time. This keeps the on-disk footprint
small (≈200 MB per EDF instead of ~13 GB).

## References

Hadad, Y., Bensimon, M., Ben-Shimol, Y., Greenberg, S. (2024). *Situational
Awareness Classification Based on EEG Signals and Spiking Neural Network*.
Applied Sciences 14(19), 8911. <https://doi.org/10.3390/app14198911>

Shoeb, A.H. (2009). *Application of Machine Learning to Epileptic Seizure
Onset Detection and Treatment*. PhD thesis, MIT.
