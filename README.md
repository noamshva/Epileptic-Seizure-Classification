# Epileptic-Seizure-Classification

This project implements an **SCTN-based Spiking Neural Network (SNN)** for detecting epileptic seizures using the **CHB-MIT EEG dataset**.
It covers preprocessing, spike generation, pre-training, supervised training, and visual analysis.

---

##  Setup

### 1. Activate virtual environment

```
.venv3_11\Scripts\activate
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Fix the SCTN library

* Copy the local SCTN parameter files into:

  ```
  .venv3_11\Lib\site-packages\sctn\
  ```
* In `sctn/resonator.py`, inside `simple_resonator()`, add the missing line:

  ```python
  network.clk_freq = clk_freq
  ```

---

##  Dataset

Download the CHB-MIT EEG dataset from PhysioNet:
👉 [https://physionet.org/content/chbmit/1.0.0/](https://physionet.org/content/chbmit/1.0.0/)

Place it locally, for example:

```
C:\<your_path>\dataset\chbmit
```

Then, in `signal2spikes.py`, update:

```python
CHBMIT_ROOT = r"C:\<your_path>\dataset\chbmit"
```

---

## Scripts and Expected Outputs

### `signal2spikes.py`

* Converts raw `.edf` EEG recordings into compressed spike files (`.npz`).
* **Outputs:**

  * Intermediate spikes → `dataset/spikes_raw/`
  * Final training windows → `dataset/spikes_windowed/`

---

### `pre_train_chbmit.py`

* Builds the SNN and runs a forward pass to compute **descriptive band-power statistics**.
* No learning — used for validating and inspecting spike data.
* **Output:** Printed stats and optional CSV summaries under `analysis/pretrain_stats/`.

---

### `run_train_chbmit.py`

* Trains the SNN model on the generated dataset using a proper learning loop.
* **Outputs:**

  * Model weights → `output_data/trained_models/`
  * Logs and metrics → `output_data/train_logs/`
  * Results summary → `output_data/results/`

---

### `run_analysis_chbmit.py`

* Generates **spectrograms and spikegrams** comparing seizure vs. non-seizure EEG segments.
* **Outputs:** Visualizations saved in `analysis/figures/`.

---

## ▶ Typical Run Order

1. `python src/signal2spikes.py` → generate spike datasets
2. `python src/pre_train_chbmit.py` → validate band power
3. `python src/run_train_chbmit.py` → train the model
4. `python src/run_analysis_chbmit.py` → visualize results

*(Adjust parameters at the top of each script before running.)*

---

##  Notes

* Ensure the **CHB-MIT dataset** and **output directories** exist before running.
* Each `.npz` file contains a **pickled dictionary** of spike tensors keyed by channel and frequency band.
* Frequency bands used:
  * **Delta:** 0.5–4 Hz
  * **Theta:** 4–8 Hz
  * **Alpha:** 8–14 Hz
  * **Beta:** 14–32 Hz
  * **Gamma:** 32–62 Hz
 
