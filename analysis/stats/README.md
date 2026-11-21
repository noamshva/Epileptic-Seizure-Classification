# stats/

Numerical outputs from `src/2.analyze_chbmit.py`.

| File | Content |
|---|---|
| `run_config.json` | Inputs that produced this run (subject, EDF, minutes, channels, bands, resonator frequencies, timestamp). Makes the CSVs reproducible. |
| `per_channel_band.csv` | One row per `(channel, band, state)` — `state ∈ {seizure, non_seizure}` — with FFT and SCTN summary stats: mean, std, max plus SCTN spike density. |
| `channel_ranking.csv` | One row per channel. Per-band log-ratio of SCTN std (seizure vs non-seizure) and an overall discriminability score `sctn_log_ratio_sum_abs`. Sorted by that score (most discriminative first). Use this to decide channel-set reductions. |
| `run.log` | Human-readable run log mirroring console output. |

The other files here are written by the scripts named in the main README
(Results section). Each script's docstring lists its outputs.

Notes on older files:

- `fft_vs_sctn_per_patient.csv` and `fft_vs_sctn_summary.json` come from the old
  per-minute peak scaling and are superseded. The reported numbers are in the
  `_fs100` files (fixed 100 µV full scale).
- `false_alarms_*.csv` without `_final` are development runs. The final test is
  in `false_alarms_*_final*_k5.csv`.
- In `output_data/results/snn_classifier_report*.csv` the column `svm_loro_auc`
  (0.875) is a stale reference from the old scaling. The current SVM AUC is 0.962.

Generated content is gitignored, except `scale_robustness_200uv.csv` and this README.
