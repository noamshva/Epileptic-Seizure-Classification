# stats/

Numerical outputs from `src/2.analyze_chbmit.py`.

| File | Content |
|---|---|
| `run_config.json` | Inputs that produced this run (subject, EDF, minutes, channels, bands, resonator frequencies, timestamp). Makes the CSVs reproducible. |
| `per_channel_band.csv` | One row per `(channel, band, state)` — `state ∈ {seizure, non_seizure}` — with FFT and SCTN summary stats: mean, std, max plus SCTN spike density. |
| `channel_ranking.csv` | One row per channel. Per-band log-ratio of SCTN std (seizure vs non-seizure) and an overall discriminability score `sctn_log_ratio_sum_abs`. Sorted by that score (most discriminative first). Use this to decide channel-set reductions. |
| `run.log` | Human-readable run log mirroring console output. |

All generated content is gitignored; only this README is tracked.
