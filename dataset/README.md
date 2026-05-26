# Dataset

EEG inputs and intermediate spike-encoded data. Heavy files are gitignored;
the directory structure and these READMEs are kept.

| Subfolder | Contents | Created by |
|---|---|---|
| `chbmit/` | Raw CHB-MIT EEG `.edf` files plus the per-subject `chbXX-summary.txt`. Download from <https://physionet.org/content/chbmit/1.0.0/>. | manual |
| `spikes_raw/` | Per-resonator binary spike trains, organized as `<subject>/<edf_stem>/<seizure\|non_seizure>/<minute>/<channel>/<clk>/<f0>.npz`. One file per (channel, clock, resonator-frequency, minute). Compressed with `np.savez_compressed`. | `src/1.signal2spikes_chbmit.py` |

There is no longer a `spikes_windowed/` directory — training and evaluation
slice 2-second windows from `spikes_raw/` in memory.

Data flow:

```
chbmit/<subject>/<edf>.edf
        │
        ▼  (DSM encoding via SCTN resonators)
spikes_raw/<subject>/<edf_stem>/<minute_label>/<minute>/<ch>/<clk>/<f0>.npz
        │
        ▼  (sliced on demand by src/3.run_train_chbmit.py)
SNN training windows
```
