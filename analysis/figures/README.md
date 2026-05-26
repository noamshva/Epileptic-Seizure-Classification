# figures/

PNGs written by `src/2.analyze_chbmit.py`, modelled on Figures 4, 6, and 7 of
Hadad et al. (2024) but adapted to seizure vs non-seizure.

| File | Content |
|---|---|
| `01_raw_signal_time.png` | Raw EEG slice + `plt.specgram` for the primary channel during the seizure minute |
| `02_fft_spectrogram_one_channel.png` | Full FFT spectrogram + 4-band bin strip for the primary channel |
| `03_sctn_one_resonator.png` | One resonator's spike density (500-tick convolution) + 1-second-bin counts |
| `04_fft_vs_sctn_bands.png` | Paper Figure 4 (a, b): FFT band strip on top, SCTN band spikegram on bottom, same minute |
| `05_per_channel_fft_grid.png` | FFT band strip per channel (5×2 grid) |
| `06_per_channel_sctn_grid.png` | SCTN band spikegram per channel (5×2 grid) |
| `07_combined_per_channel_grid.png` | SCTN and FFT side by side per channel |
| `08_topography_fft_vs_sctn.png` | Paper Figure 6: FFT row + SCTN row, one column per band (one minute) |
| `09_topography_seizure_vs_non_seizure.png` | Paper Figure 7 style: seizure row + non-seizure row, one column per band |

All generated content is gitignored; only this README is tracked.
