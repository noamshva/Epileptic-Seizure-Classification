# Dataset

This folder contains all EEG data used in the project.

- **chbmit/** – Original CHB-MIT dataset (.edf files + summaries).
- **spikes_raw/** – 1-minute spike trains generated from the raw EEG using signal2spikes.py.
- **spikes_windowed/** – Final 0.25 s windows ready for training the SNN model.

Data flow:
chbmit → spikes_raw → spikes_windowed
