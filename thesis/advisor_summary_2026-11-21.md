# Thesis status and four points to agree on

Noam, 21 November 2026. For Prof. Shlomo Greenberg and Dr. Yehuda Ben-Shimol.

## Where the work stands

The experiments are complete and frozen. All 24 CHB-MIT patients are encoded with the SCTN resonator bank (138 seizure recordings, 185 seizures). An SNN classifier is trained and tested, and a seizure detector was scored once on EEG that was held out until the end. What remains is writing.

Main results:

| Result | SCTN | FFT band power |
|---|---|---|
| Window AUC, SVM, leave one recording out | 0.962 | 0.932 |
| Window AUC, SNN on the 40 SCTN features (3 seeds) | 0.957 | |
| Window AUC, SNN directly on spike trains | 0.943 | |
| Seizures detected, SVM (final test) | 175 of 185 | 173 of 185 |
| False alarms per hour, SVM (final test, 36.3 h) | 0.80 | 1.32 |
| Median detection delay, SVM | 6 s | 7 s |

The SNN on SCTN features detects 178 to 179 of 185 seizures at 0.96 to 2.04 false alarms per hour, depending on the random seed.

## Four points I would like to agree on

**1. The July result is reversed.** In July the FFT features were slightly ahead. The cause was in my encoder: each minute was scaled to its own peak, which removed the amplitude difference between seizure and background. With a fixed 100 µV full scale, SCTN is ahead of FFT on all 24 patients (p < 0.0001). When the FFT input is clipped at the same 100 µV, the gap narrows to 0.962 against 0.949. At a 200 µV full scale SCTN is still ahead (0.953 against 0.940, p = 0.0006).

**2. I withdraw the claim of lower computational complexity.** Counting operations, the SCTN front-end needs about 960 times more operations per second than the FFT. The argument I can defend is architectural: SCTN uses no multipliers and is event-driven, so it fits neuromorphic hardware. A power saving is a hypothesis, not a result. In support, both SNN classifiers keep their accuracy with weights rounded to 5 bits.

**3. I propose moving the NEMO measurement to future work.** Without a hardware measurement the power claim cannot be tested, and the measurement does not fit the remaining time.

**4. I propose dropping STDP learning, the 2D map and the SCTN-neuron classifier.** The SNN that is delivered is a three-layer LIF network trained with surrogate gradients. It is the goal we set, but it is not built from SCTN neurons.

## Limits I will state in the thesis

- SCTN is ahead of FFT with the SVM. With a random forest the two are tied (0.963 against 0.959, p = 0.29), so the advantage depends on the classifier.
- The difference in false alarms between SCTN and FFT is not significant on 36 hours (difference 0.52 per hour, 95% interval −0.12 to +1.42).
- The false alarm rate is an interictal rate. The 5 minutes around each seizure are excluded. In the minute after a seizure ends the detector often alarms again (67 of 185 seizures).
- The detectors are patient-specific. Trained on other patients only, the AUC is 0.83 for all models.
- One patient (chb16) accounts for half of the missed seizures with the SVM (5 of 10).
- One dataset only.

## What I need from you

1. Agreement on points 1 to 4, or the changes you want.
2. Whether the thesis should be written in Word or LaTeX.
3. A target date for the first full draft of the results chapter.
