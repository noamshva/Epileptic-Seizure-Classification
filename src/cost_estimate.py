"""Operation counts per second of 10-channel EEG: SCTN front-end vs FFT,
and SNN vs SVM classifier. Counts only; no energy model.

Usage:
    python -u src/cost_estimate.py [chb01 chb08 ...]   # patients for the classifier part
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))
import features as F                                                    # noqa: E402

NEURONS_PER_RESONATOR = 5      # delta-sigma input neuron + 4-neuron loop (sctn_fast.py)


def front_end():
    clks = [F._clk_of(f) for b in F.BANDS for f in F.BAND_RESONATORS[b]]
    upd = NEURONS_PER_RESONATOR * sum(clks) * len(F.CHANNELS)
    print(f"SCTN: {len(clks)} resonators/channel, clocks {sorted(set(clks))} Hz"
          f" ({sum(c == min(clks) for c in clks)} slow, {sum(c == max(clks) for c in clks)} fast)")
    print(f"  neuron updates per second, {len(F.CHANNELS)} channels: {upd:.3g}"
          f"  (each: add, clip, compare, shift-leak; no multiplier)")
    # FFT baseline: 256-point segments, 50% overlap -> 2 new FFTs per second per channel
    n = F._FFT_FS; ffts = 2 * len(F.CHANNELS)
    mult = ffts * (2 * n * np.log2(n) + 2 * (n // 2 + 1))     # radix-2 butterflies (4 real mult each) + |X|^2
    print(f"FFT: {ffts} {n}-point FFTs per second -> about {mult:.3g} real multiplications per second")
    print(f"  ratio SCTN neuron updates / FFT multiplications: {upd / mult:.0f}x")


def classifier(subjects):
    spec = importlib.util.spec_from_file_location("snn3c", _SRC / "3c.run_classify_snn_chbmit.py")
    snn = importlib.util.module_from_spec(spec); spec.loader.exec_module(snn)
    import torch
    for s in subjects:
        X, y, g = F.load_cached(s, "band")
        m = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale",
                                                class_weight="balanced")).fit(X, y)
        nsv = int(m[-1].n_support_.sum())
        # SNN trained on all windows; count spikes on the same windows
        torch.manual_seed(0)
        Xs = torch.tensor(StandardScaler().fit_transform(X), dtype=torch.float32)
        net = snn.SNN(X.shape[1]); opt = torch.optim.Adam(net.parameters(), lr=snn.LR)
        w = torch.tensor([len(y) / (2 * (len(y) - y.sum())), len(y) / (2 * y.sum())], dtype=torch.float32)
        lossf = torch.nn.CrossEntropyLoss(weight=w); yt = torch.tensor(y, dtype=torch.long)
        gen = torch.Generator().manual_seed(0)
        for _ in range(snn.EPOCHS):
            perm = torch.randperm(len(y), generator=gen)
            for i in range(0, len(y), snn.BATCH):
                idx = perm[i:i + snn.BATCH]
                loss = lossf(net(Xs[idx])[0], yt[idx]); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            m0 = torch.zeros(len(y), X.shape[1]); m1 = torch.zeros(len(y), snn.HIDDEN)
            e = h = 0.0
            for _ in range(snn.T):
                m0, s0 = snn._lif(m0, Xs * net.gain + net.bias)
                m1, s1 = snn._lif(m1, net.fc1(s0))
                e += s0.sum(1).mean().item(); h += s1.sum(1).mean().item()
        synops = e * snn.HIDDEN + h * 2
        print(f"{s}: SVM {nsv} support vectors -> {nsv * X.shape[1]} mult + {nsv} exp per window"
              f" | SNN encoder spikes {e:.0f}, hidden spikes {h:.0f} per window"
              f" -> {synops:.0f} synaptic adds (+ {snn.T * (X.shape[1] + snn.HIDDEN + 2)} neuron updates)")


if __name__ == "__main__":
    front_end()
    classifier([a for a in sys.argv[1:] if a.startswith("chb")] or ["chb01", "chb08", "chb22"])
