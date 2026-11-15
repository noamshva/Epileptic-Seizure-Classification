"""Seizure vs non-seizure SNN classifier on SCTN burstiness features.

Replaces the SVM of 3.run_classify_chbmit.py with a three-layer spiking network
(encoder LIF -> hidden LIF -> output LIF) trained by surrogate-gradient descent.
Same features and same leave-one-recording-out protocol, so scores compare
directly with svm_classifier_report.csv.

Needs the feature cache: python -u src/cache_features.py

Usage:
    python -u src/3c.run_classify_snn_chbmit.py                 # all cached patients
    python -u src/3c.run_classify_snn_chbmit.py chb01 chb03
Env:
    INPUT = features (default) | bins   (bins: 10-ms spike counts per resonator,
                                         one time step per bin; needs cache_features.py --bins)
    FEATURE_LEVEL = band (default) | resonator
    HIDDEN=64  T=50  EPOCHS=30  LR=0.01  BETA=0.9  SEED=0  WORKERS=8
    SLOPE=10 (surrogate steepness)  CLIP=0 (gradient-norm clip, 0 = off)
    READOUT = spikes (default; output spike counts) | membrane (mean output
              membrane potential, output neurons integrate without firing)
    TAG = suffix for the report file (sweeps)
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
import features as F                                                    # noqa: E402

INPUT = os.environ.get("INPUT", "features")
FEATURE_LEVEL = os.environ.get("FEATURE_LEVEL", "band")
HIDDEN = int(os.environ.get("HIDDEN", 64))
T = int(os.environ.get("T", 50))
EPOCHS = int(os.environ.get("EPOCHS", 30))
LR = float(os.environ.get("LR", 0.01))
BETA = float(os.environ.get("BETA", 0.9))
BATCH = 128
SEED = int(os.environ.get("SEED", 0))
TAG = os.environ.get("TAG", "")
THRESH = 1.0
SLOPE = float(os.environ.get("SLOPE", 10.0))
CLIP = float(os.environ.get("CLIP", 0))
READOUT = os.environ.get("READOUT", "spikes")
STUCK_LOSS = 0.68          # final train loss above this = fold did not learn
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 1)))
LOGIT_SCALE = 10.0

_OUT_ROOT = Path(os.environ.get("PIPELINE_DATA_ROOT", str(_SCRIPT_DIR.parent)))
OUT_DIR = _OUT_ROOT / "output_data" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


class _Spike(torch.autograd.Function):
    """Heaviside forward, fast-sigmoid surrogate gradient backward."""

    @staticmethod
    def forward(ctx, v):
        ctx.save_for_backward(v)
        return (v > 0).float()

    @staticmethod
    def backward(ctx, grad):
        (v,) = ctx.saved_tensors
        return grad / (SLOPE * v.abs() + 1.0) ** 2


def _lif(mem, cur):
    mem = BETA * mem + cur
    spk = _Spike.apply(mem - THRESH)
    return mem - spk * THRESH, spk


class SNN(nn.Module):
    """Encoder LIF (one neuron per feature) -> hidden LIF -> output LIF.

    x [B, D]: each feature is a constant current for T steps.
    x [B, steps, D]: one current per time step.
    Everything after the encoder is carried by spikes. Returns logits [B, 2]
    (scaled output firing rate, or mean output membrane) and the mean hidden
    firing rate.
    """

    def __init__(self, n_in: int, n_hidden: int = HIDDEN, n_out: int = 2):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(n_in))
        self.bias = nn.Parameter(torch.full((n_in,), 0.5))
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        seq = x.dim() == 3
        steps = x.shape[1] if seq else T
        m0 = x.new_zeros(x.shape[0], self.gain.numel())
        m1 = x.new_zeros(x.shape[0], self.fc1.out_features)
        m2 = x.new_zeros(x.shape[0], self.fc2.out_features)
        counts = torch.zeros_like(m2)
        hid = 0.0
        for t in range(steps):
            m0, s0 = _lif(m0, (x[:, t] if seq else x) * self.gain + self.bias)
            m1, s1 = _lif(m1, self.fc1(s0))
            if READOUT == "membrane":
                m2 = BETA * m2 + self.fc2(s1)
                counts = counts + m2
            else:
                m2, s2 = _lif(m2, self.fc2(s1))
                counts = counts + s2
            hid = hid + s1.mean()
        scale = 1.0 if READOUT == "membrane" else LOGIT_SCALE
        return counts / steps * scale, hid / steps


def _bin_stats(X, idx, chunk: int = 256):
    """Per-resonator mean/std of the binned counts over the training windows."""
    n = 0; s = np.zeros(X.shape[2]); ss = np.zeros(X.shape[2])
    for i in range(0, len(idx), chunk):
        a = X[idx[i:i + chunk]].astype(np.float64)
        n += a.shape[0] * a.shape[1]; s += a.sum(axis=(0, 1)); ss += (a ** 2).sum(axis=(0, 1))
    mu = s / n
    sd = np.sqrt(np.maximum(ss / n - mu ** 2, 0.0))
    sd[sd == 0] = 1.0
    return torch.tensor(mu, dtype=torch.float32), torch.tensor(sd, dtype=torch.float32)


def fit_predict(X, y, tr, te, seed: int = SEED):
    """Train on windows tr; return (P(seizure), label, hidden rate, final train loss) for te."""
    torch.manual_seed(seed)
    if X.ndim == 3:
        mu, sd = _bin_stats(X, tr)
        Xt = torch.from_numpy(X)
    else:
        mu, sd = 0.0, 1.0
        Xt = torch.tensor(StandardScaler().fit(X[tr]).transform(X), dtype=torch.float32)
    tr_t = torch.as_tensor(tr); te_t = torch.as_tensor(te)
    y_t = torch.tensor(y, dtype=torch.long)

    def batch(idx):
        return (Xt[idx].float() - mu) / sd

    ytr = y[tr]
    n_pos = int(ytr.sum()); n_neg = len(ytr) - n_pos
    w = torch.tensor([len(ytr) / (2 * n_neg), len(ytr) / (2 * n_pos)], dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    net = SNN(X.shape[-1])
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    g = torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS):
        perm = torch.randperm(len(tr_t), generator=g)
        losses = []
        for i in range(0, len(perm), BATCH):
            idx = tr_t[perm[i:i + BATCH]]
            logits, _ = net(batch(idx))
            loss = loss_fn(logits, y_t[idx])
            losses.append(float(loss))
            opt.zero_grad(); loss.backward()
            if CLIP:
                nn.utils.clip_grad_norm_(net.parameters(), CLIP)
            opt.step()

    net.eval()
    rates, hids = [], []
    with torch.no_grad():
        for i in range(0, len(te_t), 512):
            rate, hid = net(batch(te_t[i:i + 512]))
            rates.append(rate); hids.append(float(hid) * rate.shape[0])
    rate = torch.cat(rates)
    prob = torch.softmax(rate, dim=1)[:, 1].numpy()
    return prob, rate.argmax(dim=1).numpy(), sum(hids) / len(te_t), float(np.mean(losses))


def run_patient(subj: str):
    """LORO over one patient's recordings; returns a report row or a skip message."""
    try:
        X, y, g = F.load_bins(subj) if INPUT == "bins" else F.load_cached(subj, FEATURE_LEVEL)
    except FileNotFoundError as e:
        return subj, str(e)
    n_rec = len(set(g.tolist()))
    if n_rec < 2 or y.sum() in (0, len(y)):
        return subj, "needs >=2 recordings and both classes"

    t0 = time.time()
    prob = np.full(len(y), np.nan); pred = np.zeros(len(y), int); rates = []; losses = []
    for tr, te in LeaveOneGroupOut().split(X, y, groups=g):
        if len(set(y[tr].tolist())) < 2:
            continue
        prob[te], pred[te], r, l = fit_predict(X, y, tr, te)
        rates.append(r); losses.append(l)
    ok = ~np.isnan(prob)
    return subj, {"patient": subj, "n_rec": n_rec, "n_win": len(y), "seizure": int(y.sum()),
                  "loro_auc": roc_auc_score(y[ok], prob[ok]),
                  "loro_f1": f1_score(y[ok], pred[ok], zero_division=0),
                  "loro_acc": accuracy_score(y[ok], pred[ok]),
                  "hidden_rate": float(np.mean(rates)), "train_loss": float(np.mean(losses)),
                  "stuck_folds": int(np.sum(np.asarray(losses) > STUCK_LOSS)), "n_folds": len(losses),
                  "seconds": time.time() - t0}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    patients = args or [p.stem for p in sorted((F.BINS_ROOT if INPUT == "bins" else F.CACHE_ROOT).glob("chb*.npz"),
                                               key=lambda p: -p.stat().st_size)]
    workers = min(int(os.environ.get("WORKERS", 8)), len(patients))
    print(f"SNN {'bins' if INPUT == 'bins' else FEATURE_LEVEL}: hidden={HIDDEN} T={T} epochs={EPOCHS} lr={LR} beta={BETA} slope={SLOPE} clip={CLIP} readout={READOUT} seed={SEED}"
          f" | patients: {len(patients)} | workers: {workers}", flush=True)

    svm_csv = OUT_DIR / "svm_classifier_report.csv"
    svm = pd.read_csv(svm_csv).set_index("patient")["loro_auc"] if svm_csv.is_file() else {}

    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(run_patient, s) for s in patients]):
            subj, row = fut.result()
            if isinstance(row, str):
                print(f"  {subj}: skip ({row})", flush=True); continue
            row["svm_loro_auc"] = float(svm[subj]) if subj in svm else np.nan
            rows.append(row)
            print(f"  {subj}: SNN AUC={row['loro_auc']:.3f} F1={row['loro_f1']:.3f} | SVM AUC="
                  f"{row['svm_loro_auc']:.3f} | hidden rate={row['hidden_rate']:.3f}"
                  f" | loss={row['train_loss']:.3f} stuck={row['stuck_folds']}/{row['n_folds']}"
                  f" | {row.pop('seconds'):.0f}s", flush=True)

    if not rows:
        print("No patients classified."); return
    df = pd.DataFrame(rows).sort_values("patient")
    out = OUT_DIR / ("snn_classifier_report" + (f"_{TAG}" if TAG else "") + ".csv")
    df.to_csv(out, index=False)
    print(f"\nSNN  LORO AUC: mean {df['loro_auc'].mean():.3f}  median {df['loro_auc'].median():.3f}")
    if df["svm_loro_auc"].notna().any():
        print(f"SVM  LORO AUC: mean {df['svm_loro_auc'].mean():.3f}  "
              f"median {df['svm_loro_auc'].median():.3f}  (same patients)")
        print(f"SNN >= SVM on {(df['loro_auc'] >= df['svm_loro_auc']).sum()}/{len(df)} patients")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
