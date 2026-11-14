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
    FEATURE_LEVEL = band (default) | resonator
    HIDDEN=64  T=50  EPOCHS=30  LR=0.01  BETA=0.9  WORKERS=8
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

FEATURE_LEVEL = os.environ.get("FEATURE_LEVEL", "band")
HIDDEN = int(os.environ.get("HIDDEN", 64))
T = int(os.environ.get("T", 50))
EPOCHS = int(os.environ.get("EPOCHS", 30))
LR = float(os.environ.get("LR", 0.01))
BETA = float(os.environ.get("BETA", 0.9))
BATCH = 128
SEED = 0
THRESH = 1.0
SLOPE = 10.0
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

    Each feature drives its encoder neuron with a constant current for T steps;
    everything after the encoder is carried by spikes. Returns output spike
    counts [B, 2] and the mean hidden firing rate.
    """

    def __init__(self, n_in: int, n_hidden: int = HIDDEN, n_out: int = 2):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(n_in))
        self.bias = nn.Parameter(torch.full((n_in,), 0.5))
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        cur0 = x * self.gain + self.bias
        m0 = torch.zeros_like(cur0)
        m1 = x.new_zeros(x.shape[0], self.fc1.out_features)
        m2 = x.new_zeros(x.shape[0], self.fc2.out_features)
        counts = torch.zeros_like(m2)
        hid = 0.0
        for _ in range(T):
            m0, s0 = _lif(m0, cur0)
            m1, s1 = _lif(m1, self.fc1(s0))
            m2, s2 = _lif(m2, self.fc2(s1))
            counts = counts + s2
            hid = hid + s1.mean()
        return counts, hid / T


def _logits(counts):
    return counts / T * LOGIT_SCALE


def fit_predict(Xtr, ytr, Xte, seed: int = SEED):
    """Train on one fold; return (P(seizure), predicted label, hidden rate) for Xte."""
    torch.manual_seed(seed)
    scaler = StandardScaler().fit(Xtr)
    Xtr_t = torch.tensor(scaler.transform(Xtr), dtype=torch.float32)
    Xte_t = torch.tensor(scaler.transform(Xte), dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)

    n_pos = int(ytr.sum()); n_neg = len(ytr) - n_pos
    w = torch.tensor([len(ytr) / (2 * n_neg), len(ytr) / (2 * n_pos)], dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    net = SNN(Xtr.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    g = torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS):
        perm = torch.randperm(len(ytr_t), generator=g)
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            counts, _ = net(Xtr_t[idx])
            loss = loss_fn(_logits(counts), ytr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()

    net.eval()
    with torch.no_grad():
        counts, hid = net(Xte_t)
        prob = torch.softmax(_logits(counts), dim=1)[:, 1].numpy()
        pred = counts.argmax(dim=1).numpy()
    return prob, pred, float(hid)


def run_patient(subj: str):
    """LORO over one patient's recordings; returns a report row or a skip message."""
    try:
        X, y, g = F.load_cached(subj, FEATURE_LEVEL)
    except FileNotFoundError as e:
        return subj, str(e)
    n_rec = len(set(g.tolist()))
    if n_rec < 2 or y.sum() in (0, len(y)):
        return subj, "needs >=2 recordings and both classes"

    t0 = time.time()
    prob = np.full(len(y), np.nan); pred = np.zeros(len(y), int); rates = []
    for tr, te in LeaveOneGroupOut().split(X, y, groups=g):
        if len(set(y[tr].tolist())) < 2:
            continue
        prob[te], pred[te], r = fit_predict(X[tr], y[tr], X[te])
        rates.append(r)
    ok = ~np.isnan(prob)
    return subj, {"patient": subj, "n_rec": n_rec, "n_win": len(y), "seizure": int(y.sum()),
                  "loro_auc": roc_auc_score(y[ok], prob[ok]),
                  "loro_f1": f1_score(y[ok], pred[ok], zero_division=0),
                  "loro_acc": accuracy_score(y[ok], pred[ok]),
                  "hidden_rate": float(np.mean(rates)), "seconds": time.time() - t0}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    patients = args or sorted(p.stem for p in F.CACHE_ROOT.glob("chb*.npz"))
    workers = min(int(os.environ.get("WORKERS", 8)), len(patients))
    print(f"SNN {FEATURE_LEVEL}: hidden={HIDDEN} T={T} epochs={EPOCHS} lr={LR} beta={BETA}"
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
                  f" | {row.pop('seconds'):.0f}s", flush=True)

    if not rows:
        print("No patients classified."); return
    df = pd.DataFrame(rows).sort_values("patient")
    out = OUT_DIR / "snn_classifier_report.csv"
    df.to_csv(out, index=False)
    print(f"\nSNN  LORO AUC: mean {df['loro_auc'].mean():.3f}  median {df['loro_auc'].median():.3f}")
    if df["svm_loro_auc"].notna().any():
        print(f"SVM  LORO AUC: mean {df['svm_loro_auc'].mean():.3f}  "
              f"median {df['svm_loro_auc'].median():.3f}  (same patients)")
        print(f"SNN >= SVM on {(df['loro_auc'] >= df['svm_loro_auc']).sum()}/{len(df)} patients")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
