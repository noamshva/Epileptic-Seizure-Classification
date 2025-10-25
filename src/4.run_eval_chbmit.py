#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate a trained SNN classifier on CHB-MIT spike dataset.

Loads the model from:
    ../output_data/trained_models/trained_chbmit_model.pth

Evaluates on all .npz windows from:
    ../dataset/spikes_windowed/

Outputs metrics, confusion matrix, and ROC/PR plots to:
    ../output_data/results/
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
)
import pandas as pd
import seaborn as sns
import sys, types
if "torch._six" not in sys.modules:
    m = types.ModuleType("torch._six")
    m.container_abcs = ()
    m.string_classes = (str,)
    m.int_classes = (int,)
    sys.modules["torch._six"] = m

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
DATASET_PATH = Path("../dataset/spikes_windowed")
MODEL_PATH = Path("../output_data/trained_models/trained_chbmit_model.pth")
OUT_DIR = Path("../output_data/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16

# ------------------------------------------------------------
# Model definition (same as used in training)
# ------------------------------------------------------------
from utils import Topographies2SNN, Bands2TopographyLayer, Resonators2BandsLayer

full_resonator_array = torch.tensor([
    1.10, 1.30, 1.60, 1.90, 2.20, 2.50, 2.88, 3.05, 3.39,
    3.70, 4.12, 4.62, 5.09, 5.45, 5.87, 6.36, 6.80, 7.60, 8.60
], dtype=torch.float32)

bands = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 14),
    "Beta": (14, 32),
    "Gamma": (32, 62),
}

channels = [
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4", "AF3",
    "F7", "F3", "FC5", "T7", "P7", "O1"
]

N = 11
xs = (np.array([-0.7,-0.66,0,0.2,0.4,0.5,0.7,0.7,0.5,0.4,0.2,0,-0.66,-0.7]) + 1) / 2
ys = (-np.array([0.2,0.6,0.95,0.6,0.3,0.6,0.2,-0.2,-0.6,-0.3,-0.6,-0.95,-0.6,-0.2]) + 1) / 2
ch_pos = {ch: (x, y) for ch, (x, y) in zip(channels, zip(ys, xs))}
ch_pos_N = {k: (int(x * N), int(y * N)) for k, (x, y) in ch_pos.items()}


def build_snn():
    bands2topographies = {}
    for band, (lf, hf) in bands.items():
        resonators2band = {}
        for ch in ch_pos.keys():
            freqs = full_resonator_array[
                (full_resonator_array >= lf) & (full_resonator_array < hf)
            ]
            if len(freqs) == 0:
                continue
            clk_freqs = torch.where(freqs < 10, 15360, 153600)
            resonators2band[ch] = Resonators2BandsLayer(ch, band, clk_freqs)
        bands2topographies[band] = Bands2TopographyLayer(
            band, resonators2band, shape=(N, N), ch_pos=ch_pos_N
        )
    snn = Topographies2SNN(bands2topographies, N * N, N * int(np.sqrt(N)), 3)
    return snn


class EEGClassifier(nn.Module):
    def __init__(self, snn_out_dim, num_classes=2):
        super().__init__()
        self.snn = build_snn()
        self.fc = nn.Sequential(
            nn.Linear(snn_out_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.fc(x)


# ------------------------------------------------------------
# Dataset loading
# ------------------------------------------------------------
def load_dataset(path):
    files = list(Path(path).rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found under {path.resolve()}")

    X, y = [], []
    for f in tqdm(files, desc=f"Loading dataset from {path.name}"):
        label = 1 if "seizure" in f.parts[-2].lower() else 0
        try:
            with open(f, "rb") as fp:
                data = pickle.load(fp)
            arrs = []
            for v in data.values():
                if hasattr(v, "shape"):
                    arrs.append(v.detach().cpu().numpy().flatten())
                elif isinstance(v, (list, tuple)):
                    arrs.extend(np.array(v, dtype=float).flatten())
            if not arrs:
                continue
            X.append(np.concatenate(arrs))
            y.append(label)
        except Exception as e:
            print(f"⚠️ Skipping {f.name}: {e}")
            continue

    X = np.stack(X)
    y = np.array(y)
    print(f"✅ Loaded {len(X)} samples ({sum(y)} seizure, {len(y)-sum(y)} non-seizure)")
    return X, y


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------
def evaluate(model, X, y):
    model.to(DEVICE)
    model.eval()
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long)
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    preds, probs, trues = [], [], []
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(DEVICE)
            outputs = model(batch_X)
            p = torch.softmax(outputs, dim=1)
            preds.extend(torch.argmax(p, dim=1).cpu().numpy())
            probs.extend(p[:, 1].cpu().numpy())  # seizure prob
            trues.extend(batch_y.cpu().numpy())

    preds = np.array(preds)
    probs = np.array(probs)
    trues = np.array(trues)

    # Metrics
    cm = confusion_matrix(trues, preds)
    report = classification_report(trues, preds, output_dict=True)
    acc = report["accuracy"]

    print(f"\n✅ Evaluation complete | Accuracy = {acc:.4f}\n")
    print(pd.DataFrame(report).T)

    # Save metrics
    metrics_path = OUT_DIR / "eval_summary.csv"
    pd.DataFrame(report).T.to_csv(metrics_path)
    print(f"[INFO] Saved metrics: {metrics_path}")

    # Confusion matrix
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Seizure", "Seizure"],
                yticklabels=["Non-Seizure", "Seizure"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    cm_path = OUT_DIR / "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    print(f"[INFO] Saved confusion matrix: {cm_path}")

    # ROC curve
    fpr, tpr, _ = roc_curve(trues, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    roc_path = OUT_DIR / "roc_curve.png"
    plt.savefig(roc_path)
    plt.close()
    print(f"[INFO] Saved ROC curve: {roc_path}")

    # Precision-recall
    precision, recall, _ = precision_recall_curve(trues, probs)
    plt.figure()
    plt.plot(recall, precision, label="PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    pr_path = OUT_DIR / "precision_recall_curve.png"
    plt.savefig(pr_path)
    plt.close()
    print(f"[INFO] Saved PR curve: {pr_path}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    print(f"Running evaluation on: {DEVICE}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    X, y = load_dataset(DATASET_PATH)
    model = EEGClassifier(snn_out_dim=X.shape[1])
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    evaluate(model, X, y)
