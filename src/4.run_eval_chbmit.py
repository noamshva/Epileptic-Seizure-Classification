"""
Evaluate the trained SA-style SNN on CHB-MIT seizure windows.

Inference is a pure BindsNet forward pass (no clamp). The predicted label is the
argmax of the per-window output-neuron spike counts.
"""
import sys
import types

# bindsnet 0.3.x compat shim — same as 3.run_train_chbmit.py.
if "torch._six" not in sys.modules:
    _shim = types.ModuleType("torch._six")
    _shim.container_abcs = ()
    _shim.string_classes = (str,)
    _shim.int_classes = (int,)
    sys.modules["torch._six"] = _shim

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
)
from bindsnet.network.monitors import Monitor

# Reuse the build / config from the training script (single source of truth).
import importlib.util as _ilu

_TRAIN_PATH = Path(__file__).resolve().parent / "3.run_train_chbmit.py"
_spec = _ilu.spec_from_file_location("train_chbmit", _TRAIN_PATH)
_train_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_train_mod)

BANDS = _train_mod.BANDS
CHANNELS = _train_mod.CHANNELS
build_snn = _train_mod.build_snn
_build_window_inputs = _train_mod._build_window_inputs
_MinuteCache = _train_mod._MinuteCache
enumerate_windows = _train_mod.enumerate_windows
WINDOW_SECONDS = _train_mod.WINDOW_SECONDS
WINDOW_OVERLAP = _train_mod.WINDOW_OVERLAP
DEVICE = _train_mod.DEVICE


# ==========================================================
# Paths
# ==========================================================
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
MODEL_PATH = _PROJECT_ROOT / "output_data" / "trained_models" / "snn_weights.pt"
OUT_DIR = _PROJECT_ROOT / "output_data" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUBJECTS = ["chb01"]


# ==========================================================
# Load weights
# ==========================================================
def load_weights_into(snn, weights_path: Path):
    weights = torch.load(weights_path, map_location=DEVICE)
    loaded = 0
    for (src, tgt), conn in snn.connections.items():
        key = f"{src}__{tgt}"
        if key in weights:
            w = weights[key].to(conn.w.device).to(conn.w.dtype)
            if w.shape != conn.w.shape:
                print(f"  shape mismatch on {key}: saved={w.shape} vs current={conn.w.shape}; skipping")
                continue
            conn.w.copy_(w)
            loaded += 1
    print(f"Loaded weights into {loaded}/{len(snn.connections)} connections")


# ==========================================================
# Inference
# ==========================================================
def predict(snn, subjects):
    """Lazy-load each window from `spikes_raw/`, run the SNN, return per-window
    (true, pred, prob_of_seizure) arrays."""
    snn.train(mode=False)
    out_layer_name = "output_layer"

    windows = list(enumerate_windows(subjects, WINDOW_SECONDS, WINDOW_OVERLAP))
    cache = _MinuteCache(capacity=4)

    trues, preds, scores = [], [], []
    for win in tqdm(windows, desc="Evaluating"):
        try:
            inputs, label = _build_window_inputs(win, cache)
        except FileNotFoundError as e:
            print(f"  skip window {win['abs_start_s']:.2f}s: missing file {e}")
            continue
        if not inputs:
            continue

        T = next(iter(inputs.values())).shape[0]
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # Fresh monitor per window so the spike count is per-window only.
        monitor = Monitor(snn.layers[out_layer_name], state_vars=("s",), time=T)
        snn.add_monitor(monitor, name="out_eval")
        try:
            snn.run(inputs=inputs, time=T)
            spikes = monitor.get("s")
            counts = spikes.sum(dim=tuple(range(spikes.ndim - 1))).float().cpu()
            pred = int(counts.argmax().item())
            total = counts.sum().item()
            score = float(counts[1].item() / total) if total > 0 else 0.5
        finally:
            del snn.monitors["out_eval"]
            snn.reset_state_variables()

        trues.append(label)
        preds.append(pred)
        scores.append(score)

    return np.array(trues), np.array(preds), np.array(scores)


# ==========================================================
# Metrics + plots
# ==========================================================
def report_and_plot(trues, preds, scores):
    cm = confusion_matrix(trues, preds, labels=[0, 1])
    report = classification_report(
        trues, preds, target_names=["non_seizure", "seizure"], output_dict=True
    )
    acc = report["accuracy"]
    print(f"\nAccuracy = {acc:.4f}\n")
    print(pd.DataFrame(report).T)

    pd.DataFrame(report).T.to_csv(OUT_DIR / "eval_summary.csv")
    print(f"Saved: {OUT_DIR / 'eval_summary.csv'}")

    plt.figure(figsize=(4, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-Seizure", "Seizure"],
        yticklabels=["Non-Seizure", "Seizure"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix.png")
    plt.close()
    print(f"Saved: {OUT_DIR / 'confusion_matrix.png'}")

    # ROC / PR only make sense if both classes are present.
    if len(np.unique(trues)) == 2:
        fpr, tpr, _ = roc_curve(trues, scores)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / "roc_curve.png")
        plt.close()

        precision, recall, _ = precision_recall_curve(trues, scores)
        plt.figure()
        plt.plot(recall, precision, label="PR Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / "precision_recall_curve.png")
        plt.close()
        print(f"Saved: {OUT_DIR / 'roc_curve.png'}, {OUT_DIR / 'precision_recall_curve.png'}")


# ==========================================================
# Main
# ==========================================================
def main():
    print(f"Device = {DEVICE}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    snn = build_snn()
    snn.to(DEVICE)
    load_weights_into(snn, MODEL_PATH)

    trues, preds, scores = predict(snn, SUBJECTS)
    report_and_plot(trues, preds, scores)


if __name__ == "__main__":
    main()
