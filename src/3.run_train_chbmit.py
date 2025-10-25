import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import sys, types
if "torch._six" not in sys.modules:
    m = types.ModuleType("torch._six")
    m.container_abcs = ()
    m.string_classes = (str,)
    m.int_classes = (int,)
    sys.modules["torch._six"] = m
    
from utils import Topographies2SNN, Bands2TopographyLayer, Resonators2BandsLayer
import pickle

# ==========================================================
# 1. Configuration
# ==========================================================
DATASET_PATH = Path("../dataset/spikes_windowed")
OUTPUT_MODEL_DIR = Path("../output_data/trained_models")
OUTPUT_LOG_DIR = Path("../output_data/train_logs")

EPOCHS = 10
BATCH_SIZE = 8
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Base resonator frequencies
full_resonator_array = torch.tensor([
    1.10, 1.30, 1.60, 1.90, 2.20, 2.50, 2.88, 3.05, 3.39,
    3.70, 4.12, 4.62, 5.09, 5.45, 5.87, 6.36, 6.80, 7.60, 8.60
], dtype=torch.float32)

# full_resonator_array = torch.tensor([
#     1.1, 1.3, 1.6, 1.9, 2.2, 2.5, 2.88, 3.05, 3.39, 3.7, 4.12, 4.62, 5.09, 5.45, 5.87, 6.36, 6.8, 7.6, 8.6,
#     10.5, 11.5, 12.8, 15.8, 16.6, 19.4, 22.0, 24.8, 28.4, 30.5, 34.7, 37.2, 40.2, 43.2, 47.7, 52.6, 57.2
# ], dtype=torch.float32)


bands = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 14),
    "Beta": (14, 32),
    "Gamma": (32, 62),
}

channels = [
    "O2","P8","T8","FC6","F4","F8","AF4","AF3",
    "F7","F3","FC5","T7","P7","O1"
]
N = 11
xs = (np.array([-0.7,-0.66,0,0.2,0.4,0.5,0.7,0.7,0.5,0.4,0.2,0,-0.66,-0.7]) + 1) / 2
ys = (-np.array([0.2,0.6,0.95,0.6,0.3,0.6,0.2,-0.2,-0.6,-0.3,-0.6,-0.95,-0.6,-0.2]) + 1) / 2
ch_pos = {ch: (x, y) for ch, (x, y) in zip(channels, zip(ys, xs))}
ch_pos_N = {k: (int(x * N), int(y * N)) for k, (x, y) in ch_pos.items()}

# ==========================================================
# 2. Model definition
# ==========================================================
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
    """Simple classifier head on top of SNN band power outputs"""
    def __init__(self, snn_out_dim, num_classes=2):
        super().__init__()
        self.snn = build_snn()
        self.fc = nn.Sequential(
            nn.Linear(snn_out_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: [B, features]
        return self.fc(x)


# ==========================================================
# 3. Dataset loader
# ==========================================================
def load_dataset(path):
    files = list(Path(path).rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found under {path.resolve()}")

    X, y = [], []
    for f in tqdm(files, desc=f"Loading dataset from {path.name}"):
        label = 1 if "seizure" in f.parts[-2].lower() else 0
        try:
            # load pickled dict
            with open(f, "rb") as fp:
                data = pickle.load(fp)
            # each dict entry: "<ch>-<band>" -> torch.Tensor [n_resonators, time]
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

    if not X:
        raise RuntimeError(f"No valid samples loaded from {path}")
    X = np.stack(X)
    y = np.array(y)
    print(f"✅ Loaded {len(X)} samples ({sum(y)} seizure, {len(y)-sum(y)} non-seizure)")
    return X, y



# ==========================================================
# 4. Training loop
# ==========================================================
def train_model(model, X, y):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long)
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    epoch_accs, epoch_losses = [], []

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct = 0, 0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_X.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_y).sum().item()

        acc = correct / len(dataset)
        avg_loss = total_loss / len(dataset)

        epoch_accs.append(acc)
        epoch_losses.append(avg_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss={total_loss/len(dataset):.4f} | Acc={acc:.4f}")

    # Save model
    model_path = OUTPUT_MODEL_DIR / "trained_chbmit_model.pth"
    torch.save(model.state_dict(), model_path)

    log_df = pd.DataFrame({"epoch": np.arange(1, EPOCHS+1), "accuracy": epoch_accs, "loss": epoch_losses})
    log_csv = OUTPUT_LOG_DIR / "training_log.csv"
    log_df.to_csv(log_csv, index=False)

    print(f"\n✅ Training completed.")
    print(f"   - Model saved to: {model_path}")
    print(f"   - Training log saved to: {log_csv}")


# ==========================================================
# 5. Main
# ==========================================================
if __name__ == "__main__":
    print(f"Running on Device = {DEVICE}")
    X, y = load_dataset(DATASET_PATH)
    print(f"Loaded {len(X)} samples with shape {X.shape[1:]}")

    model = EEGClassifier(snn_out_dim=X.shape[1])
    train_model(model, X, y)
