# GPU Setup Guide for Ubuntu

This guide walks you through setting up your Ubuntu machine with CUDA support for running the Epileptic Seizure Classification project.

---

## Prerequisites

- Ubuntu 20.04+ with NVIDIA GPU
- SSH access to the remote machine
- Sudo privileges

---

## Step 1: Install NVIDIA Drivers

Check if drivers are already installed:
```bash
nvidia-smi
```

If not installed, install NVIDIA drivers:
```bash
# Update package lists
sudo apt update

# Install NVIDIA drivers (recommended method)
sudo ubuntu-drivers devices
sudo ubuntu-drivers autoinstall

# Reboot
sudo reboot

# After reboot, verify installation
nvidia-smi
```

You should see your GPU information (model, memory, CUDA version).

---

## Step 2: Install CUDA Toolkit (Optional)

PyTorch comes with its own CUDA runtime, but installing the toolkit can help with debugging:

```bash
# For CUDA 12.x (check your driver's supported version with nvidia-smi)
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
sudo sh cuda_12.4.0_550.54.14_linux.run

# Add to PATH (add to ~/.bashrc for persistence)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

---

## Step 3: Install Python and Dependencies

```bash
# Install Python 3.10+
sudo apt install python3 python3-pip python3-venv

# Create virtual environment
cd /path/to/Epileptic-Seizure-Classification
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

---

## Step 4: Install PyTorch with CUDA

**CRITICAL**: Install PyTorch with CUDA support (NOT the CPU-only version)

Check your CUDA version first:
```bash
nvidia-smi  # Look for "CUDA Version" in the top right
```

Then install PyTorch accordingly:

### For CUDA 12.x (most common):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### For CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Verify installation:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

Should output: `CUDA available: True`

---

## Step 5: Install Project Dependencies

```bash
# Install remaining dependencies
pip install numpy scipy matplotlib mne tqdm seaborn pandas scikit-learn

# Install BindsNet (SNN framework)
pip install bindsnet

# Install SCTN (if available via pip, otherwise clone from repo)
# If SCTN is custom/local, you may need to install from source
```

---

## Step 6: Test GPU Setup

Run the diagnostic script:
```bash
cd src
python check_gpu.py
```

Expected output:
```
============================================================
PyTorch GPU Diagnostic
============================================================

✓ PyTorch version: 2.x.x+cu124
✓ CUDA available: True
✓ CUDA version: 12.4
✓ Number of GPUs: 1
✓ Current GPU: 0
✓ GPU Name: NVIDIA GeForce RTX 4090 (or your GPU model)
✓ GPU Memory: 24.00 GB
✓ GPU computation test: PASSED
============================================================
```

---

## Step 7: Transfer Data to Ubuntu Machine

You have two options:

### Option A: Direct download (if data is public)
```bash
# CHB-MIT dataset from PhysioNet
cd dataset
wget -r -N -c -np https://physionet.org/files/chbmit/1.0.0/
```

### Option B: SCP from your Windows machine
```bash
# From Windows (in PowerShell/CMD)
scp -r "C:\Users\talsh\Desktop\Noam\Msc\Thesa\Epileptic-Seizure-Classification\dataset\chbmit" user@ubuntu-ip:/path/to/project/dataset/

# From Ubuntu, verify:
ls -lh dataset/chbmit/
```

---

## Step 8: Run Training on GPU

```bash
cd src

# Run spike conversion (CPU-intensive, uses multiprocessing)
python 1.signal2spikes_chbmit.py

# Run analysis (CPU-based visualization)
python 2.analyze_and_pre_train_chbmit.py

# Train model on GPU (this will use CUDA automatically)
python 3.run_train_chbmit.py

# Evaluate model on GPU
python 4.run_eval_chbmit.py
```

---

## Performance Optimization Tips

### 1. Increase Batch Size
If you have a good GPU (e.g., RTX 3090, RTX 4090, A100), increase batch size in the training scripts:

Edit `src/3.run_train_chbmit.py`:
```python
BATCH_SIZE = 32  # or 64, 128 depending on GPU memory
```

### 2. Monitor GPU Usage
```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Or use nvtop (more detailed)
sudo apt install nvtop
nvtop
```

### 3. Parallel Data Loading
If data loading is a bottleneck, add workers to the DataLoader:

```python
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,      # Add this
    pin_memory=True     # Add this for faster GPU transfer
)
```

### 4. Mixed Precision Training
Already enabled in the updated code! This uses FP16 for faster training while maintaining FP32 accuracy.

---

## Troubleshooting

### Issue: `CUDA available: False`
**Solution**: Reinstall PyTorch with CUDA support (see Step 4)

### Issue: `RuntimeError: CUDA out of memory`
**Solutions**:
1. Reduce batch size
2. Clear GPU cache: `torch.cuda.empty_cache()`
3. Use gradient accumulation for effective larger batch sizes

### Issue: Slow training despite GPU
**Possible causes**:
1. Data loading bottleneck (add `num_workers` to DataLoader)
2. Small batch size (GPU not fully utilized)
3. CPU-GPU transfer overhead (use `pin_memory=True`)

### Issue: `nvidia-smi` not found
**Solution**: NVIDIA drivers not installed correctly. Reinstall (see Step 1)

---

## Expected Speedup

With a modern GPU (e.g., RTX 3090/4090), expect:
- **Training**: 5-10x faster than CPU
- **Evaluation**: 10-20x faster than CPU
- **Stage 1 (spike conversion)**: No speedup (CPU-bound, uses multiprocessing)

---

## Remote Development Tips

### SSH with Port Forwarding (for Jupyter)
```bash
ssh -L 8888:localhost:8888 user@ubuntu-ip
```

### Running in Background (tmux/screen)
```bash
# Install tmux
sudo apt install tmux

# Start session
tmux new -s training

# Run training
python 3.run_train_chbmit.py

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t training
```

### Monitor Training Remotely
```bash
# Use tensorboard (if you add logging)
tensorboard --logdir=output_data/train_logs --host=0.0.0.0 --port=6006

# Then access from Windows: http://ubuntu-ip:6006
```

---

## Summary of Changes Made

1. ✅ Added GPU detection and diagnostics ([src/check_gpu.py](src/check_gpu.py))
2. ✅ Added GPU memory optimization ([src/3.run_train_chbmit.py](src/3.run_train_chbmit.py))
3. ✅ Added mixed precision training for faster performance
4. ✅ Updated evaluation script with GPU optimizations ([src/4.run_eval_chbmit.py](src/4.run_eval_chbmit.py))
5. ✅ Updated requirements.txt with CUDA installation notes

**No changes needed** in:
- `1.signal2spikes_chbmit.py` (CPU multiprocessing, cannot be GPU-accelerated easily)
- `2.analyze_and_pre_train_chbmit.py` (CPU-based visualization)
- `utils.py` (BindsNet layers automatically use GPU when model is on GPU)

The code will **automatically detect and use GPU** when available. If CUDA is not available, it falls back to CPU gracefully.

---

## Questions?

If you encounter issues, check:
1. `nvidia-smi` output
2. `python src/check_gpu.py` diagnostic results
3. PyTorch installation: `pip show torch` (should show `+cu124` or similar)
