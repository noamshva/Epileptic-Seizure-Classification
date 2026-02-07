# Quick Start: GPU Setup for Ubuntu

**TL;DR**: Your code is already GPU-ready! Just install CUDA-enabled PyTorch on your Ubuntu machine.

---

## On Your Ubuntu Machine: 3 Simple Steps

### Step 1: Check GPU
```bash
nvidia-smi
```
Should show your GPU info. If not, install NVIDIA drivers first.

### Step 2: Install PyTorch with CUDA
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA 12.x (check your CUDA version with nvidia-smi)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Verify
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Step 3: Install Other Dependencies
```bash
pip install numpy scipy matplotlib mne tqdm seaborn pandas scikit-learn bindsnet
# Install SCTN according to its installation instructions
```

---

## Test GPU
```bash
cd src
python check_gpu.py
```

Should output:
```
✓ CUDA available: True
✓ GPU Name: [Your GPU Model]
✓ GPU computation test: PASSED
```

---

## Run Training on GPU
```bash
python 3.run_train_chbmit.py  # Will automatically use GPU
```

---

## What Changed in Your Code?

### Files Modified:
1. **[src/3.run_train_chbmit.py](src/3.run_train_chbmit.py)** - Added GPU detection, memory optimization, mixed precision training
2. **[src/4.run_eval_chbmit.py](src/4.run_eval_chbmit.py)** - Added GPU detection and memory optimization
3. **[requirements.txt](requirements.txt)** - Added notes about CUDA installation

### Files Created:
1. **[src/check_gpu.py](src/check_gpu.py)** - GPU diagnostic script
2. **[GPU_SETUP_GUIDE.md](GPU_SETUP_GUIDE.md)** - Comprehensive setup guide
3. **This file** - Quick reference

### No Changes Needed:
- `1.signal2spikes_chbmit.py` - Uses CPU multiprocessing (not GPU-acceleratable)
- `2.analyze_and_pre_train_chbmit.py` - CPU-based visualization
- `utils.py` - BindsNet layers work on GPU automatically when model is moved to GPU

---

## Key Points

✅ **Your code already had basic GPU support** - just needed optimization
✅ **Automatic fallback to CPU** if CUDA not available
✅ **Mixed precision training** enabled for 2x speedup
✅ **No code changes needed** when switching between Windows and Ubuntu

The DEVICE variable (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`) handles everything automatically!

---

## Performance Tips

| Setting | Current | Recommended for GPU |
|---------|---------|-------------------|
| Batch size (train) | 8 | 32-64 |
| Batch size (eval) | 16 | 64-128 |
| Mixed precision | ✅ Enabled | Keep enabled |
| DataLoader workers | 0 | 4-8 |

To increase batch size, edit the `BATCH_SIZE` variable in the training/eval scripts.

---

## Troubleshooting

**Problem**: `torch.cuda.is_available()` returns False
**Solution**: Reinstall PyTorch with `--index-url https://download.pytorch.org/whl/cu124`

**Problem**: Out of memory errors
**Solution**: Reduce `BATCH_SIZE`

**Problem**: Training not faster on GPU
**Solution**: Increase `BATCH_SIZE` (GPU not fully utilized with small batches)

---

## Full Details

See [GPU_SETUP_GUIDE.md](GPU_SETUP_GUIDE.md) for comprehensive setup instructions, troubleshooting, and optimization tips.
