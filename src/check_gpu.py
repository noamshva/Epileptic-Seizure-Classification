#!/usr/bin/env python3
"""
Quick script to verify CUDA/GPU availability for PyTorch
Run this on your Ubuntu machine after installing PyTorch with CUDA support
"""

import torch

print("=" * 60)
print("PyTorch GPU Diagnostic")
print("=" * 60)

print(f"\n✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
    print(f"✓ Current GPU: {torch.cuda.current_device()}")
    print(f"✓ GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Test tensor creation on GPU
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print(f"✓ GPU computation test: PASSED")
    except Exception as e:
        print(f"✗ GPU computation test: FAILED - {e}")
else:
    print("\n⚠️  WARNING: CUDA not available!")
    print("   - Check if nvidia-smi works")
    print("   - Reinstall PyTorch with CUDA support:")
    print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")

print("=" * 60)
