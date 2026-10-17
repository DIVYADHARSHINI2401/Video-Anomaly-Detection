"""
Phase 2: Comprehensive Environment & Hardware Verification Script.
Checks:
- Python executable & version
- PyTorch & CUDA availability on RTX 3050
- CUDA GPU memory & compute capability
- Live GPU matrix multiplication tensor operation
- Core vision & ML libraries: timm, cv2, sklearn, pandas, scipy
"""
import sys
import platform

print("=" * 65)
print("PHASE 2: COMPREHENSIVE ENVIRONMENT & GPU VERIFICATION")
print("=" * 65)

# 1. Python Environment
print(f"Python Executable: {sys.executable}")
print(f"Python Version:    {sys.version.split()[0]}")
print(f"Platform:          {platform.platform()}")
print("-" * 65)

# 2. PyTorch & CUDA Verification
import torch
print(f"PyTorch Version:   {torch.__version__}")
cuda_ok = torch.cuda.is_available()
print(f"CUDA Available:    {cuda_ok}")

if not cuda_ok:
    print("FATAL: PyTorch CUDA is not available!")
    sys.exit(1)

gpu_count = torch.cuda.device_count()
gpu_name = torch.cuda.get_device_name(0)
gpu_cap = torch.cuda.get_device_capability(0)
total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

print(f"GPU Count:         {gpu_count}")
print(f"GPU Device:        {gpu_name}")
print(f"Compute Capability:{gpu_cap}")
print(f"Total VRAM:        {total_vram_gb:.2f} GB")
print("-" * 65)

# 3. Live CUDA Tensor Operation
print("Testing CUDA Tensor Operation (Matrix Multiplication on GPU)...")
device = torch.device("cuda:0")
x = torch.randn(2048, 2048, device=device, dtype=torch.float32)
y = torch.randn(2048, 2048, device=device, dtype=torch.float32)
z = torch.matmul(x, y)
torch.cuda.synchronize()
alloc_mem_mb = torch.cuda.memory_allocated(0) / (1024**2)
print(f"  Result Shape:    {z.shape}")
print(f"  Result Mean:     {z.mean().item():.5f}")
print(f"  VRAM Allocated:  {alloc_mem_mb:.2f} MB")
print("  CUDA Operation:  SUCCESS")
print("-" * 65)

# 4. Secondary Project Dependencies Verification
print("Verifying Project Vision & ML Dependencies:")
import torchvision
print(f"  torchvision:     {torchvision.__version__}")

import timm
print(f"  timm:            {timm.__version__}")

import cv2
print(f"  OpenCV (cv2):    {cv2.__version__}")

import sklearn
print(f"  scikit-learn:    {sklearn.__version__}")

import pandas as pd
print(f"  pandas:          {pd.__version__}")

import scipy
print(f"  scipy:           {scipy.__version__}")

import tqdm
print(f"  tqdm:            {tqdm.__version__}")

import yaml
print(f"  pyyaml:          {yaml.__version__}")

print("=" * 65)
print("ALL PHASE 2 VERIFICATIONS PASSED SUCCESSFULLY.")
print("=" * 65)
