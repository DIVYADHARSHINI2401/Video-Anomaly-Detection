"""
Standalone Smoke Test for Pretrained Vision Transformer Feature Extractor.
Loads the official pretrained timm backbone, verifies weights on CPU and CUDA (RTX 3050),
tests forward pass on test image and batched clips, and verifies numerical integrity.
"""
import os
import sys
import numpy as np

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.models.vit_extractor import ViTFeatureExtractor


def run_smoke_test(device_str: str = "cuda" if torch.cuda.is_available() else "cpu"):
    print("=" * 65)
    print(f"VIT FEATURE EXTRACTOR SMOKE TEST (Target Device: {device_str})")
    print("=" * 65)

    # 1. Inspect weight cache directories
    torch_cache = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")
    hf_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    print("Model Weight Cache Locations:")
    print(f"  Torch Checkpoints: {torch_cache} (Exists: {os.path.exists(torch_cache)})")
    print(f"  HuggingFace Hub:   {hf_cache} (Exists: {os.path.exists(hf_cache)})")
    print("-" * 65)

    # 2. Instantiate extractor with real pretrained weights
    backbone = "vit_tiny_patch16_224"
    print(f"Instantiating ViTFeatureExtractor with backbone='{backbone}', pretrained=True...")
    
    try:
        extractor = ViTFeatureExtractor(
            backbone_name=backbone,
            pretrained=True,
            freeze_backbone=True,
            device=device_str
        )
    except Exception as err:
        print(f"\n[FAILURE] Failed to initialize pretrained extractor: {err}")
        return False

    print(f"  Backbone Name:   {extractor.backbone_name}")
    print(f"  Device:          {extractor.device}")
    print(f"  Output Feature Dim: {extractor.output_dim}")
    print(f"  Input Resolution:   {extractor.input_size}")
    print(f"  Param Count:     {sum(p.numel() for p in extractor.parameters()):,}")
    print(f"  Trainable Params:{sum(p.numel() for p in extractor.parameters() if p.requires_grad):,}")
    print("-" * 65)

    # 3. Create a synthetic test image (Clearly labeled)
    print("[NOTE] Generating a 224x224 RGB test pattern:")
    print("       * TYPE: SYNTHETIC TEST PATTERN (Gradient + geometric shapes)")
    print("       * PURPOSE: Smoke-testing tensor dimensions & finite values only.")
    print("       * DISCLAIMER: This is NOT a real surveillance/CCTV frame.")

    # Create synthetic gradient test image
    test_img = np.zeros((224, 224, 3), dtype=np.uint8)
    for y in range(224):
        for x in range(224):
            test_img[y, x] = [x % 256, y % 256, (x + y) % 256]

    # Preprocess via extractor
    tensor_single = extractor.preprocess_image(test_img)
    print(f"\nPreprocessed Single Frame Shape: {tensor_single.shape} (dtype: {tensor_single.dtype})")
    print(f"  Mean: {tensor_single.mean().item():.4f}, Std: {tensor_single.std().item():.4f}")

    # 4. Single Frame Forward Pass
    print("\nRunning Forward Pass on Single Frame...")
    feat_single = extractor(tensor_single)
    print(f"  Extracted Feature Shape: {feat_single.shape}")
    print(f"  Feature Mean:            {feat_single.mean().item():.4f}")
    print(f"  Feature Min / Max:       {feat_single.min().item():.4f} / {feat_single.max().item():.4f}")
    
    if not torch.isfinite(feat_single).all():
        print("[FAILURE] Extracted features contain NaN or Inf values!")
        return False
    print("  Numerical Integrity:     PASSED (All values finite)")

    # 5. Batched Temporal Clip Forward Pass (B=2, T=8)
    print("\nRunning Forward Pass on Batched Temporal Clips (Batch=2, Time=8 frames)...")
    batch_clips = tensor_single.unsqueeze(0).unsqueeze(0).repeat(2, 8, 1, 1, 1)
    print(f"  Input Batch Shape:       {batch_clips.shape}")
    
    feat_batch = extractor(batch_clips)
    print(f"  Output Feature Shape:    {feat_batch.shape} (Expected: [2, 8, {extractor.final_dim}])")

    if not torch.isfinite(feat_batch).all():
        print("[FAILURE] Batched features contain NaN or Inf values!")
        return False
    print("  Batch Numerical Integrity: PASSED (All values finite)")

    # 6. Check GPU Memory if running on CUDA
    if extractor.device.type == "cuda":
        alloc_mb = torch.cuda.memory_allocated(extractor.device) / (1024**2)
        reserved_mb = torch.cuda.memory_reserved(extractor.device) / (1024**2)
        print("-" * 65)
        print("CUDA Memory Footprint on RTX 3050:")
        print(f"  Allocated VRAM: {alloc_mb:.2f} MB")
        print(f"  Reserved VRAM:  {reserved_mb:.2f} MB")

    print("=" * 65)
    print(f"SUCCESS: Pretrained Vision Transformer smoke test PASSED on {device_str}.")
    print("=================================================================\n")
    return True


if __name__ == "__main__":
    # Test CPU first, then CUDA if available
    print("Running initial CPU verification...")
    cpu_ok = run_smoke_test("cpu")
    if not cpu_ok:
        sys.exit(1)

    if torch.cuda.is_available():
        print("\nRunning CUDA GPU verification on NVIDIA RTX 3050...")
        cuda_ok = run_smoke_test("cuda:0")
        if not cuda_ok:
            sys.exit(1)
    else:
        print("CUDA unavailable, skipping GPU run.")
