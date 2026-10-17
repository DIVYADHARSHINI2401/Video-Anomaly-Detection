"""
Speed and Efficiency Benchmark Script.
Evaluates parameter count, inference latency (ms), and frame throughput (FPS)
for ViT backbones and temporal aggregation modules.
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.models.vit_extractor import ViTFeatureExtractor
from src.models.temporal_aggregator import LightweightTemporalAggregator
from src.eval.efficiency import ModelEfficiencyProfiler


class CombinedPipelineBenchmarkModel:
    """Coupled model for end-to-end throughput benchmarking."""
    def __init__(self, vit_backbone: str = "vit_tiny_patch16_224", temporal_type: str = "gru", device: str = "cpu"):
        self.vit = ViTFeatureExtractor(backbone_name=vit_backbone, pretrained=True, device=device)
        self.temporal = LightweightTemporalAggregator(
            input_dim=self.vit.final_dim,
            hidden_dim=128,
            module_type=temporal_type
        )
        if HAS_TORCH:
            self.temporal = self.temporal.to(device)

    def eval(self):
        self.vit.eval()
        self.temporal.eval()
        return self

    def to(self, *args, **kwargs):
        if HAS_TORCH and hasattr(self.temporal, "to"):
            self.temporal.to(*args, **kwargs)
        return self

    def parameters(self):
        if HAS_TORCH:
            for p in self.vit.parameters() if hasattr(self.vit, "parameters") else []:
                yield p
            for p in self.temporal.parameters() if hasattr(self.temporal, "parameters") else []:
                yield p
        return iter([])

    def __call__(self, clips):
        feats = self.vit(clips)
        agg, _ = self.temporal(feats)
        return agg


def main():
    parser = argparse.ArgumentParser(description="Benchmark ViT + Temporal Aggregator Efficiency")
    parser.add_argument("--backbone", type=str, default="vit_tiny_patch16_224",
                        choices=["vit_tiny_patch16_224", "vit_small_patch16_224", "vit_base_patch16_224"])
    parser.add_argument("--temporal", type=str, default="gru", choices=["gru", "tcn", "attention"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--clip_length", type=int, default=16)
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()

    print("=" * 65)
    print("SURVEILLANCE ANOMALY DETECTION: EFFICIENCY & SPEED BENCHMARK")
    print("=" * 65)
    print(f"Backbone:        {args.backbone}")
    print(f"Temporal Module: {args.temporal.upper()}")
    print(f"Device:          {args.device}")
    print(f"Clip Window:     {args.clip_length} frames")
    print("-" * 65)

    if not HAS_TORCH:
        print("[Notice] PyTorch not installed in the active environment.")
        print("  - To run live GPU/CPU tensor forward-pass profiling:")
        print("    pip install torch torchvision timm")
        print("\nTheoretical Architectural Profiles:")
        print(f"  - Model: {args.backbone} + {args.temporal.upper()} Temporal Aggregator")
        print(f"  - Parameters (vit_tiny):  ~5.7 Million (Frozen Backbone) + ~0.2M (Aggregator)")
        print(f"  - Parameters (vit_small): ~22.0 Million + ~0.4M")
        print(f"  - Parameters (vit_base):  ~86.5 Million + ~0.8M")
        print(f"  - Target Edge Throughput: 45 - 90+ FPS on edge GPU / 15 - 30 FPS on multi-core CPU")
        print("=" * 65)
        return

    model = CombinedPipelineBenchmarkModel(
        vit_backbone=args.backbone,
        temporal_type=args.temporal,
        device=args.device
    )

    # 1. Parameter Analysis
    params = ModelEfficiencyProfiler.count_parameters(model)
    print(f"Total Parameters:     {params['total_parameters']:,} ({params['total_millions']} M)")
    print(f"Trainable Parameters: {params['trainable_parameters']:,} ({params['trainable_millions']} M)")
    print(f"Frozen Backbone:      {params['frozen_parameters']:,}")
    print("-" * 65)

    # 2. Latency & Throughput Benchmark
    input_shape = (1, args.clip_length, 3, 224, 224)
    print(f"Benchmarking inference latency (warmup=5, runs={args.runs})...")
    perf = ModelEfficiencyProfiler.benchmark_throughput(
        model=model,
        input_shape=input_shape,
        device=args.device,
        num_warmup=5,
        num_runs=args.runs
    )

    print(f"Mean Clip Latency:    {perf['mean_latency_ms']} ms")
    print(f"Min / Max Latency:    {perf['min_latency_ms']} ms / {perf['max_latency_ms']} ms")
    print(f"Estimated Throughput: {perf['fps']} FPS")
    print("=" * 65)


if __name__ == "__main__":
    main()
