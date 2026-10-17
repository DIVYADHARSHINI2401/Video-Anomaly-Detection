"""
Hardware Efficiency Profiler for Video Anomaly Detection.
Measures parameter counts, inference latency (ms), throughput (FPS), and memory usage.
"""
import time
from typing import Dict, Tuple

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class ModelEfficiencyProfiler:
    """
    Benchmarks model execution speed, throughput, and size.
    """
    @staticmethod
    def count_parameters(model) -> Dict[str, object]:
        """Counts total, trainable, and frozen parameters."""
        if not HAS_TORCH:
            return {
                "total_parameters": 0,
                "trainable_parameters": 0,
                "frozen_parameters": 0,
                "total_millions": 0.0,
                "trainable_millions": 0.0,
                "note": "PyTorch not detected. Install torch to inspect live model parameters."
            }

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": frozen_params,
            "total_millions": round(total_params / 1e6, 2),
            "trainable_millions": round(trainable_params / 1e6, 2)
        }

    @staticmethod
    def benchmark_throughput(
        model,
        input_shape: Tuple[int, ...],
        device: str = "cpu",
        num_warmup: int = 5,
        num_runs: int = 20
    ) -> Dict[str, object]:
        """
        Benchmarks inference latency and throughput (FPS).
        """
        if not HAS_TORCH:
            return {
                "mean_latency_ms": 0.0,
                "min_latency_ms": 0.0,
                "max_latency_ms": 0.0,
                "fps": 0.0,
                "device": device,
                "note": "PyTorch not detected. Install torch to benchmark neural forward pass."
            }

        dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        model.to(dev)
        model.eval()

        dummy_input = torch.randn(*input_shape, device=dev)

        # Warm-up
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(dummy_input)

        if dev.type == "cuda":
            torch.cuda.synchronize()

        # Measurement
        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = model(dummy_input)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                end = time.perf_counter()
                latencies.append((end - start) * 1000.0)  # ms

        mean_latency_ms = float(sum(latencies) / len(latencies))
        
        if len(input_shape) == 5:
            frames_processed = input_shape[0] * input_shape[1]
        else:
            frames_processed = input_shape[0]
            
        fps = float((frames_processed / (mean_latency_ms / 1000.0))) if mean_latency_ms > 0 else 0.0

        return {
            "mean_latency_ms": round(mean_latency_ms, 2),
            "min_latency_ms": round(min(latencies), 2),
            "max_latency_ms": round(max(latencies), 2),
            "fps": round(fps, 1),
            "device": str(dev)
        }
