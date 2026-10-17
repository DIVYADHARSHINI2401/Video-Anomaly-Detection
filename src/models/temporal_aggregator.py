"""
Lightweight Temporal Feature Aggregator Module.
Combines frame-level ViT embeddings across temporal windows to model dynamic
surveillance behaviors without heavy 3D-CNN overhead.
"""
from typing import Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = object


class TemporalTCNBlock(nn.Module if HAS_TORCH else object):
    """1D Temporal Convolutional Block with Residual Connection."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1):
        if HAS_TORCH:
            super().__init__()
            padding = (kernel_size - 1) * dilation // 2
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
            self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        res = self.downsample(x)
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + res)


class LightweightTemporalAggregator(nn.Module if HAS_TORCH else object):
    """
    Lightweight Temporal Aggregator supporting GRU, TCN, and Self-Attention Pooling.
    """
    def __init__(
        self,
        input_dim: int = 192,
        hidden_dim: int = 128,
        num_layers: int = 2,
        module_type: str = "gru",
        dropout: float = 0.1
    ):
        self.module_type = module_type.lower()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if HAS_TORCH:
            super().__init__()
            if self.module_type == "gru":
                self.temporal_net = nn.GRU(
                    input_size=input_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    bidirectional=True,
                    dropout=dropout if num_layers > 1 else 0.0
                )
                self.proj = nn.Linear(hidden_dim * 2, hidden_dim)

            elif self.module_type == "tcn":
                layers = []
                curr_in = input_dim
                for i in range(num_layers):
                    layers.append(TemporalTCNBlock(curr_in, hidden_dim, kernel_size=3, dilation=2**i))
                    curr_in = hidden_dim
                self.temporal_net = nn.Sequential(*layers)
                self.proj = nn.Linear(hidden_dim, hidden_dim)

            elif self.module_type == "attention":
                self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4, batch_first=True)
                self.proj = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            elif self.module_type in ("mean", "avg", "pooling"):
                self.temporal_net = nn.Identity()
                self.proj = nn.Identity() if input_dim == hidden_dim else nn.Linear(input_dim, hidden_dim)
            else:
                raise ValueError(f"Unsupported temporal module type: {module_type}")

            self.norm = nn.LayerNorm(hidden_dim)

    def eval(self):
        if HAS_TORCH and hasattr(super(), "eval"):
            super().eval()
        return self

    def to(self, *args, **kwargs):
        if HAS_TORCH and hasattr(super(), "to"):
            super().to(*args, **kwargs)
        return self

    def forward(self, x):
        """
        Args:
            x: Input feature sequence of shape (Batch, Time, input_dim)
        """
        if not HAS_TORCH:
            # NumPy temporal pooling fallback
            if isinstance(x, np.ndarray):
                # x: (Batch, Time, input_dim)
                # Mean pool over time + linear slice
                agg = x.mean(axis=1)[:, :self.hidden_dim]
                if agg.shape[-1] < self.hidden_dim:
                    pad = np.zeros((agg.shape[0], self.hidden_dim - agg.shape[-1]), dtype=np.float32)
                    agg = np.concatenate([agg, pad], axis=-1)
                return agg, x
            return None, None

        if self.module_type == "gru":
            out_seq, _ = self.temporal_net(x)
            proj_seq = self.proj(out_seq)
            aggregated = proj_seq.mean(dim=1)
            
        elif self.module_type == "tcn":
            x_perm = x.permute(0, 2, 1)
            tcn_out = self.temporal_net(x_perm)
            proj_seq = self.proj(tcn_out.permute(0, 2, 1))
            aggregated = proj_seq.mean(dim=1)

        elif self.module_type == "attention":
            attn_out, _ = self.attn(x, x, x)
            proj_seq = self.proj(attn_out)
            aggregated = proj_seq.mean(dim=1)

        elif self.module_type in ("mean", "avg", "pooling"):
            proj_seq = self.proj(x)
            aggregated = proj_seq.mean(dim=1)

        aggregated = self.norm(aggregated)
        proj_seq = self.norm(proj_seq)
        return aggregated, proj_seq

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
