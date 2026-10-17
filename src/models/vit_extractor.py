"""
Vision Transformer (ViT) Feature Extractor Module.
Loads genuine pretrained Vision Transformer weights from `timm` and extracts
compact spatial representations without mock fallbacks or random weights.
"""
import os
import sys
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

try:
    import timm
except ImportError as err:
    raise RuntimeError(
        "The 'timm' package is required for ViT feature extraction. "
        "Please install it in your environment: pip install timm"
    ) from err


class ViTFeatureExtractor(nn.Module):
    """
    Extracts spatial representation vectors from input video frames using
    an official pretrained Vision Transformer backbone from `timm`.
    """
    def __init__(
        self,
        backbone_name: str = "vit_tiny_patch16_224",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        feature_dim: Optional[int] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.device = torch.device(device)

        if not self.pretrained:
            raise ValueError(
                "ViTFeatureExtractor requires 'pretrained=True'. "
                "Randomly initialized Vision Transformers must not be silently used "
                "for feature extraction in this pipeline."
            )

        print(f"[ViTFeatureExtractor] Loading pretrained backbone '{backbone_name}' on {self.device}...")

        try:
            # num_classes=0 outputs the pooled feature representation (CLS token or global pool)
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=True,
                num_classes=0
            )
            self.output_dim = self.backbone.num_features

        except torch.cuda.OutOfMemoryError as oom_err:
            raise RuntimeError(
                f"Insufficient GPU VRAM to allocate backbone '{backbone_name}' on device '{self.device}'. "
                "Try using 'vit_tiny_patch16_224' or running on 'cpu'."
            ) from oom_err

        except Exception as load_err:
            cache_dir = os.environ.get("TORCH_HOME", os.path.join(os.path.expanduser("~"), ".cache", "torch"))
            hf_cache_dir = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
            raise RuntimeError(
                f"Failed to load pretrained weights for '{backbone_name}'.\n"
                f"Error Details: {load_err}\n"
                "Troubleshooting Steps:\n"
                "  1. Check network connectivity to Hugging Face / GitHub releases.\n"
                f"  2. Inspect or clear potentially corrupt cached weights in:\n"
                f"     - Torch Cache: {cache_dir}\n"
                f"     - HuggingFace Cache: {hf_cache_dir}\n"
                "  3. Ensure no firewall/proxy is blocking SSL connections."
            ) from load_err

        # Freeze backbone parameters
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        # Optional linear adapter to match downstream dimension if explicitly configured
        if feature_dim is not None and feature_dim != self.output_dim:
            self.adapter = nn.Sequential(
                nn.Linear(self.output_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.ReLU()
            )
            self.final_dim = feature_dim
        else:
            self.adapter = nn.Identity()
            self.final_dim = self.output_dim

        self.to(self.device)

        # Retrieve pretrained model normalization statistics
        data_cfg = timm.data.resolve_model_data_config(self.backbone)
        self.input_size = data_cfg.get("input_size", (3, 224, 224))[1:]
        self.mean = np.array(data_cfg.get("mean", (0.485, 0.456, 0.406)), dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(data_cfg.get("std", (0.229, 0.224, 0.225)), dtype=np.float32).reshape(1, 1, 3)

    def preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocesses a single image (OpenCV BGR ndarray or PIL RGB Image).
        Performs RGB conversion, resize to input_size (224x224), ImageNet normalization.
        Returns:
            tensor: Shape (3, H, W) float32 tensor
        """
        if isinstance(image, Image.Image):
            rgb_img = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        elif isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"Expected 3-channel image, got shape {image.shape}")
            # OpenCV loads as BGR; convert to RGB
            rgb_img = image[:, :, ::-1].astype(np.float32) / 255.0
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Resize to model input dimensions
        target_h, target_w = self.input_size
        if rgb_img.shape[0] != target_h or rgb_img.shape[1] != target_w:
            import cv2
            rgb_img = cv2.resize(rgb_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Standardize: (X - mean) / std
        norm_img = (rgb_img - self.mean) / self.std
        # HWC -> CHW
        tensor = torch.from_numpy(norm_img.transpose(2, 0, 1)).float()
        return tensor

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Extract features from preprocessed frame tensors.
        Args:
            frames: Tensor of shape:
                    - (3, H, W)          -> Single Frame
                    - (B, 3, H, W)       -> Batch of Frames
                    - (B, T, 3, H, W)    -> Batch of Temporal Clips
        Returns:
            features: Tensor of shape:
                    - (1, final_dim)
                    - (B, final_dim)
                    - (B, T, final_dim)
        """
        if not isinstance(frames, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(frames)}")

        orig_ndim = frames.ndim
        if orig_ndim == 3:
            # (3, H, W) -> (1, 3, H, W)
            frames = frames.unsqueeze(0)
            orig_ndim = 4

        is_temporal = (orig_ndim == 5)
        if is_temporal:
            b, t, c, h, w = frames.shape
            frames = frames.reshape(b * t, c, h, w)

        frames = frames.to(self.device, non_blocking=True)

        try:
            with torch.inference_mode():
                raw_feats = self.backbone(frames)
                projected = self.adapter(raw_feats)

        except torch.cuda.OutOfMemoryError as oom_err:
            raise RuntimeError(
                f"GPU Out-Of-Memory while processing frame batch of shape {frames.shape} on {self.device}. "
                "Reduce batch size or clip length."
            ) from oom_err

        # Validate numerical integrity (no NaN or Inf values from model output)
        if not torch.isfinite(projected).all():
            raise FloatingPointError("Model output contains non-finite values (NaN or Inf).")

        if is_temporal:
            projected = projected.view(b, t, -1)

        return projected
