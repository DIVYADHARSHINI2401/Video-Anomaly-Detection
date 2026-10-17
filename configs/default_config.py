"""
Default Configuration for ViT Open-Set Video Anomaly Detection
with Risk Severity Estimation.
"""
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class PreprocessingConfig:
    frame_size: Tuple[int, int] = (224, 224)
    fps_sample_rate: int = 2          # Sample 1 frame every 2 frames (matches UCSD benchmark)
    clip_length: int = 16             # Sliding window temporal length (frames)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

@dataclass
class ModelConfig:
    # ViT Backbone: 'vit_tiny_patch16_224', 'vit_small_patch16_224', 'vit_base_patch16_224'
    backbone_name: str = "vit_tiny_patch16_224"
    pretrained: bool = True
    freeze_backbone: bool = True
    feature_dim: int = 192            # 192 for vit_tiny, 384 for vit_small, 768 for vit_base
    
    # Lightweight Temporal Module: 'gru', 'tcn', or 'transformer'
    temporal_module_type: str = "gru"
    hidden_dim: int = 128
    num_temporal_layers: int = 2
    dropout: float = 0.1

@dataclass
class OpenSetConfig:
    # Thresholds for anomaly detection & novelty classification
    anomaly_threshold: float = 0.50   # Anomaly score >= threshold is flagged as anomaly
    unseen_threshold: float = 0.65    # Distance/Novelty score >= threshold flags anomaly as "Unseen"
    
    # Known training categories vs. held-out unseen categories
    known_categories: List[str] = field(default_factory=lambda: [
        "Normal", "Fighting", "Burglary", "Vandalism"
    ])
    unseen_categories: List[str] = field(default_factory=lambda: [
        "Arson", "Robbery", "Accident"
    ])

@dataclass
class RiskSeverityConfig:
    # Severity weightings: w_deviation + w_persistence + w_novelty = 1.0
    w_deviation: float = 0.45
    w_persistence: float = 0.35
    w_novelty: float = 0.20
    
    # Thresholds for categorizing final composite risk score in [0.0, 1.0]
    low_risk_threshold: float = 0.35
    medium_risk_threshold: float = 0.65
    # Above medium_risk_threshold -> High Risk

@dataclass
class PipelineConfig:
    preprocess: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    openset: OpenSetConfig = field(default_factory=OpenSetConfig)
    risk: RiskSeverityConfig = field(default_factory=RiskSeverityConfig)
    device: str = "cuda"              # Will fallback to "cpu" if cuda unavailable
    output_dir: str = "outputs"
