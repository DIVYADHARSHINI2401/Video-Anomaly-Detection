"""
Neural Network Models for ViT Open-Set Video Anomaly Detection.
"""
from src.models.vit_extractor import ViTFeatureExtractor
from src.models.temporal_aggregator import LightweightTemporalAggregator
from src.models.openset_detector import ReferenceBasedAnomalyDetector
from src.models.risk_estimator import RiskSeverityEstimator

# Backward-compatible alias
OpenSetAnomalyDetector = ReferenceBasedAnomalyDetector

__all__ = [
    "ViTFeatureExtractor",
    "LightweightTemporalAggregator",
    "ReferenceBasedAnomalyDetector",
    "OpenSetAnomalyDetector",
    "RiskSeverityEstimator",
]
