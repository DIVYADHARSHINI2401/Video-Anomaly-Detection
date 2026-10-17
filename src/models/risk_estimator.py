"""
Risk Severity Estimation Module.
Computes multi-tier operational alert levels (Low / Medium / High) from anomaly scores,
temporal persistence, and open-set novelty factors to mitigate surveillance alert fatigue.
"""
from typing import Dict, List, Union
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = object


class RiskSeverityEstimator(nn.Module if HAS_TORCH else object):
    """
    Translates raw anomaly detections into practical operational risk priorities.
    """
    def __init__(
        self,
        w_deviation: float = 0.45,
        w_persistence: float = 0.35,
        w_novelty: float = 0.20,
        low_threshold: float = 0.35,
        medium_threshold: float = 0.65
    ):
        if HAS_TORCH:
            super().__init__()
            
        total_w = w_deviation + w_persistence + w_novelty
        self.w_deviation = w_deviation / total_w
        self.w_persistence = w_persistence / total_w
        self.w_novelty = w_novelty / total_w
        
        self.low_threshold = low_threshold
        self.medium_threshold = medium_threshold

    def calculate_persistence(
        self,
        recent_scores: Union[List[float], np.ndarray]
    ) -> float:
        """
        Computes temporal persistence score: measures the duration and consistency
        of anomalous activity over a rolling temporal history window.
        """
        if len(recent_scores) == 0:
            return 0.0
        if HAS_TORCH and isinstance(recent_scores, torch.Tensor):
            arr = recent_scores.detach().cpu().numpy()
        else:
            arr = np.array(recent_scores)
        
        mean_score = float(np.mean(arr))
        consistency = float(np.sum(arr > 0.4) / len(arr))
        persistence = 0.6 * mean_score + 0.4 * consistency
        return min(max(persistence, 0.0), 1.0)

    def estimate_risk(
        self,
        anomaly_score: float,
        novelty_score: float,
        recent_anomaly_history: List[float]
    ) -> Dict[str, Union[float, str]]:
        """
        Estimates composite operational risk severity.
        """
        persistence_score = self.calculate_persistence(recent_anomaly_history)
        
        composite_score = (
            self.w_deviation * float(anomaly_score) +
            self.w_persistence * persistence_score +
            self.w_novelty * float(novelty_score)
        )
        composite_score = min(max(composite_score, 0.0), 1.0)

        # Categorize into priority tier
        if composite_score < self.low_threshold:
            severity = "Low"
            action = "Routine Log / Background Monitor"
            badge_color = (0, 200, 0)      # Green (BGR for OpenCV)
        elif composite_score < self.medium_threshold:
            severity = "Medium"
            action = "Operator Visual Verification Recommended"
            badge_color = (0, 200, 255)    # Amber/Yellow (BGR)
        else:
            severity = "High"
            action = "Urgent: Immediate Security Dispatch Alert"
            badge_color = (0, 0, 255)      # Red (BGR)

        return {
            "composite_risk_score": round(composite_score, 4),
            "severity_level": severity,
            "deviation_score": round(float(anomaly_score), 4),
            "persistence_score": round(persistence_score, 4),
            "novelty_score": round(float(novelty_score), 4),
            "recommended_action": action,
            "badge_color": badge_color,
            "validation_status": "EXPERIMENTAL — UNVALIDATED",
            "metric_type": "heuristic_composite_risk"
        }
