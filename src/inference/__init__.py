"""
End-to-End Inference Pipeline and Video Annotation.
"""
from src.inference.pipeline import VideoAnomalyInferencePipeline
from src.inference.video_annotator import SurveillanceVideoAnnotator

__all__ = ["VideoAnomalyInferencePipeline", "SurveillanceVideoAnnotator"]
