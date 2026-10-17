"""
Full Inference Pipeline for Video Anomaly Detection.
Integrates frame sampling, ViT feature extraction, temporal aggregation,
reference-based distance anomaly detection, risk estimation, and structured logging (CSV/JSON).
"""
import json
import os
import time
import warnings
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from configs.default_config import PipelineConfig
from src.data.frame_sampler import VideoFrameSampler
from src.models.vit_extractor import ViTFeatureExtractor
from src.models.temporal_aggregator import LightweightTemporalAggregator
from src.models.openset_detector import ReferenceBasedAnomalyDetector
from src.models.risk_estimator import RiskSeverityEstimator
from src.inference.video_annotator import SurveillanceVideoAnnotator


SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov"}


class VideoAnomalyInferencePipeline:
    """
    Unified end-to-end inference engine for CCTV video anomaly detection.
    Enforces reference-based distance anomaly detection with explicit experimental labeling.
    """
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        reference_checkpoint: Optional[str] = None
    ):
        self.config = config or PipelineConfig()
        
        # Hardware selection
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() and self.config.device == "cuda" else "cpu"
        )
        print(f"[Inference Pipeline] Initializing on device: {self.device}")

        # 1. Frame Sampler
        self.sampler = VideoFrameSampler(
            frame_size=self.config.preprocess.frame_size,
            sample_stride=self.config.preprocess.fps_sample_rate,
            clip_length=self.config.preprocess.clip_length,
            mean=self.config.preprocess.mean,
            std=self.config.preprocess.std
        )

        # 2. Pretrained ViT Feature Extractor
        self.vit_extractor = ViTFeatureExtractor(
            backbone_name=self.config.model.backbone_name,
            pretrained=self.config.model.pretrained,
            freeze_backbone=self.config.model.freeze_backbone,
            feature_dim=self.config.model.feature_dim,
            device=str(self.device)
        )

        # 3. Lightweight Temporal Aggregator
        self.temporal_aggregator = LightweightTemporalAggregator(
            input_dim=self.vit_extractor.final_dim,
            hidden_dim=self.config.model.hidden_dim,
            num_layers=self.config.model.num_temporal_layers,
            module_type=self.config.model.temporal_module_type,
            dropout=self.config.model.dropout
        ).to(self.device)

        # 4. Reference-Based Anomaly Detector
        self.openset_detector = ReferenceBasedAnomalyDetector(
            embedding_dim=self.config.model.hidden_dim,
            anomaly_threshold=self.config.openset.anomaly_threshold,
            unseen_threshold=self.config.openset.unseen_threshold,
            known_class_names=self.config.openset.known_categories
        ).to(self.device)

        # Track path used for reference fitting to detect train/test leakage
        self.reference_source_path: Optional[str] = None

        if reference_checkpoint:
            if not os.path.exists(reference_checkpoint):
                raise FileNotFoundError(
                    f"Supplied reference checkpoint not found at: {reference_checkpoint}\n"
                    "Please provide a valid checkpoint path or run 'fit-reference' first."
                )
            ckpt_state = self.openset_detector.load(reference_checkpoint)
            if ckpt_state and "temporal_aggregator_state" in ckpt_state:
                self.temporal_aggregator.load_state_dict(ckpt_state["temporal_aggregator_state"])
                print("[Inference Pipeline] Restored matched temporal aggregator weights from checkpoint.")

        # 5. Risk Severity Estimator
        self.risk_estimator = RiskSeverityEstimator(
            w_deviation=self.config.risk.w_deviation,
            w_persistence=self.config.risk.w_persistence,
            w_novelty=self.config.risk.w_novelty,
            low_threshold=self.config.risk.low_risk_threshold,
            medium_threshold=self.config.risk.medium_risk_threshold
        )

        # 6. Video Annotator
        self.annotator = SurveillanceVideoAnnotator()

        self.vit_extractor.eval()
        self.temporal_aggregator.eval()
        self.openset_detector.eval()

    def _validate_video_path(self, video_path: str):
        """Validates that a video path exists and has a supported extension or is a directory of frames."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file or directory not found at: {video_path}")
        if os.path.isdir(video_path):
            return
        ext = os.path.splitext(video_path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported video format '{ext}' for file {video_path}. "
                f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))} or directory of image frames"
            )

    def fit_normal_reference_video(
        self,
        video_path: Union[str, List[str]],
        save_checkpoint_path: Optional[str] = None
    ):
        """
        Extracts features from explicitly designated normal surveillance videos
        and fits the reference centroid and dispersion statistics.
        Supports a single video file/directory or a list of normal video paths.
        """
        paths = [video_path] if isinstance(video_path, str) else video_path
        if len(paths) == 0:
            raise ValueError("No video paths provided for normal reference fitting.")

        for p in paths:
            self._validate_video_path(p)

        print(f"[Inference Pipeline] Extracting reference features from {len(paths)} normal video(s)...")
        temporal_embeddings = []

        with torch.inference_mode():
            for p in paths:
                raw_frames, frame_tensors, fps, total_f = self.sampler.extract_all_sampled_frames(p)
                clips, _ = self.sampler.generate_sliding_clips(frame_tensors)
                if clips.shape[0] < 1:
                    continue

                for i in range(clips.shape[0]):
                    clip = clips[i : i + 1].to(self.device)
                    vit_feats = self.vit_extractor(clip)
                    emb, _ = self.temporal_aggregator(vit_feats)
                    temporal_embeddings.append(emb)

        if len(temporal_embeddings) < 2:
            raise ValueError(
                f"Reference videos yielded fewer than 2 total clips ({len(temporal_embeddings)}). "
                "More normal reference frames are required to fit a baseline."
            )

        stacked_embs = torch.cat(temporal_embeddings, dim=0)
        self.openset_detector.fit_normal_reference(stacked_embs)
        self.reference_source_paths = [os.path.abspath(p) for p in paths]
        if len(paths) == 1:
            self.reference_source_path = self.reference_source_paths[0]

        if save_checkpoint_path:
            extra = {"temporal_aggregator_state": self.temporal_aggregator.state_dict()}
            self.openset_detector.save(save_checkpoint_path, extra_state=extra)

    def process_video(
        self,
        video_path: str,
        render_annotated: bool = False,
        output_dir: Optional[str] = None
    ) -> Dict[str, object]:
        """
        Processes a full surveillance video file through the proposed pipeline.
        Fails clearly if the detector has not been fitted with a normal baseline.
        """
        self._validate_video_path(video_path)

        if not self.openset_detector.is_fitted:
            raise RuntimeError(
                "ReferenceBasedAnomalyDetector is NOT fitted. "
                "You must fit normal reference data first using 'pipeline.fit_normal_reference_video(path)' "
                "or load a fitted checkpoint before scoring test videos."
            )

        # Train/Test Leakage Check
        abs_test_path = os.path.abspath(video_path)
        if self.reference_source_path and abs_test_path == self.reference_source_path:
            warnings.warn(
                "TRAIN/TEST LEAKAGE WARNING: The video being evaluated is the EXACT SAME file used "
                "to fit the normal reference baseline. Results will be artificially low.",
                UserWarning,
                stacklevel=2
            )

        start_time = time.perf_counter()
        out_folder = output_dir or self.config.output_dir
        os.makedirs(out_folder, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(video_path))[0]

        # Step 1: Sample video frames
        raw_frames, frame_tensors, video_fps, total_raw_frames = self.sampler.extract_all_sampled_frames(video_path)
        num_sampled = frame_tensors.shape[0]

        # Step 2: Form temporal sliding clips
        clips, clip_ranges = self.sampler.generate_sliding_clips(frame_tensors)
        num_clips = clips.shape[0]

        # Step 3: Run Inference per clip
        clip_results = []
        csv_records = []
        recent_scores_history = []
        frame_predictions = []

        with torch.inference_mode():
            for i in range(num_clips):
                clip = clips[i : i + 1].to(self.device)
                
                # Visual feature extraction
                vit_features = self.vit_extractor(clip)
                
                # Temporal aggregation
                temporal_emb, _ = self.temporal_aggregator(vit_features)
                
                # Reference-based anomaly detection
                detection = self.openset_detector(temporal_emb)
                
                anomaly_score = float(detection["anomaly_score"][0].item())
                raw_dist = float(detection["raw_cosine_distance"][0].item())
                novelty_score = float(detection["novelty_score"][0].item())
                is_anomaly = bool(detection["is_anomaly"][0].item())
                is_unseen = bool(detection["is_unseen"][0].item())
                best_known_cat = detection["best_known_category"]

                # Resolve Category Label
                if not is_anomaly:
                    category = "Normal"
                elif is_unseen:
                    category = "Unseen / Novel Anomaly"
                elif best_known_cat == "Uncharacterized":
                    category = "Visual Anomaly (Out-of-Distribution)"
                else:
                    category = f"Known ({best_known_cat})"

                # Risk severity estimation
                recent_scores_history.append(anomaly_score)
                if len(recent_scores_history) > 10:
                    recent_scores_history.pop(0)

                risk_data = self.risk_estimator.estimate_risk(
                    anomaly_score=anomaly_score,
                    novelty_score=novelty_score,
                    recent_anomaly_history=recent_scores_history
                )

                start_idx, end_idx = clip_ranges[i]
                start_sec = (start_idx * self.sampler.sample_stride) / video_fps
                end_sec = (end_idx * self.sampler.sample_stride) / video_fps

                clip_res = {
                    "clip_index": i,
                    "frame_start": start_idx,
                    "frame_end": end_idx,
                    "time_start_sec": round(start_sec, 2),
                    "time_end_sec": round(end_sec, 2),
                    "anomaly_score": round(anomaly_score, 4),
                    "raw_cosine_distance": round(raw_dist, 4),
                    "novelty_score": round(novelty_score, 4),
                    "is_anomaly": is_anomaly,
                    "category": category,
                    "risk_data": risk_data,
                    "validation_status": "EXPERIMENTAL — UNVALIDATED"
                }
                clip_results.append(clip_res)

                # Append to CSV row
                csv_records.append({
                    "clip_index": i,
                    "time_start_sec": round(start_sec, 2),
                    "time_end_sec": round(end_sec, 2),
                    "frame_start": start_idx,
                    "frame_end": end_idx,
                    "raw_cosine_distance": round(raw_dist, 4),
                    "anomaly_relative_score": round(anomaly_score, 4),
                    "novelty_score": round(novelty_score, 4),
                    "is_anomaly": is_anomaly,
                    "predicted_category": category,
                    "risk_severity": risk_data["severity_level"],
                    "composite_risk_score": risk_data["composite_risk_score"],
                    "recommended_action": risk_data["recommended_action"],
                    "validation_status": "EXPERIMENTAL — UNVALIDATED",
                    "model_backbone": self.config.model.backbone_name,
                    "input_filename": os.path.basename(video_path)
                })

        # Align predictions back to sampled frames
        for f_idx in range(num_sampled):
            clip_idx = min(f_idx, num_clips - 1)
            c_info = clip_results[clip_idx]
            frame_predictions.append({
                "frame_idx": f_idx,
                "timestamp_sec": round((f_idx * self.sampler.sample_stride) / video_fps, 2),
                "is_anomaly": c_info["is_anomaly"],
                "category": c_info["category"],
                "anomaly_score": c_info["anomaly_score"],
                "risk_data": c_info["risk_data"]
            })

        # Synthesize Overall Video Verdict
        anomalous_clips = [c for c in clip_results if c["is_anomaly"]]
        video_is_anomalous = len(anomalous_clips) > 0

        if video_is_anomalous:
            highest_risk_clip = max(anomalous_clips, key=lambda x: x["risk_data"]["composite_risk_score"])
            final_status = "Anomaly"
            final_category = highest_risk_clip["category"]
            final_risk_level = highest_risk_clip["risk_data"]["severity_level"]
            final_confidence = round(highest_risk_clip["anomaly_score"], 4)
            detected_segments = [
                f"{c['time_start_sec']}s - {c['time_end_sec']}s ({c['category']}, Risk: {c['risk_data']['severity_level']})"
                for c in anomalous_clips
            ]
        else:
            final_status = "Normal"
            final_category = "Normal"
            final_risk_level = "Low"
            final_confidence = 0.0
            detected_segments = []

        total_proc_time = time.perf_counter() - start_time
        fps_throughput = round(num_sampled / total_proc_time, 2) if total_proc_time > 0 else 0.0

        # Export CSV report
        csv_path = os.path.join(out_folder, f"{base_name}_results.csv")
        df = pd.DataFrame(csv_records)
        df.to_csv(csv_path, index=False)
        print(f"[Inference Pipeline] Saved timestamped CSV results to: {csv_path}")

        # Render annotated video if requested
        annotated_video_path = None
        if render_annotated:
            annotated_video_path = os.path.join(out_folder, f"{base_name}_annotated.mp4")
            self.annotator.render_annotated_video(
                raw_frames=raw_frames,
                frame_predictions=frame_predictions,
                output_video_path=annotated_video_path,
                fps=max(video_fps / self.sampler.sample_stride, 1.0)
            )
            print(f"[Inference Pipeline] Saved annotated video to: {annotated_video_path}")

        json_path = os.path.join(out_folder, f"{base_name}_summary.json")

        summary_report = {
            "video_path": video_path,
            "input_filename": os.path.basename(video_path),
            "status": final_status,
            "category": final_category,
            "risk_level": final_risk_level,
            "overall_video_verdict": final_status,
            "overall_risk_severity": final_risk_level,
            "highest_risk_category": final_category,
            "peak_anomaly_score": final_confidence,
            "anomalous_clips_count": len(anomalous_clips),
            "total_temporal_clips": num_clips,
            "score_metric": "uncalibrated_cosine_feature_distance",
            "anomaly_score": final_confidence,
            "detected_segments": detected_segments,
            "total_frames_analyzed": num_sampled,
            "processing_time_sec": round(total_proc_time, 2),
            "throughput_fps": fps_throughput,
            "fps_throughput": fps_throughput,
            "csv_results_path": csv_path,
            "csv_path": csv_path,
            "json_path": json_path,
            "annotated_video_path": annotated_video_path,
            "validation_status": "EXPERIMENTAL — UNVALIDATED",
            "model_backbone": self.config.model.backbone_name,
            "device": str(self.device),
            "clip_results": clip_results,
            "summary": {}  # populated below for self-reference
        }
        summary_report["summary"] = summary_report

        # Export JSON report
        with open(json_path, "w") as f:
            json.dump({k: v for k, v in summary_report.items() if k != "summary"}, f, indent=2)
        print(f"[Inference Pipeline] Saved JSON summary to: {json_path}")

        return summary_report
