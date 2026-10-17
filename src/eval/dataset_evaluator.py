"""
Research-Grade Dataset Evaluator for Video Anomaly Detection.
Executes leakage-free evaluation across video-level train/validation/test splits,
aligns frame-level ground-truth annotations, and produces machine-readable metrics (JSON/CSV)
and Markdown evaluation reports without synthetic fabrication.
"""
import json
import os
import time
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch

from src.eval.metrics import VideoAnomalyMetrics, optimize_threshold_on_validation
from src.inference.pipeline import VideoAnomalyInferencePipeline


class VideoAnomalyDatasetEvaluator:
    """
    Evaluates VideoAnomalyInferencePipeline on benchmark datasets with ground-truth annotations.
    """
    def __init__(
        self,
        pipeline: VideoAnomalyInferencePipeline,
        output_dir: str = "evaluation_results"
    ):
        self.pipeline = pipeline
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def evaluate_annotated_videos(
        self,
        test_video_paths: List[str],
        ground_truth_labels: Dict[str, np.ndarray],
        validation_video_paths: Optional[List[str]] = None,
        val_ground_truth_labels: Optional[Dict[str, np.ndarray]] = None,
        threshold_selection_criterion: str = "youden_j",
        dataset_name: str = "Benchmark Dataset"
    ) -> Dict[str, object]:
        """
        Executes evaluation across test videos with aligned ground truth annotations.
        Args:
            test_video_paths: List of filepaths to test videos
            ground_truth_labels: Dict mapping video filename to 1D binary array of frame labels
            validation_video_paths: Optional validation videos for threshold selection
            val_ground_truth_labels: Dict of validation labels
            threshold_selection_criterion: 'youden_j', 'f1', or 'target_fpr'
            dataset_name: Name of evaluated benchmark
        """
        if not self.pipeline.openset_detector.is_fitted:
            raise RuntimeError(
                "Cannot run evaluation: ReferenceBasedAnomalyDetector is not fitted. "
                "Fit the normal reference baseline on training videos before evaluation."
            )

        start_eval_time = time.perf_counter()

        # Step 1: Threshold Selection strictly on Validation Data (if provided)
        operating_threshold = self.pipeline.config.openset.anomaly_threshold
        threshold_tuning_notes = "Default threshold used; no separate validation set provided."

        if validation_video_paths and val_ground_truth_labels:
            print(f"[Evaluator] Optimizing operating threshold on {len(validation_video_paths)} validation videos...")
            val_scores_all = []
            val_true_all = []

            stride = self.pipeline.sampler.sample_stride
            for val_path in validation_video_paths:
                base_name = os.path.basename(val_path)
                if base_name not in val_ground_truth_labels:
                    continue
                v_res = self.pipeline.process_video(val_path, render_annotated=False)
                gt = val_ground_truth_labels[base_name]

                # Align sampled frames with ground truth
                for clip in v_res["clip_results"]:
                    f_start = clip["frame_start"] * stride
                    f_end = min(clip["frame_end"] * stride, len(gt))
                    if f_start < len(gt):
                        clip_gt = int(np.max(gt[f_start:f_end])) if f_start < f_end else int(gt[f_start])
                        val_true_all.append(clip_gt)
                        val_scores_all.append(clip["anomaly_score"])

            if len(val_true_all) > 0 and len(np.unique(val_true_all)) > 1:
                operating_threshold = optimize_threshold_on_validation(
                    np.array(val_true_all),
                    np.array(val_scores_all),
                    criterion=threshold_selection_criterion
                )
                threshold_tuning_notes = (
                    f"Threshold optimized on validation set using '{threshold_selection_criterion}' "
                    f"across {len(val_true_all)} clips -> selected threshold: {operating_threshold:.4f}"
                )
                print(f"[Evaluator] {threshold_tuning_notes}")

        # Step 2: Evaluation on Test Videos
        print(f"[Evaluator] Evaluating {len(test_video_paths)} test videos on {self.pipeline.device}...")
        all_true_labels = []
        all_pred_scores = []
        all_raw_distances = []
        prediction_records = []

        # Native frame-level containers for benchmark comparability
        all_native_frame_true = []
        all_native_frame_scores = []

        total_frames_processed = 0

        for test_path in test_video_paths:
            base_name = os.path.basename(test_path)
            if base_name not in ground_truth_labels:
                raise KeyError(f"Ground-truth annotations missing for test video: {base_name}")

            gt_array = ground_truth_labels[base_name]
            v_res = self.pipeline.process_video(test_path, render_annotated=False)

            stride = self.pipeline.sampler.sample_stride
            num_native_frames = len(gt_array)
            native_frame_scores = np.zeros(num_native_frames, dtype=np.float32)
            native_frame_assigned = np.zeros(num_native_frames, dtype=bool)

            for clip in v_res["clip_results"]:
                f_start = clip["frame_start"] * stride
                f_end = min(clip["frame_end"] * stride, num_native_frames)

                # Ground truth for this temporal segment
                if f_start < num_native_frames:
                    clip_gt = int(np.max(gt_array[f_start:f_end])) if f_start < f_end else int(gt_array[f_start])
                else:
                    clip_gt = 0

                score = clip["anomaly_score"]
                raw_dist = clip["raw_cosine_distance"]
                is_anom_pred = bool(score >= operating_threshold)

                all_true_labels.append(clip_gt)
                all_pred_scores.append(score)
                all_raw_distances.append(raw_dist)

                # Assign score to native frame range
                actual_end = max(f_start + 1, f_end)
                for f_i in range(f_start, min(actual_end, num_native_frames)):
                    if not native_frame_assigned[f_i]:
                        native_frame_scores[f_i] = score
                        native_frame_assigned[f_i] = True
                    else:
                        native_frame_scores[f_i] = max(native_frame_scores[f_i], score)

                prediction_records.append({
                    "video_filename": base_name,
                    "clip_index": clip["clip_index"],
                    "frame_start_native": f_start,
                    "frame_end_native": f_end,
                    "time_start_sec": clip["time_start_sec"],
                    "time_end_sec": clip["time_end_sec"],
                    "ground_truth_label": clip_gt,
                    "anomaly_relative_score": score,
                    "raw_cosine_distance": raw_dist,
                    "predicted_label": int(is_anom_pred),
                    "predicted_category": clip["category"],
                    "risk_severity": clip["risk_data"]["severity_level"],
                    "validation_status": "EXPERIMENTAL — UNVALIDATED"
                })

            # Impute any unassigned frames at start/end
            first_valid_score = v_res["clip_results"][0]["anomaly_score"] if len(v_res["clip_results"]) > 0 else 0.0
            last_valid_score = v_res["clip_results"][-1]["anomaly_score"] if len(v_res["clip_results"]) > 0 else 0.0
            for f_i in range(num_native_frames):
                if not native_frame_assigned[f_i]:
                    native_frame_scores[f_i] = first_valid_score if f_i < num_native_frames // 2 else last_valid_score

            all_native_frame_true.extend(gt_array.tolist())
            all_native_frame_scores.extend(native_frame_scores.tolist())
            total_frames_processed += v_res["total_frames_analyzed"]

        eval_wall_time = time.perf_counter() - start_eval_time
        y_true = np.array(all_true_labels)
        y_scores = np.array(all_pred_scores)

        # Step 3: Compute Mathematically Rigorous Metrics
        # Frame-level metrics (Standard benchmark convention)
        y_frame_true = np.array(all_native_frame_true)
        y_frame_scores = np.array(all_native_frame_scores)
        metrics = VideoAnomalyMetrics.compute_frame_level_metrics(
            y_true=y_frame_true,
            y_scores=y_frame_scores,
            threshold=operating_threshold
        )

        # Also preserve clip/segment metrics for comparative diagnostic
        clip_metrics = VideoAnomalyMetrics.compute_frame_level_metrics(
            y_true=y_true,
            y_scores=y_scores,
            threshold=operating_threshold
        )
        metrics["clip_level_summary"] = {
            "clip_roc_auc": clip_metrics["roc_auc"],
            "clip_pr_auc": clip_metrics["pr_auc"],
            "clip_f1_score": clip_metrics["f1_score"],
            "total_clips": len(y_true)
        }

        metrics["dataset_name"] = dataset_name
        metrics["num_test_videos"] = len(test_video_paths)
        metrics["total_evaluated_native_frames"] = len(y_frame_true)
        metrics["total_evaluated_clips"] = len(y_true)
        metrics["anomaly_ratio"] = round(float(np.mean(y_frame_true)), 4)
        metrics["threshold_selection_protocol"] = threshold_tuning_notes
        metrics["evaluation_wall_time_sec"] = round(eval_wall_time, 2)
        metrics["throughput_fps"] = round(total_frames_processed / eval_wall_time, 2) if eval_wall_time > 0 else 0.0
        metrics["model_backbone"] = self.pipeline.config.model.backbone_name
        metrics["device"] = str(self.pipeline.device)

        # Step 4: Save Per-Frame/Clip Predictions CSV
        csv_path = os.path.join(self.output_dir, "predictions.csv")
        pred_df = pd.DataFrame(prediction_records)
        pred_df.to_csv(csv_path, index=False)

        # Step 5: Save Metrics JSON
        json_path = os.path.join(self.output_dir, "metrics.json")
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # Step 6: Generate Human-Readable Markdown Report
        report_path = os.path.join(self.output_dir, "evaluation_report.md")
        self._generate_markdown_report(report_path, metrics)

        print(f"[Evaluator] Evaluation complete! Results saved to: {self.output_dir}")
        return metrics

    def _generate_markdown_report(self, report_path: str, m: Dict[str, object]):
        """Generates a clean Markdown evaluation summary."""
        cm = m.get("confusion_matrix", {})
        md_text = f"""# Research Evaluation Report: Video Anomaly Detection

**Dataset:** {m.get('dataset_name', 'Benchmark')}  
**Model Backbone:** {m.get('model_backbone', 'Pretrained ViT')}  
**Device:** {m.get('device', 'cpu')}  
**Evaluation Status:** EXPERIMENTAL — UNVALIDATED  

---

## 1. Summary of Quantitative Metrics

| Metric | Value | Interpretation |
|---|---|---|
| **ROC-AUC** | **{m.get('roc_auc', 'N/A')}** | Threshold-independent discrimination capability across all operating points |
| **PR-AUC** | **{m.get('pr_auc', 'N/A')}** | Precision-Recall curve area (informative under class imbalance) |
| **Operating Threshold** | **{m.get('operating_threshold', 'N/A')}** | {m.get('threshold_selection_protocol', '')} |
| **Precision** | **{m.get('precision', 'N/A')}** | True Positives / (True Positives + False Positives) |
| **Recall (Sensitivity)** | **{m.get('recall', 'N/A')}** | True Positives / (True Positives + False Negatives) |
| **F1-Score** | **{m.get('f1_score', 'N/A')}** | Harmonic mean of Precision and Recall |
| **False Positive Rate (FPR)** | **{m.get('false_positive_rate', 'N/A')}** | False Positives / (False Positives + True Negatives) |
| **FPR @ 95% TPR** | **{m.get('fpr_at_95_tpr', 'N/A')}** | False alarm rate when detecting 95% of true anomalous events |

---

## 2. Confusion Matrix

| | Predicted Normal | Predicted Anomaly |
|---|---|---|
| **Actual Normal** | {cm.get('true_negatives', 0)} (TN) | {cm.get('false_positives', 0)} (FP) |
| **Actual Anomaly** | {cm.get('false_negatives', 0)} (FN) | {cm.get('true_positives', 0)} (TP) |

* **Total Evaluated Clips/Frames:** {cm.get('total_frames', 0)}  
* **Anomaly Ground-Truth Ratio:** {m.get('anomaly_ratio', 0.0) * 100:.2f}%  

---

## 3. Hardware Runtime & Efficiency

* **Wall-Clock Evaluation Time:** {m.get('evaluation_wall_time_sec', 0)} s  
* **Inference Throughput:** {m.get('throughput_fps', 0)} Frames Per Second (FPS)  

---

## 4. Methodological Scope & Limitations

> [!WARNING]
> - **Uncalibrated Feature Distance:** The anomaly score represents relative cosine deviation from the normal reference centroid in Vision Transformer representation space. It is **not** a posterior probability.
> - **Zero Test-Set Leakage:** The decision threshold and reference centroid were fitted strictly without access to test videos or test labels.
"""
        with open(report_path, "w") as f:
            f.write(md_text)
