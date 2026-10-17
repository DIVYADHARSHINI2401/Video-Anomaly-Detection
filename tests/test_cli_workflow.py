"""
Unit and Integration Tests for CLI Workflow (Phase 5).
Validates CLI error handling, checkpoint creation, inference execution,
CSV/JSON schema structure, and timestamp generation.

DISCLAIMER:
All test fixtures in this file use SMALL SYNTHETIC VIDEO CLIPS generated on the fly.
They are strictly used to test CLI subcommands, file I/O, and data structures.
They do NOT represent a real surveillance dataset or accuracy benchmark.
"""
import json
import os
import shutil
import tempfile
import unittest
import cv2
import numpy as np
import pandas as pd

from configs.default_config import PipelineConfig
from src.inference.pipeline import VideoAnomalyInferencePipeline


def _generate_synthetic_test_video(path: str, num_frames: int = 90, fps: int = 20, is_anomaly: bool = False):
    """
    Generates a tiny synthetic video clip for numerical CLI smoke-testing.
    [SYNTHETIC TEST FIXTURE - NOT A REAL SURVEILLANCE DATASET]
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w, h = 160, 120
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))

    for i in range(num_frames):
        # Base background: dim gray
        frame = np.full((h, w, 3), 60, dtype=np.uint8)
        if is_anomaly and i >= 15:
            # Add dynamic high-contrast element
            cv2.rectangle(frame, (20, 20), (80, 80), (255, 0, 0), -1)
        else:
            # Subtle low-contrast circle
            cv2.circle(frame, (80, 60), 10, (100, 100, 100), -1)
        writer.write(frame)

    writer.release()


class TestCLIWorkflow(unittest.TestCase):
    """Test suite for CLI workflow and output serialization."""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp(prefix="vad_test_cli_")
        cls.normal_video = os.path.join(cls.test_dir, "synthetic_normal.mp4")
        cls.test_video = os.path.join(cls.test_dir, "synthetic_test.mp4")
        cls.checkpoint_path = os.path.join(cls.test_dir, "test_reference.pt")
        cls.output_dir = os.path.join(cls.test_dir, "outputs")

        print("\n[SYNTHETIC TEST FIXTURES GENERATION]")
        print("  Notice: Generating synthetic MP4 fixtures for CLI smoke-testing...")
        _generate_synthetic_test_video(cls.normal_video, num_frames=90, fps=20, is_anomaly=False)
        _generate_synthetic_test_video(cls.test_video, num_frames=90, fps=20, is_anomaly=True)
        print("  Generated: synthetic_normal.mp4 and synthetic_test.mp4")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_missing_checkpoint_raises_error(self):
        """Inference pipeline without a fitted checkpoint must raise RuntimeError."""
        cfg = PipelineConfig()
        cfg.device = "cpu"
        pipeline = VideoAnomalyInferencePipeline(config=cfg)

        with self.assertRaises(RuntimeError) as ctx:
            pipeline.process_video(self.test_video)
        self.assertIn("ReferenceBasedAnomalyDetector is NOT fitted", str(ctx.exception))

    def test_invalid_video_extension_raises_error(self):
        """Passing non-video extensions (e.g. .txt, .csv) must raise ValueError."""
        cfg = PipelineConfig()
        cfg.device = "cpu"
        pipeline = VideoAnomalyInferencePipeline(config=cfg)

        invalid_file = os.path.join(self.test_dir, "dummy.txt")
        with open(invalid_file, "w") as f:
            f.write("not a video")

        with self.assertRaises(ValueError) as ctx:
            pipeline._validate_video_path(invalid_file)
        self.assertIn("Unsupported video format", str(ctx.exception))

    def test_nonexistent_video_path_raises_error(self):
        """Passing non-existent video path must raise FileNotFoundError."""
        cfg = PipelineConfig()
        cfg.device = "cpu"
        pipeline = VideoAnomalyInferencePipeline(config=cfg)

        with self.assertRaises(FileNotFoundError):
            pipeline._validate_video_path("non_existent_video_12345.mp4")

    def test_two_step_workflow_and_output_schemas(self):
        """
        Executes the two-step workflow:
        Step 1: fit_normal_reference_video on synthetic_normal.mp4 -> saves checkpoint.
        Step 2: process_video on synthetic_test.mp4 -> produces CSV, JSON, and annotated MP4.
        Verifies:
        - Checkpoint file creation.
        - CSV file creation and column headers.
        - JSON summary file creation and metadata fields.
        - Monotonically increasing timestamps.
        - Presence of 'EXPERIMENTAL — UNVALIDATED' tag.
        """
        cfg = PipelineConfig()
        cfg.device = "cpu"
        cfg.output_dir = self.output_dir

        pipeline = VideoAnomalyInferencePipeline(config=cfg)

        # --- Step 1: Fit Reference ---
        pipeline.fit_normal_reference_video(
            video_path=self.normal_video,
            save_checkpoint_path=self.checkpoint_path
        )
        self.assertTrue(os.path.exists(self.checkpoint_path))

        # --- Step 2: Load Checkpoint in Fresh Pipeline & Infer ---
        infer_pipeline = VideoAnomalyInferencePipeline(
            config=cfg,
            reference_checkpoint=self.checkpoint_path
        )
        self.assertTrue(infer_pipeline.openset_detector.is_fitted.item())

        results = infer_pipeline.process_video(
            video_path=self.test_video,
            render_annotated=True,
            output_dir=self.output_dir
        )

        # 1. Verify JSON output
        json_file = os.path.join(self.output_dir, "synthetic_test_summary.json")
        self.assertTrue(os.path.exists(json_file))
        with open(json_file, "r") as f:
            summary = json.load(f)

        self.assertEqual(summary["validation_status"], "EXPERIMENTAL — UNVALIDATED")
        self.assertIn("status", summary)
        self.assertIn("category", summary)
        self.assertIn("risk_level", summary)
        self.assertIn("throughput_fps", summary)
        self.assertGreater(summary["total_frames_analyzed"], 0)

        # 2. Verify CSV output
        csv_file = os.path.join(self.output_dir, "synthetic_test_results.csv")
        self.assertTrue(os.path.exists(csv_file))
        df = pd.read_csv(csv_file)

        expected_columns = {
            "clip_index", "time_start_sec", "time_end_sec", "frame_start", "frame_end",
            "raw_cosine_distance", "anomaly_relative_score", "novelty_score",
            "is_anomaly", "predicted_category", "risk_severity", "composite_risk_score",
            "recommended_action", "validation_status", "model_backbone", "input_filename"
        }
        self.assertTrue(expected_columns.issubset(set(df.columns)))
        self.assertTrue((df["validation_status"] == "EXPERIMENTAL — UNVALIDATED").all())

        # Check timestamp monotonicity
        timestamps = df["time_start_sec"].tolist()
        for i in range(1, len(timestamps)):
            self.assertGreaterEqual(timestamps[i], timestamps[i - 1])

        # 3. Verify Annotated Video output
        annotated_mp4 = os.path.join(self.output_dir, "synthetic_test_annotated.mp4")
        self.assertTrue(os.path.exists(annotated_mp4))
        self.assertGreater(os.path.getsize(annotated_mp4), 1000)

        print("\n[SYNTHETIC TEST WORKFLOW VERIFIED SUCCESSFULLY]")
        print(f"  Checkpoint: {self.checkpoint_path}")
        print(f"  CSV Output: {csv_file} (Rows: {len(df)})")
        print(f"  JSON Output: {json_file}")
        print(f"  Annotated Video: {annotated_mp4} ({os.path.getsize(annotated_mp4)} bytes)")


if __name__ == "__main__":
    unittest.main()
