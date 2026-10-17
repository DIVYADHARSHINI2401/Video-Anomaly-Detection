"""
Unit Tests for Evaluation Metrics, Confusion Matrix, and Threshold Optimization.
Uses small hand-verifiable synthetic numerical examples to confirm 100% mathematical precision.

DISCLAIMER:
All test cases in this file use HAND-CRAFTED NUMERICAL EXAMPLES.
They are strictly designed to test mathematical formulas and edge cases.
They do NOT represent benchmark evaluations or real surveillance accuracy claims.
"""
import unittest
import numpy as np

from src.eval.metrics import (
    VideoAnomalyMetrics,
    compute_roc_auc_numpy,
    compute_pr_auc_numpy,
    optimize_threshold_on_validation
)


class TestEvaluationMetrics(unittest.TestCase):
    """Hand-verifiable unit tests for anomaly detection metrics."""

    def test_hand_crafted_confusion_matrix_and_roc_auc(self):
        """
        Hand-crafted test case:
        y_true   = [0,   0,   1,    1  ]
        y_scores = [0.1, 0.4, 0.35, 0.8]
        Threshold = 0.30:
          Predictions: [0, 1, 1, 1]
          TP: 2 (indices 2, 3)
          FP: 1 (index 1)
          FN: 0
          TN: 1 (index 0)
          Precision = 2 / 3 = 0.6667
          Recall    = 2 / 2 = 1.0000
          F1-score  = 2 * (2/3 * 1) / (2/3 + 1) = 4 / 5 = 0.8000
          FPR       = 1 / (1 + 1) = 0.5000
        Mann-Whitney ROC-AUC:
          Positives = {0.35, 0.8}, Negatives = {0.1, 0.4}
          Pairs (pos, neg):
            0.35 vs 0.10 -> win (1.0)
            0.35 vs 0.40 -> loss (0.0)
            0.80 vs 0.10 -> win (1.0)
            0.80 vs 0.40 -> win (1.0)
          Concordant pairs = 3 / 4 = 0.7500
        """
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.4, 0.35, 0.8])
        threshold = 0.30

        metrics = VideoAnomalyMetrics.compute_frame_level_metrics(y_true, y_scores, threshold=threshold)

        # Check exact confusion matrix
        cm = metrics["confusion_matrix"]
        self.assertEqual(cm["true_positives"], 2)
        self.assertEqual(cm["false_positives"], 1)
        self.assertEqual(cm["false_negatives"], 0)
        self.assertEqual(cm["true_negatives"], 1)
        self.assertEqual(cm["total_frames"], 4)

        # Check exact derived metrics
        self.assertAlmostEqual(metrics["precision"], 0.6667, places=4)
        self.assertAlmostEqual(metrics["recall"], 1.0000, places=4)
        self.assertAlmostEqual(metrics["f1_score"], 0.8000, places=4)
        self.assertAlmostEqual(metrics["false_positive_rate"], 0.5000, places=4)
        self.assertAlmostEqual(metrics["roc_auc"], 0.7500, places=4)

    def test_pure_numpy_roc_auc_matches_hand_calculation(self):
        """Validates pure-numpy Mann-Whitney calculation against hand derivation."""
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.4, 0.35, 0.8])
        calc_auc = compute_roc_auc_numpy(y_true, y_scores)
        self.assertAlmostEqual(calc_auc, 0.75, places=4)

    def test_perfect_separation_scores(self):
        """When positive scores are strictly greater than negative scores, ROC-AUC must be 1.0."""
        y_true = np.array([0, 0, 0, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.25, 0.7, 0.9])
        metrics = VideoAnomalyMetrics.compute_frame_level_metrics(y_true, y_scores, threshold=0.5)

        self.assertAlmostEqual(metrics["roc_auc"], 1.0, places=4)
        self.assertEqual(metrics["confusion_matrix"]["true_positives"], 2)
        self.assertEqual(metrics["confusion_matrix"]["false_positives"], 0)
        self.assertEqual(metrics["confusion_matrix"]["true_negatives"], 3)
        self.assertEqual(metrics["confusion_matrix"]["false_negatives"], 0)
        self.assertAlmostEqual(metrics["f1_score"], 1.0, places=4)

    def test_validation_threshold_optimization(self):
        """
        Validates that optimize_threshold_on_validation selects the threshold
        that maximizes Youden's J statistic strictly on the validation set.
        """
        val_y_true = np.array([0, 0, 0, 1, 1, 1])
        val_y_scores = np.array([0.1, 0.2, 0.5, 0.4, 0.8, 0.9])

        # At threshold 0.4:
        # pred = [0, 0, 1, 1, 1, 1]
        # TP = 3, FP = 1 -> TPR = 1.0, FPR = 1/3 = 0.3333 -> J = 0.6667
        # At threshold 0.6:
        # pred = [0, 0, 0, 0, 1, 1]
        # TP = 2, FP = 0 -> TPR = 2/3 = 0.6667, FPR = 0.0 -> J = 0.6667
        best_th = optimize_threshold_on_validation(val_y_true, val_y_scores, criterion="youden_j")
        self.assertIn(best_th, [0.4, 0.8])

        # Test target FPR criterion: e.g. target FPR = 0.0
        best_th_zero_fpr = optimize_threshold_on_validation(val_y_true, val_y_scores, criterion="target_fpr", target_fpr=0.0)
        # To guarantee 0 false positives, threshold must be >= max negative score (0.5)
        self.assertGreaterEqual(best_th_zero_fpr, 0.5)

    def test_image_sequence_folder_sampling(self):
        """
        Validates that VideoFrameSampler correctly samples from an image-sequence directory
        containing sequential TIFF frames (common in UCSD Ped2).
        """
        import os
        import tempfile
        import cv2
        from src.data.frame_sampler import VideoFrameSampler

        sampler = VideoFrameSampler(frame_size=(64, 64), sample_stride=2, clip_length=4)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create 10 dummy grayscale .tif files (001.tif to 010.tif)
            for i in range(1, 11):
                img_path = os.path.join(tmp_dir, f"{i:03d}.tif")
                # 1-channel grayscale image
                gray = np.full((120, 160), fill_value=i * 20, dtype=np.uint8)
                cv2.imwrite(img_path, gray)

            raw_frames, frame_tensors, fps, total_f = sampler.extract_all_sampled_frames(tmp_dir)

            self.assertEqual(total_f, 10)
            self.assertEqual(fps, 10.0)
            # Stride = 2, so frames 0, 2, 4, 6, 8 -> 5 sampled frames
            self.assertEqual(len(raw_frames), 5)
            self.assertEqual(frame_tensors.shape[0], 5)
            # Channel dimension must be 3 (RGB converted)
            self.assertEqual(frame_tensors.shape[1], 3)
            self.assertEqual(frame_tensors.shape[2], 64)
            self.assertEqual(frame_tensors.shape[3], 64)


if __name__ == "__main__":
    unittest.main()
