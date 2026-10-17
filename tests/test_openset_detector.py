"""
Unit Tests for ReferenceBasedAnomalyDetector.
Verifies normal-reference fitting, distance scoring, save/load consistency,
input validation, missing reference error handling, and output determinism.

DISCLAIMER:
All test fixtures in this file use SYNTHETIC NUMERICAL VECTORS.
They are strictly used to verify mathematical properties, exceptions, and consistency.
They do NOT represent benchmark evaluations or real surveillance accuracy claims.
"""
import os
import tempfile
import unittest
import torch
import torch.nn.functional as F

from src.models.openset_detector import ReferenceBasedAnomalyDetector


class TestReferenceBasedAnomalyDetector(unittest.TestCase):
    """
    Test suite for ReferenceBasedAnomalyDetector.
    """
    def setUp(self):
        torch.manual_seed(42)
        self.embedding_dim = 192
        self.detector = ReferenceBasedAnomalyDetector(
            embedding_dim=self.embedding_dim,
            anomaly_threshold=0.40,
            unseen_threshold=0.60
        )
        print("\n[SYNTHETIC UNIT TEST FIXTURE INITIALIZED]")

    def test_unfitted_detector_raises_error(self):
        """Unfitted detector must raise RuntimeError on inference."""
        dummy_query = torch.randn(2, self.embedding_dim)
        with self.assertRaises(RuntimeError) as ctx:
            self.detector(dummy_query)
        self.assertIn("NOT been fitted", str(ctx.exception))

    def test_normal_reference_fitting(self):
        """Fitting on normal reference vectors properly updates state."""
        # Synthetic normal reference cluster centered around a fixed unit vector
        true_centroid = F.normalize(torch.randn(self.embedding_dim), p=2, dim=0)
        # Generate 50 synthetic normal vectors with small random perturbations
        noise = torch.randn(50, self.embedding_dim) * 0.05
        normal_vectors = true_centroid.unsqueeze(0) + noise

        self.assertFalse(self.detector.is_fitted.item())
        self.detector.fit_normal_reference(normal_vectors)
        self.assertTrue(self.detector.is_fitted.item())

        # Centroid must be unit length
        centroid_norm = torch.norm(self.detector.normal_centroid, p=2).item()
        self.assertAlmostEqual(centroid_norm, 1.0, places=5)
        # Cosine similarity to the generating vector should be very high (>0.99)
        dot_product = torch.dot(self.detector.normal_centroid, true_centroid).item()
        self.assertGreater(dot_product, 0.99)

    def test_distance_scoring_ordering(self):
        """In-distribution vectors must yield lower distance than out-of-distribution vectors."""
        normal_base = F.normalize(torch.randn(self.embedding_dim), p=2, dim=0)
        normal_data = normal_base.unsqueeze(0) + torch.randn(30, self.embedding_dim) * 0.05
        self.detector.fit_normal_reference(normal_data)

        # In-distribution test vector
        in_dist = F.normalize(normal_base.unsqueeze(0) + torch.randn(1, self.embedding_dim) * 0.02, p=2, dim=-1)
        # Out-of-distribution orthogonal / distant test vector
        out_dist = F.normalize(torch.randn(1, self.embedding_dim), p=2, dim=-1)

        res_in = self.detector(in_dist)
        res_out = self.detector(out_dist)

        dist_in = res_in["raw_cosine_distance"].item()
        dist_out = res_out["raw_cosine_distance"].item()

        print(f"  Synthetic In-Dist Cosine Dist:  {dist_in:.4f}")
        print(f"  Synthetic Out-Dist Cosine Dist: {dist_out:.4f}")
        self.assertLess(dist_in, dist_out)
        self.assertLess(res_in["anomaly_score"].item(), res_out["anomaly_score"].item())

    def test_save_and_load_consistency(self):
        """Saved and loaded detector must produce identical outputs on identical inputs."""
        normal_data = torch.randn(40, self.embedding_dim)
        self.detector.fit_normal_reference(normal_data)

        test_query = torch.randn(5, self.embedding_dim)
        original_output = self.detector(test_query)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "detector_ref.pt")
            self.detector.save(ckpt_path)
            self.assertTrue(os.path.exists(ckpt_path))

            # Create new fresh detector instance and load
            loaded_detector = ReferenceBasedAnomalyDetector(embedding_dim=self.embedding_dim)
            loaded_detector.load(ckpt_path)
            self.assertTrue(loaded_detector.is_fitted.item())

            loaded_output = loaded_detector(test_query)

            # Assert identical outputs
            torch.testing.assert_close(
                original_output["raw_cosine_distance"],
                loaded_output["raw_cosine_distance"],
                rtol=1e-5,
                atol=1e-6
            )
            torch.testing.assert_close(
                original_output["anomaly_score"],
                loaded_output["anomaly_score"],
                rtol=1e-5,
                atol=1e-6
            )

    def test_invalid_inputs_handling(self):
        """Invalid shapes, NaNs, and insufficient fitting data must raise errors."""
        # 1. Fitting with fewer than 2 vectors
        with self.assertRaises(ValueError):
            self.detector.fit_normal_reference(torch.randn(1, self.embedding_dim))

        # 2. Fitting with NaNs
        nan_vecs = torch.randn(10, self.embedding_dim)
        nan_vecs[0, 0] = float('nan')
        with self.assertRaises(ValueError):
            self.detector.fit_normal_reference(nan_vecs)

        # 3. Mismatched feature dimensions
        with self.assertRaises(ValueError):
            self.detector.fit_normal_reference(torch.randn(10, 64))

        # 4. Inference input with NaNs after fitting
        self.detector.fit_normal_reference(torch.randn(10, self.embedding_dim))
        query_nan = torch.randn(2, self.embedding_dim)
        query_nan[0, 0] = float('nan')
        with self.assertRaises(FloatingPointError):
            self.detector(query_nan)

    def test_deterministic_outputs(self):
        """Repeated evaluations of the exact same input must yield identical outputs."""
        self.detector.fit_normal_reference(torch.randn(20, self.embedding_dim))
        query = torch.randn(4, self.embedding_dim)

        out1 = self.detector(query)
        out2 = self.detector(query)

        self.assertTrue(torch.equal(out1["raw_cosine_distance"], out2["raw_cosine_distance"]))
        self.assertTrue(torch.equal(out1["anomaly_score"], out2["anomaly_score"]))


if __name__ == "__main__":
    unittest.main()
