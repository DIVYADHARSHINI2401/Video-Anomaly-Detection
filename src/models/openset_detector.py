"""
Reference-Based Feature-Space Anomaly and Open-Set Detector.
Replaces random untrained classification heads with a transparent, mathematically defined
cosine-distance and nearest-centroid baseline fitted strictly on normal reference data.
"""
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReferenceBasedAnomalyDetector(nn.Module):
    """
    Reference-Based Anomaly Detector.
    
    Mathematical Formulation:
    - Given L2-normalized feature vectors x in R^D from frozen pretrained ViT.
    - Normal prototype mu_normal = Normalize( sum(x_i) / N ).
    - Query cosine distance: d(z, mu_normal) = 1.0 - <z, mu_normal>.
    - Relative anomaly distance score: scaled relative to normal reference dispersion.
    
    LIMITATION NOTICE:
    This is an uncalibrated geometric feature distance in ViT representation space.
    It does NOT represent a posterior Bayesian probability or ground-truth confidence.
    """
    def __init__(
        self,
        embedding_dim: int = 192,
        anomaly_threshold: float = 0.40,
        unseen_threshold: float = 0.60,
        known_class_names: Optional[List[str]] = None
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.anomaly_threshold = anomaly_threshold
        self.unseen_threshold = unseen_threshold
        self.known_class_names = known_class_names or ["Normal"]

        # Buffers for persistence
        self.register_buffer("normal_centroid", torch.zeros(embedding_dim))
        self.register_buffer("is_fitted", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("ref_mean_dist", torch.tensor(0.0))
        self.register_buffer("ref_std_dist", torch.tensor(1.0))
        self.register_buffer("ref_max_dist", torch.tensor(1.0))

        # Known anomaly prototypes (if provided during supervised/known calibration)
        self.known_prototypes: Dict[str, torch.Tensor] = {}

    def fit_normal_reference(self, normal_embeddings: torch.Tensor):
        """
        Fits the normal baseline centroid strictly on designated normal reference frames.
        Args:
            normal_embeddings: Tensor of shape (N, embedding_dim)
        """
        if not isinstance(normal_embeddings, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(normal_embeddings)}")
        if normal_embeddings.ndim != 2 or normal_embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected normal_embeddings of shape (N, {self.embedding_dim}), got {normal_embeddings.shape}"
            )
        if normal_embeddings.shape[0] < 2:
            raise ValueError("Fitting requires at least 2 normal reference embedding vectors.")
        if not torch.isfinite(normal_embeddings).all():
            raise ValueError("normal_embeddings contains NaN or Inf values.")

        # L2-normalize all reference vectors
        norm_x = F.normalize(normal_embeddings, p=2, dim=-1)

        # Compute empirical mean centroid and project back to unit hypersphere
        mean_vec = norm_x.mean(dim=0)
        unit_centroid = F.normalize(mean_vec, p=2, dim=0)

        # Compute reference dispersion strictly on normal training data
        ref_cosine_dists = 1.0 - torch.mv(norm_x, unit_centroid)  # shape (N,)
        mean_d = ref_cosine_dists.mean()
        std_d = ref_cosine_dists.std()
        max_d = ref_cosine_dists.max()

        self.normal_centroid.copy_(unit_centroid)
        self.ref_mean_dist.copy_(mean_d)
        self.ref_std_dist.copy_(std_d if std_d > 1e-6 else torch.tensor(0.1))
        self.ref_max_dist.copy_(max_d if max_d > 1e-6 else torch.tensor(0.5))
        self.is_fitted.copy_(torch.tensor(True, dtype=torch.bool))

        print(f"[ReferenceBasedAnomalyDetector] Fitted on {normal_embeddings.shape[0]} normal vectors.")
        print(f"  Ref Mean Cosine Dist: {mean_d.item():.4f} +/- {std_d.item():.4f} (Max: {max_d.item():.4f})")

    def register_known_anomaly_prototypes(self, category_name: str, embeddings: torch.Tensor):
        """
        Registers a prototype centroid for an explicitly known anomaly category.
        """
        if not self.is_fitted:
            raise RuntimeError("Fit normal reference baseline before registering anomaly prototypes.")
        norm_embs = F.normalize(embeddings, p=2, dim=-1)
        centroid = F.normalize(norm_embs.mean(dim=0), p=2, dim=0)
        self.known_prototypes[category_name] = centroid
        if category_name not in self.known_class_names:
            self.known_class_names.append(category_name)

    def compute_anomaly_distance(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes cosine distance from normal reference centroid and relative score.
        Args:
            embeddings: (Batch, embedding_dim)
        Returns:
            raw_cosine_dist: In range [0.0, 2.0]
            relative_score: Scaled score in [0.0, 1.0] relative to normal reference bounds
        """
        if not self.is_fitted:
            raise RuntimeError(
                "ReferenceBasedAnomalyDetector has NOT been fitted. "
                "Call 'fit_normal_reference()' or 'load()' a fitted checkpoint before scoring."
            )

        norm_z = F.normalize(embeddings, p=2, dim=-1)
        # Cosine distance: 1.0 - <z, normal_centroid>
        centroid = self.normal_centroid.to(embeddings.device)
        raw_cosine_dist = 1.0 - torch.mv(norm_z, centroid)

        # Scale relative to normal reference dispersion:
        # Distance is mapped monotonically into [0.0, 1.0] without a hard flat ceiling clamp
        # preserving fine-grained ranking for ROC-AUC and PR-AUC calculation
        scale = max(self.ref_max_dist.item() * 2.0, 0.20)
        relative_score = torch.clamp(raw_cosine_dist, min=0.0) / (torch.clamp(raw_cosine_dist, min=0.0) + scale)
        return raw_cosine_dist, relative_score

    def compute_novelty_distance(
        self,
        embeddings: torch.Tensor,
        anomaly_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, str]:
        """
        Computes open-set novelty distance.
        - If known anomaly prototypes exist: distance to closest known prototype.
        - If only normal reference exists: novelty corresponds to extreme deviation beyond normal sphere.
        """
        norm_z = F.normalize(embeddings, p=2, dim=-1)
        batch_size = embeddings.shape[0]

        if len(self.known_prototypes) > 0:
            proto_names = list(self.known_prototypes.keys())
            proto_matrix = torch.stack([self.known_prototypes[name].to(embeddings.device) for name in proto_names], dim=0)
            # Similarities to known anomaly prototypes: (Batch, Num_Prototypes)
            sims = torch.mm(norm_z, proto_matrix.t())
            max_sim, best_idx = sims.max(dim=-1)
            # Lower similarity to any known anomaly means higher novelty/unseen probability
            novelty = torch.clamp((1.0 - max_sim) / 1.2, 0.0, 1.0)
            return novelty, proto_names[best_idx[0].item()]
        else:
            # Class-agnostic novelty: distance beyond normal max reference
            novelty = torch.clamp((anomaly_scores - self.anomaly_threshold) / (1.0 - self.anomaly_threshold + 1e-6), 0.0, 1.0)
            return novelty, "Uncharacterized"

    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Evaluates input embeddings against reference model.
        Args:
            embeddings: (Batch, embedding_dim)
        Returns:
            Dictionary containing:
                - anomaly_score: (Batch,) continuous score in [0.0, 1.0]
                - raw_cosine_distance: (Batch,) continuous cosine distance
                - is_anomaly: (Batch,) boolean flag
                - novelty_score: (Batch,) continuous score in [0.0, 1.0]
                - is_unseen: (Batch,) boolean flag
                - validation_status: "EXPERIMENTAL — UNVALIDATED"
        """
        if not self.is_fitted:
            raise RuntimeError(
                "ReferenceBasedAnomalyDetector has NOT been fitted with normal reference data. "
                "Detector cannot produce inference scores without a valid normal reference baseline."
            )

        if not isinstance(embeddings, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(embeddings)}")
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected embeddings shape (B, {self.embedding_dim}), got {embeddings.shape}")
        if not torch.isfinite(embeddings).all():
            raise FloatingPointError("Input embeddings contain NaN or Inf values.")

        raw_dist, rel_score = self.compute_anomaly_distance(embeddings)
        novelty, best_known_cat = self.compute_novelty_distance(embeddings, rel_score)

        is_anomaly = rel_score >= self.anomaly_threshold
        is_unseen = is_anomaly & (novelty >= self.unseen_threshold)

        return {
            "anomaly_score": rel_score,
            "raw_cosine_distance": raw_dist,
            "is_anomaly": is_anomaly,
            "novelty_score": novelty,
            "is_unseen": is_unseen,
            "best_known_category": best_known_cat,
            "validation_status": "EXPERIMENTAL — UNVALIDATED"
        }

    def save(self, filepath: str, extra_state: Optional[Dict] = None):
        """Saves fitted detector state and prototypes to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted detector.")
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        state = {
            "embedding_dim": self.embedding_dim,
            "normal_centroid": self.normal_centroid.cpu(),
            "ref_mean_dist": self.ref_mean_dist.cpu(),
            "ref_std_dist": self.ref_std_dist.cpu(),
            "ref_max_dist": self.ref_max_dist.cpu(),
            "is_fitted": self.is_fitted.cpu(),
            "anomaly_threshold": self.anomaly_threshold,
            "unseen_threshold": self.unseen_threshold,
            "known_prototypes": {k: v.cpu() for k, v in self.known_prototypes.items()},
            "known_class_names": self.known_class_names
        }
        if extra_state:
            state.update(extra_state)
        torch.save(state, filepath)
        print(f"[ReferenceBasedAnomalyDetector] Saved fitted reference model to: {filepath}")

    def load(self, filepath: str):
        """Loads fitted detector state and prototypes from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Reference detector checkpoint not found at: {filepath}")
        state = torch.load(filepath, map_location="cpu", weights_only=False)
        self.embedding_dim = state["embedding_dim"]
        self.normal_centroid.copy_(state["normal_centroid"])
        self.ref_mean_dist.copy_(state["ref_mean_dist"])
        self.ref_std_dist.copy_(state["ref_std_dist"])
        self.ref_max_dist.copy_(state["ref_max_dist"])
        self.is_fitted.copy_(state["is_fitted"])
        self.anomaly_threshold = state["anomaly_threshold"]
        self.unseen_threshold = state["unseen_threshold"]
        self.known_prototypes = state.get("known_prototypes", {})
        self.known_class_names = state.get("known_class_names", ["Normal"])
        print(f"[ReferenceBasedAnomalyDetector] Loaded fitted reference model from: {filepath}")
        return state
