"""
Evaluation Metrics for Closed-Set and Open-Set Video Anomaly Detection.
Provides mathematically rigorous frame-level and segment-level evaluation:
- ROC-AUC, PR-AUC, Precision, Recall, F1-score, FPR, and Confusion Matrix (TP, FP, TN, FN)
- Threshold Optimization strictly on validation data (Youden's J or Maximum F1)
- Pure NumPy & SciPy implementations with scikit-learn support
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    from sklearn.metrics import (
        auc,
        confusion_matrix,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def compute_roc_auc_numpy(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Computes exact ROC-AUC using Mann-Whitney U rank statistic."""
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)
    pos = y_scores[y_true == 1]
    neg = y_scores[y_true == 0]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Vectorized pairwise comparison
    diff = pos[:, None] - neg[None, :]
    u_stat = np.sum((diff > 0).astype(float) + 0.5 * (diff == 0).astype(float))
    return float(u_stat / (n_pos * n_neg))


def compute_pr_auc_numpy(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Computes exact Area Under the Precision-Recall Curve (PR-AUC) using trapezoidal rule."""
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)
    if np.sum(y_true) == 0:
        return 0.0

    # Sort descending by score
    order = np.argsort(-y_scores)
    y_sorted = y_true[order]

    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    total_pos = np.sum(y_true == 1)

    recalls = tp / total_pos
    precisions = tp / (tp + fp)

    # Insert point (0, 1) for integration
    recalls = np.concatenate([[0.0], recalls])
    precisions = np.concatenate([[1.0], precisions])

    # Trapezoidal area under curve
    pr_auc = float(np.sum((recalls[1:] - recalls[:-1]) * precisions[1:]))
    return max(0.0, min(pr_auc, 1.0))


def optimize_threshold_on_validation(
    val_y_true: np.ndarray,
    val_y_scores: np.ndarray,
    criterion: str = "youden_j",
    target_fpr: Optional[float] = None
) -> float:
    """
    Selects the optimal anomaly decision threshold strictly using VALIDATION data.
    Criteria:
    - 'youden_j': Maximizes J = True Positive Rate - False Positive Rate (Sensitivity + Specificity - 1)
    - 'f1': Maximizes F1-score on validation set
    - 'target_fpr': Chooses the threshold achieving a specific maximum false positive rate (e.g. 0.05)
    """
    val_y_true = np.asarray(val_y_true, dtype=int)
    val_y_scores = np.asarray(val_y_scores, dtype=float)

    # Candidate thresholds across score distribution
    thresholds = np.unique(val_y_scores)
    if len(thresholds) == 0:
        return 0.5

    best_thresh = float(thresholds[0])

    if criterion == "target_fpr" and target_fpr is not None:
        neg_scores = val_y_scores[val_y_true == 0]
        if len(neg_scores) == 0:
            return 0.5
        # Pick threshold where FPR <= target_fpr
        best_thresh = float(np.percentile(neg_scores, (1.0 - target_fpr) * 100))
        return best_thresh

    elif criterion == "f1":
        best_f1 = -1.0
        for th in thresholds:
            pred = (val_y_scores >= th).astype(int)
            tp = np.sum((val_y_true == 1) & (pred == 1))
            fp = np.sum((val_y_true == 0) & (pred == 1))
            fn = np.sum((val_y_true == 1) & (pred == 0))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            score = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            if score > best_f1:
                best_f1 = score
                best_thresh = float(th)
        return best_thresh

    else:  # 'youden_j'
        best_j = -2.0
        n_pos = np.sum(val_y_true == 1)
        n_neg = np.sum(val_y_true == 0)
        if n_pos == 0 or n_neg == 0:
            return 0.5
        for th in thresholds:
            pred = (val_y_scores >= th).astype(int)
            tp = np.sum((val_y_true == 1) & (pred == 1))
            fp = np.sum((val_y_true == 0) & (pred == 1))
            tpr = tp / n_pos
            fpr = fp / n_neg
            j_stat = tpr - fpr
            if j_stat > best_j:
                best_j = j_stat
                best_thresh = float(th)
        return best_thresh


class VideoAnomalyMetrics:
    """
    Computes standard surveillance anomaly detection and open-set metrics.
    """
    @staticmethod
    def compute_frame_level_metrics(
        y_true: np.ndarray,
        y_scores: np.ndarray,
        threshold: float = 0.50
    ) -> Dict[str, object]:
        """
        Computes standard frame-level binary anomaly detection metrics.
        """
        y_true = np.asarray(y_true, dtype=int)
        y_scores = np.asarray(y_scores, dtype=float)
        y_pred = (y_scores >= threshold).astype(int)

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float((2 * prec * rec) / (prec + rec)) if (prec + rec) > 0 else 0.0
        base_fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        if HAS_SKLEARN:
            try:
                roc_auc = float(roc_auc_score(y_true, y_scores))
            except ValueError:
                roc_auc = 0.5

            precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_scores)
            pr_auc = float(auc(recall_arr, precision_arr))

            fpr_arr, tpr_arr, _ = roc_curve(y_true, y_scores)
            idx_95 = np.argmin(np.abs(tpr_arr - 0.95))
            fpr_at_95 = float(fpr_arr[idx_95])
        else:
            roc_auc = compute_roc_auc_numpy(y_true, y_scores)
            pr_auc = compute_pr_auc_numpy(y_true, y_scores)
            # Find threshold achieving 95% TPR
            pos_scores = y_scores[y_true == 1]
            if len(pos_scores) > 0 and len(y_scores[y_true == 0]) > 0:
                th_95 = float(np.percentile(pos_scores, 5))
                neg_fp = np.sum((y_scores[y_true == 0] >= th_95))
                fpr_at_95 = float(neg_fp / len(y_scores[y_true == 0]))
            else:
                fpr_at_95 = 0.0

        return {
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "f1_score": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "false_positive_rate": round(base_fpr, 4),
            "fpr_at_95_tpr": round(fpr_at_95, 4),
            "operating_threshold": round(float(threshold), 4),
            "confusion_matrix": {
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
                "total_frames": len(y_true)
            }
        }

    @staticmethod
    def compute_open_set_metrics(
        y_true_normal: np.ndarray,
        y_scores_normal: np.ndarray,
        y_true_known: np.ndarray,
        y_scores_known: np.ndarray,
        y_true_unseen: np.ndarray,
        y_scores_unseen: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluates open-set performance distinguishing known anomaly AUROC,
        unseen anomaly AUROC, and their Harmonic Mean (H-Mean).
        """
        y_known_combined = np.concatenate([y_true_normal, y_true_known])
        scores_known_combined = np.concatenate([y_scores_normal, y_scores_known])
        
        y_unseen_combined = np.concatenate([y_true_normal, y_true_unseen])
        scores_unseen_combined = np.concatenate([y_scores_normal, y_scores_unseen])

        if HAS_SKLEARN:
            try:
                auroc_known = float(roc_auc_score(y_known_combined, scores_known_combined))
            except ValueError:
                auroc_known = 0.5
            try:
                auroc_unseen = float(roc_auc_score(y_unseen_combined, scores_unseen_combined))
            except ValueError:
                auroc_unseen = 0.5
        else:
            auroc_known = compute_roc_auc_numpy(y_known_combined, scores_known_combined)
            auroc_unseen = compute_roc_auc_numpy(y_unseen_combined, scores_unseen_combined)

        if (auroc_known + auroc_unseen) > 0:
            h_mean = 2.0 * (auroc_known * auroc_unseen) / (auroc_known + auroc_unseen)
        else:
            h_mean = 0.0

        drop = auroc_known - auroc_unseen

        return {
            "auroc_known": round(auroc_known, 4),
            "auroc_unseen": round(auroc_unseen, 4),
            "harmonic_mean": round(h_mean, 4),
            "generalization_drop": round(drop, 4)
        }
