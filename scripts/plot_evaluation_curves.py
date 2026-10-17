"""
ROC and Precision-Recall Curve Visualizer for UCSD Ped2 Evaluation.
Generates publication-quality curves from predictions.csv and metrics.json.
"""
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation_results", "ucsd_ped2")
PRED_CSV = os.path.join(EVAL_DIR, "predictions.csv")
METRICS_JSON = os.path.join(EVAL_DIR, "metrics.json")
OUTPUT_PNG = os.path.join(EVAL_DIR, "roc_pr_curves.png")


def main():
    if not os.path.exists(PRED_CSV):
        print(f"Predictions CSV not found at: {PRED_CSV}")
        return

    df = pd.read_csv(PRED_CSV)
    y_true = df["ground_truth_label"].values
    score_col = "anomaly_relative_score" if "anomaly_relative_score" in df.columns else "predicted_anomaly_score"
    y_scores = df[score_col].values

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc_val = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc_val = auc(recall, precision)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # 1. ROC Curve
    ax1.plot(fpr, tpr, color="#1f77b4", lw=2.2, label=f"ViT-Tiny Ped2 (ROC-AUC = {roc_auc_val:.4f})")
    ax1.plot([0, 1], [0, 1], color="gray", lw=1.2, linestyle="--", label="Random Chance (0.50)")
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.05])
    ax1.set_xlabel("False Positive Rate (FPR)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("True Positive Rate (TPR / Recall)", fontsize=11, fontweight="bold")
    ax1.set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # 2. Precision-Recall Curve
    ax2.plot(recall, precision, color="#2ca02c", lw=2.2, label=f"ViT-Tiny Ped2 (PR-AUC = {pr_auc_val:.4f})")
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.05])
    ax2.set_xlabel("Recall (TPR)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Precision", fontsize=11, fontweight="bold")
    ax2.set_title("Precision-Recall Curve (PR)", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower left", frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("UCSD Ped2 Benchmark Evaluation Curves — Vision Transformer Anomaly Detection", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, bbox_inches="tight")
    plt.close()

    print(f"Successfully generated evaluation curves at: {OUTPUT_PNG}")
    print(f"ROC-AUC: {roc_auc_val:.4f} | PR-AUC: {pr_auc_val:.4f}")


if __name__ == "__main__":
    main()
