# Research Evaluation Report: Video Anomaly Detection

**Dataset:** ucsd_ped2  
**Model Backbone:** vit_tiny_patch16_224  
**Device:** cuda  
**Evaluation Status:** EXPERIMENTAL — UNVALIDATED  

---

## 1. Summary of Quantitative Metrics

| Metric | Value | Interpretation |
|---|---|---|
| **ROC-AUC** | **0.6769** | Threshold-independent discrimination capability across all operating points |
| **PR-AUC** | **0.9189** | Precision-Recall curve area (informative under class imbalance) |
| **Operating Threshold** | **0.5** | Default threshold used; no separate validation set provided. |
| **Precision** | **0.0** | True Positives / (True Positives + False Positives) |
| **Recall (Sensitivity)** | **0.0** | True Positives / (True Positives + False Negatives) |
| **F1-Score** | **0.0** | Harmonic mean of Precision and Recall |
| **False Positive Rate (FPR)** | **0.0** | False Positives / (False Positives + True Negatives) |
| **FPR @ 95% TPR** | **0.942** | False alarm rate when detecting 95% of true anomalous events |

---

## 2. Confusion Matrix

| | Predicted Normal | Predicted Anomaly |
|---|---|---|
| **Actual Normal** | 362 (TN) | 0 (FP) |
| **Actual Anomaly** | 1648 (FN) | 0 (TP) |

* **Total Evaluated Clips/Frames:** 2010  
* **Anomaly Ground-Truth Ratio:** 81.99%  

---

## 3. Hardware Runtime & Efficiency

* **Wall-Clock Evaluation Time:** 27.58 s  
* **Inference Throughput:** 36.44 Frames Per Second (FPS)  

---

## 4. Methodological Scope & Limitations

> [!WARNING]
> - **Uncalibrated Feature Distance:** The anomaly score represents relative cosine deviation from the normal reference centroid in Vision Transformer representation space. It is **not** a posterior probability.
> - **Zero Test-Set Leakage:** The decision threshold and reference centroid were fitted strictly without access to test videos or test labels.
