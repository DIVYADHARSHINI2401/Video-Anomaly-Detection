# Focused Literature Review & Gap Analysis: Open-Set VAD with Risk Severity Estimation

**Document Version:** 1.0  
**Project:** Vision Transformer-Based Open-Set Video Anomaly Detection with Risk Severity Estimation  
**Status:** Verification Completed & Scope Frozen  

---

## 1. Executive Review & Objectives

Before finalizing model code and experimental protocols, this literature verification checks current state-of-the-art developments across four core pillars:
1. **Vision Transformers (ViT) in Video Anomaly Detection (VAD)**
2. **Open-Set / Unseen Video Anomaly Detection**
3. **Operational Risk & Severity Estimation in Surveillance**
4. **Lightweight & Computationally Efficient Video Architectures**

This document establishes the exact theoretical boundaries, verifies the literature gap, prevents overclaiming, and freezes the project novelty profile.

---

## 2. Deep Dive Across the 4 Research Pillars

### Pillar 1: Vision Transformers in Surveillance VAD
* **Background:** Standard surveillance VAD has historically relied on 2D CNNs (ResNet, VGG), 3D-CNNs (C3D, I3D, SlowFast), or Optical Flow networks (Flownet). 3D-CNNs capture spatiotemporal patterns but suffer from enormous memory footprints, slow frame inference rates (<15 FPS on standard edge hardware), and susceptibility to domain-specific artifacts.
* **Transformers in VAD:** Self-attention architectures (e.g., standard ViT, Swin Transformer, TimeSformer) model global spatial dependencies better than local convolution receptive fields. In surveillance, a frozen or lightly adapted ViT backbone captures fine-grained posture, anomalous object interactions, and structural anomalies without requiring multi-day training from scratch.
* **Literature Reality Check:** ViT is already widely applied to vision tasks. Merely plugging a ViT into an anomaly detection pipeline does **not** constitute a standalone research novelty. Our pipeline therefore leverages ViT as an **efficient visual representation backbone**, optimizing for latency and feature quality.

---

### Pillar 2: Open-Set and Unseen Anomaly Detection
* **Closed-Set vs. Open-Set VAD:**
  * **Unsupervised One-Class VAD (OC-VAD):** Trained exclusively on normal videos (e.g., CUHK Avenue, ShanghaiTech). Theoretically detects any deviation, but produces prohibitive false positive rates (FPR) because routine novelties (e.g., someone wearing a brightly colored backpack or a stray animal) trigger alarms.
  * **Weakly Supervised Closed-Set VAD:** Trained on known anomaly categories (e.g., UCF-Crime: *Abuse, Arrest, Arson, Assault, Burglary, Robbery, Shooting, Stealing*). When tested on anomalies outside these exact taxonomy labels (e.g., a person collapsing or a chemical spill), closed-set classifiers fail to detect or misclassify them.
  * **Open-Set VAD:** Introduced explicitly in recent benchmarks such as **UBnormal (CVPR 2022)** and **RSOB + HPF (2026)**. In this setting, the training set contains normal videos and a curated subset of *Known Anomalies*, while the test set evaluates both *Known Anomalies* and completely *Unseen Anomalies*.
* **Literature Reality Check:** Detecting unseen anomalies is **not** an unexplored problem. The RSOB + HPF (2026) paper and UBnormal benchmark have formalized supervised open-set VAD. Therefore, claiming "we are the first to detect unseen anomalies" is factually invalid. 
* **Our Refined Positioning:** We adopt a **controlled known vs. unseen benchmark split**, comparing how lightweight temporal aggregators combined with ViT representations perform on unseen classes relative to known classes.

---

### Pillar 3: Operational Risk & Severity Estimation
* **The "Alert Fatigue" Problem:** Surveillance operators monitor 20–100 feeds simultaneously. Standard VAD systems output binary flags (Normal vs. Anomaly). If every minor infraction (e.g., loitering or a bike parked on a footpath) generates the same alarm tier as an active physical fight or fire hazard, security personnel quickly ignore or disable the system.
* **State of the Art in Severity Modeling:**
  * Most existing research focuses exclusively on frame-level ROC-AUC.
  * Few works incorporate operational triage (e.g., multi-tier thresholding, persistence-weighted risk, or composite risk scoring).
* **Our Methodological Formulation:**
  We model Risk Severity $R(t)$ as a composite function of:
  $$R(t) = w_1 \cdot \mathcal{S}_{\text{anomaly}}(t) + w_2 \cdot \mathcal{P}_{\text{temporal}}(t) + w_3 \cdot \mathcal{U}_{\text{novelty}}(t)$$
  Where:
  * $\mathcal{S}_{\text{anomaly}}(t)$: Feature space deviation from normal centroid.
  * $\mathcal{P}_{\text{temporal}}(t)$: Persistence duration of the abnormal pattern across the temporal window.
  * $\mathcal{U}_{\text{novelty}}(t)$: Open-set novelty factor (flagging whether the event is known high-risk or novel uncharacterized behavior).
  * Output Tiers: **Low Risk** (logged for review), **Medium Risk** (operator visual check), **High Risk** (immediate alarm/dispatch).

---

### Pillar 4: Computational Efficiency & Lightweight Modeling
* **The Real-World Constraint:** Full 3D Transformers or heavy multi-modal Large Vision-Language Models (LVLMs) require massive GPUs (A100/H100) and cannot run economically on campus edge servers or standard workstations.
* **Our Architectural Solution:**
  1. Extract patch-level or CLS token embeddings from a compact pretrained ViT (e.g., `vit_tiny_patch16_224` or `vit_small_patch16_224`).
  2. Aggregate temporal sequence context over sliding temporal windows ($T = 16$ or $32$ frames) using a lightweight Gated Recurrent Unit (GRU) or 1D Temporal Convolution Network (TCN).
  3. Measure wall-clock latency (ms), throughput (FPS), and parameter count to establish real-world viability.

---

## 3. Comparison Matrix of Related Approaches

| Approach | Visual Backbone | Temporal Modeling | Open-Set Protocol | Risk / Priority Tiers | Target Deployment |
|---|---|---|---|---|---|
| **C3D / I3D Baselines** | 3D-CNN | Native 3D Spatiotemporal | ❌ Closed-set only | ❌ Binary alarm | High-cost GPU |
| **Sultani et al. (UCF-Crime)** | 2D/3D CNN (MIL) | Sliding clip average | ❌ Closed-set only | ❌ Anomaly score only | Cloud GPU |
| **UBnormal (CVPR 2022)** | Object Detector + GCN | Graph temporal convolution | ✅ Synthesized open-set | ❌ Binary alarm | Research benchmark |
| **RSOB + HPF (2026)** | Custom HPF Feature Space | Multi-stage alignment | ✅ Supervised open-set | ❌ Binary detection | High complexity |
| **Our Proposed Pipeline** | **Pretrained ViT (`timm`)** | **Lightweight GRU / TCN** | **✅ Explicit Known vs. Unseen Split** | **✅ Low / Med / High Severity** | **Practical Campus / CCTV Edge** |

---

## 4. Frozen Claims & Non-Claims Boundary

### What the Project DOES Claim:
1. **Pragmatic Integration:** A clean, modular pipeline coupling off-the-shelf ViT representations with a lightweight temporal module, avoiding the overhead of heavy 3D-CNNs.
2. **Transparent Open-Set Benchmarking:** Explicit measurement of performance degradation between known anomaly categories and novel unseen anomaly categories.
3. **Actionable Risk Prioritization:** An interpretable 3-tier severity scoring mechanism (Low, Medium, High) that translates raw anomaly scores into operational surveillance alerts.
4. **Efficiency Profiling:** Systematic benchmarking of throughput (FPS), memory footprint, and inference latency across CPU and GPU hardware.

### What the Project Does NOT Claim:
1. We do **not** claim to invent Vision Transformers.
2. We do **not** claim to invent the concept of open-set video anomaly detection.
3. We do **not** claim to outperform massive multi-modal foundation models on zero-shot benchmarks.
4. We do **not** claim the risk severity module replaces human security judgment; it functions as an operator triage assistant.

---

## 5. Frozen Title and Final Scope

* **Final Project Title:**  
  `Vision Transformer-Based Open-Set Video Anomaly Detection with Risk Severity Estimation`
* **Primary Target Datasets:**  
  * Controlled partition on **UCF-Crime** or **ShanghaiTech / Avenue** (partitioning categories into $K_{\text{train}}$ known classes and $U_{\text{test}}$ unseen classes).
* **Core Evaluation Metrics:**  
  * Detection: Frame-level ROC-AUC, PR-AUC, False Positive Rate (FPR), Precision, Recall, F1-score.
  * Open-set: Known AUROC vs. Unseen AUROC vs. Harmonic Mean.
  * Efficiency: Inference Latency (ms), Frames Per Second (FPS), Model Parameter Count.
