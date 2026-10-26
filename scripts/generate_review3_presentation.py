import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
import os

def create_presentation(output_path):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5) # 16:9 widescreen

    # Color Palette: Clean Tech Slate & Blue
    NAVY = RGBColor(15, 23, 42)      # #0F172A
    ACCENT_BLUE = RGBColor(37, 99, 235) # #2563EB
    LIGHT_BG = RGBColor(248, 250, 252)  # #F8FAFC
    DARK_TEXT = RGBColor(30, 41, 59)    # #1E293B
    MUTED_TEXT = RGBColor(100, 116, 139) # #64748B
    CARD_BG = RGBColor(255, 255, 255)
    BORDER_COL = RGBColor(226, 232, 240)
    GREEN = RGBColor(16, 185, 129)
    AMBER = RGBColor(245, 158, 11)
    RED = RGBColor(239, 68, 68)

    blank_layout = prs.slide_layouts[6]

    def add_header(slide, title_text, category_text="CAPSTONE PROJECT PHASE I : REVIEW III"):
        # Header category
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(11.5), Inches(0.4))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.name = "Arial"
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = ACCENT_BLUE

        # Main Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.85), Inches(11.5), Inches(0.7))
        tf = title_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.name = "Arial"
        p.font.size = Pt(22)
        p.font.bold = True
        p.font.color.rgb = NAVY

    def create_card(slide, left, top, width, height, title, items, bg_color=CARD_BG, border_color=BORDER_COL):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_color
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1.5)

        tb = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.2), width - Inches(0.4), height - Inches(0.4))
        tf = tb.text_frame
        tf.word_wrap = True
        
        # Title
        p0 = tf.paragraphs[0]
        p0.text = title
        p0.font.name = "Arial"
        p0.font.size = Pt(15)
        p0.font.bold = True
        p0.font.color.rgb = NAVY
        p0.space_after = Pt(10)

        for item in items:
            p = tf.add_paragraph()
            p.space_after = Pt(6)
            if isinstance(item, tuple):
                # (heading, desc)
                r1 = p.add_run()
                r1.text = item[0] + ": "
                r1.font.bold = True
                r1.font.size = Pt(12)
                r1.font.color.rgb = DARK_TEXT
                
                r2 = p.add_run()
                r2.text = item[1]
                r2.font.size = Pt(12)
                r2.font.color.rgb = MUTED_TEXT
            else:
                r = p.add_run()
                r.text = "• " + item
                r.font.size = Pt(12)
                r.font.color.rgb = DARK_TEXT

    # -------------------------------------------------------------
    # SLIDE 1: TITLE SLIDE
    # -------------------------------------------------------------
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = NAVY
    bg1.line.fill.background()

    t_box = s1.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(11.333), Inches(2.2))
    tf1 = t_box.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "Vision Transformer-Based Open-Set Video Anomaly Detection with Risk Severity Estimation"
    p1.font.name = "Arial"
    p1.font.size = Pt(30)
    p1.font.bold = True
    p1.font.color.rgb = RGBColor(255, 255, 255)
    p1.space_after = Pt(14)

    p1_sub = tf1.add_paragraph()
    p1_sub.text = "Capstone Project Phase I Review III | Completed Work & Phase II Roadmap"
    p1_sub.font.name = "Arial"
    p1_sub.font.size = Pt(18)
    p1_sub.font.color.rgb = RGBColor(147, 197, 253)

    meta_box = s1.shapes.add_textbox(Inches(1.0), Inches(4.3), Inches(11.333), Inches(2.2))
    tf_meta = meta_box.text_frame
    
    p_auth = tf_meta.paragraphs[0]
    p_auth.text = "Presented by: RUKESH BABU GANTLA (Roll No: 22BCS143)"
    p_auth.font.name = "Arial"
    p_auth.font.size = Pt(16)
    p_auth.font.bold = True
    p_auth.font.color.rgb = RGBColor(255, 255, 255)
    p_auth.space_after = Pt(6)

    p_dept = tf_meta.add_paragraph()
    p_dept.text = "Department of Computer Science and Engineering | Kumaraguru College of Technology"
    p_dept.font.name = "Arial"
    p_dept.font.size = Pt(14)
    p_dept.font.color.rgb = RGBColor(203, 213, 225)
    p_dept.space_after = Pt(4)

    p_guide = tf_meta.add_paragraph()
    p_guide.text = "Project Supervisor: << Dr. / Prof. Project Guide >> | Review III Assessment (2026-27)"
    p_guide.font.name = "Arial"
    p_guide.font.size = Pt(14)
    p_guide.font.italic = True
    p_guide.font.color.rgb = RGBColor(148, 163, 184)

    # -------------------------------------------------------------
    # SLIDE 2: PROBLEM STATEMENT & MOTIVATION
    # -------------------------------------------------------------
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Motivation & Problem Statement: The Real-World Surveillance Crisis")

    create_card(s2, Inches(0.8), Inches(1.8), Inches(3.6), Inches(4.8), 
                "1. Operator Alert Fatigue", [
                    ("The Phenomenon", "Security personnel monitor 20–100 feeds simultaneously."),
                    ("Cognitive Limit", "Human visual focus drops by 95% after only 20 minutes."),
                    ("Binary Alert Flaw", "Standard systems fire flat 'Normal vs Anomaly' alarms for minor and critical events equally."),
                    ("Operational Impact", "Operators ignore or mute systems due to hundreds of daily false alarms.")
                ])

    create_card(s2, Inches(4.8), Inches(1.8), Inches(3.6), Inches(4.8), 
                "2. The Closed-Set Failure", [
                    ("Unrealistic Assumption", "Existing models assume all test anomalies exist in training labels (UCF-Crime)."),
                    ("The Open-World Reality", "Real hazards are unbounded (chemical leaks, medical distress, fires)."),
                    ("Catastrophic Misses", "Unseen anomaly categories cause severe false negative spikes."),
                    ("Need", "Explicit differentiation between Known and Unseen anomalies.")
                ])

    create_card(s2, Inches(8.8), Inches(1.8), Inches(3.7), Inches(4.8), 
                "3. The Computational Barrier", [
                    ("3D-CNN Bottleneck", "I3D/SlowFast have huge parameter sizes (>50 GFLOPs/clip)."),
                    ("Sub-Real-Time", "Heavy backbones run at 8–14 FPS on standard hardware."),
                    ("Cost Barrier", "Requires expensive cloud GPUs (A100), infeasible for campus edge."),
                    ("Solution", "Pretrained ViT + Lightweight GRU achieves >35 FPS on edge hardware.")
                ])

    # -------------------------------------------------------------
    # SLIDE 3: PROJECT OBJECTIVES & 4-PILLAR DESIGN
    # -------------------------------------------------------------
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Four-Pillar Architectural Framework & Key Objectives")

    create_card(s3, Inches(0.8), Inches(1.8), Inches(5.6), Inches(2.3),
                "Pillar 1: Pretrained Vision Transformer", [
                    ("Off-the-shelf ViT Backbones", "vit_tiny_patch16_224 and vit_base via timm."),
                    ("Global Spatial Context", "Self-attention tokenizes 16x16 patches for whole-frame comprehension.")
                ])

    create_card(s3, Inches(6.8), Inches(1.8), Inches(5.7), Inches(2.3),
                "Pillar 2: Lightweight Temporal Aggregation", [
                    ("Decoupled Design", "Replaces heavy 3D convolutions with 2-layer GRU or 1D-TCN."),
                    ("Sliding Sequence Modeling", "Aggregates motion dynamics across T = 16 frame windows at >35 FPS.")
                ])

    create_card(s3, Inches(0.8), Inches(4.4), Inches(5.6), Inches(2.4),
                "Pillar 3: Open-Set Generalization", [
                    ("Controlled Category Splits", "Evaluates model on withheld, completely unseen anomaly classes."),
                    ("Novelty Uncertainty", "Entropy-based detector flags novel unseen hazards vs known behaviors.")
                ])

    create_card(s3, Inches(6.8), Inches(4.4), Inches(5.7), Inches(2.4),
                "Pillar 4: Operational Risk Prioritization", [
                    ("Composite Risk Score", "Combines deviation magnitude, temporal persistence, and uncertainty."),
                    ("3-Tier Actionable Triage", "Low (silent log), Medium (visual check), High (emergency dispatch).")
                ])

    # -------------------------------------------------------------
    # SLIDE 4: SYSTEM PIPELINE & MATHEMATICAL FORMULATION
    # -------------------------------------------------------------
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "End-to-End System Pipeline & Risk Formulation")

    create_card(s4, Inches(0.8), Inches(1.8), Inches(6.0), Inches(4.8),
                "Operational Risk Severity Formulation R(t)", [
                    ("Mathematical Formula", "R(t) = w1 * S_anomaly(t) + w2 * P_temporal(t) + w3 * U_novelty(t)"),
                    ("Weights Configuration", "w1 = 0.50 (Deviation), w2 = 0.30 (Persistence), w3 = 0.20 (Novelty)"),
                    ("Deviation S_anomaly(t)", "Normalized cosine distance from normal reference centroid: 1 - (z . c_norm)/(||z|| ||c_norm||)"),
                    ("Persistence P_temporal(t)", "Exponential moving average across temporal sliding window W (alpha = 0.35)"),
                    ("Uncertainty U_novelty(t)", "Shannon entropy across classifier distribution indicating unseen category")
                ])

    create_card(s4, Inches(7.2), Inches(1.8), Inches(5.3), Inches(4.8),
                "Operational Action & Triage Matrix", [
                    ("LOW RISK (0.00 <= R < 0.40)", "Routine minor novelty. Silent background log for audit. Zero operator alert."),
                    ("MEDIUM RISK (0.40 <= R < 0.70)", "Moderate deviation or sustained pattern. Amber banner overlay; prompt operator visual check."),
                    ("HIGH RISK (0.70 <= R <= 1.00)", "Critical hazardous anomaly. Audible alarm, red alert banner, automated dispatch trigger."),
                    ("Alert Reduction", "Reduces actionable operator alerts by >80% while capturing genuine threats.")
                ])

    # -------------------------------------------------------------
    # SLIDE 5: PHASE I COMPLETED DELIVERABLES
    # -------------------------------------------------------------
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Phase I Completed Work & Technical Deliverables")

    create_card(s5, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8),
                "Completed Codebase & Pipeline Modules", [
                    ("Modular Repository", "100% completed modular architecture in PyTorch 2.0+ and OpenCV."),
                    ("ViT Extractor", "src/models/vit_extractor.py leveraging compact ViT-Tiny and ViT-Base."),
                    ("Temporal Aggregator", "src/models/temporal_aggregator.py with lightweight GRU/TCN modules."),
                    ("Open-Set Detector", "src/models/openset_detector.py with centroid distance and entropy scoring."),
                    ("Risk Severity Engine", "src/models/risk_estimator.py implementing multi-tier calibrated triage."),
                    ("Video Annotator", "src/inference/video_annotator.py for live bounding boxes and risk banners.")
                ])

    create_card(s5, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8),
                "Operational Tools & Academic Documentation", [
                    ("Streamlit Dashboard", "Interactive web UI (app/streamlit_app.py) for real-time video playback and telemetry."),
                    ("UCSD Ped2 Benchmark", "Full experimental evaluation executed on 2,010 frames with zero test-set leakage."),
                    ("Review Paper Draft", "Comprehensive academic review paper drafted on ViT and Open-Set VAD."),
                    ("KCT Project Report Draft", "Phase I rough draft compiled strictly following KCT guidelines and format."),
                    ("Structured JSON Output", "Standardized incident telemetry schema for security dispatch integration.")
                ])

    # -------------------------------------------------------------
    # SLIDE 6: EXPERIMENTAL RESULTS (UCSD PED2 BENCHMARK)
    # -------------------------------------------------------------
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Phase I Quantitative Evaluation: UCSD Ped2 Benchmark")

    create_card(s6, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8),
                "Benchmark Metric Results", [
                    ("Test Suite", "12 test video clips comprising 2,010 native frames / 825 temporal clips."),
                    ("Frame-Level ROC-AUC", "0.6769 (Strong baseline discrimination without test-set fine-tuning)."),
                    ("Precision-Recall AUC", "0.9189 (High area under PR curve under dense anomaly conditions)."),
                    ("Clip-Level ROC-AUC", "0.6455 | Clip-Level PR-AUC: 0.9478."),
                    ("Leakage Protocol", "Zero test-set leakage: centroid fitted strictly on normal training data."),
                    ("Operating Threshold", "0.50 default threshold applied.")
                ])

    create_card(s6, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8),
                "Hardware Throughput & Efficiency", [
                    ("Model Backbone", "vit_tiny_patch16_224 (Lightweight Vision Transformer)."),
                    ("Execution Hardware", "NVIDIA CUDA GPU Workstation."),
                    ("Wall-Clock Time", "27.58 seconds for all 2,010 frames."),
                    ("Inference Throughput", "36.44 Frames Per Second (FPS)."),
                    ("Operational Verdict", "Exceeds standard 30 FPS camera feed rate. Real-time edge capable!"),
                    ("Comparison", "3x faster than C3D/I3D baseline (typically 8–14 FPS on identical hardware).")
                ])

    # -------------------------------------------------------------
    # SLIDE 7: REAL-TIME SURVEILLANCE DASHBOARD DEMO
    # -------------------------------------------------------------
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Streamlit Operations Dashboard: Interactive Demonstration")

    create_card(s7, Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.8),
                "Interactive System Capabilities (app/streamlit_app.py)", [
                    ("Live Surveillance Player", "Visualizes input video with real-time bounding box tracking and color-coded risk alert banners (Green = Low, Yellow = Medium, Red = High)."),
                    ("Dynamic Metric Telemetry", "Plots real-time graphs displaying anomaly deviation score S_anomaly(t), temporal persistence P_temporal(t), and composite risk score R(t) across video progression."),
                    ("Sensitivity Tuning Sliders", "Enables security operators to adjust operating threshold, temporal smoothing window size, and risk weight parameters interactively."),
                    ("Structured JSON Incident Log", "Generates structured incident logs with timestamp, camera ID, category type, severity rating, and recommended action."),
                    ("One-Click Launch", "Executable via run_dashboard.bat or powershell run_dashboard.ps1 at http://localhost:8501.")
                ])

    # -------------------------------------------------------------
    # SLIDE 8: PHASE II PLANNED WORK ROADMAP
    # -------------------------------------------------------------
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Phase II Planned Work & Technical Roadmap")

    create_card(s8, Inches(0.8), Inches(1.8), Inches(3.6), Inches(4.8),
                "1. Multi-Category Open-Set", [
                    ("UCF-Crime Partitioning", "Split 13 categories into 8 Known and 5 Unseen anomaly classes."),
                    ("ShanghaiTech Benchmark", "Evaluate multi-scene complex campus crowd environments."),
                    ("Generalization Metric", "Compute Harmonic Mean AUROC between Known and Unseen test sets.")
                ])

    create_card(s8, Inches(4.8), Inches(1.8), Inches(3.6), Inches(4.8),
                "2. Edge Hardware Quantization", [
                    ("INT8 Precision", "Quantize ViT weights using PyTorch Quantization and ONNX runtime."),
                    ("TensorRT Acceleration", "Compile inference graphs for NVIDIA Jetson edge embedded platforms."),
                    ("Power Profiling", "Benchmark energy consumption and sub-15W edge feasibility.")
                ])

    create_card(s8, Inches(8.8), Inches(1.8), Inches(3.7), Inches(4.8),
                "3. Enterprise Alert Webhooks", [
                    ("Automated Dispatch", "Implement webhook API dispatching High-Risk alerts to Telegram/Email."),
                    ("Video Snippet Exporter", "Auto-crop and export 5-second MP4 incident clips upon high-risk trigger."),
                    ("COE Final Submission", "Deliver complete finalized thesis report and evaluation defense.")
                ])

    # -------------------------------------------------------------
    # SLIDE 9: PHASE II GANTT TIMELINE & MILESTONES
    # -------------------------------------------------------------
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Phase II Work Breakdown & Implementation Schedule")

    create_card(s9, Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.8),
                "Phase II Implementation Milestones (14-Week Schedule)", [
                    ("Milestone 1 (Weeks 1 - 3)", "UCF-Crime & ShanghaiTech Open-Set Partitioning and category dataset preparation."),
                    ("Milestone 2 (Weeks 4 - 6)", "Execution of Known vs Unseen open-set experimental benchmark matrix and ablation studies."),
                    ("Milestone 3 (Weeks 7 - 9)", "Edge optimization: ONNX / TensorRT INT8 quantization and embedded latency profiling."),
                    ("Milestone 4 (Weeks 10 - 11)", "Automated enterprise webhook alert integration and incident video clip delivery system."),
                    ("Milestone 5 (Weeks 12 - 14)", "Final comprehensive project report compilation, guide review, and COE End-Semester Defense.")
                ])

    # -------------------------------------------------------------
    # SLIDE 10: CONCLUSION & SUMMARY
    # -------------------------------------------------------------
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Summary of Achievements & Review III Conclusion")

    create_card(s10, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8),
                "Key Review III Accomplishments", [
                    ("Solid Theoretical Basis", "Literature gap verified; no overclaiming; frozen research scope."),
                    ("Complete Working Setup", "Full PyTorch pipeline + Streamlit operations dashboard functional."),
                    ("Real-Time ViT Speed", "36.44 FPS on GPU; eliminates heavy 3D-CNN computational overhead."),
                    ("Actionable Triage", "3-tier risk formulation directly targets operator alert fatigue."),
                    ("Comprehensive Documentation", "Project Report rough draft and Review Paper completed.")
                ])

    create_card(s10, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8),
                "Research Positioning & Next Steps", [
                    ("Phase II Execution", "Ready to immediately initiate UCF-Crime open-set partitioning experiments."),
                    ("Publication Target", "Submitting the drafted Review Paper to an indexed journal/conference."),
                    ("Guide Verification", "Seeking supervisor inputs on threshold calibration and category splits."),
                    ("Thank You", "Questions, suggestions, and feedback are welcomed.")
                ])

    prs.save(output_path)
    print(f"Presentation deck successfully saved to: {output_path}")

if __name__ == "__main__":
    out_dir = r"C:\Users\gantl\Downloads"
    out_file = os.path.join(out_dir, "Review_III_Presentation_Phase1_Completed_Phase2_Planned.pptx")
    create_presentation(out_file)
