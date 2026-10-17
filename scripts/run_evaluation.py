"""
Research-Grade Benchmark Evaluation CLI.
Executes leakage-free evaluation across video-level splits on real datasets with ground-truth labels.
Exports metrics.json, predictions.csv, and evaluation_report.md.
"""
import argparse
import glob
import os
import sys
import numpy as np

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from configs.default_config import PipelineConfig
from src.inference.pipeline import VideoAnomalyInferencePipeline
from src.eval.dataset_evaluator import VideoAnomalyDatasetEvaluator


def main():
    parser = argparse.ArgumentParser(
        description="Research-Grade Video Anomaly Detection Benchmark Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset root folder containing Test videos and annotations")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to fitted normal reference checkpoint (.pt)")
    parser.add_argument("--output-dir", type=str, default="evaluation_results", help="Directory to save evaluation reports and metrics")
    parser.add_argument("--threshold-criterion", type=str, default="youden_j", choices=["youden_j", "f1", "target_fpr"], help="Validation threshold selection criterion")
    parser.add_argument("--sample-stride", type=int, default=2, help="Sample stride rate for frame extraction (default: 2)")
    parser.add_argument("--clip-length", type=int, default=16, help="Sliding window temporal length in frames (default: 16)")
    parser.add_argument("--backbone", type=str, default="vit_tiny_patch16_224", help="ViT backbone name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device ('cuda' or 'cpu')")

    args = parser.parse_args()

    print("=" * 65)
    print("RESEARCH-GRADE BENCHMARK EVALUATION ENGINE")
    print("=" * 65)
    print(f"Dataset Directory:    {args.data_dir}")
    print(f"Reference Checkpoint: {args.checkpoint}")
    print(f"Output Directory:     {args.output_dir}")
    print(f"Device:               {args.device}")
    print("-" * 65)

    if not os.path.exists(args.data_dir):
        print(f"[ERROR] Dataset directory not found: {args.data_dir}")
        print("Please verify the dataset folder path.")
        sys.exit(1)

    # Check if checkpoint exists, or if fit-reference is requested
    if not os.path.exists(args.checkpoint):
        # Look for normal training videos in data-dir to fit reference if requested
        train_dir = os.path.join(args.data_dir, "Train")
        if os.path.exists(train_dir):
            print(f"[INFO] Checkpoint not found at {args.checkpoint}.")
            print(f"[INFO] Found Train folder at {train_dir}. Automatically fitting normal reference baseline...")
            train_items = []
            for ext in ("*.mp4", "*.avi", "*.mkv"):
                train_items.extend(glob.glob(os.path.join(train_dir, ext)))
            if len(train_items) == 0:
                # Subdirectories containing image frames
                train_items = [
                    os.path.join(train_dir, d) for d in sorted(os.listdir(train_dir))
                    if os.path.isdir(os.path.join(train_dir, d)) and not d.endswith("_gt")
                ]
            if len(train_items) > 0:
                print(f"[INFO] Fitting reference on {len(train_items)} normal training sequence(s)...")
                cfg = PipelineConfig()
                cfg.model.backbone_name = args.backbone
                cfg.device = args.device
                cfg.preprocess.fps_sample_rate = args.sample_stride
                cfg.preprocess.clip_length = args.clip_length
                fit_pipe = VideoAnomalyInferencePipeline(config=cfg)
                os.makedirs(os.path.dirname(os.path.abspath(args.checkpoint)), exist_ok=True)
                fit_pipe.fit_normal_reference_video(train_items, save_checkpoint_path=args.checkpoint)
                print(f"[INFO] Normal reference fitted and saved to: {args.checkpoint}")
            else:
                print(f"[ERROR] No training videos or frame folders found in {train_dir}")
                sys.exit(1)
        else:
            print(f"[ERROR] Reference checkpoint not found: {args.checkpoint}")
            print("Please fit the normal reference baseline on training videos first using:")
            print("  .venv\\Scripts\\python.exe scripts/run_cli.py fit-reference --video <train_normal.mp4> --checkpoint <save.pt>")
            sys.exit(1)

    # Search for test videos: support video containers (.mp4/.avi) and image sequence directories
    video_extensions = ("*.mp4", "*.avi", "*.mkv")
    test_videos = []
    test_dir = os.path.join(args.data_dir, "Test")
    target_dir = test_dir if os.path.exists(test_dir) else args.data_dir

    for ext in video_extensions:
        test_videos.extend(sorted(glob.glob(os.path.join(target_dir, ext))))

    # If no video files, check for image sequence subdirectories
    if len(test_videos) == 0 and os.path.isdir(target_dir):
        subdirs = sorted([
            os.path.join(target_dir, d) for d in os.listdir(target_dir)
            if os.path.isdir(os.path.join(target_dir, d)) and not d.endswith("_gt") and not d.startswith(".")
        ])
        for sdir in subdirs:
            # Check if directory contains valid frames
            frames = [f for f in os.listdir(sdir) if not f.startswith(".") and os.path.splitext(f)[1].lower() in {".tif", ".png", ".bmp", ".jpg"}]
            if len(frames) > 0:
                test_videos.append(sdir)

    if len(test_videos) == 0:
        print(f"[ERROR] No video files or frame sequence folders found in: {target_dir}")
        print("Expected layout:")
        print(f"  {target_dir}/*.mp4 (or *.avi)")
        print(f"  OR {target_dir}/Test001/*.tif (image sequence directories)")
        sys.exit(1)

    print(f"Found {len(test_videos)} test video sequence(s).")

    # Standard UCSD Ped2 official frame-level anomaly intervals (1-indexed, inclusive)
    # Total frames: 2010 frames across 12 test sequences
    UCSD_PED2_OFFICIAL_INTERVALS = {
        "Test001": [(61, 180)],
        "Test002": [(95, 180)],
        "Test003": [(1, 146)],
        "Test004": [(31, 180)],
        "Test005": [(1, 129)],
        "Test006": [(1, 159)],
        "Test007": [(46, 180)],
        "Test008": [(1, 180)],
        "Test009": [(1, 120)],
        "Test010": [(1, 150)],
        "Test011": [(1, 180)],
        "Test012": [(88, 180)],
    }

    # Search for annotations
    ground_truth_map = {}
    anno_json = os.path.join(args.data_dir, "annotations.json")
    if os.path.exists(anno_json):
        import json
        with open(anno_json, "r") as f:
            raw_annos = json.load(f)
            for k, v in raw_annos.items():
                ground_truth_map[k] = np.array(v, dtype=int)
    else:
        # Check per-video formats: _gt folders, text files, numpy arrays, or official intervals
        for v_path in test_videos:
            v_name = os.path.basename(v_path)
            clean_name = os.path.splitext(v_name)[0]

            # 1. Check for pixel ground-truth mask folder (e.g. Test001_gt/*.bmp)
            gt_folder_candidates = [
                os.path.join(os.path.dirname(v_path), f"{clean_name}_gt"),
                os.path.join(args.data_dir, "Test", f"{clean_name}_gt"),
                os.path.join(args.data_dir, f"{clean_name}_gt")
            ]
            found_gt_folder = next((p for p in gt_folder_candidates if os.path.isdir(p)), None)

            if found_gt_folder:
                import cv2
                import re
                mask_files = sorted([
                    f for f in os.listdir(found_gt_folder)
                    if not f.startswith(".") and os.path.splitext(f)[1].lower() in {".bmp", ".png", ".tif"}
                ])
                # Count total frames in the test sequence
                if os.path.isdir(v_path):
                    tot_f = len([f for f in os.listdir(v_path) if not f.startswith(".") and os.path.splitext(f)[1].lower() in {".tif", ".png", ".bmp", ".jpg"}])
                else:
                    cap = cv2.VideoCapture(v_path)
                    tot_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()

                labels = np.zeros(tot_f, dtype=int)
                for mf in mask_files:
                    stem = os.path.splitext(mf)[0]
                    digits = re.findall(r'\d+', stem)
                    if not digits:
                        continue
                    m_idx = int(digits[-1]) - 1  # 1-based filename to 0-based index
                    if 0 <= m_idx < tot_f:
                        m_img = cv2.imread(os.path.join(found_gt_folder, mf), cv2.IMREAD_GRAYSCALE)
                        if m_img is not None and np.any(m_img > 0):
                            labels[m_idx] = 1
                ground_truth_map[v_name] = labels

            # 2. Check for official UCSD Ped2 interval table if applicable
            elif clean_name in UCSD_PED2_OFFICIAL_INTERVALS:
                if os.path.isdir(v_path):
                    tot_f = len([f for f in os.listdir(v_path) if os.path.splitext(f)[1].lower() in {".tif", ".png", ".bmp", ".jpg"}])
                else:
                    import cv2
                    cap = cv2.VideoCapture(v_path)
                    tot_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()

                labels = np.zeros(tot_f, dtype=int)
                for start_f, end_f in UCSD_PED2_OFFICIAL_INTERVALS[clean_name]:
                    # Convert 1-based [start_f, end_f] inclusive to 0-based slice
                    s_idx = max(0, start_f - 1)
                    e_idx = min(tot_f, end_f)
                    labels[s_idx:e_idx] = 1
                ground_truth_map[v_name] = labels

            # 3. Check for .txt or .npy ground truth
            else:
                txt_anno = os.path.join(args.data_dir, "annotations", f"{clean_name}.txt")
                npy_anno = os.path.join(args.data_dir, "annotations", f"{clean_name}.npy")
                if os.path.exists(npy_anno):
                    ground_truth_map[v_name] = np.load(npy_anno).astype(int)
                elif os.path.exists(txt_anno):
                    labels = np.loadtxt(txt_anno, dtype=int)
                    ground_truth_map[v_name] = labels

    if len(ground_truth_map) == 0:
        print(f"[ERROR] No ground truth annotations found in {args.data_dir}")
        print("Cannot run quantitative evaluation without ground-truth labels.")
        print("To run inference on unlabelled videos, use 'scripts/run_cli.py infer' instead.")
        sys.exit(1)

    # Initialize Pipeline & Evaluator
    cfg = PipelineConfig()
    cfg.model.backbone_name = args.backbone
    cfg.device = args.device
    cfg.preprocess.fps_sample_rate = args.sample_stride
    cfg.preprocess.clip_length = args.clip_length

    pipeline = VideoAnomalyInferencePipeline(config=cfg, reference_checkpoint=args.checkpoint)
    evaluator = VideoAnomalyDatasetEvaluator(pipeline=pipeline, output_dir=args.output_dir)

    metrics = evaluator.evaluate_annotated_videos(
        test_video_paths=test_videos,
        ground_truth_labels=ground_truth_map,
        threshold_selection_criterion=args.threshold_criterion,
        dataset_name=os.path.basename(os.path.abspath(args.data_dir))
    )

    print("\n" + "=" * 65)
    print("QUANTITATIVE BENCHMARK EVALUATION SUMMARY")
    print("=" * 65)
    print(f"ROC-AUC:           {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:            {metrics['pr_auc']:.4f}")
    print(f"F1-Score:          {metrics['f1_score']:.4f}")
    print(f"Precision:         {metrics['precision']:.4f}")
    print(f"Recall:            {metrics['recall']:.4f}")
    print(f"FPR:               {metrics['false_positive_rate']:.4f}")
    print(f"FPR @ 95% TPR:     {metrics['fpr_at_95_tpr']:.4f}")
    print(f"Throughput:        {metrics['throughput_fps']} FPS")
    print("=" * 65)
    print(f"Full report saved to: {os.path.join(args.output_dir, 'evaluation_report.md')}\n")


if __name__ == "__main__":
    main()
