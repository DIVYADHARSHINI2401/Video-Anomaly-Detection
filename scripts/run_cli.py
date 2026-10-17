"""
Surveillance Video Anomaly Detection CLI.
Provides explicit subcommands for:
1. 'fit-reference': Fits a normal reference baseline from a designated normal video.
2. 'infer': Evaluates a target video against a saved reference checkpoint.
"""
import argparse
import os
import sys

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from configs.default_config import PipelineConfig
from src.inference.pipeline import VideoAnomalyInferencePipeline


def cmd_fit_reference(args):
    print("=" * 65)
    print("SURVEILLANCE ANOMALY PIPELINE: FIT NORMAL REFERENCE BASELINE")
    print("=" * 65)
    print(f"Normal Reference Video: {args.video}")
    print(f"Output Checkpoint:      {args.checkpoint}")
    print(f"ViT Backbone:           {args.backbone}")
    print(f"Device:                 {args.device}")
    print("-" * 65)

    if not os.path.exists(args.video):
        print(f"[ERROR] Reference video file not found at: {args.video}")
        sys.exit(1)

    cfg = PipelineConfig()
    cfg.model.backbone_name = args.backbone
    cfg.device = args.device

    pipeline = VideoAnomalyInferencePipeline(config=cfg)

    try:
        pipeline.fit_normal_reference_video(
            video_path=args.video,
            save_checkpoint_path=args.checkpoint
        )
        print("-" * 65)
        print(f"[SUCCESS] Normal reference baseline fitted and saved to: {args.checkpoint}")
        print("You can now evaluate test surveillance videos using:")
        print(f"  .venv\\Scripts\\python.exe scripts/run_cli.py infer --video <target_video.mp4> --checkpoint {args.checkpoint}")
        print("=" * 65)
    except Exception as err:
        print(f"[ERROR] Failed to fit reference baseline: {err}")
        sys.exit(1)


def cmd_infer(args):
    print("=" * 65)
    print("SURVEILLANCE ANOMALY PIPELINE: LOCAL VIDEO INFERENCE")
    print("=" * 65)
    print(f"Target Video:           {args.video}")
    print(f"Reference Checkpoint:   {args.checkpoint}")
    print(f"Output Directory:       {args.output_dir}")
    print(f"Device:                 {args.device}")
    print(f"Render Video HUD:       {args.render}")
    print("-" * 65)

    # Validate checkpoint
    if not args.checkpoint:
        print("\n[ERROR] Missing required '--checkpoint' argument.")
        print("The anomaly detector requires a fitted normal reference model to compute distances.")
        print("Step 1: Fit a reference baseline from a normal video first:")
        print("  .venv\\Scripts\\python.exe scripts/run_cli.py fit-reference --video <normal.mp4> --checkpoint checkpoints/normal_ref.pt")
        print("Step 2: Run inference using that checkpoint:")
        print("  .venv\\Scripts\\python.exe scripts/run_cli.py infer --video <test.mp4> --checkpoint checkpoints/normal_ref.pt")
        sys.exit(1)

    if not os.path.exists(args.checkpoint):
        print(f"\n[ERROR] Checkpoint file not found: {args.checkpoint}")
        print("Please verify the path or create the reference checkpoint first using 'fit-reference'.")
        sys.exit(1)

    if not os.path.exists(args.video):
        print(f"\n[ERROR] Target video file not found: {args.video}")
        sys.exit(1)

    cfg = PipelineConfig()
    cfg.model.backbone_name = args.backbone
    cfg.device = args.device
    cfg.output_dir = args.output_dir

    try:
        pipeline = VideoAnomalyInferencePipeline(config=cfg, reference_checkpoint=args.checkpoint)
        results = pipeline.process_video(
            video_path=args.video,
            render_annotated=args.render,
            output_dir=args.output_dir
        )

        # Standard Structured Output
        print("\n" + "#" * 55)
        print("VIDEO ANALYSIS RESULT [EXPERIMENTAL — UNVALIDATED]")
        print(f"Input Video:  {results['input_filename']}")
        print(f"Status:       {results['status']}")
        print(f"Category:     {results['category']}")
        print(f"Risk Level:   {results['risk_level']}")
        if results['detected_segments']:
            print(f"Segments:     {', '.join(results['detected_segments'])}")
        else:
            print("Segments:     No anomalous segments detected")
        print(f"Relative Dist: {results['anomaly_score']} (Scale: [0.0 - 1.0])")
        print(f"Throughput:   {results['throughput_fps']} FPS")
        print("#" * 55)

        print("\nGenerated Reports:")
        print(f"  CSV Results:  {results['csv_results_path']}")
        print(f"  JSON Summary: {os.path.join(args.output_dir, os.path.splitext(results['input_filename'])[0] + '_summary.json')}")
        if results.get("annotated_video_path"):
            print(f"  Annotated MP4:{results['annotated_video_path']}")
        print("=" * 65 + "\n")

    except Exception as err:
        print(f"\n[ERROR] Video inference failed: {err}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="ViT Open-Set Video Anomaly Detection CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to execute")

    # Subcommand: fit-reference
    fit_p = subparsers.add_parser("fit-reference", help="Fit normal reference model from a normal surveillance video")
    fit_p.add_argument("--video", type=str, required=True, help="Path to designated normal surveillance video (.mp4/.avi)")
    fit_p.add_argument("--checkpoint", type=str, default="checkpoints/normal_reference.pt", help="Output path for fitted checkpoint")
    fit_p.add_argument("--backbone", type=str, default="vit_tiny_patch16_224", help="ViT backbone name")
    fit_p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device ('cuda' or 'cpu')")

    # Subcommand: infer
    inf_p = subparsers.add_parser("infer", help="Process and evaluate a test video against a reference checkpoint")
    inf_p.add_argument("--video", type=str, required=True, help="Path to input surveillance video to analyze (.mp4/.avi)")
    inf_p.add_argument("--checkpoint", type=str, required=True, help="Path to fitted normal reference checkpoint (.pt)")
    inf_p.add_argument("--output-dir", type=str, default="outputs", help="Output directory for reports & videos")
    inf_p.add_argument("--render", action="store_true", default=True, help="Render annotated output video")
    inf_p.add_argument("--no-render", dest="render", action="store_false", help="Skip rendering output video")
    inf_p.add_argument("--backbone", type=str, default="vit_tiny_patch16_224", help="ViT backbone name")
    inf_p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device ('cuda' or 'cpu')")

    args = parser.parse_args()

    if args.command == "fit-reference":
        cmd_fit_reference(args)
    elif args.command == "infer":
        cmd_infer(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
