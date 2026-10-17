"""
Surveillance Video Anomaly Detection Demo Script.
Wrapper around scripts/run_cli.py providing seamless command routing
for fitting reference baselines and running video inference.
"""
import sys
import os

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.run_cli import main

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("=" * 65)
        print("SURVEILLANCE VIDEO ANOMALY DETECTION CLI")
        print("=" * 65)
        print("To process surveillance videos, use the two-step reference workflow:\n")
        print("Step 1: Fit a normal reference baseline from a normal video:")
        print("  .venv\\Scripts\\python.exe scripts/run_cli.py fit-reference \\")
        print("      --video path/to/normal_video.mp4 \\")
        print("      --checkpoint checkpoints/normal_reference.pt\n")
        print("Step 2: Run inference on an independent target video:")
        print("  .venv\\Scripts\\python.exe scripts/run_cli.py infer \\")
        print("      --video path/to/test_video.mp4 \\")
        print("      --checkpoint checkpoints/normal_reference.pt \\")
        print("      --output-dir outputs\n")
        print("For detailed arguments, run:")
        print("  .venv\\Scripts\\python.exe scripts/run_cli.py --help")
        print("=" * 65)
        sys.exit(0)

    main()
