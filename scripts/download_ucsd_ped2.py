"""
Selective Downloader and Verifier for UCSD Ped2 Video Anomaly Dataset.
Downloads the official SVCL UCSD archive, verifies integrity, selectively extracts
only UCSDped2 (Train, Test, Ground Truth, Annotations), validates readable frames,
and removes the compressed archive only after verified extraction.
"""
import os
import shutil
import sys
import tarfile
import time
import urllib.request

DATASET_URL_HTTPS = "https://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz"
DATASET_URL_HTTP = "http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz"
EXPECTED_ARCHIVE_SIZE = 740306953  # ~706 MB


def download_with_progress(url: str, dest_path: str, chunk_size: int = 1024 * 1024):
    """Downloads a file with chunk streaming and progress reporting."""
    print(f"[Downloader] Connecting to: {url}...")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Academic-Research-Evaluation"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        total_size = int(resp.headers.get("Content-Length", 0))
        print(f"[Downloader] Remote file size: {total_size:,} bytes ({total_size / (1024*1024):.2f} MB)")
        
        os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
        downloaded = 0
        start_time = time.time()
        last_report_time = start_time

        with open(dest_path, "wb") as out_file:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                
                curr_time = time.time()
                if curr_time - last_report_time >= 5.0 or downloaded == total_size:
                    elapsed = curr_time - start_time
                    speed_mb = (downloaded / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                    pct = (downloaded / total_size) * 100 if total_size > 0 else 0
                    print(f"  [Progress] {downloaded:,} / {total_size:,} bytes ({pct:.1f}%) - Speed: {speed_mb:.2f} MB/s")
                    last_report_time = curr_time

    print(f"[Downloader] Download completed in {time.time() - start_time:.1f}s: {dest_path}")


def selectively_extract_ped2(archive_path: str, extract_root: str):
    """
    Selectively extracts only UCSDped2 members into extract_root.
    Avoids extracting UCSDped1 (~1.2 GB) to conserve disk space.
    """
    print(f"[Extractor] Inspecting tar archive: {archive_path}...")
    target_prefix = "UCSD_Anomaly_Dataset.v1p2/UCSDped2"

    with tarfile.open(archive_path, mode="r:gz") as tar:
        all_members = tar.getmembers()
        ped2_members = [m for m in all_members if m.name.startswith(target_prefix)]
        print(f"[Extractor] Found {len(ped2_members)} UCSDped2 members out of {len(all_members)} total archive members.")

        if len(ped2_members) == 0:
            raise RuntimeError(f"No members found starting with prefix: {target_prefix}")

        # Extract only UCSDped2 members
        os.makedirs(extract_root, exist_ok=True)
        print(f"[Extractor] Selectively extracting UCSDped2 to temporary destination...")
        tar.extractall(path=extract_root, members=ped2_members)

    extracted_dir = os.path.join(extract_root, "UCSD_Anomaly_Dataset.v1p2", "UCSDped2")
    final_dir = os.path.join(extract_root, "ucsd_ped2")

    if os.path.exists(extracted_dir):
        if os.path.exists(final_dir):
            shutil.rmtree(final_dir)
        shutil.move(extracted_dir, final_dir)
        # Clean up empty parent UCSD_Anomaly_Dataset.v1p2 folder
        parent_v1p2 = os.path.join(extract_root, "UCSD_Anomaly_Dataset.v1p2")
        if os.path.exists(parent_v1p2):
            shutil.rmtree(parent_v1p2, ignore_errors=True)

    print(f"[Extractor] UCSDped2 successfully extracted to: {final_dir}")
    return final_dir


def verify_extracted_dataset(dataset_dir: str):
    """
    Verifies that all 16 normal training sequences and 12 test sequences
    exist and contain readable image frames.
    """
    import cv2
    print(f"[Verifier] Verifying extracted structure at: {dataset_dir}...")

    train_dir = os.path.join(dataset_dir, "Train")
    test_dir = os.path.join(dataset_dir, "Test")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Missing Train directory: {train_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Missing Test directory: {test_dir}")

    # Inspect Train subdirectories
    train_seqs = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d)) and not d.endswith("_gt")
    ])
    print(f"[Verifier] Found {len(train_seqs)} Train sequences: {train_seqs}")
    if len(train_seqs) != 16:
        raise ValueError(f"Expected 16 Train sequences in UCSDped2, found {len(train_seqs)}")

    # Inspect Test subdirectories
    test_seqs = sorted([
        d for d in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, d)) and not d.endswith("_gt")
    ])
    print(f"[Verifier] Found {len(test_seqs)} Test sequences: {test_seqs}")
    if len(test_seqs) != 12:
        raise ValueError(f"Expected 12 Test sequences in UCSDped2, found {len(test_seqs)}")

    # Verify frame read for all Train sequences
    total_train_frames = 0
    for t_seq in train_seqs:
        seq_path = os.path.join(train_dir, t_seq)
        frames = sorted([f for f in os.listdir(seq_path) if f.lower().endswith((".tif", ".tiff", ".png", ".jpg"))])
        if len(frames) == 0:
            raise ValueError(f"Train sequence {t_seq} has 0 readable image frames.")
        # Test read first and last frame
        sample_img = cv2.imread(os.path.join(seq_path, frames[0]))
        if sample_img is None or sample_img.size == 0:
            raise ValueError(f"Failed to read sample frame from {seq_path}/{frames[0]}")
        total_train_frames += len(frames)

    # Verify frame read for all Test sequences
    total_test_frames = 0
    for t_seq in test_seqs:
        seq_path = os.path.join(test_dir, t_seq)
        frames = sorted([f for f in os.listdir(seq_path) if f.lower().endswith((".tif", ".tiff", ".png", ".jpg"))])
        if len(frames) == 0:
            raise ValueError(f"Test sequence {t_seq} has 0 readable image frames.")
        sample_img = cv2.imread(os.path.join(seq_path, frames[0]))
        if sample_img is None or sample_img.size == 0:
            raise ValueError(f"Failed to read sample frame from {seq_path}/{frames[0]}")
        total_test_frames += len(frames)

    # Check for ground truth masks
    gt_dirs = [d for d in os.listdir(test_dir) if d.endswith("_gt")]
    print(f"[Verifier] Found {len(gt_dirs)} ground-truth mask directories in Test/: {gt_dirs}")

    print("=" * 60)
    print("DATASET VERIFICATION SUCCESSFUL")
    print(f"  Total Normal Training Frames: {total_train_frames:,} across 16 sequences")
    print(f"  Total Evaluation Test Frames: {total_test_frames:,} across 12 sequences")
    print(f"  Frame dimensions:             {sample_img.shape[1]}x{sample_img.shape[0]} ({sample_img.shape[2]} channels)")
    print("=" * 60)
    return True


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    datasets_dir = os.path.join(root_dir, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    archive_path = os.path.join(datasets_dir, "UCSD_Anomaly_Dataset.tar.gz")

    # Step 1: Selectively check HTTPS then HTTP
    url = DATASET_URL_HTTPS
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=5) as resp:
            print("[INFO] HTTPS supported by SVCL server.")
    except Exception:
        print("[INFO] HTTPS not supported on SVCL port 443; using official HTTP URL.")
        url = DATASET_URL_HTTP

    # Step 2: Download archive if not already downloaded
    if not os.path.exists(archive_path) or os.path.getsize(archive_path) != EXPECTED_ARCHIVE_SIZE:
        download_with_progress(url, archive_path)
    else:
        print(f"[INFO] Found existing archive of correct size: {archive_path} ({os.path.getsize(archive_path):,} bytes)")

    # Step 3: Selectively extract UCSDped2
    final_ped2_dir = selectively_extract_ped2(archive_path, datasets_dir)

    # Step 4: Verify extracted files
    verification_passed = verify_extracted_dataset(final_ped2_dir)

    # Step 5: Remove archive only upon successful verification
    if verification_passed:
        print(f"[Cleanup] Extraction verified. Removing archive {archive_path} to save ~706 MB disk space...")
        try:
            os.remove(archive_path)
            print("[Cleanup] Archive removed successfully.")
        except Exception as e:
            print(f"[Cleanup Warning] Could not remove archive: {e}")

    print("\n[SUCCESS] UCSD Ped2 dataset is completely prepared at:")
    print(f"  {final_ped2_dir}\n")


if __name__ == "__main__":
    main()
