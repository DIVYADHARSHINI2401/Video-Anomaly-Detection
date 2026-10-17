"""
Video Frame Sampler and Preprocessor using OpenCV and PyTorch.
Handles video reading, stride sampling, image normalization, and sliding window clipping.
"""
import os
from typing import Generator, List, Optional, Tuple
import cv2
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class VideoFrameSampler:
    """
    Extracts and normalizes frames from video files for ViT feature extraction.
    """
    def __init__(
        self,
        frame_size: Tuple[int, int] = (224, 224),
        sample_stride: int = 5,
        clip_length: int = 16,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        self.frame_size = frame_size
        self.sample_stride = max(1, sample_stride)
        self.clip_length = clip_length
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def preprocess_frame(self, frame_bgr: np.ndarray):
        """
        Preprocesses an individual OpenCV BGR frame.
        Returns:
            resized_bgr: Resized BGR image for OpenCV annotation display
            norm_array: Normalized array/tensor of shape (3, H, W)
        """
        if len(frame_bgr.shape) == 2:
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
        elif frame_bgr.shape[2] == 4:
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGRA2BGR)

        resized_bgr = cv2.resize(frame_bgr, self.frame_size, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        norm_rgb = (rgb - self.mean) / self.std
        chw = norm_rgb.transpose(2, 0, 1)

        if HAS_TORCH:
            norm_tensor = torch.from_numpy(chw).float()
            return resized_bgr, norm_tensor
        return resized_bgr, chw

    def extract_all_sampled_frames(
        self,
        video_path: str,
        fps_override: Optional[float] = None
    ):
        """
        Reads a video file or a directory of sequential image frames (.tif, .png, .jpg)
        and samples frames at regular stride intervals.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video or image directory not found at: {video_path}")

        raw_frames = []
        frame_arrays = []

        # Case 1: Directory of image frames (e.g. UCSD Ped2 sequences)
        if os.path.isdir(video_path):
            valid_exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
            img_files = sorted([
                f for f in os.listdir(video_path)
                if os.path.splitext(f)[1].lower() in valid_exts
            ])
            if len(img_files) == 0:
                raise ValueError(f"No image files ({', '.join(sorted(valid_exts))}) found in directory: {video_path}")

            video_fps = fps_override or 10.0  # UCSD Ped2 native capture rate is 10 FPS
            total_raw_frames = len(img_files)

            for frame_idx, img_file in enumerate(img_files):
                if frame_idx % self.sample_stride == 0:
                    img_path = os.path.join(video_path, img_file)
                    frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    if frame is None:
                        raise ValueError(f"Failed to read image file: {img_path}")
                    resized_bgr, item = self.preprocess_frame(frame)
                    raw_frames.append(resized_bgr)
                    frame_arrays.append(item)

        # Case 2: Standard video container (.mp4, .avi, etc.)
        else:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Failed to open video file: {video_path}")

            video_fps = fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_raw_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % self.sample_stride == 0:
                    resized_bgr, item = self.preprocess_frame(frame)
                    raw_frames.append(resized_bgr)
                    frame_arrays.append(item)

                frame_idx += 1

            cap.release()

        if len(frame_arrays) == 0:
            raise ValueError(f"No frames could be extracted from {video_path}")

        if HAS_TORCH:
            frames_stacked = torch.stack(frame_arrays, dim=0)
        else:
            frames_stacked = np.stack(frame_arrays, axis=0)

        return raw_frames, frames_stacked, video_fps, total_raw_frames

    def generate_sliding_clips(
        self,
        frame_tensors,
        clip_length: Optional[int] = None,
        stride: int = 1
    ):
        """
        Constructs sliding temporal clips of length T from the sequence of frame representations.
        """
        T = clip_length or self.clip_length
        total_frames = frame_tensors.shape[0]

        if total_frames < T:
            pad_needed = T - total_frames
            if HAS_TORCH:
                last_frame = frame_tensors[-1:].repeat(pad_needed, 1, 1, 1)
                frame_tensors = torch.cat([frame_tensors, last_frame], dim=0)
            else:
                last_frame = np.tile(frame_tensors[-1:], (pad_needed, 1, 1, 1))
                frame_tensors = np.concatenate([frame_tensors, last_frame], axis=0)
            total_frames = frame_tensors.shape[0]

        clips = []
        ranges = []
        for i in range(0, total_frames - T + 1, stride):
            clip = frame_tensors[i : i + T]
            clips.append(clip)
            ranges.append((i, i + T))

        if HAS_TORCH:
            stacked_clips = torch.stack(clips, dim=0)
        else:
            stacked_clips = np.stack(clips, axis=0)

        return stacked_clips, ranges
