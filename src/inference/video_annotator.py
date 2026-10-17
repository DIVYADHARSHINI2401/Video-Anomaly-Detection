"""
Surveillance Video Annotation Module using OpenCV.
Overlays a modern security HUD with status badge, risk level, anomaly score bar,
and timestamp on surveillance frames, preserving native aspect ratio where practical.
"""
from typing import Dict, List, Optional, Tuple
import warnings
import cv2
import numpy as np


class SurveillanceVideoAnnotator:
    """
    Renders structured anomaly detection and risk severity overlays on video frames.
    """
    def __init__(self, target_size: Optional[Tuple[int, int]] = None):
        # If target_size is None, preserves native frame dimensions
        self.target_size = target_size
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def annotate_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp_sec: float,
        is_anomaly: bool,
        category: str,
        anomaly_score: float,
        risk_data: Dict[str, object]
    ) -> np.ndarray:
        """
        Draws CCTV HUD overlay on an individual frame.
        """
        if self.target_size is not None:
            canvas = cv2.resize(frame, self.target_size)
        else:
            canvas = frame.copy()

        h, w = canvas.shape[:2]

        severity = str(risk_data.get("severity_level", "Low"))
        composite_risk = float(risk_data.get("composite_risk_score", 0.0))
        badge_bgr = tuple(risk_data.get("badge_color", (0, 200, 0)))

        # 1. Top HUD Banner (semi-transparent dark bar)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, 65), (15, 15, 18), -1)
        cv2.rectangle(overlay, (0, h - 35), (w, h), (15, 15, 18), -1)
        canvas = cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0)

        # 2. Status Badge (top-left)
        if not is_anomaly:
            status_text = "NORMAL ACTIVITY"
            status_color = (60, 200, 80)     # Green
        else:
            status_text = f"ANOMALY: {category.upper()}"
            status_color = (50, 50, 240) if "Unseen" in category else (30, 180, 255)

        cv2.putText(canvas, status_text, (15, 26), self.font, 0.60, status_color, 2, cv2.LINE_AA)

        # Sub-status detail with explicit disclaimer
        cv2.putText(
            canvas,
            f"Rel Dist: {anomaly_score:.3f} | Nov: {risk_data.get('novelty_score', 0.0):.2f} [EXPERIMENTAL - UNVALIDATED]",
            (15, 48),
            self.font,
            0.38,
            (180, 180, 240),
            1,
            cv2.LINE_AA
        )

        # 3. Risk Priority Badge (top-right)
        risk_badge_text = f"RISK: {severity.upper()} ({composite_risk:.2f})"
        (tw, th), _ = cv2.getTextSize(risk_badge_text, self.font, 0.55, 2)
        rx = max(w - tw - 20, 20)
        cv2.rectangle(canvas, (rx - 8, 8), (w - 12, 38), badge_bgr, -1)
        cv2.putText(canvas, risk_badge_text, (rx, 28), self.font, 0.50, (255, 255, 255), 2, cv2.LINE_AA)

        # 4. Anomaly Meter Bar (horizontal bar below top header)
        meter_w = 140
        meter_x = max(w - meter_w - 20, 20)
        meter_y = 48
        meter_h = 10
        cv2.rectangle(canvas, (meter_x, meter_y), (meter_x + meter_w, meter_y + meter_h), (50, 50, 50), -1)
        fill_w = int(meter_w * min(max(anomaly_score, 0.0), 1.0))
        bar_color = (50, 50, 240) if is_anomaly else (60, 200, 80)
        if fill_w > 0:
            cv2.rectangle(canvas, (meter_x, meter_y), (meter_x + fill_w, meter_y + meter_h), bar_color, -1)
        cv2.rectangle(canvas, (meter_x, meter_y), (meter_x + meter_w, meter_y + meter_h), (120, 120, 120), 1)

        # 5. Bottom Info Bar: Timestamp & Frame counter
        mins = int(timestamp_sec // 60)
        secs = int(timestamp_sec % 60)
        millis = int((timestamp_sec - int(timestamp_sec)) * 1000)
        time_str = f"T: {mins:02d}:{secs:02d}.{millis:03d} | Frame: {frame_idx}"
        cv2.putText(canvas, time_str, (15, h - 12), self.font, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

        # Action note
        action_note = str(risk_data.get("recommended_action", ""))
        cv2.putText(canvas, f"Action: {action_note}", (max(w - 360, 200), h - 12), self.font, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

        # Border strobe if High Risk
        if severity == "High" and (frame_idx % 4 < 2):
            cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), (0, 0, 255), 3)

        return canvas

    def render_annotated_video(
        self,
        raw_frames: List[np.ndarray],
        frame_predictions: List[Dict[str, object]],
        output_video_path: str,
        fps: float = 15.0
    ):
        """
        Renders and saves full annotated video to disk using OpenCV.
        """
        if len(raw_frames) == 0:
            raise ValueError("No frames provided for video rendering.")

        first_frame = raw_frames[0]
        if self.target_size is not None:
            out_w, out_h = self.target_size
        else:
            out_h, out_w = first_frame.shape[:2]

        # Generate annotated frames
        annotated_frames_bgr = []
        for i, frame in enumerate(raw_frames):
            pred = frame_predictions[i] if i < len(frame_predictions) else {}
            is_anomaly = bool(pred.get("is_anomaly", False))
            category = str(pred.get("category", "Normal"))
            anomaly_score = float(pred.get("anomaly_score", 0.0))
            risk_data = pred.get("risk_data", {
                "severity_level": "Low",
                "composite_risk_score": 0.0,
                "badge_color": (0, 200, 0),
                "recommended_action": "Normal"
            })
            timestamp_sec = float(pred.get("timestamp_sec", i / fps))

            annotated = self.annotate_frame(
                frame=frame,
                frame_idx=i,
                timestamp_sec=timestamp_sec,
                is_anomaly=is_anomaly,
                category=category,
                anomaly_score=anomaly_score,
                risk_data=risk_data
            )
            annotated_frames_bgr.append(annotated)

        # Attempt high-compatibility H.264 export via imageio (browser-native HTML5 decode)
        h264_success = False
        try:
            import imageio.v2 as imageio
            fps_val = max(float(fps), 1.0)
            writer = imageio.get_writer(
                output_video_path,
                fps=fps_val,
                codec="libx264",
                pixelformat="yuv420p",
                macro_block_size=1
            )
            for f_bgr in annotated_frames_bgr:
                f_rgb = cv2.cvtColor(f_bgr, cv2.COLOR_BGR2RGB)
                writer.append_data(f_rgb)
            writer.close()
            h264_success = True
        except Exception as e:
            warnings.warn(f"imageio H.264 encoding failed ({e}). Falling back to OpenCV VideoWriter.")

        if not h264_success:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video_path, fourcc, max(float(fps), 1.0), (out_w, out_h))
            if not writer.isOpened():
                raise RuntimeError(f"OpenCV failed to open VideoWriter for path: {output_video_path}")
            for f_bgr in annotated_frames_bgr:
                writer.write(f_bgr)
            writer.release()
