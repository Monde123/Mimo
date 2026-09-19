from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from backend.pose_estimator import extract_holistic
from backend.smoothing import smooth_landmarks
from backend.retargeting import convert_frames_to_mixamo_clip
from backend.clip_export import export_mixamo_clip
from backend.rig_calibration import load_rig_calibration


def process_video(input_path: Path, output_path: Path, fps: float | None = None, rig_path: Path | None = None) -> Path:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open input video: {input_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    capture.release()
    target_fps = float(fps or source_fps)
    raw_frames = list(extract_holistic(str(input_path)))
    if not raw_frames:
        raise RuntimeError("No usable pose was detected in the video.")
    cleaned = smooth_landmarks(raw_frames, fps=target_fps)
    calibration = load_rig_calibration(rig_path)
    clip = convert_frames_to_mixamo_clip(cleaned, fps=target_fps, calibration=calibration)
    clip["source"] = {"file": input_path.name, "fps": source_fps}
    export_mixamo_clip(clip, str(output_path))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a 2D video into a validated Mixamo animation JSON clip.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--rig", type=Path, default=None, help="Rig calibration JSON")
    args = parser.parse_args()
    result = process_video(args.input, args.output, args.fps, args.rig)
    print(json.dumps({"status": "complete", "output": str(result)}))


if __name__ == "__main__":
    main()
