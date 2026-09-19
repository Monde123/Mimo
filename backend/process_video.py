from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from backend.pose_estimator import extract_holistic
from backend.quality import trim_unstable_sequence
from backend.smoothing import smooth_landmarks
from backend.retargeting import convert_frames_to_mixamo_clip
from backend.clip_export import export_mixamo_clip
from backend.rig_calibration import load_rig_calibration


def process_video(input_path: Path, output_path: Path, fps: float | None = None, rig_path: Path | None = None, max_bad_frames: int = 10) -> Path:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open input video: {input_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    capture.release()
    target_fps = float(fps or source_fps)
    raw_frames = list(extract_holistic(str(input_path)))
    retained, report = trim_unstable_sequence(raw_frames, max_bad_frames=max_bad_frames)
    cleaned = smooth_landmarks(retained, fps=target_fps, max_gap_frames=min(max_bad_frames, 5))
    calibration = load_rig_calibration(rig_path)
    clip = convert_frames_to_mixamo_clip(cleaned, fps=target_fps, calibration=calibration, report=report)
    clip["source"] = {"file": input_path.name, "fps": source_fps}
    export_mixamo_clip(clip, str(output_path))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a 2D video into a quality-scored Mixamo animation JSON clip.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--rig", type=Path, default=None)
    parser.add_argument("--max-bad-frames", type=int, default=10)
    args = parser.parse_args()
    result = process_video(args.input, args.output, args.fps, args.rig, args.max_bad_frames)
    print(json.dumps({"status": "complete", "output": str(result)}))


if __name__ == "__main__":
    main()
