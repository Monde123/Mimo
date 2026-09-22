"""Extract MediaPipe landmarks with any of the three pipelines and export
straight to BVH -- deliberately bypassing body_solver/hand_solver/Mixamo
retargeting entirely, so MediaPipe's own output can be inspected on its own.

Usage:
    python export_raw_bvh.py video.mp4 out_prefix --mode holistic
    python export_raw_bvh.py video.mp4 out_prefix --mode precise
    python export_raw_bvh.py video.mp4 out_prefix --mode hybrid

Writes out_prefix_body.bvh, out_prefix_hand_left.bvh, out_prefix_hand_right.bvh.
"""
from __future__ import annotations

import argparse
import json

from backend.pose_estimator import extract_holistic
from backend.precise_extraction import extract_precise_frames
from backend.hybrid_extractor import extract_holistic_body_precise_hands
from mediapipe_to_bvh import export_body_bvh, export_hand_bvh

EXTRACTORS = {
    "holistic": extract_holistic,
    "precise": extract_precise_frames,
    "hybrid": extract_holistic_body_precise_hands,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Path to the input video")
    parser.add_argument("out_prefix", help="Prefix for the three output .bvh files")
    parser.add_argument("--mode", choices=list(EXTRACTORS), required=True)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--save-raw-json", default=None,
                         help="Optionally also dump the raw extracted frames to this JSON path")
    args = parser.parse_args()

    import cv2
    capture = cv2.VideoCapture(str(args.video))
    fps = args.fps or float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    capture.release()

    frames = list(EXTRACTORS[args.mode](str(args.video)))

    if args.save_raw_json:
        with open(args.save_raw_json, "w", encoding="utf-8") as handle:
            json.dump(frames, handle, indent=2)

    report = {
        "mode": args.mode,
        "fps": fps,
        "framesExtracted": len(frames),
        "body": export_body_bvh(frames, f"{args.out_prefix}_body.bvh", fps=fps),
        "handLeft": export_hand_bvh(frames, "left", f"{args.out_prefix}_hand_left.bvh", fps=fps),
        "handRight": export_hand_bvh(frames, "right", f"{args.out_prefix}_hand_right.bvh", fps=fps),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
