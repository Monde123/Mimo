from __future__ import annotations

from typing import Any

from backend.fused_extractor import extract_pose_and_hands


def merge_pose_and_hands(pose_frames: list[dict[str, Any]], hand_frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge independent extractor outputs by source frame number."""
    hands_by_frame = {int(frame["frame"]): frame for frame in hand_frames}
    merged = []
    for pose in pose_frames:
        hand = hands_by_frame.get(int(pose["frame"]), {})
        merged.append({
            **pose,
            "bodyPose": pose.get("predictions", []),
            "handsL": hand.get("handsL", []),
            "handsR": hand.get("handsR", []),
        })
    return merged


def extract_precise_frames(video_path: str):
    """Compatibility entrypoint for the one-pass precise extractor."""
    return extract_pose_and_hands(video_path)
