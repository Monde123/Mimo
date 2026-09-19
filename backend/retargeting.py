from __future__ import annotations

from typing import Any

from backend.body_solver import solve_body
from backend.clip_schema import clip_metadata
from backend.hand_solver import solve_hand
from backend.rig_calibration import load_rig_calibration


def convert_frames_to_mixamo_clip(frames: list[dict[str, Any]], fps: float = 30.0, calibration: dict[str, Any] | None = None) -> dict[str, Any]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    calibration = calibration or load_rig_calibration()
    output = []
    for index, frame in enumerate(frames):
        bones = solve_body(frame.get("bodyPose") or [], calibration)
        left = solve_hand(frame.get("handsL") or [], "left", calibration)
        right = solve_hand(frame.get("handsR") or [], "right", calibration)
        bones.update(left)
        bones.update(right)
        output.append({
            "frame": index,
            "time": index / float(fps),
            "bones": bones,
            "quality": {
                "body": 1.0 if frame.get("bodyPose") else 0.0,
                "leftHand": 1.0 if len(frame.get("handsL") or []) == 21 else 0.0,
                "rightHand": 1.0 if len(frame.get("handsR") or []) == 21 else 0.0,
            },
        })
    clip = clip_metadata()
    clip.update({"fps": float(fps), "duration": len(output) / float(fps), "frames": output})
    return clip


def convert_hands_to_mixamo_clip(frames: list[dict[str, Any]], fps: float = 30.0) -> dict[str, Any]:
    return convert_frames_to_mixamo_clip(frames, fps)
