from __future__ import annotations

from typing import Any

from backend.body_solver import solve_body
from backend.clip_schema import clip_metadata
from backend.hand_solver import solve_hand
from backend.rig_calibration import load_rig_calibration


def convert_frames_to_mixamo_clip(frames: list[dict[str, Any]], fps: float = 30.0, calibration: dict[str, Any] | None = None, report: dict[str, Any] | None = None) -> dict[str, Any]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    calibration = calibration or load_rig_calibration()
    output = []
    for index, frame in enumerate(frames):
        bones = solve_body(frame.get("bodyPose") or [], calibration)
        bones.update(solve_hand(frame.get("handsL") or [], "left", calibration))
        bones.update(solve_hand(frame.get("handsR") or [], "right", calibration))
        quality = frame.get("detectionQuality", {})
        output.append({
            "frame": index,
            "time": index / float(fps),
            "bones": bones,
            "quality": {
                "body": float(quality.get("body", 0.0)),
                "leftHand": float(quality.get("leftHand", 0.0)),
                "rightHand": float(quality.get("rightHand", 0.0)),
                "overall": float(quality.get("overall", 0.0)),
                "interpolated": list(frame.get("interpolated", [])),
            },
        })
    clip = clip_metadata()
    clip.update({"fps": float(fps), "duration": len(output) / float(fps), "frames": output})
    if report is not None:
        clip["processing"] = report
    return clip
