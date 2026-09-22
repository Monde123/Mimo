from __future__ import annotations

from typing import Any

from backend.mixamo.body_solver import solve_body
from backend.mixamo.clip_schema import clip_metadata
from backend.mixamo.hand_solver import solve_hand
from backend.mixamo.rig_calibration import load_rig_calibration
from backend.mixamo.despike import despike_clip


def convert_frames_to_mixamo_clip(frames: list[dict[str, Any]], fps: float = 30.0, calibration: dict[str, Any] | None = None, report: dict[str, Any] | None = None) -> dict[str, Any]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    calibration = calibration or load_rig_calibration()
    output = []
    for index, frame in enumerate(frames):
        bones, world = solve_body(frame.get("bodyPose") or [], calibration, return_world=True)
        # body_solver_final keys `world` by SLOT ("leftLowerArm"), not by
        # mixamorig:* node name -- unlike the old body_solver_patched.
        bones.update(solve_hand(frame.get("handsL") or [], "left", calibration,
                                 parent_world_rotation=world.get("leftLowerArm")))
        bones.update(solve_hand(frame.get("handsR") or [], "right", calibration,
                                 parent_world_rotation=world.get("rightLowerArm")))
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
    clip["frames"], despike_report = despike_clip(clip["frames"])
    clip.setdefault("processing", {})["despike"] = despike_report
    return clip