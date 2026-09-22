from __future__ import annotations

import math
from typing import Any

from backend.mixamo.clip_schema import SUPPORTED_FORMAT, SUPPORTED_VERSION


def _quaternion(values: Any, bone: str) -> list[float]:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError(f"{bone}: quaternion must contain four values")
    q = [float(value) for value in values]
    if not all(math.isfinite(value) for value in q):
        raise ValueError(f"{bone}: quaternion contains a non-finite value")
    norm = math.sqrt(sum(value * value for value in q))
    if not 0.999 <= norm <= 1.001:
        raise ValueError(f"{bone}: quaternion is not normalized (norm={norm:.6f})")
    return q


def validate_clip(clip: dict[str, Any]) -> None:
    if clip.get("format") != SUPPORTED_FORMAT or clip.get("version") != SUPPORTED_VERSION:
        raise ValueError("Unsupported Mimo clip format or version")
    fps = float(clip.get("fps", 0))
    if not math.isfinite(fps) or fps <= 0 or fps > 240:
        raise ValueError("Clip FPS must be between 0 and 240")
    frames = clip.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Clip must contain frames")

    previous_time = -1.0
    for expected_index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError("Every frame must be an object")
        time = float(frame.get("time", -1))
        if not math.isfinite(time) or time <= previous_time:
            raise ValueError("Frame times must be finite and strictly increasing")
        previous_time = time
        if frame.get("frame", expected_index) != expected_index:
            raise ValueError("Frame indices must be contiguous")
        bones = frame.get("bones", {})
        if not isinstance(bones, dict):
            raise ValueError("Frame bones must be an object")
        for bone, payload in bones.items():
            if not bone.startswith("mixamorig:"):
                raise ValueError(f"Unsupported bone name: {bone}")
            if not isinstance(payload, dict):
                raise ValueError(f"{bone}: payload must be an object")
            payload["rotation"] = _quaternion(payload.get("rotation"), bone)
            if "quality" in payload:
                quality = float(payload["quality"])
                if not 0 <= quality <= 1:
                    raise ValueError(f"{bone}: quality must be between 0 and 1")
