from __future__ import annotations

from typing import Any

from backend.clip_schema import DEFAULT_CLIP_SCHEMA, normalize_rotation_quaternion, validate_bone_name


def validate_clip(clip: dict[str, Any]) -> None:
    if not isinstance(clip, dict):
        raise ValueError("Clip must be a dictionary.")

    if clip.get("format") != DEFAULT_CLIP_SCHEMA.format:
        raise ValueError(f"Unsupported clip format: {clip.get('format')!r}")
    if int(clip.get("version", 0)) != DEFAULT_CLIP_SCHEMA.version:
        raise ValueError(f"Unsupported clip version: {clip.get('version')!r}")
    if not isinstance(clip.get("fps"), (int, float)) or float(clip["fps"]) <= 0:
        raise ValueError("Clip fps must be a positive number.")
    if not isinstance(clip.get("frames"), list) or not clip["frames"]:
        raise ValueError("Clip must contain at least one animation frame.")

    coordinate_system = clip.get("coordinateSystem", {})
    if not isinstance(coordinate_system, dict):
        raise ValueError("coordinateSystem must be an object.")

    for frame in clip["frames"]:
        if not isinstance(frame, dict):
            raise ValueError("Each frame must be an object.")
        if "time" not in frame or "bones" not in frame:
            raise ValueError("Each frame requires 'time' and 'bones'.")
        if not isinstance(frame["bones"], dict):
            raise ValueError("Each frame 'bones' must be an object.")

        for bone_name, payload in frame["bones"].items():
            validate_bone_name(bone_name)
            if not isinstance(payload, dict) or "rotation" not in payload:
                raise ValueError(f"Bone '{bone_name}' must contain a rotation payload.")
            normalize_rotation_quaternion(payload["rotation"], bone_name)

            # quality is optional but if supplied it should be numeric
            if "quality" in payload and not isinstance(payload["quality"], (int, float)):
                raise ValueError(f"Bone '{bone_name}' quality must be numeric if supplied.")

    return None
