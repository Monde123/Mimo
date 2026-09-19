from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUPPORTED_FORMATS = {"mimo.mixamo.animation"}
SUPPORTED_VERSIONS = {1}
REQUIRED_BONE_PREFIX = "mixamorig:"


@dataclass(frozen=True)
class ClipSchema:
    format: str = "mimo.mixamo.animation"
    version: int = 1
    skeleton: str = "mixamo"
    up_axis: str = "Y"
    forward_axis: str = "Z"
    quaternion_order: str = "xyzw"
    rotation_space: str = "local"
    handedness: str = "right"


DEFAULT_CLIP_SCHEMA = ClipSchema()


def clip_metadata() -> dict[str, Any]:
    return {
        "format": DEFAULT_CLIP_SCHEMA.format,
        "version": DEFAULT_CLIP_SCHEMA.version,
        "skeleton": DEFAULT_CLIP_SCHEMA.skeleton,
        "coordinateSystem": {
            "upAxis": DEFAULT_CLIP_SCHEMA.up_axis,
            "forwardAxis": DEFAULT_CLIP_SCHEMA.forward_axis,
            "quaternionOrder": DEFAULT_CLIP_SCHEMA.quaternion_order,
            "rotationSpace": DEFAULT_CLIP_SCHEMA.rotation_space,
            "handedness": DEFAULT_CLIP_SCHEMA.handedness,
        },
    }


def validate_bone_name(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError("Each bone name must be a non-empty string.")
    if not name.startswith(REQUIRED_BONE_PREFIX):
        raise ValueError(f"Bone name '{name}' is not a Mixamo-style bone name.")
    return name


def normalize_rotation_quaternion(rotation: Any, bone_name: str) -> list[float]:
    if not isinstance(rotation, (list, tuple)) or len(rotation) != 4:
        raise ValueError(f"Bone '{bone_name}' must export a quaternion with 4 values.")
    values = [float(v) for v in rotation]
    if any(not (value == value) for value in values):
        raise ValueError(f"Bone '{bone_name}' contains NaN values.")
    return values


def empty_frame(time: float, frame_index: int) -> dict[str, Any]:
    return {
        "time": float(time),
        "frame": int(frame_index),
        "bones": {},
        "quality": {"body": 0.0, "leftHand": 0.0, "rightHand": 0.0},
    }
