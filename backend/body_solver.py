from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

BODY_BONES = {
    "mixamorig:LeftArm": (11, 13), "mixamorig:LeftForeArm": (13, 15),
    "mixamorig:RightArm": (12, 14), "mixamorig:RightForeArm": (14, 16),
}


def _p(points: list[dict[str, float]], index: int) -> np.ndarray:
    item = points[index]
    return np.array([item.get("x", 0), item.get("y", 0), item.get("z", 0)], dtype=float)


def _direction(points: list[dict[str, float]], pair: tuple[int, int]) -> np.ndarray | None:
    vector = _p(points, pair[1]) - _p(points, pair[0])
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-8 else None


def _swing(source: np.ndarray, target: np.ndarray) -> Rotation:
    """Shortest swing from a canonical bone axis to an observed segment."""
    source = source / (np.linalg.norm(source) + 1e-8)
    target = target / (np.linalg.norm(target) + 1e-8)
    dot = float(np.clip(np.dot(source, target), -1, 1))
    if dot > 1 - 1e-7:
        return Rotation.identity()
    if dot < -1 + 1e-7:
        axis = np.cross(source, np.array([0.0, 1.0, 0.0]))
        if np.linalg.norm(axis) < 1e-7:
            axis = np.cross(source, np.array([1.0, 0.0, 0.0]))
        return Rotation.from_rotvec(axis / np.linalg.norm(axis) * np.pi)
    axis = np.cross(source, target)
    return Rotation.from_rotvec(axis / np.linalg.norm(axis) * np.arccos(dot))


def solve_body(points: list[dict[str, float]], calibration: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Solve upper-body swing rotations in a documented canonical frame.

    Output is parent-local only after `to_local_tracks` is applied in a full
    sequence; this function exposes measured world swings per frame.
    """
    if len(points) < 33:
        return {}
    result = {}
    for bone, pair in BODY_BONES.items():
        direction = _direction(points, pair)
        if direction is None:
            continue
        canonical = np.array([1.0, 0.0, 0.0]) if "Left" in bone else np.array([-1.0, 0.0, 0.0])
        result[bone] = {"rotation": _swing(canonical, direction).as_quat().tolist(), "quality": 1.0}
    return result
