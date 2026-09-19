from __future__ import annotations

from typing import Any
import numpy as np
from scipy.spatial.transform import Rotation

# Kalidokit-style landmarks-to-local hand rotations, with explicit side validation.
FINGERS = {"Thumb": (0, 1, 2, 3, 4), "Index": (0, 5, 6, 7, 8), "Middle": (0, 9, 10, 11, 12), "Ring": (0, 13, 14, 15, 16), "Pinky": (0, 17, 18, 19, 20)}


def _point(points, index):
    p = points[index]
    return np.asarray([p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)], dtype=float)


def _unit(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-8 else None


def _angle(a, b, c):
    first, second = _unit(a - b), _unit(c - b)
    if first is None or second is None:
        return 0.0
    return float(np.arccos(np.clip(np.dot(first, second), -1.0, 1.0)))


def solve_hand(points: list[dict[str, float]], side: str, calibration: dict[str, Any] | None = None):
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    if len(points) != 21:
        return {}
    calibration = calibration or {}
    hand_cfg = calibration.get("hand", {})
    bend_axis = np.asarray(hand_cfg.get("bendAxis", [1.0, 0.0, 0.0]), dtype=float)
    thumb_axis = np.asarray(hand_cfg.get("thumbBendAxis", bend_axis), dtype=float)
    prefix = "mixamorig:" + ("Left" if side == "left" else "Right")
    wrist, index, pinky = (_point(points, i) for i in (0, 5, 17))
    forward, across = _unit(index - wrist), _unit(pinky - index)
    if forward is None or across is None:
        return {}
    normal = _unit(np.cross(across, forward))
    if normal is None:
        return {}
    forward = _unit(np.cross(normal, across))
    matrix = np.column_stack((across, forward, normal))
    if np.linalg.det(matrix) < 0:
        matrix[:, 2] *= -1
    result = {prefix + "Hand": {"rotation": Rotation.from_matrix(matrix).as_quat().tolist(), "quality": 1.0}}
    for finger, indices in FINGERS.items():
        axis = thumb_axis if finger == "Thumb" else bend_axis
        axis = axis / (np.linalg.norm(axis) + 1e-8)
        for segment in range(3):
            a, b, c = (_point(points, indices[segment + offset]) for offset in (0, 1, 2))
            flexion = float(np.clip(np.pi - _angle(a, b, c), 0.0, 1.6 if finger == "Thumb" else np.pi))
            quaternion = Rotation.from_rotvec(axis * flexion).as_quat().tolist()
            result[f"{prefix}Hand{finger}{segment + 1}"] = {"rotation": quaternion, "quality": 1.0}
    return result
