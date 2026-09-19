from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

FINGERS = {
    "Thumb": (0, 1, 2, 3, 4),
    "Index": (0, 5, 6, 7, 8),
    "Middle": (0, 9, 10, 11, 12),
    "Ring": (0, 13, 14, 15, 16),
    "Pinky": (0, 17, 18, 19, 20),
}


def _point(points: list[dict[str, float]], index: int) -> np.ndarray:
    p = points[index]
    return np.array([p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)], dtype=float)


def _unit(vector: np.ndarray) -> np.ndarray | None:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-8 else None


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    first = _unit(a - b)
    second = _unit(c - b)
    if first is None or second is None:
        return 0.0
    return float(np.arccos(np.clip(np.dot(first, second), -1.0, 1.0)))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _palm_rotation(points: list[dict[str, float]], side: str) -> np.ndarray:
    """Kalidokit-inspired palm frame: wrist, index MCP and pinky MCP."""
    wrist, index, pinky = (_point(points, i) for i in (0, 5, 17))
    forward = _unit(index - wrist)
    across = _unit(pinky - index)
    if forward is None or across is None:
        return np.array([0.0, 0.0, 0.0, 1.0])
    normal = _unit(np.cross(across, forward))
    if normal is None:
        return np.array([0.0, 0.0, 0.0, 1.0])
    forward = _unit(np.cross(normal, across))
    matrix = np.column_stack((across, forward, normal))
    if np.linalg.det(matrix) < 0:
        matrix[:, 2] *= -1
    # The calibration layer can add a model-specific correction later.
    return Rotation.from_matrix(matrix).as_quat()


def solve_hand(points: list[dict[str, float]], side: str, calibration: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Solve a MediaPipe hand into Mixamo-local rotations.

    The method follows Kalidokit's defensible decomposition: a three-point palm
    frame, then one angle for each phalanx from three consecutive landmarks.
    Flexion is clamped; the configured bend axis is deliberately explicit so
    calibration, rather than an undocumented axis assumption, controls a rig.
    """
    if len(points) != 21:
        return {}
    calibration = calibration or {}
    hand_config = calibration.get("hand", {})
    bend_axis = np.asarray(hand_config.get("bendAxis", [1.0, 0.0, 0.0]), dtype=float)
    thumb_axis = np.asarray(hand_config.get("thumbBendAxis", bend_axis), dtype=float)
    prefix = "mixamorig:" + ("Left" if side == "left" else "Right")
    result: dict[str, dict[str, Any]] = {
        prefix + "Hand": {"rotation": _palm_rotation(points, side).tolist(), "quality": 1.0}
    }

    for finger, indices in FINGERS.items():
        axis = thumb_axis if finger == "Thumb" else bend_axis
        for segment in range(3):
            a, b, c = (_point(points, indices[segment + offset]) for offset in (0, 1, 2))
            raw = _angle(a, b, c)
            # MediaPipe's straight finger is close to pi; Mixamo flexion is 0.
            flexion = _clamp(np.pi - raw, 0.0, np.pi)
            # Thumb needs a smaller, more conservative range than finger flexion.
            if finger == "Thumb":
                flexion = _clamp(flexion, 0.0, 1.6)
            quaternion = Rotation.from_rotvec(axis / (np.linalg.norm(axis) + 1e-8) * flexion).as_quat()
            bone = f"{prefix}Hand{finger}{segment + 1}"
            result[bone] = {"rotation": quaternion.tolist(), "quality": 1.0}
    return result
