from __future__ import annotations

from typing import Any
import numpy as np
from scipy.spatial.transform import Rotation

# MediaPipe hand indices: wrist, thumb(1..4), index(5..8), middle(9..12), ring(13..16), pinky(17..20).
FINGERS = {
    "Thumb": (0, 1, 2, 3, 4),
    "Index": (0, 5, 6, 7, 8),
    "Middle": (0, 9, 10, 11, 12),
    "Ring": (0, 13, 14, 15, 16),
    "Pinky": (0, 17, 18, 19, 20),
}


def _safe(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else np.array([0.0, 1.0, 0.0])


def _quat_from_to(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return [x,y,z,w] shortest-arc quaternion from source to target."""
    a, b = _safe(source), _safe(target)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 1.0 - 1e-7:
        return np.array([0.0, 0.0, 0.0, 1.0])
    if dot < -1.0 + 1e-7:
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-7:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis = _safe(axis)
        return np.r_[axis, 0.0]
    axis = np.cross(a, b)
    q = np.r_[axis, 1.0 + dot]
    return q / np.linalg.norm(q)


def _hand_rotation(points: list[dict[str, float]], side: str) -> list[float]:
    wrist, index_mcp, pinky_mcp = (_point(points, i) for i in (0, 5, 17))
    across = _safe(pinky_mcp - index_mcp)
    forward = _safe(np.cross(across, _safe(index_mcp - wrist)))
    up = _safe(index_mcp - wrist)
    # Matrix columns are the canonical hand basis transformed into world basis.
    matrix = np.column_stack((across, up, forward))
    if side == "left":
        matrix[:, 0] *= -1
    return Rotation.from_matrix(matrix).as_quat().tolist()


def _point(points: list[dict[str, float]], index: int) -> np.ndarray:
    p = points[index]
    return np.array([p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)], dtype=float)


def _finger_rotations(points: list[dict[str, float]], side: str) -> dict[str, dict[str, list[float]]]:
    result = {}
    prefix = "mixamorig:" + ("Left" if side == "left" else "Right")
    for finger, indices in FINGERS.items():
        for segment in range(3):
            a, b = _point(points, indices[segment]), _point(points, indices[segment + 1])
            name = f"{prefix}Hand{finger}{segment + 1}"
            # Mixamo finger bones point along their local +Y axis in the canonical rest pose.
            result[name] = {"rotation": _quat_from_to(np.array([0.0, 1.0, 0.0]), b - a).tolist()}
    return result


def _body_rotations(frame: dict[str, Any]) -> dict[str, dict[str, list[float]]]:
    points = frame.get("bodyPose") or []
    if len(points) < 33:
        return {}
    # Body rotations use the same parent-local direction concept as Mimic's rotation_solver.
    mapping = {
        "mixamorig:LeftArm": (11, 13), "mixamorig:LeftForeArm": (13, 15), "mixamorig:LeftHand": (15, 19),
        "mixamorig:RightArm": (12, 14), "mixamorig:RightForeArm": (14, 16), "mixamorig:RightHand": (16, 20),
    }
    result = {}
    for bone, (a, b) in mapping.items():
        result[bone] = {"rotation": _quat_from_to(np.array([1.0, 0.0, 0.0]), _point(points, b) - _point(points, a)).tolist()}
    return result


def convert_frames_to_mixamo_clip(frames: list[dict[str, Any]], fps: float = 30.0) -> dict[str, Any]:
    output = []
    for index, frame in enumerate(frames):
        bones = _body_rotations(frame)
        for side in ("left", "right"):
            points = frame.get("handsL" if side == "left" else "handsR") or []
            if len(points) == 21:
                prefix = "mixamorig:" + ("Left" if side == "left" else "Right")
                bones[prefix + "Hand"] = {"rotation": _hand_rotation(points, side)}
                bones.update(_finger_rotations(points, side))
        output.append({"time": index / fps, "bones": bones})
    return {
        "format": "mimo.mixamo.animation",
        "version": 1,
        "skeleton": "mixamo",
        "coordinateSystem": "MediaPipe world coordinates; quaternion local tracks",
        "fps": fps,
        "duration": len(output) / fps,
        "frames": output,
    }


def convert_hands_to_mixamo_clip(frames: list[dict[str, Any]], fps: float = 30.0) -> dict[str, Any]:
    return convert_frames_to_mixamo_clip(frames, fps)
