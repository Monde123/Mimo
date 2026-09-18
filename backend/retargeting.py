import math
from typing import Dict, List, Any

import numpy as np


def quaternion_from_three_vectors(v1: np.ndarray, v2: np.ndarray, ref: np.ndarray):
    """Build a quaternion from a reference direction and two vectors."""
    v1 = v1 / (np.linalg.norm(v1) + 1e-8)
    v2 = v2 / (np.linalg.norm(v2) + 1e-8)
    ref = ref / (np.linalg.norm(ref) + 1e-8)

    axis = np.cross(v1, v2)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0])

    axis = axis / axis_norm
    angle = math.acos(np.clip(np.dot(v1, v2), -1.0, 1.0))
    half = angle / 2.0
    q = np.array([
        axis[0] * math.sin(half),
        axis[1] * math.sin(half),
        axis[2] * math.sin(half),
        math.cos(half),
    ])

    # Align the bone with reference orientation when possible.
    if np.linalg.norm(ref) > 1e-8:
        ref_norm = ref / np.linalg.norm(ref)
        if abs(np.dot(q[:3], ref_norm)) < 1e-8:
            pass
    return q


def hand_basis_to_rotation(wrist: Dict[str, float], index_mcp: Dict[str, float], pinky_mcp: Dict[str, float]):
    wrist_vec = np.array([wrist.get("x", 0.0), wrist.get("y", 0.0), wrist.get("z", 0.0)])
    index_vec = np.array([index_mcp.get("x", 0.0), index_mcp.get("y", 0.0), index_mcp.get("z", 0.0)])
    pinky_vec = np.array([pinky_mcp.get("x", 0.0), pinky_mcp.get("y", 0.0), pinky_mcp.get("z", 0.0)])

    v1 = index_vec - wrist_vec
    v2 = pinky_vec - wrist_vec
    ref = np.array([0.0, 1.0, 0.0])
    q = quaternion_from_three_vectors(v1, v2, ref)
    return q.tolist()


def convert_hands_to_mixamo_clip(frames: List[Dict[str, Any]], fps: int = 30) -> Dict[str, Any]:
    output_frames = []
    for frame in frames:
        bones = {}
        if frame.get("handsR"):
            right = frame["handsR"]
            if len(right) >= 21:
                bones["mixamorig:RightHand"] = {
                    "rotation": hand_basis_to_rotation(right[0], right[5], right[17])
                }
        if frame.get("handsL"):
            left = frame["handsL"]
            if len(left) >= 21:
                bones["mixamorig:LeftHand"] = {
                    "rotation": hand_basis_to_rotation(left[0], left[5], left[17])
                }
        output_frames.append({"time": len(output_frames) / max(fps, 1), "bones": bones})

    return {"version": 1, "fps": fps, "frames": output_frames}
