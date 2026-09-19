from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

NOSE = 0
L_EAR, R_EAR = 7, 8
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_INDEX, R_INDEX = 19, 20
L_HIP, R_HIP = 23, 24

BONE_CHAIN = [
    "mixamorig:Hips", "mixamorig:Spine", "mixamorig:Neck", "mixamorig:Head",
    "mixamorig:LeftArm", "mixamorig:LeftForeArm",
    "mixamorig:RightArm", "mixamorig:RightForeArm",
]
PARENT = {
    "mixamorig:Spine": "mixamorig:Hips", "mixamorig:Neck": "mixamorig:Spine",
    "mixamorig:Head": "mixamorig:Neck", "mixamorig:LeftArm": "mixamorig:Spine",
    "mixamorig:LeftForeArm": "mixamorig:LeftArm", "mixamorig:RightArm": "mixamorig:Spine",
    "mixamorig:RightForeArm": "mixamorig:RightArm",
}


def _p(points, index):
    item = points[index]
    return np.asarray([item.get("x", 0.0), item.get("y", 0.0), item.get("z", 0.0)], dtype=float)


def _unit(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-8 else None


def _frame_rotation(canonical_fwd, canonical_up, observed_fwd, observed_up):
    fwd = _unit(observed_fwd)
    if fwd is None:
        return None
    up = _unit(observed_up - fwd * np.dot(observed_up, fwd))
    if up is None:
        return None
    right = _unit(np.cross(up, fwd))
    if right is None:
        return None
    c_fwd = _unit(canonical_fwd)
    c_up = _unit(canonical_up - c_fwd * np.dot(canonical_up, c_fwd))
    if c_fwd is None or c_up is None:
        return None
    c_right = _unit(np.cross(c_up, c_fwd))
    if c_right is None:
        return None
    observed_basis = np.column_stack((right, up, fwd))
    canonical_basis = np.column_stack((c_right, c_up, c_fwd))
    matrix = observed_basis @ canonical_basis.T
    if np.linalg.det(matrix) < 0:
        matrix[:, 2] *= -1
    return Rotation.from_matrix(matrix)


def _world_rotations(points):
    world = {}
    l_shoulder, r_shoulder = _p(points, L_SHOULDER), _p(points, R_SHOULDER)
    l_hip, r_hip = _p(points, L_HIP), _p(points, R_HIP)
    hip_mid = (l_hip + r_hip) / 2.0
    shoulder_mid = (l_shoulder + r_shoulder) / 2.0
    torso_up = shoulder_mid - hip_mid
    torso_side = (r_hip - l_hip) + (r_shoulder - l_shoulder)
    torso_fwd = np.cross(torso_up, torso_side)
    canonical_fwd = np.array([0.0, 0.0, 1.0])
    canonical_up = np.array([0.0, 1.0, 0.0])

    hips = _frame_rotation(canonical_fwd, canonical_up, torso_fwd, torso_up)
    if hips is not None:
        world["mixamorig:Hips"] = hips
        world["mixamorig:Spine"] = hips

    head_up = _p(points, NOSE) - shoulder_mid
    neck = _frame_rotation(canonical_fwd, canonical_up, torso_fwd, head_up)
    if neck is not None:
        world["mixamorig:Neck"] = neck
    ear_span = _p(points, R_EAR) - _p(points, L_EAR)
    head = _frame_rotation(canonical_fwd, canonical_up, np.cross(head_up, ear_span), head_up)
    if head is not None:
        world["mixamorig:Head"] = head

    for side, shoulder_i, elbow_i, wrist_i, index_i in (
        ("Left", L_SHOULDER, L_ELBOW, L_WRIST, L_INDEX),
        ("Right", R_SHOULDER, R_ELBOW, R_WRIST, R_INDEX),
    ):
        shoulder, elbow, wrist, index = (_p(points, i) for i in (shoulder_i, elbow_i, wrist_i, index_i))
        side_axis = np.array([1.0, 0.0, 0.0]) if side == "Left" else np.array([-1.0, 0.0, 0.0])
        arm = _frame_rotation(side_axis, canonical_up, elbow - shoulder, wrist - elbow)
        forearm = _frame_rotation(side_axis, canonical_up, wrist - elbow, index - wrist)
        if arm is not None:
            world[f"mixamorig:{side}Arm"] = arm
        if forearm is not None:
            world[f"mixamorig:{side}ForeArm"] = forearm
    return world, hip_mid


def solve_body(points: list[dict[str, float]], calibration: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    if len(points) < 33:
        return {}
    world, hip_mid = _world_rotations(points)
    result = {}
    for bone in BONE_CHAIN:
        current = world.get(bone)
        if current is None:
            continue
        parent = PARENT.get(bone)
        parent_world = world.get(parent) if parent else None
        local = parent_world.inv() * current if parent_world is not None else current
        result[bone] = {"rotation": local.as_quat().tolist(), "quality": 1.0}
    if "mixamorig:Hips" in result:
        result["mixamorig:Hips"]["position"] = hip_mid.tolist()
    return result
