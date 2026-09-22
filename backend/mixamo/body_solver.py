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
    "hips", "spine", "chest", "neck", "head",
    "leftUpperArm", "leftLowerArm",
    "rightUpperArm", "rightLowerArm",
]

PARENT = {
    "spine": "hips", "chest": "spine", "neck": "chest", "head": "neck",
    "leftUpperArm": "chest", "leftLowerArm": "leftUpperArm",
    "rightUpperArm": "chest", "rightLowerArm": "rightUpperArm",
}

DEFAULT_NODE_NAMES = {
    "hips": "mixamorig:Hips", "spine": "mixamorig:Spine1", "chest": "mixamorig:Spine2",
    "neck": "mixamorig:Neck", "head": "mixamorig:Head",
    "leftUpperArm": "mixamorig:LeftArm", "leftLowerArm": "mixamorig:LeftForeArm",
    "rightUpperArm": "mixamorig:RightArm", "rightLowerArm": "mixamorig:RightForeArm",
}

# Universal Mixamo/glTF authoring convention, confirmed empirically on
# Clara's rig (mixamo.json: every bone's canonicalForward ~= [0,1,0]):
# in a bone's OWN local frame (before its own bind rotation is applied),
# "toward the child" is the local Y axis. The roll-reference axis (local
# X) is arbitrary in absolute terms but that's fine -- it only needs to be
# used consistently on the bind side and the observed side, which it now
# is, both being derived from the SAME real bindLocalRotation quaternion.
FWD_LOCAL = np.array([0.0, 1.0, 0.0])
ROLL_REF_LOCAL = np.array([1.0, 0.0, 0.0])

# Bones where the observed secondary landmark direction (index-finger side
# of the forearm/upper arm) plausibly corresponds to a real roll axis on
# the model. For the torso/head chain, the available "twist hint" (a
# cross-product of shoulder/hip vectors, or ear span) has NO verified
# correspondence to ROLL_REF_LOCAL -- combining them produced a spurious
# 90 degree error on a perfectly neutral pose in testing. Swing-only for
# those until a genuinely corresponding reference is calibrated.
ROLL_CAPABLE_SLOTS = {"leftUpperArm", "leftLowerArm", "rightUpperArm", "rightLowerArm"}


def _p(points: list[dict[str, float]], index: int) -> np.ndarray:
    item = points[index]
    return np.array([item.get("x", 0.0), item.get("y", 0.0), item.get("z", 0.0)], dtype=float)


def _unit(vector: np.ndarray) -> np.ndarray | None:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-8 else None


def _project_perp(vector: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Component of vector perpendicular to axis -- the 3D generalization
    of Gizmo.calc_roll's `local_point.x = 0`."""
    return vector - axis * np.dot(vector, axis)


def _swing_and_roll(bind_fwd_world: np.ndarray, bind_roll_ref_world: np.ndarray,
                     target_fwd_world: np.ndarray, target_roll_hint_world: np.ndarray,
                     allow_roll: bool) -> Rotation | None:
    """Gizmo.calc_rotation_matrix (swing) + Gizmo.calc_roll (residual twist
    around the new axis), combined into a single delta rotation.

    allow_roll=False skips the roll step entirely (swing only) -- for bones
    where the "roll hint" landmark direction has no verified correspondence
    to ROLL_REF_LOCAL, computing a "roll" from it is not a small error, it
    is a random rotation around the bone's own axis (see ROLL_CAPABLE_SLOTS).
    """
    bind_fwd = _unit(bind_fwd_world)
    target_fwd = _unit(target_fwd_world)
    if bind_fwd is None or target_fwd is None:
        return None

    swing, _ = Rotation.align_vectors([target_fwd], [bind_fwd])
    if not allow_roll:
        return swing

    predicted_roll_ref = swing.apply(bind_roll_ref_world)
    predicted_perp = _unit(_project_perp(predicted_roll_ref, target_fwd))
    observed_perp = _unit(_project_perp(target_roll_hint_world, target_fwd))
    if predicted_perp is None or observed_perp is None:
        return swing  # no usable roll signal this frame; swing only

    cos_angle = np.clip(np.dot(predicted_perp, observed_perp), -1.0, 1.0)
    sin_sign = np.dot(np.cross(predicted_perp, observed_perp), target_fwd)
    roll_angle = np.arctan2(sin_sign, cos_angle)
    roll = Rotation.from_rotvec(target_fwd * roll_angle)

    return roll * swing


def _observed_world_axes(points: list[dict[str, float]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    axes: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    l_shoulder, r_shoulder = _p(points, L_SHOULDER), _p(points, R_SHOULDER)
    l_hip, r_hip = _p(points, L_HIP), _p(points, R_HIP)
    hip_mid = (l_hip + r_hip) / 2.0
    shoulder_mid = (l_shoulder + r_shoulder) / 2.0
    nose = _p(points, NOSE)

    torso_up = shoulder_mid - hip_mid
    torso_side = (r_hip - l_hip) + (r_shoulder - l_shoulder)
    torso_twist_hint = np.cross(torso_up, torso_side)

    axes["hips"] = (torso_up, torso_twist_hint)
    axes["spine"] = (torso_up, torso_twist_hint)
    axes["chest"] = (torso_up, torso_twist_hint)

    head_dir = nose - shoulder_mid
    ear_span = _p(points, R_EAR) - _p(points, L_EAR)
    axes["neck"] = (head_dir, ear_span)
    axes["head"] = (head_dir, ear_span)

    for side, sh_i, el_i, wr_i, idx_i in (
        ("left", L_SHOULDER, L_ELBOW, L_WRIST, L_INDEX),
        ("right", R_SHOULDER, R_ELBOW, R_WRIST, R_INDEX),
    ):
        shoulder, elbow, wrist, index = (_p(points, i) for i in (sh_i, el_i, wr_i, idx_i))
        axes[f"{side}UpperArm"] = (elbow - shoulder, wrist - elbow)
        axes[f"{side}LowerArm"] = (wrist - elbow, index - wrist)

    return axes


def solve_body(points: list[dict[str, float]], calibration: dict[str, Any] | None = None,
               return_world: bool = False):
    """Local rotations for the upper body: swing (aim) + roll (twist),
    grounded in the model's REAL bind-pose local rotation per bone
    (bindLocalRotation from rig_introspector), not a single derived
    direction -- which is what makes the roll calculation trustworthy
    this time (bind forward and bind roll-reference come from the same
    real rotation, so they can't disagree the way the old synthetic
    twist reference did).
    """
    if len(points) < 33:
        return ({}, {}) if return_world else {}

    calibration = calibration or {}
    cal_bones = calibration.get("bones", {})
    observed = _observed_world_axes(points)

    l_hip, r_hip = _p(points, L_HIP), _p(points, R_HIP)
    hip_mid = (l_hip + r_hip) / 2.0

    world_rotations: dict[str, Rotation] = {}
    result: dict[str, dict[str, Any]] = {}

    for slot in BONE_CHAIN:
        bind_quat = cal_bones.get(slot, {}).get("bindLocalRotation", [0.0, 0.0, 0.0, 1.0])
        bind_local_rotation = Rotation.from_quat(bind_quat)

        obs_fwd_world, obs_roll_hint_world = observed.get(slot, (None, None))
        if obs_fwd_world is None:
            continue

        parent_slot = PARENT.get(slot)
        parent_world = world_rotations.get(parent_slot) if parent_slot else Rotation.identity()
        if parent_slot and parent_world is None:
            continue

        # "Where would this bone's axes be right now if it hadn't moved on
        # its own, only its parent did" -- the reference to swing/roll from.
        bone_bind_world_now = parent_world * bind_local_rotation
        bind_fwd_world = bone_bind_world_now.apply(FWD_LOCAL)
        bind_roll_ref_world = bone_bind_world_now.apply(ROLL_REF_LOCAL)

        delta = _swing_and_roll(bind_fwd_world, bind_roll_ref_world, obs_fwd_world, obs_roll_hint_world,
                                 allow_roll=slot in ROLL_CAPABLE_SLOTS)
        if delta is None:
            continue

        bone_world_final = delta * bone_bind_world_now
        local_rotation = parent_world.inv() * bone_world_final if parent_slot else bone_world_final

        world_rotations[slot] = bone_world_final
        node_name = cal_bones.get(slot, {}).get("nodeName", DEFAULT_NODE_NAMES[slot])
        result[node_name] = {"rotation": local_rotation.as_quat().tolist(), "quality": 1.0}

    hips_node = cal_bones.get("hips", {}).get("nodeName", DEFAULT_NODE_NAMES["hips"])
    if hips_node in result:
        result[hips_node]["position"] = hip_mid.tolist()

    return (result, world_rotations) if return_world else result