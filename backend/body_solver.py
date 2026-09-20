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

# Canonical slots, walked in hierarchy order (parent before child) so each
# bone's parent WORLD rotation is already known when we process it.
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

# Used only when a slot is missing from the model's calibration (e.g. no
# rig_introspector run yet, or the model lacks that bone). This matches the
# common Mixamo/glTF convention (bone-length axis = local Y) confirmed by
# actually reading Clara's rig — a much better default than an arbitrary
# world axis, but still just a fallback: always prefer real calibration.
FALLBACK_LOCAL_FORWARD = np.array([0.0, 1.0, 0.0])

# Mapping used only when calibration doesn't supply a nodeName for a slot.
DEFAULT_NODE_NAMES = {
    "hips": "mixamorig:Hips", "spine": "mixamorig:Spine1", "chest": "mixamorig:Spine2",
    "neck": "mixamorig:Neck", "head": "mixamorig:Head",
    "leftUpperArm": "mixamorig:LeftArm", "leftLowerArm": "mixamorig:LeftForeArm",
    "rightUpperArm": "mixamorig:RightArm", "rightLowerArm": "mixamorig:RightForeArm",
}


def _p(points: list[dict[str, float]], index: int) -> np.ndarray:
    item = points[index]
    return np.array([item.get("x", 0.0), item.get("y", 0.0), item.get("z", 0.0)], dtype=float)


def _unit(vector: np.ndarray) -> np.ndarray | None:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-8 else None


def _orthonormal_pair(forward: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """A deterministic (forward, twist-reference) pair built from a single
    axis. Used identically on the bind side and the observed side so that,
    when the observed direction exactly matches bind, the result is the
    identity rotation — not guaranteed by the previous world-template design.
    """
    fwd = _unit(forward)
    if fwd is None:
        return None
    reference = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(fwd, reference)) > 0.9:
        reference = np.array([0.0, 0.0, 1.0])
    twist = _unit(reference - fwd * np.dot(reference, fwd))
    return fwd, twist


def _observed_world_axes(points: list[dict[str, float]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """World-space (forward, twist-hint) pair per slot, straight from the
    MediaPipe landmarks. 'Forward' is the real signal (direction to the
    child); 'twist-hint' is a rough secondary reference for roll — it was
    already a best-effort heuristic before calibration and stays one here.
    """
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


def solve_body(points: list[dict[str, float]], calibration: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Local rotations for the upper body, using the model's own bind-pose
    calibration (rig_introspector output) instead of a hardcoded axis.

    For each bone: rotate its bind-local forward axis (from calibration)
    onto the observed direction, expressed in the bone's own local frame
    (i.e. relative to its parent's accumulated world rotation) — the
    parent chain is walked in order so this is always available.
    """
    if len(points) < 33:
        return {}

    calibration = calibration or {}
    cal_bones = calibration.get("bones", {})
    observed = _observed_world_axes(points)

    l_hip, r_hip = _p(points, L_HIP), _p(points, R_HIP)
    hip_mid = (l_hip + r_hip) / 2.0

    world_rotations: dict[str, Rotation] = {}
    result: dict[str, dict[str, Any]] = {}

    for slot in BONE_CHAIN:
        bind_forward = np.array(cal_bones.get(slot, {}).get("canonicalForward", FALLBACK_LOCAL_FORWARD))
        bind_pair = _orthonormal_pair(bind_forward)
        if bind_pair is None:
            continue
        bind_fwd_local, bind_twist_local = bind_pair

        obs_fwd_world, obs_twist_world = observed.get(slot, (None, None))
        if obs_fwd_world is None:
            continue

        parent_slot = PARENT.get(slot)
        parent_world = world_rotations.get(parent_slot) if parent_slot else Rotation.identity()
        if parent_slot and parent_world is None:
            continue  # parent failed to resolve; skip this bone this frame

        target_fwd_local = _unit(parent_world.inv().apply(_unit(obs_fwd_world)))
        if target_fwd_local is None:
            continue

        # NOTE: twist/roll is intentionally NOT solved here. Aligning it
        # would need a real bind-pose twist reference from the model (e.g.
        # the bone's local X or Z bind axis), which rig_introspector does
        # not currently export (only the Y-ish forward axis). A synthetic
        # placeholder twist reference was tried and produced a 90 degree
        # error on a neutral pose in testing -- worse than not solving
        # twist at all -- so this is a pure aim/swing rotation (forward
        # axis only) until calibration is extended with a real roll axis.
        local_rotation, _rmsd = Rotation.align_vectors(
            [target_fwd_local],
            [bind_fwd_local],
        )

        world_rotations[slot] = (parent_world * local_rotation) if parent_slot else local_rotation
        node_name = cal_bones.get(slot, {}).get("nodeName", DEFAULT_NODE_NAMES[slot])
        result[node_name] = {"rotation": local_rotation.as_quat().tolist(), "quality": 1.0}

    hips_node = cal_bones.get("hips", {}).get("nodeName", DEFAULT_NODE_NAMES["hips"])
    if hips_node in result:
        result[hips_node]["position"] = hip_mid.tolist()

    return result