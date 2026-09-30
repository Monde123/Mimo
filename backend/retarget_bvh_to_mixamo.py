"""
Retargeting d'un BVH (export_mediapipe_bvh, mode "rotations") vers un modèle Mixamo.

Principe : celui documenté par UniVRM pour son ControlRig (et repris par Rokoko
Retargeting dans Blender) — la conversion de pose entre deux squelettes de T-pose
différente passe par un format pivot indépendant du repos :

    PoseForA -> NormalizedLocalRotation -> PoseForB

    NormalizedLocalRotation = W_A · L_A⁻¹ · A.local · W_A⁻¹
    B.local                 = L_B · W_B⁻¹ · NormalizedLocalRotation · W_B

où L = rotation locale de repos, W = rotation monde de repos.

Simplification qui s'applique ici : notre BVH est construit avec un repos
identité par construction (les tuples de _SKELETON sont des directions
d'OFFSET, pas des rotations — à valeur de canal nulle, chaque os pointe déjà
le long de son offset). Donc L_A = W_A = identité, et la première étape
disparaît : le canal de rotation du BVH EST déjà NormalizedLocalRotation.
Il ne reste que la deuxième étape, avec (L_B, W_B) lus dans la pose de repos
du modèle Mixamo cible.

Usage:
    python -m backend.retarget_bvh_to_mixamo capture.bvh Clara.glb sortie.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


# --------------------------------------------------------------------------
# 1. Lecture du BVH
# --------------------------------------------------------------------------
def parse_bvh(path: str | Path) -> tuple[list[dict[str, Any]], float, np.ndarray]:
    """
    Parseur BVH minimal mais générique (respecte l'ordre des canaux déclaré,
    ne suppose pas Zrotation/Xrotation/Yrotation comme le fait le générateur
    par défaut — un BVH d'une autre origine peut utiliser un autre ordre).
    Retourne (joints, frame_time, motion) où motion est un tableau (T, total_canaux).
    """
    lines = [l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    joints: list[dict[str, Any]] = []
    stack: list[str] = []
    i = 0

    def parse_block() -> None:
        nonlocal i
        kind, name = lines[i].split(None, 1)
        i += 1
        assert lines[i] == "{"
        i += 1
        assert lines[i].startswith("OFFSET")
        i += 1
        parts = lines[i].split()
        assert parts[0] == "CHANNELS"
        n = int(parts[1])
        channels = parts[2 : 2 + n]
        i += 1
        joints.append({"name": name, "parent": stack[-1] if stack else None, "channels": channels})
        stack.append(name)
        while lines[i] != "}":
            if lines[i].startswith("JOINT"):
                parse_block()
            elif lines[i].startswith("End Site"):
                i += 1  # '{'
                while lines[i] != "}":
                    i += 1
                i += 1  # '}' de fermeture d'End Site
            else:
                i += 1
        i += 1  # '}' de fermeture du joint
        stack.pop()

    assert lines[i] == "HIERARCHY"
    i += 1
    parse_block()  # ROOT
    assert lines[i] == "MOTION"
    i += 1
    n_frames = int(lines[i].split(":")[1])
    i += 1
    frame_time = float(lines[i].split(":")[1])
    i += 1
    motion = np.array([[float(x) for x in lines[i + k].split()] for k in range(n_frames)])
    return joints, frame_time, motion


def _euler_bvh_to_matrix(order: list[str], angles_deg: list[float]) -> np.ndarray:
    """R = R_order[0](a0) @ R_order[1](a1) @ R_order[2](a2), ordre = celui déclaré par le fichier."""
    r = np.eye(3)
    for axis, ang in zip(order, angles_deg):
        r = r @ Rotation.from_euler(axis.lower(), ang, degrees=True).as_matrix()
    return r


def extract_bvh_rotations(
    joints: list[dict[str, Any]], motion: np.ndarray
) -> dict[str, list[Rotation]]:
    """Reconstruit, pour chaque joint, la liste des rotations locales (une par frame)."""
    col = 0
    col_of: dict[str, tuple[int, list[str]]] = {}
    for j in joints:
        col_of[j["name"]] = (col, j["channels"])
        col += len(j["channels"])

    out: dict[str, list[Rotation]] = {}
    for j in joints:
        start, channels = col_of[j["name"]]
        rot_order = [c[0] for c in channels if c.endswith("rotation")]
        rot_cols = [start + k for k, c in enumerate(channels) if c.endswith("rotation")]
        if not rot_order:
            continue
        rots = []
        for frame in motion:
            angles = [frame[c] for c in rot_cols]
            rots.append(Rotation.from_matrix(_euler_bvh_to_matrix(rot_order, angles)))
        out[j["name"]] = rots
    return out


# --------------------------------------------------------------------------
# 2. Lecture de la pose de repos du modèle Mixamo cible
# --------------------------------------------------------------------------
def _rest_pose_from_gltf(gltf: Any) -> dict[str, tuple[Rotation, Rotation]]:
    """Calcule (rotation locale, rotation monde) de repos pour chaque nœud nommé."""
    n = len(gltf.nodes)
    parent = [None] * n
    for idx, node in enumerate(gltf.nodes):
        for child in node.children or []:
            parent[child] = idx

    local: list[Rotation] = [
        Rotation.from_quat(node.rotation) if node.rotation else Rotation.identity() for node in gltf.nodes
    ]
    world: dict[int, Rotation] = {}

    def get_world(idx: int) -> Rotation:
        if idx not in world:
            world[idx] = local[idx] if parent[idx] is None else get_world(parent[idx]) * local[idx]
        return world[idx]

    return {
        node.name: (local[idx], get_world(idx)) for idx, node in enumerate(gltf.nodes) if node.name
    }


def read_gltf_rest_pose(path: str | Path) -> dict[str, tuple[Rotation, Rotation]]:
    from pygltflib import GLTF2

    path = Path(path)
    gltf = GLTF2().load_binary(str(path)) if path.suffix.lower() in (".glb", ".vrm") else GLTF2().load(str(path))
    return _rest_pose_from_gltf(gltf)


# --------------------------------------------------------------------------
# 3. Table de correspondance BVH -> Mixamo (à ajuster à ta nomenclature exacte)
# --------------------------------------------------------------------------
BVH_TO_MIXAMO: dict[str, str] = {
    "MID_HIP": "mixamorig:Hips",
    "MID_SHOULDER": "mixamorig:Spine2",
    "HEAD": "mixamorig:Head",
    "LEFT_SHOULDER": "mixamorig:LeftArm",
    "LEFT_ELBOW": "mixamorig:LeftForeArm",
    "LEFT_WRIST": "mixamorig:LeftHand",
    "RIGHT_SHOULDER": "mixamorig:RightArm",
    "RIGHT_ELBOW": "mixamorig:RightForeArm",
    "RIGHT_WRIST": "mixamorig:RightHand",
    "LEFT_HIP": "mixamorig:LeftUpLeg",
    "LEFT_KNEE": "mixamorig:LeftLeg",
    "LEFT_ANKLE": "mixamorig:LeftFoot",
    "RIGHT_HIP": "mixamorig:RightUpLeg",
    "RIGHT_KNEE": "mixamorig:RightLeg",
    "RIGHT_ANKLE": "mixamorig:RightFoot",
}
# Doigts : LEFT_HAND_THUMB_CMC -> mixamorig:LeftHandThumb1, etc. (CMC/MCP/IP/TIP ou
# MCP/PIP/DIP/TIP selon le doigt, cf. _FINGERS dans le script BVH).
_FINGER_MIXAMO_SUFFIX = {"THUMB": ["1", "2", "3"], "INDEX_FINGER": ["1", "2", "3"],
                         "MIDDLE_FINGER": ["1", "2", "3"], "RING_FINGER": ["1", "2", "3"], "PINKY": ["1", "2", "3"]}
_FINGER_MIXAMO_NAME = {"THUMB": "Thumb", "INDEX_FINGER": "Index", "MIDDLE_FINGER": "Middle",
                       "RING_FINGER": "Ring", "PINKY": "Pinky"}
for _side_bvh, _side_mixamo in (("LEFT", "Left"), ("RIGHT", "Right")):
    for _finger, _suffixes in _FINGER_MIXAMO_SUFFIX.items():
        _joints = ["CMC", "MCP", "IP"] if _finger == "THUMB" else ["MCP", "PIP", "DIP"]
        for _joint, _suffix in zip(_joints, _suffixes):
            _bvh_name = f"{_side_bvh}_HAND_{_finger}_{_joint}"
            BVH_TO_MIXAMO[_bvh_name] = f"mixamorig:{_side_mixamo}Hand{_FINGER_MIXAMO_NAME[_finger]}{_suffix}"


# --------------------------------------------------------------------------
# 4. Retargeting
# --------------------------------------------------------------------------
def retarget_bvh_to_mixamo(
    bvh_path: str | Path,
    target_glb_path: str | Path,
    bone_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Applique B.local = L_B · W_B⁻¹ · N · W_B pour chaque os mappé.
    N (NormalizedLocalRotation) = rotation locale lue directement dans le BVH,
    car son propre repos est déjà l'identité (voir docstring du module).
    """
    bone_map = bone_map or BVH_TO_MIXAMO
    joints, frame_time, motion = parse_bvh(bvh_path)
    bvh_rotations = extract_bvh_rotations(joints, motion)
    rest_pose = read_gltf_rest_pose(target_glb_path)

    tracks: dict[str, list[dict[str, Any]]] = {}
    skipped: list[str] = []
    fps = 1.0 / frame_time

    for bvh_name, mixamo_name in bone_map.items():
        if bvh_name not in bvh_rotations:
            continue
        if mixamo_name not in rest_pose:
            skipped.append(mixamo_name)
            continue
        l_b, w_b = rest_pose[mixamo_name]
        kfs = []
        for idx, n_rot in enumerate(bvh_rotations[bvh_name]):
            b_local = l_b * w_b.inv() * n_rot * w_b
            kfs.append({"time": idx / fps, "rotation": b_local.as_quat().tolist()})
        tracks[mixamo_name] = kfs

    return {
        "format": "mimo.mixamo.retarget",
        "fps": fps,
        "frameCount": len(motion),
        "tracks": tracks,
        "meta": {"source": str(bvh_path), "target": str(target_glb_path), "bonesSkipped": skipped},
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Usage: python -m backend.retarget_bvh_to_mixamo capture.bvh Clara.glb sortie.json")
        sys.exit(1)
    result = retarget_bvh_to_mixamo(sys.argv[1], sys.argv[2])
    Path(sys.argv[3]).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{len(result['tracks'])} os retargetés, {len(result['meta']['bonesSkipped'])} introuvables dans la cible.")
