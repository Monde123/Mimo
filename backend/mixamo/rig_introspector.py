from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from pygltflib import GLTF2
from scipy.spatial.transform import Rotation

# Fixed vocabulary of canonical bone "slots". Every model, whatever its own
# naming scheme, is mapped ONTO these — the rest of the pipeline
# (body_solver.py / hand_solver.py) only ever talks about these names.
CANONICAL_BONES = [
    "hips", "spine", "chest", "neck", "head",
    "leftUpperArm", "leftLowerArm", "leftHand",
    "rightUpperArm", "rightLowerArm", "rightHand",
]

# Each canonical bone's expected child, used only to derive a bind-pose
# "forward" direction (bone -> child). Slot names again, never model names.
CHILD_OF = {
    "hips": "spine", "spine": "chest", "chest": "neck", "neck": "head",
    "leftUpperArm": "leftLowerArm", "leftLowerArm": "leftHand",
    "rightUpperArm": "rightLowerArm", "rightLowerArm": "rightHand",
}

# VRM humanoid bone names (VRM 0.x and 1.0 use the same identifiers) mapped
# to our own slot names.
VRM_NAME_MAP = {
    "hips": "hips", "spine": "spine", "chest": "chest", "neck": "neck", "head": "head",
    "leftUpperArm": "leftUpperArm", "leftLowerArm": "leftLowerArm", "leftHand": "leftHand",
    "rightUpperArm": "rightUpperArm", "rightLowerArm": "rightLowerArm", "rightHand": "rightHand",
}

# Fallback ONLY used when the model has no VRM humanoid extension — i.e. a
# raw Mixamo / Unity-Humanoid / generic glTF export. Covers the common
# naming conventions (mixamorig:LeftArm, Left_Arm, LeftUpperArm...).
NAME_PATTERNS = {
    "hips": re.compile(r"hips?$", re.I),
    "spine": re.compile(r"spine1?$", re.I),
    "chest": re.compile(r"spine2$|chest$", re.I),
    "neck": re.compile(r"neck$", re.I),
    "head": re.compile(r"head$", re.I),
    "leftUpperArm": re.compile(r"left.?(arm|upperarm)$", re.I),
    "leftLowerArm": re.compile(r"left.?(forearm|lowerarm)$", re.I),
    "leftHand": re.compile(r"left.?hand$", re.I),
    "rightUpperArm": re.compile(r"right.?(arm|upperarm)$", re.I),
    "rightLowerArm": re.compile(r"right.?(forearm|lowerarm)$", re.I),
    "rightHand": re.compile(r"right.?hand$", re.I),
}


def _local_trs(gltf: GLTF2, node_index: int) -> np.ndarray:
    """4x4 local transform of a glTF node, from its matrix or TRS fields."""
    node = gltf.nodes[node_index]
    if node.matrix:
        return np.array(node.matrix, dtype=float).reshape(4, 4).T
    t = np.array(node.translation or [0.0, 0.0, 0.0], dtype=float)
    r = node.rotation or [0.0, 0.0, 0.0, 1.0]  # xyzw
    s = np.array(node.scale or [1.0, 1.0, 1.0], dtype=float)
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(r).as_matrix() * s  # scale columns
    matrix[:3, 3] = t
    return matrix


def compute_world_transforms(gltf: GLTF2) -> dict[int, np.ndarray]:
    """World-space 4x4 transform for every node, walked from the scene roots.
    This is the bind pose, since glTF files carry no separate rest-pose
    channel — node transforms ARE the rest pose unless an animation moves them.
    """
    world: dict[int, np.ndarray] = {}

    def visit(node_index: int, parent_world: np.ndarray) -> None:
        local = _local_trs(gltf, node_index)
        current = parent_world @ local
        world[node_index] = current
        for child_index in gltf.nodes[node_index].children or []:
            visit(child_index, current)

    scene = gltf.scenes[gltf.scene or 0]
    for root_index in scene.nodes:
        visit(root_index, np.eye(4))
    return world


def _vrm_bone_nodes(gltf: GLTF2) -> dict[str, int]:
    """Read the VRM humanoid extension (0.x or 1.0) if present."""
    extensions = gltf.extensions or {}
    vrm1 = extensions.get("VRMC_vrm")
    if vrm1:
        human_bones = vrm1.get("humanoid", {}).get("humanBones", {})
        return {
            slot: human_bones[vrm_name]["node"]
            for slot, vrm_name in VRM_NAME_MAP.items()
            if vrm_name in human_bones
        }
    vrm0 = extensions.get("VRM")
    if vrm0:
        bones_list = vrm0.get("humanoid", {}).get("humanBones", [])
        by_name = {entry["bone"]: entry["node"] for entry in bones_list}
        return {
            slot: by_name[vrm_name]
            for slot, vrm_name in VRM_NAME_MAP.items()
            if vrm_name in by_name
        }
    return {}


def _pattern_bone_nodes(gltf: GLTF2) -> dict[str, int]:
    """Fallback: match node names against NAME_PATTERNS. Restricted to a
    skin's joints when a skin exists, so a mesh or prop with a matching
    name (e.g. a "Head" accessory) is never picked up by mistake.
    """
    candidate_indices = set(range(len(gltf.nodes)))
    if gltf.skins:
        candidate_indices = set(gltf.skins[0].joints)

    resolved: dict[str, int] = {}
    for slot, pattern in NAME_PATTERNS.items():
        for index in candidate_indices:
            name = gltf.nodes[index].name or ""
            if pattern.search(name):
                resolved[slot] = index
                break
    return resolved


def resolve_humanoid_bones(gltf: GLTF2) -> tuple[dict[str, int], str]:
    """Canonical slot -> node index, independently of the model.
    Returns (mapping, source) where source is "vrm-humanoid" or
    "name-pattern-fallback", so the caller/report can flag low-confidence
    calibrations (fallback) versus authoritative ones (VRM metadata).
    """
    vrm_bones = _vrm_bone_nodes(gltf)
    if vrm_bones:
        return vrm_bones, "vrm-humanoid"
    return _pattern_bone_nodes(gltf), "name-pattern-fallback"


def build_rig_calibration(model_path: str | Path) -> dict[str, Any]:
    """Load any .glb/.gltf/.vrm file and derive a calibration dict with the
    same shape rig_calibration.py already expects, but generated from the
    model's own bind pose instead of the axes currently hardcoded in
    body_solver.py ([0,0,1] forward / [0,1,0] up for every model).

    For each canonical bone: the bind-pose direction toward its canonical
    child, expressed in the bone's OWN local frame — which is what makes
    this work identically for a T-pose model and an A-pose model.
    """
    gltf = GLTF2().load(str(model_path))
    bone_nodes, source = resolve_humanoid_bones(gltf)
    world_transforms = compute_world_transforms(gltf)
    parent_of = {child: parent for parent, child in CHILD_OF.items()}

    calibration: dict[str, Any] = {
        "bones": {},
        "meta": {
            "sourceModel": Path(model_path).name,
            "resolutionMethod": source,
            "missingSlots": [slot for slot in CANONICAL_BONES if slot not in bone_nodes],
        },
    }

    # bindLocalRotation: the bone's REAL local rotation in bind pose,
    # relative to its parent -- not derived from where the child happens to
    # sit (that only gives ONE axis), but read from the node's own rotation
    # in the glTF, exactly like ModelNode.set_mixamo reads a Pixel3D node's
    # "rotation" field directly instead of inferring it from positions.
    # This gives every bone (leaves included: head, hands) a FULL basis --
    # forward AND a genuine roll reference -- both mutually consistent
    # since they come from the same real rotation, unlike the old
    # canonicalForward-only approach where a synthetic "twist" reference
    # had no real relationship to the observed one (see body_solver_v2's
    # earlier 90-degree bug on a neutral pose).
    for slot, bone_index in bone_nodes.items():
        bone_world_rot = world_transforms[bone_index][:3, :3]
        parent_slot = parent_of.get(slot)
        parent_index = bone_nodes.get(parent_slot) if parent_slot else None
        if parent_index is not None:
            parent_world_rot = world_transforms[parent_index][:3, :3]
            local_rot_matrix = parent_world_rot.T @ bone_world_rot
        else:
            local_rot_matrix = bone_world_rot  # root (hips): local == world
        calibration["bones"].setdefault(slot, {})["nodeName"] = gltf.nodes[bone_index].name
        calibration["bones"][slot]["bindLocalRotation"] = Rotation.from_matrix(local_rot_matrix).as_quat().tolist()

    # canonicalForward kept for backward compatibility with any code still
    # reading it directly; bindLocalRotation above is the complete version
    # and should be preferred for anything computing rotations (it also
    # covers leaf bones -- head, hands -- that canonicalForward never did).
    for slot, child_slot in CHILD_OF.items():
        bone_index = bone_nodes.get(slot)
        child_index = bone_nodes.get(child_slot)
        if bone_index is None or child_index is None:
            continue

        bone_world = world_transforms[bone_index]
        child_world = world_transforms[child_index]

        bone_world_pos = bone_world[:3, 3]
        child_world_pos = child_world[:3, 3]
        world_dir = child_world_pos - bone_world_pos
        world_dir = world_dir / (np.linalg.norm(world_dir) + 1e-8)

        # Rotate the world-space direction back into the bone's own local
        # frame, so the stored axis stays meaningful regardless of the
        # bone's bind-pose rotation.
        bone_world_rot = Rotation.from_matrix(bone_world[:3, :3])
        local_dir = bone_world_rot.inv().apply(world_dir)

        calibration["bones"].setdefault(slot, {})["canonicalForward"] = local_dir.tolist()


    return calibration


def generate_calibration_file(model_path: str | Path, output_path: str | Path) -> Path:
    """CLI-friendly entry point: run once per avatar, write the result next
    to the other rig configs (e.g. configs/mixamo_default.json), then pass
    --rig <output_path> to process_video.py / process_precise.py as usual.
    """
    calibration = build_rig_calibration(model_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a rig calibration file from a .glb/.gltf/.vrm model's bind pose."
    )
    parser.add_argument("model", type=Path, help="Path to the .glb/.gltf/.vrm file")
    parser.add_argument("output", type=Path, help="Where to write the calibration JSON")
    args = parser.parse_args()
    result = generate_calibration_file(args.model, args.output)
    report = build_rig_calibration(args.model)["meta"]
    print(json.dumps({"status": "complete", "output": str(result), **report}, indent=2))


if __name__ == "__main__":
    main()