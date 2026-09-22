"""Export raw MediaPipe landmarks to BVH, with ZERO retargeting.

Every joint gets its own Xposition/Yposition/Zposition channels, written
directly from MediaPipe's landmark coordinates each frame -- no rotation is
ever computed, no Mixamo bone, no calibration, no bind pose. The BVH
HIERARCHY's OFFSET values are cosmetic (a rough rest-pose shape for
viewers that draw bones from OFFSET); the actual motion is carried
entirely by the per-frame absolute position channels, which override it.

This is deliberately "positions-only" BVH -- unusual outside ROOT, but
valid per the format grammar (each JOINT declares its own CHANNELS list).
Most viewers (Blender included) handle it; a strict parser expecting only
ROOT to have position channels may not.

Body and hands are exported to SEPARATE files: MediaPipe's Pose and Hand
Landmarker are different models with no guaranteed shared coordinate frame.
Merging them here would silently reintroduce an unverified assumption --
exactly what this export exists to avoid. Load both in Blender and see
whether the wrist landmarks actually coincide before assuming they do.

Usage:
    python mediapipe_to_bvh.py raw_frames.json out_body.bvh out_hand_left.bvh out_hand_right.bvh
where raw_frames.json is a list of frames as produced by pose_estimator.py /
fused_extractor.py: {"bodyPose": [...], "handsL": [...], "handsR": [...]}.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Real MediaPipe Pose topology (33 landmarks) -- BlazePose's own official
# connection list, not an invented hierarchy.
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), (0, 9),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (15, 17), (15, 19), (15, 21), (17, 19),
    (16, 18), (16, 20), (16, 22), (18, 20),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
    # NOTE: MediaPipe's own POSE_CONNECTIONS leaves the face cluster
    # (0-10: nose/eyes/ears/mouth) as a SEPARATE component from the body --
    # it never connects to the shoulders in the real topology. BVH needs a
    # single tree, so two edges are added here (ear-to-shoulder, same
    # side) purely for file structure. This changes NOTHING about any
    # joint's position or motion -- only which parent's local OFFSET a
    # face joint is nested under in the HIERARCHY text.
    (7, 11), (8, 12),
]
POSE_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye",
    "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_pinky", "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
]
POSE_ROOT = 23  # left_hip -- an actual landmark, no invented midpoint

# Real MediaPipe Hands topology (21 landmarks, already a tree from the wrist).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
HAND_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]
HAND_ROOT = 0  # wrist


def build_spanning_tree(connections: list[tuple[int, int]], root: int, node_count: int) -> dict[int, int]:
    """BFS over the real connection graph -> {child: parent}. No invented
    joints, no assumed hierarchy beyond what MediaPipe's own topology gives.
    """
    adjacency: dict[int, list[int]] = {i: [] for i in range(node_count)}
    for a, b in connections:
        adjacency[a].append(b)
        adjacency[b].append(a)

    parent: dict[int, int] = {}
    visited = {root}
    queue = [root]
    while queue:
        current = queue.pop(0)
        for neighbor in adjacency[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)
    unreached = set(range(node_count)) - visited
    if unreached:
        raise ValueError(f"Landmarks not connected to root in the topology: {unreached}")
    return parent


def _point(landmark: dict[str, float]) -> tuple[float, float, float]:
    return landmark.get("x", 0.0), landmark.get("y", 0.0), landmark.get("z", 0.0)


def _average_bone_lengths(frames_points: list[list[tuple[float, float, float]]],
                           parent: dict[int, int]) -> dict[int, float]:
    """Average, across every valid frame, the distance between each joint
    and its parent -- the FIXED bone length used for the rest pose. Chosen
    over a single reference frame per your decision: less sensitive to any
    one frame's estimation noise, at the cost of not matching any single
    real instant exactly.
    """
    sums: dict[int, float] = {j: 0.0 for j in parent}
    counts: dict[int, int] = {j: 0 for j in parent}
    for frame in frames_points:
        for joint, par in parent.items():
            jx, jy, jz = frame[joint]
            px, py, pz = frame[par]
            length = ((jx - px) ** 2 + (jy - py) ** 2 + (jz - pz) ** 2) ** 0.5
            sums[joint] += length
            counts[joint] += 1
    return {j: (sums[j] / counts[j] if counts[j] else 0.0) for j in parent}


def write_bvh_rotational(frames_points: list[list[tuple[float, float, float]]], names: list[str],
                          parent: dict[int, int], root_index: int, rest_direction: dict[int, tuple[float, float, float]],
                          fps: float, output_path: str | Path, scale: float = 100.0,
                          euler_order: str = "ZXY") -> dict:
    """Real rotational BVH: fixed bone lengths (averaged over the whole
    clip), a chosen rest-pose direction per bone (this skeleton is our own
    -- we get to define what "rest" means, unlike Mixamo retargeting where
    the rest pose is dictated by someone else's model), and per-frame local
    rotations computed by swinging the rest direction onto the observed
    direction -- same principle as body_solver.py's aim step, but simpler:
    no per-model calibration needed, since we define our own convention.

    Root (hips) gets Xposition/Yposition/Zposition (real motion) PLUS
    rotation. Every other joint gets ONLY rotation (standard BVH shape,
    unlike the positions-only export above) -- this is what makes the
    result retargetable by Blender/Rokoko/UE's own tools.

    Only swing is computed (no roll/twist) for every bone here, consistent
    with the ROLL_CAPABLE_SLOTS caution already applied in body_solver.py:
    no verified twist reference exists for a generic MediaPipe skeleton
    either, so it is left out rather than guessed.
    """
    from scipy.spatial.transform import Rotation
    import numpy as np

    children: dict[int, list[int]] = {}
    for child, par in parent.items():
        children.setdefault(par, []).append(child)

    bone_lengths = _average_bone_lengths(frames_points, parent)

    lines: list[str] = ["HIERARCHY"]
    channel_order: list[tuple[int, bool]] = []  # (joint_index, has_position)

    def write_joint(joint_index: int, depth: int, is_root: bool) -> None:
        indent = "  " * depth
        keyword = "ROOT" if is_root else "JOINT"
        lines.append(f"{indent}{keyword} {names[joint_index]}")
        lines.append(f"{indent}{{")
        if is_root:
            ox, oy, oz = 0.0, 0.0, 0.0
        else:
            dx, dy, dz = rest_direction[joint_index]
            length = bone_lengths[joint_index]
            ox, oy, oz = dx * length * scale, dy * length * scale, dz * length * scale
        lines.append(f"{indent}  OFFSET {ox:.6f} {oy:.6f} {oz:.6f}")
        if is_root:
            lines.append(f"{indent}  CHANNELS 6 Xposition Yposition Zposition "
                          f"{euler_order[0]}rotation {euler_order[1]}rotation {euler_order[2]}rotation")
            channel_order.append((joint_index, True))
        else:
            lines.append(f"{indent}  CHANNELS 3 {euler_order[0]}rotation {euler_order[1]}rotation {euler_order[2]}rotation")
            channel_order.append((joint_index, False))

        child_indices = children.get(joint_index, [])
        if not child_indices:
            lines.append(f"{indent}  End Site")
            lines.append(f"{indent}  {{")
            lines.append(f"{indent}    OFFSET 0.000000 0.000000 0.000000")
            lines.append(f"{indent}  }}")
        else:
            for child_index in child_indices:
                write_joint(child_index, depth + 1, is_root=False)
        lines.append(f"{indent}}}")

    write_joint(root_index, depth=0, is_root=True)

    lines.append("MOTION")
    lines.append(f"Frames: {len(frames_points)}")
    lines.append(f"Frame Time: {1.0 / fps:.6f}")

    for frame in frames_points:
        world_rotations: dict[int, Rotation] = {root_index: Rotation.identity()}
        row: list[float] = []

        rx, ry, rz = frame[root_index]
        row.extend([rx * scale, ry * scale, rz * scale])

        def process(joint_index: int, is_root: bool) -> None:
            if not is_root:
                parent_index = parent[joint_index]
                parent_world = world_rotations[parent_index]
                jx, jy, jz = frame[joint_index]
                px, py, pz = frame[parent_index]
                observed_world = np.array([jx - px, jy - py, jz - pz])
                norm = np.linalg.norm(observed_world)
                target_local = (parent_world.inv().apply(observed_world / norm)
                                 if norm > 1e-8 else np.array(rest_direction[joint_index]))
                rest = np.array(rest_direction[joint_index])
                local_rotation, _ = Rotation.align_vectors([target_local], [rest])
                world_rotations[joint_index] = parent_world * local_rotation
            else:
                local_rotation = Rotation.identity()

            euler = local_rotation.as_euler(euler_order.lower(), degrees=True)
            row.extend(euler.tolist())

            for child_index in children.get(joint_index, []):
                process(child_index, is_root=False)

        process(root_index, is_root=True)
        lines.append(" ".join(f"{v:.6f}" for v in row))

    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"output": str(output_path), "boneLengths": {names[j]: round(bone_lengths[j], 4) for j in bone_lengths}}


def write_bvh(frames_points: list[list[tuple[float, float, float]]], names: list[str],
              parent: dict[int, int], root_index: int, fps: float, output_path: str | Path,
              scale: float = 100.0) -> None:
    """frames_points[frame_idx][joint_idx] = (x, y, z) in MediaPipe's own
    units (metres). scale converts to BVH's conventional centimetre-ish
    units purely for viewer friendliness -- it is a unit conversion, NOT a
    retargeting scale (see bake_animation.py's position_scale for that
    different, model-fitting concern).
    """
    children: dict[int, list[int]] = {}
    for child, par in parent.items():
        children.setdefault(par, []).append(child)

    lines: list[str] = ["HIERARCHY"]
    channel_order: list[int] = []  # joint index in the order channels appear, for MOTION section

    def rest_offset(joint_index: int) -> tuple[float, float, float]:
        # Cosmetic only (see module docstring) -- use frame 0's position
        # relative to its parent so a viewer's static bind pose looks
        # roughly human, even though motion channels override it entirely.
        x, y, z = frames_points[0][joint_index]
        if joint_index == root_index:
            return (0.0, 0.0, 0.0)
        px, py, pz = frames_points[0][parent[joint_index]]
        return ((x - px) * scale, (y - py) * scale, (z - pz) * scale)

    def write_joint(joint_index: int, depth: int, is_root: bool) -> None:
        indent = "  " * depth
        keyword = "ROOT" if is_root else "JOINT"
        lines.append(f"{indent}{keyword} {names[joint_index]}")
        lines.append(f"{indent}{{")
        ox, oy, oz = rest_offset(joint_index)
        lines.append(f"{indent}  OFFSET {ox:.6f} {oy:.6f} {oz:.6f}")
        lines.append(f"{indent}  CHANNELS 3 Xposition Yposition Zposition")
        channel_order.append(joint_index)

        child_indices = children.get(joint_index, [])
        if not child_indices:
            lines.append(f"{indent}  End Site")
            lines.append(f"{indent}  {{")
            lines.append(f"{indent}    OFFSET 0.000000 0.000000 0.000000")
            lines.append(f"{indent}  }}")
        else:
            for child_index in child_indices:
                write_joint(child_index, depth + 1, is_root=False)
        lines.append(f"{indent}}}")

    write_joint(root_index, depth=0, is_root=True)

    lines.append("MOTION")
    lines.append(f"Frames: {len(frames_points)}")
    lines.append(f"Frame Time: {1.0 / fps:.6f}")

    for frame in frames_points:
        values = []
        for joint_index in channel_order:
            x, y, z = frame[joint_index]
            values.extend([x * scale, y * scale, z * scale])
        lines.append(" ".join(f"{v:.6f}" for v in values))

    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    n = (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5
    return (v[0] / n, v[1] / n, v[2] / n)


# A chosen neutral A-pose direction (parent -> child) for every bone in the
# real BFS tree above. This is OUR OWN convention -- unlike Mixamo
# retargeting, nothing external dictates what "rest" means here, so any
# consistent choice is valid. Y = up, Z = forward (arbitrary but fixed).
POSE_REST_DIRECTIONS_RAW = {
    "left_shoulder": (-0.3, 0.95, 0.0), "right_shoulder": (1.0, 0.0, 0.0),
    "left_ear": (0.0, 1.0, 0.3), "right_ear": (0.0, 1.0, 0.3),
    "left_eye_outer": (0.0, 0.0, 1.0), "left_eye": (0.0, 0.0, 1.0), "left_eye_inner": (0.0, 0.0, 1.0),
    "nose": (0.0, 0.0, 1.0), "mouth_left": (0.0, -1.0, 0.0), "mouth_right": (1.0, 0.0, 0.0),
    "right_eye_outer": (0.0, 0.0, 1.0), "right_eye": (0.0, 0.0, 1.0), "right_eye_inner": (0.0, 0.0, 1.0),
    "left_elbow": (-0.7, -0.7, 0.0), "right_elbow": (0.7, -0.7, 0.0),
    "left_wrist": (-0.3, -0.95, 0.0), "right_wrist": (0.3, -0.95, 0.0),
    "left_pinky": (-0.4, -0.9, 0.2), "left_index": (-0.2, -0.95, 0.2), "left_thumb": (-0.6, -0.7, 0.4),
    "right_pinky": (0.4, -0.9, 0.2), "right_index": (0.2, -0.95, 0.2), "right_thumb": (0.6, -0.7, 0.4),
    "right_hip": (1.0, 0.0, 0.0), "left_knee": (0.0, -1.0, 0.0), "right_knee": (0.0, -1.0, 0.0),
    "left_ankle": (0.0, -1.0, 0.0), "right_ankle": (0.0, -1.0, 0.0),
    "left_heel": (0.0, -0.3, -0.95), "right_heel": (0.0, -0.3, -0.95),
    "left_foot_index": (0.0, -0.2, 0.98), "right_foot_index": (0.0, -0.2, 0.98),
}


def export_body_bvh_rotational(frames: list[dict[str, Any]], output_path: str | Path, fps: float = 30.0) -> dict:
    parent = build_spanning_tree(POSE_CONNECTIONS, POSE_ROOT, len(POSE_NAMES))
    rest_direction = {
        POSE_NAMES.index(name): _normalize(vector)
        for name, vector in POSE_REST_DIRECTIONS_RAW.items()
    }
    frames_points = []
    for frame in frames:
        body = frame.get("bodyPose") or frame.get("predictions") or []
        if len(body) < len(POSE_NAMES):
            continue
        frames_points.append([_point(body[i]) for i in range(len(POSE_NAMES))])
    if not frames_points:
        raise ValueError("No frame has a complete body pose (33 points) -- nothing to export.")
    report = write_bvh_rotational(frames_points, POSE_NAMES, parent, POSE_ROOT, rest_direction, fps, output_path)
    report["framesExported"] = len(frames_points)
    return report


def export_body_bvh(frames: list[dict[str, Any]], output_path: str | Path, fps: float = 30.0) -> dict:
    parent = build_spanning_tree(POSE_CONNECTIONS, POSE_ROOT, len(POSE_NAMES))
    frames_points = []
    usable_frames = 0
    for frame in frames:
        body = frame.get("bodyPose") or frame.get("predictions") or []
        if len(body) < len(POSE_NAMES):
            continue
        frames_points.append([_point(body[i]) for i in range(len(POSE_NAMES))])
        usable_frames += 1
    if not frames_points:
        raise ValueError("No frame has a complete body pose (33 points) -- nothing to export.")
    write_bvh(frames_points, POSE_NAMES, parent, POSE_ROOT, fps, output_path)
    return {"output": str(output_path), "framesExported": usable_frames, "framesSkipped": len(frames) - usable_frames}


def export_hand_bvh(frames: list[dict[str, Any]], side: str, output_path: str | Path, fps: float = 30.0) -> dict:
    key = "handsL" if side == "left" else "handsR"
    parent = build_spanning_tree(HAND_CONNECTIONS, HAND_ROOT, len(HAND_NAMES))
    frames_points = []
    usable_frames = 0
    for frame in frames:
        hand = frame.get(key) or []
        if len(hand) < len(HAND_NAMES):
            continue
        frames_points.append([_point(hand[i]) for i in range(len(HAND_NAMES))])
        usable_frames += 1
    if not frames_points:
        raise ValueError(f"No frame has a complete {side} hand (21 points) -- nothing to export.")
    write_bvh(frames_points, HAND_NAMES, parent, HAND_ROOT, fps, output_path)
    return {"output": str(output_path), "framesExported": usable_frames, "framesSkipped": len(frames) - usable_frames}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", help="Path to raw frames JSON (list of {bodyPose, handsL, handsR})")
    parser.add_argument("out_body", help="Output .bvh for the body")
    parser.add_argument("out_hand_left", nargs="?", default=None, help="Output .bvh for the left hand")
    parser.add_argument("out_hand_right", nargs="?", default=None, help="Output .bvh for the right hand")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    with open(args.frames, "r", encoding="utf-8") as handle:
        frames = json.load(handle)

    report = {"body": export_body_bvh(frames, args.out_body, args.fps)}
    if args.out_hand_left:
        report["handLeft"] = export_hand_bvh(frames, "left", args.out_hand_left, args.fps)
    if args.out_hand_right:
        report["handRight"] = export_hand_bvh(frames, "right", args.out_hand_right, args.fps)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
