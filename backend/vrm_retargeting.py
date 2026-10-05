from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

"""
Solveur de rotations VRM 1.0 a partir des reperes 3D MediaPipe (Pose + mains).

Entree : convention MediaPipe world (y vers le BAS, +z s'eloigne de la camera,
x vers la droite de l'image = gauche anatomique d'une personne face camera).
Sortie : rotations locales VRM 1.0 (Y haut, l'avatar regarde +Z, son bras gauche suit +X).
Conversion : (x, y, z) -> (x, -y, -z), rotation de 180 deg autour de X (det = +1).

T-pose VRM 1.0 (rotations nulles) :
  - bras / avant-bras / doigts gauches le long de +X, droits le long de -X ;
  - paume vers le bas (-Y), pouce vers l'avant (+Z) ;
  - pouce : suppose a 45 deg entre +-X et +Z (non verifie sur un modele reel).
Chaque os est resolu par l'arc minimal entre sa direction de repos et la direction
observee, exprimee dans le repere de son parent : la direction est exacte, la torsion
autour de l'os n'est pas observee (sauf pour la paume, orientee par 3 points).
Le torse n'est pas anime : les bras sont exprimes dans le repere monde.
"""

_THUMB_REST = 1.0 / math.sqrt(2.0)
_FINGERS = [
    ("Thumb", ["Metacarpal", "Proximal", "Distal"], [1, 2, 3, 4]),
    ("Index", ["Proximal", "Intermediate", "Distal"], [5, 6, 7, 8]),
    ("Middle", ["Proximal", "Intermediate", "Distal"], [9, 10, 11, 12]),
    ("Ring", ["Proximal", "Intermediate", "Distal"], [13, 14, 15, 16]),
    ("Little", ["Proximal", "Intermediate", "Distal"], [17, 18, 19, 20]),
]


def mediapipe_to_vrm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return p * np.array([1.0, -1.0, -1.0])


def _point(landmarks: list[Any], idx: int) -> np.ndarray:
    """Point idx en convention MediaPipe ; NaN si absent ou illisible (jamais 0 par defaut)."""
    if idx >= len(landmarks):
        return np.full(3, np.nan)
    p = landmarks[idx]
    try:
        if isinstance(p, dict):
            return np.array([p["x"], p["y"], p["z"]], dtype=float)
        if hasattr(p, "x"):
            return np.array([p.x, p.y, p.z], dtype=float)
        return np.array(list(p)[:3], dtype=float) if len(p) >= 3 else np.full(3, np.nan)
    except (KeyError, TypeError, ValueError):
        return np.full(3, np.nan)


def _vrm_points(landmarks: list[Any], indices) -> np.ndarray:
    return np.stack([mediapipe_to_vrm(_point(landmarks, i)) for i in indices])


def _arc(a: np.ndarray, b: np.ndarray) -> Rotation:
    """Rotation d'angle minimal qui envoie la direction a sur la direction b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if s < 1e-9:
        if c > 0:
            return Rotation.identity()
        perp = np.cross(a, [0.0, 1.0, 0.0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(a, [0.0, 0.0, 1.0])
        return Rotation.from_rotvec(perp / np.linalg.norm(perp) * math.pi)
    return Rotation.from_rotvec(axis / s * math.atan2(s, c))


def _direction(a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    d = b - a
    n = np.linalg.norm(d)
    if not np.isfinite(d).all() or n < 1e-9:
        return None
    return d / n


def solve_arm_kinematics(shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray,
                         side: str = "left") -> tuple[Rotation, Rotation] | None:
    """Points en repere VRM. -> (rotation monde du bras, rotation locale de l'avant-bras) ou None."""
    rest = np.array([1.0, 0.0, 0.0]) if side == "left" else np.array([-1.0, 0.0, 0.0])
    upper = _direction(shoulder, elbow)
    lower = _direction(elbow, wrist)
    if upper is None or lower is None:
        return None
    upper_rot = _arc(rest, upper)
    lower_rot = _arc(rest, upper_rot.inv().apply(lower))
    return upper_rot, lower_rot


def palm_rotation(hand: np.ndarray, side: str) -> Rotation | None:
    """hand (21, 3) en repere VRM -> rotation monde de la main (identite en T-pose), ou None."""
    wrist, index_mcp, middle_mcp, pinky_mcp = hand[0], hand[5], hand[9], hand[17]
    if not np.isfinite(np.stack([wrist, index_mcp, middle_mcp, pinky_mcp])).all():
        return None
    sign = 1.0 if side == "left" else -1.0
    forward = _direction(wrist, middle_mcp)
    across = _direction(index_mcp, pinky_mcp) if side == "left" else _direction(pinky_mcp, index_mcp)
    if forward is None or across is None:
        return None
    normal = np.cross(across, forward)
    if np.linalg.norm(normal) < 1e-6:
        return None
    normal /= np.linalg.norm(normal)
    across = np.cross(forward, normal)
    # repos gauche : forward=+X, across=-Z, normal=-Y ; droite : forward=-X, across=+Z, normal=-Y
    m = np.column_stack((sign * forward, -normal, -sign * across))
    return Rotation.from_matrix(m)


def solve_hand_kinematics(landmarks: list[Any], side: str = "left",
                          parent_forearm_rot: Rotation | None = None) -> dict[str, list[float]]:
    """landmarks : 21 points MediaPipe (convention MediaPipe). -> quaternions locaux (x, y, z, w)."""
    if len(landmarks) < 21:
        return {}
    hand = _vrm_points(landmarks, range(21))
    prefix = "left" if side == "left" else "right"
    sign = 1.0 if side == "left" else -1.0
    palm = palm_rotation(hand, side)
    if palm is None:
        return {}
    parent = parent_forearm_rot if parent_forearm_rot is not None else Rotation.identity()
    out = {f"{prefix}Hand": (parent.inv() * palm).as_quat().tolist()}
    for finger, bones, idx in _FINGERS:
        rest = (np.array([sign * _THUMB_REST, 0.0, _THUMB_REST]) if finger == "Thumb"
                else np.array([sign, 0.0, 0.0]))
        world = palm
        for bone, a, b in zip(bones, idx[:-1], idx[1:]):
            d = _direction(hand[a], hand[b])
            if d is None:
                break
            local = _arc(rest, world.inv().apply(d))
            out[f"{prefix}{finger}{bone}"] = local.as_quat().tolist()
            world = world * local
    return out


def process_mimo_landmarks_to_vrma(
    frames: list[dict[str, Any]],
    fps: float = 30.0,
    sign_label: str = "Sign Language Sequence"
) -> dict[str, Any]:
    """Frames MediaPipe (bodyPose 33 + handsL / handsR 21) -> clip VRMA (quaternions locaux)."""
    if not (fps > 0 and math.isfinite(fps)):
        raise ValueError(f"fps invalide : {fps}")
    total_frames = len(frames)
    duration = total_frames / fps
    tracks: dict[str, list[dict[str, Any]]] = {}

    for idx, frame in enumerate(frames):
        t = float(idx) / float(fps)
        body = frame.get("bodyPose") or []
        hands_l = frame.get("handsL") or []
        hands_r = frame.get("handsR") or []
        rotations: dict[str, list[float]] = {}
        forearm = {"left": None, "right": None}

        if len(body) >= 25:
            pts = {i: mediapipe_to_vrm(_point(body, i)) for i in (0, 7, 8, 11, 12, 13, 14, 15, 16)}
            for side, (s, e, w) in (("left", (11, 13, 15)), ("right", (12, 14, 16))):
                arm = solve_arm_kinematics(pts[s], pts[e], pts[w], side=side)
                if arm is None:
                    continue
                up, low = arm
                rotations[f"{side}UpperArm"] = up.as_quat().tolist()
                rotations[f"{side}LowerArm"] = low.as_quat().tolist()
                forearm[side] = up * low
            head_dir = _direction((pts[7] + pts[8]) * 0.5, pts[0])
            if head_dir is not None:
                yaw = math.atan2(head_dir[0], head_dir[2])
                rotations["head"] = Rotation.from_rotvec([0.0, yaw, 0.0]).as_quat().tolist()

        for side, hand in (("left", hands_l), ("right", hands_r)):
            if not hand:
                continue
            solved = solve_hand_kinematics(hand, side=side, parent_forearm_rot=forearm[side])
            if forearm[side] is None:
                solved.pop(f"{side}Hand", None)        # orientation de la main sans parent connu
            rotations.update(solved)

        for bone_name, q in rotations.items():
            tracks.setdefault(bone_name, []).append({"time": t, "rotation": q})

    return {
        "format": "mimo.vrm.animation",
        "version": 1,
        "standard": "VRM1.0 / OpenXR Humanoid",
        "fps": fps,
        "duration": duration,
        "tracks": tracks,
        "meta": {
            "frameCount": total_frames,
            "bonesAnimated": list(tracks.keys()),
            "specialization": "Langue des Signes (LSF/ASL)",
            "signWord": sign_label,
            "solver": "Direction-exact arc solver v3 (MediaPipe -> VRM axes)",
            "assumptions": [
                "torse non anime : bras exprimes dans le repere monde",
                "torsion des os non observee (sauf paume)",
                "repos du pouce suppose a 45 deg entre X et +Z",
            ],
        }
    }
