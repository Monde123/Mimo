
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-8 else np.array([0.0, 1.0, 0.0])


def _point(landmarks: list[dict[str, float]], idx: int) -> np.ndarray:
    if idx < len(landmarks):
        p = landmarks[idx]
        return np.array([p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)], dtype=float)
    return np.zeros(3, dtype=float)


def to_vrm_space(
    landmarks: list[dict[str, float]],
    mirror_x: bool = False,
    aspect: float = 1.0,
) -> list[dict[str, float]]:
    """
    MediaPipe -> repère VRM : (x, y, z) -> (x, -y, -z).
      - y : MediaPipe va vers le bas, VRM vers le haut
      - z : MediaPipe négatif = vers la caméra ; le modèle VRM regarde +Z, donc vers la caméra = +Z
    mirror_x : à activer si la vidéo est en miroir (selfie).
    aspect   : largeur/hauteur de la vidéo. Les coordonnées normalisées x et z sont
               en unités de largeur, y en unités de hauteur ; sans correction les
               vecteurs sont déformés. Laisser 1.0 pour des pose_world_landmarks (mètres).
    """
    out: list[dict[str, float]] = []
    for p in landmarks:
        x = p.get("x", 0.0) * aspect
        y = p.get("y", 0.0)
        z = p.get("z", 0.0) * aspect
        out.append({"x": -x if mirror_x else x, "y": -y, "z": -z})
    return out


def _basis_from_x_z(x_axis: np.ndarray, z_hint: np.ndarray) -> tuple[Rotation, np.ndarray]:
    """
    Construit la rotation dont la base image (1,0,0)->x_axis, (0,0,1)->z_axis (orthogonalisé),
    (0,1,0)->y_axis. Contrairement à une rotation "arc le plus court" (axe-angle), le twist
    autour de x_axis est fixé par z_hint plutôt que laissé libre : c'est ce qui élimine les
    sauts de roulis. Retourne aussi le z_axis réellement utilisé (pour le fil de continuité
    d'une frame à l'autre).
    """
    x = _unit(x_axis)
    z = z_hint - np.dot(z_hint, x) * x  # orthogonalisation de Gram-Schmidt
    if np.linalg.norm(z) < 1e-6:
        # z_hint quasi colinéaire à x : axe de repli arbitraire mais déterministe
        fallback = np.array([0.0, 1.0, 0.0]) if abs(x[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
        z = fallback - np.dot(fallback, x) * x
    z = _unit(z)
    y = _unit(np.cross(z, x))
    z = np.cross(x, y)  # ré-orthogonalisation finale
    return Rotation.from_matrix(np.column_stack((x, y, z))), z


def solve_arm_kinematics(
    shoulder: np.ndarray,
    elbow: np.ndarray,
    wrist: np.ndarray,
    side: str = "left",
    prev_plane_normal: np.ndarray | None = None,
) -> tuple[Rotation, Rotation, np.ndarray]:
    """
    Orientation de l'épaule et du coude, torsion fixée par le plan de flexion du coude
    (épaule, coude, poignet) plutôt que laissée libre.
    prev_plane_normal : normale du plan à la frame précédente, pour choisir le même signe
    (le produit vectoriel donne deux normales opposées valides ; sans ce fil, le solveur peut
    en choisir une différente d'une frame à l'autre et faire "flipper" tout l'avant-bras).
    Retourne (upper_rot, lower_rot, plane_normal_utilisée) — à repasser en prev_plane_normal
    à la frame suivante.
    Entrées attendues dans le repère VRM (voir to_vrm_space).
    """
    is_left = side == "left"
    rest_arm = np.array([1.0, 0.0, 0.0]) if is_left else np.array([-1.0, 0.0, 0.0])

    v_upper = _unit(elbow - shoulder)
    v_lower = _unit(wrist - elbow)

    normal = np.cross(v_upper, v_lower)
    n = np.linalg.norm(normal)
    if n > 1e-4:
        plane_normal = normal / n
        if prev_plane_normal is not None and np.dot(plane_normal, prev_plane_normal) < 0:
            plane_normal = -plane_normal
    elif prev_plane_normal is not None:
        # bras (quasi) tendu : le plan est indéfini, on garde la torsion de la frame précédente
        plane_normal = prev_plane_normal
    else:
        plane_normal = np.array([0.0, 0.0, 1.0])

    upper_rot, _ = _basis_from_x_z(v_upper, plane_normal)

    local_lower_dir = upper_rot.inv().apply(v_lower)
    local_plane_normal = upper_rot.inv().apply(plane_normal)
    lower_rot, _ = _basis_from_x_z(local_lower_dir, local_plane_normal)

    return upper_rot, lower_rot, plane_normal


def solve_hand_kinematics(
    landmarks: list[dict[str, float]],
    side: str = "left",
    parent_forearm_rot: Rotation | None = None,
) -> dict[str, list[float]]:
    """
    Orientation de la paume et des 15 phalanges.
    Entrées attendues dans le repère VRM (voir to_vrm_space).
    """
    if len(landmarks) < 21:
        return {}

    is_left = side == "left"
    sign = 1.0 if is_left else -1.0  # +1 gauche, -1 droite
    prefix = "left" if is_left else "right"
    results: dict[str, list[float]] = {}

    wrist = _point(landmarks, 0)
    index_mcp = _point(landmarks, 5)
    middle_mcp = _point(landmarks, 9)
    pinky_mcp = _point(landmarks, 17)

    # Base de la paume alignée sur le repère de repos VRM :
    #   X local = direction des doigts (+X à gauche, -X à droite)
    #   Y local = dos de la main (vers le haut quand la paume est vers le bas)
    #   Z local = X x Y (côté du pouce, vers l'avant au repos)
    forward = _unit(middle_mcp - wrist)
    back = sign * np.cross(index_mcp - wrist, pinky_mcp - wrist)
    back = back - np.dot(back, forward) * forward  # orthogonalisation
    back = _unit(back)
    x_col = sign * forward
    z_col = _unit(np.cross(x_col, back))
    rot_matrix = np.column_stack((x_col, back, z_col))

    try:
        world_palm_rot = Rotation.from_matrix(rot_matrix)
        local_palm_rot = parent_forearm_rot.inv() * world_palm_rot if parent_forearm_rot is not None else world_palm_rot
        results[f"{prefix}Hand"] = local_palm_rot.as_quat().tolist()
    except Exception:
        results[f"{prefix}Hand"] = [0.0, 0.0, 0.0, 1.0]

    finger_definitions = [
        ("Thumb", [("Metacarpal", 1, 2), ("Proximal", 2, 3), ("Distal", 3, 4)], True),
        ("Index", [("Proximal", 5, 6), ("Intermediate", 6, 7), ("Distal", 7, 8)], False),
        ("Middle", [("Proximal", 9, 10), ("Intermediate", 10, 11), ("Distal", 11, 12)], False),
        ("Ring", [("Proximal", 13, 14), ("Intermediate", 14, 15), ("Distal", 15, 16)], False),
        ("Little", [("Proximal", 17, 18), ("Intermediate", 18, 19), ("Distal", 19, 20)], False),
    ]

    for finger_name, segments, is_thumb in finger_definitions:
        prev_dir = forward
        for seg_name, idx_a, idx_b in segments:
            seg_dir = _unit(_point(landmarks, idx_b) - _point(landmarks, idx_a))
            bone_key = f"{prefix}{finger_name}{seg_name}"

            flex_max = 1.4 if is_thumb else 1.6
            flexion = float(np.clip(np.arccos(np.clip(np.dot(seg_dir, prev_dir), -1.0, 1.0)), 0.0, flex_max))

            if is_thumb:
                # Non validé : articulation en selle, axe d'opposition approximatif
                rot_axis = _unit(np.array([0.3, sign * 0.4, 0.8]))
                q = Rotation.from_rotvec(rot_axis * flexion)
            else:
                # Flexion vers la paume (-Y) : rotation autour de Z de signe -sign
                #   gauche (repos +X) : angle négatif ; droite (repos -X) : angle positif
                abduction = 0.0
                if seg_name == "Proximal":  # l'écartement ne se cumule pas sur les phalanges suivantes
                    lateral = float(np.dot(seg_dir, z_col))
                    abduction = float(np.clip(-sign * lateral * 0.4, -0.4, 0.4))
                q = Rotation.from_rotvec(np.array([0.0, abduction, -sign * flexion]))

            results[bone_key] = q.as_quat().tolist()
            prev_dir = seg_dir

    return results


def _finalize_tracks(tracks: dict[str, list[dict[str, Any]]]) -> None:
    """Normalise chaque quaternion et force la continuité de signe (q et -q = même rotation)."""
    for kfs in tracks.values():
        prev: np.ndarray | None = None
        for kf in kfs:
            q = np.asarray(kf["rotation"], dtype=float)
            n = np.linalg.norm(q)
            q = q / n if n > 1e-8 else np.array([0.0, 0.0, 0.0, 1.0])
            if prev is not None and np.dot(prev, q) < 0.0:
                q = -q
            kf["rotation"] = q.tolist()
            prev = q


def process_mimo_landmarks_to_vrma(
    frames: list[dict[str, Any]],
    fps: float = 30.0,
    sign_label: str = "Sign Language Sequence",
    mirror_x: bool = False,
    aspect: float = 1.0,
) -> dict[str, Any]:
    """
    Convertit une séquence de frames MediaPipe en clip VRMA (format JSON Mimo).
    mirror_x / aspect : voir to_vrm_space.
    """
    total_frames = len(frames)
    duration = total_frames / max(fps, 1.0)
    tracks: dict[str, list[dict[str, Any]]] = {}
    prev_normal_l: np.ndarray | None = None
    prev_normal_r: np.ndarray | None = None

    for idx, frame in enumerate(frames):
        t = float(idx) / float(fps)
        body = to_vrm_space(frame.get("bodyPose") or [], mirror_x, aspect)
        hands_l = to_vrm_space(frame.get("handsL") or [], mirror_x, aspect)
        hands_r = to_vrm_space(frame.get("handsR") or [], mirror_x, aspect)

        rotations: dict[str, list[float]] = {}
        forearm_rot_l = None
        forearm_rot_r = None

        if len(body) >= 25:
            # 11/12 épaules G/D, 13/14 coudes, 15/16 poignets
            up_l, low_l, prev_normal_l = solve_arm_kinematics(
                _point(body, 11), _point(body, 13), _point(body, 15), side="left", prev_plane_normal=prev_normal_l
            )
            up_r, low_r, prev_normal_r = solve_arm_kinematics(
                _point(body, 12), _point(body, 14), _point(body, 16), side="right", prev_plane_normal=prev_normal_r
            )

            rotations["leftUpperArm"] = up_l.as_quat().tolist()
            rotations["leftLowerArm"] = low_l.as_quat().tolist()
            rotations["rightUpperArm"] = up_r.as_quat().tolist()
            rotations["rightLowerArm"] = low_r.as_quat().tolist()

            forearm_rot_l = up_l * low_l
            forearm_rot_r = up_r * low_r

            # Tête : rotation en lacet approximative (0 nez, 7/8 oreilles)
            nose = _point(body, 0)
            mid_ears = (_point(body, 7) + _point(body, 8)) * 0.5
            head_dir = _unit(nose - mid_ears)
            head_rot = Rotation.from_rotvec(np.array([0.0, 1.0, 0.0]) * (head_dir[0] * 0.6))
            rotations["head"] = head_rot.as_quat().tolist()

        if hands_l:
            rotations.update(solve_hand_kinematics(hands_l, side="left", parent_forearm_rot=forearm_rot_l))
        if hands_r:
            rotations.update(solve_hand_kinematics(hands_r, side="right", parent_forearm_rot=forearm_rot_r))

        for bone_name, q in rotations.items():
            tracks.setdefault(bone_name, []).append({"time": t, "rotation": q})

    _finalize_tracks(tracks)

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
            "solver": "Anatomical IK Solver v2.1 (axes MediaPipe->VRM)",
        },
    }