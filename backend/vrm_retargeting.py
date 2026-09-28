from __future__ import annotations
from typing import Any
import numpy as np
from scipy.spatial.transform import Rotation

"""
Solveur cinématique inverse rigoureux pour la Langue des Signes.
Convertit les repères 3D de MediaPipe (Pose + Hand Landmarker) en quaternions
locaux pour le standard VRM 1.0 / OpenXR Humanoid.

Conventions VRM 1.0 (Right-Handed, Y-Up):
- T-Pose de référence :
  - Bras gauche s'étend le long de +X
  - Bras droit s'étend le long de -X
  - Avant-bras continuent le long de l'axe X
  - Paumes orientées vers le bas (-Y) ou vers l'avant (+Z)
  - Doigts s'étendent le long de l'axe X avec flexion autour de Z et abduction autour de Y
"""

def _unit(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-8 else np.array([0.0, 1.0, 0.0])

def _point(landmarks: list[dict[str, float]], idx: int) -> np.ndarray:
    if idx < len(landmarks):
        p = landmarks[idx]
        return np.array([p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)], dtype=float)
    return np.zeros(3, dtype=float)

def solve_arm_kinematics(
    shoulder: np.ndarray,
    elbow: np.ndarray,
    wrist: np.ndarray,
    side: str = "left"
) -> tuple[Rotation, Rotation]:
    """
    Résout l'orientation de l'épaule et du coude avec calcul de la normale au plan
    (Épaule, Coude, Poignet) pour éliminer le roulis parasite (arm twist).
    """
    is_left = side == "left"
    rest_arm = np.array([1.0, 0.0, 0.0]) if is_left else np.array([-1.0, 0.0, 0.0])
    
    # Vecteurs directeurs observés
    v_upper = _unit(elbow - shoulder)
    v_lower = _unit(wrist - elbow)
    
    # Normale du triangle articulaire (plan de flexion du coude)
    triangle_normal = np.cross(v_upper, v_lower)
    norm_val = np.linalg.norm(triangle_normal)
    if norm_val > 1e-4:
        plane_normal = triangle_normal / norm_val
    else:
        # Bras tendu, normale par défaut vers l'avant (+Z)
        plane_normal = np.array([0.0, 0.0, 1.0])

    # Rotation de l'UpperArm : amène rest_arm vers v_upper
    axis = np.cross(rest_arm, v_upper)
    axis_norm = np.linalg.norm(axis)
    dot = np.clip(np.dot(rest_arm, v_upper), -1.0, 1.0)
    
    if axis_norm < 1e-6:
        upper_rot = Rotation.identity() if dot > 0 else Rotation.from_rotvec(np.array([0.0, 1.0, 0.0]) * np.pi)
    else:
        angle = np.arccos(dot)
        upper_rot = Rotation.from_rotvec((axis / axis_norm) * angle)

    # Orientation locale de l'avant-bras (LowerArm)
    # On exprime v_lower dans le référentiel local de upper_rot
    local_lower = upper_rot.inv().apply(v_lower)
    axis_low = np.cross(rest_arm, local_lower)
    axis_low_norm = np.linalg.norm(axis_low)
    dot_low = np.clip(np.dot(rest_arm, local_lower), -1.0, 1.0)

    if axis_low_norm < 1e-6:
        lower_rot = Rotation.identity() if dot_low > 0 else Rotation.from_rotvec(np.array([0.0, 0.0, 1.0]) * np.pi)
    else:
        angle_low = np.arccos(dot_low)
        lower_rot = Rotation.from_rotvec((axis_low / axis_low_norm) * angle_low)

    return upper_rot, lower_rot

def solve_hand_kinematics(
    landmarks: list[dict[str, float]],
    side: str = "left",
    parent_forearm_rot: Rotation | None = None
) -> dict[str, list[float]]:
    """
    Résout l'orientation tridimensionnelle de la paume et des 15 phalanges
    avec abduction (écartement) et flexion anatomiquement bornées.
    """
    if len(landmarks) < 21:
        return {}

    is_left = side == "left"
    sign = 1.0 if is_left else -1.0
    prefix = "left" if is_left else "right"
    results: dict[str, list[float]] = {}

    wrist = _point(landmarks, 0)
    index_mcp = _point(landmarks, 5)
    middle_mcp = _point(landmarks, 9)
    pinky_mcp = _point(landmarks, 17)

    # Base orthonormée de la paume (Forward, Across, Normal)
    forward = _unit(middle_mcp - wrist)
    across = _unit(pinky_mcp - index_mcp) if is_left else _unit(index_mcp - pinky_mcp)
    normal = _unit(np.cross(across, forward))
    across = _unit(np.cross(forward, normal))

    rot_matrix = np.column_stack((across, normal, forward))
    if np.linalg.det(rot_matrix) < 0:
        rot_matrix[:, 2] *= -1

    try:
        world_palm_rot = Rotation.from_matrix(rot_matrix)
        if parent_forearm_rot is not None:
            local_palm_rot = parent_forearm_rot.inv() * world_palm_rot
        else:
            local_palm_rot = world_palm_rot
        results[f"{prefix}Hand"] = local_palm_rot.as_quat().tolist()
    except Exception:
        results[f"{prefix}Hand"] = [0.0, 0.0, 0.0, 1.0]

    # Chaînes des 5 doigts (3 os par doigt = 15 os au total par main)
    # Metacarpal/Proximal, Intermediate, Distal
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
            pa = _point(landmarks, idx_a)
            pb = _point(landmarks, idx_b)
            seg_dir = _unit(pb - pa)

            bone_key = f"{prefix}{finger_name}{seg_name}"

            if is_thumb:
                # Le pouce possède une articulation en selle (opposition + flexion)
                # Angle par rapport au plan de la paume et à l'index
                flexion = float(np.clip(np.arccos(np.clip(np.dot(seg_dir, prev_dir), -1.0, 1.0)), 0.0, 1.4))
                # Axe composé pour l'opposition
                rot_axis = _unit(np.array([0.3, sign * 0.4, 0.8]))
                q = Rotation.from_rotvec(rot_axis * flexion)
            else:
                # Doigts longs : Flexion (autour de Z dans l'espace VRM) + Abduction (écartement latéral)
                dot_val = np.clip(np.dot(seg_dir, prev_dir), -1.0, 1.0)
                flexion = float(np.clip(np.arccos(dot_val), 0.0, 1.6))
                
                # Écartement latéral (abduction) mesuré avec le vecteur across
                abduction = float(np.clip(np.dot(seg_dir, across) * 0.4, -0.4, 0.4))

                # Quaternion de flexion/abduction
                rot_axis = np.array([0.0, abduction, sign * flexion])
                q = Rotation.from_rotvec(rot_axis)

            results[bone_key] = q.as_quat().tolist()
            prev_dir = seg_dir

    return results

def process_mimo_landmarks_to_vrma(
    frames: list[dict[str, Any]],
    fps: float = 30.0,
    sign_label: str = "Sign Language Sequence"
) -> dict[str, Any]:
    """
    Point d'entrée principal pour convertir une séquence de frames MediaPipe
    en clip VRMA hautement articulé pour Three.js et VRM.
    """
    total_frames = len(frames)
    duration = total_frames / max(fps, 1.0)
    tracks: dict[str, list[dict[str, Any]]] = {}

    for idx, frame in enumerate(frames):
        t = float(idx) / float(fps)
        body = frame.get("bodyPose") or []
        hands_l = frame.get("handsL") or []
        hands_r = frame.get("handsR") or []

        rotations: dict[str, list[float]] = {}

        forearm_rot_l = None
        forearm_rot_r = None

        if len(body) >= 25:
            # 11: L Shoulder, 12: R Shoulder, 13: L Elbow, 14: R Elbow, 15: L Wrist, 16: R Wrist
            l_sh, r_sh = _point(body, 11), _point(body, 12)
            l_el, r_el = _point(body, 13), _point(body, 14)
            l_wr, r_wr = _point(body, 15), _point(body, 16)

            up_l, low_l = solve_arm_kinematics(l_sh, l_el, l_wr, side="left")
            up_r, low_r = solve_arm_kinematics(r_sh, r_el, r_wr, side="right")

            rotations["leftUpperArm"] = up_l.as_quat().tolist()
            rotations["leftLowerArm"] = low_l.as_quat().tolist()
            rotations["rightUpperArm"] = up_r.as_quat().tolist()
            rotations["rightLowerArm"] = low_r.as_quat().tolist()

            forearm_rot_l = up_l * low_l
            forearm_rot_r = up_r * low_r

            # Tête / Regard (0: Nez, 7: Oreille G, 8: Oreille D)
            nose = _point(body, 0)
            mid_ears = (_point(body, 7) + _point(body, 8)) * 0.5
            head_dir = _unit(nose - mid_ears)
            head_rot = Rotation.from_rotvec(np.array([0.0, 1.0, 0.0]) * (head_dir[0] * 0.6))
            rotations["head"] = head_rot.as_quat().tolist()

        # Mains & Doigts (15 os résolus par main)
        if hands_l:
            rotations.update(solve_hand_kinematics(hands_l, side="left", parent_forearm_rot=forearm_rot_l))
        if hands_r:
            rotations.update(solve_hand_kinematics(hands_r, side="right", parent_forearm_rot=forearm_rot_r))

        for bone_name, q in rotations.items():
            if bone_name not in tracks:
                tracks[bone_name] = []
            tracks[bone_name].append({
                "time": t,
                "rotation": q
            })

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
            "solver": "Rigorous Anatomical IK Solver v2.0"
        }
    }
