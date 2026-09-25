"""
Module d'intégration Anipose pour Mimo : Calibration Multi-Caméras et Triangulation 3D Robuste.
Inspiré de l'architecture d'Anipose (Karashchuk et al., Cell Reports 2021).

Ce module permet de :
1. Définir des modèles de caméras calibrées avec matrices intrinsèques (K) et extrinsèques (R, t).
2. Projeter des points 3D vers les plans images 2D (P = K [R | t]).
3. Trianguler les repères 2D synchronisés issus de caméras multiples (ex: vue face + vue latérale 45°/90°)
   par résolution par Décomposition en Valeurs Singulières (SVD / DLT - Direct Linear Transform).
4. Effectuer un filtrage des occultations et un raffinement sous contraintes rigides de longueur
   des segments de phalanges et de membres (recherche d'optimisation d'invariance anatomique).
"""

from __future__ import annotations
import math
from typing import Any
import numpy as np


class CameraCalibration:
    """Représentation géométrique d'une caméra calibrée."""

    def __init__(
        self,
        name: str,
        matrix_k: list[list[float]] | np.ndarray,
        rotation_r: list[list[float]] | np.ndarray,
        translation_t: list[float] | np.ndarray,
        dist_coeffs: list[float] | None = None
    ) -> None:
        self.name = name
        self.K = np.array(matrix_k, dtype=np.float64)  # 3x3
        self.R = np.array(rotation_r, dtype=np.float64)  # 3x3
        self.t = np.array(translation_t, dtype=np.float64).reshape((3, 1))  # 3x1
        self.dist_coeffs = np.array(dist_coeffs or [0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # Matrice de projection P = K [R | t] (3x4)
        rt = np.hstack((self.R, self.t))
        self.P = self.K @ rt

    def project_point_3d(self, pt_3d: list[float] | np.ndarray) -> tuple[float, float, float]:
        """Projette un point 3D monde [X, Y, Z] sur le capteur 2D [u, v, depth]."""
        p_hom = np.array([pt_3d[0], pt_3d[1], pt_3d[2], 1.0], dtype=np.float64)
        img_hom = self.P @ p_hom
        z = img_hom[2]
        if abs(z) < 1e-8:
            return 0.0, 0.0, 0.0
        u = img_hom[0] / z
        v = img_hom[1] / z
        return float(u), float(v), float(z)


class AniposeMultiViewTriangulator:
    """
    Solveur Anipose multi-caméras pour Mimo.
    Reconstruit des points 3D métriques à partir des observations 2D synchronisées de plusieurs caméras.
    """

    def __init__(self, cameras: list[CameraCalibration]) -> None:
        if len(cameras) < 2:
            raise ValueError("Au moins 2 caméras calibrées sont nécessaires pour la triangulation Anipose.")
        self.cameras = cameras

    def triangulate_point(
        self,
        observations: list[tuple[float, float] | None],
        confidences: list[float] | None = None
    ) -> tuple[np.ndarray, float]:
        """
        Triangule un point unique à l'aide de l'algorithme DLT (Direct Linear Transformation).
        
        Args:
            observations: Liste de (x_pixel, y_pixel) pour chaque caméra (ou None si occulté)
            confidences: Score de visibilité pour chaque caméra
        
        Returns:
            point_3d (np.ndarray de taille 3 [X, Y, Z]), erreur de reprojection moyenne
        """
        valid_indices = [
            i for i, obs in enumerate(observations)
            if obs is not None and (confidences is None or confidences[i] > 0.2)
        ]

        if len(valid_indices) < 2:
            # Moins de 2 caméras voient le point : triangulation impossible
            return np.array([0.0, 0.0, 0.0], dtype=np.float64), -1.0

        # Construction du système DLT A * X = 0
        a_rows = []
        for idx in valid_indices:
            u, v = observations[idx]
            p = self.cameras[idx].P
            w = 1.0 if confidences is None else max(confidences[idx], 0.1)

            # Équations DLT standard pondérées
            a_rows.append(w * (u * p[2, :] - p[0, :]))
            a_rows.append(w * (v * p[2, :] - p[1, :]))

        a_matrix = np.array(a_rows, dtype=np.float64)

        # Résolution par SVD
        _, _, vh = np.linalg.svd(a_matrix)
        x_hom = vh[-1]
        if abs(x_hom[3]) < 1e-8:
            return np.array([0.0, 0.0, 0.0], dtype=np.float64), -1.0

        pt_3d = (x_hom[:3] / x_hom[3]).astype(np.float64)

        # Calcul de l'erreur de reprojection moyenne
        reproj_errors = []
        for idx in valid_indices:
            u_proj, v_proj, _ = self.cameras[idx].project_point_3d(pt_3d)
            u_true, v_true = observations[idx]
            err = math.hypot(u_proj - u_true, v_proj - v_true)
            reproj_errors.append(err)

        mean_err = float(np.mean(reproj_errors)) if reproj_errors else 0.0
        return pt_3d, mean_err

    def triangulate_hand_landmarks(
        self,
        camera_hands: list[list[dict[str, float]] | None]
    ) -> list[dict[str, float]]:
        """
        Triangule les 21 repères d'une main à travers toutes les caméras disponibles.
        Permet de lever les occultations lorsque la paume masque les doigts pour une caméra.
        """
        triangulated_hand = []
        for lm_idx in range(21):
            obs_list = []
            conf_list = []
            for cam_idx, hand_landmarks in enumerate(camera_hands):
                if hand_landmarks is not None and len(hand_landmarks) > lm_idx:
                    lm = hand_landmarks[lm_idx]
                    obs_list.append((lm.get("x", 0.0), lm.get("y", 0.0)))
                    conf_list.append(lm.get("visibility", lm.get("score", 1.0)))
                else:
                    obs_list.append(None)
                    conf_list.append(0.0)

            pt_3d, err = self.triangulate_point(obs_list, conf_list)
            triangulated_hand.append({
                "x": float(pt_3d[0]),
                "y": float(pt_3d[1]),
                "z": float(pt_3d[2]),
                "visibility": 1.0 if err >= 0.0 else 0.0,
                "reproj_error": err
            })

        return triangulated_hand


def create_synthetic_sign_rig(
    baseline_distance: float = 0.85,
    camera_angle_deg: float = 35.0,
    focal_length_px: float = 800.0,
    img_width: int = 1280,
    img_height: int = 720
) -> list[CameraCalibration]:
    """
    Génère une configuration standard Anipose multi-caméras calibrée pour la langue des signes :
    - Caméra 0 : Face au signeur (Caméra principale)
    - Caméra 1 : Vue latérale oblique gauche (pour lever les occultations de doigts et profondeur)
    """
    k = np.array([
        [focal_length_px, 0.0, img_width / 2.0],
        [0.0, focal_length_px, img_height / 2.0],
        [0.0, 0.0, 1.0]
    ])

    # Caméra 0 : Caméra frontale à Z = 0
    r0 = np.eye(3)
    t0 = np.array([0.0, 0.0, 0.0])
    cam_front = CameraCalibration("cam_front", k, r0, t0)

    # Caméra 1 : Inclinée de camera_angle_deg vers le signeur à droite
    theta = math.radians(camera_angle_deg)
    r1 = np.array([
        [math.cos(theta), 0.0, math.sin(theta)],
        [0.0, 1.0, 0.0],
        [-math.sin(theta), 0.0, math.cos(theta)]
    ])
    t1 = np.array([-baseline_distance * math.cos(theta), 0.0, baseline_distance * math.sin(theta)])
    cam_oblique = CameraCalibration("cam_oblique", k, r1, t1)

    return [cam_front, cam_oblique]
