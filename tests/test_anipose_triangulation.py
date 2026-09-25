"""
Tests unitaires pour le module Anipose multi-caméras de Mimo.
Vérifie la projection géométrique P = K[R|t], la triangulation SVD (DLT) et la résolution des occultations.
"""

from __future__ import annotations
import math
import numpy as np

from backend.anipose_triangulation import (
    CameraCalibration,
    AniposeMultiViewTriangulator,
    create_synthetic_sign_rig,
)


def test_camera_projection():
    cameras = create_synthetic_sign_rig()
    cam_front = cameras[0]

    # Point 3D situé à 1.5 mètre devant la caméra au centre
    p_3d = np.array([0.0, 0.0, 1.5])
    u, v, z = cam_front.project_point_3d(p_3d)

    # Doit être projeté exactement au centre optique (1280/2 = 640, 720/2 = 360)
    assert math.isclose(u, 640.0, abs_tol=1e-3)
    assert math.isclose(v, 360.0, abs_tol=1e-3)
    assert math.isclose(z, 1.5, abs_tol=1e-3)


def test_anipose_triangulation_precision():
    cameras = create_synthetic_sign_rig(baseline_distance=0.85, camera_angle_deg=35.0)
    triangulator = AniposeMultiViewTriangulator(cameras)

    # Point cible représentant par exemple le bout d'un index en LSF
    target_3d = np.array([0.15, -0.05, 1.40])

    # Projection sur les 2 caméras
    u0, v0, _ = cameras[0].project_point_3d(target_3d)
    u1, v1, _ = cameras[1].project_point_3d(target_3d)

    observations = [(u0, v0), (u1, v1)]
    reconstructed_3d, err = triangulator.triangulate_point(observations)

    # L'erreur de reconstruction doit être inférieure à 1 millimètre
    dist = np.linalg.norm(reconstructed_3d - target_3d)
    assert dist < 1e-4, f"Erreur de reconstruction trop élevée : {dist}"
    assert err < 1e-3


def test_anipose_occlusion_handling():
    cameras = create_synthetic_sign_rig()
    triangulator = AniposeMultiViewTriangulator(cameras)

    # Test quand une caméra a une occultation totale (None)
    observations = [(640.0, 360.0), None]
    pt_3d, err = triangulator.triangulate_point(observations)

    # Ne doit pas planter, mais indiquer l'impossibilité de trianguler sans 2 caméras
    assert err == -1.0
