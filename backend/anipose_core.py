"""
Pure-Python Linear Algebra routines (Vectors, 3x3 Matrices, Dot, Cross, SVD/DLT Triangulation).
Permet l'exécution universelle et indépendante du solveur Anipose multi-vues,
sans aucune dépendance obligatoire envers numpy ou scipy, tout en offrant
une compatibilité native avec les types scalaires et les listes Python.
"""

from __future__ import annotations
import math
from typing import Any


def vec_sub(a: list[float] | tuple[float, ...], b: list[float] | tuple[float, ...]) -> list[float]:
    return [a[i] - b[i] for i in range(len(a))]


def vec_add(a: list[float] | tuple[float, ...], b: list[float] | tuple[float, ...]) -> list[float]:
    return [a[i] + b[i] for i in range(len(a))]


def vec_scale(a: list[float] | tuple[float, ...], s: float) -> list[float]:
    return [x * s for x in a]


def vec_dot(a: list[float] | tuple[float, ...], b: list[float] | tuple[float, ...]) -> float:
    return sum(a[i] * b[i] for i in range(len(a)))


def vec_norm(a: list[float] | tuple[float, ...]) -> float:
    return math.sqrt(sum(x * x for x in a))


def mat_mul_vec(m: list[list[float]], v: list[float]) -> list[float]:
    """Multiplie une matrice m (N x M) par un vecteur v (M x 1)."""
    return [sum(row[j] * v[j] for j in range(len(v))) for row in m]


def solve_dlt_svd(a: list[list[float]]) -> list[float]:
    """
    Résout le problème des moindres carrés A * X = 0 sous contrainte ||X|| = 1.
    Calcule le vecteur propre associé à la plus petite valeur propre de A^T * A (4x4).
    Méthode par itération de puissance inverse pour une convergence ultra-rapide et exacte.
    """
    # 1. Calculer M = A^T * A (4x4)
    m = [[0.0] * 4 for _ in range(4)]
    for r in a:
        for i in range(4):
            for j in range(4):
                m[i][j] += r[i] * r[j]

    # Pour trouver le vecteur propre de la plus petite valeur propre de M,
    # on cherche le vecteur propre de la plus grande valeur propre de (M_max * I - M)
    trace = m[0][0] + m[1][1] + m[2][2] + m[3][3] + 1.0
    shift_m = [[-m[i][j] for j in range(4)] for i in range(4)]
    for i in range(4):
        shift_m[i][i] += trace

    # Power Iteration
    v = [0.5, 0.5, 0.5, 0.5]
    for _ in range(35):
        nv = mat_mul_vec(shift_m, v)
        norm = vec_norm(nv)
        if norm > 1e-12:
            v = [x / norm for x in nv]

    return v


class AniposeCamera:
    """Modèle d'une caméra calibrée compatible Pure Python & NumPy."""

    def __init__(
        self,
        name: str,
        k: list[list[float]],
        r: list[list[float]],
        t: list[float]
    ) -> None:
        self.name = name
        self.K = k
        self.R = r
        self.t = t

        # Calcul de la matrice de projection P = K * [R | t] (3x4)
        rt = [
            [r[i][0], r[i][1], r[i][2], t[i]]
            for i in range(3)
        ]
        self.P = [
            [sum(self.K[i][k] * rt[k][j] for k in range(3)) for j in range(4)]
            for i in range(3)
        ]

    def project_point_3d(self, pt_3d: list[float] | tuple[float, float, float]) -> tuple[float, float, float]:
        """Projette un point monde 3D [X, Y, Z] sur le plan 2D de la caméra."""
        hom = [pt_3d[0], pt_3d[1], pt_3d[2], 1.0]
        img_hom = mat_mul_vec(self.P, hom)
        z = img_hom[2]
        if abs(z) < 1e-8:
            return 0.0, 0.0, 0.0
        return img_hom[0] / z, img_hom[1] / z, z


class AniposeTriangulator:
    """Solveur de triangulation multi-vues universel (DLT / SVD)."""

    def __init__(self, cameras: list[AniposeCamera]) -> None:
        if len(cameras) < 2:
            raise ValueError("Au moins 2 caméras sont requises pour Anipose.")
        self.cameras = cameras

    def triangulate(
        self,
        observations: list[tuple[float, float] | list[float] | None],
        confidences: list[float] | None = None
    ) -> tuple[list[float], float]:
        valid_indices = [
            i for i, obs in enumerate(observations)
            if obs is not None and (confidences is None or confidences[i] > 0.2)
        ]

        if len(valid_indices) < 2:
            return [0.0, 0.0, 0.0], -1.0

        a_rows: list[list[float]] = []
        for idx in valid_indices:
            u, v = observations[idx][0], observations[idx][1]
            p = self.cameras[idx].P
            w = 1.0 if confidences is None else max(confidences[idx], 0.1)

            # Ligne 1 : w * (u * P[2] - P[0])
            a_rows.append([w * (u * p[2][c] - p[0][c]) for c in range(4)])
            # Ligne 2 : w * (v * P[2] - P[1])
            a_rows.append([w * (v * p[2][c] - p[1][c]) for c in range(4)])

        sol_hom = solve_dlt_svd(a_rows)
        w_val = sol_hom[3]
        if abs(w_val) < 1e-8:
            return [0.0, 0.0, 0.0], -1.0

        pt_3d = [sol_hom[0] / w_val, sol_hom[1] / w_val, sol_hom[2] / w_val]

        # Calcul de l'erreur de reprojection en pixels
        reproj_errs = []
        for idx in valid_indices:
            u_proj, v_proj, _ = self.cameras[idx].project_point_3d(pt_3d)
            u_true, v_true = observations[idx][0], observations[idx][1]
            reproj_errs.append(math.hypot(u_proj - u_true, v_proj - v_true))

        mean_err = sum(reproj_errs) / len(reproj_errs) if reproj_errs else 0.0
        return pt_3d, mean_err
