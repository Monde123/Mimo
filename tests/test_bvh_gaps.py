"""Trous de detection (points rejetes / non vus) dans l'export BVH.

Bug vise : un point absent quelques frames etait fige sur sa derniere position pendant que
ses voisins bougeaient -> os qui pivote brutalement (mesure : 146 deg en une frame sur COMFI).

Oracle : mouvement connu d'avance. On compare l'angle parcouru d'une frame a l'autre par
l'avant-bras dans le BVH (FK independante de test_retarget_bvh_to_mixamo) a l'angle vrai.
Un angle entre deux directions est invariant par le changement de repere de l'export.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from backend.bvh_export import _fill_gaps, export_mediapipe_bvh
from test_retarget_bvh_to_mixamo import _bvh_fk

T = 40
STEP_DEG = 3.0


def _standing_pose() -> np.ndarray:
    """33 points MediaPipe world (metres, y vers le bas), sujet debout bras le long du corps."""
    p = np.zeros((33, 3))
    p[0] = [0.0, -0.65, 0.0]                                   # nez
    for i in range(1, 11):
        p[i] = [0.0, -0.62, -0.02]
    p[11], p[12] = [0.18, -0.45, 0.0], [-0.18, -0.45, 0.0]     # epaules (gauche = +x)
    p[13], p[14] = [0.20, -0.17, 0.0], [-0.20, -0.17, 0.0]     # coudes
    p[15], p[16] = [0.21, 0.08, 0.0], [-0.21, 0.08, 0.0]       # poignets
    for i, j in ((17, 15), (19, 15), (21, 15), (18, 16), (20, 16), (22, 16)):
        p[i] = p[j] + [0.0, 0.07, 0.02 if i in (19, 20) else 0.0] + ([0.02, 0, 0] if i in (17, 18) else 0)
    p[23], p[24] = [0.10, 0.0, 0.0], [-0.10, 0.0, 0.0]         # hanches
    p[25], p[26] = [0.10, 0.42, 0.0], [-0.10, 0.42, 0.0]
    p[27], p[28] = [0.10, 0.82, 0.0], [-0.10, 0.82, 0.0]
    p[29], p[30] = [0.10, 0.86, 0.05], [-0.10, 0.86, 0.05]     # talons
    p[31], p[32] = [0.10, 0.86, -0.12], [-0.10, 0.86, -0.12]   # pointes
    return p


def _moving_arm() -> np.ndarray:
    """Bras gauche : l'humerus tourne de STEP_DEG / frame vers l'avant (autour de x),
    l'avant-bras garde un coude plie de 40 deg. Le reste du corps est immobile."""
    base = _standing_pose()
    out = np.repeat(base[None], T, axis=0)
    sh = base[11]
    up, fore = 0.28, 0.25
    for t in range(T):
        a = math.radians(STEP_DEG * t)
        u = np.array([0.0, math.cos(a), -math.sin(a)])                      # humerus
        f = np.array([0.0, math.cos(a + math.radians(40)), -math.sin(a + math.radians(40))])
        out[t, 13] = sh + up * u
        out[t, 15] = out[t, 13] + fore * f
        for i in (17, 19, 21):
            out[t, i] = out[t, 15] + 0.07 * f + [0.01 * (i - 19), 0, 0]
    return out


def _frames(p3d: np.ndarray) -> list[dict]:
    return [{"frame": t + 1, "bodyPose": [{"x": float(x), "y": float(y), "z": float(z), "visibility": 0.99}
                                          for x, y, z in p3d[t]], "handsL": [], "handsR": []}
            for t in range(len(p3d))]


def _forearm_steps(bvh_path) -> np.ndarray:
    pos = _bvh_fk(bvh_path)
    d = pos["LEFT_WRIST"] - pos["LEFT_ELBOW"]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip((d[1:] * d[:-1]).sum(1), -1, 1)))


def _export(tmp_path, p3d, **kw):
    out = tmp_path / "arm.bvh"
    rep = export_mediapipe_bvh(_frames(p3d), out, 30.0, source="world", aspect=1.0, body="full",
                               hands="off", min_visibility=0.0, **kw)
    return out, rep


def test_short_gap_does_not_make_the_forearm_jump(tmp_path):
    p = _moving_arm()
    p[10:13, 13] = np.nan                                     # coude rejete 3 frames
    out, rep = _export(tmp_path, p)
    steps = _forearm_steps(out)
    # vrai pas = STEP_DEG ; la corde interpolee devie de moins de 1 deg sur 3 frames de 3 deg
    assert np.abs(steps - STEP_DEG).max() < 1.0, np.round(steps, 2)
    assert rep["interpolated_fraction_by_joint"]["LEFT_ELBOW"] == pytest.approx(3 / T)
    assert rep["held_fraction_by_joint"]["LEFT_ELBOW"] == 0.0


def test_long_gap_is_held_and_reported(tmp_path):
    p = _moving_arm()
    p[5:30, 13] = np.nan                                      # 25 frames = 0.83 s > 0.5 s
    out, rep = _export(tmp_path, p)
    assert rep["held_fraction_by_joint"]["LEFT_ELBOW"] == pytest.approx(25 / T)
    assert rep["interpolated_fraction_by_joint"]["LEFT_ELBOW"] == 0.0
    assert any("LEFT_ELBOW" in w for w in rep["warnings"])


def test_no_gap_reports_zero(tmp_path):
    out, rep = _export(tmp_path, _moving_arm())
    assert np.abs(_forearm_steps(out) - STEP_DEG).max() < 0.01
    assert set(rep["interpolated_fraction_by_joint"].values()) == {0.0}


# ---- _fill_gaps : valeurs calculees a la main ------------------------------------------
def _col(vals):
    return np.array([[v, 2 * v, -v] if v is not None else [np.nan] * 3 for v in vals], dtype=float)


def test_fill_gaps_interior_linear():
    a = _col([0, None, None, None, 4])
    f, held, interp = _fill_gaps(a, "x", max_gap=3)
    assert np.allclose(f[:, 0], [0, 1, 2, 3, 4]) and np.allclose(f[:, 1], [0, 2, 4, 6, 8])
    assert (held, interp) == (0.0, 3 / 5)


def test_fill_gaps_too_long_holds_previous():
    a = _col([0, None, None, None, 4])
    f, held, interp = _fill_gaps(a, "x", max_gap=2)
    assert np.allclose(f[:, 0], [0, 0, 0, 0, 4])
    assert (held, interp) == (3 / 5, 0.0)


def test_fill_gaps_edges_hold_nearest_never_extrapolate():
    a = _col([None, None, 5, 6, None])
    f, held, interp = _fill_gaps(a, "x", max_gap=10)
    assert np.allclose(f[:, 0], [5, 5, 5, 6, 6])
    assert (held, interp) == (3 / 5, 0.0)


@pytest.mark.parametrize("vals,max_gap", [([None, None], 3), ([], 3), ([1, None, 2], -1)])
def test_fill_gaps_invalid(vals, max_gap):
    with pytest.raises(ValueError):
        _fill_gaps(_col(vals).reshape(-1, 3), "x", max_gap=max_gap)


def test_fill_gaps_single_value_and_partial_nan():
    f, held, interp = _fill_gaps(_col([7]), "x", max_gap=3)
    assert np.allclose(f, [[7, 14, -7]]) and (held, interp) == (0.0, 0.0)
    a = np.array([[0, 0, 0], [1, np.nan, 1], [2, 2, 2]], float)   # une coordonnee manquante = point manquant
    f, held, interp = _fill_gaps(a, "x", max_gap=3)
    assert np.allclose(f[1], [1, 1, 1]) and interp == pytest.approx(1 / 3)


def test_report_written_with_new_field(tmp_path):
    out, rep = _export(tmp_path, _moving_arm())
    on_disk = json.loads(out.with_suffix(".report.json").read_text(encoding="utf-8"))
    assert on_disk["max_gap_frames"] == 15                      # 0.5 s a 30 fps
    assert "interpolated_fraction_by_joint" in on_disk
