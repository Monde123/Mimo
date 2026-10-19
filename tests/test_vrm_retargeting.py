"""Directions physiques du solveur VRM.

Oracle : cinematique directe independante. Les quaternions produits sont appliques
(scipy) aux directions de repos VRM 1.0 (bras / doigts gauches +X, droits -X, Y haut,
avatar face a +Z) et doivent redonner les directions voulues. Les entrees sont
ecrites en convention MediaPipe (y bas, z loin de la camera) a partir de directions
VRM, via (x, -y, -z) -- la meme formule ecrite ici, independamment du module.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from backend.vrm_retargeting import process_mimo_landmarks_to_vrma

X, Y, Z = np.eye(3)
S_L, S_R = np.array([0.18, 1.4, 0.0]), np.array([-0.18, 1.4, 0.0])     # epaules, repere VRM


def _mp(p):
    p = np.asarray(p, dtype=float)
    return {"x": float(p[0]), "y": float(-p[1]), "z": float(-p[2]), "visibility": 1.0}


def _body(left_dirs=(X, X), right_dirs=(-X, -X)):
    pts = [np.array([0.0, 1.6, 0.1])] + [np.array([0.0, 1.6, 0.0])] * 32
    pts[7], pts[8] = np.array([0.07, 1.6, 0.0]), np.array([-0.07, 1.6, 0.0])
    pts[0] = np.array([0.0, 1.6, 0.1])
    for s, e, w, sh, (d1, d2) in ((11, 13, 15, S_L, left_dirs), (12, 14, 16, S_R, right_dirs)):
        pts[s] = sh
        pts[e] = sh + 0.3 * np.asarray(d1, float) / np.linalg.norm(d1)
        pts[w] = pts[e] + 0.25 * np.asarray(d2, float) / np.linalg.norm(d2)
    return [_mp(p) for p in pts]


def _angle(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return math.degrees(math.acos(float(np.clip(c, -1.0, 1.0))))


def _solve(frame):
    clip = process_mimo_landmarks_to_vrma([frame], fps=30.0)
    return {k: Rotation.from_quat(v[0]["rotation"]) for k, v in clip["tracks"].items()}


@pytest.mark.parametrize("name,d", [
    ("haut", Y), ("avant", Z), ("cote", X), ("bas", -Y), ("arriere", -Z),
    ("diagonale", np.array([0.5, 0.6, 0.62])),
])
def test_left_and_right_arm_point_where_observed(name, d):
    d_r = d * np.array([-1, 1, 1])                       # miroir pour le bras droit
    rot = _solve({"bodyPose": _body((d, d), (d_r, d_r))})
    assert _angle(rot["leftUpperArm"].apply(X), d) < 1e-6
    assert _angle((rot["leftUpperArm"] * rot["leftLowerArm"]).apply(X), d) < 1e-6
    assert _angle(rot["rightUpperArm"].apply(-X), d_r) < 1e-6
    assert _angle((rot["rightUpperArm"] * rot["rightLowerArm"]).apply(-X), d_r) < 1e-6


def test_side_arm_is_identity_and_bent_elbow_points_forward():
    rot = _solve({"bodyPose": _body((X, Z), (-X, Z))})
    assert rot["leftUpperArm"].magnitude() < 1e-9            # T-pose = rotation nulle
    assert _angle((rot["leftUpperArm"] * rot["leftLowerArm"]).apply(X), Z) < 1e-6
    assert _angle(rot["leftLowerArm"].apply(X), Z) < 1e-6    # parent = identite
    assert _angle((rot["rightUpperArm"] * rot["rightLowerArm"]).apply(-X), Z) < 1e-6


def test_head_yaw_follows_nose():
    body = _body()
    nose = np.array([0.1, 1.6, 0.0]) + 0.0                   # nez vers la gauche de l'avatar (+X)
    body[0] = _mp(nose)
    rot = _solve({"bodyPose": body})
    assert _angle(rot["head"].apply(Z), X) < 1e-6


# -- mains ---------------------------------------------------------------------
def _hand_rest(side="left"):
    """Main en T-pose VRM, paume vers le bas, doigts tendus le long de +-X, pouce a 45 deg vers +Z."""
    s = 1.0 if side == "left" else -1.0
    pts = np.zeros((21, 3))
    lateral = {5: 0.02, 9: 0.0, 13: -0.02, 17: -0.04}     # majeur sur l'axe : definit l'avant de la paume
    for base, z in lateral.items():
        pts[base] = [s * 0.09, 0.0, z]
        for k in range(1, 4):
            pts[base + k] = pts[base] + [s * 0.03 * k, 0.0, 0.0]
    t = np.array([s, 0.0, 1.0]) / math.sqrt(2.0)
    pts[1] = [s * 0.02, 0.0, 0.02]
    for k in range(2, 5):
        pts[k] = pts[k - 1] + 0.03 * t
    return pts


def _frame_with_hand(hand_vrm, side="left", arm=None):
    rest = X if side == "left" else -X
    arm = arm or (rest, rest)
    wrist = (S_L if side == "left" else S_R) + 0.55 * rest
    left_dirs = arm if side == "left" else (X, X)
    right_dirs = arm if side == "right" else (-X, -X)
    body = _body(left_dirs, right_dirs)
    hand = [_mp(p + wrist) for p in hand_vrm]
    return {"bodyPose": body, "handsL" if side == "left" else "handsR": hand}


@pytest.mark.parametrize("side", ["left", "right"])
def test_rest_hand_gives_identity_everywhere(side):
    rot = _solve(_frame_with_hand(_hand_rest(side), side))
    keys = [k for k in rot if k.startswith(side) and k not in (f"{side}UpperArm", f"{side}LowerArm")]
    assert f"{side}Hand" in keys and len(keys) == 16
    for k in keys:
        assert rot[k].magnitude() < 1e-6, (k, math.degrees(rot[k].magnitude()))


def test_curled_index_bends_toward_palm():
    hand = _hand_rest("left")
    mcp = hand[5]
    hand[6] = mcp + [0.04, 0.0, 0.0]          # proximale : tendue (+X)
    hand[7] = hand[6] + [0.0, -0.03, 0.0]     # intermediaire : vers la paume (-Y, paume en bas)
    hand[8] = hand[7] + [-0.02, 0.0, 0.0]     # distale : revient vers le poignet
    rot = _solve(_frame_with_hand(hand))
    palm = rot["leftHand"]                    # avant-bras = identite (bras en T)
    w1 = palm * rot["leftIndexProximal"]
    w2 = w1 * rot["leftIndexIntermediate"]
    w3 = w2 * rot["leftIndexDistal"]
    assert _angle(w1.apply(X), X) < 1e-6
    assert _angle(w2.apply(X), -Y) < 1e-6
    assert _angle(w3.apply(X), -X) < 1e-6


def test_rotated_arm_and_hand_round_trip():
    G = Rotation.from_euler("zyx", [70, -30, 25], degrees=True)
    hand = G.apply(_hand_rest("left"))
    d = G.apply(X)
    frame = _frame_with_hand(hand, "left", arm=(d, d))
    frame["bodyPose"][15] = _mp(S_L + 0.55 * d)
    frame["handsL"] = [_mp(p + S_L + 0.55 * d) for p in hand]
    rot = _solve(frame)
    forearm = rot["leftUpperArm"] * rot["leftLowerArm"]
    palm_world = forearm * rot["leftHand"]
    assert (palm_world * G.inv()).magnitude() < 1e-6         # paume monde = rotation appliquee
    for finger, idx in (("Index", (5, 6, 7, 8)), ("Little", (17, 18, 19, 20))):
        world = palm_world
        for bone, a, b in zip(("Proximal", "Intermediate", "Distal"), idx[:-1], idx[1:]):
            world = world * rot[f"left{finger}{bone}"]
            assert _angle(world.apply(X), hand[b] - hand[a]) < 1e-6


def test_missing_points_never_become_zero_rotations():
    body = _body()
    body[13] = {"x": float("nan"), "y": float("nan"), "z": float("nan"), "visibility": 0.0}
    hand = _hand_rest("left")
    frame = {"bodyPose": body, "handsL": [_mp(p + S_L + 0.55 * X) for p in hand]}
    frame["handsL"][7] = {"x": None, "y": 0.0, "z": 0.0}
    clip = process_mimo_landmarks_to_vrma([frame], fps=30.0)
    tracks = clip["tracks"]
    assert "leftUpperArm" not in tracks and "leftLowerArm" not in tracks
    assert "leftHand" not in tracks                                # pas de parent connu
    assert "leftIndexProximal" in tracks and "leftIndexIntermediate" not in tracks
    assert "rightUpperArm" in tracks
    for keys in tracks.values():
        q = np.array(keys[0]["rotation"])
        assert np.isfinite(q).all() and abs(np.linalg.norm(q) - 1.0) < 1e-9


def test_short_body_and_bad_fps():
    clip = process_mimo_landmarks_to_vrma([{"bodyPose": _body()[:20]}], fps=30.0)
    assert clip["tracks"] == {}
    for fps in (0.0, -5.0, float("nan")):
        with pytest.raises(ValueError, match="fps"):
            process_mimo_landmarks_to_vrma([], fps=fps)
