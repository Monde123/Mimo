from __future__ import annotations

import math

import numpy as np
import pytest

from backend.mixamo.hand_solver import solve_hand


def straight_hand():
    # A synthetic palm and five straight fingers. Each segment is collinear,
    # so the Kalidokit-style flexion angle should be approximately zero.
    points = [None] * 21
    points[0] = {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 1.0}
    bases = {1: (-0.18, 0.08), 5: (-0.10, 0.22), 9: (0.0, 0.25), 13: (0.10, 0.22), 17: (0.18, 0.16)}
    for base, (x, y) in bases.items():
        for offset in range(4):
            points[base + offset] = {"x": x, "y": y + offset * 0.12, "z": 0.0, "visibility": 1.0}
    return points


def bent_index_hand():
    points = straight_hand()
    points[6] = {"x": -0.02, "y": 0.30, "z": 0.0, "visibility": 1.0}
    points[7] = {"x": 0.06, "y": 0.28, "z": 0.0, "visibility": 1.0}
    points[8] = {"x": 0.12, "y": 0.20, "z": 0.0, "visibility": 1.0}
    return points


def test_straight_hand_emits_full_hand_and_fingers():
    result = solve_hand(straight_hand(), "right")
    assert len(result) == 16
    assert "mixamorig:RightHand" in result
    assert "mixamorig:RightHandIndex1" in result
    for payload in result.values():
        quaternion = payload["rotation"]
        assert len(quaternion) == 4
        assert math.isclose(float(np.linalg.norm(quaternion)), 1.0, abs_tol=1e-6)


def test_bent_index_differs_from_straight_index():
    straight = solve_hand(straight_hand(), "right")
    bent = solve_hand(bent_index_hand(), "right")
    assert not np.allclose(
        straight["mixamorig:RightHandIndex2"]["rotation"],
        bent["mixamorig:RightHandIndex2"]["rotation"],
    )


def test_left_and_right_use_distinct_bone_names():
    left = solve_hand(straight_hand(), "left")
    assert "mixamorig:LeftHand" in left
    assert all("Right" not in name for name in left)


def test_invalid_hand_is_rejected_without_fake_pose():
    assert solve_hand(straight_hand()[:20], "right") == {}


def test_invalid_side_is_explicitly_rejected():
    with pytest.raises(ValueError):
        solve_hand(straight_hand(), "front")
