from __future__ import annotations

import math

import numpy as np

from backend.mixamo.body_solver import solve_body


def body_points():
    points = [{"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 1.0} for _ in range(33)]
    points[11] = {"x": -0.3, "y": 1.5, "z": 0.0, "visibility": 1.0}
    points[12] = {"x": 0.3, "y": 1.5, "z": 0.0, "visibility": 1.0}
    points[13] = {"x": -0.6, "y": 1.5, "z": 0.0, "visibility": 1.0}
    points[14] = {"x": 0.6, "y": 1.5, "z": 0.0, "visibility": 1.0}
    points[15] = {"x": -0.9, "y": 1.5, "z": 0.0, "visibility": 1.0}
    points[16] = {"x": 0.9, "y": 1.5, "z": 0.0, "visibility": 1.0}
    return points


def test_body_solver_emits_both_arm_segments():
    result = solve_body(body_points())
    assert set(result) == {
        "mixamorig:LeftArm", "mixamorig:LeftForeArm",
        "mixamorig:RightArm", "mixamorig:RightForeArm",
    }
    for payload in result.values():
        assert math.isclose(np.linalg.norm(payload["rotation"]), 1.0, abs_tol=1e-6)


def test_body_solver_rejects_incomplete_pose():
    assert solve_body(body_points()[:32]) == {}


def test_body_solver_changes_when_arm_direction_changes():
    first = solve_body(body_points())
    points = body_points()
    points[15] = {"x": -0.6, "y": 1.8, "z": 0.0, "visibility": 1.0}
    second = solve_body(points)
    assert not np.allclose(
        first["mixamorig:LeftForeArm"]["rotation"],
        second["mixamorig:LeftForeArm"]["rotation"],
    )
