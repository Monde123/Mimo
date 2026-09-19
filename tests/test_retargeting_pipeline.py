from __future__ import annotations

import numpy as np

from backend.hand_solver import solve_hand
from backend.retargeting import convert_frames_to_mixamo_clip


def hand_points():
    points = [{"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 1.0} for _ in range(21)]
    for base in (1, 5, 9, 13, 17):
        for offset in range(4):
            points[base + offset] = {
                "x": (base - 10) / 50,
                "y": offset / 10 + 0.1,
                "z": 0.0,
                "visibility": 1.0,
            }
    return points


def test_pipeline_emits_both_hands_and_quality():
    clip = convert_frames_to_mixamo_clip(
        [{"bodyPose": [], "handsL": hand_points(), "handsR": hand_points()}], fps=24
    )
    frame = clip["frames"][0]
    assert frame["quality"]["leftHand"] == 1.0
    assert frame["quality"]["rightHand"] == 1.0
    assert "mixamorig:LeftHandIndex1" in frame["bones"]
    assert "mixamorig:RightHandPinky3" in frame["bones"]
    for payload in frame["bones"].values():
        assert np.isclose(np.linalg.norm(payload["rotation"]), 1.0, atol=1e-6)
