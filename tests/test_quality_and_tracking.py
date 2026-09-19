from __future__ import annotations

import numpy as np
import pytest

from backend.quality import score_frame, trim_unstable_sequence
from backend.smoothing import smooth_landmarks


def points(count, visibility=1.0, offset=0.0):
    return [{"x": offset + i * 0.01, "y": i * 0.02, "z": 0.01, "visibility": visibility} for i in range(count)]


def frame(good=True, both_hands=True):
    return {
        "frame": 0,
        "bodyPose": points(33) if good else [],
        "handsL": points(21) if good else [],
        "handsR": points(21) if good and both_hands else [],
    }


def test_quality_is_component_specific():
    quality = score_frame(frame(good=True, both_hands=False))
    assert quality["body"] > 0
    assert quality["leftHand"] > 0
    assert quality["rightHand"] == 0
    assert quality["usable"] is True


def test_trim_stops_after_sustained_detection_loss():
    frames = [frame() for _ in range(3)] + [frame(False) for _ in range(4)] + [frame()]
    retained, report = trim_unstable_sequence(frames, max_bad_frames=2)
    assert len(retained) == 5
    assert report["stoppedBecause"] == "sustained_detection_loss"


def test_short_hand_dropout_is_interpolated_and_long_dropout_is_not():
    frames = [frame() for _ in range(2)] + [frame(False) for _ in range(2)] + [frame()]
    result = smooth_landmarks(frames, fps=30, max_gap_frames=2)
    assert len(result[2]["handsL"]) == 21
    long_frames = [frame()] + [frame(False) for _ in range(3)]
    result = smooth_landmarks(long_frames, fps=30, max_gap_frames=2)
    assert result[-1]["handsL"] == []
