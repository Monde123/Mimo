from __future__ import annotations

import math

import pytest

from backend.mixamo.clip_validator import validate_clip
from backend.mixamo.clip_schema import clip_metadata
from backend.mixamo.clip_export import export_mixamo_clip
from backend.mixamo.retargeting import convert_frames_to_mixamo_clip


def valid_frame(index=0):
    return {
        "frame": index,
        "time": index / 30.0,
        "bones": {
            "mixamorig:RightHand": {
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "quality": 1.0,
            }
        },
        "quality": {"body": 0.0, "leftHand": 0.0, "rightHand": 1.0},
    }


def valid_clip():
    clip = clip_metadata()
    clip.update({"fps": 30.0, "duration": 1 / 30.0, "frames": [valid_frame()]})
    return clip


def test_validator_accepts_valid_clip():
    validate_clip(valid_clip())


def test_validator_rejects_non_normalized_quaternion():
    clip = valid_clip()
    clip["frames"][0]["bones"]["mixamorig:RightHand"]["rotation"] = [0, 0, 0, 2]
    with pytest.raises(ValueError, match="normalized"):
        validate_clip(clip)


def test_validator_rejects_non_monotonic_time():
    clip = valid_clip()
    clip["frames"].append(valid_frame(1))
    clip["frames"][1]["time"] = 0.0
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_clip(clip)


def test_export_writes_validated_json(tmp_path):
    output = tmp_path / "clip.json"
    assert export_mixamo_clip(valid_clip(), str(output)) == str(output)
    assert output.exists()


def test_retargeting_returns_versioned_clip_for_empty_frames():
    clip = convert_frames_to_mixamo_clip([{"bodyPose": [], "handsL": [], "handsR": []}], fps=30)
    assert clip["format"] == "mimo.mixamo.animation"
    assert clip["version"] == 1
    assert clip["frames"][0]["frame"] == 0
