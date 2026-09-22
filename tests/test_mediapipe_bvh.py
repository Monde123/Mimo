from pathlib import Path

from backend.bvh_export import export_mediapipe_bvh
from backend.pose_estimator import landmarks_to_array, landmarks_list_to_array


def _body_points():
    return [
        {"x": float(index % 5) * 0.02, "y": float(index // 5) * 0.02,
         "z": float(index) * 0.01, "visibility": 1.0}
        for index in range(33)
    ]


def _hand_points():
    return [
        {"x": float(index) * 0.01, "y": float(index) * 0.01,
         "z": 0.0, "visibility": 1.0}
        for index in range(21)
    ]


def test_landmark_conversion_alias_is_kept():
    assert landmarks_to_array is landmarks_list_to_array


def test_upper_body_bvh_contains_dedicated_hand_chains(tmp_path: Path):
    frames = [
        {"frame": index, "bodyPose": _body_points(),
         "handsL": _hand_points(), "handsR": _hand_points()}
        for index in range(6)
    ]
    output = tmp_path / "signer_upper.bvh"

    report = export_mediapipe_bvh(
        frames, output, fps=24, mode="positions", source="normalized",
        body="upper", hands="on",
    )

    text = output.read_text(encoding="utf-8")
    assert report["body"] == "upper"
    assert report["hands_used"] == ["L", "R"]
    assert "LEFT_HAND_INDEX_FINGER_TIP" in text
    assert "RIGHT_HAND_INDEX_FINGER_TIP" in text
    assert "LEFT_KNEE" not in text
    assert "RIGHT_ANKLE" not in text
