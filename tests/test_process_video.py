from __future__ import annotations

import json
from pathlib import Path

import pytest

import backend.process_video as process_video_module
from backend.frames import normalize_frame
from backend.vrm_retargeting import process_mimo_landmarks_to_vrma
from mimo.main import main


def _body_points(offset: float = 0.0, count: int = 33):
    return [
        {"x": float(index % 5) * 0.02 + offset, "y": float(index // 5) * 0.02,
         "z": float(index) * 0.01, "visibility": 1.0}
        for index in range(count)
    ]


def _hand_points(offset: float = 0.0):
    return [
        {"x": float(index) * 0.01 + offset, "y": float(index % 4) * 0.01,
         "z": float(index // 4) * 0.005, "visibility": 1.0}
        for index in range(21)
    ]


def _holistic_frames(count: int = 8):
    """Frames shaped like backend.pose_estimator.extract_holistic output."""
    return [
        {
            "bodyPose": {"predictions": _body_points(i * 0.001, count=37), "frame": i, "width": 640, "height": 480},
            "handsPose": {"handsL": _hand_points(i * 0.001), "handsR": _hand_points(0.3 + i * 0.001), "frame": i},
            "frame": i,
        }
        for i in range(count)
    ]


@pytest.fixture
def fake_holistic_video(monkeypatch, tmp_path: Path) -> Path:
    video = tmp_path / "signer.mp4"
    video.write_bytes(b"not decoded: extraction is stubbed")
    monkeypatch.setattr(process_video_module, "_probe_video", lambda path: (24.0, 640.0, 480.0))
    monkeypatch.setattr(process_video_module, "_select_extractor",
                        lambda pipeline, *args, **kwargs: (lambda path: _holistic_frames()))
    return video


def test_normalize_frame_flattens_holistic_shape():
    frame = normalize_frame(_holistic_frames(1)[0])
    assert len(frame["bodyPose"]) == 33
    assert len(frame["handsL"]) == 21 and len(frame["handsR"]) == 21
    assert "handsPose" not in frame


def test_normalize_frame_keeps_flat_shape():
    flat = {"frame": 1, "bodyPose": _body_points(), "handsL": [], "handsR": _hand_points()}
    frame = normalize_frame(flat)
    assert frame["bodyPose"] == flat["bodyPose"]
    assert frame["handsL"] == [] and frame["handsR"] == flat["handsR"]


def test_process_video_writes_bvh_and_report(fake_holistic_video: Path, tmp_path: Path):
    output = tmp_path / "out" / "signer.bvh"
    result = process_video_module.process_video(
        fake_holistic_video, output, source="normalized", body="upper", hands="on", pipeline="holistic",
    )
    assert result["status"] == "complete"
    assert output.is_file()
    text = output.read_text(encoding="utf-8")
    assert text.startswith("HIERARCHY")
    assert "Frames: 8" in text
    report = json.loads(output.with_suffix(".report.json").read_text(encoding="utf-8"))
    assert report["pipeline"]["pipeline_name"] == "holistic"
    assert report["hands_used"] == ["L", "R"]


def test_cli_bvh_writes_file(fake_holistic_video: Path, tmp_path: Path, capsys):
    output = tmp_path / "cli.bvh"
    code = main(["bvh", str(fake_holistic_video), str(output), "--preset", "full",
                 "--source", "normalized", "--body", "upper", "--hands", "on"])
    assert code == 0
    assert output.is_file()
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["output"] == str(output)


def test_extract_frames_feeds_vrma_arms_and_fingers_for_holistic(fake_holistic_video: Path):
    frames, fps = process_video_module.extract_frames(fake_holistic_video, "holistic")
    assert fps == 24.0
    clip = process_mimo_landmarks_to_vrma(frames, fps=fps)
    bones = set(clip["tracks"])
    assert {"leftUpperArm", "leftLowerArm", "rightUpperArm", "rightLowerArm"} <= bones
    assert any(name.startswith("leftIndex") for name in bones)
    assert any(name.startswith("rightIndex") for name in bones)


def test_cli_vrma_holistic_animates_body_and_hands(fake_holistic_video: Path, tmp_path: Path):
    output = tmp_path / "clip.vrma.json"
    assert main(["vrma", str(fake_holistic_video), str(output), "--pipeline", "holistic"]) == 0
    clip = json.loads(output.read_text(encoding="utf-8"))
    assert "leftUpperArm" in clip["tracks"]
    assert any(name.startswith("rightIndex") for name in clip["tracks"])


def test_unknown_pipeline_is_rejected():
    with pytest.raises(ValueError, match="pipeline"):
        process_video_module._select_extractor("unknown")
