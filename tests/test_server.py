from __future__ import annotations

import io
import uuid
from pathlib import Path

import pytest

import backend.server as server_module
from backend.mixamo.retargeting import convert_frames_to_mixamo_clip
from mimo.main import _build_parser


def _hand_points():
    points = [{"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 1.0} for _ in range(21)]
    for base in (1, 5, 9, 13, 17):
        for offset in range(4):
            points[base + offset] = {"x": (base - 10) / 50, "y": offset / 10 + 0.1, "z": 0.0, "visibility": 1.0}
    return points


def _valid_clip():
    return convert_frames_to_mixamo_clip(
        [{"bodyPose": [], "handsL": _hand_points(), "handsR": _hand_points()}] * 3, fps=30
    )


@pytest.fixture
def dirs(tmp_path: Path):
    return tmp_path / "temp", tmp_path / "exports"


@pytest.fixture
def client(dirs, monkeypatch):
    monkeypatch.setattr(server_module, "_build_clip", lambda path: _valid_clip())
    app = server_module.create_app(temp_dir=dirs[0], export_dir=dirs[1], max_upload_mb=1)
    return app.test_client()


def _upload(client, name="sign.mp4", payload=b"video-bytes"):
    return client.post("/process-video", data={"file": (io.BytesIO(payload), name)},
                       content_type="multipart/form-data")


def test_health(client):
    assert client.get("/health").get_json()["status"] == "ok"


def test_upload_exports_clip_without_leaking_paths_and_cleans_temp(client, dirs):
    response = _upload(client)
    assert response.status_code == 200
    body = response.get_json()
    assert set(body) == {"clipId", "status", "fps", "duration", "downloadUrl", "message"}
    assert body["downloadUrl"] == f"/download/{body['clipId']}"
    assert str(dirs[1]) not in response.get_data(as_text=True)
    assert list(dirs[0].iterdir()) == []
    download = client.get(body["downloadUrl"])
    assert download.status_code == 200
    assert download.get_json()["fps"] == 30.0


def test_rejects_non_video_extension(client, dirs):
    assert _upload(client, name="payload.exe").status_code == 415
    assert list(dirs[0].iterdir()) == []


def test_rejects_oversized_upload(client):
    assert _upload(client, payload=b"0" * (2 * 1024 * 1024)).status_code == 413


def test_unusable_video_is_reported_as_422(client, monkeypatch, dirs):
    def no_detection(path):
        raise ValueError("The video does not contain enough usable body/hand detections.")
    monkeypatch.setattr(server_module, "_build_clip", no_detection)
    response = _upload(client)
    assert response.status_code == 422
    assert "usable" in response.get_json()["error"]
    assert list(dirs[0].iterdir()) == []


def test_internal_errors_do_not_leak_details(client, monkeypatch):
    def crash(path):
        raise RuntimeError(r"secret C:\internal\path")
    monkeypatch.setattr(server_module, "_build_clip", crash)
    response = _upload(client)
    assert response.status_code == 500
    assert "secret" not in response.get_data(as_text=True)


@pytest.mark.parametrize("clip_id", ["..%5C..%5Cpyproject", "not-a-uuid", str(uuid.uuid4()).upper()])
def test_download_rejects_non_canonical_ids(client, dirs, clip_id):
    (dirs[1].parent / "pyproject.json").write_text("{}", encoding="utf-8")
    assert client.get(f"/download/{clip_id}").status_code == 404


def test_server_cli_defaults_are_local_and_not_debug():
    args = _build_parser().parse_args(["server"])
    assert (args.host, args.port, args.debug) == ("127.0.0.1", 5000, False)
