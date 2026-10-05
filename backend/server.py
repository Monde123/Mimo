from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory, url_for

from backend.mixamo.clip_export import export_mixamo_clip

LOGGER = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
DEFAULT_MAX_UPLOAD_MB = 200


def _build_clip(video_path: Path) -> dict[str, Any]:
    from backend.mixamo.process_precise import build_clip

    return build_clip(video_path)


def _parse_clip_id(clip_id: str) -> str | None:
    try:
        parsed = uuid.UUID(clip_id)
    except ValueError:
        return None
    return str(parsed) if str(parsed) == clip_id else None


def create_app(
    temp_dir: str | Path | None = None,
    export_dir: str | Path | None = None,
    max_upload_mb: int | None = None,
) -> Flask:
    app = Flask(__name__)
    if max_upload_mb is None:
        max_upload_mb = int(os.environ.get("MIMO_MAX_UPLOAD_MB", DEFAULT_MAX_UPLOAD_MB))
    app.config["MAX_CONTENT_LENGTH"] = max_upload_mb * 1024 * 1024
    temp = Path(temp_dir or os.environ.get("MIMO_TEMP_DIR") or ROOT_DIR / "temp").resolve()
    exports = Path(export_dir or os.environ.get("MIMO_EXPORT_DIR") or ROOT_DIR / "exports").resolve()
    temp.mkdir(parents=True, exist_ok=True)
    exports.mkdir(parents=True, exist_ok=True)

    @app.errorhandler(413)
    def upload_too_large(_error):
        return jsonify({"error": f"Video too large (limit: {max_upload_mb} MB)."}), 413

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "service": "mimo"})

    @app.post("/process-video")
    def process_video():
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded."}), 400
        uploaded = request.files["file"]
        if not uploaded.filename:
            return jsonify({"error": "Empty filename."}), 400
        suffix = Path(uploaded.filename).suffix.lower()
        if suffix not in ALLOWED_VIDEO_SUFFIXES:
            allowed = ", ".join(sorted(ALLOWED_VIDEO_SUFFIXES))
            return jsonify({"error": f"Unsupported video type (allowed: {allowed})."}), 415

        clip_id = str(uuid.uuid4())
        src_path = temp / f"{clip_id}{suffix}"
        uploaded.save(src_path)
        try:
            clip = _build_clip(src_path)
            export_mixamo_clip(clip, str(exports / f"{clip_id}.json"))
        except (ValueError, FileNotFoundError) as exc:
            return jsonify({"error": f"Video could not be processed: {exc}"}), 422
        except Exception:
            LOGGER.exception("Processing failed for clip %s", clip_id)
            return jsonify({"error": "Internal processing error."}), 500
        finally:
            src_path.unlink(missing_ok=True)

        return jsonify({
            "clipId": clip_id,
            "status": "processed",
            "fps": clip["fps"],
            "duration": clip["duration"],
            "downloadUrl": url_for("download_clip", clip_id=clip_id),
            "message": "Animation clip exported successfully.",
        })

    @app.get("/download/<clip_id>")
    def download_clip(clip_id: str):
        safe_id = _parse_clip_id(clip_id)
        if safe_id is None or not (exports / f"{safe_id}.json").is_file():
            return jsonify({"error": "Clip not found."}), 404
        return send_from_directory(exports, f"{safe_id}.json", mimetype="application/json")

    return app


def run_server(host: str = "127.0.0.1", port: int = 5000, debug: bool = False) -> None:
    create_app().run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_server()
