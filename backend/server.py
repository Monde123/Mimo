from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory, url_for

LOGGER = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
DEFAULT_MAX_UPLOAD_MB = 200


def _process_video_to_bvh(video_path: Path, output_path: Path) -> dict[str, Any]:
    """Extract landmarks from video and export as BVH using the hybrid pipeline."""
    from backend.process_video import process_video

    return process_video(
        video_path,
        output_path,
        pipeline="hybrid",
        body="upper",
        hands="on",
    )


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
        bvh_path = exports / f"{clip_id}.bvh"
        uploaded.save(src_path)
        try:
            result = _process_video_to_bvh(src_path, bvh_path)
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
            "report": result["report"],
            "downloadUrl": url_for("download_clip", clip_id=clip_id),
            "message": "BVH exported successfully.",
        })

    @app.get("/download/<clip_id>")
    def download_clip(clip_id: str):
        safe_id = _parse_clip_id(clip_id)
        if safe_id is None or not (exports / f"{safe_id}.bvh").is_file():
            return jsonify({"error": "Clip not found."}), 404
        return send_from_directory(exports, f"{safe_id}.bvh", mimetype="application/octet-stream")

    return app


def run_server(host: str = "127.0.0.1", port: int = 5000, debug: bool = False) -> None:
    create_app().run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_server()
