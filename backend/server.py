import os
import uuid
from threading import Thread
from typing import Dict, Any

from flask import Flask, jsonify, request, send_file

from backend.pose_estimator import extract_hands
from backend.smoothing import smooth_landmarks
from backend.mixamo.retargeting import convert_hands_to_mixamo_clip
from backend.mixamo.clip_export import export_mixamo_clip

app = Flask(__name__)

TEMP_DIR = os.path.join(os.path.dirname(__file__), "../temp")
EXPORT_DIR = os.path.join(os.path.dirname(__file__), "../exports")

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)


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

    file_id = str(uuid.uuid4())
    src_path = os.path.join(TEMP_DIR, f"{file_id}.mp4")
    uploaded.save(src_path)

    try:
        raw_frames = list(extract_hands(src_path))
        cleaned = smooth_landmarks(raw_frames)
        clip = convert_hands_to_mixamo_clip(cleaned, fps=30)
        output_path = os.path.join(EXPORT_DIR, f"{file_id}.json")
        export_mixamo_clip(clip, output_path)

        return jsonify({
            "clipId": file_id,
            "status": "processed",
            "fps": 30,
            "duration": max(len(cleaned) / 30, 0.0),
            "path": output_path,
            "message": "Animation clip exported successfully."
        })
    except Exception as exc:  # pragma: no cover - runtime safety
        return jsonify({"error": str(exc)}), 500


@app.get("/download/<clip_id>")
def download_clip(clip_id: str):
    path = os.path.join(EXPORT_DIR, f"{clip_id}.json")
    if not os.path.exists(path):
        return jsonify({"error": "Clip not found."}), 404
    return send_file(path, mimetype="application/json")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
