# Mimo

Mimo is a minimal MediaPipe-based pipeline for extracting 2D sign-language motion from video and exporting a machine-usable animation clip for a 3D avatar or Mixamo-style skeleton.

This repository intentionally keeps only the parts that matter for the goal:
- video input
- MediaPipe pose and hand landmark extraction
- smoothing and filtering of noisy frames
- conversion of landmark data into Mixamo-compatible bone rotations
- export to an animation JSON or a machine-readable clip format
- a small HTTP API for app integration

It deliberately avoids Unity, rendering layers, face generation, and unrelated research assets from the original DigiHuman project.

## Goals

- process a 2D sign-language video
- detect body and hand landmarks with MediaPipe
- clean noisy motion
- convert stream data to Mixamo-style bone rotations
- export frames ready for app/mobile consumption

## Project structure

- backend/
  - pose_estimator.py: MediaPipe extraction logic
  - mediapipe_compat.py: compatibility helpers for MediaPipe tasks
  - smoothing.py: temporal smoothing for landmarks
  - retargeting.py: landmark-to-bone conversion logic
  - clip_export.py: export animation clip JSON
  - server.py: small Flask API
- requirements.txt: Python dependencies
- usage.md: step-by-step installation and usage guide

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m backend.server
```

Then upload a video via the API or use the Python functions directly.

## Typical workflow

1. Upload video file
2. Extract body and hand landmarks
3. Smooth each frame temporally
4. Compute bone rotations from key landmarks
5. Export animation clip JSON
6. Send the clip to the mobile app or game runtime

## Notes

This repository is a foundation. The conversion from landmarks to final Mixamo rotations is the critical step for sign-language animation quality, and that logic is implemented as a dedicated retargeting module instead of a Unity scene.
