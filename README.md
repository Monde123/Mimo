# Mimo

Mimo is a minimal MediaPipe-based pipeline for extracting 2D sign-language motion from video and exporting a machine-usable animation clip for a 3D avatar or Mixamo-style skeleton.

The public command-line entry point is `mimo`. Install the repository in
editable mode with `python -m pip install -e .`, then use `mimo --help`.

This repository intentionally keeps only the parts that matter for the goal:
- video input
- MediaPipe pose and hand landmark extraction
- smoothing and filtering of noisy frames
- direct BVH export for inspecting MediaPipe landmarks before retargeting
- optional conversion of landmark data into Mixamo-compatible bone rotations
- export to an animation JSON or a machine-readable clip format
- a small HTTP API for app integration

It deliberately avoids Unity, rendering layers, face generation, and unrelated research assets from the original DigiHuman project.

## Goals

- process a 2D sign-language video
- detect body and hand landmarks with MediaPipe
- clean noisy motion
- compare Holistic hands with the dedicated Hand Landmarker on an upper-body BVH
- convert stream data to Mixamo-style bone rotations
- export frames ready for app/mobile consumption

## Project structure

- backend/
  - pose_estimator.py: MediaPipe extraction logic
  - mediapipe_compat.py: compatibility helpers for MediaPipe tasks
  - smoothing.py: temporal smoothing for landmarks
  - bvh_export.py: direct MediaPipe landmark-to-BVH export
  - hybrid_extractor.py: Holistic body plus dedicated hand extraction
  - parallel_extractor.py: optional YOLOv8 plus ViTPose/MMPose body extraction
  - mixamo/: isolated optional retargeting and JSON export chain
  - server.py: small Flask API
- requirements.txt: Python dependencies
- usage.md: step-by-step installation and usage guide

## Quick start

Create the environment first. The project is validated with Python 3.13 and
supports Python 3.10 through 3.13:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The optional parallel pipeline uses a separate `requirements-parallel.txt`
file because torch and MMPose are heavyweight. ViTPose requires the MMPose
configuration matching its checkpoint; a checkpoint alone is not sufficient.
Hands remain sourced from the MediaPipe Hand Landmarker.

On Windows PowerShell, use `py -3.13 -m venv .venv` and
`.\.venv\Scripts\Activate.ps1`. Then use the BVH commands in
[usage.md](usage.md).

## Inspect MediaPipe as BVH

Pour les commandes courantes, des préréglages évitent de répéter les options :

```bash
python -m backend.process_video input.mp4 signer.bvh --preset sign
python -m backend.process_video input.mp4 corps_complet.bvh --preset full
```

For sign-language footage, inspect the upper body and hands before enabling any
Mixamo retargeting:

```bash
python -m backend.process_video input.mp4 output_holistic.bvh \
  --body upper --hands on --pipeline holistic

python -m backend.process_video input.mp4 output_hybrid.bvh \
  --body upper --hands on --pipeline hybrid
```

Optional parallel body extraction:

```bash
python -m backend.process_video input.mp4 output_parallel.bvh \
  --pipeline parallel --yolo-model yolov8n.pt \
  --vitpose-config vitpose_config.py --vitpose-checkpoint vitpose.pth
```

The optional model files are kept in
`backend/models/parallel/`; the ViTPose configuration files are kept in
`backend/vitPose/`. Copy the YOLO checkpoint and ViTPose checkpoint there;
model weights are ignored by Git. `MIMO_MODEL_DIR` can point to another local
model directory.

The `parallel` pipeline uses YOLOv8 for person detection and ViTPose/MMPose
for the body, while MediaPipe Hand Landmarker supplies the 21 points per hand.
The ViTPose checkpoint must be paired with its matching MMPose config.

The `holistic` pipeline uses the hands produced by Holistic. The `hybrid`
pipeline keeps Holistic for the upper body and uses the dedicated Hand
Landmarker for 21 points per hand. Each BVH has a matching `.report.json`
containing detection coverage, held/interpolated points, coordinate metadata,
and reconstruction warnings. The live server and Mixamo conversion are not
required for this diagnostic workflow.

The BVH pipeline is intentionally independent from `backend/mixamo/`. The
Mixamo chain contains the body/hand solvers, retargeting, rig calibration,
JSON schema/export, and its CLI tools. It is only loaded when explicitly
using a module from that package.

## Typical workflow

1. Upload video file
2. Extract body and hand landmarks
3. Smooth each frame temporally
4. Compute bone rotations from key landmarks
5. Export animation clip JSON
6. Send the clip to the mobile app or game runtime

## Notes

This repository is a foundation. The conversion from landmarks to final Mixamo rotations is the critical step for sign-language animation quality, and that logic is implemented as a dedicated retargeting module instead of a Unity scene.
