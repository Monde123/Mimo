# Mimo

Mimo is a minimal MediaPipe-based pipeline for extracting 2D sign-language motion from video and exporting a machine-usable animation clip for a 3D avatar or Mixamo-style skeleton.

The public command-line entry point is `mimo`. Install the repository in
editable mode with `python -m pip install -e .`, then use `mimo --help`.
Optional extras: `.[server]` for `mimo server`, `.[glb]` for the GLB/VRM
baking tools, `.[multiview]` for
`mimo calibrate` / `mimo multiview` (2+ synchronized cameras triangulated with
Anipose calibration into one BVH / VRMA), `.[dev]` for the tests.

This repository intentionally keeps only the parts that matter for the goal:
- video input
- MediaPipe pose and hand landmark extraction (Holistic, Hybrid, Pose+Hands)
- smoothing and filtering of noisy frames (OneEuro filter)
- direct BVH export for inspecting MediaPipe landmarks in Blender before retargeting
- **VRM Animation (.vrma / JSON) export**: Standard Humanoid OpenXR/VRM 1.0 animation format for sign language, with 21 3D finger joints per hand
- optional conversion of landmark data into Mixamo-compatible bone rotations
- export to an animation JSON or a machine-readable clip format
- Three.js + @pixiv/three-vrm web application for real-time sign language avatar playback
- a small HTTP API for app integration

It deliberately avoids Unity, rendering layers, face generation, and unrelated research assets from the original DigiHuman project.

## Get Started

### 1. Installation

Windows PowerShell:

```powershell
git clone https://github.com/Monde123/Mimo.git
cd Mimo
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
git clone https://github.com/Monde123/Mimo.git
cd Mimo
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check the active interpreter:

```bash
python --version
python -c "import cv2, mediapipe, numpy, scipy; print('Mimo runtime: OK')"
```

### 2. Add a 3D model

Download a Mixamo 3D model (`.glb`) from this link: https://drive.google.com/file/d/171Rs1GwlNxBaAvxrSpIthHE55fr-R4O4/view?usp=drive_link

You can also get a model from mixamo.com and convert it from `.fbx` to `.glb` with Blender or any other tool.

Then place `model3D.glb` in the `mimo/` folder.

## Commands

```powershell
python -m pip install -e .[glb]          # + .[multiview], .[server], .[dev] as needed

# All in one command: video(s) -> BVH (kept alongside) -> animated Mixamo .glb
mimo animate a.mp4 --model clara.glb -o out\clara_a.glb --preset sign
mimo animate cam0.mp4 cam1.mp4 --calibration calibration.toml --model clara.glb -o out\clara_mv.glb --body full

# Video -> BVH (intermediate you can inspect in Blender) + .report.json report
mimo bvh a.mp4 out\a_sign.bvh --preset sign

# BVH -> Mixamo clip (glTF quaternions per bone) for a .glb model
mimo retarget out\a_sign.bvh clara.glb -o out\a_sign.mixamo.json
# ... or directly the animated model, playable in three.js / Blender
mimo retarget out\a_sign.bvh clara.glb -o out\clara_a_sign.glb
mimo bake-mixamo out\a_sign.mixamo.json clara.glb -o out\clara_a_sign.glb

# Video -> VRM Animation clip, then injection into a VRM avatar
mimo vrma a.mp4 out\a.vrma                # binary, for three-vrm-animation
mimo vrma a.mp4 out\a.vrma.json           # JSON, for bake-vrm
mimo bake-vrm avatar.vrm out\a.vrma.json out\avatar_anime.vrm

# 2+ synchronized cameras -> triangulated BVH / VRMA
mimo calibrate cam0_board.mp4 cam1_board.mp4 -o calibration.toml `
  --squares-x 7 --squares-y 5 --square-length-m 0.06 --marker-length-m 0.045
mimo multiview cam0.mp4 cam1.mp4 -o out\signe.bvh --calibration calibration.toml

mimo server                               # local HTTP API
mimo <command> --help                     # all options
```

`mimo retarget` aligns the model's rest pose to the BVH's. On `clara.glb`,
without this alignment, the arm drifts by 7° and the thumb by more than 20°. With it,
every bone points the same way as in the BVH. `--no-align-rest` disables this alignment.
Pelvis translation is not transferred. Details and JSON format:
[usage.md, section 7](usage.md#7-retargeting-bvh--mixamo).

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
  - retarget_bvh_to_mixamo.py: BVH -> Mixamo glTF clip (`mimo retarget`)
  - bake_mixamo_animation.py: clip -> animated Mixamo `.glb` (`mimo bake-mixamo`, `mimo retarget -o x.glb`)
  - multiview.py: calibration and multi-camera triangulation (`mimo calibrate`, `mimo multiview`)
  - vrm_retargeting.py: MediaPipe -> VRM Animation (`mimo vrma`)
  - mixamo/: older direct video -> Mixamo JSON chain (`mimo mixamo`)
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

## Inspect MediaPipe as BVH or Export a VRMA Clip

### 1. Export VRM Animation (recommended for 3D avatars and sign language)

To animate a VRM avatar (VRM 0.x or 1.0) directly in the browser or in Unity, with full articulation of all 10 fingers:

```bash
mimo vrma input.mp4 animation_sign.vrma.json --pipeline hybrid
```

This JSON file standardizes:
- the Humanoid hierarchy (chest, neck, head, arms, forearms, hands);
- the 15 bones per hand (thumb, index, middle, ring, pinky, with metacarpals and phalanges);
- the local rotation space and normalized quaternions for `@pixiv/three-vrm`.

### 2. BVH export for inspection in Blender

For common commands, presets avoid repeating options:

```bash
python -m backend.process_video input.mp4 signer.bvh --preset sign
python -m backend.process_video input.mp4 full_body.bvh --preset full
```

For sign-language footage, inspect the upper body and hands before enabling any
Mixamo retargeting:

```bash
python -m backend.process_video input.mp4 output_holistic.bvh \
  --body upper --hands on --pipeline holistic

python -m backend.process_video input.mp4 output_hybrid.bvh \
  --body upper --hands on --pipeline hybrid
```

The dedicated Pose + Hand Landmarker combination is also available as a BVH
pipeline:

```bash
mimo bvh input.mp4 output_pose_hands.bvh \
  --pipeline pose_hands --body upper --hands on
```

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

1. `mimo bvh`: extract body and hand landmarks, smooth them, compute BVH rotations
2. Inspect the BVH in Blender and read its `.report.json` (coverage, warnings)
3. `mimo retarget`: replay the BVH on the Mixamo model (`.glb`)
4. Load the animated `.glb` in the app (three.js `GLTFLoader` + `AnimationMixer`), or the
   JSON clip (one `QuaternionKeyframeTrack` per bone)

## Notes

This repository is a foundation. The conversion from landmarks to final Mixamo rotations is the critical step for sign-language animation quality, and that logic is implemented as a dedicated retargeting module instead of a Unity scene.