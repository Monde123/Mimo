# Mimo

Mimo is a minimal MediaPipe-based pipeline for extracting 2D sign-language motion from video and exporting a machine-usable animation clip for a 3D avatar or Mixamo-style skeleton.

The public command-line entry point is `mimo`. Install the repository in
editable mode with `python -m pip install -e .`, then use `mimo --help`.
Optional extras: `.[server]` for `mimo server`, `.[glb]` for the GLB/VRM
baking tools, `.[parallel]` for YOLOv8 + ViTPose, `.[multiview]` for
`mimo calibrate` / `mimo multiview` (2+ synchronized cameras triangulated with
Anipose calibration into one BVH / VRMA), `.[dev]` for the tests.

This repository intentionally keeps only the parts that matter for the goal:
- video input
- MediaPipe pose and hand landmark extraction (Holistic, Hybrid, Pose+Hands)
- smoothing and filtering of noisy frames (OneEuro filter)
- direct BVH export for inspecting MediaPipe landmarks in Blender before retargeting
- **VRM Animation (.vrma / JSON) export**: Standard Humanoid OpenXR/VRM 1.0 animation format for sign language with 21 3D finger joints per hand
- optional conversion of landmark data into Mixamo-compatible bone rotations
- export to an animation JSON or a machine-readable clip format
- Three.js + @pixiv/three-vrm web application for real-time sign language avatar playback
- a small HTTP API for app integration

It deliberately avoids Unity, rendering layers, face generation, and unrelated research assets from the original DigiHuman project.

## Commandes

```powershell
python -m pip install -e .[glb]          # + .[multiview], .[server], .[dev] selon les besoins

# Vidéo -> BVH (intermédiaire inspectable dans Blender) + rapport .report.json
mimo bvh a.mp4 out\a_sign.bvh --preset sign

# BVH -> clip Mixamo (quaternions glTF par os) pour un modèle .glb
mimo retarget out\a_sign.bvh clara.glb -o out\a_sign.mixamo.json
# ... ou directement le modèle animé, jouable dans three.js / Blender
mimo retarget out\a_sign.bvh clara.glb -o out\clara_a_sign.glb
mimo bake-mixamo out\a_sign.mixamo.json clara.glb -o out\clara_a_sign.glb

# Vidéo -> clip VRM Animation, puis injection dans un avatar VRM
mimo vrma a.mp4 out\a.vrma                # binaire, pour three-vrm-animation
mimo vrma a.mp4 out\a.vrma.json           # JSON, pour bake-vrm
mimo bake-vrm avatar.vrm out\a.vrma.json out\avatar_anime.vrm

# 2 caméras synchronisées ou plus -> BVH / VRMA triangulé
mimo calibrate cam0_board.mp4 cam1_board.mp4 -o calibration.toml `
  --squares-x 7 --squares-y 5 --square-length-m 0.06 --marker-length-m 0.045
mimo multiview cam0.mp4 cam1.mp4 -o out\signe.bvh --calibration calibration.toml

mimo server                               # API HTTP locale
mimo <commande> --help                    # toutes les options
```

`mimo retarget` recale la pose de repos du modèle sur celle du BVH. Sur `clara.glb`,
sans ce recalage, le bras s'écarte de 7° et le pouce de plus de 20°. Avec, chaque os
pointe comme dans le BVH. `--no-align-rest` désactive ce recalage. La translation du
bassin n'est pas transférée. Détails et format du JSON :
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

## Inspect MediaPipe as BVH or Export VRMA Clip

### 1. Export VRM Animation (Recommandé pour avatars 3D & Langue des Signes)

Pour animer directement un avatar VRM (VRM 0.x ou 1.0) dans le navigateur ou Unity avec les articulations complètes des 10 doigts :

```bash
mimo vrma input.mp4 animation_sign.vrma.json --pipeline hybrid
```

Ce fichier JSON standardise :
- La hiérarchie Humanoid (buste, cou, tête, bras, avant-bras, mains).
- Les 15 os par main (pouce, index, majeur, annulaire, auriculaire avec métacarpes et phalanges).
- L'espace de rotation local et quaternions normalisés pour `@pixiv/three-vrm`.

### 2. Export BVH pour inspection dans Blender

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

The dedicated Pose + Hand Landmarker combination is also available as a BVH
pipeline:

```bash
mimo bvh input.mp4 output_pose_hands.bvh \
  --pipeline pose_hands --body upper --hands on
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

1. `mimo bvh` : extract body and hand landmarks, smooth them, compute BVH rotations
2. Inspect the BVH in Blender and read its `.report.json` (coverage, warnings)
3. `mimo retarget` : replay the BVH on the Mixamo model (`.glb`)
4. Load the animated `.glb` in the app (three.js `GLTFLoader` + `AnimationMixer`), or the
   JSON clip (one `QuaternionKeyframeTrack` per bone)

## Notes

This repository is a foundation. The conversion from landmarks to final Mixamo rotations is the critical step for sign-language animation quality, and that logic is implemented as a dedicated retargeting module instead of a Unity scene.
