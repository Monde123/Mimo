# Test suite

Run the tests from the repository root:

```bash
python -m pip install -r requirements.txt
python -m pytest
```

The suite currently checks:

- all hand bones are emitted for valid 21-landmark inputs;
- bent and straight fingers produce different rotations;
- left/right hand naming is not mixed;
- incomplete hands do not create a fake pose;
- body arm rotations are normalized and responsive to movement;
- clip metadata, timestamps, bone names, quality values, and quaternions are valid;
- export is atomic and refuses invalid clips;
- the complete hand retargeting path produces both hands;
- `process_video` and `mimo bvh` write the BVH and its `.report.json`;
- Holistic frames are normalized so `mimo vrma --pipeline holistic` animates arms and fingers;
- the Flask server rejects oversized or non-video uploads, cleans temporary files,
  hides internal paths and errors, and only serves UUID clip ids;
- MediaPipe timestamps restart for each video;
- the synthetic Anipose rig triangulates with 2 to 5 cameras;
- multi-view (`test_multiview.py`):
  - A real 75-point skeleton extracted from `a.mp4` by MediaPipe is projected with
    `cv2.projectPoints` into known distorted cameras. Triangulation recovers it within
    1 mm, and matches `aniposelib`'s own DLT.
  - Confidence weighting, worst-view removal, and the reprojection, behind-camera and
    low-parallax rejections are tested. The low-parallax case is the same video given twice.
  - Frame offsets realign a moving subject.
  - A ChArUco board and a plain checkerboard are rendered by exact homography into two
    known cameras. Calibration recovers the focal length within 1 %, the relative
    rotation within 0.5°, and the 50 cm baseline within 5 mm.
  - The BVH export keeps measured hand and shoulder sizes, and VRMA export works.
  - Invalid calibration, offsets, sizes, units and paths raise clear errors.
- VRM directions (`test_vrm_retargeting.py`): independent forward kinematics
  shows arms up / forward / sideways, a curled finger and a rotated hand land
  exactly where observed, a T-pose gives identity rotations, and missing points
  never become zero rotations.

The multi-view tests need `aniposelib` (`.[multiview]`) and are skipped without it. Tests that read
`a.mp4` use the MediaPipe `.task` models and are skipped if the video is missing.
No test uses a real two-camera recording, because none is available yet.
