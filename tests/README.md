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
- the complete hand retargeting path produces both hands.

These tests are deterministic unit tests and do not require MediaPipe model files or a real video. A separate integration test should be added once representative sign-language videos and `.task` models are available.
