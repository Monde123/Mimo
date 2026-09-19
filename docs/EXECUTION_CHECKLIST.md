# Mimo execution checklist

## Local checks

```bash
python -m pip install -r requirements.txt
python -m pytest
python -m compileall backend
```

## Process a real video

```bash
python -m backend.process_video input.mp4 output.json --rig configs/mixamo_default.json
```

Required model:

```text
backend/models/holistic_landmarker.task
```

## Quality behavior

- Each frame receives body, left-hand, right-hand, overall, and usable quality scores.
- A frame is usable when the body or at least one hand is credible.
- Short gaps are held for interpolation only up to `max_gap_frames`.
- A prolonged loss of body and both hands ends the retained animation.
- The final JSON contains the processing report and per-frame quality metadata.

## Efficiency rules

- MediaPipe is loaded once per video, not once per frame.
- Frames are decoded once and processed in memory in the current offline mode.
- Short gaps are filtered without inventing indefinitely long hand tracks.
- JSON is validated before atomic publication.
- Use a smaller output FPS only when the application accepts reduced temporal precision.

The pipeline currently requires an actual `.task` model and a real video for end-to-end execution. Unit tests do not require model files.
