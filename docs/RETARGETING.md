# Mimo hand/body retargeting

## Justification

The solver follows the decomposition used by Kalidokit without copying its TypeScript implementation:

1. The wrist, index MCP, and pinky MCP define a palm plane.
2. Each finger articulation is computed from three consecutive landmarks.
3. Thumb and non-thumb fingers have separate limits.
4. Left/right signs are represented by explicit bone names.
5. Body segments are solved as swings from a canonical axis.
6. The resulting quaternions are exported as local-track data and validated.

This is not a claim that monocular MediaPipe recovers invisible twist. The calibration file must define the target rig's axes, and the output should be evaluated with real sign-language clips.

## Current command

```bash
python -m backend.process_video input.mp4 output.json --rig configs/mixamo_default.json
```

The current stage provides a principled baseline. The next required improvement is parent-relative composition across the full frame sequence: body and hand rotations must be composed against the previous parent track and the rig bind correction, rather than treated as independent world swings.
