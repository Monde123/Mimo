from __future__ import annotations

from typing import Any
import math
import numpy as np


def _visibility(points: list[dict[str, Any]]) -> float:
    if not points:
        return 0.0
    values = [float(p.get("visibility", 1.0)) for p in points]
    values = [v for v in values if math.isfinite(v)]
    return float(np.clip(np.mean(values), 0.0, 1.0)) if values else 0.0


def _geometric_quality(points: list[dict[str, Any]], minimum: int) -> float:
    if len(points) < minimum:
        return 0.0
    values = np.asarray([[p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)] for p in points], dtype=float)
    if not np.isfinite(values).all():
        return 0.0
    if np.max(np.linalg.norm(values[1:] - values[:-1], axis=1)) < 1e-8:
        return 0.15
    return 1.0


def component_quality(points: list[dict[str, Any]], expected: int) -> float:
    if not points:
        return 0.0
    completeness = min(len(points) / expected, 1.0)
    return float(completeness * _visibility(points) * _geometric_quality(points, expected))


def score_frame(frame: dict[str, Any]) -> dict[str, float | bool]:
    body = component_quality(frame.get("bodyPose") or [], 33)
    left = component_quality(frame.get("handsL") or [], 21)
    right = component_quality(frame.get("handsR") or [], 21)
    # A signing frame is usable if the body or at least one hand is credible.
    usable = bool(body >= 0.35 or left >= 0.45 or right >= 0.45)
    overall = max(body, (left + right) / 2.0, left, right)
    return {"body": body, "leftHand": left, "rightHand": right, "overall": overall, "usable": usable}


def annotate_quality(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for frame in frames:
        clone = dict(frame)
        clone["detectionQuality"] = score_frame(frame)
        result.append(clone)
    return result


def trim_unstable_sequence(
    frames: list[dict[str, Any]],
    max_bad_frames: int = 10,
    min_good_frames: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Trim unusable edges and stop after a sustained tracking failure.

    A short dropout is retained for interpolation; a sustained dropout ends the
    clip instead of freezing the avatar indefinitely.
    """
    annotated = annotate_quality(frames)
    good_indices = [i for i, frame in enumerate(annotated) if frame["detectionQuality"]["usable"]]
    if len(good_indices) < min_good_frames:
        raise ValueError("The video does not contain enough usable body/hand detections.")
    start = good_indices[0]
    selected = []
    bad_run = 0
    for frame in annotated[start:]:
        if frame["detectionQuality"]["usable"]:
            bad_run = 0
        else:
            bad_run += 1
            if bad_run > max_bad_frames:
                break
        selected.append(frame)
    if not selected:
        raise ValueError("Tracking was unstable for the complete input sequence.")
    report = {
        "inputFrames": len(frames),
        "retainedFrames": len(selected),
        "firstFrame": int(selected[0].get("frame", start)),
        "stoppedBecause": "sustained_detection_loss" if len(selected) < len(annotated) - start else None,
        "badFrameBudget": max_bad_frames,
        "quality": [frame["detectionQuality"] for frame in selected],
    }
    return selected, report
