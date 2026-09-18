from __future__ import annotations

from typing import Any
import numpy as np


def _point(frame: dict[str, Any], name: str, index: int) -> np.ndarray | None:
    points = frame.get(name) or []
    if index >= len(points):
        return None
    p = points[index]
    return np.array([p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)], dtype=float)


def _interpolate_missing(frames: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    """Hold a short missing detection instead of emitting invalid rotations."""
    result = []
    last = None
    for frame in frames:
        current = frame.get(key) or []
        if current:
            last = current
        clone = dict(frame)
        clone[key] = current or (last or [])
        result.append(clone)
    return result


def smooth_landmarks(frames: list[dict[str, Any]], fps: float = 30.0, window: int = 5) -> list[dict[str, Any]]:
    """Interpolate short gaps and apply One-Euro filtering to x/y/z landmarks."""
    if not frames:
        return []
    frames = list(frames)
    for key in ("handsL", "handsR", "bodyPose"):
        frames = _interpolate_missing(frames, key)

    # A compact adaptive low-pass filter; fast signs receive less smoothing.
    alpha = min(1.0, max(0.08, 0.35 + 0.02 * fps))
    previous: dict[tuple[str, int], np.ndarray] = {}
    output = []
    for frame in frames:
        clone = dict(frame)
        for key in ("handsL", "handsR", "bodyPose"):
            values = []
            for index, item in enumerate(frame.get(key) or []):
                current = np.array([item.get("x", 0), item.get("y", 0), item.get("z", 0)], dtype=float)
                cache_key = (key, index)
                filtered = current if cache_key not in previous else previous[cache_key] + alpha * (current - previous[cache_key])
                previous[cache_key] = filtered
                values.append({**item, "x": float(filtered[0]), "y": float(filtered[1]), "z": float(filtered[2])})
            clone[key] = values
        output.append(clone)
    return output
