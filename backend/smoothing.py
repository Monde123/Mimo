from __future__ import annotations

from typing import Any
import numpy as np


class OneEuro:
    def __init__(self, fps: float, min_cutoff: float = 1.0, beta: float = 0.007):
        self.fps = max(float(fps), 1e-6)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.previous = None
        self.derivative = None

    def __call__(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=float)
        if self.previous is None:
            self.previous = value.copy()
            self.derivative = np.zeros_like(value)
            return value
        dt = 1.0 / self.fps
        raw_derivative = (value - self.previous) / dt
        derivative_alpha = self._alpha(1.0)
        self.derivative = derivative_alpha * raw_derivative + (1 - derivative_alpha) * self.derivative
        cutoff = self.min_cutoff + self.beta * np.abs(self.derivative)
        alpha = self._alpha(cutoff)
        filtered = alpha * value + (1 - alpha) * self.previous
        self.previous = filtered
        return filtered

    def _alpha(self, cutoff):
        tau = 1.0 / (2.0 * np.pi * np.asarray(cutoff))
        return 1.0 / (1.0 + tau / (1.0 / self.fps))


def _valid_points(points: list[dict[str, Any]], expected: int) -> bool:
    return len(points) == expected and all(
        np.isfinite([p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)]).all()
        for p in points
    )


def smooth_landmarks(
    frames: list[dict[str, Any]],
    fps: float = 30.0,
    max_gap_frames: int = 5,
    min_cutoff: float = 1.0,
    beta: float = 0.007,
) -> list[dict[str, Any]]:
    """Interpolate only short gaps, then One-Euro filter coordinates.

    Missing hands are not invented beyond max_gap_frames. Their quality remains
    the raw detection quality so consumers can distinguish real/interpolated data.
    """
    if not frames:
        return []
    keys = (("bodyPose", 33), ("handsL", 21), ("handsR", 21))
    result = [dict(frame) for frame in frames]
    for key, expected in keys:
        last_valid = None
        missing = 0
        for index, frame in enumerate(result):
            points = frame.get(key) or []
            if _valid_points(points, expected):
                last_valid = points
                missing = 0
            elif last_valid is not None and missing < max_gap_frames:
                frame[key] = last_valid
                frame.setdefault("interpolated", []).append(key)
                missing += 1
            else:
                missing += 1
                frame[key] = []

    filters: dict[tuple[str, int], OneEuro] = {}
    for frame in result:
        for key, expected in keys:
            smoothed = []
            for index, point in enumerate(frame.get(key) or []):
                filter_key = (key, index)
                filters.setdefault(filter_key, OneEuro(fps, min_cutoff, beta))
                vector = filters[filter_key](np.array([point["x"], point["y"], point["z"]], dtype=float))
                smoothed.append({**point, "x": float(vector[0]), "y": float(vector[1]), "z": float(vector[2])})
            frame[key] = smoothed
    return result
