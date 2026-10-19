from __future__ import annotations

from typing import Any
import numpy as np


MIN_CUTOFF_HZ = 5.0
# Coordonnees ~1 (metres ou image normalisee), pas des pixels : beta=0.007 n'accelere jamais le
# filtre a ces echelles (passe-bas fixe, pire que le brut sur la mocap COMFI bruitee a 1 cm).
BETA = 30.0


def _check_params(fps: float, min_cutoff: float, beta: float) -> None:
    for name, value in (("fps", fps), ("min_cutoff", min_cutoff), ("beta", beta)):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} doit etre un nombre (recu {value!r})")
    if not (np.isfinite(fps) and fps > 0):
        raise ValueError(f"fps doit etre fini et > 0 (recu {fps!r})")
    if not (np.isfinite(min_cutoff) and min_cutoff > 0):
        raise ValueError(f"min_cutoff doit etre fini et > 0 Hz (recu {min_cutoff!r})")
    if not (np.isfinite(beta) and beta >= 0):
        raise ValueError(f"beta doit etre fini et >= 0 (recu {beta!r})")


class OneEuro:
    def __init__(self, fps: float, min_cutoff: float = MIN_CUTOFF_HZ, beta: float = BETA):
        _check_params(fps, min_cutoff, beta)
        self.fps = float(fps)
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
    min_cutoff: float = MIN_CUTOFF_HZ,
    beta: float = BETA,
) -> list[dict[str, Any]]:
    """Interpolate only short gaps, then One-Euro filter coordinates.

    Missing hands are not invented beyond max_gap_frames. Their quality remains
    the raw detection quality so consumers can distinguish real/interpolated data.
    """
    _check_params(fps, min_cutoff, beta)
    if max_gap_frames < 0:
        raise ValueError(f"max_gap_frames doit etre >= 0 (recu {max_gap_frames!r})")
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
