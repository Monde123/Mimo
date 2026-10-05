from __future__ import annotations

from typing import Any

BODY_LANDMARKS = 33


def normalize_frame(frame: dict[str, Any]) -> dict[str, Any]:
    """Return a frame with flat ``bodyPose`` (33 points), ``handsL`` and ``handsR`` lists.

    Holistic yields ``bodyPose`` as ``{"predictions": [...]}`` with extra
    derived points and nests hands under ``handsPose``; the other extractors
    already yield the flat shape.
    """
    result = dict(frame)
    body_pose = result.get("bodyPose")
    if isinstance(body_pose, dict):
        body_pose = body_pose.get("predictions") or []
    result["bodyPose"] = list(body_pose or [])[:BODY_LANDMARKS]
    hands_pose = result.pop("handsPose", None)
    if isinstance(hands_pose, dict):
        result.setdefault("handsL", hands_pose.get("handsL") or [])
        result.setdefault("handsR", hands_pose.get("handsR") or [])
    result["handsL"] = list(result.get("handsL") or [])
    result["handsR"] = list(result.get("handsR") or [])
    return result


def normalize_frames(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [normalize_frame(frame) for frame in frames]
