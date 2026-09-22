"""Post-retargeting despike filter, adapted from SAM3DBody-cpp's
gmr_retarget.py despike_frames() to Mimo's clip schema (bones as
{"rotation": [x,y,z,w], "position": [x,y,z]?} per frame, keyed by
mixamorig:* name, instead of GMR's {joint: [pos, quat_wxyz]} tuples).

Where trim_unstable_sequence() DROPS bad frames and smooth_landmarks()
continuously smooths everything, this targets isolated tracking
singularities (a rotation flip, a lost-track spike) that survive both:
a frame is flagged only if SOME bone's rotation, or the root position,
jumps faster than a plausible human movement between two frames -- then
that frame (or run of frames) is replaced by slerp/lerp interpolation
between the nearest clean neighbours, not dropped and not blurred.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT_BONE = "mixamorig:Hips"


def _quat_angle_deg(q_a: list[float], q_b: list[float]) -> float:
    dot = min(1.0, abs(sum(a * b for a, b in zip(q_a, q_b))))
    return float(np.degrees(2 * np.arccos(dot)))


def _slerp(q_a: list[float], q_b: list[float], t: float) -> list[float]:
    rotations = Rotation.from_quat([q_a, q_b])
    return Slerp([0.0, 1.0], rotations)([t])[0].as_quat().tolist()


def _frame_jump(frame_a: dict, frame_b: dict, bone_deg_threshold: float, pos_m_threshold: float) -> bool:
    """True if ANY bone's rotation, or the root's position, jumps more than
    a human plausibly moves between two consecutive frames.
    """
    bones_a, bones_b = frame_a["bones"], frame_b["bones"]
    for bone_name in set(bones_a) & set(bones_b):
        if _quat_angle_deg(bones_a[bone_name]["rotation"], bones_b[bone_name]["rotation"]) > bone_deg_threshold:
            return True
    if ROOT_BONE in bones_a and "position" in bones_a[ROOT_BONE] and "position" in bones_b.get(ROOT_BONE, {}):
        pos_a = np.asarray(bones_a[ROOT_BONE]["position"])
        pos_b = np.asarray(bones_b[ROOT_BONE]["position"])
        if float(np.linalg.norm(pos_b - pos_a)) > pos_m_threshold:
            return True
    return False


def _interpolate_frame(target: dict, lo: dict, hi: dict, alpha: float) -> dict:
    new_bones: dict[str, Any] = {}
    for bone_name in set(lo["bones"]) & set(hi["bones"]):
        lo_bone, hi_bone = lo["bones"][bone_name], hi["bones"][bone_name]
        entry: dict[str, Any] = {"rotation": _slerp(lo_bone["rotation"], hi_bone["rotation"], alpha),
                                  "quality": min(lo_bone.get("quality", 1.0), hi_bone.get("quality", 1.0))}
        if "position" in lo_bone and "position" in hi_bone:
            lo_pos, hi_pos = np.asarray(lo_bone["position"]), np.asarray(hi_bone["position"])
            entry["position"] = ((1 - alpha) * lo_pos + alpha * hi_pos).tolist()
        new_bones[bone_name] = entry
    result = dict(target)
    result["bones"] = new_bones
    result["despiked"] = True
    return result


def despike_clip(frames: list[dict[str, Any]], bone_deg_threshold: float = 45.0,
                  pos_m_threshold: float = 0.30) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Returns (new_frames, report). New_frames is a shallow-copied list;
    despiked entries are replaced, everything else is untouched. Frames
    with no clean neighbour on either side (a glitch run touching the
    very start or end of the clip) are clamped to the nearest clean frame
    instead of interpolated, same as the reference implementation.
    """
    n = len(frames)
    if n < 3:
        return list(frames), {"framesReplaced": 0, "threshold": {"boneDeg": bone_deg_threshold, "posM": pos_m_threshold}}

    jump_before = [False] + [
        _frame_jump(frames[i - 1], frames[i], bone_deg_threshold, pos_m_threshold) for i in range(1, n)
    ]
    bad = [
        (i > 0 and jump_before[i]) or (i < n - 1 and jump_before[i + 1])
        for i in range(n)
    ]

    result = list(frames)
    replaced = 0
    i = 0
    while i < n:
        if not bad[i]:
            i += 1
            continue
        span_start = i
        while i < n and bad[i]:
            i += 1
        span_end = i - 1

        lo = span_start - 1
        while lo >= 0 and bad[lo]:
            lo -= 1
        hi = span_end + 1
        while hi < n and bad[hi]:
            hi += 1

        lo_ok, hi_ok = lo >= 0, hi < n
        if not lo_ok and not hi_ok:
            continue  # entire clip flagged -- leave it alone rather than guess

        for t in range(span_start, span_end + 1):
            if not (lo_ok and hi_ok):
                source = frames[hi if hi_ok else lo]
                result[t] = {**frames[t], "bones": source["bones"], "despiked": True}
            else:
                alpha = (t - lo) / (hi - lo)
                result[t] = _interpolate_frame(frames[t], frames[lo], frames[hi], alpha)
            replaced += 1

    return result, {
        "framesReplaced": replaced,
        "threshold": {"boneDeg": bone_deg_threshold, "posM": pos_m_threshold},
    }
