"""Compare two Mimo animation clips (same input video, two pipelines) and
report a per-bone angular error, so process_video.py and process_precise.py
outputs can be quantitatively compared once their schemas match.

Usage:
    python compare_clips.py holistic_output.json precise_output.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _quat_angle_deg(q_a: list[float], q_b: list[float]) -> float:
    """Geodesic angle in degrees between two rotations given as xyzw
    quaternions. abs() handles the double-cover of SO(3) by quaternions
    (q and -q represent the same rotation).
    """
    dot = abs(sum(a * b for a, b in zip(q_a, q_b)))
    dot = min(1.0, dot)
    return float(np.degrees(2 * np.arccos(dot)))


def _align_frames(clip_a: dict[str, Any], clip_b: dict[str, Any]) -> list[tuple[dict, dict]]:
    """Pair up frames by index if counts match, else by nearest timestamp."""
    frames_a, frames_b = clip_a["frames"], clip_b["frames"]
    if len(frames_a) == len(frames_b):
        return list(zip(frames_a, frames_b))

    times_b = np.array([f["time"] for f in frames_b])
    pairs = []
    for frame_a in frames_a:
        nearest_index = int(np.argmin(np.abs(times_b - frame_a["time"])))
        pairs.append((frame_a, frames_b[nearest_index]))
    return pairs


def compare_clips(path_a: str | Path, path_b: str | Path, warn_threshold_deg: float = 15.0) -> dict[str, Any]:
    with open(path_a, "r", encoding="utf-8") as handle:
        clip_a = json.load(handle)
    with open(path_b, "r", encoding="utf-8") as handle:
        clip_b = json.load(handle)

    for label, clip in (("A", clip_a), ("B", clip_b)):
        space = clip.get("coordinateSystem", {}).get("rotationSpace")
        if space != "local":
            raise ValueError(f"Clip {label} is not in local rotation space (got {space!r}); "
                              "angular comparison assumes both clips use the same convention.")

    frame_pairs = _align_frames(clip_a, clip_b)
    per_bone_errors: dict[str, list[float]] = {}

    for frame_a, frame_b in frame_pairs:
        bones_a, bones_b = frame_a["bones"], frame_b["bones"]
        for bone_name in set(bones_a) & set(bones_b):
            angle = _quat_angle_deg(bones_a[bone_name]["rotation"], bones_b[bone_name]["rotation"])
            per_bone_errors.setdefault(bone_name, []).append(angle)

    only_in_a = sorted(set(clip_a["frames"][0]["bones"]) - set(clip_b["frames"][0]["bones"]))
    only_in_b = sorted(set(clip_b["frames"][0]["bones"]) - set(clip_a["frames"][0]["bones"]))

    per_bone_summary = {
        bone: {
            "meanDeg": float(np.mean(errors)),
            "maxDeg": float(np.max(errors)),
            "p95Deg": float(np.percentile(errors, 95)),
            "framesAboveThreshold": int(sum(1 for e in errors if e > warn_threshold_deg)),
        }
        for bone, errors in per_bone_errors.items()
    }

    all_errors = [e for errors in per_bone_errors.values() for e in errors]
    overall = {
        "meanDeg": float(np.mean(all_errors)) if all_errors else None,
        "maxDeg": float(np.max(all_errors)) if all_errors else None,
        "p95Deg": float(np.percentile(all_errors, 95)) if all_errors else None,
        # A simple normalized ratio: mean angular error as a fraction of a
        "errorRatio": float(np.mean(all_errors) / 180.0) if all_errors else None,
        "framePairsCompared": len(frame_pairs),
        "bonesCompared": len(per_bone_errors),
        "bonesOnlyInA": only_in_a,
        "bonesOnlyInB": only_in_b,
    }

    return {"overall": overall, "perBone": per_bone_summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip_a", help="First clip JSON (e.g. process_video.py output)")
    parser.add_argument("clip_b", help="Second clip JSON (e.g. process_precise.py output)")
    parser.add_argument("--threshold-deg", type=float, default=15.0,
                         help="Angle above which a frame is flagged as a meaningful divergence")
    args = parser.parse_args()

    report = compare_clips(args.clip_a, args.clip_b, args.threshold_deg)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
