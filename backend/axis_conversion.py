"""The MediaPipe -> engine axis conversion, applied ONCE to every landmark
at extraction time, before body_solver/hand_solver ever see the data.

Value: (x, -y, -z) -- X unchanged, Y and Z negated.

Confirmed by two independent, unrelated reference implementations that both
target a right-handed engine convention (unlike DigiHuman/Unity, which is
left-handed and needs (-x, -y, z) instead -- do not copy that one):

  - Nor-s/mediapipe-to-mixamo (mp2mm), targets Mixamo directly:
        glm.vec3(landmark.x, -landmark.y, -landmark.z)

  - SAM3DBody-cpp, targets BVH world space:
        camera frame is (+X right, +Y down, +Z forward);
        bvh_writer.cpp applies diag(1, -1, -1) to reach its Y-up model pose.

Still verify this on YOUR MediaPipe version/model before trusting it
blindly -- run visualize_pose_candidates.py on a real reference frame and
confirm the (x=+1, y=-1, z=-1) candidate is the one that looks like a
normal standing person. If it isn't, override AXIS_SIGNS below; everything
downstream reads from this one place.
"""
from __future__ import annotations

AXIS_SIGNS: dict[str, int] = {"x": 1, "y": -1, "z": -1}


def apply_axis_conversion(point: dict[str, float]) -> dict[str, float]:
    """Convert ONE landmark dict, preserving any other keys (visibility...) untouched."""
    converted = dict(point)
    for axis, sign in AXIS_SIGNS.items():
        if axis in point:
            converted[axis] = sign * point[axis]
    return converted


def convert_frame(landmarks: list[dict[str, float]]) -> list[dict[str, float]]:
    """Convert every landmark in one frame (body pose OR hand, both are
    plain lists of {"x","y","z",...} dicts in this codebase)."""
    return [apply_axis_conversion(p) for p in landmarks]