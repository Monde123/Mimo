from __future__ import annotations

from typing import Any

SUPPORTED_FORMAT = "mimo.mixamo.animation"
SUPPORTED_VERSION = 1


def clip_metadata() -> dict[str, Any]:
    return {
        "format": SUPPORTED_FORMAT,
        "version": SUPPORTED_VERSION,
        "skeleton": "mixamo",
        "coordinateSystem": {
            "upAxis": "Y",
            "forwardAxis": "Z",
            "handedness": "right",
            "quaternionOrder": "xyzw",
            "rotationSpace": "local",
        },
    }
