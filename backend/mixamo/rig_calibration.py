from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_CALIBRATION = {
    "name": "mixamo_default",
    "coordinateSystem": {"upAxis": "Y", "forwardAxis": "Z", "handedness": "right"},
    "bones": {},
    "hand": {
        "bendAxis": [1.0, 0.0, 0.0],
        "thumbBendAxis": [1.0, 0.0, 0.0]
    }
}


def load_rig_calibration(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return DEFAULT_CALIBRATION.copy()
    with Path(path).open("r", encoding="utf-8") as handle:
        calibration = json.load(handle)
    if not isinstance(calibration, dict):
        raise ValueError("Rig calibration must be a JSON object")
    calibration.setdefault("bones", {})
    calibration.setdefault("hand", DEFAULT_CALIBRATION["hand"])
    return calibration


def bone_config(calibration: dict[str, Any], bone: str) -> dict[str, Any]:
    return calibration.get("bones", {}).get(bone, {})
