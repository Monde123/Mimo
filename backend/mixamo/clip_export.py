from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.mixamo.clip_validator import validate_clip


def export_mixamo_clip(clip: dict[str, Any], output_path: str) -> str:
    path = Path(output_path)
    if path.suffix.lower() != ".json":
        raise ValueError("Animation output must use the .json extension")
    validate_clip(clip)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clip, indent=2), encoding="utf-8")
    temporary.replace(path)
    return str(path)
