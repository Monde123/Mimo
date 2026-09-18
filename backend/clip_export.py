import os
import json
from typing import List, Dict, Any


def export_mixamo_clip(clip: Dict[str, Any], output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        json.dump(clip, fout, indent=2)
    return output_path


def export_clip_for_mobile(frames: List[Dict[str, Any]], fps: int = 30, output_path: str = "exports/sign_clip.json"):
    clip = {"version": 1, "fps": fps, "frames": []}
    for idx, frame in enumerate(frames):
        clip["frames"].append({
            "time": idx / max(fps, 1),
            "bones": frame.get("bones", {}),
        })
    export_mixamo_clip(clip, output_path)
    return output_path
