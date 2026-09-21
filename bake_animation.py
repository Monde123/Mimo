"""Bake a Mimo animation JSON (mimo.mixamo.animation format) onto a
Mixamo-rigged glTF/GLB model, producing a single self-contained animated .glb.

Usage:
    python bake_animation.py clara.glb output.json animated_model.glb
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pygltflib import (
    GLTF2, Animation, AnimationSampler, AnimationChannel, AnimationChannelTarget,
    Accessor, BufferView, FLOAT, ANIM_LINEAR, ARRAY_BUFFER,
)

SCALAR = "SCALAR"
VEC3 = "VEC3"
VEC4 = "VEC4"


def _find_node_index(gltf: GLTF2, bone_name: str) -> int | None:
    for index, node in enumerate(gltf.nodes):
        if node.name == bone_name:
            return index
    suffix = bone_name.split(":")[-1]
    for index, node in enumerate(gltf.nodes):
        if node.name == suffix:
            return index
    return None


def _append_accessor(gltf: GLTF2, blob: bytearray, data: np.ndarray, accessor_type: str,
                      min_max: bool = False) -> int:
    byte_offset = len(blob)
    raw = data.astype("<f4").tobytes()
    blob.extend(raw)
    while len(blob) % 4 != 0:
        blob.append(0)

    buffer_view_index = len(gltf.bufferViews)
    gltf.bufferViews.append(BufferView(
        buffer=0, byteOffset=byte_offset, byteLength=len(raw), target=ARRAY_BUFFER,
    ))

    accessor = Accessor(
        bufferView=buffer_view_index, componentType=FLOAT,
        count=len(data), type=accessor_type,
    )
    if min_max:
        accessor.min = data.min(axis=0).tolist() if data.ndim > 1 else [float(data.min())]
        accessor.max = data.max(axis=0).tolist() if data.ndim > 1 else [float(data.max())]

    accessor_index = len(gltf.accessors)
    gltf.accessors.append(accessor)
    return accessor_index


def _forward_fill_track(frames: list[dict], bone_name: str, field: str, default: list[float]) -> tuple[np.ndarray, int]:
    """Build a per-frame array for one bone's rotation/position, HOLDING the
    last known value across frames where the bone is entirely absent
    (e.g. a hand not detected that frame) instead of snapping to `default`.
    Returns (values, missing_count) so the caller can report it.
    """
    values = []
    last_known = None
    missing_count = 0
    for frame in frames:
        bone_entry = frame["bones"].get(bone_name)
        if bone_entry is not None and field in bone_entry:
            last_known = bone_entry[field]
        elif last_known is None:
            missing_count += 1  # missing from the very start -- no prior value to hold
        else:
            missing_count += 1
        values.append(last_known if last_known is not None else default)
    return np.array(values, dtype="<f4"), missing_count


def bake_animation(model_path: str | Path, clip_path: str | Path, output_path: str | Path,
                    position_scale: float = 1.0) -> dict:
    """position_scale: applied to the root (Hips) position DELTA before it is
    added to the model's own bind translation. Needed because MediaPipe
    positions are in metres in MediaPipe's own scale, while the character's
    rig may use a completely different unit/proportion -- writing the raw
    MediaPipe position directly (as this function used to) overwrites the
    model's bind translation outright and ignores any scale mismatch,
    exactly the mistake Rokoko's Blender retargeter (scale_armature) computes
    a height-ratio for before baking. Pass 1.0 to disable (no rescaling).
    """
    gltf = GLTF2().load(str(model_path))
    blob = bytearray(gltf.binary_blob() or b"")

    with open(clip_path, "r", encoding="utf-8") as handle:
        clip = json.load(handle)

    if clip.get("coordinateSystem", {}).get("rotationSpace") != "local":
        raise ValueError("bake_animation expects LOCAL rotations (as produced by body_solver/hand_solver); "
                          "this clip is not in local space.")

    frames = clip["frames"]
    times = np.array([f["time"] for f in frames], dtype="<f4")

    bone_names = sorted({bone for f in frames for bone in f["bones"]})
    matched, unmatched = {}, []
    for bone_name in bone_names:
        node_index = _find_node_index(gltf, bone_name)
        if node_index is None:
            unmatched.append(bone_name)
        else:
            matched[bone_name] = node_index

    time_accessor = _append_accessor(gltf, blob, times, SCALAR, min_max=True)

    samplers: list[AnimationSampler] = []
    channels: list[AnimationChannel] = []
    held_frames_report: dict[str, int] = {}

    for bone_name, node_index in matched.items():
        rotations, missing = _forward_fill_track(frames, bone_name, "rotation", [0.0, 0.0, 0.0, 1.0])
        if missing:
            held_frames_report[bone_name] = missing

        rot_accessor = _append_accessor(gltf, blob, rotations, VEC4)
        sampler_index = len(samplers)
        samplers.append(AnimationSampler(input=time_accessor, output=rot_accessor, interpolation=ANIM_LINEAR))
        channels.append(AnimationChannel(
            sampler=sampler_index,
            target=AnimationChannelTarget(node=node_index, path="rotation"),
        ))

        if any("position" in f["bones"].get(bone_name, {}) for f in frames):
            # A glTF translation channel REPLACES the node's local translation
            # outright (same lesson learned for "rotation" earlier) -- so we
            # must add back the model's OWN bind translation, and only ever
            # write the RELATIVE displacement from frame 0, not the raw
            # absolute MediaPipe-space position.
            raw_positions, _ = _forward_fill_track(frames, bone_name, "position", [0.0, 0.0, 0.0])
            bind_translation = np.array(gltf.nodes[node_index].translation or [0.0, 0.0, 0.0], dtype="<f4")
            relative_delta = (raw_positions - raw_positions[0]) * position_scale
            positions = bind_translation[None, :] + relative_delta

            pos_accessor = _append_accessor(gltf, blob, positions, VEC3)
            pos_sampler_index = len(samplers)
            samplers.append(AnimationSampler(input=time_accessor, output=pos_accessor, interpolation=ANIM_LINEAR))
            channels.append(AnimationChannel(
                sampler=pos_sampler_index,
                target=AnimationChannelTarget(node=node_index, path="translation"),
            ))

    gltf.animations.append(Animation(
        name=Path(clip_path).stem, samplers=samplers, channels=channels,
    ))

    gltf.set_binary_blob(bytes(blob))
    gltf.buffers[0].byteLength = len(blob)
    gltf.save(str(output_path))

    return {
        "output": str(output_path),
        "bonesAnimated": len(matched),
        "bonesUnmatched": unmatched,
        "frameCount": len(frames),
        "durationSeconds": clip.get("duration"),
        "framesHeldPerBone": held_frames_report,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Path to the Mixamo-rigged .glb/.gltf model")
    parser.add_argument("clip", help="Path to the Mimo output.json animation clip")
    parser.add_argument("output", help="Path to write the animated .glb")
    args = parser.parse_args()

    report = bake_animation(args.model, args.clip, args.output)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["bonesUnmatched"]:
        print(f"\nATTENTION : {len(report['bonesUnmatched'])} os du clip n'ont pas de node correspondant "
              f"dans le modele : {report['bonesUnmatched']}")
    if report["framesHeldPerBone"]:
        print(f"\nNote : certains os ont ete detectes absents sur certaines frames "
              f"(main hors champ, par ex.) -- derniere valeur connue maintenue plutot "
              f"qu'un saut a l'identite : {report['framesHeldPerBone']}")


if __name__ == "__main__":
    main()