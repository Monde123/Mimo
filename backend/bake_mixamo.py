"""
Injecte un clip retargeté (sortie de retarget_bvh_to_mixamo.retarget_bvh_to_mixamo)
directement dans le modèle Mixamo cible : le .glb de sortie est autonome, jouable
tel quel dans n'importe quel viewer glTF (three.js, Blender, etc.), sans fichier
BVH ni JSON séparé.

Contrairement au bake VRM (bake_vrm_animation.py), pas besoin de table humanBones :
les noms d'os Mixamo (mixamorig:LeftArm, ...) sont uniques, la correspondance se
fait directement par nom de nœud.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def bake_retarget_into_glb(
    clip: dict[str, Any],
    target_glb_path: str | Path,
    output_path: str | Path,
    animation_name: str = "MimoRetarget",
    keep_existing_animations: bool = False,
) -> dict[str, Any]:
    """
    keep_existing_animations=False (défaut) : retire TOUTE animation déjà présente
    dans le .glb cible (ex. "Idle"/"Walking" fournie par Mixamo.com) avant d'écrire
    la nôtre — garantit un fichier avec une seule animation, sans ambiguïté sur
    celle que le viewer doit jouer.
    keep_existing_animations=True : ne retire que celle qui porte le même nom que
    la nôtre (évite juste les doublons si tu re-bakes ta propre sortie), les autres
    animations du personnage restent dans le fichier à côté de la nouvelle.
    """
    from pygltflib import GLTF2, Animation, AnimationSampler, AnimationChannel, AnimationChannelTarget, Accessor, BufferView, Buffer, FLOAT

    target_glb_path, output_path = Path(target_glb_path), Path(output_path)
    tracks: dict[str, list[dict[str, Any]]] = {b: kfs for b, kfs in clip.get("tracks", {}).items() if kfs}
    if not tracks:
        raise ValueError("Le clip ne contient aucune piste exploitable.")

    gltf = GLTF2().load_binary(str(target_glb_path))
    blob = bytearray(gltf.binary_blob() or b"")
    if not gltf.buffers:
        gltf.buffers.append(Buffer(byteLength=0))

    name_to_idx = {node.name: idx for idx, node in enumerate(gltf.nodes) if node.name}
    matched: dict[str, int] = {}
    missing: list[str] = []
    for bone in tracks:
        if bone in name_to_idx:
            matched[bone] = name_to_idx[bone]
        else:
            missing.append(bone)

    if not matched:
        raise RuntimeError(
            "Aucun os du clip ne correspond à un nœud du .glb cible. "
            "Vérifie les noms exacts dans gltf.nodes (ex. avec un script d'inspection) "
            "et ajuste BVH_TO_MIXAMO en conséquence."
        )

    def push_view(raw: bytes) -> int:
        while len(blob) % 4:
            blob.append(0)
        offset = len(blob)
        blob.extend(raw)
        gltf.bufferViews.append(BufferView(buffer=0, byteOffset=offset, byteLength=len(raw)))
        return len(gltf.bufferViews) - 1

    n_before = len(gltf.animations)
    if keep_existing_animations:
        gltf.animations = [a for a in gltf.animations if a.name != animation_name]
    else:
        gltf.animations = []
    n_removed = n_before - len(gltf.animations)

    samplers, channels = [], []
    total_keyframes = 0
    for bone, node_idx in matched.items():
        kfs = tracks[bone]
        times = np.array([k["time"] for k in kfs], dtype="<f4")
        quats = np.array([k["rotation"] for k in kfs], dtype=np.float64)
        norms = np.linalg.norm(quats, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        quats = (quats / norms).astype("<f4")  # normalisation défensive

        t_bv = push_view(times.tobytes())
        gltf.accessors.append(Accessor(
            bufferView=t_bv, componentType=FLOAT, count=len(times), type="SCALAR",
            min=[float(times.min())], max=[float(times.max())],
        ))
        t_acc = len(gltf.accessors) - 1

        r_bv = push_view(quats.tobytes())
        gltf.accessors.append(Accessor(bufferView=r_bv, componentType=FLOAT, count=len(quats), type="VEC4"))
        r_acc = len(gltf.accessors) - 1

        samplers.append(AnimationSampler(input=t_acc, output=r_acc, interpolation="LINEAR"))
        channels.append(AnimationChannel(sampler=len(samplers) - 1, target=AnimationChannelTarget(node=node_idx, path="rotation")))
        total_keyframes += len(times)

    gltf.animations.append(Animation(name=animation_name, samplers=samplers, channels=channels))

    while len(blob) % 4:
        blob.append(0)
    gltf.buffers[0].byteLength = len(blob)
    gltf.set_binary_blob(bytes(blob))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gltf.save_binary(str(output_path))  # jamais .save() sur un .glb : ça écrirait du JSON + .bin externe

    return {
        "status": "success",
        "output": str(output_path),
        "bonesMatched": len(matched),
        "totalBones": len(tracks),
        "bonesMissing": missing,
        "keyframesCount": total_keyframes,
        "animationName": animation_name,
        "existingAnimationsRemoved": n_removed,
        "resultAnimationCount": len(gltf.animations),
    }


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) != 4:
        print("Usage: python -m backend.bake_mixamo_animation clip.json Clara.glb sortie.glb")
        sys.exit(1)
    clip = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    res = bake_retarget_into_glb(clip, sys.argv[2], sys.argv[3])
    print(f"✅ {res['output']} — os appariés : {res['bonesMatched']}/{res['totalBones']}")
    if res["bonesMissing"]:
        print(f"   ⚠️ Os sans correspondance dans le .glb : {res['bonesMissing']}")