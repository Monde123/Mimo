"""
Injecte un clip retargeté (sortie de retarget_bvh_to_mixamo.retarget_bvh_to_mixamo)
directement dans le modèle Mixamo cible : le .glb de sortie est autonome, jouable
tel quel dans n'importe quel viewer glTF (three.js, Blender, etc.), sans fichier
BVH ni JSON séparé.

Contrairement au bake VRM (bake_vrm_animation.py), pas besoin de table humanBones :
les noms d'os Mixamo (mixamorig:LeftArm, ...) sont uniques, la correspondance se
fait directement par nom de nœud.

Les accessors des animations retirées restent dans le buffer (données orphelines,
le fichier est valide mais un peu plus lourd).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _track_arrays(bone: str, kfs: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    try:
        times = np.array([k["time"] for k in kfs], dtype=np.float64)
        quats = np.array([k["rotation"] for k in kfs], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{bone} : chaque clé doit avoir 'time' et 'rotation' [x, y, z, w] ({exc})") from exc
    if quats.shape != (len(kfs), 4):
        raise ValueError(f"{bone} : rotation de forme {quats.shape}, attendu ({len(kfs)}, 4)")
    if not (np.isfinite(times).all() and np.isfinite(quats).all()):
        raise ValueError(f"{bone} : temps ou quaternion non fini")
    if times[0] < 0 or (np.diff(times) <= 0).any():
        raise ValueError(f"{bone} : les temps doivent être >= 0 et strictement croissants (exigence glTF)")
    norms = np.linalg.norm(quats, axis=1)
    if (norms < 1e-8).any():
        t = times[int(np.argmin(norms))]
        raise ValueError(f"{bone} t={t:.3f} : quaternion nul")
    return times.astype("<f4"), (quats / norms[:, None]).astype("<f4")


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
    from pygltflib import Animation, AnimationSampler, AnimationChannel, AnimationChannelTarget, Accessor, BufferView, Buffer, FLOAT
    from backend.retarget_bvh_to_mixamo import _load_gltf

    target_glb_path, output_path = Path(target_glb_path), Path(output_path)
    if target_glb_path.suffix.lower() != ".glb" or output_path.suffix.lower() != ".glb":
        raise ValueError(f"bake Mixamo : entrée et sortie doivent être des .glb ({target_glb_path.name} -> {output_path.name})")
    if not isinstance(clip, dict) or not isinstance(clip.get("tracks"), dict):
        raise ValueError("Le clip doit être un objet JSON avec un dictionnaire 'tracks' (sortie de mimo retarget).")
    tracks: dict[str, list[dict[str, Any]]] = {b: kfs for b, kfs in clip["tracks"].items() if kfs}
    if not tracks:
        raise ValueError("Le clip ne contient aucune piste exploitable.")
    arrays = {bone: _track_arrays(bone, kfs) for bone, kfs in tracks.items()}

    gltf = _load_gltf(target_glb_path)
    if gltf.buffers and gltf.buffers[0].uri:
        raise ValueError(f"{target_glb_path.name} : le buffer 0 est externe ({gltf.buffers[0].uri}), GLB autonome attendu")
    blob = bytearray(gltf.binary_blob() or b"")
    if not gltf.buffers:
        gltf.buffers.append(Buffer(byteLength=0))

    name_to_idx = {node.name: idx for idx, node in enumerate(gltf.nodes) if node.name}
    matched = {bone: name_to_idx[bone] for bone in tracks if bone in name_to_idx}
    missing = [bone for bone in tracks if bone not in name_to_idx]

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
        times, quats = arrays[bone]

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
    print(f"{res['output']} — os appariés : {res['bonesMatched']}/{res['totalBones']}")
    if res["bonesMissing"]:
        print(f"   Os sans correspondance dans le .glb : {res['bonesMissing']}")
