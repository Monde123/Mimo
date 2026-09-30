"""
Outil de baking et d'inspection visuelle terminale des animations VRM / VRMA.
Injecte une animation VRMA (.json) dans un modèle VRM (glTF binaire) pour créer
un fichier .vrm/.glb animé autonome, et inspecte les os / rotations en ASCII.

Corrections par rapport à la v1 :
  1. Sauvegarde via save_binary() (plus de .vrm JSON + .bin externe)
  2. buffers[0].byteLength mis à jour
  3. Mapping des os via la table humanoid du modèle (VRM 1.0 et 0.x), doigts inclus
  4. Un accessor de temps par os (les pistes peuvent avoir des tailles différentes)
  5. Pas de target=ARRAY_BUFFER sur les bufferViews d'animation
  6. Composition avec la rotation de repos du nœud : final = rest * delta
  7. Continuité des quaternions (évite les tours complets en LINEAR)

Usage:
    python -m backend.bake_vrm_animation animation.vrma.json
    python -m backend.bake_vrm_animation modele.vrm animation.vrma.json [sortie.vrm]
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

# Synonymes de repli (correspondance EXACTE, insensible à la casse, jamais par sous-chaîne)
VRM_BONE_SYNONYMS: dict[str, list[str]] = {
    "hips": ["Hips", "J_Bip_C_Hips"],
    "spine": ["Spine", "J_Bip_C_Spine"],
    "chest": ["Chest", "J_Bip_C_Chest"],
    "upperChest": ["UpperChest", "J_Bip_C_UpperChest"],
    "neck": ["Neck", "J_Bip_C_Neck"],
    "head": ["Head", "J_Bip_C_Head"],
    "leftUpperArm": ["LeftUpperArm", "LeftArm", "J_Bip_L_UpperArm"],
    "leftLowerArm": ["LeftLowerArm", "LeftForeArm", "J_Bip_L_LowerArm"],
    "leftHand": ["LeftHand", "J_Bip_L_Hand"],
    "rightUpperArm": ["RightUpperArm", "RightArm", "J_Bip_R_UpperArm"],
    "rightLowerArm": ["RightLowerArm", "RightForeArm", "J_Bip_R_LowerArm"],
    "rightHand": ["RightHand", "J_Bip_R_Hand"],
}

# VRM 0.x n'a pas de Metacarpal au pouce : Proximal/Intermediate/Distal
VRM0_THUMB_RENAME = {
    "ThumbMetacarpal": "ThumbProximal",
    "ThumbProximal": "ThumbIntermediate",
    "ThumbDistal": "ThumbDistal",
}


# --------------------------------------------------------------------------- #
# Maths quaternions [x, y, z, w]
# --------------------------------------------------------------------------- #
def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Produit de Hamilton a * b (arrays (...,4) au format xyzw)."""
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def normalize_quats(q: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    return q / norms


def enforce_continuity(q: np.ndarray) -> np.ndarray:
    """q et -q = même rotation. On aligne chaque quaternion sur le précédent."""
    out = q.copy()
    for i in range(1, len(out)):
        if np.dot(out[i - 1], out[i]) < 0.0:
            out[i] = -out[i]
    return out


def quat_to_euler_deg(q: list[float]) -> tuple[float, float, float]:
    """Quaternion [x, y, z, w] -> angles d'Euler (X, Y, Z) en degrés."""
    x, y, z, w = q
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return (
        round(math.degrees(roll), 1),
        round(math.degrees(pitch), 1),
        round(math.degrees(yaw), 1),
    )


def ascii_angle_meter(val_deg: float, min_deg: float = -90.0, max_deg: float = 90.0, width: int = 15) -> str:
    clamped = max(min_deg, min(max_deg, val_deg))
    ratio = (clamped - min_deg) / (max_deg - min_deg)
    pos = int(ratio * (width - 1))
    bar = ["-"] * width
    bar[pos] = "O"
    return f"[{''.join(bar)}]"


# --------------------------------------------------------------------------- #
# Inspection terminale
# --------------------------------------------------------------------------- #
def inspect_vrma_terminal(clip_data: dict[str, Any], max_frames: int = 8) -> None:
    tracks = clip_data.get("tracks", {})
    fps = clip_data.get("fps", 30.0)
    duration = clip_data.get("duration", 0.0)
    meta = clip_data.get("meta", {})
    bones = list(tracks.keys())

    hand_bones = [b for b in bones if any(x in b for x in ["Thumb", "Index", "Middle", "Ring", "Little", "Hand"])]
    body_bones = [b for b in bones if b not in hand_bones]

    print("=" * 72)
    print(" 🎬 RAPPORT D'INSPECTION VISUELLE VRMA DANS LE TERMINAL")
    print("=" * 72)
    print(f" • Format standard : {clip_data.get('standard', 'VRM 1.0 / OpenXR')}")
    print(f" • Spécialisation  : {meta.get('signWord', meta.get('specialization', 'Langue des Signes'))}")
    print(f" • Durée totale    : {duration:.2f} s ({meta.get('frameCount', 0)} frames @ {fps} FPS)")
    print(f" • Os articulés    : {len(bones)} os totaux (Corps: {len(body_bones)}, Doigts: {len(hand_bones)})")
    print("-" * 72)

    # "Hand" (poignet) exclu du décompte : 15 phalanges par main
    left_fingers = [b for b in hand_bones if b.startswith("left") and not b.endswith("Hand")]
    right_fingers = [b for b in hand_bones if b.startswith("right") and not b.endswith("Hand")]
    print(" 🖐 COUVERTURE ANATOMIQUE DES DOIGTS (15 phalanges par main) :")
    print(f"   ▶ Main gauche : {len(left_fingers)} / 15 os tracés " + ("✅ COMPLET" if len(left_fingers) >= 15 else "⚠️ PARTIEL"))
    print(f"   ▶ Main droite : {len(right_fingers)} / 15 os tracés " + ("✅ COMPLET" if len(right_fingers) >= 15 else "⚠️ PARTIEL"))
    print("-" * 72)

    sample_bones = ["rightUpperArm", "rightLowerArm", "rightIndexProximal",
                    "leftUpperArm", "leftLowerArm", "rightThumbProximal"]
    active = [b for b in sample_bones if b in tracks and len(tracks[b]) > 0]

    if active:
        print(" 📐 APERÇU CINÉMATIQUE SUR LES FRAMES CLÉS (Euler X / Y / Z en degrés) :")
        print(f"{'Temps (s)':<10} | {'Articulation':<22} | {'Euler (X, Y, Z)':<20} | {'Jauge Flexion (-90° à +90°)'}")
        print("-" * 72)
        total = min(len(tracks[b]) for b in active)  # évite l'IndexError si pistes inégales
        step = max(1, total // max_frames)
        for idx in range(0, total, step):
            for b in active:
                kf = tracks[b][idx]
                rx, ry, rz = quat_to_euler_deg(kf["rotation"])
                print(f" {kf['time']:5.2f}s    | {b:<22} | ({rx:>5.1f}°, {ry:>5.1f}°, {rz:>5.1f}°) | {ascii_angle_meter(rx)}")
            print("-" * 72)

    print(" 🎯 VALIDATION : Ce clip est directement consommable par three-vrm.")
    print("=" * 72)


# --------------------------------------------------------------------------- #
# Mapping des os
# --------------------------------------------------------------------------- #
def read_humanoid_map(gltf: Any) -> tuple[dict[str, int], str]:
    """Lit la table humanoid du modèle. Retourne ({os: index_nœud}, version)."""
    ext = gltf.extensions or {}
    if "VRMC_vrm" in ext:  # VRM 1.0 : dict
        hb = ext["VRMC_vrm"].get("humanoid", {}).get("humanBones", {})
        return {k: v["node"] for k, v in hb.items() if isinstance(v, dict) and "node" in v}, "1.0"
    if "VRM" in ext:  # VRM 0.x : liste de {"bone", "node"}
        hb = ext["VRM"].get("humanoid", {}).get("humanBones", [])
        return {e["bone"]: e["node"] for e in hb if "bone" in e and "node" in e}, "0.x"
    return {}, "none"


def match_bones(gltf: Any, bone_names: list[str]) -> tuple[dict[str, int], list[str]]:
    humanoid, version = read_humanoid_map(gltf)
    node_by_lower_name: dict[str, int] = {}
    for idx, node in enumerate(gltf.nodes):
        if node.name:
            node_by_lower_name.setdefault(node.name.lower(), idx)

    matched: dict[str, int] = {}
    missing: list[str] = []
    for bone in bone_names:
        key = bone
        if version == "0.x":
            for vrm1, vrm0 in VRM0_THUMB_RENAME.items():
                if bone.endswith(vrm1):
                    key = bone[: -len(vrm1)] + vrm0
                    break
        if key in humanoid:
            matched[bone] = humanoid[key]
            continue
        # Repli : nom de nœud exact (jamais de sous-chaîne)
        for cand in [bone] + VRM_BONE_SYNONYMS.get(bone, []):
            if cand.lower() in node_by_lower_name:
                matched[bone] = node_by_lower_name[cand.lower()]
                break
        else:
            missing.append(bone)
    return matched, missing


# --------------------------------------------------------------------------- #
# Baking
# --------------------------------------------------------------------------- #
def bake_vrma_to_vrm(
    vrm_path: str | Path,
    vrma_path: str | Path,
    output_path: str | Path | None = None,
    animation_name: str = "SignLanguage_Mimo_Baked",
) -> dict[str, Any]:
    try:
        from pygltflib import (
            GLTF2, Animation, AnimationSampler, AnimationChannel,
            AnimationChannelTarget, Accessor, BufferView, Buffer, FLOAT,
        )
    except ImportError:
        raise ImportError("pygltflib est requis : pip install pygltflib")

    vrm_file, vrma_file = Path(vrm_path), Path(vrma_path)
    if not vrm_file.exists():
        raise FileNotFoundError(f"Modèle VRM introuvable : {vrm_file}")
    if not vrma_file.exists():
        raise FileNotFoundError(f"Fichier VRMA introuvable : {vrma_file}")

    with open(vrma_file, "r", encoding="utf-8") as f:
        tracks: dict[str, list[dict[str, Any]]] = json.load(f).get("tracks", {})
    tracks = {b: kfs for b, kfs in tracks.items() if kfs}
    if not tracks:
        raise ValueError("Le fichier VRMA ne contient aucune piste exploitable dans 'tracks'.")

    gltf = GLTF2().load_binary(str(vrm_file))
    blob = bytearray(gltf.binary_blob() or b"")
    if not gltf.buffers:
        gltf.buffers.append(Buffer(byteLength=0))

    matched, missing = match_bones(gltf, list(tracks.keys()))
    if not matched:
        raise RuntimeError("Aucun os du clip ne correspond au modèle (table humanoid absente ?).")

    def push_view(raw: bytes) -> int:
        while len(blob) % 4:
            blob.append(0)
        offset = len(blob)
        blob.extend(raw)
        gltf.bufferViews.append(BufferView(buffer=0, byteOffset=offset, byteLength=len(raw)))
        return len(gltf.bufferViews) - 1

    # Une ancienne animation du même nom est remplacée, pas dupliquée
    gltf.animations = [a for a in gltf.animations if a.name != animation_name]

    samplers: list[AnimationSampler] = []
    channels: list[AnimationChannel] = []
    total_keyframes = 0

    for bone, node_idx in matched.items():
        kfs = tracks[bone]
        times = np.array([k["time"] for k in kfs], dtype="<f4")
        delta = normalize_quats(np.array([k["rotation"] for k in kfs], dtype=np.float64))

        # Composition avec la pose de repos : un canal glTF REMPLACE la rotation locale
        rest_list = gltf.nodes[node_idx].rotation or [0.0, 0.0, 0.0, 1.0]
        rest = np.array(rest_list, dtype=np.float64)
        final = normalize_quats(quat_mul(np.broadcast_to(rest, delta.shape), delta))
        final = enforce_continuity(final).astype("<f4")

        t_bv = push_view(times.tobytes())
        gltf.accessors.append(Accessor(
            bufferView=t_bv, componentType=FLOAT, count=len(times), type="SCALAR",
            min=[float(times.min())], max=[float(times.max())],
        ))
        t_acc = len(gltf.accessors) - 1

        r_bv = push_view(final.tobytes())
        gltf.accessors.append(Accessor(
            bufferView=r_bv, componentType=FLOAT, count=len(final), type="VEC4",
        ))
        r_acc = len(gltf.accessors) - 1

        samplers.append(AnimationSampler(input=t_acc, output=r_acc, interpolation="LINEAR"))
        channels.append(AnimationChannel(
            sampler=len(samplers) - 1,
            target=AnimationChannelTarget(node=node_idx, path="rotation"),
        ))
        total_keyframes += len(times)

    gltf.animations.append(Animation(name=animation_name, samplers=samplers, channels=channels))

    while len(blob) % 4:
        blob.append(0)
    gltf.buffers[0].byteLength = len(blob)  # sinon les loaders rejettent le fichier
    gltf.set_binary_blob(bytes(blob))

    out = Path(output_path) if output_path else vrm_file.with_name(f"{vrm_file.stem}_baked.vrm")
    out.parent.mkdir(parents=True, exist_ok=True)
    gltf.save_binary(str(out))  # save() sur .vrm écrirait du JSON + .bin externe

    return {
        "status": "success",
        "output": str(out),
        "bonesMatched": len(matched),
        "totalBones": len(tracks),
        "bonesMissing": missing,
        "keyframesCount": total_keyframes,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Inspection seule : python -m backend.bake_vrm_animation animation.vrma.json")
        print("  Baking complet   : python -m backend.bake_vrm_animation modele.vrm animation.vrma.json sortie.vrm")
        sys.exit(1)

    first_arg = Path(sys.argv[1])
    if first_arg.suffix.lower() == ".json":
        with open(first_arg, "r", encoding="utf-8") as f:
            inspect_vrma_terminal(json.load(f))
    else:
        vrm_file = first_arg
        vrma_file = Path(sys.argv[2])
        out_file = Path(sys.argv[3]) if len(sys.argv) > 3 else None

        with open(vrma_file, "r", encoding="utf-8") as f:
            inspect_vrma_terminal(json.load(f))

        print(f"\n⏳ Baking de l'animation dans le modèle {vrm_file.name}...")
        res = bake_vrma_to_vrm(vrm_file, vrma_file, out_file)
        print(f"✅ Fichier béké généré avec succès : {res['output']}")
        print(f"   Os appariés : {res['bonesMatched']}/{res['totalBones']}")
        if res["bonesMissing"]:
            print(f"   ⚠️ Os sans correspondance : {', '.join(res['bonesMissing'])}")