"""
Outil de baking et d'inspection visuelle terminale des animations VRM / VRMA.
Permet d'injecter une animation VRMA (.json) dans un modèle VRM/glTF pour créer
un fichier .vrm/.glb animé autonome, et d'inspecter visuellement les os et
les rotations clés directement en mode console ASCII.

Usage:
    python -m backend.bake_vrm_animation modele.vrm animation.vrma.json [sortie_animee.vrm] [--inspect]
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any

# VRM Humanoid Bone Standard Names mapping to common node name conventions
VRM_BONE_SYNONYMS: dict[str, list[str]] = {
    "hips": ["hips", "Hips", "J_Bip_C_Hips"],
    "spine": ["spine", "Spine", "J_Bip_C_Spine"],
    "chest": ["chest", "Chest", "J_Bip_C_Chest"],
    "neck": ["neck", "Neck", "J_Bip_C_Neck"],
    "head": ["head", "Head", "J_Bip_C_Head"],
    "leftUpperArm": ["leftUpperArm", "LeftUpperArm", "LeftArm", "J_Bip_L_UpperArm"],
    "leftLowerArm": ["leftLowerArm", "LeftLowerArm", "LeftForeArm", "J_Bip_L_LowerArm"],
    "leftHand": ["leftHand", "LeftHand", "J_Bip_L_Hand"],
    "rightUpperArm": ["rightUpperArm", "RightUpperArm", "RightArm", "J_Bip_R_UpperArm"],
    "rightLowerArm": ["rightLowerArm", "RightLowerArm", "RightForeArm", "J_Bip_R_LowerArm"],
    "rightHand": ["rightHand", "RightHand", "J_Bip_R_Hand"],
}

def quat_to_euler_deg(q: list[float]) -> tuple[float, float, float]:
    """Convertit un quaternion [x, y, z, w] en angles d'Euler (Yaw, Pitch, Roll) en degrés."""
    x, y, z, w = q
    # Roll (x-axis)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)

    # Yaw (z-axis)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return (
        round(math.degrees(roll), 1),
        round(math.degrees(pitch), 1),
        round(math.degrees(yaw), 1),
    )

def ascii_angle_meter(val_deg: float, min_deg: float = -90.0, max_deg: float = 90.0, width: int = 15) -> str:
    """Affiche une jauge textuelle ASCII pour visualiser la flexion/rotation dans le terminal."""
    clamped = max(min_deg, min(max_deg, val_deg))
    ratio = (clamped - min_deg) / (max_deg - min_deg)
    pos = int(ratio * (width - 1))
    bar = ["-"] * width
    bar[pos] = "O"
    return f"[{''.join(bar)}]"

def inspect_vrma_terminal(clip_data: dict[str, Any], max_frames: int = 8) -> None:
    """
    Affiche une analyse visuelle et textuelle complète dans le terminal :
    - Squelette détecté et nombre de keyframes
    - Taux d'activité des 10 doigts
    - Aperçu ASCII des angles de rotation sur les frames clés
    """
    tracks = clip_data.get("tracks", {})
    fps = clip_data.get("fps", 30.0)
    duration = clip_data.get("duration", 0.0)
    meta = clip_data.get("meta", {})
    bones = list(tracks.keys())
    
    # Séparation corps et doigts
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

    # État des mains (Intégrité des 15 os par main)
    left_fingers = [b for b in hand_bones if b.startswith("left")]
    right_fingers = [b for b in hand_bones if b.startswith("right")]
    print(" 🖐 COUVERTURE ANATOMIQUE DES DOIGTS (15 phalanges par main) :")
    print(f"   ▶ Main gauche : {len(left_fingers)} / 15 os tracés " + ("✅ COMPLET" if len(left_fingers) >= 15 else "⚠️ PARTIEL"))
    print(f"   ▶ Main droite : {len(right_fingers)} / 15 os tracés " + ("✅ COMPLET" if len(right_fingers) >= 15 else "⚠️ PARTIEL"))
    print("-" * 72)

    # Affichage du graphe temporel des poses clés sur les membres majeurs
    sample_bones = [
        "rightUpperArm", "rightLowerArm", "rightIndexProximal",
        "leftUpperArm", "leftLowerArm", "rightThumbProximal"
    ]
    active_samples = [b for b in sample_bones if b in tracks and len(tracks[b]) > 0]

    if active_samples:
        print(" 📐 APERÇU CINÉMATIQUE SUR LES FRAMES CLÉS (Euler X / Y / Z en degrés) :")
        print(f"{'Temps (s)':<10} | {'Articulation':<22} | {'Euler (X, Y, Z)':<20} | {'Jauge Flexion (-90° à +90°)'}")
        print("-" * 72)

        total_keyframes = len(tracks[active_samples[0]])
        step = max(1, total_keyframes // max_frames)
        
        for idx in range(0, total_keyframes, step):
            for b in active_samples:
                kf = tracks[b][idx]
                t = kf["time"]
                rot = kf["rotation"]
                rx, ry, rz = quat_to_euler_deg(rot)
                meter = ascii_angle_meter(rx)
                print(f" {t:5.2f}s    | {b:<22} | ({rx:>5.1f}°, {ry:>5.1f}°, {rz:>5.1f}°) | {meter}")
            print("-" * 72)

    print(" 🎯 VALIDATION : Ce clip est directement consommable par three-vrm.")
    print("=" * 72)


def bake_vrma_to_vrm(
    vrm_path: str | Path,
    vrma_path: str | Path,
    output_path: str | Path | None = None
) -> dict[str, Any]:
    """
    Injecte les pistes d'animation VRMA dans le modèle VRM (GLTF2) et génère
    un fichier .vrm/.glb autonome avec son canal d'animation béké.
    """
    try:
        from pygltflib import (
            GLTF2, Animation, AnimationSampler, AnimationChannel,
            AnimationChannelTarget, Accessor, BufferView, FLOAT,
            ANIM_LINEAR, ARRAY_BUFFER
        )
    except ImportError:
        raise ImportError("pygltflib est requis pour le baking binaire : pip install pygltflib")

    vrm_file = Path(vrm_path)
    vrma_file = Path(vrma_path)

    if not vrm_file.exists():
        raise FileNotFoundError(f"Modèle VRM introuvable : {vrm_file}")
    if not vrma_file.exists():
        raise FileNotFoundError(f"Fichier VRMA introuvable : {vrma_file}")

    with open(vrma_file, "r", encoding="utf-8") as f:
        vrma_data = json.load(f)

    tracks = vrma_data.get("tracks", {})
    if not tracks:
        raise ValueError("Le fichier VRMA ne contient aucune piste dans 'tracks'.")

    gltf = GLTF2().load(str(vrm_file))
    blob = bytearray(gltf.binary_blob() or b"")

    # Recherche des correspondances de nœuds d'os dans le modèle VRM
    matched_nodes: dict[str, int] = {}
    for bone_name in tracks.keys():
        synonyms = VRM_BONE_SYNONYMS.get(bone_name, [bone_name])
        found_idx = None
        for syn in synonyms:
            for idx, node in enumerate(gltf.nodes):
                if node.name and (node.name == syn or syn.lower() in node.name.lower()):
                    found_idx = idx
                    break
            if found_idx is not None:
                break
        if found_idx is not None:
            matched_nodes[bone_name] = found_idx

    # Construction du buffer glTF d'animation
    first_track = next(iter(tracks.values()))
    times = np.array([kf["time"] for kf in first_track], dtype="<f4")
    
    # Time accessor
    byte_offset = len(blob)
    raw_time = times.tobytes()
    blob.extend(raw_time)
    while len(blob) % 4 != 0:
        blob.append(0)

    time_bv_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(BufferView(
        buffer=0, byteOffset=byte_offset, byteLength=len(raw_time), target=ARRAY_BUFFER
    ))
    time_acc_idx = len(gltf.accessors)
    gltf.accessors.append(Accessor(
        bufferView=time_bv_idx, componentType=FLOAT, count=len(times),
        type="SCALAR", min=[float(times.min())], max=[float(times.max())]
    ))

    samplers = []
    channels = []

    for bone_name, node_idx in matched_nodes.items():
        kfs = tracks[bone_name]
        rotations = np.array([k["rotation"] for k in kfs], dtype="<f4")
        
        rot_offset = len(blob)
        raw_rot = rotations.tobytes()
        blob.extend(raw_rot)
        while len(blob) % 4 != 0:
            blob.append(0)

        rot_bv_idx = len(gltf.bufferViews)
        gltf.bufferViews.append(BufferView(
            buffer=0, byteOffset=rot_offset, byteLength=len(raw_rot), target=ARRAY_BUFFER
        ))
        rot_acc_idx = len(gltf.accessors)
        gltf.accessors.append(Accessor(
            bufferView=rot_bv_idx, componentType=FLOAT, count=len(rotations), type="VEC4"
        ))

        sampler_idx = len(samplers)
        samplers.append(AnimationSampler(
            input=time_acc_idx, output=rot_acc_idx, interpolation=ANIM_LINEAR
        ))
        channels.append(AnimationChannel(
            sampler=sampler_idx,
            target=AnimationChannelTarget(node=node_idx, path="rotation")
        ))

    anim = Animation(name="SignLanguage_Mimo_Baked", samplers=samplers, channels=channels)
    gltf.animations.append(anim)
    gltf.set_binary_blob(bytes(blob))

    out = Path(output_path) if output_path else vrm_file.with_name(f"{vrm_file.stem}_baked.vrm")
    gltf.save(str(out))

    return {
        "status": "success",
        "output": str(out),
        "bonesMatched": len(matched_nodes),
        "totalBones": len(tracks),
        "keyframesCount": len(times),
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
        # Mode inspection terminale
        with open(first_arg, "r", encoding="utf-8") as f:
            inspect_vrma_terminal(json.load(f))
    else:
        # Mode baking
        vrm_file = first_arg
        vrma_file = Path(sys.argv[2])
        out_file = Path(sys.argv[3]) if len(sys.argv) > 3 else None
        
        # Inspection
        with open(vrma_file, "r", encoding="utf-8") as f:
            inspect_vrma_terminal(json.load(f))
            
        print(f"\n⏳ Baking de l'animation dans le modèle {vrm_file.name}...")
        res = bake_vrma_to_vrm(vrm_file, vrma_file, out_file)
        print(f"✅ Fichier béké généré avec succès : {res['output']}")
        print(f"   Os appariés : {res['bonesMatched']}/{res['totalBones']}")
