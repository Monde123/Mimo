"""
Serializer binaire officiel pour le format .vrma (VRM Animation).
Conforme à la spécification VRMC_vrm_animation (Khronos glTF 2.0 Binary - .glb / .vrma).

Le format .vrma est un conteneur binaire glTF 2.0 (.glb) spécialisé qui embarque :
- L'extension glTF `VRMC_vrm_animation` dans `extensionsUsed` et `extensions`
- Des nœuds hiérarchiques représentant les os humanoïdes standardisés
- Des canaux d'animation (sampler LINEAR, rotation / translation) stockés
  directement dans le binaire GLB (BIN chunk)
- La cartographie des nœuds vers les `humanBones` du standard VRM 1.0
"""

from __future__ import annotations
import struct
import json
from pathlib import Path
from typing import Any

# Liste des 55 os humanoïdes VRM 1.0 officiels
VRM_HUMANOID_BONES = [
    "hips", "spine", "chest", "upperChest", "neck", "head",
    "leftEye", "rightEye", "jaw",
    "leftUpperLeg", "leftLowerLeg", "leftFoot", "leftToes",
    "rightUpperLeg", "rightLowerLeg", "rightFoot", "rightToes",
    "leftShoulder", "leftUpperArm", "leftLowerArm", "leftHand",
    "rightShoulder", "rightUpperArm", "rightLowerArm", "rightHand",
    # Doigts main gauche
    "leftThumbMetacarpal", "leftThumbProximal", "leftThumbDistal",
    "leftIndexProximal", "leftIndexIntermediate", "leftIndexDistal",
    "leftMiddleProximal", "leftMiddleIntermediate", "leftMiddleDistal",
    "leftRingProximal", "leftRingIntermediate", "leftRingDistal",
    "leftLittleProximal", "leftLittleIntermediate", "leftLittleDistal",
    # Doigts main droite
    "rightThumbMetacarpal", "rightThumbProximal", "rightThumbDistal",
    "rightIndexProximal", "rightIndexIntermediate", "rightIndexDistal",
    "rightMiddleProximal", "rightMiddleIntermediate", "rightMiddleDistal",
    "rightRingProximal", "rightRingIntermediate", "rightRingDistal",
    "rightLittleProximal", "rightLittleIntermediate", "rightLittleDistal"
]


def serialize_to_vrma(clip_payload: dict[str, Any], output_path: str | Path) -> Path:
    """
    Convertit un dictionnaire d'animation Mimo en fichier binaire standard .vrma.
    
    Structure binaire glTF 2.0 (VRMA) :
    - En-tête 12 octets : magic (0x46546C67 "glTF"), version (2), totalLength
    - Chunk 0 : JSON textuel décrivant les nœuds, buffers, animations et l'extension VRMC_vrm_animation
    - Chunk 1 : Binaire (BIN) contenant les timestamps (FLOAT) et quaternions (VEC4 FLOAT)
    """
    output_path = Path(output_path)
    tracks: dict[str, list[dict[str, Any]]] = clip_payload.get("tracks", {})
    if not tracks:
        raise ValueError("Le clip d'animation ne contient aucune piste 'tracks'.")

    # 1. Identifier les os animés et créer un nœud glTF pour chacun
    animated_bones = [b for b in tracks.keys() if len(tracks[b]) > 0]
    
    nodes = []
    bone_name_to_node_idx: dict[str, int] = {}
    human_bones_map: dict[str, dict[str, int]] = {}

    for idx, bone_name in enumerate(animated_bones):
        bone_name_to_node_idx[bone_name] = idx
        nodes.append({
            "name": bone_name
        })
        # Si c'est un os standard VRM, on le mappe dans VRMC_vrm_animation
        if bone_name in VRM_HUMANOID_BONES:
            human_bones_map[bone_name] = {"node": idx}

    # 2. Construire le buffer binaire (timestamps et rotations quaternions)
    bin_buffer = bytearray()
    buffer_views = []
    accessors = []
    animation_samplers = []
    animation_channels = []

    def pad_bin():
        """Alignement 4-octets obligatoire selon la spec glTF 2.0"""
        remainder = len(bin_buffer) % 4
        if remainder != 0:
            bin_buffer.extend(b"\x00" * (4 - remainder))

    for bone_name in animated_bones:
        node_idx = bone_name_to_node_idx[bone_name]
        key_frames = tracks[bone_name]
        
        times: list[float] = [float(k["time"]) for k in key_frames]
        rotations: list[list[float]] = [k["rotation"] for k in key_frames]
        count = len(times)

        # BufferView pour les temps (FLOAT, SCALAR)
        pad_bin()
        time_offset = len(bin_buffer)
        for t in times:
            bin_buffer.extend(struct.pack("<f", t))
        time_length = len(bin_buffer) - time_offset

        time_bv_idx = len(buffer_views)
        buffer_views.append({
            "buffer": 0,
            "byteOffset": time_offset,
            "byteLength": time_length
        })

        time_acc_idx = len(accessors)
        accessors.append({
            "bufferView": time_bv_idx,
            "byteOffset": 0,
            "componentType": 5126,  # FLOAT
            "count": count,
            "type": "SCALAR",
            "min": [min(times) if times else 0.0],
            "max": [max(times) if times else 0.0]
        })

        # BufferView pour les quaternions (FLOAT, VEC4)
        pad_bin()
        rot_offset = len(bin_buffer)
        for q in rotations:
            # Assurer [x, y, z, w]
            qx, qy, qz, qw = float(q[0]), float(q[1]), float(q[2]), float(q[3])
            bin_buffer.extend(struct.pack("<ffff", qx, qy, qz, qw))
        rot_length = len(bin_buffer) - rot_offset

        rot_bv_idx = len(buffer_views)
        buffer_views.append({
            "buffer": 0,
            "byteOffset": rot_offset,
            "byteLength": rot_length
        })

        rot_acc_idx = len(accessors)
        accessors.append({
            "bufferView": rot_bv_idx,
            "byteOffset": 0,
            "componentType": 5126,  # FLOAT
            "count": count,
            "type": "VEC4"
        })

        # Sampler et Channel
        sampler_idx = len(animation_samplers)
        animation_samplers.append({
            "input": time_acc_idx,
            "interpolation": "LINEAR",
            "output": rot_acc_idx
        })

        animation_channels.append({
            "sampler": sampler_idx,
            "target": {
                "node": node_idx,
                "path": "rotation"
            }
        })

    pad_bin()

    # 3. Assembler le document JSON glTF 2.0 avec l'extension VRMC_vrm_animation
    gltf_doc: dict[str, Any] = {
        "asset": {
            "version": "2.0",
            "generator": "Mimo Sign VRMA Serializer v2.0"
        },
        "extensionsUsed": [
            "VRMC_vrm_animation"
        ],
        "extensions": {
            "VRMC_vrm_animation": {
                "specVersion": "1.0",
                "humanoid": {
                    "humanBones": human_bones_map
                }
            }
        },
        "scenes": [
            {
                "nodes": list(range(len(nodes)))
            }
        ],
        "scene": 0,
        "nodes": nodes,
        "animations": [
            {
                "name": clip_payload.get("meta", {}).get("signWord", "Sign_Animation"),
                "samplers": animation_samplers,
                "channels": animation_channels
            }
        ],
        "buffers": [
            {
                "byteLength": len(bin_buffer)
            }
        ],
        "bufferViews": buffer_views,
        "accessors": accessors
    }

    # 4. Packaging binaire GLB
    json_bytes = json.dumps(gltf_doc, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    
    # Alignement du JSON Chunk à un multiple de 4 octets avec des espaces (0x20)
    json_pad = (4 - (len(json_bytes) % 4)) % 4
    if json_pad > 0:
        json_bytes += b" " * json_pad

    # Alignement du BIN Chunk avec des zéros
    bin_pad = (4 - (len(bin_buffer) % 4)) % 4
    if bin_pad > 0:
        bin_buffer.extend(b"\x00" * bin_pad)

    total_length = 12 + 8 + len(json_bytes) + 8 + len(bin_buffer)

    header = struct.pack("<4sII", b"glTF", 2, total_length)
    chunk0_header = struct.pack("<II", len(json_bytes), 0x4E4F534A)  # 'JSON'
    chunk1_header = struct.pack("<II", len(bin_buffer), 0x004E4942)  # 'BIN\x00'

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(header)
        f.write(chunk0_header)
        f.write(json_bytes)
        f.write(chunk1_header)
        f.write(bin_buffer)

    return output_path
