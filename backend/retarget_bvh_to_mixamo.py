"""
Retargeting d'un BVH (export_mediapipe_bvh, mode "rotations") vers un modèle Mixamo.

Principe : celui documenté par UniVRM pour son ControlRig (et repris par Rokoko
Retargeting dans Blender) — la conversion de pose entre deux squelettes de T-pose
différente passe par un format pivot indépendant du repos :

    PoseForA -> NormalizedLocalRotation -> PoseForB

    NormalizedLocalRotation = W_A · L_A⁻¹ · A.local · W_A⁻¹
    B.local                 = L_B · W_B⁻¹ · NormalizedLocalRotation · W_B

où L = rotation locale de repos, W = rotation monde de repos.

Simplification qui s'applique ici : notre BVH est construit avec un repos
identité par construction (les tuples de _SKELETON sont des directions
d'OFFSET, pas des rotations — à valeur de canal nulle, chaque os pointe déjà
le long de son offset). Donc L_A = W_A = identité, et la première étape
disparaît : le canal de rotation du BVH EST déjà NormalizedLocalRotation.
Il ne reste que la deuxième étape, avec (L_B, W_B) lus dans la pose de repos
du modèle Mixamo cible.

Limite de la formule seule (mesurée sur clara.glb) : elle aligne les conventions
d'axes, pas les poses de repos. Si la T-pose de la cible differe de celle du BVH
(bras 7 deg plus bas, doigts 13-26 deg), chaque os garde cet ecart constant dans
toute l'animation. On corrige donc, par os, la rotation C_B qui amene les
directions de repos de la cible sur celles du BVH (2 directions quand elles
existent : paume, bassin, haut du torse), et on applique en rotations monde :

    Q_B(t)    = G_A(t) · C_B · W_B          (G_A = rotation monde du joint BVH)
    B.local   = Q_parent(t)⁻¹ · Q_B(t)       (parent tel qu'anime, os intermediaires inclus)

Avec C_B = identite et des hierarchies identiques, c'est exactement la formule UniVRM.

Usage:
    python -m backend.retarget_bvh_to_mixamo capture.bvh Clara.glb sortie.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


def _finalize_tracks(tracks: dict[str, list[dict[str, Any]]]) -> None:
    """
    Normalise chaque quaternion et force la continuité de signe (q et -q = même
    rotation). Sans ça, un changement de signe d'une frame à l'autre — possible
    après composition L·W⁻¹·N·W même si N était déjà continu — peut faire
    "sauter" l'interpolation LINEAR du moteur de rendu (slerp du mauvais côté
    de la sphère). C'est un vrai levier de fluidité, contrairement à Kabsch
    qui n'a pas sa place ici (voir docstring du module).
    """
    for bone, kfs in tracks.items():
        prev: np.ndarray | None = None
        for kf in kfs:
            q = np.asarray(kf["rotation"], dtype=float)
            n = np.linalg.norm(q)
            if not (np.isfinite(q).all() and n > 1e-8):
                raise ValueError(f"{bone} t={kf['time']:.3f} : quaternion invalide {q.tolist()}")
            q = q / n
            if prev is not None and np.dot(prev, q) < 0.0:
                q = -q
            kf["rotation"] = q.tolist()
            prev = q


# --------------------------------------------------------------------------
# 1. Lecture du BVH
# --------------------------------------------------------------------------
_BVH_CHANNELS = {"Xposition", "Yposition", "Zposition", "Xrotation", "Yrotation", "Zrotation"}


def parse_bvh(path: str | Path) -> tuple[list[dict[str, Any]], float, np.ndarray]:
    """
    Parseur BVH minimal mais générique (respecte l'ordre des canaux déclaré,
    ne suppose pas Zrotation/Xrotation/Yrotation comme le fait le générateur
    par défaut — un BVH d'une autre origine peut utiliser un autre ordre).
    Retourne (joints, frame_time, motion) où motion est un tableau (T, total_canaux).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"BVH introuvable : {path}")
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    joints: list[dict[str, Any]] = []
    stack: list[str] = []
    i = 0

    def expect(cond: bool, what: str) -> None:
        if not cond:
            found = lines[i] if i < len(lines) else "fin de fichier"
            raise ValueError(f"BVH {path.name}, ligne utile {i + 1} : {what} attendu, trouve '{found}'")

    def parse_offset() -> np.ndarray:
        nonlocal i
        expect(i < len(lines) and lines[i].split()[0] == "OFFSET", "OFFSET")
        parts = lines[i].split()[1:]
        try:
            offset = np.array([float(v) for v in parts])
        except ValueError:
            offset = np.array([np.nan])
        expect(offset.shape == (3,) and np.isfinite(offset).all(), "OFFSET a 3 valeurs numeriques")
        i += 1
        return offset

    def parse_block() -> None:
        nonlocal i
        parts = lines[i].split(None, 1)
        expect(len(parts) == 2, "nom de joint")
        name = parts[1]
        i += 1
        expect(i < len(lines) and lines[i] == "{", "'{'")
        i += 1
        offset = parse_offset()
        parts = lines[i].split() if i < len(lines) else []
        expect(len(parts) >= 2 and parts[0] == "CHANNELS" and parts[1].isdigit()
               and len(parts) == 2 + int(parts[1]) and set(parts[2:]) <= _BVH_CHANNELS, "CHANNELS n canal...")
        channels = parts[2:]
        i += 1
        joints.append({"name": name, "parent": stack[-1] if stack else None, "channels": channels,
                       "offset": offset})
        stack.append(name)
        while True:
            expect(i < len(lines), "'}' ou JOINT")
            if lines[i] == "}":
                break
            if lines[i].startswith("JOINT"):
                parse_block()
            elif lines[i].startswith("End Site"):
                i += 1
                expect(i < len(lines) and lines[i] == "{", "'{'")
                i += 1
                parse_offset()
                expect(i < len(lines) and lines[i] == "}", "'}' d'End Site")
                i += 1
            else:
                expect(False, "JOINT, End Site ou '}'")
        i += 1  # '}' de fermeture du joint
        stack.pop()

    expect(bool(lines) and lines[0] == "HIERARCHY", "HIERARCHY")
    i += 1
    expect(i < len(lines) and lines[i].startswith("ROOT"), "ROOT")
    parse_block()  # ROOT
    expect(i < len(lines) and lines[i] == "MOTION", "MOTION")
    i += 1
    try:
        n_frames = int(lines[i].split(":")[1])
        frame_time = float(lines[i + 1].split(":")[1])
    except (IndexError, ValueError):
        expect(False, "'Frames: N' puis 'Frame Time: dt'")
    if n_frames <= 0 or not (frame_time > 0):
        raise ValueError(f"BVH {path.name} : Frames={n_frames}, Frame Time={frame_time} invalides")
    i += 2
    n_channels = sum(len(j["channels"]) for j in joints)
    rows = lines[i:]
    if len(rows) != n_frames:
        raise ValueError(f"BVH {path.name} : {len(rows)} lignes de mouvement pour Frames={n_frames}")
    if any(len(r.split()) != n_channels for r in rows):
        raise ValueError(f"BVH {path.name} : chaque frame doit avoir {n_channels} valeurs")
    try:
        motion = np.array([[float(x) for x in r.split()] for r in rows], dtype=float).reshape(n_frames, n_channels)
    except ValueError:
        motion = np.array([np.nan])
    if not np.isfinite(motion).all():
        raise ValueError(f"BVH {path.name} : valeurs de mouvement non numeriques ou non finies")
    return joints, frame_time, motion


def _euler_bvh_to_matrix(order: list[str], angles_deg: list[float]) -> np.ndarray:
    """R = R_order[0](a0) @ R_order[1](a1) @ R_order[2](a2), ordre = celui déclaré par le fichier."""
    r = np.eye(3)
    for axis, ang in zip(order, angles_deg):
        r = r @ Rotation.from_euler(axis.lower(), ang, degrees=True).as_matrix()
    return r


def extract_bvh_rotations(
    joints: list[dict[str, Any]], motion: np.ndarray
) -> dict[str, list[Rotation]]:
    """Reconstruit, pour chaque joint, la liste des rotations locales (une par frame)."""
    col = 0
    col_of: dict[str, tuple[int, list[str]]] = {}
    for j in joints:
        col_of[j["name"]] = (col, j["channels"])
        col += len(j["channels"])

    out: dict[str, list[Rotation]] = {}
    for j in joints:
        start, channels = col_of[j["name"]]
        rot_order = [c[0] for c in channels if c.endswith("rotation")]
        rot_cols = [start + k for k, c in enumerate(channels) if c.endswith("rotation")]
        if not rot_order:
            continue
        rots = []
        for frame in motion:
            angles = [frame[c] for c in rot_cols]
            rots.append(Rotation.from_matrix(_euler_bvh_to_matrix(rot_order, angles)))
        out[j["name"]] = rots
    return out


# --------------------------------------------------------------------------
# 2. Lecture de la pose de repos du modèle Mixamo cible
# --------------------------------------------------------------------------
class _GltfRest:
    """Pose de repos d'un glTF : parents, rotations locales / monde, positions monde."""

    def __init__(self, gltf: Any):
        if gltf.nodes is None or not gltf.nodes:
            raise ValueError("glTF sans noeuds")
        nodes = gltf.nodes
        n = len(nodes)
        self.parent: list[int | None] = [None] * n
        for idx, node in enumerate(nodes):
            for child in node.children or []:
                self.parent[child] = idx
        for node in nodes:
            s = np.asarray(node.scale if node.scale else [1.0, 1.0, 1.0], dtype=float)
            if (s <= 0).any() or np.ptp(s) > 1e-3 * s.max():
                raise ValueError(f"noeud {node.name!r} : echelle {s.tolist()} non uniforme ou negative, non supportee")
        self.names = [node.name for node in nodes]
        self.local = [Rotation.from_quat(node.rotation) if node.rotation else Rotation.identity() for node in nodes]
        self.order: list[int] = []
        self.world: list[Rotation | None] = [None] * n
        self.position = np.zeros((n, 3))
        mats: dict[int, np.ndarray] = {}

        def visit(i: int) -> None:
            node = nodes[i]
            m = np.eye(4)
            s = node.scale[0] if node.scale else 1.0
            m[:3, :3] = self.local[i].as_matrix() * s
            m[:3, 3] = node.translation or [0.0, 0.0, 0.0]
            p = self.parent[i]
            mats[i] = m if p is None else mats[p] @ m
            self.world[i] = self.local[i] if p is None else self.world[p] * self.local[i]
            self.position[i] = mats[i][:3, 3]
            self.order.append(i)
            for c in node.children or []:
                visit(c)

        for i in range(n):
            if self.parent[i] is None:
                visit(i)
        self.index = {name: i for i, name in enumerate(self.names) if name}


def _load_gltf(path: str | Path):
    from pygltflib import GLTF2

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Modele cible introuvable : {path}")
    suffix = path.suffix.lower()
    if suffix in (".glb", ".vrm"):
        with path.open("rb") as f:
            if f.read(4) != b"glTF":
                raise ValueError(f"{path.name} n'est pas un glTF binaire (en-tete 'glTF' absent)")
        return GLTF2().load_binary(str(path))
    if suffix == ".gltf":
        try:
            return GLTF2().load(str(path))
        except (ValueError, KeyError) as exc:
            raise ValueError(f"{path.name} : glTF JSON illisible ({exc})") from exc
    raise ValueError(f"{path.name} : extension {suffix or '(aucune)'} non supportee, attendu .glb, .gltf ou .vrm")


def read_gltf_rest_pose(path: str | Path) -> dict[str, tuple[Rotation, Rotation]]:
    """(rotation locale, rotation monde) de repos pour chaque noeud nomme."""
    rest = _GltfRest(_load_gltf(path))
    return {name: (rest.local[i], rest.world[i]) for name, i in rest.index.items()}


# --------------------------------------------------------------------------
# 3. Table de correspondance BVH -> Mixamo (à ajuster à ta nomenclature exacte)
# --------------------------------------------------------------------------
BVH_TO_MIXAMO: dict[str, str] = {
    "MID_HIP": "mixamorig:Hips",
    "MID_SHOULDER": "mixamorig:Spine2",
    "HEAD": "mixamorig:Head",
    "LEFT_SHOULDER": "mixamorig:LeftArm",
    "LEFT_ELBOW": "mixamorig:LeftForeArm",
    "LEFT_WRIST": "mixamorig:LeftHand",
    "RIGHT_SHOULDER": "mixamorig:RightArm",
    "RIGHT_ELBOW": "mixamorig:RightForeArm",
    "RIGHT_WRIST": "mixamorig:RightHand",
    "LEFT_HIP": "mixamorig:LeftUpLeg",
    "LEFT_KNEE": "mixamorig:LeftLeg",
    "LEFT_ANKLE": "mixamorig:LeftFoot",
    "RIGHT_HIP": "mixamorig:RightUpLeg",
    "RIGHT_KNEE": "mixamorig:RightLeg",
    "RIGHT_ANKLE": "mixamorig:RightFoot",
}
# Doigts : LEFT_HAND_THUMB_CMC -> mixamorig:LeftHandThumb1, etc. (CMC/MCP/IP/TIP ou
# MCP/PIP/DIP/TIP selon le doigt, cf. _FINGERS dans le script BVH).
_FINGER_MIXAMO_SUFFIX = {"THUMB": ["1", "2", "3"], "INDEX_FINGER": ["1", "2", "3"],
                         "MIDDLE_FINGER": ["1", "2", "3"], "RING_FINGER": ["1", "2", "3"], "PINKY": ["1", "2", "3"]}
_FINGER_MIXAMO_NAME = {"THUMB": "Thumb", "INDEX_FINGER": "Index", "MIDDLE_FINGER": "Middle",
                       "RING_FINGER": "Ring", "PINKY": "Pinky"}
for _side_bvh, _side_mixamo in (("LEFT", "Left"), ("RIGHT", "Right")):
    for _finger, _suffixes in _FINGER_MIXAMO_SUFFIX.items():
        _joints = ["CMC", "MCP", "IP"] if _finger == "THUMB" else ["MCP", "PIP", "DIP"]
        for _joint, _suffix in zip(_joints, _suffixes):
            _bvh_name = f"{_side_bvh}_HAND_{_finger}_{_joint}"
            BVH_TO_MIXAMO[_bvh_name] = f"mixamorig:{_side_mixamo}Hand{_FINGER_MIXAMO_NAME[_finger]}{_suffix}"


def _default_rest_alignment() -> dict[str, list[tuple[tuple[str, str], tuple[str, str]]]]:
    """Par os Mixamo : [(segment BVH, segment Mixamo)], le 1er donne la direction (exacte),
    le 2e (optionnel) fixe la torsion autour de cette direction."""
    m = "mixamorig:"
    out: dict[str, list[tuple[tuple[str, str], tuple[str, str]]]] = {
        m + "Hips": [(("MID_HIP", "MID_SHOULDER"), (m + "Hips", m + "Spine2")),
                     (("RIGHT_HIP", "LEFT_HIP"), (m + "RightUpLeg", m + "LeftUpLeg"))],
        m + "Spine2": [(("MID_SHOULDER", "HEAD"), (m + "Spine2", m + "Head")),
                       (("RIGHT_SHOULDER", "LEFT_SHOULDER"), (m + "RightArm", m + "LeftArm"))],
    }
    for S, s in (("LEFT", "Left"), ("RIGHT", "Right")):
        out[f"{m}{s}Arm"] = [((f"{S}_SHOULDER", f"{S}_ELBOW"), (f"{m}{s}Arm", f"{m}{s}ForeArm"))]
        out[f"{m}{s}ForeArm"] = [((f"{S}_ELBOW", f"{S}_WRIST"), (f"{m}{s}ForeArm", f"{m}{s}Hand"))]
        out[f"{m}{s}Hand"] = [
            ((f"{S}_WRIST", f"{S}_HAND_MIDDLE_FINGER_MCP"), (f"{m}{s}Hand", f"{m}{s}HandMiddle1")),
            ((f"{S}_HAND_INDEX_FINGER_MCP", f"{S}_HAND_PINKY_MCP"), (f"{m}{s}HandIndex1", f"{m}{s}HandPinky1")),
        ]
        out[f"{m}{s}UpLeg"] = [((f"{S}_HIP", f"{S}_KNEE"), (f"{m}{s}UpLeg", f"{m}{s}Leg"))]
        out[f"{m}{s}Leg"] = [((f"{S}_KNEE", f"{S}_ANKLE"), (f"{m}{s}Leg", f"{m}{s}Foot"))]
        out[f"{m}{s}Foot"] = [((f"{S}_ANKLE", f"{S}_FOOT_INDEX"), (f"{m}{s}Foot", f"{m}{s}ToeBase"))]
        for F, f in _FINGER_MIXAMO_NAME.items():
            js = ["CMC", "MCP", "IP", "TIP"] if F == "THUMB" else ["MCP", "PIP", "DIP", "TIP"]
            for k in range(3):
                out[f"{m}{s}Hand{f}{k + 1}"] = [((f"{S}_HAND_{F}_{js[k]}", f"{S}_HAND_{F}_{js[k + 1]}"),
                                                 (f"{m}{s}Hand{f}{k + 1}", f"{m}{s}Hand{f}{k + 2}"))]
    return out


REST_ALIGNMENT = _default_rest_alignment()


def _unit(v: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 and np.isfinite(n) else None


def _arc(a: np.ndarray, b: np.ndarray) -> Rotation:
    """Rotation minimale envoyant la direction unitaire a sur b."""
    axis = np.cross(a, b)
    s, c = float(np.linalg.norm(axis)), float(np.clip(np.dot(a, b), -1.0, 1.0))
    if s < 1e-12:
        if c > 0:
            return Rotation.identity()
        perp = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(a, [0.0, 1.0, 0.0])
        return Rotation.from_rotvec(perp / np.linalg.norm(perp) * np.pi)
    return Rotation.from_rotvec(axis / s * np.arctan2(s, c))


def _alignment(pairs: list[tuple[np.ndarray, np.ndarray]]) -> Rotation:
    """C tel que C · b1 = a1 exactement, puis torsion autour de a1 pour rapprocher C · b2 de a2."""
    b1, a1 = pairs[0]
    c = _arc(b1, a1)
    if len(pairs) > 1:
        b2, a2 = pairs[1]
        p = c.apply(b2) - np.dot(c.apply(b2), a1) * a1
        q = a2 - np.dot(a2, a1) * a1
        if np.linalg.norm(p) > 1e-6 and np.linalg.norm(q) > 1e-6:
            p, q = p / np.linalg.norm(p), q / np.linalg.norm(q)
            angle = np.arctan2(np.dot(np.cross(p, q), a1), np.clip(np.dot(p, q), -1.0, 1.0))
            c = Rotation.from_rotvec(a1 * angle) * c
    return c


def _bvh_rest_positions(joints: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    pos: dict[str, np.ndarray] = {}
    for j in joints:
        base = np.zeros(3) if j["parent"] is None else pos[j["parent"]]
        pos[j["name"]] = base + j["offset"]
    return pos


# --------------------------------------------------------------------------
# 4. Retargeting
# --------------------------------------------------------------------------
def retarget_bvh_to_mixamo(
    bvh_path: str | Path,
    target_glb_path: str | Path,
    bone_map: dict[str, str] | None = None,
    align_rest: bool = True,
    rest_alignment: dict[str, list[tuple[tuple[str, str], tuple[str, str]]]] | None = None,
) -> dict[str, Any]:
    """
    Q_B(t) = G_A(t) · C_B · W_B ; B.local = Q_parent(t)⁻¹ · Q_B(t) (voir docstring du module).
    align_rest=False : C_B = identite, ce qui reproduit B.local = L_B · W_B⁻¹ · N · W_B quand
    les hierarchies se correspondent 1:1.
    """
    bone_map = bone_map or BVH_TO_MIXAMO
    rest_alignment = REST_ALIGNMENT if rest_alignment is None else rest_alignment
    joints, frame_time, motion = parse_bvh(bvh_path)
    T = len(motion)
    local_rot = {k: Rotation.concatenate(v) for k, v in extract_bvh_rotations(joints, motion).items()}
    identity_T = Rotation.from_quat(np.tile([0.0, 0.0, 0.0, 1.0], (T, 1)))
    G: dict[str, Rotation] = {}
    for j in joints:
        loc = local_rot.get(j["name"], identity_T)
        G[j["name"]] = loc if j["parent"] is None else G[j["parent"]] * loc
    bvh_rest = _bvh_rest_positions(joints)

    rest = _GltfRest(_load_gltf(target_glb_path))
    fps = 1.0 / frame_time
    skipped = [mix for bvh, mix in bone_map.items() if bvh in G and mix not in rest.index]
    driven = {rest.index[mix]: bvh for bvh, mix in bone_map.items() if bvh in G and mix in rest.index}
    if not driven:
        raise ValueError(f"aucun os du BVH ne correspond a un noeud de {Path(target_glb_path).name} "
                         f"(os attendus du type {next(iter(bone_map.values()), '?')})")

    correction: dict[int, Rotation] = {}
    alignment_deg: dict[str, float | None] = {}
    for i, bvh in driven.items():
        name = rest.names[i]
        pairs = []
        for (ba, bb), (ma, mb) in (rest_alignment.get(name, []) if align_rest else []):
            if ba in bvh_rest and bb in bvh_rest and ma in rest.index and mb in rest.index:
                a = _unit(bvh_rest[bb] - bvh_rest[ba])
                b = _unit(rest.position[rest.index[mb]] - rest.position[rest.index[ma]])
                if a is not None and b is not None:
                    pairs.append((b, a))
        correction[i] = _alignment(pairs) if pairs else Rotation.identity()
        alignment_deg[name] = round(float(np.degrees(correction[i].magnitude())), 2) if pairs else None

    world: dict[int, Rotation] = {}
    tracks: dict[str, list[dict[str, Any]]] = {}
    for i in rest.order:
        p = rest.parent[i]
        parent_world = identity_T if p is None else world[p]
        if i in driven:
            world[i] = G[driven[i]] * (correction[i] * rest.world[i])
            local = parent_world.inv() * world[i]
            quats = local.as_quat()
            tracks[rest.names[i]] = [{"time": t / fps, "rotation": quats[t].tolist()} for t in range(T)]
        else:
            world[i] = parent_world * rest.local[i]

    _finalize_tracks(tracks)

    return {
        "format": "mimo.mixamo.retarget",
        "fps": fps,
        "frameCount": T,
        "tracks": tracks,
        "meta": {
            "source": str(bvh_path),
            "target": str(target_glb_path),
            "bonesSkipped": skipped,
            "restAlignment": align_rest,
            "restCorrectionDeg": alignment_deg,
            "rootMotion": "aucune (translation du BVH non transferee)",
        },
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Usage: python -m backend.retarget_bvh_to_mixamo capture.bvh Clara.glb sortie.json")
        sys.exit(1)
    result = retarget_bvh_to_mixamo(sys.argv[1], sys.argv[2])
    Path(sys.argv[3]).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{len(result['tracks'])} os retargetés, {len(result['meta']['bonesSkipped'])} introuvables dans la cible.")
