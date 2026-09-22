"""Export BVH natif MediaPipe (Pose 33 points + mains 21 points) -- sans Mixamo.

But : verifier si MediaPipe recupere correctement le mouvement AVANT de faire
un retargeting vers Mixamo.

Corps :
  * body="full"  : racine = MID_HIP (bassin), jambes incluses.
  * body="upper" : racine = MID_SHOULDER, AUCUNE jambe/bassin. Pour les videos
                   ou seul le haut du corps est visible (ex. langue des signes).
  * body="auto"  : "upper" si les hanches/genoux/chevilles sont peu visibles.

Mains : hands="auto"/"on"/"off". Si les mains Holistic (handsL / handsR, 21 pts)
sont detectees, chaque main devient une chaine de 20 articulations attachee au
poignet de la Pose. Sinon on retombe sur les 3 points de la Pose (pouce, index,
auriculaire).

Modes de sortie :
  * "positions" : 3 canaux de position par point (aucun calcul de rotation).
  * "rotations" : BVH classique (rotations calculees par Kabsch / arc minimal),
                  avec erreur de reconstruction dans le rapport.

Repere de sortie : Y haut, la personne regarde +Z, sa GAUCHE est +X.
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np

POSE_LANDMARKS = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER", "LEFT_EAR", "RIGHT_EAR",
    "MOUTH_LEFT", "MOUTH_RIGHT", "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY",
    "RIGHT_PINKY", "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB",
    "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE",
    "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX",
    "RIGHT_FOOT_INDEX",
]
_IDX = {n: i for i, n in enumerate(POSE_LANDMARKS)}
_LOWER_IDX = [23, 24, 25, 26, 27, 28]

# (nom, parent, source, direction de repos en T-pose)
# source = ("lm", i) | ("mid", i, j) | ("hand", "L"/"R", i)
_SKELETON = [
    ("MID_HIP", None, ("mid", 23, 24), (0, 0, 0)),
    ("LEFT_HIP", "MID_HIP", ("lm", 23), (1, 0, 0)),
    ("RIGHT_HIP", "MID_HIP", ("lm", 24), (-1, 0, 0)),
    ("MID_SHOULDER", "MID_HIP", ("mid", 11, 12), (0, 1, 0)),
    ("LEFT_KNEE", "LEFT_HIP", ("lm", 25), (0, -1, 0)),
    ("LEFT_ANKLE", "LEFT_KNEE", ("lm", 27), (0, -1, 0)),
    ("LEFT_HEEL", "LEFT_ANKLE", ("lm", 29), (0, -0.5, -0.866)),
    ("LEFT_FOOT_INDEX", "LEFT_ANKLE", ("lm", 31), (0, -0.4, 0.917)),
    ("RIGHT_KNEE", "RIGHT_HIP", ("lm", 26), (0, -1, 0)),
    ("RIGHT_ANKLE", "RIGHT_KNEE", ("lm", 28), (0, -1, 0)),
    ("RIGHT_HEEL", "RIGHT_ANKLE", ("lm", 30), (0, -0.5, -0.866)),
    ("RIGHT_FOOT_INDEX", "RIGHT_ANKLE", ("lm", 32), (0, -0.4, 0.917)),
    ("LEFT_SHOULDER", "MID_SHOULDER", ("lm", 11), (1, 0, 0)),
    ("LEFT_ELBOW", "LEFT_SHOULDER", ("lm", 13), (1, 0, 0)),
    ("LEFT_WRIST", "LEFT_ELBOW", ("lm", 15), (1, 0, 0)),
    ("LEFT_PINKY", "LEFT_WRIST", ("lm", 17), (0.96, 0, -0.28)),
    ("LEFT_INDEX", "LEFT_WRIST", ("lm", 19), (0.96, 0, 0.28)),
    ("LEFT_THUMB", "LEFT_WRIST", ("lm", 21), (0.6, 0, 0.8)),
    ("RIGHT_SHOULDER", "MID_SHOULDER", ("lm", 12), (-1, 0, 0)),
    ("RIGHT_ELBOW", "RIGHT_SHOULDER", ("lm", 14), (-1, 0, 0)),
    ("RIGHT_WRIST", "RIGHT_ELBOW", ("lm", 16), (-1, 0, 0)),
    ("RIGHT_PINKY", "RIGHT_WRIST", ("lm", 18), (-0.96, 0, -0.28)),
    ("RIGHT_INDEX", "RIGHT_WRIST", ("lm", 20), (-0.96, 0, 0.28)),
    ("RIGHT_THUMB", "RIGHT_WRIST", ("lm", 22), (-0.6, 0, 0.8)),
    ("HEAD", "MID_SHOULDER", ("mid", 7, 8), (0, 1, 0)),
    ("LEFT_EAR", "HEAD", ("lm", 7), (1, 0, 0)),
    ("RIGHT_EAR", "HEAD", ("lm", 8), (-1, 0, 0)),
    ("NOSE", "HEAD", ("lm", 0), (0, -0.2, 0.98)),
]
_LOWER_NAMES = {"MID_HIP", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "LEFT_ANKLE",
                "LEFT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_KNEE", "RIGHT_ANKLE",
                "RIGHT_HEEL", "RIGHT_FOOT_INDEX"}
_POSE_HAND_SIDE = {"LEFT_PINKY": "L", "LEFT_INDEX": "L", "LEFT_THUMB": "L",
                   "RIGHT_PINKY": "R", "RIGHT_INDEX": "R", "RIGHT_THUMB": "R"}
# (doigt, articulations, indice du 1er point main, dir poignet->1er point, dir suivantes)
_FINGERS = [
    ("THUMB", ["CMC", "MCP", "IP", "TIP"], 1, (0.55, 0, 0.83), (0.75, 0, 0.66)),
    ("INDEX_FINGER", ["MCP", "PIP", "DIP", "TIP"], 5, (0.95, 0, 0.30), (1, 0, 0)),
    ("MIDDLE_FINGER", ["MCP", "PIP", "DIP", "TIP"], 9, (0.99, 0, 0.10), (1, 0, 0)),
    ("RING_FINGER", ["MCP", "PIP", "DIP", "TIP"], 13, (0.99, 0, -0.10), (1, 0, 0)),
    ("PINKY", ["MCP", "PIP", "DIP", "TIP"], 17, (0.95, 0, -0.30), (1, 0, 0)),
]


class _Skel:
    def __init__(self, entries):
        self.names = [e[0] for e in entries]
        self.parent = {e[0]: e[1] for e in entries}
        self.src = {e[0]: e[2] for e in entries}
        self.rest = {e[0]: np.array(e[3], dtype=float) for e in entries}
        self.children = {n: [] for n in self.names}
        for n, p in self.parent.items():
            if p is not None:
                self.children[p].append(n)
        self.root = self.names[0]
        self.ground = [n for n in ("LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
                                   "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX") if n in self.parent]


def _hand_chain(side: str, wrist: str):
    sgn, pref = (1.0, "LEFT") if side == "L" else (-1.0, "RIGHT")
    out = []
    for finger, joints, first, d0, d1 in _FINGERS:
        parent = wrist
        for k, jn in enumerate(joints):
            d = d0 if k == 0 else d1
            name = f"{pref}_HAND_{finger}_{jn}"
            out.append((name, parent, ("hand", side, first + k), (sgn * d[0], d[1], d[2])))
            parent = name
    return out


def _build_skeleton(body: str = "full", hand_sides: tuple[str, ...] = ()) -> _Skel:
    entries = []
    for name, parent, src, rd in _SKELETON:
        if body == "upper":
            if name in _LOWER_NAMES:
                continue
            if name == "MID_SHOULDER":
                parent, rd = None, (0, 0, 0)
        if _POSE_HAND_SIDE.get(name) in hand_sides:
            continue                                   # remplace par la chaine de la main
        entries.append((name, parent, src, rd))
        if name in ("LEFT_WRIST", "RIGHT_WRIST"):
            side = "L" if name.startswith("LEFT") else "R"
            if side in hand_sides:
                entries.extend(_hand_chain(side, name))
    return _Skel(entries)


# Squelette "complet" (compatibilite avec la version precedente et les tests)
_FULL = _build_skeleton("full", ())
ROOT = _FULL.root
_NAMES, _PARENT, _REST_DIR, _CHILDREN, _FEET = (
    _FULL.names, _FULL.parent, _FULL.rest, _FULL.children, _FULL.ground)


# --------------------------------------------------------------------------
# 1. Lecture des frames (tolerant : dict / objet / liste / tableau numpy)
# --------------------------------------------------------------------------
def _read_landmark(lm: Any):
    """-> (x, y, z, visibilite ou None) ; None si illisible."""
    if lm is None:
        return None
    try:
        if isinstance(lm, dict):
            x, y = lm["x"], lm["y"]
            z = lm.get("z", 0.0)
            v = lm.get("visibility", lm.get("presence"))
        elif hasattr(lm, "x"):
            x, y, z = lm.x, lm.y, getattr(lm, "z", 0.0)
            v = getattr(lm, "visibility", None)
        else:
            seq = list(lm)
            if len(seq) < 2:
                return None
            x, y = seq[0], seq[1]
            z = seq[2] if len(seq) > 2 else 0.0
            v = seq[3] if len(seq) > 3 else None
        return float(x), float(y), float(z), (None if v is None else float(v))
    except (KeyError, TypeError, ValueError):
        return None



def describe_frames(frames: list[dict[str, Any]]) -> str:
    """Aide au debogage : montre le format reel des frames, SANS JAMAIS planter,
    quelle que soit la forme reelle (liste attendue, mais aussi dict, None, etc.)."""
    if not frames:
        return "liste de frames vide"
    for f in frames:
        if not isinstance(f, dict):
            return f"une frame n'est pas un dict : type={type(f).__name__}, valeur={f!r}"
        pose = f.get("bodyPose")
        if pose is None:
            continue
        txt = f"cles d'une frame : {sorted(f.keys())}\\nbodyPose : type={type(pose).__name__}"
        if isinstance(pose, dict):
            txt += (f", cles={sorted(pose.keys())}\\n"
                    "  -> bodyPose est un DICT, pas une liste de points. bvh_export attend "
                    "frame['bodyPose'] = liste de 33 points directement. Il faut probablement "
                    "extraire bodyPose['predictions'] (ou la cle equivalente) avant l'export.")
            return txt
        if isinstance(pose, list) and len(pose) == 0:
            continue
        try:
            length = len(pose)
        except TypeError:
            return txt + f", non indexable, valeur={pose!r}"
        txt += f", longueur={length}"
        try:
            txt += f"\\n  premier point : {pose[0]!r}"
        except Exception as e:
            txt += f"\\n  (impossible de lire pose[0] : {e})"
        for key in ("handsL", "handsR"):
            h = f.get(key)
            if h is None:
                txt += f"\\n{key} : absent de la frame (verifie s'il n'est pas range ailleurs, ex. 'handsPose')"
            elif isinstance(h, dict):
                txt += f"\\n{key} : est un DICT (cles={sorted(h.keys())}), pas une liste de points"
            else:
                txt += f"\\n{key} : {'vide' if len(h) == 0 else f'longueur={len(h)}, premier point={h[0]!r}'}"
        return txt
    return "aucune frame avec bodyPose non vide (toutes ont bodyPose=None ou liste vide)"


def _collect_pose(frames, min_visibility):
    T = len(frames)
    arr = np.full((T, 33, 3), np.nan)
    without, saw_visibility, extra_points_seen = 0, False, 0
    for t, frame in enumerate(frames):
        pose = frame.get("bodyPose")
        if isinstance(pose, dict):
            raise ValueError(
                "bodyPose est un dict, pas une liste de 33 points (ex. {'predictions': [...], ...}). "
                "Il faut d'abord extraire la liste, par ex. frame['bodyPose']['predictions'].\\n"
                + describe_frames(frames))
        if pose is None or len(pose) == 0:
            without += 1
            continue
        if len(pose) < 33:
            raise ValueError(f"bodyPose a seulement {len(pose)} points, 33 attendus au minimum "
                             "(MediaPipe Pose).\\n" + describe_frames(frames))
        if len(pose) > 33:
            extra_points_seen = max(extra_points_seen, len(pose) - 33)
            pose = pose[:33]          # ex. add_extra_points() ajoute des points a la fin : ignores ici
        for i, lm in enumerate(pose):
            row = _read_landmark(lm)
            if row is None:
                continue
            x, y, z, v = row
            if v is not None:
                saw_visibility = True
            if (v is not None and v < min_visibility) or not all(math.isfinite(c) for c in (x, y, z)):
                continue
            arr[t, i] = (x, y, z)
    if not np.isfinite(arr).any():
        raise ValueError("Aucun point exploitable dans les frames.\n" + describe_frames(frames))
    return arr, without, saw_visibility


def _collect_hand(frames, key):
    arr = np.full((len(frames), 21, 3), np.nan)
    for t, frame in enumerate(frames):
        h = frame.get(key)
        if h is None or len(h) != 21:
            continue
        for i, lm in enumerate(h):
            row = _read_landmark(lm)
            if row is not None and all(math.isfinite(c) for c in row[:3]):
                arr[t, i] = row[:3]
    return arr


def _detect_body(raw: np.ndarray) -> str:
    return "upper" if float(np.isfinite(raw[:, _LOWER_IDX, 0]).mean()) < 0.5 else "full"


def _detect_source(arr: np.ndarray) -> str:
    """world = metres centres sur les hanches ; normalized = coordonnees image 0..1."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        hips = np.nanmean(arr[:, [23, 24], :2], axis=1)
        shoulders = np.nanmean(arr[:, [11, 12], :2], axis=1)
        ref = hips if np.isfinite(hips).all(axis=1).mean() >= 0.5 else shoulders
        med = np.nanmedian(ref, axis=0)
    if not np.all(np.isfinite(med)):
        raise ValueError("Ni hanches ni epaules detectees : impossible de deviner la source.")
    return "world" if (abs(med[0]) < 0.15 and med[1] < 0.15) else "normalized"


def _to_output_space(arr, source, aspect, scale, flip):
    p = arr.copy()
    if source == "normalized":
        p[..., 0] *= aspect
        p[..., 2] *= aspect
    p *= scale
    p[..., 1] *= -1.0                # image : y vers le bas -> Y vers le haut
    p[..., 2] *= -1.0                # z camera (loin) -> la personne regarde +Z
    for axis, do_flip in enumerate(flip):
        if do_flip:
            p[..., axis] *= -1.0
    return p


def _joint_positions(p: np.ndarray, sk: _Skel | None = None,
                     hand_rel: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    sk = sk or _FULL
    out: dict[str, np.ndarray] = {}
    for name in sk.names:
        src = sk.src[name]
        if src[0] == "lm":
            out[name] = p[:, src[1]]
        elif src[0] == "mid":
            out[name] = (p[:, src[1]] + p[:, src[2]]) / 2.0
        else:
            wrist = "LEFT_WRIST" if src[1] == "L" else "RIGHT_WRIST"
            out[name] = out[wrist] + hand_rel[src[1]][:, src[2]]
    return out


def _hold_fill(a: np.ndarray, name: str) -> tuple[np.ndarray, float]:
    valid = np.isfinite(a).all(axis=1)
    if not valid.any():
        raise ValueError(f"Le point {name} n'est jamais detecte (visibilite trop basse ?).")
    idx = np.where(valid, np.arange(len(a)), -1)
    np.maximum.accumulate(idx, out=idx)
    idx[idx < 0] = int(np.argmax(valid))
    return a[idx], float(1.0 - valid.mean())


# --------------------------------------------------------------------------
# 2. Maths : rotations
# --------------------------------------------------------------------------
def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _arc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation de plus petit angle qui envoie a sur b (torsion indeterminee)."""
    a, b = _unit(a), _unit(b)
    c = float(np.dot(a, b))
    if c > 1.0 - 1e-9:
        return np.eye(3)
    if c < -1.0 + 1e-9:
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis = _unit(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    v = np.cross(a, b)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + (k @ k) / (1.0 + c)


def _kabsch(rest, obs):
    h = sum(np.outer(o, v) for o, v in zip(rest, obs))
    u, s, vt = np.linalg.svd(h)
    if s[1] < 1e-6 * max(s[0], 1e-12):
        return None                   # vecteurs (quasi) colineaires
    d = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T


def _global_rotation(rest, obs):
    rest = [_unit(r) for r in rest]
    obs = [_unit(o) for o in obs]
    if len(rest) >= 2:
        r = _kabsch(rest, obs)
        if r is not None:
            return r
    return _arc(rest[0], obs[0])


def _wrap(x: float) -> float:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def _euler_zxy(R: np.ndarray, prev):
    """R = Rz(a) @ Rx(b) @ Ry(c)  (canaux BVH 'Zrotation Xrotation Yrotation')."""
    sb = float(np.clip(R[2, 1], -1.0, 1.0))
    b = math.asin(sb)
    if abs(sb) < 0.999999:
        a = math.atan2(-R[0, 1], R[1, 1])
        c = math.atan2(-R[2, 0], R[2, 2])
        cands = [(a, b, c), (a + math.pi, math.pi - b, c + math.pi)]
    else:                              # blocage de cardan
        cands = [(math.atan2(R[1, 0], R[0, 0]), b, 0.0)]
    if prev is None:
        return cands[0]
    best, best_cost = None, float("inf")
    for cand in cands:                 # continuite temporelle (pas de saut de 360 deg)
        adj = tuple(p + _wrap(v - p) for v, p in zip(cand, prev))
        cost = sum(abs(x - p) for x, p in zip(adj, prev))
        if cost < best_cost:
            best, best_cost = adj, cost
    return best


# --------------------------------------------------------------------------
# 3. Export principal
# --------------------------------------------------------------------------
def export_mediapipe_bvh(
    frames: list[dict[str, Any]],
    output_path: str | Path,
    fps: float,
    mode: str = "rotations",
    source: str = "auto",
    aspect: float = 1.0,
    scale: float = 100.0,
    min_visibility: float = 0.5,
    flip: tuple[bool, bool, bool] = (False, False, False),
    recenter: bool = True,
    extra: dict[str, Any] | None = None,
    body: str = "auto",
    hands: str = "auto",
    hand_size_m: float = 0.095,
) -> dict[str, Any]:
    if mode not in ("rotations", "positions"):
        raise ValueError("mode doit etre 'rotations' ou 'positions'")
    if body not in ("auto", "full", "upper") or hands not in ("auto", "on", "off"):
        raise ValueError("body: auto|full|upper ; hands: auto|on|off")
    if fps <= 0:
        raise ValueError("fps doit etre positif")
    if not frames:
        raise ValueError("Aucune frame a exporter")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    warns: list[str] = []

    raw, frames_without_body, saw_visibility = _collect_pose(frames, min_visibility)
    T = raw.shape[0]
    body_used = _detect_body(raw) if body == "auto" else body
    if body_used == "full" and not saw_visibility:
        warns.append("Aucun champ 'visibility' dans bodyPose : impossible de savoir si les jambes "
                     "sont reellement visibles. Si seul le haut du corps est visible, "
                     "utilise --body upper (sinon les jambes seront inventees).")
    if source == "auto":
        source = _detect_source(raw)
    elif source not in ("world", "normalized"):
        raise ValueError("source doit etre 'auto', 'world' ou 'normalized'")

    # --- mains -----------------------------------------------------------------
    hand_arr = {"L": _collect_hand(frames, "handsL"), "R": _collect_hand(frames, "handsR")}
    detected = {s: int(np.isfinite(a).all(axis=(1, 2)).sum()) for s, a in hand_arr.items()}
    sides = () if hands == "off" else tuple(s for s in "LR" if detected[s] >= 5)
    if hands == "on" and not sides:
        raise ValueError("hands='on' mais aucune main (handsL/handsR, 21 points) n'est detectee.")
    if hands != "off":
        for s in "LR":
            if s not in sides:
                warns.append(f"Main {'gauche' if s == 'L' else 'droite'} detectee sur {detected[s]} frame(s) "
                             "seulement : repli sur les 3 points de la Pose (pouce/index/auriculaire).")

    p_raw = _to_output_space(raw, source, aspect, scale, flip)

    rel_out: dict[str, np.ndarray] = {}
    if sides:
        rels = {s: hand_arr[s] - hand_arr[s][:, :1] for s in sides}
        if source == "normalized":
            for s in sides:
                rel_out[s] = _to_output_space(rels[s], "normalized", aspect, scale, flip)
        else:                                           # pose en metres, mains en unites image
            units = {}
            for s in sides:
                u = rels[s].copy()
                u[..., 0] *= aspect
                u[..., 2] *= aspect
                units[s] = u
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                lens = np.concatenate([np.linalg.norm(units[s][:, 9], axis=1) for s in sides])
                med = float(np.nanmedian(lens))
            if not math.isfinite(med) or med <= 0:
                raise ValueError("Impossible d'estimer l'echelle des mains.")
            s_world = hand_size_m / med
            warns.append(f"Source 'world' : l'echelle des mains est ESTIMEE (poignet->milieu de la paume "
                         f"= {hand_size_m * 100:.1f} cm suppose). Les proportions de la main sont fiables, "
                         "sa taille absolue est approximative.")
            for s in sides:
                rel_out[s] = _to_output_space(units[s] * s_world, "world", 1.0, scale, flip)
        if source == "normalized":                      # controle gauche/droite
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                for s in sides:
                    own, other = (15, 16) if s == "L" else (16, 15)
                    k = np.array([aspect, 1.0])
                    d_own = np.nanmedian(np.linalg.norm((hand_arr[s][:, 0, :2] - raw[:, own, :2]) * k, axis=1))
                    d_oth = np.nanmedian(np.linalg.norm((hand_arr[s][:, 0, :2] - raw[:, other, :2]) * k, axis=1))
                    if math.isfinite(d_own) and math.isfinite(d_oth) and d_oth < 0.5 * d_own:
                        warns.append(f"handsL/handsR : la main '{s}' est plus proche du poignet oppose : "
                                     "les mains sont peut-etre INVERSEES dans ton pipeline.")

    sk = _build_skeleton(body_used, sides)
    pos_raw = _joint_positions(p_raw, sk, rel_out)

    rel_fill: dict[str, np.ndarray] = {}
    rel_held: dict[str, np.ndarray] = {}
    for s in sides:
        cols, hl = [], []
        for i in range(21):
            if i == 0:
                cols.append(np.zeros((T, 3)))
                hl.append(0.0)
                continue
            f_, h_ = _hold_fill(rel_out[s][:, i], f"main {s}[{i}]")
            cols.append(f_)
            hl.append(h_)
        rel_fill[s] = np.stack(cols, axis=1)
        rel_held[s] = np.array(hl)

    pos: dict[str, np.ndarray] = {}
    held: dict[str, float] = {}
    for name in sk.names:
        src = sk.src[name]
        if src[0] == "hand":
            wrist = "LEFT_WRIST" if src[1] == "L" else "RIGHT_WRIST"
            pos[name] = pos[wrist] + rel_fill[src[1]][:, src[2]]
            held[name] = float(rel_held[src[1]][src[2]])
        else:
            pos[name], held[name] = _hold_fill(pos_raw[name], name)

    if recenter:
        if sk.ground:                  # corps entier : depart en (0,0) et pieds sur le sol
            r0 = pos[sk.root][0]
            shift = np.array([r0[0], min(float(pos[n][:, 1].min()) for n in sk.ground), r0[2]])
        else:                          # haut du corps : racine a l'origine
            shift = pos[sk.root][0].copy()
        pos = {k: v - shift for k, v in pos.items()}

    # --- longueurs d'os ---------------------------------------------------------
    bone_len, bone_cv = {}, {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for name in sk.names[1:]:
            par = sk.parent[name]
            L = np.linalg.norm(pos_raw[name] - pos_raw[par], axis=1)
            med = float(np.nanmedian(L))
            if not math.isfinite(med) or med <= 0:
                raise ValueError(f"Os {par}->{name} : longueur inconnue.")
            bone_len[name] = med
            bone_cv[f"{par}->{name}"] = round(float(np.nanstd(L) / med), 4)
    if body_used == "full":
        ref_name, ref_len = "torso", bone_len["MID_SHOULDER"]
    else:
        ref_name, ref_len = "shoulder_width", bone_len["LEFT_SHOULDER"] + bone_len["RIGHT_SHOULDER"]

    offsets: dict[str, np.ndarray] = {sk.root: np.zeros(3)}
    for name in sk.names[1:]:
        if mode == "rotations":
            offsets[name] = _unit(sk.rest[name]) * bone_len[name]
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                offsets[name] = np.nanmedian(pos_raw[name] - pos_raw[sk.parent[name]], axis=0)

    # --- canaux -----------------------------------------------------------------
    channels: dict[str, np.ndarray] = {}
    recon: dict[str, float] = {}
    if mode == "positions":
        for name in sk.names[1:]:
            channels[name] = (pos[name] - pos[sk.parent[name]]) - offsets[name]
    else:
        has_kids = [n for n in sk.names if sk.children[n]]
        prev = {n: None for n in has_kids}
        eul = {n: np.zeros((T, 3)) for n in has_kids}
        err = {n: np.zeros(T) for n in sk.names[1:]}
        unit_rest = {n: _unit(sk.rest[n]) for n in sk.names[1:]}
        for t in range(T):
            G: dict[str, np.ndarray] = {}
            for name in has_kids:
                kids = sk.children[name]
                obs = [pos[k][t] - pos[name][t] for k in kids]
                G[name] = _global_rotation([unit_rest[k] for k in kids], obs)
                local = G[name] if name == sk.root else G[sk.parent[name]].T @ G[name]
                a, b, c = _euler_zxy(local, prev[name])
                prev[name] = (a, b, c)
                eul[name][t] = np.degrees((a, b, c))
            hat = {sk.root: pos[sk.root][t]}
            for name in sk.names[1:]:
                par = sk.parent[name]
                hat[name] = hat[par] + G[par] @ offsets[name]
                err[name][t] = np.linalg.norm(hat[name] - pos[name][t])
        channels = eul
        for name in sk.names[1:]:
            recon[name] = round(float(err[name].mean() / ref_len * 100), 2)

    # --- ecriture du BVH ----------------------------------------------------------
    lines: list[str] = ["HIERARCHY"]
    order: list[str] = []
    end_sites: dict[str, list[str]] = {}

    def fmt(v):
        return f"{v[0]:.5f} {v[1]:.5f} {v[2]:.5f}"

    def emit(name: str, depth: int) -> None:
        pad = "\t" * depth
        is_root = name == sk.root
        lines.append(f"{pad}{'ROOT' if is_root else 'JOINT'} {name}")
        lines.append(f"{pad}{{")
        lines.append(f"{pad}\tOFFSET {fmt(offsets[name])}")
        if mode == "rotations":
            chan = ("6 Xposition Yposition Zposition Zrotation Xrotation Yrotation"
                    if is_root else "3 Zrotation Xrotation Yrotation")
        else:
            chan = "3 Xposition Yposition Zposition"
        lines.append(f"{pad}\tCHANNELS {chan}")
        order.append(name)
        for kid in sk.children[name]:
            if mode == "rotations" and not sk.children[kid]:
                end_sites.setdefault(name, []).append(kid)
                lines.extend([f"{pad}\tEnd Site", f"{pad}\t{{",
                              f"{pad}\t\tOFFSET {fmt(offsets[kid])}", f"{pad}\t}}"])
            else:
                emit(kid, depth + 1)
        if mode == "positions" and not sk.children[name]:
            tip = _unit(offsets[name]) * 0.02 * ref_len
            lines.extend([f"{pad}\tEnd Site", f"{pad}\t{{", f"{pad}\t\tOFFSET {fmt(tip)}", f"{pad}\t}}"])
        lines.append(f"{pad}}}")

    emit(sk.root, 0)
    if mode == "rotations":
        cols = [pos[sk.root], channels[sk.root]] + [channels[n] for n in order[1:]]
    else:
        cols = [pos[sk.root]] + [channels[n] for n in order[1:]]
    motion = np.hstack(cols)
    lines += ["MOTION", f"Frames: {T}", f"Frame Time: {1.0 / fps:.6f}"]
    lines += [" ".join(f"{v:.5f}" for v in row) for row in motion]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- rapport -------------------------------------------------------------------
    if source == "world" and body_used == "full":
        warns.append("Source 'world' : origine = centre des hanches, donc AUCUNE translation globale.")
    if body_used == "upper":
        warns.append("Mode haut du corps : racine = milieu des epaules, pas de bassin ni de jambes. "
                     "Le torse n'a pas de colonne : seule l'orientation epaules/tete est estimee.")
    long_bones = {k: v for k, v in bone_cv.items() if "_HAND_" not in k}
    unstable = {k: v for k, v in long_bones.items() if v > 0.15}
    if unstable:
        warns.append(f"{len(unstable)} os (hors doigts) ont une longueur tres instable (CV > 15 %) : "
                     "MediaPipe deforme le squelette sur ces os.")
    poorly = [k for k, v in held.items() if v > 0.2]
    if poorly:
        warns.append(f"{len(poorly)} points absents sur >20 % des frames (valeur maintenue), ex. : {poorly[:5]}")
    report = {
        "output": str(output_path), "mode": mode, "body": body_used, "source": source,
        "hands_used": list(sides), "frames": T, "fps": fps, "scale": scale, "aspect": aspect,
        "min_visibility": min_visibility, "has_visibility_field": saw_visibility,
        "frames_without_body": frames_without_body,
        "hand_frames_detected": detected,
        "interpolated_frames": sum(1 for f in frames if f.get("interpolated")),
        "reference_length": ref_name, "reference_length_value": round(ref_len, 3),
        "held_fraction_by_joint": {k: round(v, 3) for k, v in held.items()},
        "bone_length_cv": dict(sorted(bone_cv.items(), key=lambda kv: -kv[1])),
        "reconstruction_error_pct_of_reference": dict(sorted(recon.items(), key=lambda kv: -kv[1])),
        "end_sites": end_sites,
        "warnings": warns,
    }
    if extra:
        report["pipeline"] = extra
    output_path.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


