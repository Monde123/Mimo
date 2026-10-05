"""Capture multi-vues (2 cameras ou plus) : calibration Anipose + triangulation.

Chaine :
  1. calibrate_cameras : videos d'une planche ChArUco -> calibration.toml (format Anipose).
  2. extract_view_2d   : par camera, 33 points Pose + 2 x 21 points Main en PIXELS + confiance.
  3. synchronize       : alignement par indice de frame, decalage optionnel par camera.
  4. triangulate_weighted : DLT ponderee par la confiance, apres correction de la distorsion,
                         rejet par erreur de reprojection et par profondeur negative.
  5. reconstruct_frames : points 3D dans le repere de la camera 0, en metres
                         (y vers le bas, +z loin de la camera : meme convention que MediaPipe world).

Indices des points : 0-32 corps, 33-53 main gauche, 54-74 main droite.
"""
from __future__ import annotations

import contextlib
import io
import itertools
import json
import math
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

N_BODY = 33
N_HAND = 21
LEFT_HAND = slice(N_BODY, N_BODY + N_HAND)
RIGHT_HAND = slice(N_BODY + N_HAND, N_BODY + 2 * N_HAND)
N_KEYPOINTS = N_BODY + 2 * N_HAND
_LEFT_WRIST, _RIGHT_WRIST = 15, 16
_UNITS_TO_M = {"m": 1.0, "cm": 0.01, "mm": 0.001}


def _require_aniposelib():
    try:
        from aniposelib.cameras import CameraGroup
        from aniposelib.boards import CharucoBoard
    except ImportError as exc:
        raise RuntimeError("aniposelib est requis pour le multi-vues : pip install -e .[multiview]") from exc
    return CameraGroup, CharucoBoard


# --------------------------------------------------------------------------
# 1. Calibration
# --------------------------------------------------------------------------
def _natural_key(name: str):
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name)]


def _check_camera_dict(key: str, d: Any) -> None:
    if not isinstance(d, dict):
        raise ValueError(f"calibration : [{key}] n'est pas une table TOML")
    missing = [k for k in ("name", "size", "matrix", "distortions", "rotation", "translation") if k not in d]
    if missing:
        raise ValueError(f"calibration : [{key}] sans les champs {missing}")
    try:
        K = np.asarray(d["matrix"], dtype=float)
        dist = np.asarray(d["distortions"], dtype=float).ravel()
        rvec = np.asarray(d["rotation"], dtype=float).ravel()
        tvec = np.asarray(d["translation"], dtype=float).ravel()
        size = [int(v) for v in d["size"]]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"calibration : [{key}] contient des valeurs non numeriques ({exc})") from exc
    if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError(f"calibration : [{key}].matrix doit etre une matrice 3x3 finie avec fx, fy > 0")
    if rvec.shape != (3,) or tvec.shape != (3,) or not (np.isfinite(rvec).all() and np.isfinite(tvec).all()):
        raise ValueError(f"calibration : [{key}].rotation et .translation doivent etre 3 nombres finis")
    if dist.size not in (4, 5, 8, 12, 14) or not np.isfinite(dist).all():
        raise ValueError(f"calibration : [{key}].distortions : {dist.size} coefficients (4, 5, 8, 12 ou 14 attendus)")
    if len(size) != 2 or min(size) <= 0:
        raise ValueError(f"calibration : [{key}].size doit etre [largeur, hauteur] > 0")


def load_calibration(path: str | Path):
    """Charge un calibration.toml Anipose ; ordre des cameras = cam_0, cam_1, ..., cam_10 (ordre naturel)."""
    import toml
    CameraGroup, _ = _require_aniposelib()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Calibration introuvable : {path}")
    try:
        data = toml.load(path)
    except toml.TomlDecodeError as exc:
        raise ValueError(f"calibration : TOML invalide ({exc})") from exc
    metadata = data.pop("metadata", {}) or {}
    keys = sorted(data.keys(), key=_natural_key)
    if len(keys) < 2:
        raise ValueError(f"calibration : {len(keys)} camera(s), au moins 2 sont necessaires pour trianguler")
    for k in keys:
        _check_camera_dict(k, data[k])
    names = [str(data[k]["name"]) for k in keys]
    if len(set(names)) != len(names):
        raise ValueError(f"calibration : noms de cameras en double {names}")
    cgroup = CameraGroup.from_dicts([data[k] for k in keys])
    cgroup.metadata = metadata
    return cgroup


def calibration_unit_scale(cgroup) -> float | None:
    """Facteur unites de calibration -> metres, d'apres metadata.units ; None si inconnu."""
    units = (getattr(cgroup, "metadata", None) or {}).get("units")
    return _UNITS_TO_M.get(str(units)) if units is not None else None


@dataclass
class BoardSpec:
    """Planche de calibration ; squares_x / squares_y = nombre de CASES (pas de coins)."""
    squares_x: int
    squares_y: int
    square_length_m: float
    marker_length_m: float | None = None
    marker_bits: int = 4
    dict_size: int = 50
    kind: str = "charuco"

    def validate(self) -> None:
        if self.kind not in ("charuco", "checkerboard"):
            raise ValueError(f"planche : type '{self.kind}' inconnu (charuco ou checkerboard)")
        if self.squares_x < 3 or self.squares_y < 3:
            raise ValueError("planche : au moins 3 x 3 cases")
        if not (self.square_length_m > 0 and math.isfinite(self.square_length_m)):
            raise ValueError("planche : square_length > 0 requis")
        if self.kind == "checkerboard":
            if self.marker_length_m is not None:
                raise ValueError("damier : pas de marqueurs, ne pas donner marker_length")
            if self.squares_x % 2 == self.squares_y % 2:
                raise ValueError("damier : un nombre de cases pair et l'autre impair sont requis, "
                                 "sinon l'orientation du damier est ambigue a 180 deg")
            return
        if self.marker_length_m is None or not (0 < self.marker_length_m < self.square_length_m):
            raise ValueError("planche ChArUco : 0 < marker_length < square_length requis")
        if (self.marker_bits, self.dict_size) not in {(b, s) for b in (4, 5, 6, 7) for s in (50, 100, 250, 1000)}:
            raise ValueError("planche ChArUco : marker_bits dans 4..7 et dict_size dans 50/100/250/1000")
        if self.squares_x * self.squares_y // 2 > self.dict_size:
            raise ValueError(f"planche ChArUco : {self.squares_x * self.squares_y // 2} marqueurs "
                             f"pour un dictionnaire de {self.dict_size}")

    def make(self):
        _require_aniposelib()
        self.validate()
        if self.kind == "checkerboard":
            from aniposelib.boards import Checkerboard
            return Checkerboard(self.squares_x - 1, self.squares_y - 1, self.square_length_m)
        from aniposelib.boards import CharucoBoard
        return CharucoBoard(self.squares_x, self.squares_y, self.square_length_m, self.marker_length_m,
                            marker_bits=self.marker_bits, dict_size=self.dict_size)


def calibrate_cameras(videos: Sequence[str | Path], output: str | Path, board: BoardSpec,
                      names: Sequence[str] | None = None, min_detections: int = 5) -> dict[str, Any]:
    """Calibre intrinseques + extrinseques a partir de videos ChArUco synchronisees."""
    CameraGroup, _ = _require_aniposelib()
    import cv2
    videos = [Path(v) for v in videos]
    if len(videos) < 2:
        raise ValueError("calibration : au moins 2 videos (une par camera)")
    names = list(names) if names else [f"cam{i}" for i in range(len(videos))]
    if len(names) != len(videos) or len(set(names)) != len(names):
        raise ValueError("calibration : un nom unique par video est requis")
    for v in videos:
        if not v.is_file():
            raise FileNotFoundError(f"Video introuvable : {v}")
        cap = cv2.VideoCapture(str(v))
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        if not ok:
            raise ValueError(f"Video illisible : {v}")
    ch_board = board.make()
    cgroup = CameraGroup.from_names(names)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        rows = cgroup.get_rows_videos([[str(v)] for v in videos], ch_board, verbose=False)
        counts = {n: len(r) for n, r in zip(names, rows)}
        weak = {n: c for n, c in counts.items() if c < min_detections}
        if weak:
            raise ValueError(f"calibration : planche detectee sur trop peu de frames {weak} "
                             f"(minimum {min_detections}). Verifie les dimensions de la planche et l'eclairage.")
        cgroup.set_camera_sizes_videos([[str(v)] for v in videos])
        try:
            error = float(cgroup.calibrate_rows(rows, ch_board, verbose=False))
        except ValueError as exc:
            raise ValueError(f"calibration impossible : {exc}") from exc
    if not math.isfinite(error):
        raise ValueError("calibration : erreur de reprojection non finie, calibration rejetee")
    cgroup.metadata = {"units": "m", "board": {k: v for k, v in board.__dict__.items() if v is not None}}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cgroup.dump(str(output))
    return {
        "output": str(output),
        "cameras": names,
        "detections": counts,
        "reprojection_error_px": round(error, 4),
        "units": "m",
    }


# --------------------------------------------------------------------------
# 2. Extraction 2D par camera
# --------------------------------------------------------------------------
@dataclass
class ViewTrack:
    name: str
    points: np.ndarray            # (T, 75, 2) pixels, NaN si absent
    scores: np.ndarray            # (T, 75) confiance 0..1, 0 si absent
    fps: float
    size: tuple[int, int]         # (largeur, hauteur)
    stats: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=float)
        self.scores = np.asarray(self.scores, dtype=float)
        if self.points.ndim != 3 or self.points.shape[1:] != (N_KEYPOINTS, 2):
            raise ValueError(f"vue {self.name} : points de forme {self.points.shape}, (T, {N_KEYPOINTS}, 2) attendu")
        if self.scores.shape != self.points.shape[:2]:
            raise ValueError(f"vue {self.name} : scores de forme {self.scores.shape}, {self.points.shape[:2]} attendu")
        if not (self.fps > 0 and math.isfinite(self.fps)):
            raise ValueError(f"vue {self.name} : fps invalide ({self.fps})")


def _assign_hands(hands: list[tuple[np.ndarray, float]], wrists: dict[str, np.ndarray],
                  max_dist: float) -> dict[str, tuple[np.ndarray, float]]:
    """Attribue chaque main au poignet Pose le plus proche (en pixels).

    Le label de lateralite de MediaPipe depend de la convention miroir de l'image ;
    la proximite au poignet anatomique de la Pose n'en depend pas.
    """
    sides = [s for s in ("L", "R") if s in wrists]
    best: dict[str, tuple[np.ndarray, float]] = {}
    best_cost = float("inf")
    for k in range(min(len(hands), len(sides)), 0, -1):
        for hs in itertools.permutations(range(len(hands)), k):
            for ss in itertools.permutations(sides, k):
                d = [float(np.linalg.norm(hands[h][0][0] - wrists[s])) for h, s in zip(hs, ss)]
                if max(d) > max_dist:
                    continue
                cost = sum(d)
                if cost < best_cost:
                    best_cost, best = cost, {s: hands[h] for h, s in zip(hs, ss)}
        if best:
            break
    return best


def extract_view_2d(video_path: str | Path, name: str | None = None,
                    max_frames: int | None = None) -> ViewTrack:
    import cv2
    from backend.mediapipe_compat import FrameClock, model_path, numpy_rgb_to_mp_image
    from backend.pose_estimator import (_BaseOptions, _HandLandmarker, _HandLandmarkerOptions,
                                        _PoseLandmarker, _PoseLandmarkerOptions, _RunningMode)
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video introuvable : {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Video illisible : {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or w <= 0 or h <= 0:
        cap.release()
        raise ValueError(f"Video {video_path} : fps ou taille inconnus (fps={fps}, {w}x{h})")
    pose_options = _PoseLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=model_path("pose_landmarker_full.task")),
        running_mode=_RunningMode.VIDEO,
        min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5, min_tracking_confidence=0.8)
    hand_options = _HandLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=model_path("hand_landmarker.task")),
        running_mode=_RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.6, min_hand_presence_confidence=0.5, min_tracking_confidence=0.8)
    pts_list, sc_list = [], []
    stats = {"frames": 0, "frames_with_body": 0, "hands_assigned": 0, "hands_dropped": 0,
             "points_outside_image": 0}
    max_dist = 0.15 * math.hypot(w, h)
    try:
        with _PoseLandmarker.create_from_options(pose_options) as pose, \
                _HandLandmarker.create_from_options(hand_options) as hands_model:
            clock = FrameClock(cap)
            index = 0
            while max_frames is None or index < max_frames:
                ok, image = cap.read()
                if not ok:
                    break
                index += 1
                if image.shape[1] != w or image.shape[0] != h:
                    raise ValueError(f"Video {video_path} : taille de frame variable")
                mp_image = numpy_rgb_to_mp_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                ts = clock(index)
                pose_res = pose.detect_for_video(mp_image, ts)
                hand_res = hands_model.detect_for_video(mp_image, ts)
                pts = np.full((N_KEYPOINTS, 2), np.nan)
                sc = np.zeros(N_KEYPOINTS)
                wrists: dict[str, np.ndarray] = {}
                if pose_res.pose_landmarks:
                    stats["frames_with_body"] += 1
                    for i, lm in enumerate(pose_res.pose_landmarks[0][:N_BODY]):
                        pts[i] = (lm.x * w, lm.y * h)
                        vis = lm.visibility if lm.visibility is not None else 0.0
                        pres = lm.presence if lm.presence is not None else vis
                        sc[i] = float(np.clip(min(vis, pres), 0.0, 1.0))
                    wrists = {"L": pts[_LEFT_WRIST].copy(), "R": pts[_RIGHT_WRIST].copy()}
                detected = []
                for k, hl in enumerate(hand_res.hand_landmarks or []):
                    score = 0.0
                    if hand_res.handedness and k < len(hand_res.handedness) and hand_res.handedness[k]:
                        score = float(hand_res.handedness[k][0].score)
                    detected.append((np.array([(lm.x * w, lm.y * h) for lm in hl[:N_HAND]]), score))
                assigned = _assign_hands(detected, wrists, max_dist)
                stats["hands_assigned"] += len(assigned)
                stats["hands_dropped"] += len(detected) - len(assigned)
                for side, (hp, hs) in assigned.items():
                    sl = LEFT_HAND if side == "L" else RIGHT_HAND
                    pts[sl] = hp
                    sc[sl] = hs
                outside = ~((pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h))
                stats["points_outside_image"] += int((outside & (sc > 0)).sum())
                sc[outside] = 0.0                  # extrapole par MediaPipe, pas observe
                pts_list.append(pts)
                sc_list.append(sc)
    finally:
        cap.release()
    stats["frames"] = len(pts_list)
    if not pts_list:
        raise ValueError(f"Video {video_path} : aucune frame lue")
    return ViewTrack(name or video_path.stem, np.stack(pts_list), np.stack(sc_list), fps, (w, h), stats)


# --------------------------------------------------------------------------
# 3. Synchronisation
# --------------------------------------------------------------------------
def synchronize(tracks: Sequence[ViewTrack], offsets: Sequence[int] | None = None,
                fps_tolerance: float = 0.01) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """offsets[i] = nombre de frames a ignorer au debut de la camera i (>= 0).

    -> points (C, T, 75, 2), scores (C, T, 75), infos.
    """
    if len(tracks) < 2:
        raise ValueError(f"synchronisation : {len(tracks)} vue(s), au moins 2 necessaires")
    offsets = [0] * len(tracks) if offsets is None else list(offsets)
    if len(offsets) != len(tracks):
        raise ValueError(f"synchronisation : {len(offsets)} decalage(s) pour {len(tracks)} cameras")
    if any((not isinstance(o, (int, np.integer))) or o < 0 for o in offsets):
        raise ValueError(f"synchronisation : decalages entiers >= 0 attendus, recu {offsets}")
    fps0 = tracks[0].fps
    for tr in tracks[1:]:
        if abs(tr.fps - fps0) > fps_tolerance * fps0:
            raise ValueError(f"synchronisation : fps differents ({tracks[0].name}={fps0:.3f}, "
                             f"{tr.name}={tr.fps:.3f}) ; reechantillonne les videos d'abord")
    lengths = [len(tr.points) - o for tr, o in zip(tracks, offsets)]
    T = min(lengths)
    if T <= 0:
        raise ValueError(f"synchronisation : aucun recouvrement (frames restantes par camera {lengths})")
    points = np.stack([tr.points[o:o + T] for tr, o in zip(tracks, offsets)])
    scores = np.stack([tr.scores[o:o + T] for tr, o in zip(tracks, offsets)])
    info = {"frames": T, "fps": fps0, "offsets": offsets,
            "frames_dropped_at_end": {tr.name: n - T for tr, n in zip(tracks, lengths)}}
    return points, scores, info


# --------------------------------------------------------------------------
# 4. Triangulation
# --------------------------------------------------------------------------
def _weighted_dlt(proj: np.ndarray, norm_pts: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """proj (C,3,4) ; norm_pts (C,N,2) coordonnees normalisees ; weights (C,N), 0 = vue ignoree."""
    x = np.nan_to_num(norm_pts[..., 0])[..., None]
    y = np.nan_to_num(norm_pts[..., 1])[..., None]
    w = weights[..., None]
    P0, P1, P2 = proj[:, None, 0, :], proj[:, None, 1, :], proj[:, None, 2, :]
    rows_x = w * (x * P2 - P0)
    rows_y = w * (y * P2 - P1)
    A = np.concatenate([rows_x, rows_y], axis=0).transpose(1, 0, 2)     # (N, 2C, 4)
    _, _, vh = np.linalg.svd(A)
    X = vh[:, -1, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = X[:, :3] / X[:, 3:4]
    out[np.abs(X[:, 3]) < 1e-12] = np.nan
    return out


def triangulate_weighted(cgroup, points: np.ndarray, scores: np.ndarray, min_score: float = 0.5,
                         max_reproj_px: float = 10.0, min_parallax_deg: float = 1.0) -> dict[str, np.ndarray]:
    """points (C, N, 2) en pixels bruts ; scores (C, N).

    Une vue compte si son point est fini et son score >= min_score. Un point 3D est rendu si
    >= 2 vues comptent, si les rayons des cameras forment un angle >= min_parallax_deg (sinon la
    profondeur n'est pas observable : meme video deux fois, cameras trop proches), s'il est devant
    chaque camera utilisee, et si son erreur de reprojection moyenne <= max_reproj_px. Avec >= 3 vues,
    la pire vue est retiree tant que l'erreur depasse le seuil. Sinon le point vaut NaN et la raison
    est comptee.
    """
    cams = cgroup.cameras
    points = np.asarray(points, dtype=float)
    scores = np.asarray(scores, dtype=float)
    C = len(cams)
    if points.ndim != 3 or points.shape[0] != C or points.shape[2] != 2:
        raise ValueError(f"triangulation : points de forme {points.shape}, ({C}, N, 2) attendu")
    if scores.shape != points.shape[:2]:
        raise ValueError(f"triangulation : scores de forme {scores.shape}, {points.shape[:2]} attendu")
    if not (0.0 <= min_score <= 1.0) or not (max_reproj_px > 0) or not (0.0 <= min_parallax_deg < 90.0):
        raise ValueError("triangulation : 0 <= min_score <= 1, max_reproj_px > 0 et 0 <= min_parallax_deg < 90 requis")
    N = points.shape[1]
    valid = np.isfinite(points).all(axis=2) & np.isfinite(scores) & (scores >= min_score)
    norm = np.full_like(points, np.nan)
    for c, cam in enumerate(cams):
        m = valid[c]
        if m.any():
            norm[c, m] = cam.undistort_points(np.ascontiguousarray(points[c, m])).reshape(-1, 2)
    proj = np.stack([cam.get_extrinsics_mat()[:3] for cam in cams])
    Rs = proj[:, :, :3]
    ts = proj[:, :, 3]
    centers = -np.einsum("cji,cj->ci", Rs, ts)                 # C_c = -R_c^T t_c
    cos_min = math.cos(math.radians(min_parallax_deg))
    use = valid.copy()
    p3d = np.full((N, 3), np.nan)
    err = np.full(N, np.nan)
    per_view = np.full((C, N), np.nan)
    behind = np.zeros(N, dtype=bool)
    flat = np.zeros(N, dtype=bool)
    todo = use.sum(axis=0) >= 2
    for _ in range(max(1, C - 1)):
        idx = np.where(todo)[0]
        if idx.size == 0:
            break
        X = _weighted_dlt(proj, norm[:, idx], np.where(use[:, idx], scores[:, idx], 0.0))
        finite = np.isfinite(X).all(axis=1)
        rays = X[None, :, :] - centers[:, None, :]            # (C, n, 3)
        with np.errstate(invalid="ignore", divide="ignore"):
            rays /= np.linalg.norm(rays, axis=2, keepdims=True)
        best_cos = np.ones(idx.size)
        for a, b in itertools.combinations(range(C), 2):
            pair = use[a, idx] & use[b, idx] & finite
            cosab = np.einsum("nj,nj->n", rays[a], rays[b])
            best_cos = np.where(pair, np.minimum(best_cos, cosab), best_cos)
        ok_parallax = finite & (best_cos <= cos_min)
        depth = (X @ Rs[:, 2, :].T + ts[:, 2]).T               # (C, n) : z dans chaque camera
        ok_depth = ok_parallax & np.where(use[:, idx], depth > 0, True).all(axis=0)
        e = np.full((C, idx.size), np.nan)
        for c, cam in enumerate(cams):
            m = use[c, idx] & ok_depth
            if m.any():
                pr = cam.project(np.ascontiguousarray(X[m])).reshape(-1, 2)
                e[c, m] = np.linalg.norm(pr - points[c, idx[m]], axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_e = np.nanmean(np.where(use[:, idx], e, np.nan), axis=0)
        good = ok_depth & (mean_e <= max_reproj_px)
        p3d[idx[good]] = X[good]
        err[idx[good]] = mean_e[good]
        per_view[:, idx[good]] = e[:, good]
        flat[idx[~ok_parallax]] = True
        behind[idx[ok_parallax & ~ok_depth]] = True
        retry = (~good) & ok_depth & (use[:, idx].sum(axis=0) >= 3)
        todo[:] = False
        if retry.any():
            r_idx = idx[retry]
            worst = np.nanargmax(np.where(use[:, r_idx], e[:, retry], -np.inf), axis=0)
            use[worst, r_idx] = False
            todo[r_idx] = True
    n_valid = valid.sum(axis=0)
    triangulated = np.isfinite(p3d).all(axis=1)
    return {
        "points3d": p3d,
        "error_px": err,
        "error_px_per_view": per_view,
        "views_used": np.where(triangulated, use.sum(axis=0), 0),
        "counts": {
            "points": int(N),
            "triangulated": int(triangulated.sum()),
            "fewer_than_2_views": int((n_valid < 2).sum()),
            "low_parallax": int((flat & ~triangulated).sum()),
            "behind_camera": int((behind & ~flat & ~triangulated).sum()),
            "rejected_reprojection": int(((n_valid >= 2) & ~behind & ~flat & ~triangulated).sum()),
        },
    }


def to_camera_frame(cgroup, p3d: np.ndarray, ref: int = 0) -> np.ndarray:
    """X_ref = R_ref X + t_ref : repere de la camera de reference (y bas, +z vers la scene)."""
    M = cgroup.cameras[ref].get_extrinsics_mat()
    return p3d @ M[:3, :3].T + M[:3, 3]


# --------------------------------------------------------------------------
# 5. Reconstruction complete
# --------------------------------------------------------------------------
def _landmark_dicts(p: np.ndarray, conf: np.ndarray) -> list[dict[str, float]]:
    return [{"x": float(a[0]), "y": float(a[1]), "z": float(a[2]), "visibility": float(v)}
            for a, v in zip(p, conf)]


def points_to_frames(p3d: np.ndarray, conf: np.ndarray) -> list[dict[str, Any]]:
    """(T, 75, 3) metres -> frames au format Mimo (bodyPose 33, handsL / handsR 21 ou [])."""
    frames = []
    for t in range(p3d.shape[0]):
        f: dict[str, Any] = {"frame": t + 1}
        body = p3d[t, :N_BODY]
        f["bodyPose"] = _landmark_dicts(body, conf[t, :N_BODY]) if np.isfinite(body).any() else []
        for key, sl in (("handsL", LEFT_HAND), ("handsR", RIGHT_HAND)):
            hand = p3d[t, sl]
            f[key] = _landmark_dicts(hand, conf[t, sl]) if np.isfinite(hand).any() else []
        frames.append(f)
    return frames


def reconstruct_frames(tracks: Sequence[ViewTrack], cgroup, *, offsets: Sequence[int] | None = None,
                       unit_scale: float | None = None, min_score: float = 0.5,
                       max_reproj_px: float = 10.0) -> tuple[list[dict[str, Any]], float, dict[str, Any]]:
    """Vues 2D + calibration -> (frames 3D en metres repere camera 0, fps, rapport)."""
    if len(tracks) != len(cgroup.cameras):
        raise ValueError(f"{len(tracks)} video(s) pour {len(cgroup.cameras)} camera(s) dans la calibration")
    for tr, cam in zip(tracks, cgroup.cameras):
        size = tuple(int(v) for v in (cam.get_size() or ()))
        if size != tuple(tr.size):
            raise ValueError(f"camera {cam.get_name()} calibree en {size}, video {tr.name} en {tuple(tr.size)} : "
                             "la calibration doit etre faite a la meme resolution")
    if unit_scale is None:
        unit_scale = calibration_unit_scale(cgroup)
        if unit_scale is None:
            raise ValueError("Unites de la calibration inconnues (metadata.units absent) : "
                             "precise unit_scale (facteur vers les metres, ex. 0.001 pour des mm)")
    if not (unit_scale > 0 and math.isfinite(unit_scale)):
        raise ValueError(f"unit_scale invalide : {unit_scale}")
    points, scores, sync = synchronize(tracks, offsets)
    C, T, K, _ = points.shape
    tri = triangulate_weighted(cgroup, points.reshape(C, T * K, 2), scores.reshape(C, T * K),
                               min_score=min_score, max_reproj_px=max_reproj_px)
    if tri["counts"]["triangulated"] == 0:
        raise ValueError(f"Aucun point triangule : {tri['counts']}. Verifie la calibration, "
                         "l'ordre des videos et la synchronisation.")
    p3d = to_camera_frame(cgroup, tri["points3d"]).reshape(T, K, 3) * unit_scale
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        conf = np.where(np.isfinite(p3d).all(axis=2),
                        np.nanmean(np.where(scores >= min_score, scores, np.nan), axis=0), 0.0)
    frames = points_to_frames(p3d, conf)

    ok = np.isfinite(p3d).all(axis=2)
    err = tri["error_px"].reshape(T, K)
    per_view = tri["error_px_per_view"].reshape(C, T, K)
    warns: list[str] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        shoulder = np.linalg.norm(p3d[:, 11] - p3d[:, 12], axis=1)
        sw = float(np.nanmedian(shoulder)) if np.isfinite(shoulder).any() else float("nan")
    if math.isfinite(sw) and not (0.2 <= sw <= 0.6):
        warns.append(f"Largeur d'epaules mediane = {sw:.3f} m, hors de [0.20, 0.60] m : "
                     "unites de calibration ou synchronisation suspectes.")

    def _stats(a: np.ndarray) -> dict[str, float | None]:
        a = a[np.isfinite(a)]
        if a.size == 0:
            return {"mean": None, "median": None, "p95": None}
        return {"mean": round(float(a.mean()), 3), "median": round(float(np.median(a)), 3),
                "p95": round(float(np.percentile(a, 95)), 3)}

    report = {
        "cameras": cgroup.get_names(),
        "videos": [tr.name for tr in tracks],
        "sync": sync,
        "unit_scale_to_m": unit_scale,
        "min_score": min_score,
        "max_reprojection_px": max_reproj_px,
        "frame": "camera 0 : x droite image, y bas, z vers la scene (convention MediaPipe world), metres",
        "coverage": {
            "body": round(float(ok[:, :N_BODY].mean()), 4),
            "left_hand": round(float(ok[:, LEFT_HAND].mean()), 4),
            "right_hand": round(float(ok[:, RIGHT_HAND].mean()), 4),
            "per_body_point": [round(float(v), 3) for v in ok[:, :N_BODY].mean(axis=0)],
        },
        "reprojection_error_px": {
            "all": _stats(err),
            "per_camera": {n: _stats(per_view[c]) for c, n in enumerate(cgroup.get_names())},
        },
        "triangulation_counts": tri["counts"],
        "shoulder_width_m_median": None if not math.isfinite(sw) else round(sw, 4),
        "extraction": {tr.name: tr.stats for tr in tracks},
        "warnings": warns,
    }
    return frames, sync["fps"], report


def export_multiview(frames: list[dict[str, Any]], fps: float, output: str | Path, report: dict[str, Any],
                     *, bvh_mode: str = "rotations", body: str = "auto", hands: str = "auto",
                     scale: float = 100.0) -> dict[str, Any]:
    """Ecrit .bvh, .vrma (binaire) ou .json (clip VRMA JSON) + <sortie>.multiview.json."""
    output = Path(output)
    suffix = output.suffix.lower()
    if suffix not in (".bvh", ".vrma", ".json"):
        raise ValueError(f"Sortie {output} : extension .bvh, .vrma ou .json attendue")
    output.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".bvh":
        from backend.bvh_export import export_mediapipe_bvh
        bvh_report = export_mediapipe_bvh(
            frames, output, fps, mode=bvh_mode, source="world", aspect=1.0, scale=scale,
            min_visibility=0.0, recenter=True, body=body, hands=hands,
            hand_units="metric", root_motion=True, extra={"multiview": True})
        report["bvh"] = {k: bvh_report[k] for k in ("body", "hands_used", "frames", "warnings")}
    else:
        from backend.vrm_retargeting import process_mimo_landmarks_to_vrma
        clip = process_mimo_landmarks_to_vrma(frames, fps=fps, sign_label=output.stem)
        if not clip["tracks"]:
            raise ValueError("Aucun os VRM anime : bras et mains jamais triangules")
        if suffix == ".vrma":
            from backend.vrma_serializer import serialize_to_vrma
            serialize_to_vrma(clip, output)
        else:
            output.write_text(json.dumps(clip, indent=2, ensure_ascii=False), encoding="utf-8")
        report["vrma"] = {"bones": clip["meta"]["bonesAnimated"], "duration": clip["duration"]}
    report["output"] = str(output)
    report_path = output.with_name(output.stem + ".multiview.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["report"] = str(report_path)
    return report


def run_multiview(videos: Sequence[str | Path], calibration: str | Path, output: str | Path, *,
                  offsets: Sequence[int] | None = None, unit_scale: float | None = None,
                  min_score: float = 0.5, max_reproj_px: float = 10.0, max_frames: int | None = None,
                  bvh_mode: str = "rotations", body: str = "auto", hands: str = "auto") -> dict[str, Any]:
    cgroup = load_calibration(calibration)
    if len(videos) != len(cgroup.cameras):
        raise ValueError(f"{len(videos)} video(s) pour {len(cgroup.cameras)} camera(s) dans {calibration}")
    if Path(output).suffix.lower() not in (".bvh", ".vrma", ".json"):
        raise ValueError(f"Sortie {output} : extension .bvh, .vrma ou .json attendue")
    tracks = [extract_view_2d(v, name=cam.get_name(), max_frames=max_frames)
              for v, cam in zip(videos, cgroup.cameras)]
    frames, fps, report = reconstruct_frames(tracks, cgroup, offsets=offsets, unit_scale=unit_scale,
                                             min_score=min_score, max_reproj_px=max_reproj_px)
    report["calibration"] = str(calibration)
    report["video_paths"] = [str(v) for v in videos]
    return export_multiview(frames, fps, output, report, bvh_mode=bvh_mode, body=body, hands=hands)
