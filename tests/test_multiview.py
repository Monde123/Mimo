"""Tests multi-vues.

Oracles :
  * squelette 3D REEL : 75 points metriques extraits par MediaPipe (pose world + mains world)
    de a.mp4 (fichier fourni par l'utilisateur), projetes par cv2.projectPoints dans des cameras
    aux parametres connus (avec distorsion) -> la triangulation doit retrouver ces points ;
  * aniposelib CameraGroup.triangulate : autre implementation de la DLT (jax) ;
  * planche ChArUco rendue par homographie exacte H = K [r1 r2 t] -> la calibration doit
    retrouver K et la pose relative connues.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
toml = pytest.importorskip("toml")
pytest.importorskip("aniposelib")

from backend import multiview as mv  # noqa: E402
from mimo.main import main  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "a.mp4"
W, H = 1280, 720


# --------------------------------------------------------------------------
# Donnees : squelette reel + cameras connues
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_skeleton() -> np.ndarray:
    """(T, 75, 3) metres, repere MediaPipe world, issu de a.mp4."""
    if not VIDEO.is_file():
        pytest.skip("a.mp4 absent : pas de squelette reel disponible")
    from backend.mediapipe_compat import FrameClock, model_path, numpy_rgb_to_mp_image
    from backend.pose_estimator import (_BaseOptions, _HandLandmarker, _HandLandmarkerOptions,
                                        _PoseLandmarker, _PoseLandmarkerOptions, _RunningMode)
    po = _PoseLandmarkerOptions(base_options=_BaseOptions(model_asset_path=model_path("pose_landmarker_full.task")),
                                running_mode=_RunningMode.VIDEO)
    ho = _HandLandmarkerOptions(base_options=_BaseOptions(model_asset_path=model_path("hand_landmarker.task")),
                                running_mode=_RunningMode.VIDEO, num_hands=2)
    cap = cv2.VideoCapture(str(VIDEO))
    out = []
    try:
        with _PoseLandmarker.create_from_options(po) as pose, _HandLandmarker.create_from_options(ho) as hands:
            clock = FrameClock(cap)
            i = 0
            while len(out) < 12 and i < 200:
                ok, img = cap.read()
                if not ok:
                    break
                i += 1
                mpi = numpy_rgb_to_mp_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                ts = clock(i)
                pr, hr = pose.detect_for_video(mpi, ts), hands.detect_for_video(mpi, ts)
                if not pr.pose_world_landmarks or len(hr.hand_world_landmarks or []) < 2:
                    continue
                body = np.array([[p.x, p.y, p.z] for p in pr.pose_world_landmarks[0]])
                hw = [np.array([[p.x, p.y, p.z] for p in h]) for h in hr.hand_world_landmarks[:2]]
                # attache chaque main au poignet Pose le plus proche (positions 2D normalisees)
                hn = [np.array([[p.x, p.y] for p in h]) for h in hr.hand_landmarks[:2]]
                pn = np.array([[p.x, p.y] for p in pr.pose_landmarks[0]])
                d0 = np.linalg.norm(hn[0][0] - pn[15]) + np.linalg.norm(hn[1][0] - pn[16])
                d1 = np.linalg.norm(hn[0][0] - pn[16]) + np.linalg.norm(hn[1][0] - pn[15])
                left, right = (hw[0], hw[1]) if d0 <= d1 else (hw[1], hw[0])
                sk = np.concatenate([body, left - left[0] + body[15], right - right[0] + body[16]])
                out.append(sk)
    finally:
        cap.release()
    if len(out) < 5:
        pytest.skip("moins de 5 frames avec corps + 2 mains dans a.mp4")
    return np.stack(out)


def _look_at(C, target):
    z = np.asarray(target, float) - C
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return cv2.Rodrigues(R)[0].ravel(), -R @ C


def _camera_dicts(n=2, dist=True):
    K = [[1000.0, 0.0, 650.0], [0.0, 1010.0, 350.0], [0.0, 0.0, 1.0]]
    centers = [np.array([0.0, 0.0, 0.0]), np.array([1.4, -0.2, 0.9]), np.array([-1.3, -0.4, 1.0])]
    dists = [[0.05, -0.01, 0.0, 0.0, 0.0], [-0.12, 0.03, 0.001, -0.001, 0.0], [0.08, 0.0, 0.0, 0.0, 0.0]]
    out = []
    for i in range(n):
        rvec, tvec = (np.zeros(3), np.zeros(3)) if i == 0 else _look_at(centers[i], [0.0, 0.0, 2.5])
        out.append({"name": f"cam{i}", "size": [W, H], "matrix": K,
                    "distortions": dists[i] if dist else [0.0] * 5,
                    "rotation": list(map(float, rvec)), "translation": list(map(float, tvec))})
    return out


def _write_calibration(path: Path, cams, units="m") -> Path:
    data = {f"cam_{i}": c for i, c in enumerate(cams)}
    if units is not None:
        data["metadata"] = {"units": units}
    path.write_text(toml.dumps(data), encoding="utf-8")
    return path


def _place(sk: np.ndarray) -> np.ndarray:
    """Squelette MediaPipe world (hanches en 0) -> repere camera 0, 2.5 m devant."""
    return sk + np.array([0.0, 0.1, 2.5])


def _project(cam: dict, pts: np.ndarray) -> np.ndarray:
    uv, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), np.array(cam["rotation"]), np.array(cam["translation"]),
                              np.array(cam["matrix"]), np.array(cam["distortions"]))
    return uv.reshape(*pts.shape[:-1], 2)


def _tracks(world: np.ndarray, cams, fps=30.0, scores=None):
    T = world.shape[0]
    tr = []
    for i, c in enumerate(cams):
        s = np.ones((T, mv.N_KEYPOINTS)) if scores is None else scores[i]
        tr.append(mv.ViewTrack(c["name"], _project(c, world), s, fps, (W, H)))
    return tr


# --------------------------------------------------------------------------
# Triangulation
# --------------------------------------------------------------------------
def test_triangulation_recovers_real_skeleton_with_distortion(tmp_path, real_skeleton):
    # Bug attrape : distorsion ignoree, mauvais repere de sortie, ordre des cameras, pixels mal convertis.
    cams = _camera_dicts(2)
    world = _place(real_skeleton)
    cg = mv.load_calibration(_write_calibration(tmp_path / "c.toml", cams))
    frames, fps, rep = mv.reconstruct_frames(_tracks(world, cams), cg)
    got = np.array([[[p["x"], p["y"], p["z"]] for p in f["bodyPose"] + f["handsL"] + f["handsR"]] for f in frames])
    err = np.linalg.norm(got - world, axis=2)
    assert err.max() < 1e-3, f"erreur 3D max {err.max() * 1000:.3f} mm"
    assert rep["coverage"]["body"] == 1.0 and rep["coverage"]["left_hand"] == 1.0
    assert rep["reprojection_error_px"]["all"]["p95"] < 0.05
    assert fps == 30.0

    # Sensibilite : la meme chaine sans correction de distorsion doit echouer nettement.
    for cam in cg.cameras:
        cam.zero_distortions()
    tri = mv.triangulate_weighted(cg, np.stack([_project(c, world).reshape(-1, 2) for c in cams]),
                                  np.ones((2, world.shape[0] * mv.N_KEYPOINTS)), max_reproj_px=1e6)
    wrong = np.linalg.norm(mv.to_camera_frame(cg, tri["points3d"]) - world.reshape(-1, 3), axis=1)
    assert np.nanmax(wrong) > 1e-3          # le seuil ci-dessus detecte bien l'oubli de la distorsion


def test_weighted_dlt_matches_aniposelib_with_equal_weights(tmp_path, real_skeleton):
    # Oracle : autre implementation (aniposelib, jax). Bug attrape : DLT mal construite.
    cams = _camera_dicts(3)
    world = _place(real_skeleton).reshape(-1, 3)
    cg = mv.load_calibration(_write_calibration(tmp_path / "c.toml", cams))
    p2d = np.stack([_project(c, world) for c in cams]) + np.random.default_rng(1).normal(0, 0.7, (3, len(world), 2))
    ours = mv.triangulate_weighted(cg, p2d, np.ones(p2d.shape[:2]), max_reproj_px=1e6)["points3d"]
    ref = cg.triangulate(p2d, undistort=True)
    assert np.abs(ours - ref).max() < 1e-6


def test_confidence_weighting_reduces_bad_view_influence(tmp_path, real_skeleton):
    # Bug attrape : scores ignores (DLT non ponderee).
    cams = _camera_dicts(3)
    world = _place(real_skeleton).reshape(-1, 3)
    cg = mv.load_calibration(_write_calibration(tmp_path / "c.toml", cams))
    p2d = np.stack([_project(c, world) for c in cams])
    p2d[2, :, 0] += 25.0
    scores = np.ones(p2d.shape[:2])
    scores[2] = 0.55
    weighted = mv.triangulate_weighted(cg, p2d, scores, max_reproj_px=1e6)["points3d"]
    unweighted = cg.triangulate(p2d, undistort=True)
    e_w = np.linalg.norm(weighted - world, axis=1).mean()
    e_u = np.linalg.norm(unweighted - world, axis=1).mean()
    assert e_w < 0.8 * e_u, (e_w, e_u)


def test_reprojection_rejection_drops_worst_view_or_point(tmp_path, real_skeleton):
    cams = _camera_dicts(3)
    world = _place(real_skeleton).reshape(-1, 3)
    cg = mv.load_calibration(_write_calibration(tmp_path / "c.toml", cams))
    p2d = np.stack([_project(c, world) for c in cams])
    p2d[2, :, 1] += 40.0
    tri = mv.triangulate_weighted(cg, p2d, np.ones(p2d.shape[:2]), max_reproj_px=5.0)
    assert np.linalg.norm(tri["points3d"] - world, axis=1).max() < 1e-3      # 3 vues -> pire vue retiree
    assert set(np.unique(tri["views_used"])) == {2}
    # 2 vues incoherentes (hors epipolaire) : aucun point ne doit sortir.
    cg2 = cg.subset_cameras([0, 2])
    tri2 = mv.triangulate_weighted(cg2, p2d[[0, 2]], np.ones((2, len(world))), max_reproj_px=5.0)
    assert tri2["counts"]["triangulated"] == 0
    assert tri2["counts"]["rejected_reprojection"] == len(world)


def test_point_behind_cameras_is_rejected(tmp_path):
    cams = _camera_dicts(2, dist=False)
    cg = mv.load_calibration(_write_calibration(tmp_path / "c.toml", cams))
    pts = np.array([[0.1, 0.0, 2.0], [0.0, 0.1, -1.5]])     # le 2e est derriere la camera 0
    p2d = np.stack([_project(c, pts) for c in cams])
    tri = mv.triangulate_weighted(cg, p2d, np.ones((2, 2)), max_reproj_px=1e6)
    assert np.isfinite(tri["points3d"][0]).all() and np.isnan(tri["points3d"][1]).all()
    assert tri["counts"]["behind_camera"] == 1


def test_same_video_twice_is_not_triangulated(tmp_path):
    # Entree hostile realiste : la meme video passee pour les deux cameras -> rayons paralleles,
    # point a l'infini dont l'erreur de reprojection est nulle. Il ne doit PAS etre accepte.
    cams = _camera_dicts(2, dist=False)
    cams[1]["rotation"] = [0.0, 0.0, 0.0]
    cams[1]["translation"] = [-0.5, 0.0, 0.0]
    cg = mv.load_calibration(_write_calibration(tmp_path / "c.toml", cams))
    pts = np.array([[0.1, 0.0, 2.0], [-0.2, 0.3, 3.0]])
    same = _project(cams[0], pts)
    tri = mv.triangulate_weighted(cg, np.stack([same, same]), np.ones((2, 2)), max_reproj_px=1e6)
    assert tri["counts"]["triangulated"] == 0 and tri["counts"]["low_parallax"] == 2
    # Controle : un vrai point lointain (parallaxe 0.3 deg) est refuse, un point a 2 m (14 deg) accepte.
    far_near = np.array([[0.0, 0.0, 100.0], [0.0, 0.0, 2.0]])
    p2d = np.stack([_project(c, far_near) for c in cams])
    tri = mv.triangulate_weighted(cg, p2d, np.ones((2, 2)), max_reproj_px=1e6)
    assert np.isnan(tri["points3d"][0]).all() and np.linalg.norm(tri["points3d"][1] - far_near[1]) < 1e-6


def test_low_score_and_nan_views_are_not_used(tmp_path):
    cams = _camera_dicts(2, dist=False)
    cg = mv.load_calibration(_write_calibration(tmp_path / "c.toml", cams))
    pts = np.array([[0.1, 0.0, 2.0]] * 3)
    p2d = np.stack([_project(c, pts) for c in cams])
    scores = np.ones((2, 3))
    scores[1, 0] = 0.49                 # sous le seuil
    p2d[0, 1] = np.nan                  # point absent
    scores[1, 2] = np.nan               # score illisible
    tri = mv.triangulate_weighted(cg, p2d, scores, min_score=0.5)
    assert np.isnan(tri["points3d"]).all()
    assert tri["counts"]["fewer_than_2_views"] == 3


@pytest.mark.parametrize("bad", [
    {"points": np.zeros((3, 4, 2)), "scores": np.ones((3, 4))},        # 3 vues pour 2 cameras
    {"points": np.zeros((2, 4, 3)), "scores": np.ones((2, 4))},        # 3 coordonnees
    {"points": np.zeros((2, 4, 2)), "scores": np.ones((2, 5))},        # scores en trop
    {"points": np.zeros((2, 0, 2)), "scores": np.ones((2, 0)), "min_score": 1.5},
])
def test_triangulation_rejects_bad_shapes(tmp_path, bad):
    cg = mv.load_calibration(_write_calibration(tmp_path / "c.toml", _camera_dicts(2)))
    with pytest.raises(ValueError, match="triangulation"):
        mv.triangulate_weighted(cg, **bad)


# --------------------------------------------------------------------------
# Synchronisation
# --------------------------------------------------------------------------
def test_offsets_realign_shifted_camera(tmp_path, real_skeleton):
    # La camera 1 demarre 3 frames plus tot ; le signeur se deplace de 4 cm par frame.
    cams = _camera_dicts(2)
    T = len(real_skeleton)
    move = lambda t: np.array([0.04 * t, 0.0, 0.0])
    full = np.stack([_place(real_skeleton[max(t, 0)]) + move(t) for t in range(-3, T)])
    world = full[3:]
    cg = mv.load_calibration(_write_calibration(tmp_path / "c.toml", cams))
    a = mv.ViewTrack("cam0", _project(cams[0], world), np.ones((T, mv.N_KEYPOINTS)), 30.0, (W, H))
    b = mv.ViewTrack("cam1", _project(cams[1], full), np.ones((T + 3, mv.N_KEYPOINTS)), 30.0, (W, H))

    def body(frames):
        return np.array([[[p["x"], p["y"], p["z"]] if f["bodyPose"] else [np.nan] * 3
                          for p in (f["bodyPose"] or [None] * 33)] for f in frames], dtype=float)

    frames, _, rep = mv.reconstruct_frames([a, b], cg, offsets=[0, 3])
    assert np.linalg.norm(body(frames) - world[:, :33], axis=2).max() < 1e-3
    assert rep["sync"]["frames"] == T and rep["sync"]["frames_dropped_at_end"] == {"cam0": 0, "cam1": 0}
    # Sans decalage : 12 cm d'ecart temporel -> aucun point ne doit etre a la fois accepte ET juste.
    frames0, _, _ = mv.reconstruct_frames([a, b], cg, offsets=None, max_reproj_px=1e6)
    err0 = np.linalg.norm(body(frames0) - world[:, :33], axis=2)
    assert np.nanmin(err0) > 0.01


@pytest.mark.parametrize("offsets,fps_b,msg", [
    ([0], 30.0, "decalage"),
    ([0, -1], 30.0, ">= 0"),
    ([0, 1.5], 30.0, ">= 0"),
    ([0, 0], 25.0, "fps"),
    ([0, 50], 30.0, "recouvrement"),
])
def test_synchronize_rejects_invalid_inputs(offsets, fps_b, msg):
    a = mv.ViewTrack("a", np.zeros((10, mv.N_KEYPOINTS, 2)), np.ones((10, mv.N_KEYPOINTS)), 30.0, (W, H))
    b = mv.ViewTrack("b", np.zeros((10, mv.N_KEYPOINTS, 2)), np.ones((10, mv.N_KEYPOINTS)), fps_b, (W, H))
    with pytest.raises(ValueError, match=msg):
        mv.synchronize([a, b], offsets)


def test_synchronize_needs_two_views_and_valid_tracks():
    a = mv.ViewTrack("a", np.zeros((10, mv.N_KEYPOINTS, 2)), np.ones((10, mv.N_KEYPOINTS)), 30.0, (W, H))
    with pytest.raises(ValueError, match="au moins 2"):
        mv.synchronize([a])
    with pytest.raises(ValueError, match="forme"):
        mv.ViewTrack("x", np.zeros((10, 33, 2)), np.ones((10, 33)), 30.0, (W, H))
    with pytest.raises(ValueError, match="fps"):
        mv.ViewTrack("x", np.zeros((1, mv.N_KEYPOINTS, 2)), np.ones((1, mv.N_KEYPOINTS)), 0.0, (W, H))


# --------------------------------------------------------------------------
# Calibration : fichier
# --------------------------------------------------------------------------
def test_load_calibration_natural_camera_order(tmp_path):
    cams = []
    for i in range(11):
        c = _camera_dicts(1)[0]
        c["name"] = f"n{i}"
        cams.append(c)
    cg = mv.load_calibration(_write_calibration(tmp_path / "c.toml", cams))
    assert cg.get_names() == [f"n{i}" for i in range(11)]       # cam_10 apres cam_9


@pytest.mark.parametrize("mutate,msg", [
    (lambda d: d.pop("cam_1"), "au moins 2"),
    (lambda d: d["cam_0"].pop("size"), "size"),
    (lambda d: d["cam_0"].__setitem__("matrix", [[-5, 0, 1], [0, 5, 1], [0, 0, 1]]), "fx, fy > 0"),
    (lambda d: d["cam_0"].__setitem__("matrix", [[1, 0], [0, 1]]), "3x3"),
    (lambda d: d["cam_0"].__setitem__("rotation", [0, 0]), "rotation"),
    (lambda d: d["cam_0"].__setitem__("distortions", [0.1, 0.2, 0.3]), "distortions"),
    (lambda d: d["cam_0"].__setitem__("translation", ["a", "b", "c"]), "non numeriques"),
    (lambda d: d["cam_1"].__setitem__("name", "cam0"), "double"),
])
def test_load_calibration_rejects_invalid(tmp_path, mutate, msg):
    data = {f"cam_{i}": c for i, c in enumerate(_camera_dicts(2))}
    mutate(data)
    p = tmp_path / "c.toml"
    p.write_text(toml.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match=msg):
        mv.load_calibration(p)


def test_load_calibration_missing_and_malformed(tmp_path):
    with pytest.raises(FileNotFoundError):
        mv.load_calibration(tmp_path / "absent.toml")
    bad = tmp_path / "bad.toml"
    bad.write_text("[cam_0\nname = ", encoding="utf-8")
    with pytest.raises(ValueError, match="TOML invalide"):
        mv.load_calibration(bad)


def test_reconstruct_refuses_mismatches_and_unknown_units(tmp_path, real_skeleton):
    cams = _camera_dicts(2)
    world = _place(real_skeleton)
    tracks = _tracks(world, cams)
    no_units = mv.load_calibration(_write_calibration(tmp_path / "n.toml", cams, units=None))
    with pytest.raises(ValueError, match="Unites"):
        mv.reconstruct_frames(tracks, no_units)
    # unites explicites en mm : la sortie doit etre 1000 fois plus petite
    frames, _, _ = mv.reconstruct_frames(tracks, no_units, unit_scale=0.001)
    assert abs(frames[0]["bodyPose"][0]["z"] - world[0, 0, 2] * 0.001) < 1e-6
    cg = mv.load_calibration(_write_calibration(tmp_path / "c.toml", cams))
    with pytest.raises(ValueError, match="camera"):
        mv.reconstruct_frames(tracks[:1] + tracks, cg)
    small = [mv.ViewTrack(t.name, t.points, t.scores, t.fps, (640, 360)) for t in tracks]
    with pytest.raises(ValueError, match="resolution"):
        mv.reconstruct_frames(small, cg)
    with pytest.raises(ValueError, match="unit_scale"):
        mv.reconstruct_frames(tracks, cg, unit_scale=-1.0)


# --------------------------------------------------------------------------
# Export : BVH avec mains metriques, VRMA
# --------------------------------------------------------------------------
def _parse_bvh_positions(path: Path):
    """Parseur BVH minimal independant (mode positions : 3 canaux Xposition Yposition Zposition)."""
    tokens = path.read_text(encoding="utf-8").split()
    names, parents, offsets, stack, i = [], [], [], [], 0
    while tokens[i] != "MOTION":
        tok = tokens[i]
        if tok in ("ROOT", "JOINT"):
            names.append(tokens[i + 1])
            parents.append(stack[-1] if stack else None)
            i += 2
        elif tok == "End":
            names.append(None)
            parents.append(stack[-1])
            i += 2
        elif tok == "{":
            stack.append(len(names) - 1)
            i += 1
        elif tok == "}":
            stack.pop()
            i += 1
        elif tok == "OFFSET":
            offsets.append([float(x) for x in tokens[i + 1:i + 4]])
            i += 4
        elif tok == "CHANNELS":
            i += 2 + int(tokens[i + 1])
        else:
            i += 1
    n_frames = int(tokens[i + 2])
    vals = np.array(tokens[i + 6:], dtype=float).reshape(n_frames, -1)
    joints = [k for k, n in enumerate(names) if n is not None]
    pos = {}
    for t in range(n_frames):
        world = {}
        for c, k in enumerate(joints):
            local = np.array(offsets[k]) + vals[t, 3 * c:3 * c + 3]
            world[k] = local if parents[k] is None else world[parents[k]] + local
        for k in joints:
            pos.setdefault(names[k], []).append(world[k])
    return {n: np.array(v) for n, v in pos.items()}


def test_bvh_export_keeps_metric_hand_size(tmp_path, real_skeleton):
    # Bug attrape : mains re-mises a l'echelle (taille de main supposee) alors qu'elles sont metriques.
    cams = _camera_dicts(2)
    world = _place(real_skeleton)
    cg = mv.load_calibration(_write_calibration(tmp_path / "c.toml", cams))
    frames, fps, rep = mv.reconstruct_frames(_tracks(world, cams), cg)
    out = tmp_path / "mv.bvh"
    mv.export_multiview(frames, fps, out, rep, bvh_mode="positions", body="upper", hands="on")
    pos = _parse_bvh_positions(out)
    got = np.linalg.norm(pos["LEFT_HAND_MIDDLE_FINGER_TIP"] - pos["LEFT_WRIST"], axis=1)
    truth = np.linalg.norm(world[:, 33 + 12] - world[:, 33 + 0], axis=1) * 100.0       # cm
    assert np.abs(got - truth).max() < 0.05
    shoulder = np.linalg.norm(pos["LEFT_SHOULDER"] - pos["RIGHT_SHOULDER"], axis=1)
    assert np.abs(shoulder - np.linalg.norm(world[:, 11] - world[:, 12], axis=1) * 100).max() < 0.05
    report = json.loads(out.with_name("mv.multiview.json").read_text(encoding="utf-8"))
    assert report["bvh"]["hands_used"] == ["L", "R"]
    assert not any("ESTIMEE" in w for w in report["bvh"]["warnings"])


def test_export_rejects_unknown_extension(tmp_path):
    with pytest.raises(ValueError, match="extension"):
        mv.export_multiview([], 30.0, tmp_path / "x.fbx", {})


def test_vrma_export_from_multiview(tmp_path, real_skeleton):
    pytest.importorskip("pygltflib")
    cams = _camera_dicts(2)
    cg = mv.load_calibration(_write_calibration(tmp_path / "c.toml", cams))
    frames, fps, rep = mv.reconstruct_frames(_tracks(_place(real_skeleton), cams), cg)
    out = tmp_path / "mv.vrma"
    rep = mv.export_multiview(frames, fps, out, rep)
    assert out.read_bytes()[:4] == b"glTF"
    assert {"leftUpperArm", "rightLowerArm", "leftIndexProximal", "rightThumbDistal"} <= set(rep["vrma"]["bones"])


# --------------------------------------------------------------------------
# Attribution des mains
# --------------------------------------------------------------------------
def test_assign_hands_uses_nearest_pose_wrist_not_label():
    wrists = {"L": np.array([800.0, 400.0]), "R": np.array([300.0, 400.0])}
    near_r = (np.array([[310.0, 395.0]] * 21), 0.9)
    near_l = (np.array([[790.0, 410.0]] * 21), 0.8)
    far = (np.array([[50.0, 50.0]] * 21), 0.99)
    got = mv._assign_hands([near_r, near_l], wrists, max_dist=200.0)
    assert got["L"][1] == 0.8 and got["R"][1] == 0.9
    assert mv._assign_hands([far], wrists, max_dist=200.0) == {}
    assert mv._assign_hands([near_l], {}, max_dist=200.0) == {}


def test_extract_view_2d_on_real_video_pixels_are_consistent():
    # Oracle : deux modeles independants (Pose et Hand) doivent placer le poignet au meme pixel.
    if not VIDEO.is_file():
        pytest.skip("a.mp4 absent")
    tr = mv.extract_view_2d(VIDEO, max_frames=40)
    w, h = tr.size
    assert tr.points.shape == (40, mv.N_KEYPOINTS, 2)
    gap = []
    for sl, wi in ((mv.LEFT_HAND, 15), (mv.RIGHT_HAND, 16)):
        ok = np.isfinite(tr.points[:, sl][:, 0, 0])
        gap.extend(np.linalg.norm(tr.points[ok, sl][:, 0] - tr.points[ok, wi], axis=1))
    assert len(gap) > 10
    assert np.median(gap) < 0.03 * math.hypot(w, h)
    assert ((tr.scores >= 0) & (tr.scores <= 1)).all()
    # a.mp4 coupe les jambes : MediaPipe extrapole hors image, ces points ne doivent pas compter.
    p = tr.points
    outside = np.isfinite(p[..., 0]) & ~((p[..., 0] >= 0) & (p[..., 0] < w) & (p[..., 1] >= 0) & (p[..., 1] < h))
    assert outside.any() and (tr.scores[outside] == 0).all()
    assert tr.stats["points_outside_image"] > 0


def test_extract_view_2d_rejects_missing_and_unreadable(tmp_path):
    with pytest.raises(FileNotFoundError):
        mv.extract_view_2d(tmp_path / "absent.mp4")
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"\x00" * 64)
    with pytest.raises(ValueError):
        mv.extract_view_2d(junk)


# --------------------------------------------------------------------------
# Calibration ChArUco rendue par homographie exacte
# --------------------------------------------------------------------------
def _render_board_videos(folder: Path, src: np.ndarray, metres_per_px: float, n_poses=24):
    """Rend l'image de planche src (niveaux de gris) par H = K [r1 r2 t] dans 2 cameras connues."""
    K = np.array([[900.0, 0, 639.5], [0, 900.0, 359.5], [0, 0, 1]])
    rvec1, t1 = _look_at(np.array([0.5, 0.0, 0.0]), [0.0, 0.0, 1.2])
    R1 = cv2.Rodrigues(rvec1)[0]
    cams = [(np.eye(3), np.zeros(3)), (R1, t1)]
    rng = np.random.default_rng(0)
    S = np.diag([metres_per_px, metres_per_px, 1.0])
    half = np.array([src.shape[1] / 2 * metres_per_px, src.shape[0] / 2 * metres_per_px, 0.0])
    poses = []
    for _ in range(n_poses):
        Rb = cv2.Rodrigues(rng.uniform(-0.5, 0.5, 3))[0]
        c = np.array([rng.uniform(-0.15, 0.15), rng.uniform(-0.1, 0.1), rng.uniform(0.9, 1.4)])
        poses.append((Rb, c - Rb @ half))
    paths = []
    for i, (Rc, tc) in enumerate(cams):
        p = folder / f"board{i}.avi"
        vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"MJPG"), 30, (W, H))
        for Rb, tb in poses:
            R, t = Rc @ Rb, Rc @ tb + tc
            img = cv2.warpPerspective(src, K @ np.column_stack([R[:, 0], R[:, 1], t]) @ S, (W, H),
                                      flags=cv2.INTER_AREA, borderValue=255)
            vw.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        vw.release()
        paths.append(p)
    return paths, K, R1, t1


@pytest.fixture(scope="module")
def charuco_videos(tmp_path_factory):
    board = cv2.aruco.CharucoBoard((7, 5), 0.06, 0.045, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    src = board.generateImage((700, 500), marginSize=0)
    return _render_board_videos(tmp_path_factory.mktemp("charuco"), src, 0.06 / 100)


@pytest.fixture(scope="module")
def checker_videos(tmp_path_factory):
    px, nx, ny = 60, 8, 7                                   # 8 x 7 cases de 5 cm, marge blanche d'une case
    src = np.full(((ny + 2) * px, (nx + 2) * px), 255, np.uint8)
    for y in range(ny):
        for x in range(nx):
            if (x + y) % 2 == 0:
                src[(y + 1) * px:(y + 2) * px, (x + 1) * px:(x + 2) * px] = 0
    return _render_board_videos(tmp_path_factory.mktemp("checker"), src, 0.05 / px)


def _check_recovered(toml_path: Path, R1: np.ndarray):
    cg = mv.load_calibration(toml_path)
    assert mv.calibration_unit_scale(cg) == 1.0
    for cam in cg.cameras:
        assert abs(cam.get_focal_length() - 900.0) / 900.0 < 0.01
        assert tuple(cam.get_size()) == (W, H)
    M = cg.cameras[1].get_extrinsics_mat() @ np.linalg.inv(cg.cameras[0].get_extrinsics_mat())
    angle = np.degrees(np.linalg.norm(cv2.Rodrigues(M[:3, :3] @ R1.T)[0]))
    assert angle < 0.5
    assert abs(np.linalg.norm(M[:3, 3]) - 0.5) < 0.005          # base de 50 cm, en metres


def test_checkerboard_calibration_recovers_known_cameras(tmp_path, checker_videos):
    paths, K, R1, t1 = checker_videos
    res = mv.calibrate_cameras(paths, tmp_path / "calibration.toml", mv.BoardSpec(8, 7, 0.05, kind="checkerboard"))
    assert res["reprojection_error_px"] < 0.5
    _check_recovered(tmp_path / "calibration.toml", R1)
    with pytest.raises(ValueError, match="ambigue"):
        mv.BoardSpec(8, 6, 0.05, kind="checkerboard").validate()
    with pytest.raises(ValueError, match="marqueurs"):
        mv.BoardSpec(8, 7, 0.05, 0.03, kind="checkerboard").validate()


def test_charuco_calibration_recovers_known_cameras(tmp_path, charuco_videos):
    paths, K, R1, t1 = charuco_videos
    spec = mv.BoardSpec(7, 5, 0.06, 0.045)
    res = mv.calibrate_cameras(paths, tmp_path / "calibration.toml", spec)
    assert res["detections"] == {"cam0": 24, "cam1": 24}
    assert res["reprojection_error_px"] < 0.5
    _check_recovered(tmp_path / "calibration.toml", R1)


def test_cli_calibrate_real_command(tmp_path, charuco_videos, capsys):
    paths = charuco_videos[0]
    out = tmp_path / "cli.toml"
    code = main(["calibrate", *map(str, paths), "-o", str(out), "--squares-x", "7", "--squares-y", "5",
                 "--square-length-m", "0.06", "--marker-length-m", "0.045"])
    assert code == 0
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["cameras"] == ["cam0", "cam1"] and out.is_file()


def test_calibration_rejects_wrong_board_and_bad_inputs(tmp_path, charuco_videos):
    paths = charuco_videos[0]
    with pytest.raises(ValueError, match="trop peu"):            # mauvais dictionnaire de marqueurs
        mv.calibrate_cameras(paths, tmp_path / "x.toml", mv.BoardSpec(7, 5, 0.06, 0.045, marker_bits=5))
    with pytest.raises(ValueError, match="au moins 2"):
        mv.calibrate_cameras(paths[:1], tmp_path / "x.toml", mv.BoardSpec(7, 5, 0.06, 0.045))
    with pytest.raises(FileNotFoundError):
        mv.calibrate_cameras([paths[0], tmp_path / "none.avi"], tmp_path / "x.toml", mv.BoardSpec(7, 5, 0.06, 0.045))
    with pytest.raises(ValueError, match="marker_length"):
        mv.BoardSpec(7, 5, 0.04, 0.05).validate()
    with pytest.raises(ValueError, match="marqueurs"):
        mv.BoardSpec(12, 12, 0.04, 0.03, dict_size=50).validate()
    with pytest.raises(ValueError, match="nom unique"):
        mv.calibrate_cameras(paths, tmp_path / "x.toml", mv.BoardSpec(7, 5, 0.06, 0.045), names=["a", "a"])


def test_cli_multiview_reports_clear_errors(tmp_path, capsys):
    calib = _write_calibration(tmp_path / "c.toml", _camera_dicts(2))
    code = main(["multiview", "a.mp4", "-o", str(tmp_path / "o.bvh"), "--calibration", str(calib)])
    assert code == 2
    assert "1 video(s) pour 2 camera(s)" in capsys.readouterr().err
    code = main(["multiview", "a.mp4", "b.mp4", "-o", str(tmp_path / "o.fbx"), "--calibration", str(calib)])
    assert code == 2
    assert "extension" in capsys.readouterr().err
    code = main(["multiview", "a.mp4", "b.mp4", "-o", str(tmp_path / "o.bvh"),
                 "--calibration", str(tmp_path / "none.toml")])
    assert code == 2
    assert "introuvable" in capsys.readouterr().err
