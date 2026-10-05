"""Retargeting BVH -> Mixamo.

Oracle : cinematique directe independante. Un BVH aux rotations aleatoires (poses hostiles,
+-70 deg par canal) est ecrit ici, puis rejoue sur clara.glb (modele reel fourni par
l'utilisateur) avec les rotations produites. La FK glTF (matrices T*R*S de chaque noeud,
ecrite ici) et la FK BVH (Euler intrinseque ZXY de scipy, ecrite ici) doivent donner la meme
direction monde pour chaque os.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pygltflib = pytest.importorskip("pygltflib")

from backend import retarget_bvh_to_mixamo as rt  # noqa: E402
from backend.bvh_export import _build_skeleton  # noqa: E402

CLARA = Path(__file__).resolve().parents[1] / "clara.glb"
M = "mixamorig:"


def _write_bvh(path: Path, T=6, seed=0, amplitude=70.0) -> Path:
    sk = _build_skeleton("full", ("L", "R"))
    rng = np.random.default_rng(seed)
    lines = ["HIERARCHY"]
    order = []

    def emit(name, depth):
        pad = "\t" * depth
        root = sk.parent[name] is None
        lines.append(f"{pad}{'ROOT' if root else 'JOINT'} {name}")
        lines.append(pad + "{")
        off = np.zeros(3) if root else sk.rest[name] / np.linalg.norm(sk.rest[name]) * 10.0
        lines.append(f"{pad}\tOFFSET {off[0]:.6f} {off[1]:.6f} {off[2]:.6f}")
        lines.append(f"{pad}\tCHANNELS " + ("6 Xposition Yposition Zposition " if root else "3 ")
                     + "Zrotation Xrotation Yrotation")
        order.append(name)
        for kid in sk.children[name]:
            emit(kid, depth + 1)
        if not sk.children[name]:
            lines.extend([f"{pad}\tEnd Site", f"{pad}\t{{", f"{pad}\t\tOFFSET 0 1 0", f"{pad}\t}}"])
        lines.append(pad + "}")

    emit(sk.root, 0)
    lines += ["MOTION", f"Frames: {T}", "Frame Time: 0.033333"]
    for _ in range(T):
        vals = [0.0, 90.0, 0.0] + list(rng.uniform(-amplitude, amplitude, 3 * len(order)))
        lines.append(" ".join(f"{v:.5f}" for v in vals))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _bvh_fk(path: Path) -> dict[str, np.ndarray]:
    tok = path.read_text(encoding="utf-8").split()
    names, parent, offset, chans, stack, i, pending = [], [], [], [], [], 0, None
    while tok[i] != "MOTION":
        t = tok[i]
        if t in ("ROOT", "JOINT"):
            pending = tok[i + 1]
            i += 2
        elif t == "End":
            pending = None
            i += 2
        elif t == "{":
            names.append(pending)
            parent.append(stack[-1] if stack else None)
            offset.append(None)
            chans.append([])
            stack.append(len(names) - 1)
            i += 1
        elif t == "}":
            stack.pop()
            i += 1
        elif t == "OFFSET":
            offset[stack[-1]] = np.array(tok[i + 1:i + 4], float)
            i += 4
        elif t == "CHANNELS":
            n = int(tok[i + 1])
            chans[stack[-1]] = tok[i + 2:i + 2 + n]
            i += 2 + n
        else:
            i += 1
    vals = np.array(tok[i + 6:], float).reshape(int(tok[i + 2]), -1)
    out = {n: np.zeros((len(vals), 3)) for n in names if n}
    for t, row in enumerate(vals):
        col, G, P = 0, {}, {}
        for k, n in enumerate(names):
            ch = chans[k]
            v = dict(zip(ch, row[col:col + len(ch)]))
            col += len(ch)
            R = Rotation.from_euler("ZXY", [v.get("Zrotation", 0), v.get("Xrotation", 0), v.get("Yrotation", 0)],
                                    degrees=True)
            if parent[k] is None:
                P[k], G[k] = offset[k] + [v.get("Xposition", 0), v.get("Yposition", 0), v.get("Zposition", 0)], R
            else:
                P[k], G[k] = P[parent[k]] + G[parent[k]].apply(offset[k]), G[parent[k]] * R
            if n:
                out[n][t] = P[k]
    return out


def _gltf_fk(glb: Path, tracks: dict[str, list], T: int) -> dict[str, np.ndarray]:
    g = pygltflib.GLTF2().load_binary(str(glb))
    par = [None] * len(g.nodes)
    for i, nd in enumerate(g.nodes):
        for c in nd.children or []:
            par[c] = i
    out = {nd.name: np.zeros((T, 3)) for nd in g.nodes if nd.name}
    for t in range(T):
        W: dict[int, np.ndarray] = {}

        def world(i):
            if i not in W:
                nd = g.nodes[i]
                q = tracks[nd.name][t]["rotation"] if nd.name in tracks else (nd.rotation or [0, 0, 0, 1])
                L = np.eye(4)
                L[:3, :3] = Rotation.from_quat(q).as_matrix() * np.array(nd.scale or [1, 1, 1])
                L[:3, 3] = nd.translation or [0, 0, 0]
                W[i] = L if par[i] is None else world(par[i]) @ L
            return W[i]

        for i, nd in enumerate(g.nodes):
            if nd.name:
                out[nd.name][t] = world(i)[:3, 3]
    return out


def _pairs():
    p = [("MID_HIP", "MID_SHOULDER", "Hips", "Spine2"), ("MID_SHOULDER", "HEAD", "Spine2", "Head")]
    for S, s in (("LEFT", "Left"), ("RIGHT", "Right")):
        p += [(f"{S}_SHOULDER", f"{S}_ELBOW", f"{s}Arm", f"{s}ForeArm"),
              (f"{S}_ELBOW", f"{S}_WRIST", f"{s}ForeArm", f"{s}Hand"),
              (f"{S}_WRIST", f"{S}_HAND_MIDDLE_FINGER_MCP", f"{s}Hand", f"{s}HandMiddle1"),
              (f"{S}_HIP", f"{S}_KNEE", f"{s}UpLeg", f"{s}Leg"),
              (f"{S}_KNEE", f"{S}_ANKLE", f"{s}Leg", f"{s}Foot"),
              (f"{S}_ANKLE", f"{S}_FOOT_INDEX", f"{s}Foot", f"{s}ToeBase")]
        for F, f, js in (("THUMB", "Thumb", ["CMC", "MCP", "IP", "TIP"]),
                         ("INDEX_FINGER", "Index", ["MCP", "PIP", "DIP", "TIP"]),
                         ("MIDDLE_FINGER", "Middle", ["MCP", "PIP", "DIP", "TIP"]),
                         ("RING_FINGER", "Ring", ["MCP", "PIP", "DIP", "TIP"]),
                         ("PINKY", "Pinky", ["MCP", "PIP", "DIP", "TIP"])):
            for k in range(3):
                p.append((f"{S}_HAND_{F}_{js[k]}", f"{S}_HAND_{F}_{js[k + 1]}", f"{s}Hand{f}{k + 1}", f"{s}Hand{f}{k + 2}"))
    return p


def _angle(u, v):
    c = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def _angles(bvh_pos, mix_pos):
    return {f"{ma}->{mb}": _angle(bvh_pos[b] - bvh_pos[a], mix_pos[M + mb] - mix_pos[M + ma])
            for a, b, ma, mb in _pairs()}


# (direction principale, direction secondaire) d'un meme segment rigide, cote BVH puis cote Mixamo.
SECONDARY = [
    (("MID_HIP", "MID_SHOULDER"), ("RIGHT_HIP", "LEFT_HIP"), ("Hips", "Spine2"), ("RightUpLeg", "LeftUpLeg")),
    (("MID_SHOULDER", "HEAD"), ("RIGHT_SHOULDER", "LEFT_SHOULDER"), ("Spine2", "Head"), ("RightArm", "LeftArm")),
] + [
    ((f"{S}_WRIST", f"{S}_HAND_MIDDLE_FINGER_MCP"), (f"{S}_HAND_INDEX_FINGER_MCP", f"{S}_HAND_PINKY_MCP"),
     (f"{s}Hand", f"{s}HandMiddle1"), (f"{s}HandIndex1", f"{s}HandPinky1"))
    for S, s in (("LEFT", "Left"), ("RIGHT", "Right"))
]


@pytest.fixture
def bvh(tmp_path):
    if not CLARA.is_file():
        pytest.skip("clara.glb absent : pas de modele Mixamo reel")
    return _write_bvh(tmp_path / "random.bvh")


def test_every_bone_points_like_the_bvh_on_hostile_poses(bvh):
    # Bug attrape : ecart de repos non corrige, mauvais ordre Euler, parent mal compose, os oublie.
    res = rt.retarget_bvh_to_mixamo(bvh, CLARA)
    err = _angles(_bvh_fk(bvh), _gltf_fk(CLARA, res["tracks"], res["frameCount"]))
    worst = max(err, key=lambda k: err[k].max())
    assert err[worst].max() < 0.01, (worst, err[worst])
    assert res["meta"]["bonesSkipped"] == []


def test_twist_residual_is_only_the_shape_difference(bvh):
    # Une direction secondaire (ligne des hanches, des epaules, index->auriculaire) ne peut etre
    # alignee exactement que si l'angle principal/secondaire est le meme sur les deux squelettes.
    # Oracle geometrique : principal aligne + secondaires dans le meme demi-plan => ecart = |theta_bvh - theta_mix|.
    # Bug attrape : torsion fausse ou de mauvais signe (l'ecart depasserait cette borne).
    res = rt.retarget_bvh_to_mixamo(bvh, CLARA)
    bp, mp = _bvh_fk(bvh), _gltf_fk(CLARA, res["tracks"], res["frameCount"])
    for (p0, p1), (s0, s1), (m0, m1), (n0, n1) in SECONDARY:
        bu, bs = bp[p1] - bp[p0], bp[s1] - bp[s0]
        mu, ms = mp[M + m1] - mp[M + m0], mp[M + n1] - mp[M + n0]
        expected = np.abs(_angle(bu, bs) - _angle(mu, ms))
        assert np.allclose(_angle(bs, ms), expected, atol=1e-3), (s0, _angle(bs, ms), expected)


def test_without_rest_alignment_the_rest_gap_remains(bvh):
    # Sensibilite : sans correction, la T-pose de Clara (pouce a ~20 deg) doit se voir.
    res = rt.retarget_bvh_to_mixamo(bvh, CLARA, align_rest=False)
    err = _angles(_bvh_fk(bvh), _gltf_fk(CLARA, res["tracks"], res["frameCount"]))
    assert err["LeftHandThumb1->LeftHandThumb2"].min() > 10.0
    # Ecart constant quelle que soit la pose ; 1e-3 deg = marge au-dessus du bruit float32 du fichier
    # (echelles de Clara non uniformes a 7e-6 pres).
    assert np.ptp(err["LeftArm->LeftForeArm"]) < 1e-3


def test_no_alignment_equals_univrm_formula_for_one_to_one_bones(bvh):
    # Oracle : la formule d'origine B.local = L_B . W_B^-1 . N . W_B, ecrite ici.
    res = rt.retarget_bvh_to_mixamo(bvh, CLARA, align_rest=False)
    joints, _, motion = rt.parse_bvh(bvh)
    local = rt.extract_bvh_rotations(joints, motion)
    rest = rt.read_gltf_rest_pose(CLARA)
    for bvh_name, mix in (("LEFT_ELBOW", "LeftForeArm"), ("RIGHT_HAND_INDEX_FINGER_PIP", "RightHandIndex2"),
                          ("LEFT_HAND_THUMB_IP", "LeftHandThumb3")):
        L, W = rest[M + mix]
        for t, n_rot in enumerate(local[bvh_name]):
            expected = L * W.inv() * n_rot * W
            got = Rotation.from_quat(res["tracks"][M + mix][t]["rotation"])
            assert (expected.inv() * got).magnitude() < 1e-9


def test_quaternions_are_unit_and_sign_continuous(bvh):
    res = rt.retarget_bvh_to_mixamo(bvh, CLARA)
    for bone, kfs in res["tracks"].items():
        q = np.array([k["rotation"] for k in kfs])
        assert np.allclose(np.linalg.norm(q, axis=1), 1.0), bone
        assert ((q[1:] * q[:-1]).sum(1) >= 0).all(), bone
    assert res["fps"] == pytest.approx(30.0, rel=1e-4)
    assert "aucune" in res["meta"]["rootMotion"]


@pytest.mark.parametrize("mutate,err", [
    (lambda s: s.replace("HIERARCHY", "HIERARCHIE", 1), ValueError),
    (lambda s: s.replace("Frames: 6", "Frames: 7"), ValueError),
    (lambda s: s.rsplit("\n", 2)[0] + "\n", ValueError),                    # derniere frame manquante
    (lambda s: s.replace("OFFSET 0 1 0", "OFFSET a b c", 1), ValueError),
    (lambda s: s.replace("Frame Time: 0.033333", "Frame Time: 0"), ValueError),
    (lambda s: "", ValueError),
    (lambda s: s.replace("CHANNELS 3 Zrotation", "CHANNELS 4 Zrotation", 1), ValueError),
    (lambda s: s.replace("CHANNELS 3 Zrotation Xrotation", "CHANNELS 3 Zrotation Wrotation", 1), ValueError),
    (lambda s: s.split("MOTION")[0][:-40], ValueError),                     # hierarchie tronquee
    (lambda s: s[:-10] + " nan\n", ValueError),
    (lambda s: s.rstrip("\n") + " 1.0\n", ValueError),                      # valeur en trop
])
def test_invalid_bvh_raises_clear_error(tmp_path, mutate, err):
    p = _write_bvh(tmp_path / "ok.bvh")
    bad = tmp_path / "bad.bvh"
    bad.write_text(mutate(p.read_text(encoding="utf-8")), encoding="utf-8")
    with pytest.raises(err, match="BVH"):
        rt.parse_bvh(bad)


def test_missing_files_and_bad_target(tmp_path, bvh):
    with pytest.raises(FileNotFoundError):
        rt.retarget_bvh_to_mixamo(tmp_path / "absent.bvh", CLARA)
    with pytest.raises(FileNotFoundError):
        rt.retarget_bvh_to_mixamo(bvh, tmp_path / "absent.glb")
    g = pygltflib.GLTF2().load_binary(str(CLARA))
    g.nodes[5].scale = [1.0, 2.0, 1.0]
    with pytest.raises(ValueError, match="echelle"):
        rt._GltfRest(g)


def test_cli_writes_a_clip_that_moves_clara_like_the_bvh(bvh, tmp_path, capsys):
    # Bug attrape : commande non cablee, option --no-align-rest ignoree, clip non ecrit sur disque.
    import json
    from mimo.main import main

    for flag, check in (([], lambda e: e.max() < 0.01), (["--no-align-rest"], lambda e: e.min() > 10.0)):
        out = tmp_path / "sub" / "clip.json"
        assert main(["retarget", str(bvh), str(CLARA), "-o", str(out), *flag]) == 0
        clip = json.loads(out.read_text(encoding="utf-8"))
        err = _angles(_bvh_fk(bvh), _gltf_fk(CLARA, clip["tracks"], clip["frameCount"]))
        assert check(err["LeftHandThumb1->LeftHandThumb2"]), (flag, err["LeftHandThumb1->LeftHandThumb2"])
        summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert summary["restAlignment"] is (not flag)
        assert summary["bonesAnimated"] == len(clip["tracks"]) == 45


@pytest.mark.parametrize("model,message", [
    ("exam.vrm", "aucun os"),                 # VRM reel, sans os mixamorig
    ("a.mp4", "extension .mp4"),
    ("fake.glb", "en-tete 'glTF'"),
    ("absent.glb", "introuvable"),
])
def test_cli_rejects_wrong_target_with_clear_error(bvh, tmp_path, capsys, model, message):
    from mimo.main import main

    root = CLARA.parent
    target = root / model
    if model == "fake.glb":
        target = tmp_path / model
        target.write_bytes((root / "a.mp4").read_bytes()[:4096])
    elif model == "absent.glb":
        target = tmp_path / model
    elif not target.is_file():
        pytest.skip(f"{model} absent")
    out = tmp_path / "clip.json"
    assert main(["retarget", str(bvh), str(target), "-o", str(out)]) == 2
    assert message in capsys.readouterr().err
    assert not out.exists()


def test_bones_absent_from_target_are_reported(bvh):
    bone_map = dict(rt.BVH_TO_MIXAMO)
    bone_map["LEFT_ELBOW"] = "mixamorig:NoSuchBone"
    res = rt.retarget_bvh_to_mixamo(bvh, CLARA, bone_map=bone_map)
    assert res["meta"]["bonesSkipped"] == ["mixamorig:NoSuchBone"]
    assert "mixamorig:LeftForeArm" not in res["tracks"]
