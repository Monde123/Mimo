"""Bake d'un clip retargeté dans le .glb Mixamo.

Oracle : les octets de l'animation sont relus directement dans le chunk binaire du .glb
(np.frombuffer sur bufferView/accessor), puis rejoués par la FK glTF indépendante de
test_retarget_bvh_to_mixamo et comparés aux directions du BVH source.
"""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

pygltflib = pytest.importorskip("pygltflib")

from backend import retarget_bvh_to_mixamo as rt  # noqa: E402
from backend.bake_mixamo_animation import bake_retarget_into_glb  # noqa: E402
from test_retarget_bvh_to_mixamo import CLARA, _angles, _bvh_fk, _gltf_fk, _write_bvh  # noqa: E402


def _read_animation(glb, name):
    g = pygltflib.GLTF2().load_binary(str(glb))
    blob = g.binary_blob()
    anim = next(a for a in g.animations if a.name == name)

    def data(acc_idx, width):
        acc = g.accessors[acc_idx]
        bv = g.bufferViews[acc.bufferView]
        start = bv.byteOffset + (acc.byteOffset or 0)
        return np.frombuffer(blob, "<f4", acc.count * width, start).reshape(acc.count, width)

    out = {}
    for ch in anim.channels:
        s = anim.samplers[ch.sampler]
        assert ch.target.path == "rotation" and s.interpolation == "LINEAR"
        times, quats = data(s.input, 1)[:, 0], data(s.output, 4)
        out[g.nodes[ch.target.node].name] = [{"time": float(t), "rotation": q.astype(float).tolist()}
                                             for t, q in zip(times, quats)]
    return g, out


@pytest.fixture
def clip(tmp_path):
    if not CLARA.is_file():
        pytest.skip("clara.glb absent : pas de modele Mixamo reel")
    bvh = _write_bvh(tmp_path / "random.bvh")
    return bvh, rt.retarget_bvh_to_mixamo(bvh, CLARA)


def test_baked_glb_moves_clara_like_the_bvh(clip, tmp_path):
    # Bug attrape : mauvais noeud cible, octets decales (alignement, offset), ordre xyzw inverse,
    # temps faux, quaternions tronques.
    bvh, c = clip
    out = tmp_path / "clara_anim.glb"
    res = bake_retarget_into_glb(c, CLARA, out)
    g, tracks = _read_animation(out, "MimoRetarget")
    assert set(tracks) == set(c["tracks"]) and res["bonesMissing"] == []
    times = np.array([k["time"] for k in tracks["mixamorig:LeftArm"]])
    assert np.allclose(times, np.arange(c["frameCount"]) / c["fps"], atol=1e-6)
    err = _angles(_bvh_fk(bvh), _gltf_fk(CLARA, tracks, c["frameCount"]))
    worst = max(err, key=lambda k: err[k].max())
    assert err[worst].max() < 0.01, (worst, err[worst])          # float32 : ~1e-5 deg attendu


def test_original_model_data_is_untouched(clip, tmp_path):
    # Invariant : le maillage, la peau et les animations retirees ne doivent pas deplacer les octets existants.
    _, c = clip
    out = tmp_path / "clara_anim.glb"
    bake_retarget_into_glb(c, CLARA, out)
    src = pygltflib.GLTF2().load_binary(str(CLARA))
    dst = pygltflib.GLTF2().load_binary(str(out))
    assert dst.binary_blob()[:len(src.binary_blob())] == src.binary_blob()
    # pygltflib.Attributes n'a pas d'__eq__ : comparaison sur le JSON glTF serialise.
    s, d = json.loads(src.to_json()), json.loads(dst.to_json())
    for key in ("nodes", "meshes", "skins", "materials", "textures", "images", "scenes"):
        assert d.get(key) == s.get(key), key
    assert d["bufferViews"][:len(s["bufferViews"])] == s["bufferViews"]
    assert d["accessors"][:len(s["accessors"])] == s["accessors"]


def test_existing_animations_policy(clip, tmp_path):
    _, c = clip
    once = tmp_path / "once.glb"
    bake_retarget_into_glb(c, CLARA, once, animation_name="A")
    keep = tmp_path / "keep.glb"
    r = bake_retarget_into_glb(c, once, keep, animation_name="B", keep_existing_animations=True)
    assert [a.name for a in pygltflib.GLTF2().load_binary(str(keep)).animations] == ["A", "B"]
    assert r["existingAnimationsRemoved"] == 0
    same = tmp_path / "same.glb"
    r = bake_retarget_into_glb(c, keep, same, animation_name="B", keep_existing_animations=True)
    assert [a.name for a in pygltflib.GLTF2().load_binary(str(same)).animations] == ["A", "B"]
    assert r["existingAnimationsRemoved"] == 1
    drop = tmp_path / "drop.glb"
    r = bake_retarget_into_glb(c, keep, drop, animation_name="C")
    assert [a.name for a in pygltflib.GLTF2().load_binary(str(drop)).animations] == ["C"]
    assert r["existingAnimationsRemoved"] == 2


def _mutate(c, fn):
    c = copy.deepcopy(c)
    fn(c["tracks"]["mixamorig:LeftArm"], c)
    return c


@pytest.mark.parametrize("fn,message", [
    (lambda kfs, c: c.update(tracks={}), "aucune piste"),
    (lambda kfs, c: c.pop("tracks"), "tracks"),
    (lambda kfs, c: kfs[3].update(rotation=[0, 0, 0, 0]), "quaternion nul"),
    (lambda kfs, c: kfs[3].update(rotation=[0, 0, float("nan"), 1]), "non fini"),
    (lambda kfs, c: kfs[3].update(rotation=[0, 0, 1]), r"rotation.*\[x, y, z, w\]"),
    (lambda kfs, c: [k.update(rotation=[0, 0, 0, 1, 0]) for k in kfs], "forme"),     # composante en trop
    (lambda kfs, c: kfs[3].update(time=kfs[2]["time"]), "strictement croissants"),
    (lambda kfs, c: kfs[0].update(time=-0.1), "strictement croissants"),
    (lambda kfs, c: kfs[3].pop("time"), "'time'"),
])
def test_invalid_clip_raises_clear_error(clip, tmp_path, fn, message):
    _, c = clip
    out = tmp_path / "bad.glb"
    with pytest.raises(ValueError, match=message):
        bake_retarget_into_glb(_mutate(c, fn), CLARA, out)
    assert not out.exists()


def test_wrong_target_raises_clear_error(clip, tmp_path):
    _, c = clip
    renamed = {"tracks": {k.replace("mixamorig:", "mixamorig1:"): v for k, v in c["tracks"].items()}}
    with pytest.raises(RuntimeError, match="Aucun os"):
        bake_retarget_into_glb(renamed, CLARA, tmp_path / "x.glb")
    with pytest.raises(ValueError, match=r"\.glb"):
        bake_retarget_into_glb(c, CLARA, tmp_path / "x.gltf")
    fake = tmp_path / "fake.glb"
    fake.write_bytes(b"\x00" * 64)
    with pytest.raises(ValueError, match="glTF"):
        bake_retarget_into_glb(c, fake, tmp_path / "x.glb")
    with pytest.raises(FileNotFoundError):
        bake_retarget_into_glb(c, tmp_path / "absent.glb", tmp_path / "x.glb")


def test_partial_target_reports_missing_bones(clip, tmp_path):
    _, c = clip
    c = copy.deepcopy(c)
    c["tracks"]["mixamorig:NoSuchBone"] = c["tracks"]["mixamorig:LeftArm"]
    res = bake_retarget_into_glb(c, CLARA, tmp_path / "x.glb")
    assert res["bonesMissing"] == ["mixamorig:NoSuchBone"]
    assert res["bonesMatched"] == len(c["tracks"]) - 1


def test_cli_retarget_to_glb_and_bake_mixamo_give_same_file(clip, tmp_path, capsys):
    # Bug attrape : les deux chemins CLI divergent (options non transmises, JSON relu differemment).
    from mimo.main import main

    bvh, _ = clip
    direct, js, baked = tmp_path / "direct.glb", tmp_path / "clip.json", tmp_path / "baked.glb"
    assert main(["retarget", str(bvh), str(CLARA), "-o", str(direct), "--name", "Signe"]) == 0
    assert main(["retarget", str(bvh), str(CLARA), "-o", str(js)]) == 0
    assert main(["bake-mixamo", str(js), str(CLARA), "-o", str(baked), "--name", "Signe"]) == 0
    assert direct.read_bytes() == baked.read_bytes()
    assert [a.name for a in pygltflib.GLTF2().load_binary(str(direct)).animations] == ["Signe"]
    capsys.readouterr()
    assert main(["bake-mixamo", str(tmp_path / "absent.json"), str(CLARA), "-o", str(baked)]) == 2
    assert "introuvable" in capsys.readouterr().err
    (tmp_path / "broken.json").write_text("{pas du json", encoding="utf-8")
    assert main(["bake-mixamo", str(tmp_path / "broken.json"), str(CLARA), "-o", str(tmp_path / "b.glb")]) == 2
    assert main(["retarget", str(bvh), str(CLARA), "-o", str(tmp_path / "x.bvh")]) == 2
