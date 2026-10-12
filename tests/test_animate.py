"""mimo animate : video(s) -> BVH -> .glb Mixamo anime, en une commande.

Bugs vises : option perdue entre les etapes, mauvais BVH passe au retarget, erreur d'arguments
detectee seulement APRES l'extraction (minutes perdues), monoculaire sans avertissement de profondeur.
Oracle : equivalence octet pour octet avec l'enchainement manuel deja verifie par FK
(mimo bvh / mimo multiview puis mimo retarget -o .glb, cf. test_retarget / test_bake).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pygltflib")

import backend.process_video as process_video_module  # noqa: E402
from mimo.main import main  # noqa: E402
from test_process_video import _holistic_frames  # noqa: E402
from test_retarget_bvh_to_mixamo import CLARA  # noqa: E402

ROOT = CLARA.parent


@pytest.fixture(autouse=True)
def _need_clara():
    if not CLARA.is_file():
        pytest.skip("clara.glb absent : pas de modele Mixamo reel")


@pytest.fixture
def mono_video(monkeypatch, tmp_path: Path) -> Path:
    video = tmp_path / "signer.mp4"
    video.write_bytes(b"extraction simulee")
    monkeypatch.setattr(process_video_module, "_probe_video", lambda path: (24.0, 640.0, 480.0))
    monkeypatch.setattr(process_video_module, "_select_extractor",
                        lambda pipeline, *a, **k: (lambda path: _holistic_frames(12)))
    return video


def _last_json(capsys) -> dict:
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_mono_equals_bvh_then_retarget(mono_video, tmp_path, capsys):
    manual_bvh, manual_glb = tmp_path / "m" / "s.bvh", tmp_path / "m" / "s.glb"
    # --hands off : les frames simulees ont des mains, donc "auto" donnerait un autre BVH.
    assert main(["bvh", str(mono_video), str(manual_bvh), "--body", "upper", "--hands", "off"]) == 0
    assert main(["retarget", str(manual_bvh), str(CLARA), "-o", str(manual_glb), "--name", "Signe"]) == 0
    capsys.readouterr()

    out = tmp_path / "a" / "s.glb"
    assert main(["animate", str(mono_video), "--model", str(CLARA), "-o", str(out),
                 "--body", "upper", "--hands", "off", "--name", "Signe"]) == 0
    summary = _last_json(capsys)
    assert out.with_suffix(".bvh").read_bytes() == manual_bvh.read_bytes()
    assert out.read_bytes() == manual_glb.read_bytes()
    assert summary["mode"] == "monocular"
    assert summary["bvh"] == str(out.with_suffix(".bvh"))
    assert any("profondeur" in w for w in summary["warnings"])


def test_mono_custom_bvh_path(mono_video, tmp_path, capsys):
    out, bvh = tmp_path / "s.glb", tmp_path / "inter" / "x.bvh"
    assert main(["animate", str(mono_video), "--model", str(CLARA), "-o", str(out), "--bvh", str(bvh)]) == 0
    assert bvh.read_text(encoding="utf-8").startswith("HIERARCHY")
    assert not out.with_suffix(".bvh").exists()
    assert out.read_bytes()[:4] == b"glTF"


def test_multiview_equals_multiview_then_retarget(tmp_path, capsys, monkeypatch, real_skeleton):
    from backend import multiview as mv
    from test_multiview import _camera_dicts, _place, _tracks, _write_calibration

    cams = _camera_dicts(2)
    calib = _write_calibration(tmp_path / "calibration.toml", cams)
    tracks = {c["name"]: t for c, t in zip(cams, _tracks(_place(real_skeleton), cams))}
    monkeypatch.setattr(mv, "extract_view_2d", lambda v, name=None, max_frames=None: tracks[name])
    videos = []
    for i in range(2):
        videos.append(tmp_path / f"cam{i}.mp4")
        videos[-1].write_bytes(b"extraction simulee")

    manual_bvh, manual_glb = tmp_path / "m" / "mv.bvh", tmp_path / "m" / "mv.glb"
    assert main(["multiview", *map(str, videos), "--calibration", str(calib), "-o", str(manual_bvh)]) == 0
    assert main(["retarget", str(manual_bvh), str(CLARA), "-o", str(manual_glb)]) == 0
    capsys.readouterr()

    out = tmp_path / "a" / "mv.glb"
    assert main(["animate", *map(str, videos), "--calibration", str(calib), "--model", str(CLARA),
                 "-o", str(out)]) == 0
    summary = _last_json(capsys)
    assert out.with_suffix(".bvh").read_bytes() == manual_bvh.read_bytes()
    assert out.read_bytes() == manual_glb.read_bytes()
    assert summary["mode"] == "multiview"
    assert not any("profondeur" in w for w in summary["warnings"])


from test_multiview import real_skeleton  # noqa: E402,F401  (fixture)


def _boom(*a, **k):
    raise AssertionError("extraction lancee malgre des arguments invalides")


@pytest.mark.parametrize("case, message", [
    ("two_videos_no_calibration", "--calibration"),
    ("one_video_with_calibration", "--calibration"),
    ("offsets_mono", "--offsets"),
    ("output_bvh", ".glb"),
    ("bvh_wrong_suffix", ".bvh"),
    ("model_fake", "glTF"),
    ("model_absent", "introuvable"),
    ("model_no_mixamo", "mixamorig"),
    ("video_absent", "introuvable"),
    ("preset_multiview", "--preset"),
    ("same_video_twice", "meme video"),
])
def test_bad_arguments_fail_before_extraction(case, message, tmp_path, monkeypatch, capsys):
    from backend import multiview as mv
    monkeypatch.setattr(process_video_module, "_select_extractor", _boom)
    monkeypatch.setattr(mv, "extract_view_2d", _boom)
    v0, v1 = tmp_path / "c0.mp4", tmp_path / "c1.mp4"
    v0.write_bytes(b"x")
    v1.write_bytes(b"x")
    calib = tmp_path / "calibration.toml"
    calib.write_text("", encoding="utf-8")
    out = tmp_path / "o.glb"
    model = CLARA
    args = [str(v0)]
    extra: list[str] = []
    if case == "two_videos_no_calibration":
        args = [str(v0), str(v1)]
    elif case == "one_video_with_calibration":
        extra = ["--calibration", str(calib)]
    elif case == "offsets_mono":
        extra = ["--offsets", "0"]
    elif case == "output_bvh":
        out = tmp_path / "o.bvh"
    elif case == "bvh_wrong_suffix":
        extra = ["--bvh", str(tmp_path / "x.txt")]
    elif case == "model_fake":
        model = tmp_path / "fake.glb"
        model.write_bytes(b"not a gltf file at all")
    elif case == "model_absent":
        model = tmp_path / "absent.glb"
    elif case == "model_no_mixamo":
        model = ROOT / "exam.vrm"
        if not model.is_file():
            pytest.skip("exam.vrm absent")
    elif case == "video_absent":
        args = [str(tmp_path / "absent.mp4")]
    elif case == "preset_multiview":
        args = [str(v0), str(v1)]
        extra = ["--calibration", str(calib), "--preset", "sign"]
    elif case == "same_video_twice":
        args = [str(v0), str(tmp_path / "." / "c0.mp4")]
        extra = ["--calibration", str(calib)]
    code = main(["animate", *args, "--model", str(model), "-o", str(out), *extra])
    err = capsys.readouterr().err
    assert code == 2, err
    assert message in err
    assert not out.exists()
