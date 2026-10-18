"""mimo animate : video -> BVH -> .glb Mixamo anime, en une commande (monoculaire).

Bugs vises : option perdue entre les etapes, mauvais BVH passe au retarget, erreur d'arguments
detectee seulement APRES l'extraction (minutes perdues).
Oracle : equivalence octet pour octet avec l'enchainement manuel deja verifie par FK
(mimo bvh puis mimo retarget -o .glb, cf. test_retarget / test_bake).
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
                        lambda pipeline: (lambda path: _holistic_frames(12)))
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


def test_mono_custom_bvh_path(mono_video, tmp_path, capsys):
    out, bvh = tmp_path / "s.glb", tmp_path / "inter" / "x.bvh"
    assert main(["animate", str(mono_video), "--model", str(CLARA), "-o", str(out), "--bvh", str(bvh)]) == 0
    assert bvh.read_text(encoding="utf-8").startswith("HIERARCHY")
    assert not out.with_suffix(".bvh").exists()
    assert out.read_bytes()[:4] == b"glTF"


def _boom(*a, **k):
    raise AssertionError("extraction lancee malgre des arguments invalides")


@pytest.mark.parametrize("case, message", [
    ("output_bvh", ".glb"),
    ("bvh_wrong_suffix", ".bvh"),
    ("model_fake", "glTF"),
    ("model_absent", "introuvable"),
    ("model_no_mixamo", "mixamorig"),
    ("video_absent", "introuvable"),
])
def test_bad_arguments_fail_before_extraction(case, message, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(process_video_module, "_select_extractor", _boom)
    v0 = tmp_path / "c0.mp4"
    v0.write_bytes(b"x")
    out = tmp_path / "o.glb"
    model = CLARA
    extra: list[str] = []
    if case == "output_bvh":
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
        v0 = tmp_path / "absent.mp4"
    code = main(["animate", str(v0), "--model", str(model), "-o", str(out), *extra])
    err = capsys.readouterr().err
    assert code == 2, err
    assert message in err
    assert not out.exists()
