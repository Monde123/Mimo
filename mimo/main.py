from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _add_bvh_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--max-bad-frames", type=int, default=10)
    parser.add_argument("--bvh-mode", choices=["rotations", "positions"], default="rotations")
    parser.add_argument("--source", choices=["auto", "world", "normalized"], default="auto")
    parser.add_argument("--aspect", type=float, default=None)
    parser.add_argument("--scale", type=float, default=100.0)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--flip-x", action="store_true")
    parser.add_argument("--flip-y", action="store_true")
    parser.add_argument("--flip-z", action="store_true")
    parser.add_argument("--no-recenter", action="store_true")
    parser.add_argument("--preset", choices=["sign", "full", "parallel", "precise"], default=None)
    parser.add_argument("--body", choices=["auto", "full", "upper"], default=None)
    parser.add_argument("--hands", choices=["auto", "on", "off"], default=None)
    parser.add_argument("--hand-size-cm", type=float, default=9.5)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--pipeline", choices=["holistic", "pose_hands", "hybrid"], default=None)
    
 
    parser.add_argument("--parallel-device", default="auto")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mimo",
        description="Mimo: extraction de mouvements et export d'animations.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    bvh = subparsers.add_parser("bvh", help="extraire une vidéo vers un fichier BVH")
    _add_bvh_arguments(bvh)

    mixamo = subparsers.add_parser("mixamo", help="extraire une vidéo vers un clip JSON Mixamo")
    mixamo.add_argument("input", type=Path)
    mixamo.add_argument("output", type=Path)
    mixamo.add_argument("--fps", type=float, default=None)
    mixamo.add_argument("--rig", type=Path, default=None)
    mixamo.add_argument("--max-bad-frames", type=int, default=10)

    vrma = subparsers.add_parser("vrma", help="extraire une vidéo vers un clip d'animation VRM (VRMA binaire ou JSON) pour avatar 3D")
    vrma.add_argument("input", type=Path)
    vrma.add_argument("output", type=Path)
    vrma.add_argument("--fps", type=float, default=None)
    vrma.add_argument("--pipeline", choices=["hybrid", "pose_hands", "holistic"], default="hybrid")
    vrma.add_argument("--binary", action="store_true", help="générer un fichier .vrma binaire conforme glTF/VRMC_vrm_animation (par défaut si l'extension est .vrma)")

    bake_vrm = subparsers.add_parser("bake-vrm", help="injecter et béké un clip VRMA dans un modèle 3D VRM")
    bake_vrm.add_argument("model", type=Path, help="chemin vers le modèle VRM (ex: avatar.vrm)")
    bake_vrm.add_argument("vrma", type=Path, help="chemin vers le clip VRMA json")
    bake_vrm.add_argument("output", type=Path, nargs="?", default=None, help="modèle béké de sortie")

    inspect_vrm = subparsers.add_parser("inspect-vrm", help="inspection visuelle des poses et des doigts VRMA dans le terminal")
    inspect_vrm.add_argument("vrma", type=Path, help="chemin vers le clip VRMA json")

    anipose = subparsers.add_parser("anipose", help="auto-test de triangulation multi-vues inspiré d'Anipose (rig synthétique)")
    anipose.add_argument("--test", action="store_true", help="exécuter l'auto-test de triangulation sur un rig synthétique étalonné")
    anipose.add_argument("--cams", type=int, default=2, help="nombre de caméras synthétiques (2 à 5)")

    calibrate = subparsers.add_parser("calibrate", help="calibrer 2 caméras ou plus avec une planche ChArUco (calibration.toml Anipose)")
    calibrate.add_argument("videos", type=Path, nargs="+", help="une vidéo de la planche par caméra, filmées en même temps")
    calibrate.add_argument("-o", "--output", type=Path, required=True, help="fichier calibration.toml à écrire")
    calibrate.add_argument("--squares-x", type=int, required=True, help="nombre de cases en largeur")
    calibrate.add_argument("--squares-y", type=int, required=True, help="nombre de cases en hauteur")
    calibrate.add_argument("--board", choices=["charuco", "checkerboard"], default="charuco")
    calibrate.add_argument("--square-length-m", type=float, required=True, help="côté d'une case, en mètres")
    calibrate.add_argument("--marker-length-m", type=float, default=None, help="côté d'un marqueur ArUco, en mètres (ChArUco)")
    calibrate.add_argument("--marker-bits", type=int, default=4)
    calibrate.add_argument("--dict-size", type=int, default=50)
    calibrate.add_argument("--names", nargs="+", default=None, help="noms des caméras (cam0 cam1 ... par défaut)")

    multiview = subparsers.add_parser("multiview", help="trianguler 2 vidéos synchronisées ou plus en une animation 3D (BVH / VRMA)")
    multiview.add_argument("videos", type=Path, nargs="+", help="une vidéo par caméra, dans l'ordre de la calibration")
    multiview.add_argument("-o", "--output", type=Path, required=True, help="sortie .bvh, .vrma ou .json (clip VRMA)")
    multiview.add_argument("--calibration", type=Path, required=True, help="calibration.toml (mimo calibrate ou Anipose)")
    multiview.add_argument("--offsets", type=int, nargs="+", default=None,
                           help="frames à ignorer au début de chaque vidéo pour les synchroniser (>= 0)")
    multiview.add_argument("--unit-scale", type=float, default=None,
                           help="facteur unités de calibration -> mètres, si metadata.units est absent")
    multiview.add_argument("--min-score", type=float, default=0.5, help="confiance 2D minimale par vue")
    multiview.add_argument("--max-reproj-px", type=float, default=10.0, help="erreur de reprojection maximale (pixels)")
    multiview.add_argument("--max-frames", type=int, default=None)
    multiview.add_argument("--bvh-mode", choices=["rotations", "positions"], default="rotations")
    multiview.add_argument("--body", choices=["auto", "full", "upper"], default="auto")
    multiview.add_argument("--hands", choices=["auto", "on", "off"], default="auto")

    retarget = subparsers.add_parser("retarget", help="rejouer un BVH Mimo sur un modèle Mixamo glTF (clip JSON de quaternions)")
    retarget.add_argument("bvh", type=Path, help="BVH produit par mimo bvh ou mimo multiview")
    retarget.add_argument("model", type=Path, help="modèle Mixamo .glb / .gltf (os mixamorig:*)")
    retarget.add_argument("-o", "--output", type=Path, required=True,
                          help="clip .json, ou .glb = modèle avec l'animation intégrée")
    retarget.add_argument("--no-align-rest", action="store_true",
                          help="ne pas recaler la pose de repos du modèle sur celle du BVH (formule UniVRM brute)")
    retarget.add_argument("--name", default="MimoRetarget", help="nom de l'animation (sortie .glb)")
    retarget.add_argument("--keep-animations", action="store_true",
                          help="garder les animations déjà présentes dans le modèle (sortie .glb)")

    bake_mixamo = subparsers.add_parser("bake-mixamo", help="intégrer un clip de mimo retarget dans le modèle Mixamo .glb")
    bake_mixamo.add_argument("clip", type=Path, help="clip JSON produit par mimo retarget")
    bake_mixamo.add_argument("model", type=Path, help="modèle Mixamo .glb ciblé par le clip")
    bake_mixamo.add_argument("-o", "--output", type=Path, required=True, help=".glb animé à écrire")
    bake_mixamo.add_argument("--name", default="MimoRetarget", help="nom de l'animation")
    bake_mixamo.add_argument("--keep-animations", action="store_true",
                             help="garder les animations déjà présentes (seule celle du même nom est remplacée)")

    animate = subparsers.add_parser(
        "animate", help="vidéo(s) -> BVH -> .glb Mixamo animé en une commande (1 vidéo = monoculaire, 2+ = multivue)")
    animate.add_argument("videos", type=Path, nargs="+", help="1 vidéo, ou une vidéo par caméra dans l'ordre de la calibration")
    animate.add_argument("--model", type=Path, required=True, help="modèle Mixamo .glb (os mixamorig:*)")
    animate.add_argument("-o", "--output", type=Path, required=True, help=".glb animé à écrire")
    animate.add_argument("--bvh", type=Path, default=None, help="BVH intermédiaire (défaut : à côté de la sortie)")
    animate.add_argument("--calibration", type=Path, default=None, help="calibration.toml, obligatoire avec 2 vidéos ou plus")
    animate.add_argument("--name", default="MimoRetarget", help="nom de l'animation")
    animate.add_argument("--keep-animations", action="store_true", help="garder les animations déjà présentes dans le modèle")
    animate.add_argument("--body", choices=["auto", "full", "upper"], default=None)
    animate.add_argument("--hands", choices=["auto", "on", "off"], default=None)
    animate.add_argument("--preset", choices=["sign", "full", "precise"], default=None, help="monoculaire uniquement")
    animate.add_argument("--pipeline", choices=["holistic", "pose_hands", "hybrid"], default=None, help="monoculaire uniquement")
    animate.add_argument("--fps", type=float, default=None, help="monoculaire uniquement")
    animate.add_argument("--offsets", type=int, nargs="+", default=None, help="multivue uniquement")
    animate.add_argument("--unit-scale", type=float, default=None, help="multivue uniquement")
    animate.add_argument("--max-frames", type=int, default=None, help="multivue uniquement")

    server = subparsers.add_parser("server", help="démarrer le serveur Flask optionnel")
    server.add_argument("--host", default="127.0.0.1", help="interface d'écoute (127.0.0.1 par défaut)")
    server.add_argument("--port", type=int, default=5000)
    server.add_argument("--debug", action="store_true", help="mode debug Flask, réservé au développement local")
    return parser


def _run_anipose(args: argparse.Namespace) -> int:
    if not args.test:
        print("Seul l'auto-test synthétique est disponible : relancer avec --test.")
        return 2
    import numpy as np
    from backend.anipose_triangulation import create_synthetic_sign_rig, AniposeMultiViewTriangulator

    print(f"Initialisation du banc Anipose multi-vues ({args.cams} caméras calibrées)...")
    cams = create_synthetic_sign_rig(num_cameras=args.cams)
    triangulator = AniposeMultiViewTriangulator(cams)
    test_pt = np.array([0.12, -0.04, 1.35])
    observations = []
    for cam in cams:
        u, v, _ = cam.project_point_3d(test_pt)
        observations.append((u, v))
    reconstructed, err = triangulator.triangulate_point(observations)
    print(json.dumps({
        "status": "anipose_ready",
        "cameras": [c.name for c in cams],
        "ground_truth_3d": test_pt.tolist(),
        "triangulated_3d": reconstructed.tolist(),
        "reprojection_error_px": round(err, 4),
        "spatial_precision_mm": round(float(np.linalg.norm(reconstructed - test_pt)) * 1000.0, 3)
    }, indent=2, ensure_ascii=False))
    return 0


def _run_calibrate(args: argparse.Namespace) -> int:
    from backend.multiview import BoardSpec, calibrate_cameras
    board = BoardSpec(args.squares_x, args.squares_y, args.square_length_m, args.marker_length_m,
                      args.marker_bits, args.dict_size, kind=args.board)
    result = calibrate_cameras(args.videos, args.output, board, names=args.names)
    print(json.dumps(result, ensure_ascii=False))
    return 0


def _run_multiview(args: argparse.Namespace) -> int:
    from backend.multiview import run_multiview
    report = run_multiview(
        args.videos, args.calibration, args.output, offsets=args.offsets, unit_scale=args.unit_scale,
        min_score=args.min_score, max_reproj_px=args.max_reproj_px, max_frames=args.max_frames,
        bvh_mode=args.bvh_mode, body=args.body, hands=args.hands)
    print(json.dumps({
        "status": "complete",
        "output": report["output"],
        "report": report["report"],
        "frames": report["sync"]["frames"],
        "coverage": {k: v for k, v in report["coverage"].items() if k != "per_body_point"},
        "reprojection_error_px": report["reprojection_error_px"]["all"],
        "warnings": report["warnings"],
    }, ensure_ascii=False))
    return 0


def _has_pygltflib(command: str) -> bool:
    try:
        import pygltflib  # noqa: F401
    except ModuleNotFoundError:
        print(f"pygltflib est requis pour mimo {command} : pip install -e .[glb]", file=sys.stderr)
        return False
    return True


def _retarget(bvh: Path, model: Path, output: Path, align_rest: bool = True,
              name: str = "MimoRetarget", keep_animations: bool = False) -> dict:
    from backend.retarget_bvh_to_mixamo import retarget_bvh_to_mixamo
    suffix = output.suffix.lower()
    if suffix not in (".json", ".glb"):
        raise ValueError(f"sortie {output.name} : extension .json ou .glb attendue")
    clip = retarget_bvh_to_mixamo(bvh, model, align_rest=align_rest)
    baked = None
    if suffix == ".glb":
        from backend.bake_mixamo_animation import bake_retarget_into_glb
        baked = bake_retarget_into_glb(clip, model, output, name, keep_animations)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(clip, indent=2, ensure_ascii=False), encoding="utf-8")
    corrections = {k: v for k, v in clip["meta"]["restCorrectionDeg"].items() if v is not None}
    worst = max(corrections, key=corrections.get) if corrections else None
    return {
        "status": "complete",
        "output": str(output),
        "frames": clip["frameCount"],
        "fps": round(clip["fps"], 4),
        "bonesAnimated": len(clip["tracks"]),
        "bonesSkipped": clip["meta"]["bonesSkipped"],
        "restAlignment": clip["meta"]["restAlignment"],
        "largestRestCorrectionDeg": {worst: corrections[worst]} if worst else None,
        "rootMotion": clip["meta"]["rootMotion"],
        **({"bake": baked} if baked else {}),
    }


def _run_retarget(args: argparse.Namespace) -> int:
    if not _has_pygltflib("retarget"):
        return 1
    print(json.dumps(_retarget(args.bvh, args.model, args.output, not args.no_align_rest,
                               args.name, args.keep_animations), ensure_ascii=False))
    return 0


MONOCULAR_DEPTH_WARNING = (
    "Monoculaire : la profondeur (mouvement vers ou loin de la camera) est sous-estimee. "
    "Mesure sur COMFI (saut filme de face, contre mocap) : flexion du genou retrouvee a 35-60 %. "
    "Pour le corps entier, preferer 2 cameras ou plus avec --calibration.")


def _check_animate_args(args: argparse.Namespace) -> Path:
    multi = len(args.videos) >= 2
    if multi and args.calibration is None:
        raise ValueError("2 videos ou plus = multivue : --calibration est obligatoire (voir mimo calibrate)")
    if not multi and args.calibration is not None:
        raise ValueError("une seule video = monoculaire : --calibration ne sert pas "
                         "(la retirer, ou donner une video par camera)")
    mono_only = {"--preset": args.preset, "--pipeline": args.pipeline, "--fps": args.fps}
    multi_only = {"--offsets": args.offsets, "--unit-scale": args.unit_scale, "--max-frames": args.max_frames}
    for flag, value in (mono_only if multi else multi_only).items():
        if value is not None:
            raise ValueError(f"{flag} ne s'applique qu'au mode {'monoculaire (1 video)' if multi else 'multivue (2 videos ou plus)'}")
    if args.output.suffix.lower() != ".glb":
        raise ValueError(f"sortie {args.output.name} : extension .glb attendue")
    bvh = args.bvh or args.output.with_suffix(".bvh")
    if bvh.suffix.lower() != ".bvh":
        raise ValueError(f"--bvh {bvh.name} : extension .bvh attendue")
    for video in args.videos:
        if not video.is_file():
            raise FileNotFoundError(f"video introuvable : {video}")
    resolved = [video.resolve() for video in args.videos]
    if len(set(resolved)) != len(resolved):
        raise ValueError("la meme video est donnee plusieurs fois : il faut une video differente par camera")
    if multi and not args.calibration.is_file():
        raise FileNotFoundError(f"calibration introuvable : {args.calibration}")
    from backend.retarget_bvh_to_mixamo import _load_gltf
    gltf = _load_gltf(args.model)
    if not any((node.name or "").startswith("mixamorig") for node in gltf.nodes):
        raise ValueError(f"{args.model.name} : aucun noeud mixamorig:* (modele Mixamo attendu)")
    return bvh


def _run_animate(args: argparse.Namespace) -> int:
    if not _has_pygltflib("animate"):
        return 1
    bvh = _check_animate_args(args)
    if len(args.videos) >= 2:
        from backend.multiview import run_multiview
        report = run_multiview(args.videos, args.calibration, bvh, offsets=args.offsets,
                               unit_scale=args.unit_scale, max_frames=args.max_frames,
                               body=args.body or "auto", hands=args.hands or "auto")
        mode, warnings, bvh_report = "multiview", list(report["warnings"]), report["report"]
    else:
        from backend.process_video import _apply_preset, process_video
        options = argparse.Namespace(preset=args.preset, pipeline=args.pipeline, body=args.body,
                                     hands=args.hands, yolo_model=None, vitpose_config=None,
                                     vitpose_checkpoint=None)
        _apply_preset(options)
        result = process_video(args.videos[0], bvh, args.fps, body=options.body, hands=options.hands,
                               pipeline=options.pipeline, yolo_model_path=options.yolo_model,
                               vitpose_config_path=options.vitpose_config,
                               vitpose_checkpoint_path=options.vitpose_checkpoint)
        mode, bvh_report = "monocular", result["report"]
        warnings = [*result["warnings"], MONOCULAR_DEPTH_WARNING]
    summary = _retarget(bvh, args.model, args.output, name=args.name, keep_animations=args.keep_animations)
    print(json.dumps({**summary, "mode": mode, "videos": [str(v) for v in args.videos],
                      "bvh": str(bvh), "bvhReport": bvh_report, "warnings": warnings}, ensure_ascii=False))
    return 0


def _run_bake_mixamo(args: argparse.Namespace) -> int:
    if not _has_pygltflib("bake-mixamo"):
        return 1
    from backend.bake_mixamo_animation import bake_retarget_into_glb
    if not args.clip.is_file():
        raise FileNotFoundError(f"clip introuvable : {args.clip}")
    clip = json.loads(args.clip.read_text(encoding="utf-8"))
    print(json.dumps(bake_retarget_into_glb(clip, args.model, args.output, args.name, args.keep_animations),
                     ensure_ascii=False))
    return 0


def _run_bvh(args: argparse.Namespace) -> int:
    from backend.process_video import _apply_preset, process_video as process_bvh
    _apply_preset(args)
    result = process_bvh(
        args.input,
        args.output,
        args.fps,
        args.max_bad_frames,
        bvh_mode=args.bvh_mode,
        source=args.source,
        aspect=args.aspect,
        scale=args.scale,
        min_visibility=args.min_visibility,
        flip=(args.flip_x, args.flip_y, args.flip_z),
        recenter=not args.no_recenter,
        inspect=args.inspect,
        body=args.body,
        hands=args.hands,
        hand_size_m=args.hand_size_cm / 100.0,
        pipeline=args.pipeline,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


def _run_vrma(args: argparse.Namespace) -> int:
    from backend.vrm_retargeting import process_mimo_landmarks_to_vrma
    from backend.vrma_serializer import serialize_to_vrma
    from backend.process_video import extract_frames

    print(f"Extraction des repères MediaPipe ({args.pipeline})...")
    frames, fps = extract_frames(args.input, args.pipeline, args.fps)
    print(f"Génération du clip d'animation VRM ({len(frames)} frames @ {fps} FPS)...")
    clip = process_mimo_landmarks_to_vrma(frames, fps=fps, sign_label=args.input.stem)
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    is_binary = args.binary or args.output.suffix.lower() == ".vrma"
    
    if is_binary:
        serialize_to_vrma(clip, args.output)
        format_type = "VRMA binaire (VRMC_vrm_animation)"
    else:
        args.output.write_text(json.dumps(clip, indent=2, ensure_ascii=False), encoding="utf-8")
        format_type = "JSON VRMA"

    print(json.dumps({
        "status": "complete",
        "format": format_type,
        "output": str(args.output),
        "duration": clip["duration"],
        "bonesCount": len(clip["meta"]["bonesAnimated"])
    }, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "bvh":
        return _run_bvh(args)
    if args.command == "vrma":
        return _run_vrma(args)
    if args.command == "bake-vrm":
        from backend.bake_vrm_animation import bake_vrma_to_vrm, inspect_vrma_terminal
        with open(args.vrma, "r", encoding="utf-8") as f:
            inspect_vrma_terminal(json.load(f))
        res = bake_vrma_to_vrm(args.model, args.vrma, args.output)
        print(json.dumps(res, ensure_ascii=False))
        return 0
    if args.command == "inspect-vrm":
        from backend.bake_vrm_animation import inspect_vrma_terminal
        with open(args.vrma, "r", encoding="utf-8") as f:
            inspect_vrma_terminal(json.load(f))
        return 0
    if args.command == "anipose":
        return _run_anipose(args)
    if args.command in ("calibrate", "multiview", "retarget", "bake-mixamo", "animate"):
        run = {"calibrate": _run_calibrate, "multiview": _run_multiview, "retarget": _run_retarget,
               "bake-mixamo": _run_bake_mixamo, "animate": _run_animate}[args.command]
        try:
            return run(args)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            print(f"mimo {args.command} : {exc}", file=sys.stderr)
            return 2
    if args.command == "mixamo":
        from backend.mixamo.process_precise import process_video as process_mixamo
        result = process_mixamo(args.input, args.output, args.fps, args.rig, args.max_bad_frames)
        print(json.dumps({"status": "complete", "output": str(result)}, ensure_ascii=False))
        return 0
    if args.command == "server":
        try:
            from backend.server import run_server
        except ModuleNotFoundError as exc:
            if exc.name != "flask":
                raise
            print("Flask est requis pour le serveur : pip install -e .[server]")
            return 1
        run_server(host=args.host, port=args.port, debug=args.debug)
        return 0
    raise RuntimeError(f"Commande inconnue : {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
