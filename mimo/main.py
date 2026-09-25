from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.mixamo.process_precise import process_video as process_mixamo
from backend.process_video import _apply_preset, process_video as process_bvh


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
    parser.add_argument("--pipeline", choices=["holistic", "pose_hands", "hybrid", "parallel"], default=None)
    parser.add_argument("--yolo-model", type=Path, default=None)
    parser.add_argument("--vitpose-config", type=Path, default=None)
    parser.add_argument("--vitpose-checkpoint", type=Path, default=None)
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

    vrma = subparsers.add_parser("vrma", help="extraire une vidéo vers un clip d'animation VRM (VRMA/JSON) pour avatar 3D")
    vrma.add_argument("input", type=Path)
    vrma.add_argument("output", type=Path)
    vrma.add_argument("--fps", type=float, default=None)
    vrma.add_argument("--pipeline", choices=["hybrid", "pose_hands", "holistic"], default="hybrid")

    bake_vrm = subparsers.add_parser("bake-vrm", help="injecter et béké un clip VRMA dans un modèle 3D VRM")
    bake_vrm.add_argument("model", type=Path, help="chemin vers le modèle VRM (ex: avatar.vrm)")
    bake_vrm.add_argument("vrma", type=Path, help="chemin vers le clip VRMA json")
    bake_vrm.add_argument("output", type=Path, nargs="?", default=None, help="modèle béké de sortie")

    inspect_vrm = subparsers.add_parser("inspect-vrm", help="inspection visuelle des poses et des doigts VRMA dans le terminal")
    inspect_vrm.add_argument("vrma", type=Path, help="chemin vers le clip VRMA json")

    subparsers.add_parser("server", help="démarrer le serveur Flask optionnel")
    return parser


def _run_bvh(args: argparse.Namespace) -> int:
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
        yolo_model_path=args.yolo_model,
        vitpose_config_path=args.vitpose_config,
        vitpose_checkpoint_path=args.vitpose_checkpoint,
        parallel_device=args.parallel_device,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


def _run_vrma(args: argparse.Namespace) -> int:
    from backend.vrm_retargeting import process_mimo_landmarks_to_vrma
    from backend.process_video import extract_frames

    print(f"Extraction des repères MediaPipe ({args.pipeline})...")
    frames, fps = extract_frames(args.input, args.pipeline, args.fps)
    print(f"Génération du clip d'animation VRM ({len(frames)} frames @ {fps} FPS)...")
    clip = process_mimo_landmarks_to_vrma(frames, fps=fps, sign_label=args.input.stem)
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(clip, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "status": "complete",
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
    if args.command == "mixamo":
        result = process_mixamo(args.input, args.output, args.fps, args.rig, args.max_bad_frames)
        print(json.dumps({"status": "complete", "output": str(result)}, ensure_ascii=False))
        return 0
    if args.command == "server":
        from backend.server import app

        app.run(host="0.0.0.0", port=5000, debug=True)
        return 0
    raise RuntimeError(f"Commande inconnue : {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
