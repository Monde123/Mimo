from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2

from backend.pose_estimator import extract_holistic
from backend.hybrid_extractor import extract_holistic_body_precise_hands
from backend.parallel_extractor import extract_parallel
from backend.quality import trim_unstable_sequence
from backend.smoothing import smooth_landmarks
from backend.bvh_export import describe_frames, export_mediapipe_bvh


def _model_search_roots() -> tuple[Path, ...]:
    """Return Mimo-local parallel model roots."""
    roots: list[Path] = []
    if value := os.environ.get("MIMO_MODEL_DIR"):
        roots.append(Path(value).expanduser())
    roots.append(Path(__file__).resolve().parent / "models" / "parallel")
    return tuple(dict.fromkeys(roots))


def _find_model(filename: str) -> Path | None:
    for root in _model_search_roots():
        candidate = root / filename
        if candidate.is_file():
            return candidate
    return None


def _find_parallel_config() -> Path | None:
    for root in _model_search_roots():
        candidates = (
            root.parent / "vitPose" / "ViTPose_coco_25.py",
            root.parent.parent / "vitPose" / "ViTPose_coco_25.py",
            root / "vitpose" / "ViTPose_coco_25.py",
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    return None


def _apply_preset(args: argparse.Namespace) -> None:
    presets = {
        "sign": ("hybrid", "upper", "on"),
        "full": ("holistic", "full", "off"),
        "parallel": ("parallel", "upper", "on"),
    }
    if args.preset:
        pipeline, body, hands = presets[args.preset]
        args.pipeline = args.pipeline or pipeline
        args.body = args.body or body
        args.hands = args.hands or hands
    args.pipeline = args.pipeline or "holistic"
    args.body = args.body or "auto"
    args.hands = args.hands or "auto"
    if args.pipeline == "parallel":
        args.yolo_model = args.yolo_model or _find_model("yolov8l.pt")
        args.vitpose_checkpoint = args.vitpose_checkpoint or _find_model("vitpose-s-coco_25.pth")
        args.vitpose_config = args.vitpose_config or _find_parallel_config()


def process_video(
    input_path: Path,
    output_path: Path,
    fps: float | None = None,
    max_bad_frames: int = 10,
    bvh_mode: str = "rotations",
    source: str = "auto",
    aspect: float | None = None,
    scale: float = 100.0,
    min_visibility: float = 0.5,
    flip: tuple[bool, bool, bool] = (False, False, False),
    recenter: bool = True,
    inspect: bool = False,
    body: str = "auto",
    hands: str = "auto",
    hand_size_m: float = 0.095,
    pipeline: str = "holistic",
    yolo_model_path: str | Path | None = None,
    vitpose_config_path: str | Path | None = None,
    vitpose_checkpoint_path: str | Path | None = None,
    parallel_device: str = "auto",
) -> dict:
    input_path, output_path = Path(input_path), Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Video introuvable : {input_path}")
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV ne peut pas ouvrir la video (codec non supporte ?) : {input_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    target_fps = float(fps or source_fps)

    if pipeline == "holistic":
        extractor = extract_holistic
    elif pipeline == "hybrid":
        extractor = extract_holistic_body_precise_hands
    elif pipeline == "parallel":
        extractor = lambda path: extract_parallel(
            path, yolo_model_path, vitpose_config_path, vitpose_checkpoint_path,
            device=parallel_device,
        )
    else:
        raise ValueError("pipeline doit etre 'holistic', 'hybrid' ou 'parallel'")
    raw_frames = list(extractor(str(input_path)))
    for frame in raw_frames:
        body_pose = frame.get("bodyPose")
        if isinstance(body_pose, dict):
            frame["bodyPose"] = body_pose.get("predictions", [])[:33]
        elif isinstance(body_pose, list) and len(body_pose) > 33:
            frame["bodyPose"] = body_pose[:33]
        hands_pose = frame.get("handsPose")
        if isinstance(hands_pose, dict):
            frame["handsL"] = hands_pose.get("handsL", [])
            frame["handsR"] = hands_pose.get("handsR", [])
    if inspect:
        print("--- format des frames brutes ---\n" + describe_frames(raw_frames))
    retained, report = trim_unstable_sequence(raw_frames, max_bad_frames=max_bad_frames)
    if not retained:
        raise ValueError("Aucune frame exploitable : la personne n'est jamais detectee de facon stable.")
    cleaned = smooth_landmarks(retained, fps=target_fps, max_gap_frames=min(max_bad_frames, 5))
    if inspect:
        print("--- format apres lissage ---\n" + describe_frames(cleaned))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if aspect is None:
        aspect = (width / height) if width > 0 and height > 0 else 1.0
    bvh_report = export_mediapipe_bvh(
        cleaned, output_path, fps=target_fps, mode=bvh_mode, source=source,
        aspect=aspect, scale=scale, min_visibility=min_visibility, flip=flip,
        recenter=recenter, body=body, hands=hands, hand_size_m=hand_size_m,
        extra={"input": input_path.name, "source_fps": source_fps,
               "video_size": [width, height], "trim": report,
               "pipeline_name": pipeline},
    )
    return {"status": "complete", "output": str(output_path),
            "report": str(output_path.with_suffix(".report.json")),
            "warnings": bvh_report["warnings"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Video 2D -> BVH MediaPipe natif (test) ou clip JSON Mixamo.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--max-bad-frames", type=int, default=10)
    parser.add_argument("--bvh-mode", choices=["rotations", "positions"], default="rotations",
                        help="positions = aucun calcul de rotation (test le plus pur)")
    parser.add_argument("--source", choices=["auto", "world", "normalized"], default="auto",
                        help="type de coordonnees stockees dans bodyPose")
    parser.add_argument("--aspect", type=float, default=None,
                        help="largeur/hauteur de la video (auto par defaut ; utile si portrait/rotation)")
    parser.add_argument("--scale", type=float, default=100.0, help="multiplicateur d'unites (100 : metres -> cm)")
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--flip-x", action="store_true", help="inverse X (video en miroir / selfie)")
    parser.add_argument("--flip-y", action="store_true")
    parser.add_argument("--flip-z", action="store_true")
    parser.add_argument("--no-recenter", action="store_true", help="garde la position absolue (pas de recentrage/sol)")
    parser.add_argument("--preset", choices=["sign", "full", "parallel"], default=None,
                             help="raccourci : sign=hybrid upper mains ; full=corps complet ; "
                                  "parallel=YOLOv8 + ViTPose")
    parser.add_argument("--body", choices=["auto", "full", "upper"], default=None,
                             help="upper = haut du corps seul (racine = epaules, pas de jambes). "
                             "Conseille si les jambes ne sont pas visibles")
    parser.add_argument("--hands", choices=["auto", "on", "off"], default=None,
                             help="utilise les 21 points de chaque main (handsL/handsR)")
    parser.add_argument("--hand-size-cm", type=float, default=9.5,
                        help="longueur poignet->milieu de la paume, sert a l'echelle des mains si source=world")
    parser.add_argument("--inspect", action="store_true", help="affiche le format reel des frames (debogage)")
    parser.add_argument("--pipeline", choices=["holistic", "hybrid", "parallel"], default=None,
                        help="holistic = corps et mains Holistic ; hybrid = corps Holistic + mains dediees ; "
                             "parallel = YOLOv8 + ViTPose (mains MediaPipe)")
    parser.add_argument("--yolo-model", type=Path, help="chemin du checkpoint YOLOv8")
    parser.add_argument("--vitpose-config", type=Path, help="configuration MMPose correspondant au checkpoint ViTPose")
    parser.add_argument("--vitpose-checkpoint", type=Path, help="checkpoint ViTPose")
    parser.add_argument("--parallel-device", default="auto", help="device MMPose (auto, cpu, cuda:0, ...)")
    args = parser.parse_args()
    _apply_preset(args)
    result = process_video(
        args.input, args.output, args.fps, args.max_bad_frames,
        bvh_mode=args.bvh_mode, source=args.source,
        aspect=args.aspect, scale=args.scale, min_visibility=args.min_visibility,
        flip=(args.flip_x, args.flip_y, args.flip_z), recenter=not args.no_recenter,
        inspect=args.inspect, body=args.body, hands=args.hands,
        hand_size_m=args.hand_size_cm / 100.0,
        pipeline=args.pipeline,
        yolo_model_path=args.yolo_model,
        vitpose_config_path=args.vitpose_config,
        vitpose_checkpoint_path=args.vitpose_checkpoint,
        parallel_device=args.parallel_device,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()