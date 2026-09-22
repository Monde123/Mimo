"""Optional YOLOv8 + ViTPose body extractor.

The imports for Ultralytics, MMPose, and PyTorch intentionally live inside
``extract_parallel`` so the regular MediaPipe pipelines remain usable without
the considerably heavier optional stack.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import cv2

from backend.mediapipe_compat import frame_timestamp_ms, model_path, numpy_rgb_to_mp_image


class ParallelExtractorError(RuntimeError):
    """Base error for optional parallel-extractor configuration failures."""


class ParallelExtractorDependencyError(ParallelExtractorError):
    """Raised when an optional parallel-extractor dependency is unavailable."""


class ParallelExtractorConfigError(ParallelExtractorError):
    """Raised when a model/config path is missing or invalid."""


def validate_parallel_config(
    yolo_model_path: str | Path | None,
    vitpose_config_path: str | Path | None,
    vitpose_checkpoint_path: str | Path | None,
) -> tuple[Path, Path, Path]:
    """Validate paths without importing torch, MMPose, or Ultralytics."""
    values = (
        ("YOLOv8 model", yolo_model_path),
        ("ViTPose MMPose config", vitpose_config_path),
        ("ViTPose checkpoint", vitpose_checkpoint_path),
    )
    paths: list[Path] = []
    for label, value in values:
        if not value:
            raise ParallelExtractorConfigError(
                f"{label} path is required for pipeline='parallel'."
            )
        path = Path(value).expanduser()
        if not path.is_file():
            raise ParallelExtractorConfigError(f"{label} file does not exist: {path}")
        paths.append(path)
    return paths[0], paths[1], paths[2]


# COCO-17 (ViTPose's common default) -> MediaPipe Pose-33.
_COCO_TO_MP = {
    0: 0,  # nose
    1: 11, 2: 12,  # eyes
    3: 15, 4: 16,  # ears
    5: 11, 6: 12,  # shoulders
    7: 13, 8: 14,  # elbows
    9: 15, 10: 16,  # wrists
    11: 23, 12: 24,  # hips
    13: 25, 14: 26,  # knees
    15: 27, 16: 28,  # ankles
}


def _normalise_pose(keypoints: Any, scores: Any, width: int, height: int) -> list[dict[str, float]]:
    """Convert one COCO keypoint set to exactly 33 bvh_export landmarks."""
    if hasattr(keypoints, "detach"):
        keypoints = keypoints.detach().cpu().numpy()
    if hasattr(scores, "detach"):
        scores = scores.detach().cpu().numpy()
    points = [{"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.0} for _ in range(33)]
    for coco_index, mp_index in _COCO_TO_MP.items():
        try:
            x, y = float(keypoints[coco_index][:2])
            score = float(scores[coco_index]) if scores is not None else 1.0
        except (IndexError, KeyError, TypeError, ValueError):
            continue
        points[mp_index] = {
            "x": x / max(width, 1),
            "y": y / max(height, 1),
            "z": 0.0,
            "visibility": score,
        }

    # Fill duplicate/synthetic MediaPipe landmarks so the contract is always
    # 33 points while retaining confidence information where possible.
    def midpoint(target: int, left: int, right: int) -> None:
        if points[target]["visibility"] == 0.0:
            points[target] = {
                axis: (points[left][axis] + points[right][axis]) / 2.0
                for axis in ("x", "y", "z")
            }
            points[target]["visibility"] = min(points[left]["visibility"], points[right]["visibility"])

    midpoint(7, 11, 11)
    midpoint(8, 12, 12)
    for target, source in ((1, 0), (2, 0), (3, 0), (4, 0), (9, 0), (10, 0),
                           (17, 15), (18, 16), (19, 15), (20, 16), (21, 15),
                           (22, 16), (29, 28), (30, 27), (31, 28), (32, 27)):
        if points[target]["visibility"] == 0.0:
            points[target] = dict(points[source])
    return points


def extract_parallel(
    video_path: str,
    yolo_model_path: str | Path | None = None,
    vitpose_config_path: str | Path | None = None,
    vitpose_checkpoint_path: str | Path | None = None,
    *,
    device: str = "auto",
) -> Iterator[dict[str, Any]]:
    """Yield MediaPipe-shaped frames using YOLOv8 detection and ViTPose.

    Hands are extracted by the dedicated MediaPipe Hand Landmarker. This keeps
    the experimental comparison focused on the body detector/pose estimator.
    """
    yolo_path, config_path, checkpoint_path = validate_parallel_config(
        yolo_model_path, vitpose_config_path, vitpose_checkpoint_path
    )
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ParallelExtractorDependencyError(
            "pipeline='parallel' requires ultralytics (YOLOv8). "
            "Install requirements-parallel.txt."
        ) from exc
    try:
        from mmpose.apis import inference_topdown, init_model
    except ImportError as exc:
        raise ParallelExtractorDependencyError(
            "pipeline='parallel' requires MMPose and its PyTorch dependencies. "
            "Install requirements-parallel.txt."
        ) from exc
    try:
        from backend.pose_estimator import (
            _BaseOptions,
            _HandLandmarker,
            _HandLandmarkerOptions,
            _RunningMode,
            landmarks_to_array,
        )
    except ImportError as exc:
        raise ParallelExtractorDependencyError(
            "pipeline='parallel' requires the MediaPipe Hand Landmarker."
        ) from exc

    detector = YOLO(str(yolo_path))
    mmpose_device = "cpu" if device == "auto" else device
    pose_model = init_model(str(config_path), str(checkpoint_path), device=mmpose_device)
    hand_options = _HandLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=model_path("hand_landmarker.task")),
        running_mode=_RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.8,
    )
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")
    try:
        with _HandLandmarker.create_from_options(hand_options) as hands:
            frame_index = 0
            while True:
                success, image = cap.read()
                if not success:
                    break
                frame_index += 1
                height, width = image.shape[:2]
                detections = detector.predict(image, verbose=False)
                result = detections[0] if detections else None
                boxes = []
                if result is not None and len(result.boxes):
                    xyxy = result.boxes.xyxy.detach().cpu().numpy()
                    classes = result.boxes.cls.detach().cpu().numpy()
                    boxes = xyxy[classes == 0]
                pose_results = inference_topdown(pose_model, image, bboxes=boxes)
                body = []
                if pose_results:
                    sample = pose_results[0]
                    instances = getattr(sample, "pred_instances", sample)
                    keypoints = getattr(instances, "keypoints", None)
                    scores = getattr(instances, "keypoint_scores", None)
                    if keypoints is not None and len(keypoints):
                        body = _normalise_pose(
                            keypoints[0], scores[0] if scores is not None else None,
                            width, height,
                        )
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                hand_result = hands.detect_for_video(
                    numpy_rgb_to_mp_image(rgb),
                    frame_timestamp_ms(cap, frame_index),
                )
                hands_by_side = {"left": [], "right": []}
                for index, hand_landmarks in enumerate(hand_result.hand_landmarks or []):
                    label = ""
                    if hand_result.handedness and index < len(hand_result.handedness):
                        categories = hand_result.handedness[index]
                        if categories:
                            label = (categories[0].category_name
                                     or categories[0].display_name or "").lower()
                    if label in hands_by_side:
                        hands_by_side[label] = landmarks_to_array(hand_landmarks)
                yield {
                    "frame": frame_index,
                    "bodyPose": body,
                    "handsL": hands_by_side["left"],
                    "handsR": hands_by_side["right"],
                    "width": int(width),
                    "height": int(height),
                }
    finally:
        cap.release()


# Descriptive alias for callers that prefer the model names in the function
# name; keep ``extract_parallel`` as the process_video-facing API.
extract_yolov8_vitpose = extract_parallel
