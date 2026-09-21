import json
from typing import Iterator, Dict, Any, List

import cv2
import numpy as np

try:
    from mediapipe.tasks.python.core import base_options as base_options_module
    from mediapipe.tasks.python.vision import drawing_utils as mp_drawing
    from mediapipe.tasks.python.vision import hand_landmarker as hand_landmarker_module
    from mediapipe.tasks.python.vision import pose_landmarker as pose_landmarker_module
    from mediapipe.tasks.python.vision import holistic_landmarker as holistic_landmarker_module
    from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module

    _BaseOptions = base_options_module.BaseOptions
    _RunningMode = running_mode_module.VisionTaskRunningMode
    _PoseLandmarker = pose_landmarker_module.PoseLandmarker
    _PoseLandmarkerOptions = pose_landmarker_module.PoseLandmarkerOptions
    _HandLandmarker = hand_landmarker_module.HandLandmarker
    _HandLandmarkerOptions = hand_landmarker_module.HandLandmarkerOptions
    _HolisticLandmarker = holistic_landmarker_module.HolisticLandmarker
    _HolisticLandmarkerOptions = holistic_landmarker_module.HolisticLandmarkerOptions
except Exception:  # pragma: no cover - fallback for environments without MediaPipe runtime
    _BaseOptions = None
    _RunningMode = None
    _PoseLandmarker = None
    _PoseLandmarkerOptions = None
    _HandLandmarker = None
    _HandLandmarkerOptions = None
    _HolisticLandmarker = None
    _HolisticLandmarkerOptions = None
    mp_drawing = None

from backend.mediapipe_compat import model_path, numpy_rgb_to_mp_image, frame_timestamp_ms
from backend.axis_conversion import convert_frame


def landmark_sequence(landmarks):
    if landmarks is None:
        return []
    if hasattr(landmarks, "landmark"):
        return landmarks.landmark
    return landmarks


def landmarks_to_array(landmarks):
    """Raw MediaPipe landmarks -> engine-space landmarks, in one step.
    This is the ONLY place the axis conversion is applied -- every caller
    (body pose, left hand, right hand, from any of the three extractors
    below) already routes through here, so nothing downstream needs to
    know MediaPipe's convention exists.
    """
    out = []
    for lmk in landmark_sequence(landmarks):
        vis = getattr(lmk, "visibility", None)
        if vis is None:
            vis = 1.0
        out.append({
            "x": float(lmk.x),
            "y": float(lmk.y),
            "z": float(lmk.z),
            "visibility": float(vis),
        })
    return convert_frame(out)


def extract_pose(video_path: str) -> Iterator[Dict[str, Any]]:
    if _PoseLandmarker is None:
        raise RuntimeError("MediaPipe is not installed or not available in this environment.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    options = _PoseLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=model_path("pose_landmarker_full.task")),
        running_mode=_RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.8,
    )

    with _PoseLandmarker.create_from_options(options) as detector:
        frame_index = 0
        while True:
            success, image = cap.read()
            if not success:
                break
            frame_index += 1
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = numpy_rgb_to_mp_image(image_rgb)
            timestamp_ms = frame_timestamp_ms(cap, frame_index)
            results = detector.detect_for_video(mp_image, timestamp_ms)
            poses = results.pose_world_landmarks if results.pose_world_landmarks else []
            if poses:
                # PoseLandmarker DOES nest by detected person: poses[0] is
                # the first person's List[Landmark]. Unlike HolisticLandmarker
                # below, this [0] is correct here.
                landmarks = landmarks_to_array(poses[0])
                yield {
                    "frame": int(frame_index),
                    "predictions": landmarks,
                    "width": int(image.shape[1]),
                    "height": int(image.shape[0]),
                }

    cap.release()


def extract_hands(video_path: str) -> Iterator[Dict[str, Any]]:
    if _HandLandmarker is None:
        raise RuntimeError("MediaPipe Hand Landmarker is not installed or not available.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    options = _HandLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=model_path("hand_landmarker.task")),
        running_mode=_RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.8,
    )

    with _HandLandmarker.create_from_options(options) as detector:
        frame_index = 0
        while True:
            success, image = cap.read()
            if not success:
                break
            frame_index += 1
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = numpy_rgb_to_mp_image(image_rgb)
            timestamp_ms = frame_timestamp_ms(cap, frame_index)
            results = detector.detect_for_video(mp_image, timestamp_ms)

            left = []
            right = []
            if results.hand_landmarks:
                for idx, hand_lm in enumerate(results.hand_landmarks):
                    landmark_data = landmarks_to_array(hand_lm)
                    label = ""
                    if results.handedness and idx < len(results.handedness):
                        cats = results.handedness[idx]
                        if cats:
                            label = (cats[0].category_name or cats[0].display_name or "").lower()
                    if label == "left":
                        left = landmark_data
                    elif label == "right":
                        right = landmark_data

            yield {
                "frame": int(frame_index),
                "handsL": left,
                "handsR": right,
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
            }

    cap.release()


def extract_holistic(video_path: str) -> Iterator[Dict[str, Any]]:
    if _HolisticLandmarker is None:
        raise RuntimeError("MediaPipe Holistic model is not installed or not available.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    options = _HolisticLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=model_path("holistic_landmarker.task")),
        running_mode=_RunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_face_landmarks_confidence=0.8,
        min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.8,
        min_hand_landmarks_confidence=0.8,
    )

    with _HolisticLandmarker.create_from_options(options) as detector:
        frame_index = 0
        while True:
            success, image = cap.read()
            if not success:
                break
            frame_index += 1
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = numpy_rgb_to_mp_image(image_rgb)
            timestamp_ms = frame_timestamp_ms(cap, frame_index)
            results = detector.detect_for_video(mp_image, timestamp_ms)

            body = []
            left = []
            right = []
            if getattr(results, "pose_world_landmarks", None):
                # HolisticLandmarker's Python API does NOT nest by person
                # (it only ever tracks one) -- pose_world_landmarks is
                # already List[Landmark], no [0] here (see the bug this
                # fixed: 'Landmark' object is not iterable).
                body = landmarks_to_array(results.pose_world_landmarks)
            if getattr(results, "left_hand_landmarks", None):
                left = landmarks_to_array(results.left_hand_landmarks)
            if getattr(results, "right_hand_landmarks", None):
                right = landmarks_to_array(results.right_hand_landmarks)

            yield {
                "frame": int(frame_index),
                "bodyPose": body,
                "handsL": left,
                "handsR": right,
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
            }

    cap.release()


def save_json(path: str, payload: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2)