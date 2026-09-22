from __future__ import annotations

from typing import Any, Iterator

import cv2

from backend.mediapipe_compat import frame_timestamp_ms, model_path, numpy_rgb_to_mp_image
from backend.pose_estimator import (
    _BaseOptions,
    _HandLandmarker,
    _HandLandmarkerOptions,
    _HolisticLandmarker,
    _HolisticLandmarkerOptions,
    _RunningMode,
    landmarks_to_array,
)


def extract_holistic_body_precise_hands(video_path: str) -> Iterator[dict[str, Any]]:
    """Body from HolisticLandmarker, hands from the dedicated HandLandmarker,
    in one video pass with shared frame indices -- same pattern as
    extract_pose_and_hands() in fused_extractor.py, but with Holistic
    standing in for the body model instead of the standalone PoseLandmarker.

    Rationale: Holistic is a single model call for the body (cheaper than
    running a separate PoseLandmarker), while still getting the dedicated
    HandLandmarker's higher hand precision instead of Holistic's own
    lower-precision hand output. A third, distinct cost/precision point
    from extract_holistic (cheap, lower hand precision) and
    extract_pose_and_hands (two dedicated models, highest precision, more
    compute).

    Holistic's OWN left_hand_landmarks/right_hand_landmarks are read here
    but discarded -- only its body output is used. This is intentional,
    not wasted work avoided: HolisticLandmarker does not expose an option
    to skip hand tracking internally, so the cost of running it is paid
    either way; this function simply prefers the dedicated model's hand
    result over Holistic's own once both are available.
    """
    if _HolisticLandmarker is None or _HandLandmarker is None:
        raise RuntimeError("MediaPipe Holistic and Hand Landmarker are required.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    holistic_options = _HolisticLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=model_path("holistic_landmarker.task")),
        running_mode=_RunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_face_landmarks_confidence=0.8,
        min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.8,
        min_hand_landmarks_confidence=0.8,
    )
    hand_options = _HandLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=model_path("hand_landmarker.task")),
        running_mode=_RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.8,
    )

    try:
        with _HolisticLandmarker.create_from_options(holistic_options) as holistic, \
             _HandLandmarker.create_from_options(hand_options) as hands:
            frame_index = 0
            while True:
                success, image = cap.read()
                if not success:
                    break
                frame_index += 1
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = numpy_rgb_to_mp_image(image_rgb)
                timestamp_ms = frame_timestamp_ms(cap, frame_index)

                holistic_result = holistic.detect_for_video(mp_image, timestamp_ms)
                hand_result = hands.detect_for_video(mp_image, timestamp_ms)

                body = []
                if getattr(holistic_result, "pose_world_landmarks", None):
                    # Same fix as extract_holistic: HolisticLandmarker's
                    # Python API does not nest by person, no [0] here.
                    body = landmarks_to_array(holistic_result.pose_world_landmarks)

                left, right = [], []
                if hand_result.hand_landmarks:
                    for index, hand_landmarks in enumerate(hand_result.hand_landmarks):
                        points = landmarks_to_array(hand_landmarks)
                        label = ""
                        if hand_result.handedness and index < len(hand_result.handedness):
                            categories = hand_result.handedness[index]
                            if categories:
                                label = (categories[0].category_name or categories[0].display_name or "").lower()
                        if label == "left":
                            left = points
                        elif label == "right":
                            right = points

                yield {
                    "frame": frame_index,
                    "bodyPose": body,
                    "handsL": left,
                    "handsR": right,
                    "width": int(image.shape[1]),
                    "height": int(image.shape[0]),
                }
    finally:
        cap.release()
