from __future__ import annotations

from typing import Any, Iterator

import cv2

from backend.mediapipe_compat import frame_timestamp_ms, model_path, numpy_rgb_to_mp_image
from backend.pose_estimator import (
    _BaseOptions,
    _HandLandmarker,
    _HandLandmarkerOptions,
    _PoseLandmarker,
    _PoseLandmarkerOptions,
    _RunningMode,
    landmarks_to_array,
)


def extract_pose_and_hands(video_path: str) -> Iterator[dict[str, Any]]:
    """Extract pose and hands in one video pass with shared frame indices.

    Pose and hand models are created once per video. This avoids the previous
    two-pass alignment problem and preserves frames where only hands are visible.
    """
    if _PoseLandmarker is None or _HandLandmarker is None:
        raise RuntimeError("MediaPipe Pose and Hand Landmarker are required.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    pose_options = _PoseLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=model_path("pose_landmarker_full.task")),
        running_mode=_RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.8,
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
        with _PoseLandmarker.create_from_options(pose_options) as pose, _HandLandmarker.create_from_options(hand_options) as hands:
            frame_index = 0
            while True:
                success, image = cap.read()
                if not success:
                    break
                frame_index += 1
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = numpy_rgb_to_mp_image(image_rgb)
                timestamp_ms = frame_timestamp_ms(cap, frame_index)
                pose_result = pose.detect_for_video(mp_image, timestamp_ms)
                hand_result = hands.detect_for_video(mp_image, timestamp_ms)

                body = []
                if pose_result.pose_world_landmarks:
                    body = landmarks_to_array(pose_result.pose_world_landmarks[0])
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
