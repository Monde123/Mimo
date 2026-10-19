import cv2
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.vision import drawing_utils as mp_drawing
from mediapipe.tasks.python.vision import face_landmarker as face_landmarker_module
from mediapipe.tasks.python.vision import hand_landmarker as hand_landmarker_module
from mediapipe.tasks.python.vision import holistic_landmarker as holistic_landmarker_module
from mediapipe.tasks.python.vision import pose_landmarker as pose_landmarker_module
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module



_BaseOptions = base_options_module.BaseOptions
_RunningMode = running_mode_module.VisionTaskRunningMode
_PoseLandmarker = pose_landmarker_module.PoseLandmarker
_PoseLandmarkerOptions = pose_landmarker_module.PoseLandmarkerOptions
_PoseLandmarksConnections = pose_landmarker_module.PoseLandmarksConnections
_HolisticLandmarker = holistic_landmarker_module.HolisticLandmarker
_HolisticLandmarkerOptions = holistic_landmarker_module.HolisticLandmarkerOptions
_HandLandmarker = hand_landmarker_module.HandLandmarker
_HandLandmarkerOptions = hand_landmarker_module.HandLandmarkerOptions
_HandLandmarksConnections = hand_landmarker_module.HandLandmarksConnections
_FaceLandmarksConnections = face_landmarker_module.FaceLandmarksConnections

from backend.mediapipe_compat import FrameClock, model_path, numpy_rgb_to_mp_image


def _landmark_sequence(landmarks):
    if landmarks is None:
        return []
    if hasattr(landmarks, "landmark"):
        return landmarks.landmark
    return landmarks


def add_extra_points(landmark_list):
    left_shoulder = landmark_list[11]
    right_shoulder = landmark_list[12]
    left_hip = landmark_list[23]
    right_hip = landmark_list[24]

    hip = {
        "x": (left_hip["x"] + right_hip["x"]) / 2.0,
        "y": (left_hip["y"] + right_hip["y"]) / 2.0,
        "z": (left_hip["z"] + right_hip["z"]) / 2.0,
        "visibility": (left_hip["visibility"] + right_hip["visibility"]) / 2.0,
    }
    landmark_list.append(hip)

    spine = {
        "x": (left_hip["x"] + right_hip["x"] + right_shoulder["x"] + left_shoulder["x"]) / 4.0,
        "y": (left_hip["y"] + right_hip["y"] + right_shoulder["y"] + left_shoulder["y"]) / 4.0,
        "z": (left_hip["z"] + right_hip["z"] + right_shoulder["z"] + left_shoulder["z"]) / 4.0,
        "visibility": (
            left_hip["visibility"]
            + right_hip["visibility"]
            + right_shoulder["visibility"]
            + left_shoulder["visibility"]
        )
        / 4.0,
    }
    landmark_list.append(spine)

    left_mouth = landmark_list[9]
    right_mouth = landmark_list[10]
    nose = landmark_list[0]
    left_ear = landmark_list[7]
    right_ear = landmark_list[8]
    neck = {
        "x": (left_mouth["x"] + right_mouth["x"] + right_shoulder["x"] + left_shoulder["x"]) / 4.0,
        "y": (left_mouth["y"] + right_mouth["y"] + right_shoulder["y"] + left_shoulder["y"]) / 4.0,
        "z": (left_mouth["z"] + right_mouth["z"] + right_shoulder["z"] + left_shoulder["z"]) / 4.0,
        "visibility": (
            left_mouth["visibility"]
            + right_mouth["visibility"]
            + right_shoulder["visibility"]
            + left_shoulder["visibility"]
        )
        / 4.0,
    }
    landmark_list.append(neck)

    head = {
        "x": (nose["x"] + left_ear["x"] + right_ear["x"]) / 3.0,
        "y": (nose["y"] + left_ear["y"] + right_ear["y"]) / 3.0,
        "z": (nose["z"] + left_ear["z"] + right_ear["z"]) / 3.0,
        "visibility": (
            nose["visibility"] + left_ear["visibility"] + right_ear["visibility"]
        )
        / 3.0,
    }
    landmark_list.append(head)


def landmarks_list_to_array(landmark_list):
    array = []
    for lmk in _landmark_sequence(landmark_list):
        vis = getattr(lmk, "visibility", None)
        if vis is None:
            vis = 1.0
        new_row = {
            "x": lmk.x,
            "y": lmk.y,
            "z": lmk.z,
            "visibility": vis,
        }
        array.append(new_row)
    return array


# Shared name used by the one-pass extractors.
landmarks_to_array = landmarks_list_to_array
def extract_holistic(video_path, debug=False):
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
    face_contours = _FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS
    pose_conn = _PoseLandmarksConnections.POSE_LANDMARKS
    hand_conn = _HandLandmarksConnections.HAND_CONNECTIONS
    face_spec = mp_drawing.DrawingSpec(color=mp_drawing.WHITE_COLOR, thickness=1)
    pose_spec = mp_drawing.DrawingSpec(color=mp_drawing.WHITE_COLOR, thickness=2)
    hand_spec = mp_drawing.DrawingSpec(color=mp_drawing.GREEN_COLOR, thickness=2)

    try:
        with _HolisticLandmarker.create_from_options(options) as holistic:
            clock = FrameClock(cap)
            frame_index = 0
            while True:
                success, image = cap.read()
                if not success:
                    break
                frame_index += 1
                frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = holistic.detect_for_video(numpy_rgb_to_mp_image(image_rgb), clock(frame_index))

                rows, cols, _ = image_rgb.shape
                pose_landmarks = []
                if results.pose_world_landmarks:
                    pose_landmarks = landmarks_list_to_array(results.pose_world_landmarks)
                    add_extra_points(pose_landmarks)
                body_pose = {"predictions": pose_landmarks, "frame": frame, "height": rows, "width": cols}

                hands_array_L = []
                hands_array_R = []
                if results.left_hand_landmarks:
                    hands_array_L = landmarks_list_to_array(results.left_hand_landmarks)
                if results.right_hand_landmarks:
                    hands_array_R = landmarks_list_to_array(results.right_hand_landmarks)
                hands_pose = {"handsR": hands_array_R, "handsL": hands_array_L, "frame": frame}

                yield {"bodyPose": body_pose, "handsPose": hands_pose, "frame": frame}

                if debug:
                    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                    if results.face_landmarks:
                        mp_drawing.draw_landmarks(
                            image_bgr, results.face_landmarks, face_contours,
                            landmark_drawing_spec=None, connection_drawing_spec=face_spec,
                        )
                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(
                            image_bgr, results.pose_landmarks, pose_conn, landmark_drawing_spec=pose_spec,
                        )
                    for hand_landmarks in (results.right_hand_landmarks, results.left_hand_landmarks):
                        if hand_landmarks:
                            mp_drawing.draw_landmarks(
                                image_bgr, hand_landmarks, hand_conn, landmark_drawing_spec=hand_spec,
                            )
                    cv2.imshow("MediaPipe Holistic", cv2.flip(image_bgr, 1))
                    if cv2.waitKey(5) & 0xFF == 27:
                        break
    finally:
        cap.release()
