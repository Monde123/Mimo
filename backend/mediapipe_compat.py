import os
import cv2


def model_path(name):
    base_dir = os.path.join(os.path.dirname(__file__), "models")
    return os.path.join(base_dir, name)


def frame_timestamp_ms(cap, frame_index):
    # MediaPipe in VIDEO mode expects timestamps in milliseconds.
    # OpenCV CAP_PROP_POS_MSEC is a simple fallback.
    return int(cap.get(cv2.CAP_PROP_POS_MSEC) or frame_index * 1000 / max(cap.get(cv2.CAP_PROP_FPS) or 30, 1))


def numpy_rgb_to_mp_image(image_rgb):
    try:
        from mediapipe.framework.formats import landmark_pb2
        from mediapipe.python.solutions import drawing_utils
        _ = landmark_pb2, drawing_utils
    except Exception:
        pass

    try:
        from mediapipe import Image
        return Image(image_format=ImageFormat.SRGB, data=image_rgb)
    except Exception:
        return image_rgb


# Compatibility fallback names for MediaPipe tasks.
try:
    from mediapipe import Image as MPImage
    Image = MPImage
except Exception:
    Image = None

try:
    from mediapipe import ImageFormat
    ImageFormat = ImageFormat
except Exception:
    class ImageFormat:
        SRGB = 0
