import os

import cv2
from mediapipe import Image, ImageFormat


def model_path(name):
    base_dir = os.path.join(os.path.dirname(__file__), "models")
    return os.path.join(base_dir, name)


class FrameClock:
    """Strictly increasing millisecond timestamps for one video.

    MediaPipe VIDEO mode rejects non-increasing timestamps per landmarker, so
    each video (and its landmarkers) gets its own clock.
    """

    def __init__(self, cap):
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30
        self.last = -1

    def __call__(self, frame_index):
        ts = int(round(frame_index * 1000 / max(self.fps, 1)))
        if ts <= self.last:
            ts = self.last + 1
        self.last = ts
        return ts


def numpy_rgb_to_mp_image(image_rgb):
    return Image(image_format=ImageFormat.SRGB, data=image_rgb)
