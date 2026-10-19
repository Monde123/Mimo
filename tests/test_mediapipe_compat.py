from __future__ import annotations

import cv2
import numpy as np

from backend.mediapipe_compat import FrameClock, numpy_rgb_to_mp_image


class _FakeCapture:
    def __init__(self, fps):
        self.fps = fps

    def get(self, prop):
        assert prop == cv2.CAP_PROP_FPS
        return self.fps


def test_each_video_clock_starts_from_zero():
    first = FrameClock(_FakeCapture(30))
    assert [first(i) for i in (1, 2, 3)] == [33, 67, 100]
    second = FrameClock(_FakeCapture(30))
    assert second(1) == 33


def test_clock_stays_strictly_increasing_at_high_fps():
    clock = FrameClock(_FakeCapture(5000))
    stamps = [clock(i) for i in range(1, 20)]
    assert all(b > a for a, b in zip(stamps, stamps[1:]))


def test_rgb_frame_becomes_mediapipe_image():
    image = numpy_rgb_to_mp_image(np.zeros((4, 6, 3), dtype=np.uint8))
    assert (image.height, image.width) == (4, 6)
