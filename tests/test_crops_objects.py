"""walkie_vision.crops.objects: the object sampler, the padded crop and its JPEG. No ROS graph."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from walkie_vision.crops.objects import CROP_MAX_SIDE, encode_crop, padded_crop, select_samples
from walkie_vision.providers import Detection

WIDTH, HEIGHT = 64, 48
BOX = (10.0, 8.0, 50.0, 40.0)  # 40 x 32 px
IMAGE = np.random.default_rng(0).integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)


def _detection(track_id=10, score=0.8, box=BOX, mask=None, label="chair") -> Detection:
    return Detection(box=np.array(box, dtype=np.float64), label=label, score=score, track_id=track_id, mask=mask)


# --- the sampler -----------------------------------------------------------


@pytest.mark.parametrize(
    "track_id, counter, sampled",
    [(10, 0, True), (1, 0, False), (9, 0, False), (1, 9, True), (2, 9, False), (7, 13, True), (0, 0, True), (0, 1, False)],
)
def test_a_track_is_eligible_once_in_ten_images(track_id, counter, sampled):
    assert select_samples([_detection(track_id)], counter, WIDTH, HEIGHT) == ([0] if sampled else [])


@pytest.mark.parametrize(
    "box, sampled",
    [
        ((0.0, 0.0, 32.0, 32.0), True),
        ((0.0, 0.0, 31.9, 40.0), False),
        ((0.0, 0.0, 40.0, 31.0), False),
        # A box that hangs off the image counts only its pixels inside it.
        ((-30.0, 0.0, 10.0, 40.0), False),
    ],
)
def test_a_box_needs_32_pixels_on_its_short_side(box, sampled):
    assert select_samples([_detection(box=box)], 0, WIDTH, HEIGHT) == ([0] if sampled else [])


def test_at_most_three_detections_are_sampled_by_score():
    scores = [0.5, 0.9, 0.7, 0.6, 0.8]
    detections = [_detection(10 * (i + 1), score) for i, score in enumerate(scores)]
    assert select_samples(detections, 0, WIDTH, HEIGHT) == [1, 4, 2]


def test_equal_scores_break_by_track_id_then_index():
    detections = [_detection(30, 0.5), _detection(10, 0.5), _detection(20, 0.5), _detection(10, 0.5)]
    assert select_samples(detections, 0, WIDTH, HEIGHT) == [1, 3, 2]


# --- padding and JPEG ------------------------------------------------------


def test_the_crop_grows_15_percent_on_each_side():
    image = np.arange(1000 * 1000 * 3, dtype=np.uint32).reshape(1000, 1000, 3).astype(np.uint8)
    crop = padded_crop(image, np.array([100.0, 200.0, 300.0, 600.0]))
    # 30 px left and right of a 200 px width, 60 px above and below a 400 px height.
    assert crop.shape[:2] == (520, 260)
    np.testing.assert_array_equal(crop, image[140:660, 70:330])


def test_the_crop_is_clipped_to_the_image():
    crop = padded_crop(IMAGE, np.array([0.0, 0.0, 64.0, 48.0]))
    assert crop.shape == IMAGE.shape
    crop = padded_crop(IMAGE, np.array([50.0, 40.0, 64.0, 48.0]))
    np.testing.assert_array_equal(crop, IMAGE[38:48, 47:64])


def test_a_box_outside_the_image_has_no_crop():
    with pytest.raises(ValueError, match="no pixels"):
        padded_crop(IMAGE, np.array([200.0, 200.0, 240.0, 240.0]))


def test_a_jpeg_has_its_header_and_is_reduced_but_never_enlarged():
    small = encode_crop(IMAGE)
    assert small[:2] == b"\xff\xd8"
    assert cv2.imdecode(np.frombuffer(small, np.uint8), cv2.IMREAD_COLOR).shape == IMAGE.shape

    large = np.zeros((800, 1024, 3), dtype=np.uint8)
    decoded = cv2.imdecode(np.frombuffer(encode_crop(large), np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (400, CROP_MAX_SIDE, 3)
