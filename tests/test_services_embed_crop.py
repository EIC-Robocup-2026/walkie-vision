"""walkie_vision.services.embed_crop: the request checks, the current image's check and the answers. No ROS graph."""

from __future__ import annotations

import math

import numpy as np
import pytest
from builtin_interfaces.msg import Time
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from walkie_vision_msgs.srv import EmbedCrop

from walkie_vision.fakes import FakeEmbedder
from walkie_vision.geometry import frame_laplacian, view_quality
from walkie_vision.messages import bbox_to_box, box_to_bbox
from walkie_vision.providers import CLIP_VIT_B16
from walkie_vision.services.embed_crop import (
    MAX_FUTURE_NS,
    MAX_IMAGE_AGE_NS,
    MAX_STAMP_GAP_NS,
    crop_image_problem,
    embed_crop_answer,
    embed_crop_problem,
    fill_embed_crop_response,
    rectangle_quality,
)

FRAME = "camera_optical"
WIDTH, HEIGHT = 64, 48
STAMP_NS = 10_000_000_000


def _stamp(ns: int) -> Time:
    return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


def _crop_request(ns: int = STAMP_NS, **fields) -> EmbedCrop.Request:
    request = EmbedCrop.Request(
        header=Header(stamp=_stamp(ns), frame_id=FRAME),
        image_width=WIDTH,
        image_height=HEIGHT,
        bbox=box_to_bbox((10.0, 10.0, 30.0, 30.0)),
        timeout_s=1.0,
    )
    for name, value in fields.items():
        setattr(request, name, value)
    return request


def _image(ns: int = STAMP_NS, frame: str = FRAME, width: int = WIDTH, height: int = HEIGHT) -> Image:
    return Image(header=Header(stamp=_stamp(ns), frame_id=frame), width=width, height=height)


# --- the checks ------------------------------------------------------------


def _bad_bbox(**fields):
    bbox = box_to_bbox((10.0, 10.0, 30.0, 30.0))
    for name, value in fields.items():
        if name in ("x", "y", "theta"):
            target = bbox.center if name == "theta" else bbox.center.position
            setattr(target, name, value)
        else:
            setattr(bbox, name, value)
    return bbox


@pytest.mark.parametrize(
    ("fields", "reason"),
    [
        ({"header": Header(stamp=_stamp(STAMP_NS), frame_id="")}, "frame_id"),
        ({"header": Header(frame_id=FRAME)}, "stamp"),
        ({"image_width": 0}, "image size"),
        ({"image_height": 0}, "image size"),
        ({"timeout_s": 0.0}, "timeout_s"),
        ({"timeout_s": -1.0}, "timeout_s"),
        ({"timeout_s": 5.5}, "timeout_s"),
        ({"timeout_s": math.nan}, "timeout_s"),
        ({"timeout_s": math.inf}, "timeout_s"),
        ({"bbox": _bad_bbox(x=math.nan)}, "not finite"),
        ({"bbox": _bad_bbox(size_y=math.inf)}, "not finite"),
        ({"bbox": _bad_bbox(theta=0.1)}, "theta"),
        ({"bbox": _bad_bbox(size_x=0.0)}, "positive area"),
        ({"bbox": _bad_bbox(size_y=-1.0)}, "positive area"),
        ({"bbox": _bad_bbox(x=1000.0)}, "no pixels"),
    ],
)
def test_a_bad_crop_request_is_refused(fields, reason):
    assert reason in embed_crop_problem(_crop_request(**fields))


@pytest.mark.parametrize("timeout_s", [0.001, 1.0, 5.0])
def test_a_good_crop_request_is_valid(timeout_s):
    assert embed_crop_problem(_crop_request(timeout_s=timeout_s)) is None


@pytest.mark.parametrize(
    "fields",
    [
        {"x": 1.7e308, "size_x": 1.7e308},
        {"x": -1.7e308, "size_x": 1.7e308},
        {"y": 1.7e308, "size_y": 1.7e308},
    ],
    ids=["right", "left", "bottom"],
)
def test_a_crop_request_whose_box_corner_overflows_is_refused(fields):
    # Every field is finite. Half the size added to the centre is not.
    request = _crop_request(bbox=_bad_bbox(**fields))
    assert all(math.isfinite(v) for v in fields.values())
    problem = embed_crop_problem(request)
    assert problem is not None and "bbox" in problem and "not finite" in problem


def test_a_crop_request_whose_padded_crop_overflows_is_refused():
    # The corners are 0 and 1.79e308. The crop's padding adds 15% of the width beyond them.
    request = _crop_request(bbox=_bad_bbox(x=0.895e308, size_x=1.79e308))
    assert np.all(np.isfinite(bbox_to_box(request.bbox)))
    problem = embed_crop_problem(request)
    assert problem is not None and "bbox" in problem and "not finite" in problem


def test_a_crop_request_with_a_huge_finite_box_is_valid_and_crops_the_whole_image():
    request = _crop_request(bbox=_bad_bbox(x=0.0, y=0.0, size_x=1e300, size_y=1e300))
    assert embed_crop_problem(request) is None
    image = np.random.default_rng(3).integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    embedder = FakeEmbedder()
    response = embed_crop_answer(embedder, image, Header(frame_id=FRAME), bbox_to_box(request.bbox), None, 0.2, 3.0)
    np.testing.assert_allclose(response.embedding, embedder.embed_images([image])[0], rtol=1e-6)
    assert 0.0 <= response.view_quality <= 1.0


@pytest.mark.parametrize(
    "stamp",
    [
        Time(sec=0, nanosec=1_000_000_000),
        Time(sec=1, nanosec=1_000_000_000),
        Time(sec=10, nanosec=4_294_967_295),
        Time(sec=-1, nanosec=0),
    ],
    ids=["one second as nanosec", "nanosec at one second", "largest nanosec", "before zero"],
)
def test_a_crop_request_with_a_stamp_that_is_not_a_valid_time_is_refused(stamp):
    problem = embed_crop_problem(_crop_request(header=Header(stamp=stamp, frame_id=FRAME)))
    assert problem is not None
    assert problem.startswith("header.stamp must be a valid time after zero")
    assert f"{stamp.sec} s and {stamp.nanosec} ns" in problem


@pytest.mark.parametrize("stamp", [Time(sec=0, nanosec=1), Time(sec=0, nanosec=999_999_999)])
def test_a_crop_request_stamp_may_have_nanosec_up_to_just_under_a_second(stamp):
    assert embed_crop_problem(_crop_request(header=Header(stamp=stamp, frame_id=FRAME))) is None


def test_the_current_image_answers_a_request_within_every_bound():
    # 5 ms from the image in both directions, an image 0.3 s old, and 5 ms ahead of the clock.
    gap = MAX_STAMP_GAP_NS
    assert crop_image_problem(_crop_request(STAMP_NS + gap), _image(), STAMP_NS + gap) is None
    assert crop_image_problem(_crop_request(STAMP_NS - gap), _image(), STAMP_NS) is None
    assert crop_image_problem(_crop_request(), _image(), STAMP_NS + MAX_IMAGE_AGE_NS) is None
    assert crop_image_problem(_crop_request(), _image(), STAMP_NS - MAX_FUTURE_NS) is None


@pytest.mark.parametrize(
    ("request_ns", "image", "now_ns", "reason"),
    [
        (STAMP_NS + MAX_STAMP_GAP_NS + 1, _image(), STAMP_NS + 10_000_000, "from the current image"),
        (STAMP_NS - MAX_STAMP_GAP_NS - 1, _image(), STAMP_NS, "from the current image"),
        (STAMP_NS, _image(frame="other"), STAMP_NS, "frame"),
        (STAMP_NS, _image(width=WIDTH + 2), STAMP_NS, "image is"),
        (STAMP_NS, _image(height=HEIGHT + 2), STAMP_NS, "image is"),
        (STAMP_NS, _image(), STAMP_NS + MAX_IMAGE_AGE_NS + 1, "old"),
        (STAMP_NS, _image(), STAMP_NS - MAX_FUTURE_NS - 1, "ahead of the clock"),
    ],
)
def test_the_current_image_refuses_a_request_past_any_bound(request_ns, image, now_ns, reason):
    assert reason in crop_image_problem(_crop_request(request_ns), image, now_ns)


# --- the answers -----------------------------------------------------------


def test_a_crop_answer_refuses_clip_with_the_wrong_length():
    with pytest.raises(ValueError, match="512"):
        _fill_crop(FakeEmbedder(model_name=CLIP_VIT_B16, dim=8), np.ones(8, dtype=np.float32))


def _fill_crop(embedder, vector, quality: float = 0.5) -> EmbedCrop.Response:
    return fill_embed_crop_response(EmbedCrop.Response(), Header(), b"\xff\xd8", embedder, vector, quality)


@pytest.mark.parametrize(
    ("vector", "quality", "match"),
    [
        (np.zeros(8), 0.5, "zero"),
        (np.full(8, math.nan), 0.5, "not finite"),
        (np.ones(4), 0.5, "expected 8"),
        (np.ones(8), 1.5, "view_quality"),
        (np.ones(8), math.nan, "view_quality"),
    ],
)
def test_a_bad_crop_answer_raises(vector, quality, match):
    with pytest.raises(ValueError, match=match):
        _fill_crop(FakeEmbedder(dim=8), vector, quality)


def test_a_good_crop_answer_carries_the_source_header_and_a_unit_vector():
    header = Header(stamp=_stamp(STAMP_NS), frame_id=FRAME)
    response = fill_embed_crop_response(EmbedCrop.Response(), header, b"\xff\xd8", FakeEmbedder(dim=8), np.ones(8), 0.5)
    assert response.success and response.error_code == "" and response.message == ""
    assert response.header == header and response.crop.header == header and response.crop.format == "jpeg"
    assert np.linalg.norm(np.asarray(response.embedding)) == pytest.approx(1.0)
    assert response.view_quality == 0.5


def test_a_rectangle_scores_like_a_mask_of_its_pixels():
    image = np.random.default_rng(0).integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    box = np.array([10.5, 10.0, 30.0, 30.0])
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[10:30, 10:30] = True
    expected = view_quality(mask, box, frame_laplacian(image), 0.0)
    assert rectangle_quality(image, box, None, 0.2, 3.0) == pytest.approx(expected)


def test_a_wrong_shape_from_the_embedder_raises():
    class TwoRows(FakeEmbedder):
        def embed_images(self, images):
            return np.ones((2, self.dim))

    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="shape"):
        embed_crop_answer(TwoRows(), image, Header(), np.array([10.0, 10.0, 30.0, 30.0]), None, 0.2, 3.0)
