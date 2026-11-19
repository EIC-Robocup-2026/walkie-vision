"""The pure parts of the bounded services: the job, the slot, the class gate and the checks.

The newest input and the wake are tested in tests/test_camera_mailboxes.py,
and the services themselves through the node in tests/test_node.py.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
from builtin_interfaces.msg import Time
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from walkie_vision_msgs.srv import DetectObjects, EmbedCrop, EmbedText

from walkie_vision import convert, requests
from walkie_vision.convert import box_to_bbox
from walkie_vision.fakes import FakeEmbedder
from walkie_vision.geometry import frame_laplacian, view_quality
from walkie_vision.providers import CLIP_VIT_B16

FRAME = "camera_optical"
WIDTH, HEIGHT = 64, 48
STAMP_NS = 10_000_000_000
LONG_S = 30.0


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


def _job(deadline_s: float = LONG_S) -> requests.ServiceJob:
    return requests.ServiceJob(EmbedText.Request(), time.monotonic() + deadline_s, convert.embed_text_failure)


# --- the job ---------------------------------------------------------------


def test_a_job_answered_in_time_returns_the_answer():
    job = _job()
    answer = EmbedText.Response(success=True)
    assert job.finish(answer)
    assert job.wait() is answer
    assert not job.given_up


def test_a_job_keeps_its_first_answer():
    job = _job()
    first = EmbedText.Response(success=True)
    assert job.finish(first)
    assert not job.finish(EmbedText.Response(message="later"))
    assert job.wait() is first


def test_a_job_past_its_deadline_takes_no_late_answer():
    job = _job(deadline_s=0.0)
    assert job.wait() is None
    assert job.given_up
    assert job.remaining() <= 0
    assert not job.finish(EmbedText.Response(success=True))
    assert job.wait() is None


def test_a_job_refuses_an_answer_after_its_deadline_that_reaches_the_lock_first():
    now = [100.0]
    job = requests.ServiceJob(EmbedText.Request(texts=["cup"]), 100.1, convert.embed_text_failure, clock=lambda: now[0])
    # The worker finishes late, before the waiting callback wakes and gives up.
    now[0] = 100.2
    assert not job.finish(EmbedText.Response(success=True))
    assert job.wait() is None
    assert job.given_up


def test_a_job_on_the_monotonic_clock_refuses_a_late_answer_before_its_caller_gives_up():
    job = _job(deadline_s=-0.1)
    assert not job.finish(EmbedText.Response(success=True))
    assert job.wait() is None


@pytest.mark.parametrize(("finished_at", "taken"), [(100.0999, True), (100.1, False)])
def test_a_job_takes_an_answer_only_before_its_deadline(finished_at, taken):
    now = [100.0]
    job = requests.ServiceJob(EmbedText.Request(texts=["cup"]), 100.1, convert.embed_text_failure, clock=lambda: now[0])
    answer = EmbedText.Response(success=True)
    now[0] = finished_at
    assert job.finish(answer) is taken
    # The caller may read the answer later. Only the time of finish() counts.
    now[0] = 100.2
    assert job.wait() is (answer if taken else None)


# --- the slot --------------------------------------------------------------


def test_the_slot_holds_one_job_and_wakes_the_worker():
    wakes = []
    slot = requests.JobSlot(lambda: wakes.append(1))
    first, second = _job(), _job()
    assert slot.admit(first)
    assert wakes == [1]
    assert not slot.admit(second)
    assert wakes == [1]
    assert slot.claim() is first
    assert slot.claim() is None


def test_a_claimed_job_holds_the_slot_until_its_release():
    slot = requests.JobSlot(lambda: None)
    job = _job()
    assert slot.admit(job)
    assert slot.claim() is job
    # The callback gave up, but the model still runs.
    slot.withdraw(job)
    assert not slot.admit(_job())
    slot.release(job)
    assert slot.admit(_job())


def test_withdrawing_an_unclaimed_job_frees_the_slot():
    slot = requests.JobSlot(lambda: None)
    job = _job()
    assert slot.admit(job)
    slot.withdraw(job)
    assert slot.claim() is None
    assert slot.admit(_job())


def test_an_old_job_cannot_free_the_slot_of_a_new_one():
    slot = requests.JobSlot(lambda: None)
    old, new = _job(), _job()
    assert slot.admit(old)
    slot.release(old)
    assert slot.admit(new)
    slot.release(old)
    slot.withdraw(old)
    assert not slot.admit(_job())


def test_closing_the_slot_answers_its_job_and_refuses_new_ones():
    slot = requests.JobSlot(lambda: None)
    job = _job()
    assert slot.admit(job)
    assert slot.claim() is job
    slot.close()
    assert slot.closed
    answer = job.wait()
    assert not answer.success and answer.message.startswith(requests.MODEL_UNAVAILABLE)
    assert not slot.admit(_job())
    # The worker's answer after that is the late one.
    assert not job.finish(EmbedText.Response(success=True))


def test_closing_the_slot_after_the_deadline_leaves_its_caller_the_timeout():
    now = [100.0]
    slot = requests.JobSlot(lambda: None)
    job = requests.ServiceJob(EmbedText.Request(texts=["cup"]), 100.1, convert.embed_text_failure, clock=lambda: now[0])
    assert slot.admit(job)
    assert slot.claim() is job
    now[0] = 100.2
    slot.close()
    assert slot.closed
    # The deadline has passed, so the callback answers TIMEOUT, as it would without the shutdown.
    assert job.wait() is None
    assert not slot.admit(_job())


# --- the class gate --------------------------------------------------------


def test_the_class_gate_runs_from_the_end_of_a_preparation():
    now = [100.0]
    gate = requests.ClassPrepareGate(2.0, clock=lambda: now[0])
    assert gate.open()
    now[0] = 101.5  # the preparation took 1.5 s
    gate.mark()
    assert not gate.open()
    now[0] = 103.4
    assert not gate.open()
    now[0] = 103.5
    assert gate.open()


# --- the checks ------------------------------------------------------------


def _detect_request(**fields) -> DetectObjects.Request:
    request = DetectObjects.Request(classes=["cup"])
    for name, value in fields.items():
        setattr(request, name, value)
    return request


def test_a_detect_objects_request_has_the_defaults_of_its_type():
    request = DetectObjects.Request(classes=["cup"])
    assert request.max_image_age_s == pytest.approx(0.3) and request.timeout_s == pytest.approx(4.0)
    assert requests.detect_objects_problem(request) is None


@pytest.mark.parametrize(
    ("fields", "reason"),
    [
        ({"classes": []}, "empty"),
        ({"classes": [f"c{i}" for i in range(33)]}, "at most 32"),
        ({"classes": ["cup", ""]}, "class 1 is empty"),
        ({"classes": [" cup"]}, "class 0 is empty or has spaces"),
        ({"classes": ["   "]}, "class 0 is empty or has spaces"),
        ({"classes": ["x" * 81]}, "more than 80"),
        ({"classes": ["cup", "cup"]}, "duplicates"),
        ({"max_image_age_s": 0.0}, "max_image_age_s"),
        ({"max_image_age_s": -0.1}, "max_image_age_s"),
        ({"max_image_age_s": 1.5}, "max_image_age_s"),
        ({"max_image_age_s": float("nan")}, "max_image_age_s"),
        ({"max_image_age_s": float("inf")}, "max_image_age_s"),
        ({"timeout_s": 0.0}, "timeout_s"),
        ({"timeout_s": 5.5}, "timeout_s"),
        ({"timeout_s": float("nan")}, "timeout_s"),
    ],
)
def test_a_bad_detect_objects_request_is_refused(fields, reason):
    assert reason in requests.detect_objects_problem(_detect_request(**fields))


def test_a_detect_objects_request_at_the_caps_is_valid():
    request = _detect_request(
        classes=[f"{i}".ljust(80, "x") for i in range(32)], max_image_age_s=1.0, timeout_s=5.0
    )
    assert requests.detect_objects_problem(request) is None


def test_a_detect_objects_refusal_does_not_repeat_a_class():
    assert "secret" not in requests.detect_objects_problem(_detect_request(classes=["secret" * 20]))


@pytest.mark.parametrize(
    ("age_ns", "current"),
    [
        (0, True),
        (300_000_000, True),
        (300_000_001, False),
        (-300_000_000, True),  # /clock can lag an image
        (-300_000_001, False),  # a stamp from before a sim restart
    ],
)
def test_the_image_age_counts_both_ways(age_ns, current):
    problem = requests.image_age_problem(_image(STAMP_NS), STAMP_NS + age_ns, 0.3)
    assert (problem is None) is current


@pytest.mark.parametrize(
    ("texts", "reason"),
    [
        ([], "empty"),
        (["mug"] * (requests.MAX_TEXTS + 1), "at most 16"),
        (["mug", "   "], "text 1 is blank"),
        (["mug", ""], "text 1 is blank"),
        (["x" * (requests.MAX_TEXT_CHARS + 1)], "more than 512"),
    ],
)
def test_bad_texts_are_refused_without_repeating_them(texts, reason):
    problem = requests.embed_text_problem(texts)
    assert reason in problem
    assert "xxx" not in problem and "mug" not in problem


def test_texts_at_the_caps_are_valid():
    assert requests.embed_text_problem(["x" * requests.MAX_TEXT_CHARS] * requests.MAX_TEXTS) is None


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
    assert reason in requests.embed_crop_problem(_crop_request(**fields))


@pytest.mark.parametrize("timeout_s", [0.001, 1.0, 5.0])
def test_a_good_crop_request_is_valid(timeout_s):
    assert requests.embed_crop_problem(_crop_request(timeout_s=timeout_s)) is None


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
    problem = requests.embed_crop_problem(request)
    assert problem is not None and "bbox" in problem and "not finite" in problem


def test_a_crop_request_whose_padded_crop_overflows_is_refused():
    # The corners are 0 and 1.79e308. The crop's padding adds 15% of the width beyond them.
    request = _crop_request(bbox=_bad_bbox(x=0.895e308, size_x=1.79e308))
    assert np.all(np.isfinite(convert.bbox_to_box(request.bbox)))
    problem = requests.embed_crop_problem(request)
    assert problem is not None and "bbox" in problem and "not finite" in problem


def test_a_crop_request_with_a_huge_finite_box_is_valid_and_crops_the_whole_image():
    request = _crop_request(bbox=_bad_bbox(x=0.0, y=0.0, size_x=1e300, size_y=1e300))
    assert requests.embed_crop_problem(request) is None
    image = np.random.default_rng(3).integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    embedder = FakeEmbedder()
    response = requests.embed_crop_answer(
        embedder, image, Header(frame_id=FRAME), convert.bbox_to_box(request.bbox), None, 0.2, 3.0
    )
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
    problem = requests.embed_crop_problem(_crop_request(header=Header(stamp=stamp, frame_id=FRAME)))
    assert problem is not None
    assert problem.startswith("header.stamp must be a valid time after zero")
    assert f"{stamp.sec} s and {stamp.nanosec} ns" in problem


@pytest.mark.parametrize("stamp", [Time(sec=0, nanosec=1), Time(sec=0, nanosec=999_999_999)])
def test_a_crop_request_stamp_may_have_nanosec_up_to_just_under_a_second(stamp):
    assert requests.embed_crop_problem(_crop_request(header=Header(stamp=stamp, frame_id=FRAME))) is None


def test_the_current_image_answers_a_request_within_every_bound():
    # 5 ms from the image in both directions, an image 0.3 s old, and 5 ms ahead of the clock.
    gap = requests.MAX_STAMP_GAP_NS
    assert requests.crop_image_problem(_crop_request(STAMP_NS + gap), _image(), STAMP_NS + gap) is None
    assert requests.crop_image_problem(_crop_request(STAMP_NS - gap), _image(), STAMP_NS) is None
    assert requests.crop_image_problem(_crop_request(), _image(), STAMP_NS + requests.MAX_IMAGE_AGE_NS) is None
    assert requests.crop_image_problem(_crop_request(), _image(), STAMP_NS - requests.MAX_FUTURE_NS) is None


@pytest.mark.parametrize(
    ("request_ns", "image", "now_ns", "reason"),
    [
        (STAMP_NS + requests.MAX_STAMP_GAP_NS + 1, _image(), STAMP_NS + 10_000_000, "from the current image"),
        (STAMP_NS - requests.MAX_STAMP_GAP_NS - 1, _image(), STAMP_NS, "from the current image"),
        (STAMP_NS, _image(frame="other"), STAMP_NS, "frame"),
        (STAMP_NS, _image(width=WIDTH + 2), STAMP_NS, "image is"),
        (STAMP_NS, _image(height=HEIGHT + 2), STAMP_NS, "image is"),
        (STAMP_NS, _image(), STAMP_NS + requests.MAX_IMAGE_AGE_NS + 1, "old"),
        (STAMP_NS, _image(), STAMP_NS - requests.MAX_FUTURE_NS - 1, "ahead of the clock"),
    ],
)
def test_the_current_image_refuses_a_request_past_any_bound(request_ns, image, now_ns, reason):
    assert reason in requests.crop_image_problem(_crop_request(request_ns), image, now_ns)


# --- the answers -----------------------------------------------------------


def test_a_failure_message_is_bounded():
    long = "x" * 10_000
    assert len(convert.embed_text_failure(requests.TIMEOUT, long).message) == convert.MAX_MESSAGE_CHARS
    crop_failure = convert.embed_crop_failure(requests.TIMEOUT, long)
    assert len(crop_failure.message) == convert.MAX_MESSAGE_CHARS
    assert crop_failure.error_code == requests.TIMEOUT
    assert not crop_failure.success and len(crop_failure.embedding) == 0 and len(crop_failure.crop.data) == 0


def test_an_embedder_failure_message_is_bounded():
    class Broken(FakeEmbedder):
        def embed_texts(self, texts):
            raise RuntimeError("x" * 10_000)

    response = convert.fill_embed_text_response(EmbedText.Response(), Broken(), ["mug"])
    assert not response.success and len(response.message) == convert.MAX_MESSAGE_CHARS


def test_a_crop_answer_refuses_clip_with_the_wrong_length():
    with pytest.raises(ValueError, match="512"):
        _fill_crop(FakeEmbedder(model_name=CLIP_VIT_B16, dim=8), np.ones(8, dtype=np.float32))


def _fill_crop(embedder, vector, quality: float = 0.5) -> EmbedCrop.Response:
    return convert.fill_embed_crop_response(EmbedCrop.Response(), Header(), b"\xff\xd8", embedder, vector, quality)


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
    response = convert.fill_embed_crop_response(
        EmbedCrop.Response(), header, b"\xff\xd8", FakeEmbedder(dim=8), np.ones(8), 0.5
    )
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
    assert requests.rectangle_quality(image, box, None, 0.2, 3.0) == pytest.approx(expected)


def test_a_wrong_shape_from_the_embedder_raises():
    class TwoRows(FakeEmbedder):
        def embed_images(self, images):
            return np.ones((2, self.dim))

    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="shape"):
        requests.embed_crop_answer(TwoRows(), image, Header(), np.array([10.0, 10.0, 30.0, 30.0]), None, 0.2, 3.0)
