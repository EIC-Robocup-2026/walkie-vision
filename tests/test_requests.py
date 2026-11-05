"""The pure parts of the bounded services: the job, the slot, the newest input, the checks and the wake.

The services themselves are tested through the node in tests/test_node.py.
"""

from __future__ import annotations

import gc
import math
import threading
import time
import weakref

import numpy as np
import pytest
from builtin_interfaces.msg import Time
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from walkie_vision_msgs.srv import DetectObjects, EmbedCrop, EmbedText

from walkie_vision import convert, requests
from walkie_vision.convert import box_to_bbox
from walkie_vision.depth import RgbdBuffer, frame_laplacian, view_quality
from walkie_vision.fakes import FakeEmbedder
from walkie_vision.node import NewestFrame
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


# --- the newest input ------------------------------------------------------


class _Msg:
    """Unlike a generated message, this can have a weak reference."""


def test_the_current_image_keeps_only_the_newest_of_each():
    current = requests.CurrentImage()
    assert current.read() == (None, None)
    old_colour, old_depth = _Msg(), _Msg()
    old_refs = weakref.ref(old_colour), weakref.ref(old_depth)
    current.put_colour(old_colour)
    current.put_depth(old_depth)
    assert current.read() == (old_colour, old_depth)
    new_colour, new_depth = _Msg(), _Msg()
    current.put_colour(new_colour)
    current.put_depth(new_depth)
    assert current.read() == (new_colour, new_depth)
    del old_colour, old_depth
    gc.collect()
    assert all(ref() is None for ref in old_refs), "an older image is still held"


def test_the_current_image_keeps_the_newest_pair_with_matching_stamps():
    # A still clock: every image arrives within the wait, so only the stamps decide.
    current = requests.CurrentImage(max_pair_gap_ns=5_000_000, pair_wait_s=0.02, clock=lambda: 0.0)
    assert current.read_pair() is None
    colour_1, depth_1 = _image(1_000_000_000), _image(1_002_000_000)
    current.put_colour(colour_1)
    assert current.read_pair() is None
    current.put_depth(depth_1)
    assert current.read_pair() == (colour_1, depth_1)
    # The next colour came first: the newest of each no longer match, and the
    # last pair stays whole.
    colour_2 = _image(1_100_000_000)
    current.put_colour(colour_2)
    assert current.read() == (colour_2, depth_1)
    assert current.read_pair() == (colour_1, depth_1)
    depth_2 = _image(1_100_000_000)
    current.put_depth(depth_2)
    assert current.read_pair() == (colour_2, depth_2)


def test_the_current_image_keeps_no_pair_without_a_gap():
    current = requests.CurrentImage()
    current.put_colour(_image(STAMP_NS))
    current.put_depth(_image(STAMP_NS))
    assert current.read_pair() is None


PAIR_GAP_NS = 5_000_000
PAIR_WAIT_S = 0.0625
"""Exact in binary, so two arrivals can be exactly this far apart."""


def _pairing(now: list[float]) -> requests.CurrentImage:
    return requests.CurrentImage(max_pair_gap_ns=PAIR_GAP_NS, pair_wait_s=PAIR_WAIT_S, clock=lambda: now[0])


def _assert_same(got, want) -> None:
    """Assert identity, not equality: two images with equal fields can still be two captures."""
    if want is None:
        assert got is None
    else:
        assert got is not None and len(got) == len(want) and all(g is w for g, w in zip(got, want))


def test_the_current_image_pairs_no_depth_from_before_a_clock_reset():
    now = [100.0]
    current = _pairing(now)
    # The last depth of the previous episode, at 10 s, then a colour image at 11 s.
    current.put_depth(_image(10 * 1_000_000_000))
    current.put_colour(_image(11 * 1_000_000_000))
    assert current.read_pair() is None
    # /clock goes back, and a new colour image has the old depth's stamp. No new depth came.
    now[0] += 0.5
    new_colour = _image(10 * 1_000_000_000)
    current.put_colour(new_colour)
    assert current.read_pair() is None
    _assert_same(current.read_current(), (new_colour, None))


@pytest.mark.parametrize("depth_first", [True, False])
@pytest.mark.parametrize(("apart_s", "pairs"), [(0.0, True), (PAIR_WAIT_S, True), (PAIR_WAIT_S + 0.001, False)])
def test_the_current_image_pairs_only_images_that_arrived_within_the_wait(depth_first, apart_s, pairs):
    now = [100.0]
    current = _pairing(now)
    colour, depth = _image(), _image()
    puts = [(current.put_colour, colour), (current.put_depth, depth)]
    if depth_first:
        puts.reverse()
    (put_first, first), (put_second, second) = puts
    put_first(first)
    now[0] += apart_s
    put_second(second)
    _assert_same(current.read_pair(), (colour, depth) if pairs else None)
    _assert_same(current.read_current(), (colour, depth if pairs else None))


@pytest.mark.parametrize(("colour_ns", "depth_ns"), [(0, 0), (0, 1_000_000), (1_000_000, 0)])
def test_the_current_image_never_pairs_a_zero_stamp(colour_ns, depth_ns):
    current = _pairing([100.0])
    current.put_colour(_image(colour_ns))
    current.put_depth(_image(depth_ns))
    assert current.read_pair() is None


def test_the_current_image_gives_a_depth_only_with_a_pair_of_the_newest_of_each():
    now = [100.0]
    current = _pairing(now)
    colour_1, depth_1 = _image(STAMP_NS), _image(STAMP_NS)
    current.put_colour(colour_1)
    current.put_depth(depth_1)
    _assert_same(current.read_current(), (colour_1, depth_1))
    # The next depth came before its colour image. The pair stays for
    # DetectObjects, but it no longer holds the newest depth.
    depth_2 = _image(STAMP_NS + 100_000_000)
    current.put_depth(depth_2)
    _assert_same(current.read_current(), (colour_1, None))
    _assert_same(current.read_pair(), (colour_1, depth_1))
    # Its colour image came too late to pair with it, so it gets no depth either.
    now[0] += 1.0
    colour_2 = _image(STAMP_NS + 100_000_000)
    current.put_colour(colour_2)
    _assert_same(current.read_current(), (colour_2, None))
    _assert_same(current.read_pair(), (colour_1, depth_1))
    # A depth for colour_2 within the wait pairs, and the newest of each are the pair again.
    depth_3 = _image(STAMP_NS + 100_000_000)
    current.put_depth(depth_3)
    _assert_same(current.read_current(), (colour_2, depth_3))


@pytest.mark.parametrize(("gap_ns", "wait_s"), [(5_000_000, None), (None, 0.02)])
def test_the_current_image_needs_both_pairing_bounds_or_neither(gap_ns, wait_s):
    with pytest.raises(ValueError, match="both"):
        requests.CurrentImage(max_pair_gap_ns=gap_ns, pair_wait_s=wait_s)


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


# --- the wake --------------------------------------------------------------


@pytest.mark.parametrize("make", [NewestFrame, lambda: RgbdBuffer(0.005, 0.020)], ids=["newest", "rgbd"])
def test_a_wake_ends_a_wait_with_no_image(make):
    mailbox = make()
    taken = []
    thread = threading.Thread(target=lambda: taken.append(mailbox.take(LONG_S)), daemon=True)
    thread.start()
    time.sleep(0.05)
    mailbox.wake()
    thread.join(timeout=5.0)
    assert not thread.is_alive(), "the wake did not end the wait"
    assert taken == [None]


@pytest.mark.parametrize("make", [NewestFrame, lambda: RgbdBuffer(0.005, 0.020)], ids=["newest", "rgbd"])
def test_a_wake_before_the_wait_is_not_lost_and_is_used_once(make):
    mailbox = make()
    mailbox.wake()
    started = time.monotonic()
    assert mailbox.take(LONG_S) is None
    assert time.monotonic() - started < 5.0
    # Used up: the next short wait runs its whole length.
    started = time.monotonic()
    assert mailbox.take(0.05) is None
    assert time.monotonic() - started >= 0.04


def test_a_wake_leaves_a_waiting_image_in_place():
    mailbox = NewestFrame()
    item = (Image(), 0.0)
    mailbox.put(item)
    mailbox.wake()
    assert mailbox.take(0.0) is item


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
