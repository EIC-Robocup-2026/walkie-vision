"""Registered depth: pairing, reading and the checks, without ROS communication or a GPU."""

from __future__ import annotations

import math
import threading
import time

import numpy as np
import pytest
from builtin_interfaces.msg import Time
from sensor_msgs.msg import CameraInfo, Image

from walkie_vision.depth import (
    MAX_RAY_ERROR_PX,
    DepthRefused,
    RegisteredDepth,
    RgbdBuffer,
    depth_to_metres,
    read_registered_depth,
    stamp_ns,
)
from walkie_vision.image import calibration_problem

FRAME = "zed_head_left_camera_frame_optical"
GAP_S, WAIT_S = 0.005, 0.020
BASE_MS = 100_000  # Stamps start at 100 s: a zero stamp never pairs.
W, H = 8, 6
K = [100.0, 0.0, 3.5, 0.0, 100.0, 2.5, 0.0, 0.0, 1.0]


class FakeClock:
    """A wall clock that only moves when a test moves it."""

    def __init__(self) -> None:
        self.now = 50.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _stamped(ms: float, frame: str = FRAME) -> Image:
    msg = Image()
    ns = round(ms * 1_000_000)
    msg.header.stamp = Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)
    msg.header.frame_id = frame
    return msg


def colour(offset_ms: float = 0.0) -> Image:
    return _stamped(BASE_MS + offset_ms)


def depth(offset_ms: float = 0.0) -> Image:
    return _stamped(BASE_MS + offset_ms)


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def buffer(clock):
    return RgbdBuffer(GAP_S, WAIT_S, clock=clock)


def _take(buffer: RgbdBuffer):
    """The bundle's colour and depth messages, or None, without waiting."""
    item = buffer.take(0.0)
    return None if item is None else (item[0], item[2])


# --- pairing --------------------------------------------------------------


def test_an_exact_stamp_pairs_at_once_when_the_colour_comes_first(buffer):
    c, d = colour(), depth()
    buffer.put((c, 1.0))
    assert buffer.take(0.0) is None
    buffer.put_depth(d)
    assert buffer.take(0.0) == (c, 1.0, d)


def test_an_exact_stamp_pairs_at_once_when_the_depth_comes_first(buffer):
    c, d = colour(), depth()
    buffer.put_depth(d)
    assert buffer.take(0.0) is None
    buffer.put((c, 1.0))
    assert buffer.take(0.0) == (c, 1.0, d)


@pytest.mark.parametrize("offset_ms", [0.0, 4.0, 5.0])
def test_a_depth_after_the_colour_within_the_gap_pairs_at_once(buffer, offset_ms):
    c, d = colour(), depth(offset_ms)
    buffer.put((c, 1.0))
    buffer.put_depth(d)
    assert _take(buffer) == (c, d)


@pytest.mark.parametrize("offset_ms", [4.0, 5.0])
def test_a_depth_before_the_colour_within_the_gap_pairs_when_the_wait_ends(buffer, clock, offset_ms):
    c, d = colour(), depth(-offset_ms)
    buffer.put_depth(d)
    buffer.put((c, 1.0))
    assert buffer.take(0.0) is None, "an exact depth may still arrive"
    clock.advance(WAIT_S)
    assert _take(buffer) == (c, d)


@pytest.mark.parametrize("offset_ms", [-6.0, 6.0])
def test_a_depth_past_the_gap_does_not_pair(buffer, clock, offset_ms):
    c, d = colour(), depth(offset_ms)
    buffer.put((c, 1.0))
    buffer.put_depth(d)
    assert buffer.take(0.0) is None
    clock.advance(WAIT_S)
    assert _take(buffer) == (c, None)


def test_a_colour_waits_at_most_the_wall_wait(buffer, clock):
    c = colour()
    buffer.put((c, 1.0))
    clock.advance(WAIT_S * 0.99)
    assert buffer.take(0.0) is None
    clock.advance(WAIT_S * 0.02)
    assert _take(buffer) == (c, None)


def test_an_exact_depth_beats_an_earlier_candidate(buffer, clock):
    c, early, exact = colour(), depth(-4.0), depth()
    buffer.put_depth(early)
    buffer.put((c, 1.0))
    clock.advance(WAIT_S / 2)
    buffer.put_depth(exact)
    assert _take(buffer) == (c, exact)
    assert buffer.take(0.0) is None


def test_the_nearer_of_two_inexact_depths_wins(buffer):
    c, early, late = colour(), depth(-4.0), depth(3.0)
    buffer.put_depth(early)
    buffer.put((c, 1.0))
    buffer.put_depth(late)
    assert _take(buffer) == (c, late)


def test_a_candidate_that_is_nearer_wins_and_the_later_depth_stays_for_the_next_image(buffer):
    c, early, late = colour(), depth(-2.0), depth(4.0)
    buffer.put_depth(early)
    buffer.put((c, 1.0))
    buffer.put_depth(late)
    assert _take(buffer) == (c, early)
    following = colour(4.0)
    buffer.put((following, 2.0))
    assert _take(buffer) == (following, late)


def test_a_later_depth_past_the_gap_ends_the_wait_for_a_candidate(buffer):
    c, early, late = colour(), depth(-4.0), depth(66.0)
    buffer.put_depth(early)
    buffer.put((c, 1.0))
    buffer.put_depth(late)
    assert _take(buffer) == (c, early)


def test_a_depth_after_the_wait_does_not_pair(buffer, clock):
    c, d = colour(), depth()
    buffer.put((c, 1.0))
    clock.advance(WAIT_S)
    buffer.put_depth(d)
    assert _take(buffer) == (c, None)


def test_zero_stamps_never_pair(buffer, clock):
    c, d = Image(), Image()
    buffer.put_depth(d)
    buffer.put((c, 1.0))
    assert buffer.take(0.0) is None
    clock.advance(WAIT_S)
    assert _take(buffer) == (c, None)


def test_a_zero_gap_pairs_only_an_exact_stamp(clock):
    buffer = RgbdBuffer(0.0, WAIT_S, clock=clock)
    c = colour()
    buffer.put((c, 1.0))
    buffer.put_depth(depth(1.0))
    assert buffer.take(0.0) is None
    exact = depth()
    buffer.put_depth(exact)
    assert _take(buffer) == (c, exact)


# --- no stale pair --------------------------------------------------------


def test_a_depth_is_never_paired_twice(buffer, clock):
    first, second, d = colour(), colour(), depth()
    buffer.put((first, 1.0))
    buffer.put_depth(d)
    assert _take(buffer) == (first, d)
    buffer.put((second, 2.0))
    clock.advance(WAIT_S)
    assert _take(buffer) == (second, None)


def test_a_paused_clock_ends_the_wait_and_reuses_nothing(buffer, clock):
    # /clock has stopped, so stamps repeat. Wall time still moves.
    for sent in (1.0, 2.0, 3.0):
        c = colour()
        buffer.put((c, sent))
        assert buffer.take(0.0) is None
        clock.advance(WAIT_S)
        assert _take(buffer) == (c, None)


def test_a_reset_clock_does_not_match_an_old_depth_with_an_equal_stamp(buffer, clock):
    old = depth()
    buffer.put_depth(old)
    clock.advance(1.0)  # The sim restarts and reaches the same stamp again.
    c = colour()
    buffer.put((c, 1.0))
    clock.advance(WAIT_S)
    assert _take(buffer) == (c, None)


def test_a_reset_clock_pairs_again_on_the_new_stamps(buffer, clock):
    before = colour()
    buffer.put((before, 1.0))
    after_depth = _stamped(0.5)
    buffer.put_depth(after_depth)  # 100 s behind the waiting image.
    assert buffer.take(0.0) is None
    clock.advance(WAIT_S)
    assert _take(buffer) == (before, None)

    after_colour, exact = _stamped(50.0), _stamped(50.0)
    buffer.put((after_colour, 2.0))
    buffer.put_depth(exact)
    assert _take(buffer) == (after_colour, exact)


def test_a_depth_older_than_the_wait_is_dropped_when_an_image_arrives(buffer, clock):
    buffer.put_depth(depth())
    clock.advance(WAIT_S * 1.01)
    c = colour()
    buffer.put((c, 1.0))
    # The old depth is gone, so a new one with the same stamp is the only match.
    fresh = depth()
    buffer.put_depth(fresh)
    assert _take(buffer) == (c, fresh)


# --- queue replacement ----------------------------------------------------


def test_a_newer_colour_replaces_a_waiting_one(buffer):
    old, new, d = colour(), colour(66.0), depth(66.0)
    buffer.put((old, 1.0))
    buffer.put((new, 2.0))
    assert buffer.replaced == 1
    buffer.put_depth(d)
    assert _take(buffer) == (new, d)
    assert buffer.take(0.0) is None


def test_a_newer_depth_replaces_a_waiting_one(buffer):
    gone, kept = depth(), depth(66.0)
    buffer.put_depth(gone)
    buffer.put_depth(kept)
    c = colour(0.0)
    buffer.put((c, 1.0))
    # The older depth had the exact stamp, but only the newest is kept.
    assert buffer.take(0.0) is None


def test_a_new_pair_replaces_a_pair_the_worker_has_not_taken(buffer):
    buffer.put_depth(depth())
    buffer.put((colour(), 1.0))
    c, d = colour(66.0), depth(66.0)
    buffer.put_depth(d)
    buffer.put((c, 2.0))
    assert buffer.replaced == 1
    assert _take(buffer) == (c, d)
    assert buffer.take(0.0) is None


def test_a_replaced_colour_with_a_candidate_is_finished_with_it(buffer):
    old, candidate, new = colour(), depth(-4.0), colour(66.0)
    buffer.put_depth(candidate)
    buffer.put((old, 1.0))
    buffer.put((new, 2.0))
    assert buffer.replaced == 0
    assert _take(buffer) == (old, candidate)


def test_colours_faster_than_the_wait_still_reach_the_worker(buffer, clock):
    first_sent = colour()
    buffer.put((first_sent, 1.0))
    newest = first_sent
    for i in range(1, 3):
        clock.advance(WAIT_S * 0.4)
        newest = colour(i * 8.0)
        buffer.put((newest, 1.0 + i))
        assert buffer.take(0.0) is None
    clock.advance(WAIT_S * 0.3)
    assert _take(buffer) == (newest, None)


def test_a_colour_that_ran_out_does_not_replace_a_ready_pair(buffer, clock):
    paired, d = colour(), depth()
    buffer.put((paired, 1.0))
    buffer.put_depth(d)
    unpaired = colour(66.0)
    buffer.put((unpaired, 2.0))
    clock.advance(WAIT_S)
    assert _take(buffer) == (paired, d)
    assert _take(buffer) == (unpaired, None)
    assert buffer.replaced == 0


# --- take and close -------------------------------------------------------


def test_take_with_no_timeout_does_not_wait():
    buffer = RgbdBuffer(GAP_S, 5.0)
    buffer.put((colour(), 1.0))
    started = time.monotonic()
    assert buffer.take(0.0) is None
    assert time.monotonic() - started < 1.0


def test_take_returns_when_the_wall_wait_ends():
    buffer = RgbdBuffer(GAP_S, WAIT_S)
    c = colour()
    buffer.put((c, 1.0))
    started = time.monotonic()
    item = buffer.take(5.0)
    assert time.monotonic() - started < 2.0, "take waited for its timeout, not for the end of the pair wait"
    assert item == (c, 1.0, None)


def test_take_gives_up_after_its_timeout():
    buffer = RgbdBuffer(GAP_S, WAIT_S)
    started = time.monotonic()
    assert buffer.take(0.05) is None
    assert time.monotonic() - started >= 0.05


def test_a_waiting_take_wakes_for_a_pair():
    buffer = RgbdBuffer(GAP_S, 5.0)
    got = []
    worker = threading.Thread(target=lambda: got.append(buffer.take(5.0)))
    worker.start()
    c, d = colour(), depth()
    buffer.put((c, 1.0))
    buffer.put_depth(d)
    worker.join(4.0)
    assert not worker.is_alive()
    assert got == [(c, 1.0, d)]


def test_close_wakes_a_waiting_take_and_refuses_new_images():
    buffer = RgbdBuffer(GAP_S, WAIT_S)
    got = []
    worker = threading.Thread(target=lambda: got.append(buffer.take(30.0)))
    worker.start()
    buffer.close()
    worker.join(4.0)
    assert not worker.is_alive()
    assert got == [None]
    assert buffer.closed
    buffer.put((colour(), 1.0))
    buffer.put_depth(depth())
    assert buffer.take(0.0) is None


# --- reading a raster -----------------------------------------------------


def _raster(values, encoding: str, *, bigendian: bool = False, padding: int = 0) -> Image:
    """A depth image with these values, in the given byte order and with padded rows."""
    code = {"32FC1": "f4", "16UC1": "u2"}[encoding]
    array = np.asarray(values, dtype=(">" if bigendian else "<") + code)
    rows = [row.tobytes() + b"\xee" * padding for row in array]
    msg = _stamped(BASE_MS)
    msg.height, msg.width = array.shape
    msg.encoding = encoding
    msg.is_bigendian = int(bigendian)
    msg.step = int(array.shape[1]) * int(code[1]) + padding
    msg.data = b"".join(rows)
    return msg


METRES = [[0.5, 1.25, 2.0], [3.0, float("nan"), -1.0]]
MILLIMETRES = [[500, 1250, 2000], [3000, 0, 65535]]


@pytest.mark.parametrize("bigendian", [False, True])
@pytest.mark.parametrize("padding", [0, 2, 4, 7])
def test_32fc1_is_metres_with_any_padding_and_byte_order(bigendian, padding):
    metres = depth_to_metres(_raster(METRES, "32FC1", bigendian=bigendian, padding=padding))
    assert metres.dtype == np.float32 and metres.dtype.isnative
    assert metres.flags.c_contiguous and metres.flags.writeable
    np.testing.assert_array_equal(metres, np.asarray(METRES, dtype=np.float32))


@pytest.mark.parametrize("bigendian", [False, True])
@pytest.mark.parametrize("padding", [0, 2, 3])
def test_16uc1_is_millimetres_with_any_padding_and_byte_order(bigendian, padding):
    metres = depth_to_metres(_raster(MILLIMETRES, "16UC1", bigendian=bigendian, padding=padding))
    assert metres.dtype == np.float32 and metres.dtype.isnative
    assert metres.flags.c_contiguous
    np.testing.assert_allclose(metres[0], [0.5, 1.25, 2.0], atol=1e-6)
    assert metres[1, 0] == pytest.approx(3.0)
    assert math.isnan(metres[1, 1]), "0 means no measurement"
    assert metres[1, 2] == pytest.approx(65.535)


def test_the_raster_is_a_copy():
    msg = _raster(METRES, "32FC1")
    metres = depth_to_metres(msg)
    metres[0, 0] = 9.0
    assert depth_to_metres(msg)[0, 0] == pytest.approx(0.5)


@pytest.mark.parametrize("encoding", ["mono8", "bgr8", "16SC1", "32FC3", "", "32fc1"])
def test_another_encoding_is_refused(encoding):
    msg = _raster(METRES, "32FC1")
    msg.encoding = encoding
    with pytest.raises(DepthRefused, match="encoding") as e:
        depth_to_metres(msg)
    assert e.value.reason == "depth_encoding"


@pytest.mark.parametrize(
    "change",
    [
        {"width": 0},
        {"height": 0},
        {"step": 11},  # 3 pixels of 4 bytes are 12 bytes.
        {"height": 3},  # More rows than the data holds.
        {"step": 13},  # Padding that the data does not hold.
    ],
)
def test_a_broken_raster_is_refused(change):
    msg = _raster(METRES, "32FC1")
    for key, value in change.items():
        setattr(msg, key, value)
    with pytest.raises(DepthRefused) as e:
        depth_to_metres(msg)
    assert e.value.reason == "depth_data"


# --- the registration checks ----------------------------------------------


def _info(width: int = W, height: int = H, k=None, frame: str = FRAME, d=()) -> CameraInfo:
    info = CameraInfo()
    info.header.frame_id = frame
    info.width, info.height = width, height
    info.k = list(K if k is None else k)
    info.d = list(d)
    return info


def _colour_image(width: int = W, height: int = H, frame: str = FRAME) -> Image:
    msg = _stamped(BASE_MS, frame)
    msg.width, msg.height = width, height
    msg.encoding = "bgr8"
    msg.step = width * 3
    return msg


def _depth_image(width: int = W, height: int = H, frame: str = FRAME, value: float = 1.0, encoding: str = "32FC1") -> Image:
    msg = _raster(np.full((height, width), value), encoding)
    msg.header.frame_id = frame
    return msg


def _scaled_k(scale_x: float, scale_y: float) -> list[float]:
    """K for a raster that is scaled from the colour image, by the pixel-centre rule."""
    k = list(K)
    k[0] = scale_x * K[0]
    k[2] = scale_x * (K[2] + 0.5) - 0.5
    k[4] = scale_y * K[4]
    k[5] = scale_y * (K[5] + 0.5) - 0.5
    return k


def test_a_registered_raster_of_the_same_size_is_accepted():
    got = read_registered_depth(_depth_image(value=1.5), _info(), _colour_image(), _info())
    assert isinstance(got, RegisteredDepth)
    assert got.metres.shape == (H, W)
    assert (got.fx, got.fy, got.cx, got.cy) == (K[0], K[4], K[2], K[5])
    np.testing.assert_array_equal(got.metres, np.full((H, W), 1.5, dtype=np.float32))


def test_a_lower_resolution_raster_uses_its_own_calibration():
    k = _scaled_k(0.5, 0.5)
    got = read_registered_depth(_depth_image(W // 2, H // 2), _info(W // 2, H // 2, k), _colour_image(), _info())
    assert got.metres.shape == (H // 2, W // 2)
    assert (got.fx, got.fy, got.cx, got.cy) == (k[0], k[4], k[2], k[5])


def test_a_lower_resolution_raster_with_only_the_colour_calibration_is_refused():
    # No intrinsics are made up from the colour image's.
    with pytest.raises(DepthRefused) as e:
        read_registered_depth(_depth_image(W // 2, H // 2), _info(), _colour_image(), _info())
    assert e.value.reason == "depth_camera_info"


@pytest.mark.parametrize(
    ("depth_frame", "depth_info_frame", "colour_info_frame", "reason"),
    [
        ("other", FRAME, FRAME, "depth_frame"),
        ("", FRAME, FRAME, "depth_frame"),
        (FRAME, "other", FRAME, "depth_camera_info"),
        (FRAME, FRAME, "other", "camera_info_frame"),
    ],
)
def test_a_wrong_frame_is_refused(depth_frame, depth_info_frame, colour_info_frame, reason):
    with pytest.raises(DepthRefused) as e:
        read_registered_depth(
            _depth_image(frame=depth_frame), _info(frame=depth_info_frame), _colour_image(), _info(frame=colour_info_frame)
        )
    assert e.value.reason == reason


def test_a_missing_camera_info_is_refused():
    with pytest.raises(DepthRefused) as e:
        read_registered_depth(_depth_image(), None, _colour_image(), _info())
    assert e.value.reason == "depth_camera_info"
    with pytest.raises(DepthRefused) as e:
        read_registered_depth(_depth_image(), _info(), _colour_image(), None)
    assert e.value.reason == "camera_info_frame"


def test_a_camera_info_of_another_size_is_refused():
    with pytest.raises(DepthRefused) as e:
        read_registered_depth(_depth_image(), _info(W + 1, H), _colour_image(), _info())
    assert e.value.reason == "depth_camera_info"
    with pytest.raises(DepthRefused) as e:
        read_registered_depth(_depth_image(), _info(), _colour_image(), _info(W + 1, H))
    assert e.value.reason == "depth_camera_info"


@pytest.mark.parametrize(
    "bad",
    [
        {"k": [float("nan"), 0.0, 3.5, 0.0, 100.0, 2.5, 0.0, 0.0, 1.0]},
        {"k": [100.0, 0.0, float("inf"), 0.0, 100.0, 2.5, 0.0, 0.0, 1.0]},
        {"k": [-100.0, 0.0, 3.5, 0.0, 100.0, 2.5, 0.0, 0.0, 1.0]},
        {"k": [100.0, 0.0, 3.5, 0.0, -1.0, 2.5, 0.0, 0.0, 1.0]},
        {"d": [0.1, 0.0, 0.0, 0.0, 0.0]},
    ],
)
@pytest.mark.parametrize("target", ["depth", "colour"])
def test_a_bad_calibration_is_refused(bad, target):
    depth_info = _info(**bad) if target == "depth" else _info()
    colour_info = _info(**bad) if target == "colour" else _info()
    with pytest.raises(DepthRefused) as e:
        read_registered_depth(_depth_image(), depth_info, _colour_image(), colour_info)
    assert e.value.reason == "depth_calibration"


def test_a_zero_focal_length_is_refused():
    with pytest.raises(DepthRefused) as e:
        read_registered_depth(
            _depth_image(), _info(k=[0.0, 0.0, 3.5, 0.0, 100.0, 2.5, 0.0, 0.0, 1.0]), _colour_image(), _info()
        )
    assert e.value.reason == "depth_camera_info"


@pytest.mark.parametrize(
    "k",
    [
        [100.0, 0.0, 3.5 + MAX_RAY_ERROR_PX * 2, 0.0, 100.0, 2.5, 0.0, 0.0, 1.0],
        [100.0, 0.0, 3.5, 0.0, 100.0, 2.5 - MAX_RAY_ERROR_PX * 2, 0.0, 0.0, 1.0],
        [20.0, 0.0, 3.5, 0.0, 100.0, 2.5, 0.0, 0.0, 1.0],
    ],
)
def test_other_rays_are_refused(k):
    with pytest.raises(DepthRefused, match="other rays") as e:
        read_registered_depth(_depth_image(), _info(k=k), _colour_image(), _info())
    assert e.value.reason == "depth_calibration"


def test_a_raster_cropped_from_the_colour_image_is_refused():
    # The same focal length and centre, on half the width: other rays.
    k = _scaled_k(1.0, 1.0)
    with pytest.raises(DepthRefused, match="other rays"):
        read_registered_depth(_depth_image(W // 2, H), _info(W // 2, H, k), _colour_image(), _info())


def test_rays_within_the_tolerance_are_accepted():
    k = [100.0, 0.0, 3.5 + MAX_RAY_ERROR_PX * 0.9, 0.0, 100.0, 2.5, 0.0, 0.0, 1.0]
    assert read_registered_depth(_depth_image(), _info(k=k), _colour_image(), _info()).cx == k[2]


def test_a_bad_encoding_is_refused_by_the_whole_check():
    bad = _depth_image()
    bad.encoding = "mono8"
    with pytest.raises(DepthRefused) as e:
        read_registered_depth(bad, _info(), _colour_image(), _info())
    assert e.value.reason == "depth_encoding"


def test_calibration_problem_accepts_empty_and_zero_distortion():
    assert calibration_problem(_info()) is None
    assert calibration_problem(_info(d=[0.0] * 5)) is None


def test_registered_depth_refuses_bad_intrinsics():
    with pytest.raises(ValueError, match="intrinsics"):
        RegisteredDepth(np.zeros((2, 2), np.float32), float("nan"), 1.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="dimensions"):
        RegisteredDepth(np.zeros(4, np.float32), 1.0, 1.0, 0.0, 0.0)


def test_stamp_ns():
    assert stamp_ns(_stamped(1500.5).header) == 1_500_500_000
