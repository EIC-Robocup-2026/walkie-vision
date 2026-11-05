"""Registered depth: pairing, reading, the checks, lifting and scoring, without ROS communication or a GPU."""

from __future__ import annotations

import math
import threading
import time

import cv2
import numpy as np
import pytest
from builtin_interfaces.msg import Time
from sensor_msgs.msg import CameraInfo, Image

from walkie_vision.convert import MAX_CLOUD_POINTS, MIN_CLOUD_POINTS
from walkie_vision.depth import (
    MAX_RAY_ERROR_PX,
    DepthRefused,
    RegisteredDepth,
    RgbdBuffer,
    depth_cover,
    depth_to_metres,
    erosion_radius,
    frame_laplacian,
    lift_mask,
    read_registered_depth,
    stamp_ns,
    view_quality,
)
from walkie_vision.fakes import box_mask
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


def test_clear_drops_a_held_depth_so_it_cannot_pair_within_the_wait(buffer, clock):
    buffer.put_depth(depth())
    buffer.clear()  # The sim restarts and reaches the same stamp again at once.
    c = colour()
    buffer.put((c, 1.0))
    assert buffer.take(0.0) is None
    clock.advance(WAIT_S)
    assert _take(buffer) == (c, None)


def test_clear_drops_a_waiting_colour_and_a_pair_the_worker_has_not_taken(buffer, clock):
    buffer.put_depth(depth())
    buffer.put((colour(), 1.0))
    waiting, candidate = colour(66.0), depth(62.0)
    buffer.put((waiting, 2.0))
    buffer.put_depth(candidate)
    buffer.clear()
    assert buffer.take(0.0) is None
    clock.advance(WAIT_S)
    assert buffer.take(0.0) is None and buffer.replaced == 0
    # Nothing from before the clear pairs with a new depth, and new images pair as before.
    buffer.put_depth(depth(66.0))
    assert buffer.take(0.0) is None
    after_colour, after_depth = colour(80.0), depth(80.0)
    buffer.put((after_colour, 3.0))
    buffer.put_depth(after_depth)
    assert _take(buffer) == (after_colour, after_depth)


def test_clear_with_nothing_held_changes_nothing(buffer):
    buffer.clear()
    assert buffer.take(0.0) is None and buffer.replaced == 0


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


@pytest.mark.parametrize("bigendian", [False, True])
def test_16uc1_gives_the_float32_metres_of_32fc1_for_every_millimetre(bigendian):
    # A 32FC1 image of n mm carries float32(n / 1000), so the two encodings of one depth must agree exactly.
    millimetres = np.arange(1, 65536, dtype=np.uint16).reshape(1, -1)
    from_millimetres = depth_to_metres(_raster(millimetres, "16UC1", bigendian=bigendian, padding=2))
    from_metres = depth_to_metres(_raster((millimetres / 1000).astype(np.float32), "32FC1", bigendian=bigendian))
    np.testing.assert_array_equal(from_millimetres, from_metres)
    assert from_millimetres[0, 2999] == 3.0


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


# --- lifting a mask to points --------------------------------------------------

RASTER_H, RASTER_W = 40, 60
FX, FY, CX, CY = 100.0, 120.0, 30.0, 20.0
MIN_M, MAX_M = 0.2, 3.0


def _plane(value: float = 1.0, height: int = RASTER_H, width: int = RASTER_W) -> np.ndarray:
    return np.full((height, width), value, np.float32)


def _box(x1: int, y1: int, x2: int, y2: int, height: int = RASTER_H, width: int = RASTER_W) -> np.ndarray:
    return box_mask(height, width, (x1, y1, x2, y2))


def _lift(metres: np.ndarray, mask: np.ndarray, *, min_m: float = MIN_M, max_m: float = MAX_M) -> np.ndarray:
    return lift_mask(mask, RegisteredDepth(np.asarray(metres, np.float32), FX, FY, CX, CY), min_m, max_m)


def _pixels(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the integer rows and columns the points came from, by inverting the back-projection."""
    rows = np.rint(points[:, 1] * FY / points[:, 2] + CY).astype(int)
    cols = np.rint(points[:, 0] * FX / points[:, 2] + CX).astype(int)
    return rows, cols


def test_a_plane_lifts_to_the_back_projection_of_the_eroded_mask():
    # 20 x 20 px gives a radius of 1, so one pixel goes on every side.
    points = _lift(_plane(2.0), _box(10, 5, 30, 25))
    rows, cols = np.mgrid[6:24, 11:29]
    expected = np.column_stack(
        ((cols.ravel() - CX) * 2.0 / FX, (rows.ravel() - CY) * 2.0 / FY, np.full(rows.size, 2.0))
    )
    assert points.dtype == np.float32
    np.testing.assert_allclose(points, expected, atol=1e-6)


def test_a_tilted_plane_keeps_its_slope():
    metres = np.broadcast_to(1.0 + 0.001 * np.arange(RASTER_H, dtype=np.float32)[:, None], (RASTER_H, RASTER_W))
    points = _lift(metres, _box(10, 5, 30, 35))
    rows, _ = _pixels(points)
    assert len(points) == 18 * 28
    np.testing.assert_allclose(points[:, 2], 1.0 + 0.001 * rows, atol=1e-6)


def test_a_cube_face_keeps_its_size_and_drops_the_background():
    # A cube's front face at z = 1 m on a wall at 3 m: x from -0.1 to 0.2 m and y from -0.15 to 0.15 m.
    fx, cx, cy = 100.0, 60.0, 40.0
    metres = np.full((80, 120), 3.0, np.float32)
    metres[25:55, 50:80] = 1.0
    # The mask overshoots the face by one pixel on each side. 32 px gives a radius of 2.
    mask = box_mask(80, 120, (49, 24, 81, 56))
    points = lift_mask(mask, RegisteredDepth(metres, fx, fx, cx, cy), MIN_M, MAX_M)
    np.testing.assert_array_equal(points[:, 2], 1.0)
    assert len(points) == 28 * 28
    np.testing.assert_allclose(
        [points[:, 0].min(), points[:, 0].max(), points[:, 1].min(), points[:, 1].max()],
        [(51 - cx) / fx, (78 - cx) / fx, (26 - cy) / fx, (53 - cy) / fx],
        atol=1e-6,
    )


def test_the_whole_mask_is_still_cut_by_the_depth_window():
    # 20 x 3 px: erosion leaves 18 pixels, under the floor. The whole mask has 60,
    # and its top row overshoots onto a wall at 3 m.
    metres = _plane(3.0)
    metres[10:12, 10:30] = 1.0
    points = _lift(metres, _box(10, 9, 30, 12))
    np.testing.assert_array_equal(points[:, 2], 1.0)
    assert len(points) == 2 * 20


def test_points_are_in_the_optical_frame():
    # The optical centre is at pixel (30, 20). Right is +x, down is +y, forward is +z.
    right_below = _lift(_plane(), _box(40, 25, 55, 40))
    assert (right_below[:, 0] > 0).all() and (right_below[:, 1] > 0).all() and (right_below[:, 2] > 0).all()
    left_above = _lift(_plane(), _box(5, 0, 25, 15))
    assert (left_above[:, 0] < 0).all() and (left_above[:, 1] < 0).all() and (left_above[:, 2] > 0).all()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0, MIN_M - 0.01, MAX_M + 0.5])
def test_a_bad_depth_gives_no_point(bad):
    metres = _plane()
    metres[10, 20] = bad
    points = _lift(metres, _box(10, 5, 30, 25))
    assert len(points) == 18 * 18 - 1
    assert np.isfinite(points).all()
    rows, cols = _pixels(points)
    assert not ((rows == 10) & (cols == 20)).any()


def test_the_range_limits_keep_a_depth_on_the_limit():
    assert len(_lift(_plane(1.0), _box(10, 5, 30, 25), min_m=1.0, max_m=1.0 + 1e-3)) == 18 * 18
    assert len(_lift(_plane(1.0), _box(10, 5, 30, 25), min_m=0.5, max_m=1.0)) == 18 * 18
    assert len(_lift(_plane(1.0), _box(10, 5, 30, 25), min_m=1.5, max_m=2.0)) == 0
    assert len(_lift(_plane(3.0), _box(10, 5, 30, 25), min_m=0.2, max_m=2.0)) == 0


@pytest.mark.parametrize(("millimetres", "kept"), [(199, False), (200, True), (3000, True), (3001, False)])
def test_16uc1_and_32fc1_keep_the_same_depths_at_the_default_range_limits(millimetres, kept):
    # MIN_M and MAX_M are the node's default range, 200 mm and 3000 mm, and both limits are inclusive.
    mask = np.ones((1, 1), bool)
    for raster in (_raster([[millimetres]], "16UC1"), _raster([[millimetres / 1000]], "32FC1")):
        assert depth_cover(mask, depth_to_metres(raster), MIN_M, MAX_M) == (1.0 if kept else 0.0), raster.encoding


@pytest.mark.parametrize("millimetres", [200, 3000])
def test_a_16uc1_depth_on_a_range_limit_lifts_like_the_same_32fc1_depth(millimetres):
    from_millimetres = _raster(np.full((RASTER_H, RASTER_W), millimetres), "16UC1")
    from_metres = _raster(np.full((RASTER_H, RASTER_W), millimetres / 1000), "32FC1")
    points = _lift(depth_to_metres(from_millimetres), _box(10, 5, 30, 25))
    assert len(points) == 18 * 18
    np.testing.assert_array_equal(points, _lift(depth_to_metres(from_metres), _box(10, 5, 30, 25)))


def test_a_depth_far_from_the_median_is_dropped_by_the_mad_window():
    # Three levels in equal shares: the median is 1.0 and the MAD 0.02, so the window is 0.0889 m.
    metres = _plane()
    metres[np.arange(RASTER_H) % 3 == 0, :] = 0.98
    metres[np.arange(RASTER_H) % 3 == 2, :] = 1.02
    metres[12, 15] = 1.08
    metres[12, 16] = 1.2
    points = _lift(metres, _box(10, 5, 30, 25))
    rows, cols = _pixels(points)
    assert ((rows == 12) & (cols == 15)).any()
    assert not ((rows == 12) & (cols == 16)).any()
    assert len(points) == 18 * 18 - 1


def test_a_zero_mad_still_keeps_small_noise_within_one_centimetre():
    metres = _plane()
    metres[12, 15] = 1.008
    metres[12, 16] = 1.02
    points = _lift(metres, _box(10, 5, 30, 25))
    rows, cols = _pixels(points)
    assert ((rows == 12) & (cols == 15)).any()
    assert not ((rows == 12) & (cols == 16)).any()


def test_a_thin_mask_falls_back_to_the_whole_mask():
    # 3 x 20 px: erosion leaves 18 pixels, under the floor. The whole mask has 60.
    points = _lift(_plane(), _box(10, 5, 30, 8))
    assert len(points) == 60


def test_the_fallback_applies_the_same_depth_rules():
    # 9 x 9 px: erosion leaves 49 pixels, all without a depth. The ring has 32, of which one is bad.
    metres = _plane()
    metres[6:13, 11:18] = float("nan")
    metres[5, 10] = 0.0
    points = _lift(metres, _box(10, 5, 19, 14))
    assert len(points) == 31
    assert (points[:, 2] == 1.0).all()


@pytest.mark.parametrize(("pixels", "expected"), [(MIN_CLOUD_POINTS - 1, 0), (MIN_CLOUD_POINTS, MIN_CLOUD_POINTS)])
def test_the_point_floor_of_the_whole_mask(pixels, expected):
    # One row: erosion leaves nothing, so the whole mask decides.
    assert len(_lift(_plane(), _box(0, 10, pixels, 11))) == expected


def test_thirty_eroded_points_stay_and_twenty_nine_fall_back():
    # 12 x 5 px: erosion leaves 10 x 3 = 30 pixels, and the whole mask has 60.
    mask = _box(10, 5, 22, 10)
    assert len(_lift(_plane(), mask)) == MIN_CLOUD_POINTS
    metres = _plane()
    metres[6, 11] = float("nan")
    assert len(_lift(metres, mask)) == 59


def test_too_few_points_give_an_empty_array_of_the_right_shape():
    for mask in (np.zeros((RASTER_H, RASTER_W), bool), _box(0, 10, 5, 11)):
        points = _lift(_plane(), mask)
        assert points.shape == (0, 3) and points.dtype == np.float32


def test_the_cap_keeps_evenly_spaced_pixels_in_row_major_order():
    # 100 x 100 px gives a radius of 5, so 90 x 90 = 8100 pixels survive.
    metres = _plane(1.0, 200, 300)
    mask = _box(50, 50, 150, 150, 200, 300)
    points = _lift(metres, mask)
    assert len(points) == MAX_CLOUD_POINTS
    rows, cols = _pixels(points)
    chosen = np.arange(MAX_CLOUD_POINTS) * 8100 // MAX_CLOUD_POINTS
    all_rows, all_cols = np.mgrid[55:145, 55:145]
    np.testing.assert_array_equal(rows, all_rows.ravel()[chosen])
    np.testing.assert_array_equal(cols, all_cols.ravel()[chosen])
    np.testing.assert_array_equal(points, _lift(metres, mask))


def test_exactly_the_cap_keeps_every_pixel():
    # 54 x 44 px gives a radius of 2, so 50 x 40 = 2000 pixels survive.
    points = _lift(_plane(1.0, 200, 300), _box(50, 50, 104, 94, 200, 300))
    assert len(points) == MAX_CLOUD_POINTS
    rows, cols = _pixels(points)
    all_rows, all_cols = np.mgrid[52:92, 52:102]
    np.testing.assert_array_equal(rows, all_rows.ravel())
    np.testing.assert_array_equal(cols, all_cols.ravel())


@pytest.mark.parametrize(
    ("height", "width", "radius"),
    [
        (1, 1, 1),
        (10, 100, 1),
        (20, 20, 1),
        (30, 30, 2),
        (50, 50, 2),  # 2.5 goes to the even side.
        (70, 70, 4),  # 3.5 goes to the even side.
        (100, 100, 5),
        (400, 400, 5),
        (20, 400, 1),
        (400, 20, 1),
    ],
)
def test_the_erosion_radius(height, width, radius):
    mask = np.zeros((500, 500), bool)
    mask[:height, :width] = True
    assert erosion_radius(mask) == radius


def test_the_erosion_radius_uses_the_extent_not_the_pixel_count():
    assert erosion_radius(np.eye(100, dtype=bool)) == 5


def test_an_empty_mask_has_no_erosion_radius():
    assert erosion_radius(np.zeros((5, 5), bool)) == 0


def test_a_mask_on_the_edge_of_the_raster_is_eroded_from_the_edge():
    # The whole raster: 40 px gives a radius of 2, and outside the raster counts as not in the mask.
    points = _lift(_plane(), np.ones((RASTER_H, RASTER_W), bool))
    rows, cols = _pixels(points)
    assert (rows.min(), rows.max(), cols.min(), cols.max()) == (2, 37, 2, 57)


def test_depth_cover_counts_the_valid_pixels_before_erosion():
    metres = _plane()
    mask = _box(0, 0, 10, 10)
    assert depth_cover(mask, metres, MIN_M, MAX_M) == 1.0
    # The outer ring would be gone after erosion, and it still counts.
    metres[0, :10] = float("nan")
    metres[1, :10] = 0.0
    metres[2, :10] = MAX_M + 1.0
    assert depth_cover(mask, metres, MIN_M, MAX_M) == pytest.approx(0.7)


def test_depth_cover_of_an_empty_mask_is_zero():
    assert depth_cover(np.zeros((RASTER_H, RASTER_W), bool), _plane(), MIN_M, MAX_M) == 0.0


# --- view quality ----------------------------------------------------------------

Q_H, Q_W = 100, 100
CENTRED_BOX = np.array([25.0, 25.0, 75.0, 75.0])
FLAT = np.zeros((Q_H, Q_W), np.float32)


def _noise(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (Q_H, Q_W, 3), dtype=np.uint8)


def _q_box(x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    return box_mask(Q_H, Q_W, (x1, y1, x2, y2))


def test_frame_laplacian_uses_the_recipe():
    image = _noise()
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    expected = cv2.Laplacian(grey, cv2.CV_32F, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    result = frame_laplacian(image)
    assert result.dtype == np.float32 and result.shape == (Q_H, Q_W)
    np.testing.assert_array_equal(result, expected)


def test_a_flat_image_has_no_laplacian_even_at_the_border():
    assert not frame_laplacian(np.full((Q_H, Q_W, 3), 200, np.uint8)).any()


def test_view_quality_is_the_mean_of_its_four_terms():
    # A tenth of the image gives a full area term. The box is centred, half the
    # mask has depth, and a flat image has no sharpness: 0.25 * (1 + 1 + 0.5 + 0).
    assert view_quality(_q_box(0, 45, 100, 55), np.array([0.0, 45.0, 100.0, 55.0]), FLAT, 0.5) == pytest.approx(0.625)


def test_the_area_term_is_the_square_root_of_the_share():
    # 250 px is a quarter of the full area. A box on the corner is not centred.
    score = view_quality(_q_box(0, 0, 25, 10), np.array([0.0, 0.0, 0.0, 0.0]), FLAT, 0.0)
    assert score == pytest.approx(0.25 * 0.5)


@pytest.mark.parametrize(
    ("box", "centred"),
    [
        ((50.0, 50.0, 50.0, 50.0), 1.0),
        ((0.0, 0.0, 0.0, 0.0), 0.0),
        ((100.0, 100.0, 100.0, 100.0), 0.0),
        ((100.0, 50.0, 100.0, 50.0), 1.0 - 1.0 / math.sqrt(2.0)),
    ],
)
def test_the_centred_term_falls_with_the_distance_from_the_centre(box, centred):
    # Area, depth and sharpness are all 0 or known: a tenth of the image gives 1.
    score = view_quality(_q_box(0, 0, 100, 10), np.array(box), FLAT, 0.0)
    assert score == pytest.approx(0.25 * (1.0 + centred))


def test_more_depth_cover_scores_higher_by_a_quarter_of_the_difference():
    mask = _q_box(25, 25, 75, 75)
    low, high = view_quality(mask, CENTRED_BOX, FLAT, 0.25), view_quality(mask, CENTRED_BOX, FLAT, 0.75)
    assert high - low == pytest.approx(0.125)


def test_a_sharper_view_scores_higher():
    image = _noise()
    mask = _q_box(25, 25, 75, 75)
    sharp = view_quality(mask, CENTRED_BOX, frame_laplacian(image), 0.0)
    blurred = view_quality(mask, CENTRED_BOX, frame_laplacian(cv2.GaussianBlur(image, (9, 9), 3.0)), 0.0)
    assert sharp > blurred


def test_a_mask_over_an_edge_scores_higher_than_a_mask_over_a_flat_part():
    # The same mask size and box, so only the sharpness term can differ.
    image = np.zeros((Q_H, Q_W, 3), np.uint8)
    image[:, 50:] = 255
    laplacian = frame_laplacian(image)
    over_edge = view_quality(_q_box(40, 0, 60, 20), CENTRED_BOX, laplacian, 0.0)
    over_flat = view_quality(_q_box(70, 0, 90, 20), CENTRED_BOX, laplacian, 0.0)
    assert over_edge > over_flat


def test_sharpness_needs_fifty_mask_pixels():
    laplacian = frame_laplacian(_noise())
    below = _q_box(25, 25, 74, 26)  # 49 px
    enough = _q_box(25, 25, 75, 26)  # 50 px
    score_below = view_quality(below, CENTRED_BOX, laplacian, 0.0)
    score_enough = view_quality(enough, CENTRED_BOX, laplacian, 0.0)
    assert score_below == pytest.approx(0.25 * (math.sqrt(49 / 1000) + 1.0))
    # The noise image's Laplacian variance is far above the full mark.
    assert score_enough == pytest.approx(0.25 * (math.sqrt(50 / 1000) + 1.0 + 1.0))


def test_an_empty_mask_scores_zero():
    assert view_quality(np.zeros((Q_H, Q_W), bool), CENTRED_BOX, FLAT, 1.0) == 0.0


def test_the_best_view_scores_one():
    mask = np.ones((Q_H, Q_W), bool)
    assert view_quality(mask, np.array([0.0, 0.0, 100.0, 100.0]), frame_laplacian(_noise()), 1.0) == 1.0


def test_every_score_is_finite_and_in_range():
    rng = np.random.default_rng(3)
    laplacian = frame_laplacian(_noise())
    for _ in range(200):
        x1, x2 = sorted(rng.integers(0, Q_W + 1, 2).tolist())
        y1, y2 = sorted(rng.integers(0, Q_H + 1, 2).tolist())
        score = view_quality(_q_box(x1, y1, x2, y2), np.array([x1, y1, x2, y2], float), laplacian, float(rng.random()))
        assert math.isfinite(score) and 0.0 <= score <= 1.0
