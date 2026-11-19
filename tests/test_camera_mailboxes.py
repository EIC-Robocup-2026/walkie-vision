"""walkie_vision.camera.mailboxes: the newest frame, the colour and depth pairing, the current image and the wake. No ROS graph."""

from __future__ import annotations

import gc
import threading
import time
import weakref

import pytest
from builtin_interfaces.msg import Time
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from walkie_vision.camera.mailboxes import CurrentImage, NewestFrame, RgbdBuffer

FRAME = "zed_head_left_camera_frame_optical"
GAP_S, WAIT_S = 0.005, 0.020
BASE_MS = 100_000  # Stamps start at 100 s: a zero stamp never pairs.

IMAGE_FRAME = "camera_optical"
WIDTH, HEIGHT = 64, 48
STAMP_NS = 10_000_000_000
LONG_S = 30.0


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


def _stamp(ns: int) -> Time:
    return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


def _image(ns: int = STAMP_NS, frame: str = IMAGE_FRAME, width: int = WIDTH, height: int = HEIGHT) -> Image:
    return Image(header=Header(stamp=_stamp(ns), frame_id=frame), width=width, height=height)


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


# --- the current image -----------------------------------------------------


class _Msg:
    """Unlike a generated message, this can have a weak reference."""


def test_the_current_image_keeps_only_the_newest_of_each():
    current = CurrentImage()
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
    current = CurrentImage(max_pair_gap_ns=5_000_000, pair_wait_s=0.02, clock=lambda: 0.0)
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
    current = CurrentImage()
    current.put_colour(_image(STAMP_NS))
    current.put_depth(_image(STAMP_NS))
    assert current.read_pair() is None


PAIR_GAP_NS = 5_000_000
PAIR_WAIT_S = 0.0625
"""Exact in binary, so two arrivals can be exactly this far apart."""


def _pairing(now: list[float]) -> CurrentImage:
    return CurrentImage(max_pair_gap_ns=PAIR_GAP_NS, pair_wait_s=PAIR_WAIT_S, clock=lambda: now[0])


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


def test_the_current_image_forgets_every_image_and_the_pair_on_clear():
    now = [100.0]
    current = _pairing(now)
    current.put_depth(_image(10 * 1_000_000_000))
    current.put_colour(_image(10 * 1_000_000_000))
    assert current.read_pair() is not None
    current.clear()
    assert current.read() == (None, None)
    assert current.read_current() == (None, None)
    assert current.read_pair() is None
    # Within the wait, by the arrival clock, a new image with the old stamps pairs with nothing from before.
    new_colour = _image(10 * 1_000_000_000)
    current.put_colour(new_colour)
    assert current.read_pair() is None
    _assert_same(current.read_current(), (new_colour, None))
    new_depth = _image(10 * 1_000_000_000)
    current.put_depth(new_depth)
    _assert_same(current.read_pair(), (new_colour, new_depth))


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
        CurrentImage(max_pair_gap_ns=gap_ns, pair_wait_s=wait_s)


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


def test_clear_drops_the_waiting_image():
    mailbox = NewestFrame()
    mailbox.put((Image(), 0.0))
    mailbox.clear()
    assert mailbox.take(0.0) is None and mailbox.replaced == 0
    item = (Image(), 1.0)
    mailbox.put(item)
    assert mailbox.take(0.0) is item


def test_a_wake_leaves_a_waiting_image_in_place():
    mailbox = NewestFrame()
    item = (Image(), 0.0)
    mailbox.put(item)
    mailbox.wake()
    assert mailbox.take(0.0) is item
