"""The mailboxes that hand the newest camera images to the vision node's worker and services.

NewestFrame holds the newest colour image for the worker. RgbdBuffer does the
same with depth (ADR-0014): it pairs the newest colour image with the newest
depth image. The stamps may differ by a small gap, and a colour image waits a
short wall time for its depth. Nothing is kept beyond the newest colour, the
newest depth and one finished bundle, so no backlog can grow.

CurrentImage holds the newest colour and depth images that the services
answer from (ADR-0015).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from sensor_msgs.msg import Image

from ..messages import stamp_ns

Bundle = tuple[Image, float, Image | None]
"""A colour image, the time it was received, and its depth or None."""


class NewestFrame:
    """A one-slot mailbox: a new item replaces the one that waits."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._item: tuple[Image, float] | None = None
        self._closed = False
        self._woken = False
        self.replaced = 0
        """How many waiting items a newer one replaced."""

    def put(self, item: tuple[Image, float]) -> None:
        with self._cond:
            if self._closed:
                return
            if self._item is not None:
                self.replaced += 1
            self._item = item
            self._cond.notify()

    def take(self, timeout: float) -> tuple[Image, float] | None:
        """Return the waiting item, waiting up to timeout seconds; None when there is none."""
        with self._cond:
            if self._item is None and not self._closed and not self._woken and timeout > 0:
                self._cond.wait(timeout)
            self._woken = False
            item, self._item = self._item, None
            return item

    def clear(self) -> None:
        """Drop the waiting item, as after a sim restart. An item that take() already returned goes on."""
        with self._cond:
            self._item = None

    def wake(self) -> None:
        """End the current or the next take() early, so the worker can look at a service job."""
        with self._cond:
            self._woken = True
            self._cond.notify_all()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._item = None
            self._cond.notify_all()

    @property
    def closed(self) -> bool:
        return self._closed


@dataclass
class _WaitingColour:
    msg: Image
    received: float
    deadline: float
    candidate: Image | None = None
    """A depth just before this image, within the gap. Used if nothing better arrives in time."""


class RgbdBuffer:
    """Pairs colour and depth images, with the mailbox interface of NewestFrame.

    The callbacks pair on arrival. An exact stamp pairs at once. A depth just
    after the colour image, within max_gap_s, also pairs at once, because the
    streams are in order and no nearer one can follow. A depth just before it
    is held, in case an exact or later one arrives. The colour image gets
    wait_s of wall time, and then goes to the worker with the depth it holds,
    or with None.

    Wall time bounds every wait, so a paused or reset sim clock cannot hang
    the worker. For the same reason, a depth that waited longer than wait_s is
    dropped, so a reset clock cannot match it to a new image with an equal
    stamp. Within wait_s, clear() does that, and drops every image the worker
    has not taken: the node calls it when the sim clock goes back. Pairing
    uses up both images, and a zero stamp never pairs, because two zero stamps
    say nothing about the capture.
    """

    def __init__(self, max_gap_s: float, wait_s: float, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._max_gap_ns = round(max_gap_s * 1e9)
        self._wait_s = wait_s
        self._clock = clock
        self._cond = threading.Condition()
        self._colour: _WaitingColour | None = None
        self._depth: tuple[Image, float] | None = None
        self._ready: Bundle | None = None
        self._closed = False
        self._woken = False
        self.replaced = 0
        """How many colour images a newer one replaced before the worker took them."""

    def put(self, item: tuple[Image, float]) -> None:
        """Take a colour image and the time it was received."""
        msg, received = item
        with self._cond:
            if self._closed:
                return
            now = self._clock()
            deadline = now + self._wait_s
            old = self._colour
            if old is not None:
                if old.candidate is not None:
                    self._finish(old, old.candidate)
                else:
                    self.replaced += 1
                # Above 1 / wait_s Hz, a fresh deadline per image would never end.
                if now < old.deadline:
                    deadline = old.deadline
            self._colour = _WaitingColour(msg, received, deadline)
            if self._depth is not None:
                depth, arrived = self._depth
                if now - arrived > self._wait_s or self._offer(depth):
                    self._depth = None
            self._cond.notify()

    def put_depth(self, depth: Image) -> None:
        """Take a depth image. It is used now or kept as the newest depth."""
        with self._cond:
            if self._closed:
                return
            now = self._clock()
            if self._colour is not None and now < self._colour.deadline and self._offer(depth):
                self._depth = None
            else:
                self._depth = (depth, now)
            self._cond.notify()

    def clear(self) -> None:
        """Drop every image the worker has not taken, as after a sim restart.

        The waiting colour image, the depth it holds, the newest unpaired
        depth and a finished bundle are all dropped, so nothing received before
        this call reaches the worker or pairs with anything received after it.
        A bundle that take() already returned goes on.
        """
        with self._cond:
            self._colour = None
            self._depth = None
            self._ready = None
            self._cond.notify()

    def _offer(self, depth: Image) -> bool:
        """Offer a depth to the waiting colour image. True when it was used up, or lost to a nearer one."""
        colour = self._colour
        assert colour is not None
        colour_ns, depth_ns = stamp_ns(colour.msg.header), stamp_ns(depth.header)
        if colour_ns == 0 or depth_ns == 0:
            return False
        after_ns = depth_ns - colour_ns
        within = abs(after_ns) <= self._max_gap_ns
        candidate = colour.candidate
        candidate_gap_ns = None if candidate is None else colour_ns - stamp_ns(candidate.header)
        if after_ns == 0 or (within and after_ns > 0 and (candidate is None or after_ns <= candidate_gap_ns)):
            self._finish(colour, depth)
            return True
        if after_ns > 0:
            # Later than the colour image, and no nearer than the candidate.
            if candidate is not None:
                self._finish(colour, candidate)
            return False
        if within:
            if candidate is None or -after_ns <= candidate_gap_ns:
                colour.candidate = depth
            return True
        return False

    def _finish(self, colour: _WaitingColour, depth: Image | None) -> None:
        if self._ready is not None:
            self.replaced += 1
        self._ready = (colour.msg, colour.received, depth)
        self._colour = None

    def take(self, timeout: float) -> Bundle | None:
        """Return the finished bundle, waiting up to timeout seconds; None when there is none.

        A colour image whose wait has ended becomes a bundle here, but it
        never replaces a pair that is ready: it is the newer image and comes
        next.
        """
        end = self._clock() + timeout
        with self._cond:
            while True:
                now = self._clock()
                colour = self._colour
                if self._ready is None and colour is not None and now >= colour.deadline:
                    self._finish(colour, colour.candidate)
                if self._ready is not None or self._closed:
                    item, self._ready = self._ready, None
                    self._woken = False
                    return item
                if self._woken:
                    self._woken = False
                    return None
                remaining = end - now
                if remaining <= 0:
                    return None
                if colour is not None:
                    remaining = min(remaining, colour.deadline - now)
                self._cond.wait(remaining)

    def wake(self) -> None:
        """End the current or the next take() early, so the worker can look at a service job."""
        with self._cond:
            self._woken = True
            self._cond.notify_all()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._colour = None
            self._depth = None
            self._ready = None
            self._cond.notify_all()

    @property
    def closed(self) -> bool:
        return self._closed


class CurrentImage:
    """The newest colour image and the newest depth image, each replaced by the next.

    EmbedCrop answers from these. Nothing older is kept, so a request cannot
    reach a past image.

    With max_pair_gap_ns and pair_wait_s it also keeps the newest colour and
    depth that paired, and the services take depth only from that pair. They
    pair by the bounds of RgbdBuffer: their stamps differ by at most
    max_pair_gap_ns, neither stamp is zero, and the clock, monotonic wall time
    by default, saw them arrive within pair_wait_s of each other. One image
    arrives before the other, so at any moment the newest of each can belong
    to different captures. That pair is the only older input kept.

    clear() forgets all of it. The node calls it when the sim clock goes back,
    as after a restart, because the stamps and the arrival times of the next
    run can match those of the previous one. It does not reach images that a
    read already returned.
    """

    def __init__(
        self,
        max_pair_gap_ns: int | None = None,
        pair_wait_s: float | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (max_pair_gap_ns is None) != (pair_wait_s is None):
            raise ValueError("pairing needs both max_pair_gap_ns and pair_wait_s, or neither")
        self._lock = threading.Lock()
        self._clock = clock
        self._colour: Image | None = None
        self._colour_at = 0.0
        self._depth: Image | None = None
        self._depth_at = 0.0
        self._max_pair_gap_ns = max_pair_gap_ns
        self._pair_wait_s = pair_wait_s
        self._pair: tuple[Image, Image] | None = None

    def put_colour(self, msg: Image) -> None:
        with self._lock:
            self._colour, self._colour_at = msg, self._clock()
            self._match()

    def put_depth(self, msg: Image) -> None:
        with self._lock:
            self._depth, self._depth_at = msg, self._clock()
            self._match()

    def _match(self) -> None:
        if self._max_pair_gap_ns is None or self._colour is None or self._depth is None:
            return
        colour_ns, depth_ns = stamp_ns(self._colour.header), stamp_ns(self._depth.header)
        if colour_ns == 0 or depth_ns == 0 or abs(colour_ns - depth_ns) > self._max_pair_gap_ns:
            return
        if abs(self._colour_at - self._depth_at) <= self._pair_wait_s:
            self._pair = (self._colour, self._depth)

    def clear(self) -> None:
        """Forget every image and the pair, so no read after this call returns, or pairs, one put before it."""
        with self._lock:
            self._colour = self._depth = None
            self._colour_at = self._depth_at = 0.0
            self._pair = None

    def read(self) -> tuple[Image | None, Image | None]:
        """Return the newest of each. They can belong to different captures, so take depth from a pair."""
        with self._lock:
            return self._colour, self._depth

    def read_current(self) -> tuple[Image | None, Image | None]:
        """Return the newest colour image, and the newest depth only when the pair holds these very two.

        Otherwise the depth is None, also when the pair holds this colour image
        but a newer depth has come since, as EmbedCrop takes only the newest of
        each (ADR-0015).
        """
        with self._lock:
            if self._pair is not None and self._pair[0] is self._colour and self._pair[1] is self._depth:
                return self._pair
            return self._colour, None

    def read_pair(self) -> tuple[Image, Image] | None:
        """Return the newest colour and depth that paired, or None when none has."""
        with self._lock:
            return self._pair
