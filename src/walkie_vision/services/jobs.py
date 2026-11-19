"""Bounded service work for the vision node (ADR-0015).

The worker runs every model call. A service callback only checks its request
and tries to leave one job for the worker. One slot holds that job, for all the
services together, so a second request gets BUSY at once and nothing queues.
Deadlines use monotonic time. Image age uses ROS time. A paused /clock can make
an image look young, and it cannot make a request hang.

This module holds the job, the slot, the error codes and the bound on a failure
message. Each service's checks and responses are in its own module of
walkie_vision.services. The newest input is in walkie_vision.camera.mailboxes.
The node holds the threads and the ROS objects.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

INVALID = "INVALID"
NO_IMAGE = "NO_IMAGE"
STALE_IMAGE = "STALE_IMAGE"
BUSY = "BUSY"
TIMEOUT = "TIMEOUT"
DISABLED = "DISABLED"
MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
INFERENCE_FAILED = "INFERENCE_FAILED"

MAX_TIMEOUT_S = 5.0
MAX_MESSAGE_CHARS = 256
"""The longest failure message of a service response."""


class ServiceJob:
    """One admitted service request, from its callback to the worker and back.

    The job gets one answer, and only before its deadline. A lock decides
    between the worker's finish() and wait() ending first. An answer after
    that is discarded. finish() reads the clock under the lock, so an answer
    after the deadline is discarded too, even when it gets the lock before the
    waiting callback does.
    """

    def __init__(
        self,
        request: Any,
        deadline: float,
        fail: Callable[[str, str], Any],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.request = request
        self.deadline = deadline
        """When the callback stops waiting, on clock: time.monotonic() unless a test gives another."""
        self.fail = fail
        """Builds this service's failure response from an error code and a message."""
        self._clock = clock
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._response: Any = None
        self._given_up = False

    def remaining(self) -> float:
        """The seconds left until the deadline. Negative after it."""
        return self.deadline - self._clock()

    @property
    def given_up(self) -> bool:
        with self._lock:
            return self._given_up

    def finish(self, response: Any) -> bool:
        """Answer the job. False when it has an answer already, nobody waits for one, or the deadline has passed."""
        with self._lock:
            if self._given_up or self._response is not None or self.remaining() <= 0:
                return False
            self._response = response
        self._done.set()
        return True

    def wait(self) -> Any:
        """Return the answer, waiting up to the deadline; None when there is none, and then no answer is taken."""
        self._done.wait(max(0.0, self.remaining()))
        with self._lock:
            if self._response is None:
                self._given_up = True
            return self._response


class JobSlot:
    """The one service job of the node.

    A job holds the slot from admit() until release(). The worker calls
    release() after the job and the newest frame after it, so a stream of
    requests cannot stop the stream of images. A job that a timed-out callback
    leaves behind holds the slot until the worker comes back from it.
    """

    def __init__(self, wake: Callable[[], None]) -> None:
        self._wake = wake
        self._lock = threading.Lock()
        self._job: ServiceJob | None = None
        self._claimed = False
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def admit(self, job: ServiceJob) -> bool:
        """Leave a job for the worker and wake it. False when the slot is taken or closed."""
        with self._lock:
            if self._closed or self._job is not None:
                return False
            self._job = job
            self._claimed = False
        self._wake()
        return True

    def claim(self) -> ServiceJob | None:
        """Return the job the worker has not taken yet, or None."""
        with self._lock:
            if self._job is None or self._claimed:
                return None
            self._claimed = True
            return self._job

    def withdraw(self, job: ServiceJob) -> None:
        """Free the slot for a callback that gave up, unless the worker already took the job."""
        with self._lock:
            if self._job is job and not self._claimed:
                self._job = None

    def release(self, job: ServiceJob) -> None:
        """Free the slot once the worker is done with the job and with the frame after it."""
        with self._lock:
            if self._job is job:
                self._job = None

    def close(self) -> None:
        """Refuse new jobs, and answer the one that holds the slot.

        A job past its deadline takes no answer, as from the worker. Its
        callback answers TIMEOUT.
        """
        with self._lock:
            self._closed = True
            job = self._job
        if job is not None:
            job.finish(job.fail(MODEL_UNAVAILABLE, "the node is shutting down"))


def bounded_message(text: str) -> str:
    """Cut a failure message down to MAX_MESSAGE_CHARS."""
    return text if len(text) <= MAX_MESSAGE_CHARS else text[: MAX_MESSAGE_CHARS - 3] + "..."
