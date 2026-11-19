"""DetectObjects: the request check, the class gate, the image age and the failure response (ADR-0015).

The callback checks the request with detect_objects_problem() and leaves a job
in the slot of walkie_vision.services.jobs. The worker prepares the classes
when ClassPrepareGate allows it, and answers from an image that
image_age_problem() finds current.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

from sensor_msgs.msg import Image
from walkie_vision_msgs.srv import DetectObjects

from ..messages import stamp_ns
from .jobs import MAX_TIMEOUT_S, bounded_message

MAX_CLASSES = 32
MAX_CLASS_CHARS = 80
MAX_REQUEST_IMAGE_AGE_S = 1.0
"""The largest max_image_age_s a DetectObjects request may ask for."""
CLASS_PREPARE_PERIOD_S = 2.0
"""The shortest time from the end of one on-demand class preparation to the start of the next."""


class ClassPrepareGate:
    """Limits how often DetectObjects may prepare new classes: about 1.5 s with the stream stopped.

    The time runs from the end of a preparation, so the stream gets the whole
    period between two stalls, however long one took. Only the worker uses this,
    so it holds no lock. Monotonic time, so a paused /clock cannot hold it shut.
    """

    def __init__(self, period_s: float = CLASS_PREPARE_PERIOD_S, clock: Callable[[], float] = time.monotonic) -> None:
        self.period_s = period_s
        self._clock = clock
        self._last: float | None = None

    def open(self) -> bool:
        return self._last is None or self._clock() - self._last >= self.period_s

    def mark(self) -> None:
        """Record that a preparation has ended, whether it worked or not."""
        self._last = self._clock()


def detect_objects_problem(request: DetectObjects.Request) -> str | None:
    """Return why a DetectObjects request is refused, or None. The reason never repeats a class."""
    classes = list(request.classes)
    if not classes:
        return "classes is empty"
    if len(classes) > MAX_CLASSES:
        return f"at most {MAX_CLASSES} classes, not {len(classes)}"
    for i, name in enumerate(classes):
        if not name or name != name.strip():
            return f"class {i} is empty or has spaces around it"
        if len(name) > MAX_CLASS_CHARS:
            return f"class {i} has {len(name)} characters, more than {MAX_CLASS_CHARS}"
    if len(set(classes)) != len(classes):
        return "classes has duplicates"
    age = float(request.max_image_age_s)
    if not (math.isfinite(age) and 0.0 < age <= MAX_REQUEST_IMAGE_AGE_S):
        return f"max_image_age_s must be more than 0 and at most {MAX_REQUEST_IMAGE_AGE_S:g}, not {age}"
    timeout = float(request.timeout_s)
    if not (math.isfinite(timeout) and 0.0 < timeout <= MAX_TIMEOUT_S):
        return f"timeout_s must be more than 0 and at most {MAX_TIMEOUT_S:g}, not {timeout}"
    return None


def image_age_problem(image: Image, now_ns: int, max_age_s: float) -> str | None:
    """Return why an image is not current in ROS time, or None.

    A stamp far ahead of the clock counts too, as after a sim restart: its age
    says nothing. Up to max_age_s ahead is allowed, because /clock can lag an
    image.
    """
    age_ns = now_ns - stamp_ns(image.header)
    if abs(age_ns) > round(max_age_s * 1e9):
        return f"the image is {age_ns / 1e9:.3f} s old by the clock; the request allows {max_age_s:.3f} s"
    return None


def detect_objects_failure(code: str, message: str) -> DetectObjects.Response:
    """A DetectObjects failure, with the default empty array."""
    return DetectObjects.Response(success=False, error_code=code, message=bounded_message(message))
