"""Bounded service work for the vision node (ADR-0015).

The worker runs every model call. A service callback only checks its request
and tries to leave one job for the worker. One slot holds that job, for all the
services together, so a second request gets BUSY at once and nothing queues.
Deadlines use monotonic time. Image age uses ROS time. A paused /clock can make
an image look young, and it cannot make a request hang.

This module holds the job, the slot, the newest input and the checks. The node
holds the threads and the ROS objects.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from sensor_msgs.msg import Image
from walkie_vision_msgs.srv import DetectObjects, EmbedCrop

from . import convert, crop, embeddings
from .depth import RegisteredDepth, depth_cover, frame_laplacian, stamp_ns, view_quality
from .image import resize_mask
from .providers import Embedder

INVALID = "INVALID"
NO_IMAGE = "NO_IMAGE"
STALE_IMAGE = "STALE_IMAGE"
BUSY = "BUSY"
TIMEOUT = "TIMEOUT"
DISABLED = "DISABLED"
MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
INFERENCE_FAILED = "INFERENCE_FAILED"

MAX_TEXTS = 16
MAX_TEXT_CHARS = 512
EMBED_TEXT_DEADLINE_S = 2.0
"""EmbedText's request has no timeout field. The caller allows 5 s."""
MAX_TIMEOUT_S = 5.0
MAX_CLASSES = 32
MAX_CLASS_CHARS = 80
MAX_REQUEST_IMAGE_AGE_S = 1.0
"""The largest max_image_age_s a DetectObjects request may ask for."""
CLASS_PREPARE_PERIOD_S = 2.0
"""The shortest time from the end of one on-demand class preparation to the start of the next."""
MAX_STAMP_GAP_NS = 5_000_000
"""The largest difference between a request's stamp and the current image's."""
MAX_IMAGE_AGE_NS = 300_000_000
"""The oldest current image a crop may come from, in ROS time."""
MAX_FUTURE_NS = 5_000_000
"""How far ahead of ROS time a request's stamp may be."""


class ServiceJob:
    """One admitted service request, from its callback to the worker and back.

    The job gets one answer. A lock decides between the worker's finish() and
    wait() ending first. An answer after that is discarded.
    """

    def __init__(self, request: Any, deadline: float, fail: Callable[[str, str], Any]) -> None:
        self.request = request
        self.deadline = deadline
        """When the callback stops waiting, on the time.monotonic() clock."""
        self.fail = fail
        """Builds this service's failure response from an error code and a message."""
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._response: Any = None
        self._given_up = False

    def remaining(self) -> float:
        """The seconds left until the deadline. Negative after it."""
        return self.deadline - time.monotonic()

    @property
    def given_up(self) -> bool:
        with self._lock:
            return self._given_up

    def finish(self, response: Any) -> bool:
        """Answer the job. False when it has an answer already, or nobody waits for one."""
        with self._lock:
            if self._given_up or self._response is not None:
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
        """Refuse new jobs, and answer the one that holds the slot."""
        with self._lock:
            self._closed = True
            job = self._job
        if job is not None:
            job.finish(job.fail(MODEL_UNAVAILABLE, "the node is shutting down"))


class CurrentImage:
    """The newest colour image and the newest depth image, each replaced by the next.

    EmbedCrop answers from these. Nothing older is kept, so a request cannot
    reach a past image.

    With max_pair_gap_ns it also keeps the newest colour and depth whose stamps
    match, for DetectObjects. One arrives before the other, so at any moment the
    newest of each can belong to different captures. That pair is the only older
    input kept.
    """

    def __init__(self, max_pair_gap_ns: int | None = None) -> None:
        self._lock = threading.Lock()
        self._colour: Image | None = None
        self._depth: Image | None = None
        self._max_pair_gap_ns = max_pair_gap_ns
        self._pair: tuple[Image, Image] | None = None

    def put_colour(self, msg: Image) -> None:
        with self._lock:
            self._colour = msg
            self._match()

    def put_depth(self, msg: Image) -> None:
        with self._lock:
            self._depth = msg
            self._match()

    def _match(self) -> None:
        if self._max_pair_gap_ns is None or self._colour is None or self._depth is None:
            return
        if abs(stamp_ns(self._colour.header) - stamp_ns(self._depth.header)) <= self._max_pair_gap_ns:
            self._pair = (self._colour, self._depth)

    def read(self) -> tuple[Image | None, Image | None]:
        with self._lock:
            return self._colour, self._depth

    def read_pair(self) -> tuple[Image, Image] | None:
        """Return the newest colour and depth with matching stamps, or None when none has matched."""
        with self._lock:
            return self._pair


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


def embed_text_problem(texts: Sequence[str]) -> str | None:
    """Return why an EmbedText request is refused, or None. The reason never repeats a text."""
    if not texts:
        return "texts is empty"
    if len(texts) > MAX_TEXTS:
        return f"at most {MAX_TEXTS} texts, not {len(texts)}"
    for i, text in enumerate(texts):
        if not text.strip():
            return f"text {i} is blank"
        if len(text) > MAX_TEXT_CHARS:
            return f"text {i} has {len(text)} characters, more than {MAX_TEXT_CHARS}"
    return None


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


def embed_crop_problem(request: EmbedCrop.Request) -> str | None:
    """Return why an EmbedCrop request is refused before it sees an image, or None."""
    if not request.header.frame_id:
        return "header.frame_id is empty"
    if stamp_ns(request.header) == 0:
        return "header.stamp is zero"
    width, height = int(request.image_width), int(request.image_height)
    if width == 0 or height == 0:
        return f"the image size must be positive, not {width}x{height}"
    timeout = float(request.timeout_s)
    if not (math.isfinite(timeout) and 0.0 < timeout <= MAX_TIMEOUT_S):
        return f"timeout_s must be more than 0 and at most {MAX_TIMEOUT_S:g}, not {timeout}"
    bbox = request.bbox
    values = (bbox.center.position.x, bbox.center.position.y, bbox.center.theta, bbox.size_x, bbox.size_y)
    if not all(math.isfinite(v) for v in values):
        return "bbox has a value that is not finite"
    if bbox.center.theta != 0.0:
        return f"bbox.center.theta must be 0, not {bbox.center.theta}"
    if bbox.size_x <= 0.0 or bbox.size_y <= 0.0:
        return f"bbox must have a positive area, not {bbox.size_x} by {bbox.size_y}"
    if crop.intersect_rects(crop.outer_rect(convert.bbox_to_box(bbox)), (0, 0, width, height)) is None:
        return f"bbox has no pixels in a {width}x{height} image"
    return None


def crop_image_problem(request: EmbedCrop.Request, image: Image, now_ns: int) -> str | None:
    """Return why the current image cannot answer a request, or None.

    The box belongs to the image the request names. On any other view it would
    cover something else, so this never widens to a near match.
    """
    if request.header.frame_id != image.header.frame_id:
        return f"the request is in frame {request.header.frame_id!r}, the current image in {image.header.frame_id!r}"
    if (request.image_width, request.image_height) != (image.width, image.height):
        return (
            f"the request is for a {request.image_width}x{request.image_height} image, "
            f"the current image is {image.width}x{image.height}"
        )
    request_ns, image_ns = stamp_ns(request.header), stamp_ns(image.header)
    if abs(request_ns - image_ns) > MAX_STAMP_GAP_NS:
        return f"the request's stamp is {(request_ns - image_ns) / 1e6:.1f} ms from the current image's"
    if now_ns - image_ns > MAX_IMAGE_AGE_NS:
        return f"the current image is {(now_ns - image_ns) / 1e6:.0f} ms old"
    if request_ns - now_ns > MAX_FUTURE_NS:
        return f"the request's stamp is {(request_ns - now_ns) / 1e6:.1f} ms ahead of the clock"
    return None


def rectangle_quality(
    image: np.ndarray, box: np.ndarray, depth: RegisteredDepth | None, depth_min_m: float, depth_max_m: float
) -> float:
    """Score a requested box as a view, with the rectangle as an approximate mask.

    The same four terms as a detection. The depth term is 0 without depth.
    """
    height, width = image.shape[:2]
    x1, y1, x2, y2 = crop.intersect_rects(crop.outer_rect(box), (0, 0, width, height)) or (0, 0, 0, 0)
    mask = np.zeros((height, width), dtype=bool)
    mask[y1:y2, x1:x2] = True
    cover = 0.0
    if depth is not None:
        cover = depth_cover(resize_mask(mask, *depth.metres.shape), depth.metres, depth_min_m, depth_max_m)
    return view_quality(mask, box, frame_laplacian(image), cover)


def embed_crop_answer(
    embedder: Embedder,
    image: np.ndarray,
    header: Any,
    box: np.ndarray,
    depth: RegisteredDepth | None,
    depth_min_m: float,
    depth_max_m: float,
) -> EmbedCrop.Response:
    """Crop and embed one box of one image. Raises when the box, the model or its vector fails."""
    pixels = embeddings.padded_crop(image, box)
    vectors = np.asarray(embedder.embed_images([pixels]))
    if vectors.shape != (1, embedder.dim):
        raise ValueError(f"the embedder returned shape {vectors.shape} for one crop of dim {embedder.dim}")
    return convert.fill_embed_crop_response(
        EmbedCrop.Response(),
        header,
        embeddings.encode_crop(pixels),
        embedder,
        vectors[0],
        rectangle_quality(image, box, depth, depth_min_m, depth_max_m),
    )
