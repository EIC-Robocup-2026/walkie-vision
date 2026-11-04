"""Registered depth for the vision node: pairing, checking and reading (ADR-0014).

RgbdBuffer pairs the newest colour image with the newest depth image. The
stamps may differ by a small gap, and a colour image waits a short wall time
for its depth. Nothing is kept beyond the newest colour, the newest depth and
one finished bundle, so no backlog can grow.

read_registered_depth() accepts a pair only if the depth is already registered
to the colour image: same optical frame, same rays, no distortion left. It
never reads TF. Registering depth to colour is the camera's work (walkie-ros).

Imports no model code, so the tests run it without a GPU.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header

from .image import calibration_problem, camera_info_problem

_DEPTH_FORMATS = {"32FC1": "f4", "16UC1": "u2"}
SUPPORTED_DEPTH_ENCODINGS = tuple(_DEPTH_FORMATS)
_METRES_PER_MILLIMETRE = 0.001

MAX_RAY_ERROR_PX = 1.0
"""How far the depth raster's calibration may move a ray from the colour image's, in depth pixels."""

Bundle = tuple[Image, float, Image | None]
"""A colour image, the time it was received, and its depth or None."""


def stamp_ns(header: Header) -> int:
    """Return a header's stamp in nanoseconds."""
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


class DepthRefused(ValueError):
    """A depth image that cannot be used for geometry. reason is a short counter key."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def depth_to_metres(msg: Image) -> np.ndarray:
    """Return a contiguous H x W float32 copy of a depth image, in metres.

    32FC1 is metres and 16UC1 is millimetres, where 0 means no measurement
    and becomes NaN. step and is_bigendian are honoured, including padding
    that is not a whole number of pixels. Bad values are kept as they are:
    the range check in the geometry step rejects them. Raises DepthRefused
    for another encoding, an empty image, a step shorter than one row, or
    data shorter than step * height.
    """
    encoding = msg.encoding
    if encoding not in _DEPTH_FORMATS:
        raise DepthRefused(
            "depth_encoding", f"unsupported depth encoding {encoding!r}; supported: {', '.join(SUPPORTED_DEPTH_ENCODINGS)}"
        )
    kind = _DEPTH_FORMATS[encoding]
    pixel_bytes = int(kind[1])
    width, height, step = int(msg.width), int(msg.height), int(msg.step)
    if width == 0 or height == 0:
        raise DepthRefused("depth_data", f"the depth image is empty: {width}x{height}")
    if step < width * pixel_bytes:
        raise DepthRefused(
            "depth_data", f"step {step} is shorter than one row of {width} {encoding} pixels ({width * pixel_bytes} bytes)"
        )
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if data.size < step * height:
        raise DepthRefused("depth_data", f"the depth image has {data.size} bytes, but step * height is {step * height}")
    order = ">" if msg.is_bigendian else "<"
    # strides reads rows of any padding; astype makes a native, contiguous copy.
    rows = np.ndarray((height, width), dtype=order + kind, buffer=data, strides=(step, pixel_bytes))
    metres = rows.astype(np.float32)
    if encoding == "16UC1":
        metres *= _METRES_PER_MILLIMETRE
        metres[rows == 0] = np.nan
    return metres


@dataclass(frozen=True)
class RegisteredDepth:
    """A depth raster in metres, in the colour image's optical frame, with its own intrinsics."""

    metres: np.ndarray
    """H x W float32. NaN where nothing was measured."""
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if self.metres.ndim != 2:
            raise ValueError(f"metres must have two dimensions, not {self.metres.ndim}")
        if not all(math.isfinite(v) for v in (self.fx, self.fy, self.cx, self.cy)) or self.fx <= 0 or self.fy <= 0:
            raise ValueError(f"bad intrinsics: fx {self.fx}, fy {self.fy}, cx {self.cx}, cy {self.cy}")


def _axis_ray_error(size: int, colour_size: int, focal: float, centre: float, colour_focal: float, colour_centre: float) -> float:
    """Return the worst ray difference along one axis, in depth pixels.

    The colour calibration is scaled to the raster by the pixel-centre rule:
    f' = s * f and c' = s * (c + 0.5) - 0.5. Rays are compared at both edges.
    """
    scale = size / colour_size
    scaled_focal = scale * colour_focal
    scaled_centre = scale * (colour_centre + 0.5) - 0.5
    return max(
        abs((pixel - centre) / focal - (pixel - scaled_centre) / scaled_focal) * scaled_focal for pixel in (0, size - 1)
    )


def _ray_error(depth_info: CameraInfo, colour_info: CameraInfo) -> float:
    k, colour_k = depth_info.k, colour_info.k
    return max(
        _axis_ray_error(int(depth_info.width), int(colour_info.width), k[0], k[2], colour_k[0], colour_k[2]),
        _axis_ray_error(int(depth_info.height), int(colour_info.height), k[4], k[5], colour_k[4], colour_k[5]),
    )


def read_registered_depth(
    depth: Image, depth_info: CameraInfo | None, colour: Image, colour_info: CameraInfo | None
) -> RegisteredDepth:
    """Check a paired depth image against its colour image and read it.

    The depth and both CameraInfos must be in the colour image's frame. The
    raster must have a CameraInfo of its own size, and that calibration must
    describe the same rays as the colour image's. A lower resolution is fine;
    intrinsics are never made up from the colour image's. Raises DepthRefused.
    """
    frame = colour.header.frame_id
    if depth.header.frame_id != frame:
        raise DepthRefused("depth_frame", f"the depth image is in {depth.header.frame_id!r}, the colour image in {frame!r}")
    if colour_info is None or colour_info.header.frame_id != frame:
        held = None if colour_info is None else colour_info.header.frame_id
        raise DepthRefused("camera_info_frame", f"the colour camera_info is in {held!r}, the colour image in {frame!r}")
    if depth_info is None:
        raise DepthRefused("depth_camera_info", "no depth camera_info yet")
    if depth_info.header.frame_id != frame:
        raise DepthRefused(
            "depth_camera_info", f"the depth camera_info is in {depth_info.header.frame_id!r}, the colour image in {frame!r}"
        )
    size_problem = camera_info_problem(depth_info, int(depth.width), int(depth.height))
    if size_problem is not None:
        raise DepthRefused("depth_camera_info", f"depth: {size_problem}")
    colour_size_problem = camera_info_problem(colour_info, int(colour.width), int(colour.height))
    if colour_size_problem is not None:
        raise DepthRefused("depth_camera_info", f"colour: {colour_size_problem}")
    for name, info in (("depth", depth_info), ("colour", colour_info)):
        problem = calibration_problem(info)
        if problem is not None:
            raise DepthRefused("depth_calibration", f"{name} camera_info: {problem}")
    error = _ray_error(depth_info, colour_info)
    if error > MAX_RAY_ERROR_PX:
        raise DepthRefused(
            "depth_calibration",
            f"the depth camera_info describes other rays than the colour one: {error:.1f} px apart at the edge",
        )
    k = depth_info.k
    return RegisteredDepth(depth_to_metres(depth), float(k[0]), float(k[4]), float(k[2]), float(k[5]))


@dataclass
class _WaitingColour:
    msg: Image
    received: float
    deadline: float
    candidate: Image | None = None
    """A depth just before this image, within the gap. Used if nothing better arrives in time."""


class RgbdBuffer:
    """Pairs colour and depth images, with the mailbox interface of walkie_vision.node.NewestFrame.

    The callbacks pair on arrival. An exact stamp pairs at once. A depth just
    after the colour image, within max_gap_s, also pairs at once, because the
    streams are in order and no nearer one can follow. A depth just before it
    is held, in case an exact or later one arrives. The colour image gets
    wait_s of wall time, and then goes to the worker with the depth it holds,
    or with None.

    Wall time bounds every wait, so a paused or reset sim clock cannot hang
    the worker. For the same reason, a depth that waited longer than wait_s is
    dropped, so a reset clock cannot match it to a new image with an equal
    stamp. Pairing uses up both images, and a zero stamp never pairs, because
    two zero stamps say nothing about the capture.
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
                    return item
                remaining = end - now
                if remaining <= 0:
                    return None
                if colour is not None:
                    remaining = min(remaining, colour.deadline - now)
                self._cond.wait(remaining)

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
