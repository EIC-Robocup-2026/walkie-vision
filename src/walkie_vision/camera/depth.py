"""Registered depth for the vision node: reading and checking a depth image (ADR-0014).

read_registered_depth() accepts a pair only if the depth is already registered
to the colour image: same optical frame, same rays, no distortion left. It
never reads TF. Registering depth to colour is the camera's work (walkie-ros).

Imports no model code, so the tests run it without a GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sensor_msgs.msg import CameraInfo, Image

from .image import calibration_problem, camera_info_problem

_DEPTH_FORMATS = {"32FC1": "f4", "16UC1": "u2"}
SUPPORTED_DEPTH_ENCODINGS = tuple(_DEPTH_FORMATS)
_MILLIMETRES_PER_METRE = 1000.0

MAX_RAY_ERROR_PX = 1.0
"""How far the depth raster's calibration may move a ray from the colour image's, in depth pixels."""


class DepthRefused(ValueError):
    """A depth image that cannot be used for geometry. reason is a short counter key."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def depth_to_metres(msg: Image) -> np.ndarray:
    """Return a contiguous H x W float32 copy of a depth image, in metres.

    32FC1 is metres and 16UC1 is millimetres, where 0 means no measurement
    and becomes NaN. n mm gives float32(n / 1000), the same value as a 32FC1
    image of that depth, so the two agree at a range limit. step and is_bigendian are honoured, including padding
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
    if encoding == "32FC1":
        return rows.astype(np.float32)
    # Divide in float64, then cast: n mm becomes float32(n / 1000), the value a
    # 32FC1 image of the same depth carries, so 3000 mm is exactly 3.0. Multiplying
    # by 0.001 in float32 gives 3.0000002, outside an inclusive 3 m limit.
    metres = np.divide(rows, _MILLIMETRES_PER_METRE, dtype=np.float64).astype(np.float32)
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
