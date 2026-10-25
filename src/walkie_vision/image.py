"""The camera image and its calibration, as the vision node reads them.

image_to_bgr() turns a sensor_msgs/Image into the BGR array every provider
takes (walkie_vision.providers). camera_info_problem() says whether a
CameraInfo belongs to an image of that size.
"""

from __future__ import annotations

import cv2
import numpy as np
from sensor_msgs.msg import CameraInfo, Image

# The encodings the node accepts, with the number of bytes per pixel. The
# ZED wrapper publishes bgra8; the MuJoCo sim publishes rgb8 or bgra8.
_CHANNELS = {"bgra8": 4, "bgr8": 3, "rgba8": 4, "rgb8": 3}
SUPPORTED_ENCODINGS = tuple(_CHANNELS)


def image_to_bgr(msg: Image) -> np.ndarray:
    """Return a contiguous H x W x 3 uint8 BGR copy of an image message.

    Rows may be padded: step is honoured. Raises ValueError for an
    encoding that is not in SUPPORTED_ENCODINGS, a zero width or height, a
    step shorter than one row, or data shorter than step * height.
    """
    encoding = msg.encoding
    if encoding not in _CHANNELS:
        raise ValueError(f"unsupported image encoding {encoding!r}; supported: {', '.join(SUPPORTED_ENCODINGS)}")
    channels = _CHANNELS[encoding]
    width, height, step = int(msg.width), int(msg.height), int(msg.step)
    if width == 0 or height == 0:
        raise ValueError(f"the image is empty: {width}x{height}")
    if step < width * channels:
        raise ValueError(f"step {step} is shorter than one row of {width} {encoding} pixels ({width * channels} bytes)")
    # rclpy gives data as an array.array of bytes, which numpy reads without a copy.
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if data.size < step * height:
        raise ValueError(f"the image has {data.size} bytes, but step * height is {step * height}")
    pixels = data[: step * height].reshape(height, step)[:, : width * channels].reshape(height, width, channels)
    if encoding == "bgr8":
        return pixels.copy()
    code = {"bgra8": cv2.COLOR_BGRA2BGR, "rgba8": cv2.COLOR_RGBA2BGR, "rgb8": cv2.COLOR_RGB2BGR}[encoding]
    # cv2 needs contiguous input; a padded image's rows are not.
    return np.ascontiguousarray(cv2.cvtColor(np.ascontiguousarray(pixels), code))


def camera_info_problem(info: CameraInfo | None, width: int, height: int) -> str | None:
    """Return why a camera_info does not fit an image of this size, or None when it fits."""
    if info is None:
        return "no camera_info yet"
    if int(info.width) != width or int(info.height) != height:
        return f"camera_info is {info.width}x{info.height} but the image is {width}x{height}"
    if float(info.k[0]) == 0.0 or float(info.k[4]) == 0.0:
        return "camera_info has no focal length (k[0] or k[4] is 0)"
    return None
