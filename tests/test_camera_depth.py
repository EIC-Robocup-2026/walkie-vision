"""walkie_vision.camera.depth: reading a registered depth raster and its checks, without ROS communication or a GPU."""

from __future__ import annotations

import math

import numpy as np
import pytest
from builtin_interfaces.msg import Time
from sensor_msgs.msg import CameraInfo, Image

from walkie_vision.camera.depth import (
    MAX_RAY_ERROR_PX,
    DepthRefused,
    RegisteredDepth,
    depth_to_metres,
    read_registered_depth,
)
from walkie_vision.camera.image import calibration_problem

FRAME = "zed_head_left_camera_frame_optical"
BASE_MS = 100_000  # Stamps start at 100 s: a zero stamp never pairs.
W, H = 8, 6
K = [100.0, 0.0, 3.5, 0.0, 100.0, 2.5, 0.0, 0.0, 1.0]


def _stamped(ms: float, frame: str = FRAME) -> Image:
    msg = Image()
    ns = round(ms * 1_000_000)
    msg.header.stamp = Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)
    msg.header.frame_id = frame
    return msg


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
