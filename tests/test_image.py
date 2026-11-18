"""Image messages to BGR arrays, and the camera_info check."""

import numpy as np
import pytest
from sensor_msgs.msg import CameraInfo, Image

from walkie_vision.image import SUPPORTED_ENCODINGS, camera_info_problem, image_to_bgr

HEIGHT, WIDTH = 4, 5


def _bgr() -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)


def _pixels(bgr: np.ndarray, encoding: str) -> np.ndarray:
    alpha = np.full(bgr.shape[:2] + (1,), 255, dtype=np.uint8)
    return {
        "bgr8": bgr,
        "rgb8": bgr[:, :, ::-1],
        "bgra8": np.concatenate([bgr, alpha], axis=2),
        "rgba8": np.concatenate([bgr[:, :, ::-1], alpha], axis=2),
    }[encoding]


def _message(pixels: np.ndarray, encoding: str, padding: int = 0) -> Image:
    height, width, channels = pixels.shape
    rows = np.zeros((height, width * channels + padding), dtype=np.uint8)
    rows[:, : width * channels] = pixels.reshape(height, -1)
    rows[:, width * channels :] = 77  # padding bytes must not reach the image
    msg = Image()
    msg.height, msg.width = height, width
    msg.encoding = encoding
    msg.step = width * channels + padding
    msg.data = rows.tobytes()
    return msg


@pytest.mark.parametrize("encoding", SUPPORTED_ENCODINGS)
@pytest.mark.parametrize("padding", [0, 3])
def test_every_encoding_gives_the_same_bgr_image(encoding, padding):
    bgr = _bgr()
    out = image_to_bgr(_message(_pixels(bgr, encoding), encoding, padding))
    assert out.dtype == np.uint8 and out.shape == (HEIGHT, WIDTH, 3)
    assert out.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(out, bgr)


def test_the_result_is_a_copy():
    msg = _message(_bgr(), "bgr8")
    out = image_to_bgr(msg)
    out[:] = 0
    assert image_to_bgr(msg).any()


def test_an_unsupported_encoding_is_named():
    msg = _message(_bgr(), "bgr8")
    msg.encoding = "mono8"
    with pytest.raises(ValueError, match="mono8.*bgra8"):
        image_to_bgr(msg)


def test_short_data_raises():
    msg = _message(_bgr(), "bgr8")
    msg.data = bytes(msg.data)[:-1]
    with pytest.raises(ValueError, match="bytes"):
        image_to_bgr(msg)


def test_a_short_step_raises():
    msg = _message(_bgr(), "bgr8")
    msg.step = WIDTH * 3 - 1
    with pytest.raises(ValueError, match="step"):
        image_to_bgr(msg)


@pytest.mark.parametrize("size", [(0, WIDTH), (HEIGHT, 0)])
def test_a_zero_size_raises(size):
    msg = _message(_bgr(), "bgr8")
    msg.height, msg.width = size
    with pytest.raises(ValueError, match="empty"):
        image_to_bgr(msg)


def _info(width: int = 640, height: int = 360, fx: float = 224.0, fy: float = 224.0) -> CameraInfo:
    info = CameraInfo()
    info.width, info.height = width, height
    info.k = [fx, 0.0, width / 2, 0.0, fy, height / 2, 0.0, 0.0, 1.0]
    return info


def test_camera_info_problem_none_when_it_fits():
    assert camera_info_problem(_info(), 640, 360) is None


def test_camera_info_problem_reasons():
    assert camera_info_problem(None, 640, 360) == "no camera_info yet"
    assert camera_info_problem(_info(), 2208, 1242) == "camera_info is 640x360 but the image is 2208x1242"
    no_focal = "camera_info has no focal length (k[0] or k[4] is 0)"
    assert camera_info_problem(_info(fx=0.0), 640, 360) == no_focal
    assert camera_info_problem(_info(fy=0.0), 640, 360) == no_focal
