"""walkie_vision.stereo: the input format, the depth, the calibration checks and the ONNX wrapper, without a GPU."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from builtin_interfaces.msg import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header

from walkie_vision.depth import DepthRefused, read_registered_depth
from walkie_vision.models import FFS_ONNX
from walkie_vision.stereo import (
    IMAGENET_MEAN_RGB,
    IMAGENET_STD_RGB,
    Padding,
    StereoRefused,
    centre_padding,
    depth_image,
    disparity_to_depth,
    inference_size,
    load_model,
    model_input,
    nearest_depth_m,
    read_stereo_calibration,
    scaled_camera_info,
    stereo_depth,
    unpad,
)

REPO = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 64, 48
LEFT_FRAME = "zed_head_left_camera_frame_optical"
RIGHT_FRAME = "zed_head_right_camera_frame_optical"
K = [50.0, 0.0, 32.0, 0.0, 50.0, 24.0, 0.0, 0.0, 1.0]
LEFT_P = [50.0, 0.0, 32.0, 0.0, 0.0, 50.0, 24.0, 0.0, 0.0, 0.0, 1.0, 0.0]
RIGHT_P = [50.0, 0.0, 32.0, -5.0, 0.0, 50.0, 24.0, 0.0, 0.0, 0.0, 1.0, 0.0]  # a baseline of 0.1 m
STAMP = Time(sec=1_792_400_000, nanosec=250_000_000)
# At scale 0.5, fx is 25 and depth is 25 * 0.1 / d: a disparity of 2.5 is 1.0 m.
DISPARITY = 2.5


def _header(frame: str = LEFT_FRAME, stamp: Time = STAMP) -> Header:
    return Header(stamp=stamp, frame_id=frame)


def _image(frame: str = LEFT_FRAME, value: int = 40, width: int = WIDTH, height: int = HEIGHT, stamp: Time = STAMP,
           encoding: str = "bgr8") -> Image:
    channels = 2 if encoding == "mono16" else 3
    pixels = np.full((height, width, channels), value, dtype=np.uint8)
    return Image(
        header=_header(frame, stamp),
        height=height,
        width=width,
        encoding=encoding,
        step=width * channels,
        data=pixels.tobytes(),
    )


def _info(frame: str = LEFT_FRAME, p=LEFT_P, width: int = WIDTH, height: int = HEIGHT) -> CameraInfo:
    info = CameraInfo()
    info.header = _header(frame)
    info.width, info.height = width, height
    info.distortion_model = "plumb_bob"
    info.d = [0.0] * 5
    info.k = K
    info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    info.p = list(p)
    return info


def _pair():
    left, right = _image(), _image(RIGHT_FRAME)
    left_info, right_info = _info(), _info(RIGHT_FRAME, RIGHT_P)
    return left, right, left_info, right_info


def _constant_model(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.full(left.shape[2:], DISPARITY, np.float32)


# Sizes and padding.


def test_inference_size_rounds_half_up_per_axis():
    assert inference_size(2208, 1242, 0.5, 0.5) == (1104, 621)
    assert inference_size(2208, 1242, 0.5, 1 / 3) == (1104, 414)
    assert inference_size(5, 5, 0.5, 0.5) == (3, 3)


def test_centre_padding_reaches_a_multiple_of_32_and_puts_the_extra_row_at_the_bottom():
    assert centre_padding(1104, 621) == Padding(top=9, bottom=10, left=8, right=8)
    assert centre_padding(64, 64) == Padding(0, 0, 0, 0)


def test_model_input_is_rgb_normalised_nchw_with_edge_padding():
    # Check 13 cannot catch a wrong input format: the model finds a uniform
    # shift with or without these values. They were checked against a real
    # stereo pair on the robot (ADR-0017); changing them needs that again.
    assert IMAGENET_MEAN_RGB == (123.675, 116.28, 103.53)
    assert IMAGENET_STD_RGB == (58.395, 57.12, 57.375)
    bgr = np.zeros((4, 6, 3), np.uint8)
    bgr[..., 0], bgr[..., 1], bgr[..., 2] = 10, 120, 230  # B, G, R
    bgr[0, 0] = (1, 2, 3)
    x = model_input(bgr, (6, 4), Padding(1, 1, 1, 1))
    assert x.shape == (1, 3, 6, 8) and x.dtype == np.float32 and x.flags["C_CONTIGUOUS"]
    expected = [(230 - IMAGENET_MEAN_RGB[0]) / IMAGENET_STD_RGB[0],
                (120 - IMAGENET_MEAN_RGB[1]) / IMAGENET_STD_RGB[1],
                (10 - IMAGENET_MEAN_RGB[2]) / IMAGENET_STD_RGB[2]]
    np.testing.assert_allclose(x[0, :, 2, 3], expected, rtol=1e-6)
    # The corner of the padding copies the nearest image pixel, (0, 0).
    np.testing.assert_allclose(x[0, :, 0, 0], x[0, :, 1, 1])
    np.testing.assert_allclose(x[0, 0, 1, 1], (3 - IMAGENET_MEAN_RGB[0]) / IMAGENET_STD_RGB[0], rtol=1e-6)


def test_model_input_resizes_with_area_interpolation():
    grey = np.array([[0, 2, 100, 102], [4, 6, 104, 106], [50, 50, 200, 210], [50, 50, 220, 230]], np.uint8)
    bgr = np.repeat(grey[..., None], 3, axis=2)
    x = model_input(bgr, (2, 2), Padding(0, 0, 0, 0))
    pixels = x[0, 0] * IMAGENET_STD_RGB[0] + IMAGENET_MEAN_RGB[0]
    np.testing.assert_allclose(pixels, [[3, 103], [50, 215]], atol=1e-3)


@pytest.mark.parametrize("shape", [(621, 1104), (64, 64)])
def test_unpad_inverts_centre_padding(shape):
    height, width = shape
    original = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    p = centre_padding(width, height)
    padded = np.pad(original, ((p.top, p.bottom), (p.left, p.right)))
    assert padded.shape[0] % 32 == 0 and padded.shape[1] % 32 == 0
    np.testing.assert_array_equal(unpad(padded, p), original)


# Disparity to depth.


def test_disparity_to_depth_is_fx_times_baseline_over_disparity():
    depth = disparity_to_depth(np.full((4, 10), 2.5, np.float32), 25.0, 0.1)
    assert depth.dtype == np.float32
    np.testing.assert_allclose(depth[:, 3:], 1.0, rtol=1e-6)
    assert np.isnan(depth[:, :3]).all()


# 184 to 192 px is inside the model's range, but a surface past the range can
# come out there.
@pytest.mark.parametrize("d", [0.0, -1.0, float("nan"), float("inf"), 184.0, 191.9, 192.0, 250.0])
def test_disparity_to_depth_marks_invalid_disparities_nan(d):
    disparity = np.full((2, 300), d, np.float32)
    assert np.isnan(disparity_to_depth(disparity, 541.0, 0.12)).all()
    disparity[:, 200] = 183.9
    depth = disparity_to_depth(disparity, 541.0, 0.12)
    assert np.isfinite(depth[:, 200]).all()
    assert np.isnan(np.delete(depth, 200, axis=1)).all()


def test_the_nearest_depth_grows_with_the_scale_the_model_runs_at():
    # The ZED's HD2K calibration: fx 1082.901 px and a baseline of 0.120121 m.
    assert nearest_depth_m(1082.901 * 0.5, 0.120121) == pytest.approx(0.354, abs=1e-3)
    assert nearest_depth_m(1082.901, 0.120121) == pytest.approx(0.707, abs=1e-3)
    depth = disparity_to_depth(np.full((1, 300), 183.99, np.float32), 541.45, 0.120121)
    assert np.nanmin(depth) >= nearest_depth_m(541.45, 0.120121)


# The calibration.


def test_read_stereo_calibration_takes_the_baseline_from_the_right_projection():
    left, right, left_info, right_info = _pair()
    right_info.p = [50.0, 0.0, 32.0, -50.0 * 0.12, 0.0, 50.0, 24.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    calib = read_stereo_calibration(left, right, left_info, right_info)
    assert calib.baseline_m == pytest.approx(0.12)
    assert (calib.fx, calib.fy, calib.cx, calib.cy) == (50.0, 50.0, 32.0, 24.0)
    assert (calib.width, calib.height) == (WIDTH, HEIGHT)


def _no_left_info(left, right, left_info, right_info):
    return left, right, None, right_info


def _right_info_size(left, right, left_info, right_info):
    right_info.width = 32
    return left, right, left_info, right_info


def _small_right(left, right, left_info, right_info):
    right_info.width, right_info.height = 32, 24
    return left, _image(RIGHT_FRAME, width=32, height=24), left_info, right_info


def _left_info_frame(left, right, left_info, right_info):
    left_info.header.frame_id = "other"
    return left, right, left_info, right_info


def _set(attr: str, index: int | None, value):
    def change(left, right, left_info, right_info):
        info = right_info if attr.startswith("right") else left_info
        field = attr.split("_", 1)[1]
        if index is None:
            setattr(info, field, value)
        else:
            values = list(getattr(info, field))
            values[index] = value
            setattr(info, field, values)
        return left, right, left_info, right_info

    return change


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (_no_left_info, "camera_info"),
        (_right_info_size, "camera_info"),
        (_small_right, "size"),
        (_left_info_frame, "frame"),
        (_set("left_d", None, [0.1, 0.0, 0.0, 0.0, 0.0]), "calibration"),
        (_set("right_p", 0, 51.0), "calibration"),
        (_set("right_p", 2, 33.0), "calibration"),
        (_set("right_p", 7, 1.0), "calibration"),
        (_set("left_p", 3, -1.0), "calibration"),
        (_set("right_p", 3, 0.0), "baseline"),
        (_set("right_p", 3, 5.0), "baseline"),
        (_set("right_p", 3, -100.0), "baseline"),
    ],
    ids=[
        "no-left-info", "right-info-size", "image-size", "left-info-frame", "distortion", "right-fx", "doffs",
        "right-p7", "left-p3", "zero-baseline", "negative-baseline", "two-metre-baseline",
    ],
)
def test_read_stereo_calibration_refuses_with_a_reason(change, reason):
    args = change(*_pair())
    with pytest.raises(StereoRefused) as raised:
        read_stereo_calibration(*args)
    assert raised.value.reason == reason


def test_scaled_camera_info_follows_the_pixel_centre_rule():
    header = _header(stamp=Time(sec=7))
    info = scaled_camera_info(_info(), 32, 24, header)
    assert (info.width, info.height) == (32, 24)
    np.testing.assert_allclose(info.k, [25.0, 0.0, 15.75, 0.0, 25.0, 11.75, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(info.p, [25.0, 0.0, 15.75, 0.0, 0.0, 25.0, 11.75, 0.0, 0.0, 0.0, 1.0, 0.0])
    assert info.header == header
    assert list(info.d) == [0.0] * 5 and info.distortion_model == "plumb_bob"
    right = scaled_camera_info(_info(RIGHT_FRAME, RIGHT_P), 32, 24, header)
    assert right.p[3] == pytest.approx(-2.5)


@pytest.mark.parametrize("size", [(32, 24), (32, 16)])
def test_scaled_camera_info_passes_the_vision_nodes_ray_check(size):
    width, height = size
    info = scaled_camera_info(_info(), width, height, _header())
    depth = depth_image(np.ones((height, width), np.float32), _header())
    try:
        registered = read_registered_depth(depth, info, _image(), _info())
    except DepthRefused as e:  # pragma: no cover - the assertion below says why
        pytest.fail(f"the vision node refuses the scaled camera_info: {e.reason}: {e}")
    assert registered.fx == 25.0 and registered.cx == 15.75


def test_depth_image_is_little_endian_32fc1():
    depth = np.array([[1.0, np.nan], [2.5, 3.0], [0.5, np.inf]], np.float32)
    msg = depth_image(depth, _header())
    assert (msg.encoding, msg.is_bigendian, msg.width, msg.height, msg.step) == ("32FC1", 0, 2, 3, 8)
    assert msg.header == _header()
    back = np.frombuffer(bytes(msg.data), dtype="<f4").reshape(3, 2)
    np.testing.assert_array_equal(back, depth)


# The whole step.


def test_stereo_depth_returns_depth_at_the_inference_size_with_the_left_header():
    shapes = []

    def model(left, right):
        shapes.extend([left.shape, right.shape])
        return _constant_model(left, right)

    left, right, left_info, right_info = _pair()
    frame = stereo_depth(left, right, left_info, right_info, 0.5, 0.5, model)
    assert frame.depth.shape == (24, 32)
    np.testing.assert_allclose(frame.depth[:, 3:], 1.0, rtol=1e-6)
    assert np.isnan(frame.depth[:, :3]).all()
    assert (frame.depth_info.width, frame.depth_info.height) == (32, 24)
    assert frame.depth_info.header == left.header
    assert frame.image_info.header == left.header and list(frame.image_info.k) == K
    assert shapes == [(1, 3, 32, 32), (1, 3, 32, 32)]
    assert frame.valid_share == pytest.approx(29 / 32)
    assert frame.nearest_m == pytest.approx(25 * 0.1 / 184)
    assert set(frame.times_ms) == {"prepare", "model", "depth"}


def test_stereo_depth_scales_each_axis_on_its_own_and_uses_the_rounded_width_for_fx():
    width = WIDTH + 1
    frame = stereo_depth(
        _image(width=width),
        _image(RIGHT_FRAME, width=width),
        _info(width=width),
        _info(RIGHT_FRAME, RIGHT_P, width=width),
        0.5,
        0.25,
        _constant_model,
    )
    # 65 * 0.5 rounds up to 33 columns; 48 * 0.25 is 12 rows.
    assert frame.depth.shape == (12, 33)
    fx_s = K[0] * 33 / width
    np.testing.assert_allclose(frame.depth[:, 3:], fx_s * 0.1 / DISPARITY, rtol=1e-6)
    assert (frame.depth_info.width, frame.depth_info.height) == (33, 12)
    assert frame.depth_info.k[0] == pytest.approx(fx_s)
    assert frame.depth_info.k[4] == pytest.approx(K[4] * 12 / HEIGHT)


def test_stereo_depth_hands_the_model_left_then_right():
    seen = []

    def model(left, right):
        seen.append((left, right))
        return _constant_model(left, right)

    left, _, left_info, right_info = _pair()
    left, right = _image(value=10), _image(RIGHT_FRAME, value=200)
    stereo_depth(left, right, left_info, right_info, 0.5, 0.5, model)
    expected = model_input(np.full((HEIGHT, WIDTH, 3), 10, np.uint8), (32, 24), centre_padding(32, 24))
    np.testing.assert_array_equal(seen[0][0], expected)
    assert not np.array_equal(seen[0][1], expected)


@pytest.mark.parametrize("problem", ["encoding", "stamp"])
def test_stereo_depth_refuses_an_encoding_or_a_stamp_mismatch(problem):
    calls = []
    left, right, left_info, right_info = _pair()
    if problem == "encoding":
        left, right = _image(encoding="mono16"), _image(RIGHT_FRAME, encoding="mono16")
    else:
        right = _image(RIGHT_FRAME, stamp=Time(sec=STAMP.sec + 1, nanosec=STAMP.nanosec))
    with pytest.raises(StereoRefused) as raised:
        stereo_depth(left, right, left_info, right_info, 0.5, 0.5, lambda a, b: calls.append(1))
    assert raised.value.reason == problem
    assert not calls


def test_stereo_depth_rejects_a_model_output_of_the_wrong_shape():
    left, right, left_info, right_info = _pair()

    def short(left_in, right_in):
        height, width = left_in.shape[2:]
        return np.ones((height - 1, width), np.float32)

    with pytest.raises(RuntimeError, match="shape"):
        stereo_depth(left, right, left_info, right_info, 0.5, 0.5, short)


# The ONNX wrapper, on the CPU provider with a tiny graph.


def test_stereo_imports_onnxruntime_only_when_loading_a_model():
    code = (
        "import sys, walkie_vision.stereo as s\n"
        "assert 'onnxruntime' not in sys.modules\n"
        "try:\n"
        "    s.load_model('/nonexistent/x.onnx')\n"
        "except FileNotFoundError as e:\n"
        "    print(e)\n"
        "print(' '.join(m for m in ('torch', 'ultralytics', 'onnxruntime') if m in sys.modules))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=60)
    first, rest = out.stdout.split("\n", 1)
    assert "fetch-models" in first
    assert rest.strip() == ""


@pytest.mark.parametrize("named", [True, False])
def test_check_13_loads_the_onnx_file_named_on_its_command_line(tmp_path, named):
    """Neither file exists, so the check stops at the file it tried, before any GPU work."""
    candidate, models = tmp_path / "candidate.onnx", tmp_path / "models"
    out = subprocess.run(
        [sys.executable, str(REPO / "checks" / "c13_stereo.py"), *([str(candidate)] if named else [])],
        env=dict(os.environ, WALKIE_VISION_MODELS=str(models)),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode != 0
    tried = candidate if named else models / FFS_ONNX
    assert f"FileNotFoundError: {tried} is missing" in out.stderr


def _tiny_graph(path, *, left_name: str = "left_image", keepdims: int = 1) -> None:
    from onnx import TensorProto, checker, helper, save

    shape = [1, 3, "h", "w"]
    out_shape = [1, 1, "h", "w"] if keepdims else [1, "h", "w"]
    graph = helper.make_graph(
        [
            helper.make_node("Sub", [left_name, "right_image"], ["diff"]),
            helper.make_node("ReduceMean", ["diff"], ["disparity"], axes=[1], keepdims=keepdims),
        ],
        "tiny_disparity",
        [
            helper.make_tensor_value_info(left_name, TensorProto.FLOAT, shape),
            helper.make_tensor_value_info("right_image", TensorProto.FLOAT, shape),
        ],
        [helper.make_tensor_value_info("disparity", TensorProto.FLOAT, out_shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    checker.check_model(model)
    save(model, str(path))


def test_load_model_runs_a_tiny_graph_on_the_cpu_provider(tmp_path):
    path = tmp_path / "tiny.onnx"
    _tiny_graph(path)
    model = load_model(path, provider="CPUExecutionProvider")
    assert model.providers[0] == "CPUExecutionProvider"
    rng = np.random.default_rng(0)
    left = rng.standard_normal((1, 3, 8, 12)).astype(np.float32)
    right = rng.standard_normal((1, 3, 8, 12)).astype(np.float32)
    out = model(left, right)
    assert out.shape == (8, 12) and out.dtype == np.float32
    np.testing.assert_allclose(out, (left - right)[0].mean(axis=0), rtol=1e-5, atol=1e-6)


def test_load_model_refuses_a_graph_with_other_input_names(tmp_path):
    path = tmp_path / "left.onnx"
    _tiny_graph(path, left_name="left")
    with pytest.raises(RuntimeError, match="left_image"):
        load_model(path, provider="CPUExecutionProvider")


def test_the_onnx_model_rejects_an_output_of_the_wrong_rank(tmp_path):
    path = tmp_path / "rank3.onnx"
    _tiny_graph(path, keepdims=0)
    model = load_model(path, provider="CPUExecutionProvider")
    left = np.zeros((1, 3, 8, 12), np.float32)
    with pytest.raises(RuntimeError, match="shape"):
        model(left, left)
