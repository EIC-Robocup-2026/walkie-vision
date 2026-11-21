"""Depth from a rectified stereo pair with Fast-FoundationStereo, without a ROS node.

stereo_depth() takes the left and right images and their camera_infos, runs a
disparity model on a smaller copy of the pair, and returns depth in metres at
that smaller size, with a camera_info scaled to it. The depth is registered to
the left image: same frame, same stamp, same rays. walkie_vision.stereo_depth
is the node around it (ADR-0017).

The model is NVIDIA's ONNX export of the commercial checkpoint
(nvidia/c-fast-foundationstereo, NVIDIA Open Model Agreement), which
fetch-models downloads. Only that file is used: no code from the research
repository. The input format below was measured against that file. Creating a
CUDA session for it takes 26 to 29 s of CPU, and the first run about 1.6 s.
The TensorRT provider is unavailable: the environment has no libnvinfer.so.10.

onnxruntime is imported only in load_model(), so the tests import this module
without it.
"""

from __future__ import annotations

import array
import copy
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header

from .depth import stamp_ns
from .image import calibration_problem, camera_info_problem, image_to_bgr

# The export removed the input normalisation from the graph: its first
# convolution reads the input as it is. The caller normalises 0-255 RGB.
IMAGENET_MEAN_RGB = (123.675, 116.28, 103.53)
IMAGENET_STD_RGB = (58.395, 57.12, 57.375)
PAD_MULTIPLE = 32
"""The input's height and width must be multiples of this."""
# The graph measures disparities up to about 192 px at its input size. Near
# that limit its output scatters by a few px, so a surface just past it can
# come out under 192 px and read as a depth a few percent too far. On
# bus.jpg shifted by known amounts at 1104 x 621, a cut at 184 px kept every
# pixel of a 180 px shift. It let through 2.5 % of a 192 px shift, 0.2 % of
# a 196 px shift and none from 200 to 260 px, against 54 %, 11 % and 3.5 to
# 0.5 % with a cut at 192 px. At scale 0.5 on the ZED, 184 px puts the
# nearest depth at about 0.35 m instead of 0.34 m.
# The graph's 8 refinement iterations are baked in.
MAX_DISPARITY_PX = 184.0
"""The largest disparity kept as depth: 8 px inside the model's range of about 192 px."""
INPUT_NAMES = ("left_image", "right_image")
OUTPUT_NAME = "disparity"
MAX_RECTIFICATION_ERROR_PX = 0.5
"""How far the right projection's focal lengths and centre may be from the left K."""
MIN_BASELINE_M, MAX_BASELINE_M = 0.01, 1.0

DisparityModel = Callable[[np.ndarray, np.ndarray], np.ndarray]
"""Takes the left and right inputs, each 1 x 3 x Hp x Wp float32; returns Hp x Wp float32 disparity in input pixels."""


class StereoRefused(Exception):
    """A stereo pair that cannot give depth. reason is a short counter key."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class Padding:
    top: int
    bottom: int
    left: int
    right: int


@dataclass(frozen=True)
class StereoCalibration:
    """The left camera's intrinsics at full size, and the baseline to the right camera."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    baseline_m: float


@dataclass(frozen=True)
class StereoFrame:
    depth: np.ndarray
    """h_s x w_s float32 metres, NaN where invalid."""
    depth_info: CameraInfo
    """The left camera_info scaled to the depth, with the left image's header."""
    image_info: CameraInfo
    """The left camera_info, with the left image's header."""
    valid_share: float
    """The share of depth pixels that are finite."""
    times_ms: dict[str, float]
    """Keys "prepare", "model" and "depth"."""


def inference_size(width: int, height: int, scale_x: float, scale_y: float) -> tuple[int, int]:
    """Return the (width, height) the model runs at: each side scaled and rounded half up, at least 1."""
    return (max(1, math.floor(width * scale_x + 0.5)), max(1, math.floor(height * scale_y + 0.5)))


def centre_padding(width: int, height: int, multiple: int = PAD_MULTIPLE) -> Padding:
    """Return the padding that makes both sides multiples of multiple, split evenly; the extra row or column goes last."""
    pad_w = math.ceil(width / multiple) * multiple - width
    pad_h = math.ceil(height / multiple) * multiple - height
    return Padding(top=pad_h // 2, bottom=pad_h - pad_h // 2, left=pad_w // 2, right=pad_w - pad_w // 2)


def model_input(bgr: np.ndarray, size: tuple[int, int], padding: Padding) -> np.ndarray:
    """Return a BGR image as the model's input: RGB, resized to size, padded by edge copies, normalised, 1 x 3 x H x W.

    The resize uses area interpolation. It runs before the colour conversion,
    on fewer pixels; the order does not change the result.
    """
    height, width = bgr.shape[:2]
    if (width, height) != tuple(size):
        bgr = cv2.resize(bgr, tuple(size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    padded = cv2.copyMakeBorder(rgb, padding.top, padding.bottom, padding.left, padding.right, cv2.BORDER_REPLICATE)
    x = (padded.astype(np.float32) - np.float32(IMAGENET_MEAN_RGB)) / np.float32(IMAGENET_STD_RGB)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def unpad(disparity: np.ndarray, padding: Padding) -> np.ndarray:
    """Return the part of a padded H x W array that the padding did not add."""
    height, width = disparity.shape
    return disparity[padding.top : height - padding.bottom, padding.left : width - padding.right]


def disparity_to_depth(disparity: np.ndarray, fx: float, baseline_m: float) -> np.ndarray:
    """Return fx * baseline_m / disparity as float32 metres.

    fx is in the disparity's pixels. A pixel is NaN where its disparity is not
    finite, not positive, at or above MAX_DISPARITY_PX, or points past the
    right image's left edge (column - disparity < 0).
    """
    d = np.asarray(disparity, dtype=np.float32)
    columns = np.arange(d.shape[1], dtype=np.float32)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        valid = np.isfinite(d) & (d > 0) & (d < MAX_DISPARITY_PX) & (columns - d >= 0)
        depth = np.float32(fx * baseline_m) / d
    return np.where(valid, depth, np.float32(np.nan)).astype(np.float32)


def read_stereo_calibration(
    left: Image, right: Image, left_info: CameraInfo | None, right_info: CameraInfo | None
) -> StereoCalibration:
    """Check that a pair is rectified to the left camera and return its calibration.

    Raises StereoRefused with reason camera_info, size, frame, calibration or
    baseline.
    """
    for side, image, info in (("left", left, left_info), ("right", right, right_info)):
        if info is None:
            raise StereoRefused("camera_info", f"no {side} camera_info yet")
        problem = camera_info_problem(info, int(image.width), int(image.height))
        if problem is not None:
            raise StereoRefused("camera_info", f"{side}: {problem}")
    assert left_info is not None and right_info is not None
    if (left.width, left.height) != (right.width, right.height):
        raise StereoRefused(
            "size", f"the left image is {left.width}x{left.height} but the right image is {right.width}x{right.height}"
        )
    frame = left.header.frame_id
    if not frame:
        raise StereoRefused("frame", "the left image has no frame_id")
    if left_info.header.frame_id != frame:
        raise StereoRefused("frame", f"the left camera_info is in {left_info.header.frame_id!r}, the left image in {frame!r}")
    if right_info.header.frame_id != right.header.frame_id:
        raise StereoRefused(
            "frame",
            f"the right camera_info is in {right_info.header.frame_id!r}, the right image in {right.header.frame_id!r}",
        )
    for side, info in (("left", left_info), ("right", right_info)):
        problem = calibration_problem(info)
        if problem is not None:
            raise StereoRefused("calibration", f"{side} camera_info: {problem}")
    if float(left_info.p[3]) != 0.0:
        raise StereoRefused("calibration", f"left P[3] is {left_info.p[3]}, not 0: the left camera must be the reference")
    k, p = left_info.k, right_info.p
    fx, fy, cx, cy = float(k[0]), float(k[4]), float(k[2]), float(k[5])
    right_values = {"fx": float(p[0]), "fy": float(p[5]), "cx": float(p[2]), "cy": float(p[6])}
    if not all(math.isfinite(v) for v in right_values.values()):
        raise StereoRefused("calibration", f"the right projection is not finite: {right_values}")
    for name, left_value in (("fx", fx), ("fy", fy), ("cx", cx), ("cy", cy)):
        if abs(right_values[name] - left_value) > MAX_RECTIFICATION_ERROR_PX:
            raise StereoRefused(
                "calibration",
                f"right P {name} is {right_values[name]} but the left K {name} is {left_value}: "
                "the pair is not rectified to one camera",
            )
    if float(p[7]) != 0.0:
        raise StereoRefused("calibration", f"right P[7] is {p[7]}, not 0: the cameras are not side by side")
    baseline = -float(p[3]) / float(p[0])
    if not (math.isfinite(baseline) and MIN_BASELINE_M <= baseline <= MAX_BASELINE_M):
        raise StereoRefused(
            "baseline",
            f"the baseline is {baseline} m, outside [{MIN_BASELINE_M:g}, {MAX_BASELINE_M:g}]; "
            "right P[3] must be -fx * baseline",
        )
    return StereoCalibration(int(left.width), int(left.height), fx, fy, cx, cy, baseline)


def scaled_camera_info(info: CameraInfo, width: int, height: int, header: Header) -> CameraInfo:
    """Return a camera_info for the same camera at another size, by the pixel-centre rule.

    With sx = width / info.width: fx' = sx * fx and cx' = sx * (cx + 0.5) - 0.5,
    and the same for y. walkie_vision.depth checks depth rays by that rule.
    """
    sx, sy = width / int(info.width), height / int(info.height)
    k = [float(v) for v in info.k]
    k[0], k[1], k[2] = sx * k[0], sx * k[1], sx * (k[2] + 0.5) - 0.5
    k[4], k[5] = sy * k[4], sy * (k[5] + 0.5) - 0.5
    p = [float(v) for v in info.p]
    p[0], p[1], p[2], p[3] = sx * p[0], sx * p[1], sx * (p[2] + 0.5) - 0.5, sx * p[3]
    p[5], p[6], p[7] = sy * p[5], sy * (p[6] + 0.5) - 0.5, sy * p[7]
    out = CameraInfo()
    out.header = copy.deepcopy(header)
    out.width, out.height = int(width), int(height)
    out.distortion_model = info.distortion_model
    out.d = list(info.d)
    out.k = k
    out.r = list(info.r)
    out.p = p
    return out


def depth_image(depth: np.ndarray, header: Header) -> Image:
    """Return an H x W float32 array in metres as a little-endian 32FC1 Image."""
    height, width = depth.shape
    data = array.array("B")
    data.frombytes(np.ascontiguousarray(depth, dtype="<f4").tobytes())
    return Image(
        header=copy.deepcopy(header),
        height=height,
        width=width,
        encoding="32FC1",
        is_bigendian=0,
        step=4 * width,
        data=data,
    )


def stereo_depth(
    left: Image,
    right: Image,
    left_info: CameraInfo | None,
    right_info: CameraInfo | None,
    scale_x: float,
    scale_y: float,
    model: DisparityModel,
) -> StereoFrame:
    """Return the depth of a stereo pair at the inference size, registered to the left image.

    Raises StereoRefused for a pair that cannot give depth (see
    read_stereo_calibration(), plus stamp and encoding), and RuntimeError when
    the model returns another shape than its input's.
    """
    calib = read_stereo_calibration(left, right, left_info, right_info)
    assert left_info is not None
    if stamp_ns(left.header) != stamp_ns(right.header):
        raise StereoRefused(
            "stamp", f"the left image is stamped {stamp_ns(left.header)} ns, the right image {stamp_ns(right.header)} ns"
        )
    try:
        left_bgr, right_bgr = image_to_bgr(left), image_to_bgr(right)
    except ValueError as e:
        raise StereoRefused("encoding", str(e)) from e
    size = inference_size(calib.width, calib.height, scale_x, scale_y)
    padding = centre_padding(*size)
    t0 = time.perf_counter()
    left_in, right_in = model_input(left_bgr, size, padding), model_input(right_bgr, size, padding)
    t1 = time.perf_counter()
    disparity = model(left_in, right_in)
    t2 = time.perf_counter()
    if disparity.shape != left_in.shape[2:]:
        raise RuntimeError(f"the model returned shape {disparity.shape} for an input of {left_in.shape[2:]}")
    width_s, height_s = size
    fx_s = calib.fx * (width_s / calib.width)
    depth = disparity_to_depth(unpad(disparity, padding), fx_s, calib.baseline_m)
    t3 = time.perf_counter()
    image_info = copy.deepcopy(left_info)
    image_info.header = copy.deepcopy(left.header)
    return StereoFrame(
        depth=depth,
        depth_info=scaled_camera_info(left_info, width_s, height_s, left.header),
        image_info=image_info,
        valid_share=float(np.isfinite(depth).mean()),
        times_ms={"prepare": (t1 - t0) * 1000.0, "model": (t2 - t1) * 1000.0, "depth": (t3 - t2) * 1000.0},
    )


class OnnxDisparityModel:
    """A DisparityModel on an ONNX Runtime session of the Fast-FoundationStereo export."""

    def __init__(self, session) -> None:
        self._session = session

    @property
    def providers(self) -> list[str]:
        return list(self._session.get_providers())

    def __call__(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        out = self._session.run([OUTPUT_NAME], {INPUT_NAMES[0]: left, INPUT_NAMES[1]: right})[0]
        want = (1, 1, *left.shape[2:])
        if tuple(out.shape) != want:
            raise RuntimeError(f"the model returned shape {tuple(out.shape)}, not {want}")
        return np.asarray(out[0, 0], dtype=np.float32)


def load_model(path: str | Path, provider: str = "CUDAExecutionProvider") -> OnnxDisparityModel:
    """Create a session for the ONNX file at path on provider, and check its inputs and output.

    Raises FileNotFoundError for a missing file, before onnxruntime is
    imported, and RuntimeError when the session runs on another provider or
    the graph has other inputs or outputs.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run `pixi run fetch-models`")
    import onnxruntime as ort

    ort.set_default_logger_severity(3)
    if provider == "CUDAExecutionProvider":
        # The CUDA provider library does not find the CUDA and cuDNN wheels by
        # itself; without this, ONNX Runtime falls back to the CPU provider
        # (see onnxruntime-gpu in pixi.toml).
        ort.preload_dlls()
    options = ort.SessionOptions()
    # The work runs on the GPU. Spinning CPU threads would compete with a
    # robot that already runs at a load average of 16 to 20.
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    providers = [provider] if provider == "CPUExecutionProvider" else [provider, "CPUExecutionProvider"]
    session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
    active = session.get_providers()
    if active[0] != provider:
        raise RuntimeError(
            f"{path.name} runs on {active[0]}, not {provider}; available providers: {ort.get_available_providers()}"
        )
    inputs = session.get_inputs()
    names = tuple(i.name for i in inputs)
    if names != INPUT_NAMES:
        raise RuntimeError(f"{path.name} has inputs {list(names)}, not {list(INPUT_NAMES)}")
    for i in inputs:
        if len(i.shape) != 4:
            raise RuntimeError(f"{path.name}: input {i.name} has shape {i.shape}, not four dimensions")
    outputs = session.get_outputs()
    if not outputs or outputs[0].name != OUTPUT_NAME:
        raise RuntimeError(f"{path.name} has outputs {[o.name for o in outputs]}, not {OUTPUT_NAME!r} first")
    return OnnxDisparityModel(session)
