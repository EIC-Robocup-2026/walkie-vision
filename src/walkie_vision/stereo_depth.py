"""The experimental stereo depth node: Fast-FoundationStereo on the ZED's left and right images (ADR-0017).

The node pairs the newest left and right images that carry the same stamp,
runs the disparity model on them (walkie_vision.stereo), and publishes, in
this order, each with the left image's header:

- camera_info_topic: the left camera_info.
- depth_camera_info_topic: the left camera_info scaled to the depth.
- depth_topic: 32FC1 depth in metres, NaN where invalid, at the inference
  size (1104 x 621 at the default scale of 0.5).
- image_topic: the left image itself, unchanged.

The vision node reads those four topics with task:=memory_ffs. It pairs depth
with colour within at most 0.1 s of wall time (ADR-0014), and inference takes
about 0.8 s, so it cannot pair this depth with the camera's own stream. The
relayed image carries the depth's exact stamp and arrives just after it, so
the two pair at once.

Inference runs inline in the subscription callback, on rclpy's one thread.
Pairs that arrive meanwhile are dropped by the keep-last-1 inputs, so no
backlog grows. Started by hand while it is experimental:
`pixi run stereo-depth`. The ZED publishes the right image only with
video.publish_left_right: true.
"""

from __future__ import annotations

import math
import os
import signal
import statistics
import sys
import time
import traceback
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.clock import Clock, ClockType
from rclpy.exceptions import NameValidationException
from rclpy.executors import ExternalShutdownException
from rclpy.logging import get_logger
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.validate_full_topic_name import validate_full_topic_name
from sensor_msgs.msg import CameraInfo, Image

from .depth import stamp_ns
from .models import FFS_ONNX, models_dir
from .stereo import DisparityModel, StereoRefused, depth_image, load_model, stereo_depth

NODE_NAME = "walkie_vision_stereo_depth"
EXIT_FATAL = 1
"""A failure while running, such as MAX_CONSECUTIVE_FAILURES model errors in a row."""
EXIT_STARTUP = 2
"""A failure before spinning: a bad parameter or a model that did not load."""

# A 2208 x 1242 bgra8 image is 11 MB. With keep-last-1 best-effort inputs and
# inline inference, Python takes and deserialises only the newest images
# after each inference, not 30 images a second.
INPUT_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)
# Depth 1, so that no subscriber holds old frames. Reliable, so that a
# reliable recorder still matches.
OUTPUT_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)
REPORT_PERIOD_S = 10.0
AGE_WARN_S = 1.5
"""The median age of a published depth above which the report warns.

One pair took about 0.85 s on the robot (ADR-0017), so a working node
publishes depth about 0.9 s old. The warning is for a node that is clearly
slower than that, not for its normal age.
"""
MAX_CONSECUTIVE_FAILURES = 3
_TOPICS = (
    "left_image_topic",
    "left_camera_info_topic",
    "right_image_topic",
    "right_camera_info_topic",
    "image_topic",
    "camera_info_topic",
    "depth_topic",
    "depth_camera_info_topic",
)
_STAGES = ("prepare", "model", "depth")


@dataclass(frozen=True)
class StereoConfig:
    """The node's parameters. Every field is a read-only ROS parameter of the same name."""

    # The rgb topic carries the same pixels, stamp, frame and camera_info as
    # left/color/rect, and it is the vision node's default image_topic.
    left_image_topic: str = "/zed_head/zed_node/rgb/color/rect/image"
    left_camera_info_topic: str = "/zed_head/zed_node/rgb/color/rect/camera_info"
    right_image_topic: str = "/zed_head/zed_node/right/color/rect/image"
    right_camera_info_topic: str = "/zed_head/zed_node/right/color/rect/camera_info"
    image_topic: str = "/walkie_vision/stereo/image"
    camera_info_topic: str = "/walkie_vision/stereo/camera_info"
    depth_topic: str = "/walkie_vision/stereo/depth"
    depth_camera_info_topic: str = "/walkie_vision/stereo/depth/camera_info"
    model: str = FFS_ONNX
    """A path under models/, or an absolute path."""
    scale_x: float = 0.5
    scale_y: float = 0.5
    max_rate_hz: float = 0.0
    """The most pairs per second to run; 0 runs every pair the node can take."""

    def __post_init__(self) -> None:
        seen: dict[str, str] = {}
        for name in _TOPICS:
            topic = getattr(self, name)
            try:
                validate_full_topic_name(topic)
            except NameValidationException as e:
                raise ValueError(f"{name} must be an absolute, valid topic name, not {topic!r}: {e}") from e
            if topic in seen:
                raise ValueError(f"{seen[topic]} and {name} are both {topic!r}; the topics must be distinct")
            seen[topic] = name
        if not self.model.endswith(".onnx"):
            raise ValueError(f"model must name an .onnx file, not {self.model!r}")
        for name in ("scale_x", "scale_y"):
            value = getattr(self, name)
            if not (math.isfinite(value) and 0.0 < value <= 1.0):
                raise ValueError(f"{name} must be in (0, 1], not {value}")
        if not (math.isfinite(self.max_rate_hz) and self.max_rate_hz >= 0.0):
            raise ValueError(f"max_rate_hz must be finite and not negative, not {self.max_rate_hz}")


def model_path(model: str) -> Path:
    """Return the model file: an absolute path as it is, else a path under models/."""
    path = Path(model).expanduser()
    return path if path.is_absolute() else models_dir() / model


class StereoPairer:
    """Pairs the newest left and right images that carry the same stamp.

    Each side holds at most one image. An image pairs with the other side's
    when their stamps are equal. An older image on the other side is dropped,
    and an image older than the other side's is dropped on arrival. A zero
    stamp never pairs. Every dropped image is counted as unmatched.
    """

    def __init__(self, counts: Counter | None = None) -> None:
        self.counts = counts if counts is not None else Counter()
        self._held: dict[str, Image | None] = {"left": None, "right": None}

    def offer_left(self, msg: Image) -> tuple[Image, Image] | None:
        return self._offer("left", msg)

    def offer_right(self, msg: Image) -> tuple[Image, Image] | None:
        return self._offer("right", msg)

    def _offer(self, side: str, msg: Image) -> tuple[Image, Image] | None:
        stamp = stamp_ns(msg.header)
        if stamp == 0:
            self.counts["zero_stamp"] += 1
            return None
        other_side = "right" if side == "left" else "left"
        other = self._held[other_side]
        if other is not None:
            other_stamp = stamp_ns(other.header)
            if other_stamp == stamp:
                self._held = {"left": None, "right": None}
                return (msg, other) if side == "left" else (other, msg)
            if other_stamp < stamp:
                self._held[other_side] = None
                self.counts["unmatched"] += 1
            else:
                self.counts["unmatched"] += 1
                return None
        if self._held[side] is not None:
            self.counts["unmatched"] += 1
        self._held[side] = msg
        return None


def _median(values: Sequence[float]) -> float:
    return statistics.median(values) if values else float("nan")


def report_lines(
    delta: Mapping[str, int],
    times_ms: Mapping[str, Sequence[float]],
    ages_s: Sequence[float],
    valid: Sequence[float],
    cfg: StereoConfig,
) -> list[tuple[str, str]]:
    """Return the periodic report as (level, text) pairs; level is "info" or "warning"."""
    left, right, published = delta.get("left", 0), delta.get("right", 0), delta.get("published", 0)
    unmatched, rate_skipped, failed = delta.get("unmatched", 0), delta.get("rate_skipped", 0), delta.get("failed", 0)
    refusals = {k.removeprefix("refused_"): n for k, n in sorted(delta.items()) if k.startswith("refused_") and n}
    refused = sum(refusals.values())
    breakdown = f" ({', '.join(f'{reason} {n}' for reason, n in refusals.items())})" if refused else ""
    text = (
        f"last {REPORT_PERIOD_S:g} s: left {left} right {right} published {published} "
        f"unmatched {unmatched} zero_stamp {delta.get('zero_stamp', 0)} "
        f"rate_skipped {rate_skipped} refused {refused}{breakdown} failed {failed}"
    )
    if published > 0 and ages_s:
        stages = " ".join(f"{stage} {_median(times_ms.get(stage, ())):.0f}" for stage in _STAGES)
        text += (
            f"; p50 ms {stages}; age s p50 {_median(ages_s):.2f} max {max(ages_s):.2f}; "
            f"valid p50 {_median(valid):.0%}"
        )
    lines = [("info", text)]
    if left > 0 and right == 0:
        lines.append(
            (
                "warning",
                f"no right images on {cfg.right_image_topic}; the ZED publishes them only with "
                "video.publish_left_right: true",
            )
        )
    elif right > 0 and left == 0:
        lines.append(("warning", f"no left images on {cfg.left_image_topic}"))
    elif left == 0 and right == 0:
        lines.append(("info", f"waiting for images on {cfg.left_image_topic} and {cfg.right_image_topic}"))
    elif unmatched > 0 and published + refused + failed + rate_skipped == 0:
        # Images arrive on both sides, but no left image found a right one
        # with its stamp.
        lines.append(
            (
                "warning",
                f"no left and right images share a stamp, and the node pairs only equal stamps; compare the "
                f"stamps on {cfg.left_image_topic} and {cfg.right_image_topic}",
            )
        )
    if ages_s and _median(ages_s) > AGE_WARN_S:
        lines.append(
            (
                "warning",
                f"depth is {_median(ages_s):.2f} s old when published, over {AGE_WARN_S:g} s; one pair took "
                "about 0.85 s on the robot, so the model or the images are slower than measured",
            )
        )
    return lines


class StereoDepthNode(Node):
    """Pairs the ZED's left and right images, runs the disparity model, and publishes depth and the left image."""

    def __init__(self, model: DisparityModel | None = None, **node_kwargs) -> None:
        super().__init__(NODE_NAME, **node_kwargs)
        defaults = StereoConfig()
        # One descriptor per parameter: rclpy writes the name and type into it.
        values = {
            name: self.declare_parameter(name, getattr(defaults, name), ParameterDescriptor(read_only=True)).value
            for name in StereoConfig.__dataclass_fields__
        }
        # Checked before the model loads, which takes about 30 s.
        cfg = self.config = StereoConfig(**values)
        if model is None:
            path = model_path(cfg.model)
            self.get_logger().info(f"loading {path}; this takes about 30 s")
            start = time.monotonic()
            model = load_model(path)
            self.get_logger().info(f"{path.name} on {model.providers[0]} in {time.monotonic() - start:.1f} s")
        self._model = model
        self._info_pub = self.create_publisher(CameraInfo, cfg.camera_info_topic, OUTPUT_QOS)
        self._depth_info_pub = self.create_publisher(CameraInfo, cfg.depth_camera_info_topic, OUTPUT_QOS)
        self._depth_pub = self.create_publisher(Image, cfg.depth_topic, OUTPUT_QOS)
        self._image_pub = self.create_publisher(Image, cfg.image_topic, OUTPUT_QOS)
        self.counts: Counter = Counter()
        self._pairer = StereoPairer(self.counts)
        self._left_info: CameraInfo | None = None
        self._right_info: CameraInfo | None = None
        self._failures_in_a_row = 0
        self._last_start: float | None = None
        self._monotonic = time.monotonic
        self._times: dict[str, list[float]] = {stage: [] for stage in _STAGES}
        self._ages: list[float] = []
        self._valid: list[float] = []
        self._reported: Counter = Counter()
        # Every callback runs on rclpy.spin's one thread, so nothing here needs a lock.
        self.create_subscription(Image, cfg.left_image_topic, self._on_left, INPUT_QOS)
        self.create_subscription(Image, cfg.right_image_topic, self._on_right, INPUT_QOS)
        self.create_subscription(CameraInfo, cfg.left_camera_info_topic, self._on_left_info, INPUT_QOS)
        self.create_subscription(CameraInfo, cfg.right_camera_info_topic, self._on_right_info, INPUT_QOS)
        self.create_timer(REPORT_PERIOD_S, self._report, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.get_logger().info(
            f"pairing {cfg.left_image_topic} with {cfg.right_image_topic}; publishing {cfg.depth_topic} "
            f"and relaying the image on {cfg.image_topic}"
        )

    def _on_left_info(self, msg: CameraInfo) -> None:
        self._left_info = msg

    def _on_right_info(self, msg: CameraInfo) -> None:
        self._right_info = msg

    def _on_left(self, msg: Image) -> None:
        self.counts["left"] += 1
        pair = self._pairer.offer_left(msg)
        if pair is not None:
            self._process(*pair)

    def _on_right(self, msg: Image) -> None:
        self.counts["right"] += 1
        pair = self._pairer.offer_right(msg)
        if pair is not None:
            self._process(*pair)

    def _process(self, left: Image, right: Image) -> None:
        cfg = self.config
        now = self._monotonic()
        if cfg.max_rate_hz > 0 and self._last_start is not None and now - self._last_start < 1.0 / cfg.max_rate_hz:
            self.counts["rate_skipped"] += 1
            return
        self._last_start = now
        try:
            frame = stereo_depth(left, right, self._left_info, self._right_info, cfg.scale_x, cfg.scale_y, self._model)
        except StereoRefused as e:
            # A refusal is about the input, not the model: it neither counts
            # as a failure nor ends a run of failures.
            self.counts[f"refused_{e.reason}"] += 1
            self.get_logger().warning(f"refused a stereo pair ({e.reason}): {e}", throttle_duration_sec=10.0)
            return
        except Exception as e:
            self.counts["failed"] += 1
            self._failures_in_a_row += 1
            if self.counts["failed"] == 1:
                self.get_logger().error(f"the stereo model failed: {traceback.format_exc()}")
            else:
                self.get_logger().warning(
                    f"the stereo model failed: {type(e).__name__}: {e}", throttle_duration_sec=10.0
                )
            if self._failures_in_a_row >= MAX_CONSECUTIVE_FAILURES:
                raise
            return
        # The vision node then holds both infos and the depth when the image
        # arrives, so its RgbdBuffer pairs on the exact stamp without waiting.
        self._info_pub.publish(frame.image_info)
        self._depth_info_pub.publish(frame.depth_info)
        self._depth_pub.publish(depth_image(frame.depth, left.header))
        age_s = (self.get_clock().now().nanoseconds - stamp_ns(left.header)) / 1e9
        self._image_pub.publish(left)
        self.counts["published"] += 1
        self._failures_in_a_row = 0
        for stage in _STAGES:
            self._times[stage].append(frame.times_ms[stage])
        self._ages.append(age_s)
        self._valid.append(frame.valid_share)

    def _report(self) -> None:
        delta = {k: self.counts[k] - self._reported[k] for k in self.counts}
        logger = self.get_logger()
        for level, text in report_lines(delta, self._times, self._ages, self._valid, self.config):
            # One call per severity: rclpy keeps each call site's severity and
            # raises ValueError when one call site logs at two.
            if level == "warning":
                logger.warning(text)
            else:
                logger.info(text)
        self._reported = Counter(self.counts)
        for values in (*self._times.values(), self._ages, self._valid):
            values.clear()


def _flush() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass


def main(args: list[str] | None = None) -> int:
    """Run until Ctrl-C (exit 0), a failure while running (exit 1) or a startup failure (exit 2)."""
    logger = get_logger(NODE_NAME)
    node: StereoDepthNode | None = None
    code = 0
    fatal: str | None = None
    try:
        try:
            rclpy.init(args=args)
            node = StereoDepthNode()
        except (KeyboardInterrupt, ExternalShutdownException):
            raise
        except Exception as e:
            logger.fatal(f"could not start: {type(e).__name__}: {e}")
            code = EXIT_STARTUP
        else:
            try:
                rclpy.spin(node)
            except (KeyboardInterrupt, ExternalShutdownException):
                raise
            except Exception as e:
                fatal = f"stopped: {type(e).__name__}: {e}"
    except (KeyboardInterrupt, ExternalShutdownException):
        code = 0
    finally:
        # A second SIGINT must not cut the cleanup short. pixi run forwards
        # Ctrl-C to its child, and ros2 launch sends one of its own when its
        # stdin is not a terminal, so a node often gets two.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if fatal is not None:
            logger.fatal(fatal)
            _flush()
            # Skip the ONNX Runtime destructors, which can hang with a broken
            # CUDA context.
            os._exit(EXIT_FATAL)
        if node is not None:
            node.destroy_node()
        try:
            rclpy.try_shutdown()
        except Exception:
            # After Ctrl-C, rclpy's signal handler shuts the context down from
            # its own thread a moment later, and can do it between
            # try_shutdown's check and its shutdown. The context is down either way.
            if rclpy.ok():
                raise
    return code


if __name__ == "__main__":
    sys.exit(main())
