"""The vision node: objects and people from the head camera, in one process (ADR-0010).

For every image it processes, the node publishes one PersonDetectionArray
and one ObjectDetectionArray, both with the image's header and a copy of
its camera_info. The contract that consumers rely on is in
docs/architecture/vision.md#contract.

Threads:

- The main thread spins the executor. The image callback only puts the
  message in a one-slot mailbox, so a newer image replaces one that waits.
  The camera_info callback keeps the newest message. A timer on wall time
  logs the latency summary.
- One worker thread takes the newest image, runs walkie_vision.pipeline on
  it and publishes the two arrays. It is the only thread that touches a
  model.
- main() checks the worker's health on every loop. A failure while running
  exits with EXIT_FATAL (1), and the launch file starts the node again. A
  failure before the worker starts, such as a missing model file or a bad
  parameter, exits with EXIT_STARTUP (2).

torch, Ultralytics and ONNX Runtime are imported only in build_providers(),
so the tests import this module and inject fakes.
"""

from __future__ import annotations

import collections
import logging
import os
import re
import sys
import threading
import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.clock import Clock, ClockType
from rclpy.exceptions import NameValidationException, ParameterException
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.logging import get_logger
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.validate_full_topic_name import validate_full_topic_name
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
from walkie_vision_msgs.msg import ObjectDetectionArray, PersonDetectionArray

from .image import camera_info_problem, image_to_bgr
from .pipeline import Providers, TrackIdBase, VisionPipeline

NODE_NAME = "walkie_vision"

# The camera's newest image only: a late image is worth less than none.
IMAGE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)
CAMERA_INFO_QOS = IMAGE_QOS
# RELIABLE, as walkie-agent-ros's oracle_perception: a reliable subscriber
# such as a bag recorder gets every array, and a BEST_EFFORT one, such as the
# agent's VisionFeed, matches too.
OUTPUT_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)

TARGET_RATE_HZ = 5.0
"""The contract's lowest output rate."""
EXIT_FATAL = 1
"""A failure while running. The launch file starts the node again."""
EXIT_STARTUP = 2
"""A failure before the worker started: a missing file, a bad parameter, a model that did not load."""

# Ultralytics' own tracker files, by name. The same names as
# walkie_vision.yolo.TRACKER_NAMES, which this module does not import.
TRACKER_FILES = ("bytetrack.yaml", "botsort.yaml")
_DEVICE = re.compile(r"^(cpu|cuda(:\d+)?)$")
# The files build_providers() needs in models/buffalo_l, as named in
# walkie_vision.face.analysis (which imports onnxruntime).
_FACE_FILES = ("det_10g.onnx", "w600k_r50.onnx")
# How often a repeated warning is logged, in seconds.
_WARN_PERIOD_S = 5.0
_OTHER_PUBLISHER_WARN_PERIOD_S = 30.0
_JOIN_TIMEOUT_S = 5.0


class StartupError(RuntimeError):
    """A startup failure whose message says what to do; main() logs it without a traceback."""


class StartupLogger(Protocol):
    """What build_providers() logs to: an rclpy logger or a logging.Logger."""

    def info(self, message: str) -> object: ...

    def warning(self, message: str) -> object: ...


@dataclass(frozen=True)
class VisionConfig:
    """The node's parameters. The defaults equal config/vision.yaml.

    Every field is a read-only ROS parameter of the same name. classes has
    no default: it comes from the task's file in config/tasks/.
    """

    classes: tuple[str, ...]
    image_topic: str = "/zed_head/zed_node/rgb/color/rect/image"
    camera_info_topic: str = "/zed_head/zed_node/depth/camera_info"
    people_topic: str = "/walkie_vision/people"
    objects_topic: str = "/walkie_vision/objects"
    device: str = "cuda:0"
    objects_weights: str = "yoloe-26s-seg.pt"
    objects_imgsz: int = 640
    objects_conf: float = 0.1
    objects_agnostic_nms: bool = True
    objects_tracker: str = "bytetrack.yaml"
    objects_min_score: float = 0.25
    pose_weights: str = "yolo26s-pose.pt"
    pose_imgsz: int = 640
    pose_conf: float = 0.1
    pose_tracker: str = "botsort.yaml"
    people_min_score: float = 0.25
    people_dedupe_iou: float = 0.5
    face_det_thresh: float = 0.5
    face_input_sizes: tuple[int, ...] = (160, 256, 384, 640)
    face_max_people: int = 6
    face_max_offset: float = 0.25
    reid_min_crop_px: int = 16
    reid_batch_size: int = 16
    max_rate_hz: float = 10.0
    max_consecutive_failures: int = 30
    process_timeout_s: float = 60.0
    latency_log_period_s: float = 10.0
    stale_warn_s: float = 0.5

    def __post_init__(self) -> None:
        # Lists become tuples, so the config stays hashable and immutable.
        object.__setattr__(self, "classes", tuple(self.classes))
        object.__setattr__(self, "face_input_sizes", tuple(int(s) for s in self.face_input_sizes))

        if not self.classes:
            raise ValueError("classes must not be empty")
        for c in self.classes:
            if not isinstance(c, str) or not c or c != c.strip():
                raise ValueError(f"classes: every class must be a non-empty string without surrounding spaces, not {c!r}")
        if len(set(self.classes)) != len(self.classes):
            raise ValueError(f"classes has duplicates: {list(self.classes)}")

        for name in ("image_topic", "camera_info_topic", "people_topic", "objects_topic"):
            topic = getattr(self, name)
            try:
                validate_full_topic_name(topic)
            except NameValidationException as e:
                raise ValueError(f"{name} must be an absolute, valid topic name, not {topic!r}: {e}") from e

        if not _DEVICE.match(self.device):
            raise ValueError(f"device must be cpu, cuda or cuda:N, not {self.device!r}")
        for name in ("objects_weights", "pose_weights"):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        for name in ("objects_tracker", "pose_tracker"):
            tracker = getattr(self, name)
            if tracker not in TRACKER_FILES and not (Path(tracker).is_absolute() and tracker.endswith(".yaml")):
                raise ValueError(
                    f"{name} must be one of {', '.join(TRACKER_FILES)} or an absolute path to a .yaml file, "
                    f"not {tracker!r}"
                )

        for name in ("objects_conf", "pose_conf", "objects_min_score", "people_min_score"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], not {value}")
        for name in ("people_dedupe_iou", "face_det_thresh", "face_max_offset"):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], not {value}")

        for name in ("objects_imgsz", "pose_imgsz"):
            value = getattr(self, name)
            if value <= 0 or value % 32:
                raise ValueError(f"{name} must be a positive multiple of 32, not {value}")
        if not self.face_input_sizes or any(s <= 0 or s % 32 for s in self.face_input_sizes):
            raise ValueError(
                f"face_input_sizes must be a non-empty list of positive multiples of 32, not {list(self.face_input_sizes)}"
            )

        for name, low in (
            ("face_max_people", 0),
            ("reid_min_crop_px", 1),
            ("reid_batch_size", 1),
            ("max_consecutive_failures", 1),
        ):
            value = getattr(self, name)
            if value < low:
                raise ValueError(f"{name} must be at least {low}, not {value}")
        if self.max_rate_hz < 0:
            raise ValueError(f"max_rate_hz must be 0 (no cap) or positive, not {self.max_rate_hz}")
        for name in ("process_timeout_s", "latency_log_period_s", "stale_warn_s"):
            value = getattr(self, name)
            if not value > 0:
                raise ValueError(f"{name} must be positive, not {value}")

    @classmethod
    def from_node(cls, node: Node) -> VisionConfig:
        """Declare every parameter on node, read-only, and return their values.

        Raises ValueError naming the parameter when one is missing, has the
        wrong type or a bad value.
        """
        values: dict[str, Any] = {}
        for f in fields(cls):
            descriptor = ParameterDescriptor(read_only=True)
            try:
                if f.name == "classes":
                    node.declare_parameter(f.name, Parameter.Type.STRING_ARRAY, descriptor)
                else:
                    default = list(f.default) if isinstance(f.default, tuple) else f.default
                    node.declare_parameter(f.name, default, descriptor)
                # get_parameter() raises for a typed parameter that was never set.
                value = node.get_parameter(f.name).value
            except (ParameterException, TypeError, NameValidationException) as e:
                raise ValueError(f"parameter {f.name}: {e}") from e
            if f.name in ("classes", "face_input_sizes"):
                value = tuple(value)
            values[f.name] = value
        return cls(**values)


class NewestFrame:
    """A one-slot mailbox: a new item replaces the one that waits."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._item: tuple[Image, float] | None = None
        self._closed = False
        self.replaced = 0
        """How many waiting items a newer one replaced."""

    def put(self, item: tuple[Image, float]) -> None:
        with self._cond:
            if self._closed:
                return
            if self._item is not None:
                self.replaced += 1
            self._item = item
            self._cond.notify()

    def take(self, timeout: float) -> tuple[Image, float] | None:
        """Return the waiting item, waiting up to timeout seconds; None when there is none."""
        with self._cond:
            if self._item is None and not self._closed and timeout > 0:
                self._cond.wait(timeout)
            item, self._item = self._item, None
            return item

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._item = None
            self._cond.notify_all()

    @property
    def closed(self) -> bool:
        return self._closed


class LatencyLog:
    """Per-stage latencies of the processed images, summarised once per period.

    add() runs on the worker thread and summary() on the timer, so both
    take a lock. summary() swaps out the samples, so a sample is never lost
    or counted twice, and it reports the counts as the change since the
    previous summary.
    """

    STAGES = ("wait", "convert", "objects", "pose", "faces", "reid", "build", "publish", "age")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._samples: dict[str, list[float]] = {s: [] for s in self.STAGES}
        self._processed = 0
        self._last_counts: dict[str, int] = {}
        self.last_processed = 0
        """The number of images in the latest summary."""
        self.last_deltas: dict[str, int] = {}
        """The counts in the latest summary: the change since the one before."""

    def add(self, **ms: float) -> None:
        """Record one processed image: milliseconds per stage, by the names in STAGES."""
        unknown = set(ms) - set(self.STAGES)
        if unknown:
            raise ValueError(f"unknown stages: {sorted(unknown)}")
        with self._lock:
            for stage, value in ms.items():
                self._samples[stage].append(float(value))
            self._processed += 1

    def summary(self, elapsed_s: float, counts: Mapping[str, int]) -> str:
        """Return the line for the period that just ended, and start a new one.

        counts holds running totals, such as received; the line shows each
        one's change since the previous summary, in the order given.
        """
        with self._lock:
            samples, self._samples = self._samples, {s: [] for s in self.STAGES}
            processed, self._processed = self._processed, 0
            deltas = {k: int(v) - self._last_counts.get(k, 0) for k, v in counts.items()}
            self._last_counts = {k: int(v) for k, v in counts.items()}
        self.last_processed = processed
        self.last_deltas = deltas
        rate = processed / elapsed_s if elapsed_s > 0 else 0.0
        counted = ", ".join(f"{k} {v}" for k, v in deltas.items())
        stages = " ".join(f"{s} {_p50_p95(samples[s])}" for s in self.STAGES if s != "age")
        head = f"processed {processed} in {elapsed_s:.1f} s ({rate:.1f} Hz)"
        if counted:
            head += f"; {counted}"
        return f"{head} | ms p50/p95: {stages} | age {_p50_p95(samples['age'])}"


def _p50_p95(values: list[float]) -> str:
    if not values:
        return "-/-"
    p50, p95 = np.percentile(np.asarray(values, dtype=np.float64), [50, 95])
    return f"{p50:.0f}/{p95:.0f}"


def build_providers(config: VisionConfig, logger: StartupLogger) -> Providers:
    """Load every model the node runs, after checking that their files are there.

    The order makes a broken setup fail fast: the model files first, then
    the OSNet checkpoint in the Hugging Face cache, both before torch or
    Ultralytics is imported. Then the PyTorch models, then the ONNX Runtime
    sessions. Raises FileNotFoundError naming every missing file, and
    StartupError (a RuntimeError) when OSNet is not cached and cannot be
    downloaded. A model that fails to load raises its own error.
    """
    started = time.perf_counter()
    from .models import BUFFALO_L_DIR, HUB_MODELS, OSNET_CHECKPOINT, ULTRALYTICS_DIR, models_dir
    from .yolo import MOBILECLIP_FILE

    root = models_dir()
    ultralytics_dir = (root / ULTRALYTICS_DIR).absolute()
    face_dir = (root / BUFFALO_L_DIR).absolute()

    def weights(name: str) -> Path:
        path = Path(name)
        return path if path.is_absolute() else ultralytics_dir / path

    objects_path = weights(config.objects_weights)
    pose_path = weights(config.pose_weights)
    # YOLOE loads its text encoder by file name from the folder of its weights.
    needed = [objects_path, objects_path.parent / MOBILECLIP_FILE, pose_path, *(face_dir / f for f in _FACE_FILES)]
    missing = [str(p) for p in needed if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"model files are missing: {', '.join(missing)}; run `pixi run fetch-models`")

    import huggingface_hub

    hub = HUB_MODELS["osnet"]
    cached = huggingface_hub.try_to_load_from_cache(hub.repo, OSNET_CHECKPOINT, revision=hub.revision)
    if not isinstance(cached, str):
        logger.warning("OSNet is not in the Hugging Face cache; downloading it now, which needs the internet once")
        try:
            huggingface_hub.hf_hub_download(hub.repo, OSNET_CHECKPOINT, revision=hub.revision)
        except Exception as e:
            raise StartupError(
                f"OSNet is not in the Hugging Face cache and could not be downloaded: {e}. "
                "Run once with internet access; see docs/guides/vision-node.md"
            ) from e

    from walkie_vision.yolo import YoloeTracker, YoloPoseTracker

    objects = YoloeTracker(
        objects_path,
        list(config.classes),
        device=config.device,
        imgsz=config.objects_imgsz,
        conf=config.objects_conf,
        agnostic_nms=config.objects_agnostic_nms,
        tracker=config.objects_tracker,
    )
    pose = YoloPoseTracker(
        pose_path,
        device=config.device,
        imgsz=config.pose_imgsz,
        conf=config.pose_conf,
        tracker=config.pose_tracker,
    )

    from walkie_vision.reid.identifier import OSNetReIdentifier

    reid = OSNetReIdentifier(device=config.device, batch_size=config.reid_batch_size)

    import onnxruntime

    from walkie_vision.face import FaceAnalyzer, HeadCropConfig, HeadFaceRecognizer

    # ERROR and above only, for every session created from here on. SCRFD's
    # file declares its output shapes for a 640 input and ArcFace's for a
    # batch of 1, so ONNX Runtime would warn on every smaller crop and every
    # batch of faces, burying the node's own warnings. A session takes the
    # level when it is created, so this must come before FaceAnalyzer.
    onnxruntime.set_default_logger_severity(3)
    logger.info("warming up the face models; the first run on this GPU can take minutes")
    faces = HeadFaceRecognizer(
        FaceAnalyzer(face_dir, det_thresh=config.face_det_thresh),
        HeadCropConfig(
            input_sizes=tuple(config.face_input_sizes),
            max_people=config.face_max_people,
            max_face_offset=config.face_max_offset,
        ),
    )

    import torch

    if torch.cuda.is_initialized():
        # Return the blocks freed during the loads and warm-ups, such as the
        # text encoder that set_classes() drops, so ONNX Runtime can use them.
        torch.cuda.empty_cache()
    logger.info(f"loaded the models in {time.perf_counter() - started:.1f} s")
    return Providers(objects=objects, pose=pose, faces=faces, reid=reid)


class VisionNode(Node):
    """Publishes one people array and one object array for every image it processes.

    providers defaults to the real models (build_providers()); the tests
    pass fakes. track_ids defaults to an id base from the current time.
    node_kwargs go to rclpy's Node, for example context, cli_args and
    parameter_overrides.
    """

    def __init__(
        self,
        providers: Providers | None = None,
        *,
        track_ids: TrackIdBase | None = None,
        **node_kwargs: Any,
    ) -> None:
        # Taken before the models load, so a slow load cannot make two runs' ranges meet.
        track_ids = TrackIdBase() if track_ids is None else track_ids
        super().__init__(NODE_NAME, **node_kwargs)
        try:
            self._setup(providers, track_ids)
        except BaseException:
            self.destroy_node()
            raise

    def _setup(self, providers: Providers | None, track_ids: TrackIdBase) -> None:
        self.config = VisionConfig.from_node(self)
        cfg = self.config
        if providers is None:
            providers = build_providers(cfg, self.get_logger())
        self.pipeline = VisionPipeline(
            providers,
            cfg.classes,
            track_ids,
            reid_min_crop_px=cfg.reid_min_crop_px,
            people_min_score=cfg.people_min_score,
            objects_min_score=cfg.objects_min_score,
            people_dedupe_iou=cfg.people_dedupe_iou,
        )
        self.track_ids = track_ids

        self.frames_received = 0
        self.frames_processed = 0
        self.frames_failed = 0
        self.summaries_logged = 0
        self.skipped: collections.Counter[str] = collections.Counter()
        self.dropped: collections.Counter[str] = collections.Counter()
        self.duplicates = 0
        """Poses left out as a second box of a person; YOLO26 pose makes them now and then."""
        self.consecutive_failures = 0
        self.fatal: str | None = None
        self.latency = LatencyLog()

        self._counts_lock = threading.Lock()
        self._warned_at: dict[str, float] = {}
        self._warn_lock = threading.Lock()
        self._mailbox = NewestFrame()
        self._rate_replaced = 0
        self._camera_info: CameraInfo | None = None
        self._camera_info_lock = threading.Lock()
        self._busy_since: float | None = None
        self._stop = threading.Event()
        self._closed = False
        self._period_start = time.monotonic()

        self._people_pub = self.create_publisher(PersonDetectionArray, cfg.people_topic, OUTPUT_QOS)
        self._objects_pub = self.create_publisher(ObjectDetectionArray, cfg.objects_topic, OUTPUT_QOS)
        # Subscribed only once the models are loaded, so nothing waits during the load.
        self.create_subscription(Image, cfg.image_topic, self._on_image, IMAGE_QOS)
        self.create_subscription(CameraInfo, cfg.camera_info_topic, self._on_camera_info, CAMERA_INFO_QOS)
        # Wall time, so the summary also comes under use_sim_time when /clock stops.
        self._latency_timer = self.create_timer(
            cfg.latency_log_period_s, self._on_latency_timer, clock=Clock(clock_type=ClockType.STEADY_TIME)
        )
        self._worker = threading.Thread(target=self._work, daemon=True, name="walkie_vision_worker")
        self._worker.start()
        self.get_logger().info(
            f"classes {list(cfg.classes)}; image {cfg.image_topic}, camera_info {cfg.camera_info_topic}; "
            f"publishing {cfg.people_topic} and {cfg.objects_topic}; trackers: objects {cfg.objects_tracker}, "
            f"people {cfg.pose_tracker}; track id offset {track_ids.offset}"
        )

    # --- counters and state --------------------------------------------

    @property
    def frames_replaced(self) -> int:
        """Images that a newer one replaced before the worker took them."""
        return self._mailbox.replaced + self._rate_replaced

    def other_publishers(self) -> list[str]:
        """The output topics on which another node publishes too."""
        return [t for t in (self.config.people_topic, self.config.objects_topic) if self.count_publishers(t) > 1]

    def check_health(self) -> None:
        """Set fatal if the worker has stopped, or if one image has taken longer than process_timeout_s."""
        if self.fatal is not None or self._stop.is_set():
            return
        if not self._worker.is_alive():
            self.fatal = "the worker thread stopped"
            return
        busy = self._busy_since
        if busy is not None and time.monotonic() - busy > self.config.process_timeout_s:
            self.fatal = f"one image has taken more than {self.config.process_timeout_s:g} s; the GPU may hang"

    def close(self) -> bool:
        """Stop the worker and destroy the node. False if the worker did not stop in time.

        When the worker does not stop, the node is not destroyed: the worker
        may still publish.
        """
        if self._closed:
            return True
        self._stop.set()
        self._mailbox.close()
        self._worker.join(timeout=_JOIN_TIMEOUT_S)
        if self._worker.is_alive():
            return False
        self._closed = True
        self.destroy_node()
        return True

    # --- callbacks on the executor thread -------------------------------

    def _on_image(self, msg: Image) -> None:
        self.frames_received += 1
        self._mailbox.put((msg, time.perf_counter()))

    def _on_camera_info(self, msg: CameraInfo) -> None:
        with self._camera_info_lock:
            self._camera_info = msg

    def _on_latency_timer(self) -> None:
        now = time.monotonic()
        elapsed, self._period_start = now - self._period_start, now
        with self._counts_lock:
            counts = {
                "received": self.frames_received,
                "replaced": self.frames_replaced,
                "skipped": sum(self.skipped.values()),
                "failed": self.frames_failed,
                "dropped": sum(self.dropped.values()),
                "duplicates": self.duplicates,
            }
        self.get_logger().info(self.latency.summary(elapsed, counts))
        self.summaries_logged += 1
        if elapsed > 0:
            processed_hz = self.latency.last_processed / elapsed
            received_hz = self.latency.last_deltas.get("received", 0) / elapsed
            if processed_hz < TARGET_RATE_HZ <= received_hz:
                self.get_logger().warning(
                    f"processing at {processed_hz:.1f} Hz, under the {TARGET_RATE_HZ:g} Hz the contract targets"
                )
        for topic in self.other_publishers():
            self._warn_throttled(
                f"publishers {topic}",
                f"another node also publishes on {topic}; consumers would mix two sources",
                _OTHER_PUBLISHER_WARN_PERIOD_S,
            )

    def _warn_throttled(self, key: str, message: str, period_s: float = _WARN_PERIOD_S) -> None:
        now = time.monotonic()
        with self._warn_lock:
            last = self._warned_at.get(key)
            if last is not None and now - last < period_s:
                return
            self._warned_at[key] = now
        self.get_logger().warning(message)

    # --- the worker -----------------------------------------------------

    def _work(self) -> None:
        try:
            last_start: float | None = None
            while not self._stop.is_set():
                item = self._mailbox.take(0.5)
                if item is None:
                    if self._mailbox.closed:
                        return
                    continue
                rate = self.config.max_rate_hz
                if rate > 0 and last_start is not None:
                    delay = last_start + 1.0 / rate - time.monotonic()
                    if delay > 0:
                        if self._stop.wait(delay):
                            return
                        newer = self._mailbox.take(0.0)
                        if newer is not None:
                            self._rate_replaced += 1
                            item = newer
                last_start = time.monotonic()
                if not self._process_one(*item):
                    return
        except BaseException as e:
            self.get_logger().fatal(f"the worker stopped:\n{traceback.format_exc()}")
            self.fatal = f"the worker stopped: {type(e).__name__}: {e}"

    def _process_one(self, msg: Image, received: float) -> bool:
        """Process one image and publish its arrays. False when the worker should stop."""
        cfg = self.config
        start = time.perf_counter()
        wait_ms = (start - received) * 1000.0
        with self._camera_info_lock:
            info = self._camera_info
        problem = camera_info_problem(info, int(msg.width), int(msg.height))
        if problem is not None:
            with self._counts_lock:
                self.skipped["camera_info"] += 1
            self._warn_throttled("camera_info", f"skipping images from {cfg.image_topic}: {problem} on {cfg.camera_info_topic}")
            return True
        if info.header.frame_id != msg.header.frame_id:
            self.get_logger().warning(
                f"camera_info's frame_id {info.header.frame_id!r} is not the image's {msg.header.frame_id!r}; "
                "the arrays use the image's",
                once=True,
            )
        try:
            image = image_to_bgr(msg)
        except ValueError as e:
            with self._counts_lock:
                self.skipped["encoding"] += 1
            self._warn_throttled("encoding", f"skipping an image from {cfg.image_topic}: {e}")
            return True
        convert_ms = (time.perf_counter() - start) * 1000.0

        header = Header(stamp=msg.header.stamp, frame_id=msg.header.frame_id)
        self._busy_since = time.monotonic()
        try:
            result = self.pipeline.process(image, header, info)
        except Exception as e:
            self.consecutive_failures += 1
            with self._counts_lock:
                self.frames_failed += 1
            if self.frames_failed == 1:
                self.get_logger().error(f"an image failed:\n{traceback.format_exc()}")
            else:
                self._warn_throttled("failed", f"an image failed: {type(e).__name__}: {e}")
            if self.consecutive_failures >= cfg.max_consecutive_failures:
                self.fatal = f"{self.consecutive_failures} images in a row failed: {type(e).__name__}: {e}"
                return False
            return True
        finally:
            self._busy_since = None
        self.consecutive_failures = 0
        if result.duplicates:
            # Counted in the latency line only: a second box is the model's
            # normal behaviour, not a bad field.
            with self._counts_lock:
                self.duplicates += result.duplicates
        if result.dropped:
            with self._counts_lock:
                self.dropped.update(result.dropped)
            counts = ", ".join(f"{k} {v}" for k, v in sorted(result.dropped.items()))
            self._warn_throttled("dropped", f"left out bad fields: {counts}")

        if self._stop.is_set():
            return False
        t = time.perf_counter()
        try:
            # People first, as oracle_perception does.
            self._people_pub.publish(result.people)
            self._objects_pub.publish(result.objects)
        except Exception:
            if self._stop.is_set():
                return False  # the context is shutting down
            raise
        publish_ms = (time.perf_counter() - t) * 1000.0

        self.frames_processed += 1
        stamp = msg.header.stamp
        age_s = (self.get_clock().now().nanoseconds - (stamp.sec * 1_000_000_000 + stamp.nanosec)) / 1e9
        times = result.times
        self.latency.add(
            wait=wait_ms,
            convert=convert_ms,
            objects=times.objects,
            pose=times.pose,
            faces=times.faces,
            reid=times.reid,
            build=times.build,
            publish=publish_ms,
            age=age_s * 1000.0,
        )
        if age_s > cfg.stale_warn_s:
            self._warn_throttled(
                "stale",
                f"arrays are {age_s:.2f} s old when published; walkie-agent-ros drops arrays older than 1 s",
            )
        return True


def _log_startup_failure(logger: Any, error: BaseException) -> None:
    message = f"cannot start: {type(error).__name__}: {error}"
    # A missing file, a bad parameter or OSNet offline explains itself;
    # anything else, such as a model that fails to load, gets its traceback.
    if not isinstance(error, (FileNotFoundError, ValueError, StartupError)):
        message += "\n" + "".join(traceback.format_exception(error))
    logger.fatal(message)


def _flush() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    for handler in logging.getLogger().handlers:
        try:
            handler.flush()
        except Exception:
            pass


def main(args: list[str] | None = None) -> int:
    """Run the node until Ctrl-C, a fatal failure (exit 1) or a startup failure (exit 2)."""
    logger = get_logger(NODE_NAME)
    node: VisionNode | None = None
    code = 0
    stopped = True
    try:
        try:
            rclpy.init(args=args)
            node = VisionNode()
        except (KeyboardInterrupt, ExternalShutdownException):
            raise
        except Exception as e:
            _log_startup_failure(logger, e)
            code = EXIT_STARTUP
        else:
            executor = SingleThreadedExecutor()
            executor.add_node(node)
            while rclpy.ok() and node.fatal is None:
                executor.spin_once(timeout_sec=0.5)
                node.check_health()
    except (KeyboardInterrupt, ExternalShutdownException):
        code = 0
    finally:
        if node is not None:
            stopped = node.close()
    if node is not None and node.fatal is not None:
        logger.fatal(node.fatal)
        _flush()
        # Skip the torch and ONNX Runtime destructors, which can hang with a
        # broken CUDA context, and rclpy's shutdown while a stuck worker may
        # still publish.
        os._exit(EXIT_FATAL)
    if not stopped:
        logger.warning("the worker did not stop in time; exiting without cleanup")
        _flush()
        os._exit(code)
    rclpy.try_shutdown()
    return code


if __name__ == "__main__":
    sys.exit(main())
