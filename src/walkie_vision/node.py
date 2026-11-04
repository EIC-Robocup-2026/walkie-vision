"""The vision node: objects and people from the head camera, in one process (ADR-0010).

For every image it processes, the node publishes one PersonDetectionArray
and one ObjectDetectionArray, both with the image's header and a copy of
its camera_info. The contract that consumers rely on is in
docs/architecture/vision.md#contract. It also serves GetPersonCrop: one
person's crop and colour shares from the next image (ADR-0011). And it serves
EmbedText, EmbedCrop, DescribeCrop and DetectObjects, with one bounded service
job (ADR-0015).

DetectObjects answers its caller only, and publishes nothing. It runs a separate
detector on the newest image after its classes are ready. It returns at most
pipeline.ON_DEMAND_MAX_OBJECTS detections, those with the highest scores, each
with track id 0. New classes are prepared at most once every
requests.CLASS_PREPARE_PERIOD_S, and a preparation stops the stream for about
1.5 s, because one worker runs every model (ADR-0009).

Threads:

- The main thread spins a MultiThreadedExecutor with EXECUTOR_THREADS
  threads. The image callback only puts the message in a one-slot mailbox,
  so a newer image replaces one that waits. The camera_info callback keeps
  the newest message. A timer on wall time logs the latency summary. These
  stay in the node's default callback group, so they still run one at a
  time.
- With geometry_enabled, a depth callback joins them, and the mailbox is a
  walkie_vision.depth.RgbdBuffer that pairs each image with its depth. Its
  wait is bounded by wall time (ADR-0014). Without it, the node does not
  subscribe to depth at all.
- The GetPersonCrop callback has its own callback group. It leaves the
  request in a one-slot mailbox and waits for the worker's answer, while
  another of the executor's threads runs the callbacks above.
- The EmbedText, EmbedCrop, DescribeCrop and DetectObjects callbacks share a
  reentrant group. Each checks its request and tries to leave one job in the slot of
  walkie_vision.requests. One that gets the slot waits for the worker, up to
  a deadline in wall time. Another gets BUSY at once, on another thread.
- One worker thread takes the newest image, runs walkie_vision.pipeline on
  it and publishes the two arrays. It is the only thread that touches a
  model. When a GetPersonCrop request waits, the worker answers it from
  that image and its people (walkie_vision.crop), and then drops both.
  Between images it runs the service job, and a job wakes it with no image.
  The slot stays taken until the worker has also processed the newest image
  after the job, so requests cannot stop the stream.
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
import math
import os
import re
import sys
import threading
import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.clock import Clock, ClockType
from rclpy.exceptions import NameValidationException, ParameterException
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.logging import get_logger
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.validate_full_topic_name import validate_full_topic_name
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
from walkie_vision_msgs.msg import ObjectDetectionArray, PersonDetectionArray
from walkie_vision_msgs.srv import DescribeCrop, DetectObjects, EmbedCrop, EmbedText, GetPersonCrop

from . import convert, crop, requests
from .crop import CropStatus
from .depth import DepthRefused, RegisteredDepth, RgbdBuffer, read_registered_depth, stamp_ns
from .image import camera_info_problem, image_to_bgr
from .pipeline import ON_DEMAND_MAX_OBJECTS, Providers, TrackIdBase, VisionPipeline

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
EXECUTOR_THREADS = 4
"""One for the input callbacks, one for a waiting GetPersonCrop, one for a waiting service job, one for a BUSY answer."""
MAX_CROP_TIMEOUT_S = 5.0
"""The longest timeout_s a GetPersonCrop request may ask for: the service answers one request at a time."""
MAX_PAIR_WINDOW_S = 0.1
"""The largest max_pair_gap_s and pair_wait_s: the output must stay near 5 Hz, and a wider gap pairs other captures."""

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
    person_crop_service: str = "/walkie_vision/get_person_crop"
    embed_text_service: str = "/walkie_vision/embed_text"
    embed_crop_service: str = "/walkie_vision/embed_crop"
    describe_crop_service: str = "/walkie_vision/describe_crop"
    detect_objects_service: str = "/walkie_vision/detect_objects"
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
    geometry_enabled: bool = False
    depth_topic: str = "/zed_head/zed_node/depth/depth_registered"
    depth_camera_info_topic: str = "/zed_head/zed_node/depth/camera_info"
    max_pair_gap_s: float = 0.005
    pair_wait_s: float = 0.020
    require_depth: bool = False
    depth_min_m: float = 0.2
    depth_max_m: float = 3.0
    object_crops_enabled: bool = False
    object_embeddings_enabled: bool = False
    on_demand_enabled: bool = False
    captions_enabled: bool = False

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

        for name in (
            "image_topic",
            "camera_info_topic",
            "people_topic",
            "objects_topic",
            "depth_topic",
            "depth_camera_info_topic",
        ):
            topic = getattr(self, name)
            try:
                validate_full_topic_name(topic)
            except NameValidationException as e:
                raise ValueError(f"{name} must be an absolute, valid topic name, not {topic!r}: {e}") from e
        for name in (
            "person_crop_service",
            "embed_text_service",
            "embed_crop_service",
            "describe_crop_service",
            "detect_objects_service",
        ):
            service = getattr(self, name)
            try:
                validate_full_topic_name(service, is_service=True)
            except NameValidationException as e:
                raise ValueError(f"{name} must be an absolute, valid service name, not {service!r}: {e}") from e
        if self.captions_enabled:
            raise ValueError("captions_enabled must be false: this build has no caption model")

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
        if not 0.0 <= self.max_pair_gap_s <= MAX_PAIR_WINDOW_S:
            raise ValueError(f"max_pair_gap_s must be in [0, {MAX_PAIR_WINDOW_S:g}], not {self.max_pair_gap_s}")
        # Zero would drop every depth that arrives before its image.
        if not 0.0 < self.pair_wait_s <= MAX_PAIR_WINDOW_S:
            raise ValueError(f"pair_wait_s must be in (0, {MAX_PAIR_WINDOW_S:g}], not {self.pair_wait_s}")
        if self.require_depth and not self.geometry_enabled:
            raise ValueError("require_depth needs geometry_enabled: without it the node does not subscribe to depth")
        if not 0.0 < self.depth_min_m < math.inf:
            raise ValueError(f"depth_min_m must be positive and finite, not {self.depth_min_m}")
        if not self.depth_min_m < self.depth_max_m < math.inf:
            raise ValueError(f"depth_max_m must be finite and above depth_min_m, not {self.depth_max_m}")

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
        self._woken = False
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
            if self._item is None and not self._closed and not self._woken and timeout > 0:
                self._cond.wait(timeout)
            self._woken = False
            item, self._item = self._item, None
            return item

    def wake(self) -> None:
        """End the current or the next take() early, so the worker can look at a service job."""
        with self._cond:
            self._woken = True
            self._cond.notify_all()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._item = None
            self._cond.notify_all()

    @property
    def closed(self) -> bool:
        return self._closed


def _crop_answer(status: CropStatus, message: str) -> GetPersonCrop.Response:
    """A GetPersonCrop response with only a status and a message."""
    return GetPersonCrop.Response(status=int(status), message=message)


class CropJob:
    """One GetPersonCrop request, from the service callback to the worker and back.

    The job gets one answer: the worker's through finish(), or none when
    wait() stops waiting first. A lock decides which comes first.
    """

    def __init__(self, request: GetPersonCrop.Request, frames_received: int) -> None:
        self.request = request
        self.frames_received = frames_received
        """The node's frames_received when the request arrived, to tell NO_IMAGE from TIMEOUT."""
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._response: GetPersonCrop.Response | None = None
        self._given_up = False

    def finish(self, response: GetPersonCrop.Response) -> bool:
        """Answer the job. False when it has an answer already, or nobody waits for one."""
        with self._lock:
            if self._given_up or self._response is not None:
                return False
            self._response = response
        self._done.set()
        return True

    def wait(self, timeout: float) -> GetPersonCrop.Response | None:
        """Return the answer, waiting up to timeout seconds; None when there is none, and then no answer is taken."""
        self._done.wait(timeout)
        with self._lock:
            if self._response is None:
                self._given_up = True
            return self._response


class CropMailbox:
    """A one-slot mailbox for the GetPersonCrop job that waits for the next image.

    The service's callback group is mutually exclusive, so at most one job
    waits at a time.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._job: CropJob | None = None
        self._closed = False

    def put(self, job: CropJob) -> bool:
        """Leave a job for the worker. False when the mailbox is closed."""
        with self._lock:
            if self._closed:
                return False
            self._job = job
            return True

    def take(self) -> CropJob | None:
        """Return the waiting job without waiting; None when there is none."""
        with self._lock:
            job, self._job = self._job, None
            return job

    def withdraw(self, job: CropJob) -> None:
        """Remove job if it still waits."""
        with self._lock:
            if self._job is job:
                self._job = None

    def close(self) -> None:
        """Refuse new jobs, and answer the waiting one with FAILED."""
        with self._lock:
            self._closed = True
            job, self._job = self._job, None
        if job is not None:
            job.finish(_crop_answer(CropStatus.FAILED, "the node is shutting down"))


def crop_request_problem(request: GetPersonCrop.Request) -> str | None:
    """Return why a GetPersonCrop request is refused, or None when it is valid."""
    if request.track_id == 0:
        return "track_id 0 is not a track"
    timeout = float(request.timeout_s)
    if not (math.isfinite(timeout) and 0.0 < timeout <= MAX_CROP_TIMEOUT_S):
        return f"timeout_s must be more than 0 and at most {MAX_CROP_TIMEOUT_S:g}, not {timeout}"
    try:
        crop.check_options(float(request.padding), int(request.max_side), float(request.min_keypoint_score))
    except ValueError as e:
        return str(e)
    return None


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

    CLIP is checked and loaded only with object_embeddings_enabled, and only
    from the cache: it never downloads here. The second YOLOE is built only with
    on_demand_enabled, and here at start, so a first request never loads a model.
    """
    started = time.perf_counter()
    from .embeddings import ClipEmbedder, clip_checkpoint
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
    clip_path = clip_checkpoint() if config.object_embeddings_enabled else None

    from walkie_vision.yolo import YoloeDetector, YoloeTracker, YoloPoseTracker

    # Asked for only when geometry or sampling reads them: copying masks to
    # the CPU costs time.
    keep_masks = (
        {"keep_masks": True}
        if config.geometry_enabled or config.object_crops_enabled or config.object_embeddings_enabled
        else {}
    )
    objects = YoloeTracker(
        objects_path,
        list(config.classes),
        device=config.device,
        imgsz=config.objects_imgsz,
        conf=config.objects_conf,
        agnostic_nms=config.objects_agnostic_nms,
        tracker=config.objects_tracker,
        **keep_masks,
    )
    # With no tracker, nothing reads a box under the published floor.
    on_demand = (
        YoloeDetector(
            objects_path,
            list(config.classes),
            device=config.device,
            imgsz=config.objects_imgsz,
            conf=config.objects_min_score,
            agnostic_nms=config.objects_agnostic_nms,
            max_det=ON_DEMAND_MAX_OBJECTS,
            **keep_masks,
        )
        if config.on_demand_enabled
        else None
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
    embedder = None if clip_path is None else ClipEmbedder(clip_path, device=config.device)

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
    return Providers(objects=objects, pose=pose, faces=faces, reid=reid, embedder=embedder, on_demand=on_demand)


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
        injected = providers is not None
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
            geometry_enabled=cfg.geometry_enabled,
            depth_min_m=cfg.depth_min_m,
            depth_max_m=cfg.depth_max_m,
            object_crops_enabled=cfg.object_crops_enabled,
            object_embeddings_enabled=cfg.object_embeddings_enabled,
            allow_fake_embedder=injected,
            on_demand_enabled=cfg.on_demand_enabled,
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
        self.crop_requests = 0
        """GetPersonCrop requests that reached the worker's mailbox."""
        self.service_jobs = 0
        """EmbedText, EmbedCrop and DetectObjects requests that the worker took."""
        self.service_busy = 0
        """Those requests refused because the slot was taken, or new classes came too soon."""
        self.classes_prepared = 0
        """The on-demand class lists that the worker prepared. Each one stops the stream for about 1.5 s."""
        self.late_results = 0
        """Service answers discarded because they came after their deadline."""
        self.fatal: str | None = None
        self.latency = LatencyLog()
        self.depth_received = 0
        self.depth_paired = 0
        """Images processed with a depth raster that passed every check."""
        self.no_depth: collections.Counter[str] = collections.Counter()
        """Images published without geometry, by reason. With require_depth they count in skipped instead."""

        self._counts_lock = threading.Lock()
        self._warned_at: dict[str, float] = {}
        self._warn_lock = threading.Lock()
        self._mailbox: NewestFrame | RgbdBuffer = (
            RgbdBuffer(cfg.max_pair_gap_s, cfg.pair_wait_s) if cfg.geometry_enabled else NewestFrame()
        )
        self._crops = CropMailbox()
        self._embedder = providers.embedder
        self._jobs = requests.JobSlot(self._mailbox.wake)
        # Only what EmbedCrop and DetectObjects can answer from. Off, no image
        # outlives its processing.
        self._current = (
            requests.CurrentImage(round(cfg.max_pair_gap_s * 1e9))
            if cfg.object_embeddings_enabled or cfg.on_demand_enabled
            else None
        )
        self._class_gate = requests.ClassPrepareGate()
        self._rate_replaced = 0
        self._camera_info: CameraInfo | None = None
        self._depth_camera_info: CameraInfo | None = None
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
        if cfg.geometry_enabled:
            self.create_subscription(Image, cfg.depth_topic, self._on_depth, IMAGE_QOS)
            if cfg.depth_camera_info_topic != cfg.camera_info_topic:
                self.create_subscription(CameraInfo, cfg.depth_camera_info_topic, self._on_depth_camera_info, CAMERA_INFO_QOS)
        # Wall time, so the summary also comes under use_sim_time when /clock stops.
        self._latency_timer = self.create_timer(
            cfg.latency_log_period_s, self._on_latency_timer, clock=Clock(clock_type=ClockType.STEADY_TIME)
        )
        # Its own group: the callback waits for the worker, and the image
        # callback must run in the meantime.
        self._crop_service = self.create_service(
            GetPersonCrop,
            cfg.person_crop_service,
            self._on_get_person_crop,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        # Reentrant, so a second request can run and get BUSY while the first waits.
        service_group = ReentrantCallbackGroup()
        self._embed_text_service = self.create_service(
            EmbedText, cfg.embed_text_service, self._on_embed_text, callback_group=service_group
        )
        self._embed_crop_service = self.create_service(
            EmbedCrop, cfg.embed_crop_service, self._on_embed_crop, callback_group=service_group
        )
        self._describe_crop_service = self.create_service(
            DescribeCrop, cfg.describe_crop_service, self._on_describe_crop, callback_group=service_group
        )
        self._detect_objects_service = self.create_service(
            DetectObjects, cfg.detect_objects_service, self._on_detect_objects, callback_group=service_group
        )
        self._worker = threading.Thread(target=self._work, daemon=True, name="walkie_vision_worker")
        self._worker.start()
        self.get_logger().info(
            f"classes {list(cfg.classes)}; image {cfg.image_topic}, camera_info {cfg.camera_info_topic}; "
            f"publishing {cfg.people_topic} and {cfg.objects_topic}; serving {cfg.person_crop_service}; "
            f"trackers: objects {cfg.objects_tracker}, people {cfg.pose_tracker}; track id offset {track_ids.offset}"
        )
        if cfg.geometry_enabled:
            self.get_logger().info(
                f"geometry on: depth {cfg.depth_topic}, depth camera_info {cfg.depth_camera_info_topic}; "
                f"gap at most {cfg.max_pair_gap_s:g} s, wait at most {cfg.pair_wait_s:g} s; "
                f"images without depth are {'skipped' if cfg.require_depth else 'published without points'}"
            )
        if cfg.object_crops_enabled or cfg.object_embeddings_enabled:
            self.get_logger().info(
                f"sampling objects: crops {'on' if cfg.object_crops_enabled else 'off'}, "
                f"embeddings {'on' if cfg.object_embeddings_enabled else 'off'}"
            )
        if cfg.object_embeddings_enabled:
            self.get_logger().info(f"serving {cfg.embed_text_service} and {cfg.embed_crop_service}")
        if cfg.on_demand_enabled:
            self.get_logger().info(
                f"serving {cfg.detect_objects_service} with a second detector: at most {ON_DEMAND_MAX_OBJECTS} objects; "
                f"new classes at most once every {requests.CLASS_PREPARE_PERIOD_S:g} s, and they stop the stream"
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
        """Set fatal if the worker has stopped, or if one image or service job has taken longer than process_timeout_s."""
        if self.fatal is not None or self._stop.is_set():
            return
        if not self._worker.is_alive():
            self.fatal = "the worker thread stopped"
            return
        busy = self._busy_since
        if busy is not None and time.monotonic() - busy > self.config.process_timeout_s:
            self.fatal = f"one image or service job has taken more than {self.config.process_timeout_s:g} s; the GPU may hang"

    def close(self) -> bool:
        """Stop the worker and destroy the node. False if the worker did not stop in time.

        When the worker does not stop, the node is not destroyed: the worker
        may still publish.
        """
        if self._closed:
            return True
        self._stop.set()
        self._mailbox.close()
        self._crops.close()
        self._jobs.close()
        self._worker.join(timeout=_JOIN_TIMEOUT_S)
        if self._worker.is_alive():
            return False
        self._closed = True
        self.destroy_node()
        return True

    # --- callbacks on the executor threads ------------------------------

    def _on_image(self, msg: Image) -> None:
        self.frames_received += 1
        if self._current is not None:
            self._current.put_colour(msg)
        self._mailbox.put((msg, time.perf_counter()))

    def _on_camera_info(self, msg: CameraInfo) -> None:
        with self._camera_info_lock:
            self._camera_info = msg

    def _on_depth(self, msg: Image) -> None:
        self.depth_received += 1
        if self._current is not None:
            self._current.put_depth(msg)
        self._mailbox.put_depth(msg)

    def _on_depth_camera_info(self, msg: CameraInfo) -> None:
        with self._camera_info_lock:
            self._depth_camera_info = msg

    def _on_get_person_crop(
        self, request: GetPersonCrop.Request, response: GetPersonCrop.Response
    ) -> GetPersonCrop.Response:
        started = time.perf_counter()
        answer = self._person_crop(request)
        ms = (time.perf_counter() - started) * 1000.0
        line = f"person crop for track {request.track_id}: {CropStatus(answer.status).name} in {ms:.0f} ms"
        if answer.status == CropStatus.OK:
            line += f", {len(answer.crop.data) / 1024:.0f} KB"
        if answer.message:
            line += f"; {answer.message}"
        self.get_logger().info(line)
        return answer

    def _person_crop(self, request: GetPersonCrop.Request) -> GetPersonCrop.Response:
        """Leave the request for the worker and wait for its answer, up to timeout_s of wall time."""
        problem = crop_request_problem(request)
        if problem is not None:
            return _crop_answer(CropStatus.FAILED, problem)
        job = CropJob(request, self.frames_received)
        if not self._crops.put(job):
            return _crop_answer(CropStatus.FAILED, "the node is shutting down")
        self.crop_requests += 1
        timeout = float(request.timeout_s)
        answer = job.wait(timeout)
        if answer is not None:
            return answer
        self._crops.withdraw(job)
        if self.frames_received == job.frames_received:
            return _crop_answer(CropStatus.NO_IMAGE, f"no image arrived on {self.config.image_topic} in {timeout:g} s")
        return _crop_answer(CropStatus.TIMEOUT, f"images arrived, but none was processed in {timeout:g} s")

    def _on_embed_text(self, request: EmbedText.Request, response: EmbedText.Response) -> EmbedText.Response:
        if not self.config.object_embeddings_enabled:
            return convert.embed_text_failure(requests.DISABLED, "object_embeddings_enabled is off")
        problem = requests.embed_text_problem(request.texts)
        if problem is not None:
            return convert.embed_text_failure(requests.INVALID, problem)
        return self._serve(request, requests.EMBED_TEXT_DEADLINE_S, convert.embed_text_failure)

    def _on_embed_crop(self, request: EmbedCrop.Request, response: EmbedCrop.Response) -> EmbedCrop.Response:
        if not self.config.object_embeddings_enabled:
            return convert.embed_crop_failure(requests.DISABLED, "object_embeddings_enabled is off")
        problem = requests.embed_crop_problem(request)
        if problem is not None:
            return convert.embed_crop_failure(requests.INVALID, problem)
        return self._serve(request, float(request.timeout_s), convert.embed_crop_failure)

    def _on_describe_crop(self, request: DescribeCrop.Request, response: DescribeCrop.Response) -> DescribeCrop.Response:
        return DescribeCrop.Response(
            success=False, error_code=requests.DISABLED, message="this build has no caption model"
        )

    def _on_detect_objects(self, request: DetectObjects.Request, response: DetectObjects.Response) -> DetectObjects.Response:
        if not self.config.on_demand_enabled:
            return convert.detect_objects_failure(requests.DISABLED, "on_demand_enabled is off")
        problem = requests.detect_objects_problem(request)
        if problem is not None:
            return convert.detect_objects_failure(requests.INVALID, problem)
        return self._serve(request, float(request.timeout_s), convert.detect_objects_failure)

    def _serve(self, request: Any, deadline_s: float, fail: Callable[[str, str], Any]) -> Any:
        """Leave one job for the worker and wait for its answer, up to deadline_s of wall time."""
        job = requests.ServiceJob(request, time.monotonic() + deadline_s, fail)
        if not self._jobs.admit(job):
            if self._jobs.closed:
                return fail(requests.MODEL_UNAVAILABLE, "the node is shutting down")
            with self._counts_lock:
                self.service_busy += 1
            return fail(requests.BUSY, "another service job holds the worker")
        answer = job.wait()
        if answer is not None:
            return answer
        self._jobs.withdraw(job)
        return fail(requests.TIMEOUT, f"no answer in {deadline_s:g} s")

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
            if self.config.geometry_enabled:
                counts["depth"] = self.depth_received
                counts["paired"] = self.depth_paired
                counts["no_depth"] = sum(self.no_depth.values())
            if self.config.object_embeddings_enabled or self.config.on_demand_enabled:
                counts["jobs"] = self.service_jobs
                counts["busy"] = self.service_busy
                counts["late"] = self.late_results
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
                # The slot stays taken through the frame after a job, so the
                # next request cannot get in before the newest image.
                job = self._jobs.claim()
                try:
                    if job is not None:
                        self._run_job(job)
                    item = self._mailbox.take(0.0 if job is not None else 0.5)
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
                finally:
                    if job is not None:
                        self._jobs.release(job)
        except BaseException as e:
            self.get_logger().fatal(f"the worker stopped:\n{traceback.format_exc()}")
            self.fatal = f"the worker stopped: {type(e).__name__}: {e}"

    def _run_job(self, job: requests.ServiceJob) -> None:
        """Run one service job. A failure answers it, and leaves the stream and its counters as they are."""
        if job.given_up or job.remaining() <= 0:
            return  # its callback answers TIMEOUT
        with self._counts_lock:
            self.service_jobs += 1
        started = self._busy_since = time.monotonic()
        try:
            if isinstance(job.request, EmbedText.Request):
                response = convert.fill_embed_text_response(EmbedText.Response(), self._embedder, job.request.texts)
                if not response.success:
                    self._warn_throttled("service job", f"a service job failed: {response.message}")
                    response = job.fail(requests.INFERENCE_FAILED, response.message)
            elif isinstance(job.request, DetectObjects.Request):
                response = self._detect_objects(job)
            else:
                response = self._embed_crop(job.request)
        except Exception as e:
            self._warn_throttled("service job", f"a service job failed: {type(e).__name__}: {e}")
            response = job.fail(requests.INFERENCE_FAILED, f"{type(e).__name__}: {e}")
        finally:
            self._busy_since = None
        if not job.finish(response):
            with self._counts_lock:
                self.late_results += 1
            # A deadline cannot stop a model call. The stream waited this long.
            self._warn_throttled(
                "late",
                f"a service answer came after its deadline and was discarded; the job held the worker for "
                f"{time.monotonic() - started:.2f} s",
            )

    def _detect_objects(self, job: requests.ServiceJob) -> DetectObjects.Response:
        """Answer a DetectObjects job from the newest image after its classes are ready.

        The image is chosen only after the classes are prepared, so a slow
        preparation cannot make it old before the detector runs. Its age is
        checked again before success, so an old image is never labelled current.
        """
        request = job.request
        fail = convert.detect_objects_failure
        cfg = self.config
        classes = list(request.classes)
        if self.pipeline.on_demand_classes != classes:
            if not self._class_gate.open():
                with self._counts_lock:
                    self.service_busy += 1
                return fail(requests.BUSY, f"new classes are prepared at most once every {self._class_gate.period_s:g} s")
            try:
                self.pipeline.prepare_on_demand(classes)
            except Exception as e:
                self._warn_throttled("service job", f"preparing on-demand classes failed: {type(e).__name__}: {e}")
                return fail(requests.MODEL_UNAVAILABLE, f"the classes could not be prepared: {type(e).__name__}: {e}")
            finally:
                self._class_gate.mark()
            with self._counts_lock:
                self.classes_prepared += 1
        if job.given_up or job.remaining() <= 0:
            return fail(requests.TIMEOUT, "the deadline passed while the classes were prepared")

        max_age_s = float(request.max_image_age_s)
        msg, depth_msg = self._newest_bundle(max_age_s)
        if msg is None:
            return fail(requests.NO_IMAGE, f"no image has arrived on {cfg.image_topic}")
        problem = requests.image_age_problem(msg, self.get_clock().now().nanoseconds, max_age_s)
        if problem is not None:
            return fail(requests.STALE_IMAGE, problem)
        with self._camera_info_lock:
            info = self._camera_info
        problem = camera_info_problem(info, int(msg.width), int(msg.height))
        if problem is not None:
            return fail(requests.NO_IMAGE, f"{problem} on {cfg.camera_info_topic}")
        depth = self._current_depth(msg, depth_msg)
        if depth is None and cfg.require_depth:
            return fail(requests.NO_IMAGE, f"the newest image has no usable depth from {cfg.depth_topic}")
        try:
            image = image_to_bgr(msg)
        except ValueError as e:
            return fail(requests.NO_IMAGE, f"the newest image cannot be read: {e}")

        header = Header(stamp=msg.header.stamp, frame_id=msg.header.frame_id)
        result = self.pipeline.detect_on_demand(image, header, info, depth)
        if result.dropped:
            with self._counts_lock:
                self.dropped.update(result.dropped)
        problem = requests.image_age_problem(msg, self.get_clock().now().nanoseconds, max_age_s)
        if problem is not None:
            return fail(requests.STALE_IMAGE, f"after detection, {problem}")
        return DetectObjects.Response(success=True, objects=result.objects)

    def _newest_bundle(self, max_age_s: float) -> tuple[Image | None, Image | None]:
        """Return the newest image, with its depth when geometry is on and a matching pair is still current.

        A pair can be one image older than the newest colour. That is better
        than an image with no points, as long as the pair is within the age.
        """
        colour, _ = self._current.read()
        if self.config.geometry_enabled:
            pair = self._current.read_pair()
            now_ns = self.get_clock().now().nanoseconds
            if pair is not None and requests.image_age_problem(pair[0], now_ns, max_age_s) is None:
                return pair
        return colour, None

    def _embed_crop(self, request: EmbedCrop.Request) -> EmbedCrop.Response:
        """Answer an EmbedCrop request from the current image, if the request names it."""
        cfg = self.config
        msg, depth_msg = self._current.read()
        if msg is None:
            return convert.embed_crop_failure(requests.NO_IMAGE, f"no image has arrived on {cfg.image_topic}")
        problem = requests.crop_image_problem(request, msg, self.get_clock().now().nanoseconds)
        if problem is not None:
            return convert.embed_crop_failure(requests.STALE_IMAGE, problem)
        return requests.embed_crop_answer(
            self._embedder,
            image_to_bgr(msg),
            Header(stamp=msg.header.stamp, frame_id=msg.header.frame_id),
            convert.bbox_to_box(request.bbox),
            self._current_depth(msg, depth_msg),
            cfg.depth_min_m,
            cfg.depth_max_m,
        )

    def _current_depth(self, msg: Image, depth_msg: Image | None) -> RegisteredDepth | None:
        """Return the checked depth of the current image, or None. A service job counts nothing here."""
        cfg = self.config
        if depth_msg is None or abs(stamp_ns(msg.header) - stamp_ns(depth_msg.header)) > round(cfg.max_pair_gap_s * 1e9):
            return None
        with self._camera_info_lock:
            info = self._camera_info
            depth_info = info if cfg.depth_camera_info_topic == cfg.camera_info_topic else self._depth_camera_info
        try:
            return read_registered_depth(depth_msg, depth_info, msg, info)
        except DepthRefused:
            return None

    def _read_depth(self, msg: Image, info: CameraInfo, depth_msg: Image | None) -> RegisteredDepth | None:
        """Return the checked depth of an image, or None after counting why there is none.

        An unpaired image and a refused pair count alike: neither can give
        points. With require_depth they count as skipped.
        """
        cfg = self.config
        if depth_msg is None:
            reason, problem = "unpaired", f"no depth from {cfg.depth_topic} within {cfg.max_pair_gap_s:g} s of the image"
        else:
            with self._camera_info_lock:
                depth_info = info if cfg.depth_camera_info_topic == cfg.camera_info_topic else self._depth_camera_info
            try:
                depth = read_registered_depth(depth_msg, depth_info, msg, info)
            except DepthRefused as e:
                reason, problem = e.reason, str(e)
            else:
                with self._counts_lock:
                    self.depth_paired += 1
                return depth
        with self._counts_lock:
            (self.skipped if cfg.require_depth else self.no_depth)[reason] += 1
        action = "skipping the image" if cfg.require_depth else "publishing without points"
        self._warn_throttled(f"depth {reason}", f"{action}: {problem}")
        return None

    def _process_one(self, msg: Image, received: float, depth_msg: Image | None = None) -> bool:
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
        depth: RegisteredDepth | None = None
        if cfg.geometry_enabled:
            depth = self._read_depth(msg, info, depth_msg)
            if depth is None and cfg.require_depth:
                return True
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
            result = self.pipeline.process(image, header, info, depth)
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
        # Taken as soon as this image's people are known, so a request that
        # arrived while they were found is answered from this image, and one
        # that arrives after their array is published waits for the next.
        job = self._crops.take()
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
            if job is not None:
                job.finish(_crop_answer(CropStatus.FAILED, "the node is shutting down"))
            return False
        t = time.perf_counter()
        try:
            # People first, as oracle_perception does.
            self._people_pub.publish(result.people)
            self._objects_pub.publish(result.objects)
        except Exception:
            if job is not None:
                job.finish(_crop_answer(CropStatus.FAILED, "the arrays of the image could not be published"))
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
        # After the arrays, so a crop never delays them.
        if job is not None:
            self._answer_crop(job, image, result.people)
        return True

    def _answer_crop(self, job: CropJob, image: np.ndarray, people: PersonDetectionArray) -> None:
        """Answer a GetPersonCrop job from one image and its published people.

        A crop that fails answers FAILED; the worker goes on.
        """
        request = job.request
        try:
            found = crop.crop_person(
                image,
                [convert.pose_from_person_detection(p) for p in people.people],
                int(request.track_id),
                padding=float(request.padding),
                max_side=int(request.max_side),
                mask_others=bool(request.mask_others),
                min_keypoint_score=float(request.min_keypoint_score),
            )
            response = convert.fill_person_crop_response(GetPersonCrop.Response(), people, found)
        except Exception as e:
            self.get_logger().error(f"a person crop failed:\n{traceback.format_exc()}")
            response = _crop_answer(CropStatus.FAILED, f"{type(e).__name__}: {e}")
        job.finish(response)


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
    executor: MultiThreadedExecutor | None = None
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
            # Enough threads that the image callback runs while a GetPersonCrop
            # callback and a service job's callback wait for the worker, and a
            # second request can still get BUSY (ADR-0015).
            executor = MultiThreadedExecutor(num_threads=EXECUTOR_THREADS)
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
    if executor is not None:
        # Joins the executor's threads. close() has answered a waiting
        # request, so none of them waits for long.
        executor.shutdown()
    rclpy.try_shutdown()
    return code


if __name__ == "__main__":
    sys.exit(main())
