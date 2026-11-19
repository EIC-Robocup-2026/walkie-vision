"""The vision node's parameters: VisionConfig, with the check of every value.

Every field of VisionConfig is a read-only ROS parameter of the vision node
(walkie_vision.node), and config/vision.yaml sets each to its default.
VisionConfig.from_node() declares them on the node and reads them back. This
module imports rclpy for that and for the name checks, and no model library.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from rcl_interfaces.msg import ParameterDescriptor
from rclpy.exceptions import NameValidationException, ParameterException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.validate_full_topic_name import validate_full_topic_name

from .providers._ultralytics import TRACKER_NAMES

MAX_PAIR_WINDOW_S = 0.1
"""The largest max_pair_gap_s and pair_wait_s: the output must stay near 5 Hz, and a wider gap pairs other captures."""

_DEVICE = re.compile(r"^(cpu|cuda(:\d+)?)$")


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
            if tracker not in TRACKER_NAMES and not (Path(tracker).is_absolute() and tracker.endswith(".yaml")):
                raise ValueError(
                    f"{name} must be one of {', '.join(TRACKER_NAMES)} or an absolute path to a .yaml file, "
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
