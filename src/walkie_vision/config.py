"""The vision node's parameters: VisionConfig, with the checks of the flat ones.

VisionConfig holds every read-only ROS parameter of the vision node
(walkie_vision.node), and config/vision.yaml sets each to its default. Most
parameters are flat. The models are chosen per vision task: the parameter
<task>.provider names a provider in the task's registry
(walkie_vision.providers.<task>.registry), and that provider's options are
the parameters <task>.<provider>.<option>, checked by its options dataclass.
VisionConfig.from_node() declares them on the node and reads them back. This
module imports rclpy for that and for the name checks, and no model library.
"""

from __future__ import annotations

import math
import re
from dataclasses import MISSING, dataclass, field, fields
from types import ModuleType
from typing import Any

from rcl_interfaces.msg import ParameterDescriptor
from rclpy.exceptions import NameValidationException, ParameterException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.validate_full_topic_name import validate_full_topic_name

from .providers.face_recognition import registry as face_recognition_registry
from .providers.face_recognition.registry import BuffaloLOptions
from .providers.image_embed import registry as image_embed_registry
from .providers.image_embed.registry import ClipOptions
from .providers.object_detection import registry as object_detection_registry
from .providers.object_detection.registry import YoloeOptions
from .providers.pose_estimation import registry as pose_estimation_registry
from .providers.pose_estimation.registry import YoloPoseOptions
from .providers.reid import registry as reid_registry
from .providers.reid.registry import OsnetOptions
from .providers.spec import ProviderSpec

MAX_PAIR_WINDOW_S = 0.1
"""The largest max_pair_gap_s and pair_wait_s: the output must stay near 5 Hz, and a wider gap pairs other captures."""

_DEVICE = re.compile(r"^(cpu|cuda(:\d+)?)$")

REGISTRIES: dict[str, dict[str, ProviderSpec[Any]]] = {
    "object_detection": object_detection_registry.PROVIDERS,
    "pose_estimation": pose_estimation_registry.PROVIDERS,
    "face_recognition": face_recognition_registry.PROVIDERS,
    "reid": reid_registry.PROVIDERS,
    "image_embed": image_embed_registry.PROVIDERS,
}
"""Each vision task's registry, in the order the node checks the files and prepares the Hugging Face cache.

VisionConfig has two fields per task: <task>_provider, the provider's key in
the registry, and <task>, that provider's options.
"""

_TASK_FIELDS = frozenset(REGISTRIES) | {f"{task}_provider" for task in REGISTRIES}


def _default_options(registry: ModuleType) -> Any:
    """A field whose default is the options of the registry's default provider, with their own defaults."""
    return field(default_factory=registry.PROVIDERS[registry.DEFAULT_PROVIDER].options)


def _check_provider(task: str, provider: str) -> None:
    if provider not in REGISTRIES[task]:
        raise ValueError(f"{task}.provider must be one of {', '.join(REGISTRIES[task])}, not {provider!r}")


def _declare(node: Node, name: str, default: Any) -> Any:
    """Declare one read-only parameter on node and return its value, with lists as tuples.

    A MISSING default declares a string array that the parameter files must set.
    """
    descriptor = ParameterDescriptor(read_only=True)
    try:
        if default is MISSING:
            node.declare_parameter(name, Parameter.Type.STRING_ARRAY, descriptor)
        else:
            node.declare_parameter(name, list(default) if isinstance(default, tuple) else default, descriptor)
        # get_parameter() raises for a typed parameter that was never set.
        value = node.get_parameter(name).value
    except (ParameterException, TypeError, NameValidationException) as e:
        raise ValueError(f"parameter {name}: {e}") from e
    return tuple(value) if default is MISSING or isinstance(default, tuple) else value


@dataclass(frozen=True)
class VisionConfig:
    """The node's parameters. The defaults equal config/vision.yaml.

    A flat field is the read-only ROS parameter of the same name. classes
    has no default: it comes from the task's file in config/tasks/. Each
    vision task has two fields: <task>_provider is the parameter
    <task>.provider, and <task> holds that provider's options, the
    parameters <task>.<provider>.<option>. Their defaults come from the
    task's registry, so a config built in Python with flat fields only is
    the node's default config.
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
    object_detection_provider: str = object_detection_registry.DEFAULT_PROVIDER
    object_detection: YoloeOptions = _default_options(object_detection_registry)
    objects_min_score: float = 0.25
    pose_estimation_provider: str = pose_estimation_registry.DEFAULT_PROVIDER
    pose_estimation: YoloPoseOptions = _default_options(pose_estimation_registry)
    people_min_score: float = 0.25
    people_dedupe_iou: float = 0.5
    face_recognition_provider: str = face_recognition_registry.DEFAULT_PROVIDER
    face_recognition: BuffaloLOptions = _default_options(face_recognition_registry)
    reid_min_crop_px: int = 16
    reid_provider: str = reid_registry.DEFAULT_PROVIDER
    reid: OsnetOptions = _default_options(reid_registry)
    max_rate_hz: float = 10.0
    max_consecutive_failures: int = 30
    process_timeout_s: float = 60.0
    latency_log_period_s: float = 10.0
    stale_warn_s: float = 0.5
    geometry_enabled: bool = False
    depth_topic: str = "/zed_head/zed_node/depth/depth_registered"
    depth_camera_info_topic: str = "/zed_head/zed_node/depth/camera_info"
    max_pair_gap_s: float = 0.005
    pair_wait_s: float = 0.060
    require_depth: bool = False
    depth_min_m: float = 0.2
    depth_max_m: float = 3.0
    object_crops_enabled: bool = False
    object_embeddings_enabled: bool = False
    image_embed_provider: str = image_embed_registry.DEFAULT_PROVIDER
    image_embed: ClipOptions = _default_options(image_embed_registry)
    on_demand_enabled: bool = False
    captions_enabled: bool = False

    def __post_init__(self) -> None:
        # Lists become tuples, so the config stays hashable and immutable.
        object.__setattr__(self, "classes", tuple(self.classes))

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
        # Each provider's options check their own values when they are built.
        for task, registry in REGISTRIES.items():
            provider = getattr(self, f"{task}_provider")
            _check_provider(task, provider)
            options = getattr(self, task)
            if not isinstance(options, registry[provider].options):
                raise ValueError(
                    f"{task}: {provider} takes {registry[provider].options.__name__}, not {type(options).__name__}"
                )

        for name in ("objects_min_score", "people_min_score"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], not {value}")
        if not 0.0 < self.people_dedupe_iou <= 1.0:
            raise ValueError(f"people_dedupe_iou must be in (0, 1], not {self.people_dedupe_iou}")

        for name, low in (
            ("reid_min_crop_px", 1),
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

    def parameters(self) -> dict[str, Any]:
        """Every parameter that from_node() declares, by its ROS name, with its value here."""
        values = {f.name: getattr(self, f.name) for f in fields(self) if f.name not in _TASK_FIELDS}
        for task in REGISTRIES:
            provider = getattr(self, f"{task}_provider")
            options = getattr(self, task)
            values[f"{task}.provider"] = provider
            values.update({f"{task}.{provider}.{f.name}": getattr(options, f.name) for f in fields(options)})
        return values

    @classmethod
    def from_node(cls, node: Node) -> VisionConfig:
        """Declare every parameter on node, read-only, and return their values.

        The flat parameters come first. Then, per vision task, <task>.provider,
        and only the chosen provider's options, with the defaults of its
        options dataclass. Raises ValueError naming the parameter when one is
        missing, has the wrong type or a bad value, and naming the registry's
        providers when <task>.provider is not one of them.
        """
        defaults = {f.name: f.default for f in fields(cls)}
        values: dict[str, Any] = {}
        for name, default in defaults.items():
            if name not in _TASK_FIELDS:
                values[name] = _declare(node, name, MISSING if name == "classes" else default)
        for task, registry in REGISTRIES.items():
            provider = _declare(node, f"{task}.provider", defaults[f"{task}_provider"])
            _check_provider(task, provider)
            options = registry[provider].options
            values[f"{task}_provider"] = provider
            values[task] = options(
                **{f.name: _declare(node, f"{task}.{provider}.{f.name}", f.default) for f in fields(options)}
            )
        return cls(**values)
