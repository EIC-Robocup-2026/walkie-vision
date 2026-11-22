"""The vision node's declared parameters and endpoints, frozen. No ROS graph traffic.

Each test builds VisionNode with fake providers in a private rclpy context,
from the same --ros-args the launch file passes (tasks hri and memory, with
and without sim.yaml), or with only classes set, and compares what the node
declares with the tables below:

- DEFAULTS: every declared parameter's type, value and read_only, rclpy's
  own use_sim_time and start_type_description_service included;
- CHANGED_BY_LAUNCH: per launch combination, the parameters whose row
  differs from DEFAULTS;
- ENDPOINTS: per node, the publishers, subscriptions, services and clients
  with their type names and QoS;
- DESCRIBE_CROP: the digest of the DescribeCrop answer.

The tables were frozen on the flat package layout (ad7c833), before its
modules moved: moving code changes no row, so a changed row means the node
reads or serves something else. The one change since is the rename of the
model parameters to <task>.<provider>.<option>, with a <task>.provider per
vision task: RENAMED maps each old name to its new one and its row at
ad7c833, and DEFAULTS was refrozen with the new names.
test_every_flat_override_lands sets each declared parameter with its own -p
override, because rclpy drops an override for an undeclared name without a
message: a renamed parameter would leave the node on its default and nothing
else would fail. For the same reason test_each_old_name_lands_at_its_new_name
checks that the parameter files set each new name to the old value, and
test_no_parameter_file_names_an_old_parameter that none still sets an old one.
"""

from __future__ import annotations

import array
import hashlib
import importlib.util
import os
import re
import struct
from pathlib import Path

import numpy as np
import pytest
import rclpy
import yaml
from rclpy.parameter import Parameter
from rclpy.serialization import deserialize_message, serialize_message
from walkie_vision_msgs.srv import DescribeCrop

from walkie_vision.fakes import (
    FakeEmbedder,
    FakeObjectDetector,
    FakePersonFaceRecognizer,
    FakePoseEstimator,
    FakeReIdentifier,
)
from walkie_vision.node import VisionNode
from walkie_vision.pipeline import Providers, TrackIdBase

REPO = Path(__file__).resolve().parents[1]
COMBINATIONS = {"hri": ("hri", False), "hri sim": ("hri", True), "memory": ("memory", False), "memory sim": ("memory", True)}
ROS_OWN_PARAMETERS = ("start_type_description_service", "use_sim_time")
"""rclpy declares these on every node; every other row of DEFAULTS is a parameter VisionConfig.from_node() declares."""

DEFAULTS = {
    "camera_info_topic": ("STRING", "/zed_head/zed_node/depth/camera_info", True),
    "captions_enabled": ("BOOL", False, True),
    "classes": ("STRING_ARRAY", ["chair"], True),
    "depth_camera_info_topic": ("STRING", "/zed_head/zed_node/depth/camera_info", True),
    "depth_max_m": ("DOUBLE", 3.0, True),
    "depth_min_m": ("DOUBLE", 0.2, True),
    "depth_topic": ("STRING", "/zed_head/zed_node/depth/depth_registered", True),
    "describe_crop_service": ("STRING", "/walkie_vision/describe_crop", True),
    "detect_objects_service": ("STRING", "/walkie_vision/detect_objects", True),
    "device": ("STRING", "cuda:0", True),
    "embed_crop_service": ("STRING", "/walkie_vision/embed_crop", True),
    "embed_text_service": ("STRING", "/walkie_vision/embed_text", True),
    "face_recognition.buffalo_l.det_thresh": ("DOUBLE", 0.5, True),
    "face_recognition.buffalo_l.input_sizes": ("INTEGER_ARRAY", [160, 256, 384, 640], True),
    "face_recognition.buffalo_l.max_offset": ("DOUBLE", 0.25, True),
    "face_recognition.buffalo_l.max_people": ("INTEGER", 6, True),
    "face_recognition.provider": ("STRING", "buffalo_l", True),
    "geometry_enabled": ("BOOL", False, True),
    "image_embed.provider": ("STRING", "clip", True),
    "image_topic": ("STRING", "/zed_head/zed_node/rgb/color/rect/image", True),
    "latency_log_period_s": ("DOUBLE", 10.0, True),
    "max_consecutive_failures": ("INTEGER", 30, True),
    "max_pair_gap_s": ("DOUBLE", 0.005, True),
    "max_rate_hz": ("DOUBLE", 10.0, True),
    "object_crops_enabled": ("BOOL", False, True),
    "object_detection.provider": ("STRING", "yoloe", True),
    "object_detection.yoloe.agnostic_nms": ("BOOL", True, True),
    "object_detection.yoloe.conf": ("DOUBLE", 0.1, True),
    "object_detection.yoloe.imgsz": ("INTEGER", 640, True),
    "object_detection.yoloe.tracker": ("STRING", "bytetrack.yaml", True),
    "object_detection.yoloe.weights": ("STRING", "yoloe-26s-seg.pt", True),
    "object_embeddings_enabled": ("BOOL", False, True),
    "objects_min_score": ("DOUBLE", 0.25, True),
    "objects_topic": ("STRING", "/walkie_vision/objects", True),
    "on_demand_enabled": ("BOOL", False, True),
    "pair_wait_s": ("DOUBLE", 0.06, True),
    "people_dedupe_iou": ("DOUBLE", 0.5, True),
    "people_min_score": ("DOUBLE", 0.25, True),
    "people_topic": ("STRING", "/walkie_vision/people", True),
    "person_crop_service": ("STRING", "/walkie_vision/get_person_crop", True),
    "pose_estimation.provider": ("STRING", "yolo_pose", True),
    "pose_estimation.yolo_pose.conf": ("DOUBLE", 0.1, True),
    "pose_estimation.yolo_pose.imgsz": ("INTEGER", 640, True),
    "pose_estimation.yolo_pose.tracker": ("STRING", "botsort.yaml", True),
    "pose_estimation.yolo_pose.weights": ("STRING", "yolo26s-pose.pt", True),
    "process_timeout_s": ("DOUBLE", 60.0, True),
    "reid.osnet.batch_size": ("INTEGER", 16, True),
    "reid.provider": ("STRING", "osnet", True),
    "reid_min_crop_px": ("INTEGER", 16, True),
    "require_depth": ("BOOL", False, True),
    "stale_warn_s": ("DOUBLE", 0.5, True),
    "start_type_description_service": ("BOOL", True, True),
    "use_sim_time": ("BOOL", False, False),
}
"""Every declared parameter of a node with only classes set: (type, value, read_only)."""

RENAMED = {
    "objects_weights": ("object_detection.yoloe.weights", ("STRING", "yoloe-26s-seg.pt", True)),
    "objects_imgsz": ("object_detection.yoloe.imgsz", ("INTEGER", 640, True)),
    "objects_conf": ("object_detection.yoloe.conf", ("DOUBLE", 0.1, True)),
    "objects_agnostic_nms": ("object_detection.yoloe.agnostic_nms", ("BOOL", True, True)),
    "objects_tracker": ("object_detection.yoloe.tracker", ("STRING", "bytetrack.yaml", True)),
    "pose_weights": ("pose_estimation.yolo_pose.weights", ("STRING", "yolo26s-pose.pt", True)),
    "pose_imgsz": ("pose_estimation.yolo_pose.imgsz", ("INTEGER", 640, True)),
    "pose_conf": ("pose_estimation.yolo_pose.conf", ("DOUBLE", 0.1, True)),
    "pose_tracker": ("pose_estimation.yolo_pose.tracker", ("STRING", "botsort.yaml", True)),
    "face_det_thresh": ("face_recognition.buffalo_l.det_thresh", ("DOUBLE", 0.5, True)),
    "face_input_sizes": ("face_recognition.buffalo_l.input_sizes", ("INTEGER_ARRAY", [160, 256, 384, 640], True)),
    "face_max_people": ("face_recognition.buffalo_l.max_people", ("INTEGER", 6, True)),
    "face_max_offset": ("face_recognition.buffalo_l.max_offset", ("DOUBLE", 0.25, True)),
    "reid_batch_size": ("reid.osnet.batch_size", ("INTEGER", 16, True)),
}
"""The flat model parameters of the flat layout, by old name: the new name, and the row frozen at ad7c833.

No launch combination changed any of them, so the row holds in every combination.
"""

PROVIDERS = {
    "object_detection.provider": "yoloe",
    "pose_estimation.provider": "yolo_pose",
    "face_recognition.provider": "buffalo_l",
    "reid.provider": "osnet",
    "image_embed.provider": "clip",
}
"""The parameters that choose each vision task's provider, with the one provider each registry has."""

UNTRACKED_FOLDERS = {
    ".git",
    ".pixi",
    ".pytest_cache",
    "__pycache__",
    "bakeoff-data",
    "checks/output",
    "models",
    "ros2_ws/build",
    "ros2_ws/install",
    "ros2_ws/log",
    "site",
    "weights",
}
"""Folders the YAML scan skips, by name or by path from the repository root: what .gitignore leaves out."""

HRI_CLASSES = ["chair", "sofa", "armchair", "stool"]
MEMORY_CLASSES = ["cup", "bottle", "bowl", "apple", "banana", "backpack", "chair", "table", "shelf", "cabinet", "sofa"]
SIM_NAMES = {
    "describe_crop_service": ("STRING", "/walkie_vision/sim/describe_crop", True),
    "detect_objects_service": ("STRING", "/walkie_vision/sim/detect_objects", True),
    "embed_crop_service": ("STRING", "/walkie_vision/sim/embed_crop", True),
    "embed_text_service": ("STRING", "/walkie_vision/sim/embed_text", True),
    "objects_topic": ("STRING", "/walkie_vision/sim/objects", True),
    "people_topic": ("STRING", "/walkie_vision/sim/people", True),
    "person_crop_service": ("STRING", "/walkie_vision/sim/get_person_crop", True),
    "use_sim_time": ("BOOL", True, False),
}
MEMORY_SWITCHES = {
    "geometry_enabled": ("BOOL", True, True),
    "object_crops_enabled": ("BOOL", True, True),
    "object_embeddings_enabled": ("BOOL", True, True),
    "on_demand_enabled": ("BOOL", True, True),
    "require_depth": ("BOOL", True, True),
}
CHANGED_BY_LAUNCH = {
    "hri": {"classes": ("STRING_ARRAY", HRI_CLASSES, True)},
    "hri sim": {"classes": ("STRING_ARRAY", HRI_CLASSES, True), **SIM_NAMES},
    "memory": {"classes": ("STRING_ARRAY", MEMORY_CLASSES, True), **MEMORY_SWITCHES},
    "memory sim": {"classes": ("STRING_ARRAY", MEMORY_CLASSES, True), **MEMORY_SWITCHES, **SIM_NAMES},
}
"""Per launch combination, the parameters whose (type, value, read_only) differs from DEFAULTS."""

# One endpoint per line: kind, name, type, then QoS history, depth, reliability and durability.
ROS_OWN_ENDPOINTS = (
    "publisher /parameter_events rcl_interfaces/msg/ParameterEvent KEEP_LAST 1000 RELIABLE VOLATILE",
    "service walkie_vision/describe_parameters rcl_interfaces/srv/DescribeParameters KEEP_LAST 1000 RELIABLE VOLATILE",
    "service walkie_vision/get_parameter_types rcl_interfaces/srv/GetParameterTypes KEEP_LAST 1000 RELIABLE VOLATILE",
    "service walkie_vision/get_parameters rcl_interfaces/srv/GetParameters KEEP_LAST 1000 RELIABLE VOLATILE",
    "service walkie_vision/get_type_description type_description_interfaces/srv/GetTypeDescription"
    " KEEP_LAST 10 RELIABLE VOLATILE",
    "service walkie_vision/list_parameters rcl_interfaces/srv/ListParameters KEEP_LAST 1000 RELIABLE VOLATILE",
    "service walkie_vision/set_parameters rcl_interfaces/srv/SetParameters KEEP_LAST 1000 RELIABLE VOLATILE",
    "service walkie_vision/set_parameters_atomically rcl_interfaces/srv/SetParametersAtomically"
    " KEEP_LAST 1000 RELIABLE VOLATILE",
)
"""What rclpy creates on every node, named after the node."""

CAMERA_ENDPOINTS = (
    "subscription /zed_head/zed_node/depth/camera_info sensor_msgs/msg/CameraInfo KEEP_LAST 1 BEST_EFFORT VOLATILE",
    "subscription /zed_head/zed_node/rgb/color/rect/image sensor_msgs/msg/Image KEEP_LAST 1 BEST_EFFORT VOLATILE",
)
DEPTH_ENDPOINTS = (
    "subscription /zed_head/zed_node/depth/depth_registered sensor_msgs/msg/Image KEEP_LAST 1 BEST_EFFORT VOLATILE",
)
CLOCK_ENDPOINTS = ("subscription /clock rosgraph_msgs/msg/Clock KEEP_LAST 1 BEST_EFFORT VOLATILE",)
VISION_ENDPOINTS = (
    "publisher /walkie_vision/objects walkie_vision_msgs/msg/ObjectDetectionArray KEEP_LAST 5 RELIABLE VOLATILE",
    "publisher /walkie_vision/people walkie_vision_msgs/msg/PersonDetectionArray KEEP_LAST 5 RELIABLE VOLATILE",
    "service /walkie_vision/describe_crop walkie_vision_msgs/srv/DescribeCrop KEEP_LAST 10 RELIABLE VOLATILE",
    "service /walkie_vision/detect_objects walkie_vision_msgs/srv/DetectObjects KEEP_LAST 10 RELIABLE VOLATILE",
    "service /walkie_vision/embed_crop walkie_vision_msgs/srv/EmbedCrop KEEP_LAST 10 RELIABLE VOLATILE",
    "service /walkie_vision/embed_text walkie_vision_msgs/srv/EmbedText KEEP_LAST 10 RELIABLE VOLATILE",
    "service /walkie_vision/get_person_crop walkie_vision_msgs/srv/GetPersonCrop KEEP_LAST 10 RELIABLE VOLATILE",
)
SIM_VISION_ENDPOINTS = (
    "publisher /walkie_vision/sim/objects walkie_vision_msgs/msg/ObjectDetectionArray KEEP_LAST 5 RELIABLE VOLATILE",
    "publisher /walkie_vision/sim/people walkie_vision_msgs/msg/PersonDetectionArray KEEP_LAST 5 RELIABLE VOLATILE",
    "service /walkie_vision/sim/describe_crop walkie_vision_msgs/srv/DescribeCrop KEEP_LAST 10 RELIABLE VOLATILE",
    "service /walkie_vision/sim/detect_objects walkie_vision_msgs/srv/DetectObjects KEEP_LAST 10 RELIABLE VOLATILE",
    "service /walkie_vision/sim/embed_crop walkie_vision_msgs/srv/EmbedCrop KEEP_LAST 10 RELIABLE VOLATILE",
    "service /walkie_vision/sim/embed_text walkie_vision_msgs/srv/EmbedText KEEP_LAST 10 RELIABLE VOLATILE",
    "service /walkie_vision/sim/get_person_crop walkie_vision_msgs/srv/GetPersonCrop KEEP_LAST 10 RELIABLE VOLATILE",
)
ENDPOINTS = {
    "classes only": ROS_OWN_ENDPOINTS + CAMERA_ENDPOINTS + VISION_ENDPOINTS,
    "hri": ROS_OWN_ENDPOINTS + CAMERA_ENDPOINTS + VISION_ENDPOINTS,
    "hri sim": ROS_OWN_ENDPOINTS + CAMERA_ENDPOINTS + CLOCK_ENDPOINTS + SIM_VISION_ENDPOINTS,
    "memory": ROS_OWN_ENDPOINTS + CAMERA_ENDPOINTS + DEPTH_ENDPOINTS + VISION_ENDPOINTS,
    "memory sim": ROS_OWN_ENDPOINTS + CAMERA_ENDPOINTS + DEPTH_ENDPOINTS + CLOCK_ENDPOINTS + SIM_VISION_ENDPOINTS,
}
"""Every endpoint of the node, per launch combination and for the node with only classes set."""

DESCRIBE_CROP = "7b47a71145708e8a264e4f8d"
"""cdr() of the DescribeCrop answer of the hri node: DISABLED, this build has no caption model."""

OVERRIDES = {
    "classes": ("[cup, sofa]", ["cup", "sofa"]),
    "image_topic": ("/override/image", "/override/image"),
    "camera_info_topic": ("/override/camera_info", "/override/camera_info"),
    "people_topic": ("/override/people", "/override/people"),
    "objects_topic": ("/override/objects", "/override/objects"),
    "person_crop_service": ("/override/get_person_crop", "/override/get_person_crop"),
    "embed_text_service": ("/override/embed_text", "/override/embed_text"),
    "embed_crop_service": ("/override/embed_crop", "/override/embed_crop"),
    "describe_crop_service": ("/override/describe_crop", "/override/describe_crop"),
    "detect_objects_service": ("/override/detect_objects", "/override/detect_objects"),
    "device": ("cpu", "cpu"),
    "object_detection.provider": ("unknown", None),
    "object_detection.yoloe.weights": ("yoloe-26m-seg.pt", "yoloe-26m-seg.pt"),
    "object_detection.yoloe.imgsz": ("320", 320),
    "object_detection.yoloe.conf": ("0.3", 0.3),
    "object_detection.yoloe.agnostic_nms": ("false", False),
    "object_detection.yoloe.tracker": ("botsort.yaml", "botsort.yaml"),
    "objects_min_score": ("0.4", 0.4),
    "pose_estimation.provider": ("unknown", None),
    "pose_estimation.yolo_pose.weights": ("yolo26m-pose.pt", "yolo26m-pose.pt"),
    "pose_estimation.yolo_pose.imgsz": ("320", 320),
    "pose_estimation.yolo_pose.conf": ("0.3", 0.3),
    "pose_estimation.yolo_pose.tracker": ("bytetrack.yaml", "bytetrack.yaml"),
    "people_min_score": ("0.4", 0.4),
    "people_dedupe_iou": ("0.7", 0.7),
    "face_recognition.provider": ("unknown", None),
    "face_recognition.buffalo_l.det_thresh": ("0.6", 0.6),
    "face_recognition.buffalo_l.input_sizes": ("[320, 640]", [320, 640]),
    "face_recognition.buffalo_l.max_people": ("3", 3),
    "face_recognition.buffalo_l.max_offset": ("0.5", 0.5),
    "reid_min_crop_px": ("32", 32),
    "reid.provider": ("unknown", None),
    "reid.osnet.batch_size": ("8", 8),
    "image_embed.provider": ("unknown", None),
    "max_rate_hz": ("5.0", 5.0),
    "max_consecutive_failures": ("10", 10),
    "process_timeout_s": ("30.0", 30.0),
    "latency_log_period_s": ("5.0", 5.0),
    "stale_warn_s": ("1.0", 1.0),
    "geometry_enabled": ("true", True),
    "depth_topic": ("/override/depth", "/override/depth"),
    "depth_camera_info_topic": ("/override/depth_camera_info", "/override/depth_camera_info"),
    "max_pair_gap_s": ("0.01", 0.01),
    "pair_wait_s": ("0.05", 0.05),
    "require_depth": ("true", True),
    "depth_min_m": ("0.5", 0.5),
    "depth_max_m": ("5.0", 5.0),
    "object_crops_enabled": ("true", True),
    "object_embeddings_enabled": ("true", True),
    "on_demand_enabled": ("true", True),
    "captions_enabled": ("true", True),
}
"""Per declared parameter: a -p value other than the default, and the value the node must report.

A <task>.provider has no other valid value while each registry has one
provider, so its override must be refused with the parameter's name (None).
"""

NEEDS = {"require_depth": ["-p", "geometry_enabled:=true"]}
"""Other overrides a field's value needs to pass validation."""


def _feed(value, h) -> None:
    """Hash a message value field by field, with exact bit patterns and no CDR padding."""
    if hasattr(value, "get_fields_and_field_types"):
        h.update(b"{" + type(value).__name__.encode())
        for name in value.get_fields_and_field_types():
            h.update(b"." + name.encode() + b"=")
            _feed(getattr(value, name), h)
        h.update(b"}")
    elif isinstance(value, array.array):
        h.update(b"a" + value.typecode.encode() + len(value).to_bytes(8, "little") + value.tobytes())
    elif isinstance(value, np.ndarray):
        h.update(b"n" + value.dtype.str.encode() + repr(value.shape).encode() + np.ascontiguousarray(value).tobytes())
    elif isinstance(value, (bytes, bytearray)):
        h.update(b"b" + len(value).to_bytes(8, "little") + bytes(value))
    elif isinstance(value, (list, tuple)):
        h.update(b"[" + len(value).to_bytes(8, "little"))
        for item in value:
            _feed(item, h)
    elif isinstance(value, str):
        data = value.encode()
        h.update(b"s" + len(data).to_bytes(8, "little") + data)
    elif isinstance(value, bool):
        h.update(b"T" if value else b"F")
    elif isinstance(value, int):
        h.update(b"i" + str(value).encode() + b";")
    elif isinstance(value, float):
        h.update(b"d" + struct.pack("<d", value))
    else:
        raise TypeError(f"cannot hash a {type(value).__name__}")


def cdr(msg) -> str:
    """Digest of a message as a subscriber receives it, as in test_golden_messages.py."""
    h = hashlib.sha256()
    _feed(deserialize_message(serialize_message(msg), type(msg)), h)
    return h.hexdigest()[:24]


@pytest.fixture
def ctx():
    context = rclpy.Context()
    rclpy.init(context=context, domain_id=77)
    yield context
    context.try_shutdown()


@pytest.fixture(scope="module")
def vision_launch():
    spec = importlib.util.spec_from_file_location("vision_launch", REPO / "launch" / "vision.launch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _providers(classes) -> Providers:
    """Fakes for every vision task, so any combination of switches can start."""
    return Providers(
        objects=FakeObjectDetector([], classes=list(classes)),
        pose=FakePoseEstimator([]),
        faces=FakePersonFaceRecognizer([]),
        reid=FakeReIdentifier(),
        embedder=FakeEmbedder(),
        on_demand=FakeObjectDetector([]),
    )


def _node(ctx, classes, **node_kwargs) -> VisionNode:
    return VisionNode(_providers(classes), track_ids=TrackIdBase(start_ns=0), context=ctx, **node_kwargs)


def _launched_node(ctx, vision_launch, combination: str) -> VisionNode:
    """The node as the launch file starts it for one combination of task and sim."""
    task, sim = COMBINATIONS[combination]
    cmd = vision_launch.node_command(task, sim)
    return _node(ctx, vision_launch.node_parameters(task, sim)["classes"], cli_args=cmd[cmd.index("--ros-args") :])


def _classes_only_node(ctx) -> VisionNode:
    return _node(ctx, ["chair"], parameter_overrides=[Parameter("classes", value=["chair"])])


def _plain(value):
    """A parameter value as a plain list or scalar, whatever sequence type rclpy returns."""
    if isinstance(value, (list, tuple, array.array)):
        return list(value)
    return value


def _parameters(node) -> dict:
    """Every declared parameter of node: (type, value, read_only)."""
    rows = {}
    for name in node.list_parameters([], 0).names:
        p = node.get_parameter(name)
        rows[name] = (p.type_.name, _plain(p.value), node.describe_parameter(name).read_only)
    return dict(sorted(rows.items()))


def _type_name(t) -> str:
    parts = t.__module__.split(".")
    return f"{parts[0]}/{parts[1]}/{t.__name__}"


def _endpoint(kind: str, name: str, msg_type, qos) -> str:
    return (
        f"{kind} {name} {_type_name(msg_type)} {qos.history.name} {qos.depth} {qos.reliability.name} "
        f"{qos.durability.name}"
    )


def _endpoints(node) -> list[str]:
    rows = [_endpoint("publisher", p.topic_name, p.msg_type, p.qos_profile) for p in node.publishers]
    rows += [_endpoint("subscription", s.topic_name, s.msg_type, s.qos_profile) for s in node.subscriptions]
    rows += [_endpoint("service", s.srv_name, s.srv_type, s.qos_profile) for s in node.services]
    rows += [_endpoint("client", c.srv_name, c.srv_type, c.qos_profile) for c in node.clients]
    return sorted(rows)


def test_the_declared_parameters_and_their_defaults(ctx):
    node = _classes_only_node(ctx)
    try:
        assert _parameters(node) == DEFAULTS
    finally:
        assert node.close()
    assert len(DEFAULTS) == 53
    assert sorted(OVERRIDES) == sorted(set(DEFAULTS) - set(ROS_OWN_PARAMETERS))
    assert {new for new, _ in RENAMED.values()} | set(PROVIDERS) <= set(DEFAULTS)
    assert not set(RENAMED) & set(DEFAULTS)
    assert {name: DEFAULTS[name][1] for name in PROVIDERS} == PROVIDERS


@pytest.mark.parametrize("combination", sorted(COMBINATIONS))
def test_a_launch_combination_changes_only_these_parameters(ctx, vision_launch, combination):
    node = _launched_node(ctx, vision_launch, combination)
    try:
        rows = _parameters(node)
    finally:
        assert node.close()
    assert sorted(rows) == sorted(DEFAULTS)
    changed = {name: row for name, row in rows.items() if row != DEFAULTS[name]}
    assert changed == CHANGED_BY_LAUNCH[combination]


def _flat(parameters: dict, prefix: str = "") -> dict:
    """Nested blocks as dotted names, the way rcl reads a parameter file."""
    flat = {}
    for key, value in parameters.items():
        if isinstance(value, dict):
            flat.update(_flat(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def _file_parameters(path: Path) -> dict:
    data = yaml.safe_load(path.read_text()) or {}
    return _flat(data.get("walkie_vision", {}).get("ros__parameters", {}))


@pytest.mark.parametrize("combination", sorted(COMBINATIONS))
def test_each_old_name_lands_at_its_new_name(ctx, vision_launch, combination):
    # The value each flat model parameter had at ad7c833 is now set by the
    # parameter files at its dotted name and reaches the node there.
    task, sim = COMBINATIONS[combination]
    cmd = vision_launch.node_command(task, sim)
    files: dict = {}
    for i, arg in enumerate(cmd):
        if arg == "--params-file":
            files.update(_file_parameters(Path(cmd[i + 1])))
    node = _launched_node(ctx, vision_launch, combination)
    try:
        rows = _parameters(node)
    finally:
        assert node.close()
    for old, (new, row) in RENAMED.items():
        assert old not in files, f"{combination}: the parameter files still set {old}"
        assert old not in rows, f"{combination}: the node declares {old}"
        assert files.get(new) == row[1], f"{combination}: the parameter files do not set {new} to {row[1]!r}"
        assert rows[new] == row, f"{combination}: {new}"
    for name, provider in PROVIDERS.items():
        assert files.get(name) == provider, f"{combination}: {name}"


def _yaml_files() -> list[Path]:
    """Every YAML file in the repository, without the folders .gitignore leaves out."""
    paths = []
    for folder, subfolders, files in os.walk(REPO):
        here = Path(folder)
        subfolders[:] = sorted(
            name
            for name in subfolders
            if name not in UNTRACKED_FOLDERS and (here / name).relative_to(REPO).as_posix() not in UNTRACKED_FOLDERS
        )
        paths += [here / name for name in files if name.endswith((".yaml", ".yml"))]
    return sorted(paths)


def test_no_parameter_file_names_an_old_parameter():
    # No YAML file in the repository names an old parameter, in config/ or
    # anywhere else. rclpy ignores a name the node does not declare, so a flat
    # name left in a parameter file would fall back to its default without a
    # message.
    old = re.compile(r"\b(" + "|".join(RENAMED) + r")\b")
    paths = _yaml_files()
    assert REPO / "config" / "vision.yaml" in paths
    assert len(paths) >= 4
    for path in paths:
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            assert not old.search(line), f"{path.relative_to(REPO)}:{number}: {line.strip()}"


@pytest.mark.parametrize("combination", sorted(ENDPOINTS))
def test_a_node_has_these_endpoints(ctx, vision_launch, combination):
    if combination == "classes only":
        node = _classes_only_node(ctx)
    else:
        node = _launched_node(ctx, vision_launch, combination)
    try:
        assert _endpoints(node) == sorted(ENDPOINTS[combination])
    finally:
        assert node.close()


def test_describe_crop_answers_disabled(ctx, vision_launch):
    node = _launched_node(ctx, vision_launch, "hri")
    try:
        (service,) = [s for s in node.services if s.srv_type is DescribeCrop]
        answer = service.callback(DescribeCrop.Request(), DescribeCrop.Response())
    finally:
        assert node.close()
    assert cdr(answer) == DESCRIBE_CROP


@pytest.mark.parametrize("name", sorted(OVERRIDES))
def test_every_flat_override_lands(ctx, name):
    text, expected = OVERRIDES[name]
    args = ["--ros-args"]
    if name != "classes":
        args += ["-p", "classes:=[chair]"]
    args += NEEDS.get(name, []) + ["-p", f"{name}:={text}"]
    refusal = {"captions_enabled": "captions_enabled must be false"}.get(name)
    if name in PROVIDERS:
        refusal = f"{name} must be one of {PROVIDERS[name]}, not 'unknown'"
    if refusal:
        # A parameter with no other valid value: the refusal shows the override was read.
        try:
            node = _node(ctx, ["chair"], cli_args=args)
        except ValueError as e:
            assert refusal in str(e)
            return
        assert node.close()
        pytest.fail(f"the node started although {name}:={text} was passed")
    node = _node(ctx, expected if name == "classes" else ["chair"], cli_args=args)
    try:
        rows = _parameters(node)
        config_value = _plain(node.config.parameters()[name])
    finally:
        assert node.close()
    assert name in rows, f"the node declares no parameter {name}"
    assert rows[name][1] != DEFAULTS[name][1]
    assert rows[name] == (DEFAULTS[name][0], expected, True)
    assert config_value == expected
