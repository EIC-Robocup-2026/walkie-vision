"""A deterministic stand-in for the object stream: synthetic boxes and no models.

walkie-database's tests use this publisher to exercise the object contract
(ADR-0013) without a camera or a GPU. A scene file describes a fixed pinhole
camera, axis-aligned boxes in the map frame and a finite sequence of phases.

Each phase is rendered once, with one ray per pixel. A box that wins at least
one pixel gets a detection. Its box comes from those pixels, and its cloud
holds the surface points they hit, in the optical frame. A hidden box has
fewer pixels or none, and a box that a phase leaves out is not in the render
at all.

This is synthetic geometry. It tests plumbing, and it says nothing about how
well a real detector, depth image or embedder works. A vector here is a hash
of the box's name under a fake_ model name, and is never labelled as CLIP.

This module imports no model library. It has no people topic, and it refuses
the class `person`.
"""

from __future__ import annotations

import array
import json
import math
import signal
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Quaternion, Transform, TransformStamped, Vector3
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.clock import Clock, ClockType
from rclpy.exceptions import ParameterUninitializedException
from rclpy.executors import ExternalShutdownException
from rclpy.logging import get_logger
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Header
from tf2_msgs.msg import TFMessage
from walkie_vision_msgs.msg import ObjectDetection, ObjectDetectionArray

from . import convert, crop
from .boxes import pad_rect
from .embeddings import CROP_MAX_SIDE, CROP_PADDING, MIN_COLOR_PIXELS, PALETTE_RGB
from .fakes import hash_vector
from .providers import FAKE_MODEL_PREFIX, Detection

NODE_NAME = "walkie_vision_fake"
EXIT_STARTUP = 2
DEFAULT_SCENE = Path(__file__).resolve().parents[2] / "config" / "fakes" / "boxes.json"
_BACKGROUND_BGR = (60, 60, 60)
_NO_COLOR_BGR = (128, 128, 128)
_REFUSED_CLASS = "person"

# The vision node's OUTPUT_QOS, repeated so that this module does not import the
# node; tests/test_fake_publisher.py checks that the two are equal. RELIABLE
# subscribers match it, and so do BEST_EFFORT ones such as walkie-database's.
OBSERVATION_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)
# The usual profile of /tf_static, so that a late listener still gets the camera.
TF_STATIC_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


@dataclass(frozen=True, eq=False)
class FakeCamera:
    """A fixed pinhole camera with no distortion."""

    frame_id: str
    parent_frame: str
    position: np.ndarray
    orientation_xyzw: np.ndarray
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    min_depth: float
    max_depth: float

    def rotation(self) -> np.ndarray:
        """Return the 3 x 3 rotation from the optical frame to the parent frame."""
        x, y, z, w = self.orientation_xyzw
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )

    def camera_info(self, header: Header) -> CameraInfo:
        """Return the camera's calibration under the given header."""
        info = CameraInfo(header=header, width=self.width, height=self.height, distortion_model="plumb_bob")
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        info.k = [self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [self.fx, 0.0, self.cx, 0.0, 0.0, self.fy, self.cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return info

    def transform(self, stamp: Time) -> TFMessage:
        """Return the camera's static transform from the parent frame."""
        px, py, pz = (float(v) for v in self.position)
        qx, qy, qz, qw = (float(v) for v in self.orientation_xyzw)
        return TFMessage(
            transforms=[
                TransformStamped(
                    header=Header(stamp=stamp, frame_id=self.parent_frame),
                    child_frame_id=self.frame_id,
                    transform=Transform(
                        translation=Vector3(x=px, y=py, z=pz),
                        rotation=Quaternion(x=qx, y=qy, z=qz, w=qw),
                    ),
                )
            ]
        )


@dataclass(frozen=True, eq=False)
class FakeBox:
    """One axis-aligned box in the camera's parent frame."""

    name: str
    label: str
    track_id: int
    score: float
    view_quality: float
    color: str | None
    center: np.ndarray
    size: np.ndarray


@dataclass(frozen=True, eq=False)
class FakePhase:
    """A number of identical frames. A move stays until the sequence restarts."""

    frames: int
    visible: tuple[str, ...]
    move: Mapping[str, np.ndarray]


@dataclass(frozen=True, eq=False)
class FakeScene:
    """A parsed scene file."""

    camera: FakeCamera
    classes: tuple[str, ...]
    boxes: tuple[FakeBox, ...]
    phases: tuple[FakePhase, ...]


def _check_fields(data: Any, where: str, required: Sequence[str], optional: Sequence[str] = ()) -> None:
    if not isinstance(data, Mapping):
        raise ValueError(f"{where} must be an object")
    missing = sorted(set(required) - set(data))
    unknown = sorted(set(data) - set(required) - set(optional))
    if missing or unknown:
        raise ValueError(f"{where}: missing fields {missing}, unknown fields {unknown}")


def _number(value: Any, where: str, low: float = -math.inf, high: float = math.inf) -> float:
    """Return a finite number in [low, high]. JSON lets NaN through, so this checks for it."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a number, not {value!r}")
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{where} must be finite and in [{low}, {high}], not {value!r}")
    return float(value)


def _positive(value: Any, where: str) -> float:
    number = _number(value, where)
    if number <= 0:
        raise ValueError(f"{where} must be above 0, not {value!r}")
    return number


def _integer(value: Any, where: str, low: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < low:
        raise ValueError(f"{where} must be a whole number of at least {low}, not {value!r}")
    return value


def _vector(value: Any, size: int, where: str) -> np.ndarray:
    if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != size:
        raise ValueError(f"{where} must hold {size} numbers, not {value!r}")
    return np.array([_number(v, where) for v in value], dtype=np.float64)


def _name(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{where} must be a non-empty string without surrounding spaces, not {value!r}")
    return value


def _check_classes(classes: Any, where: str) -> tuple[str, ...]:
    if isinstance(classes, str) or not isinstance(classes, Sequence) or not classes:
        raise ValueError(f"{where} must be a list with at least one class, not {classes!r}")
    names = tuple(_name(c, where) for c in classes)
    if len(set(names)) != len(names):
        raise ValueError(f"{where} has duplicates: {list(names)}")
    if _REFUSED_CLASS in names:
        raise ValueError(f"{where} must not hold {_REFUSED_CLASS!r}: this publisher makes no people")
    return names


def _parse_camera(data: Any) -> FakeCamera:
    _check_fields(
        data,
        "camera",
        (
            "frame_id",
            "parent_frame",
            "position",
            "orientation_xyzw",
            "width",
            "height",
            "fx",
            "fy",
            "cx",
            "cy",
            "depth_range",
        ),
    )
    orientation = _vector(data["orientation_xyzw"], 4, "camera.orientation_xyzw")
    if abs(float(np.linalg.norm(orientation)) - 1.0) > 1e-6:
        raise ValueError(f"camera.orientation_xyzw must be a unit quaternion, not {orientation.tolist()}")
    depth_range = _vector(data["depth_range"], 2, "camera.depth_range")
    if not 0 < depth_range[0] < depth_range[1]:
        raise ValueError(f"camera.depth_range must be 0 < near < far, not {depth_range.tolist()}")
    return FakeCamera(
        frame_id=_name(data["frame_id"], "camera.frame_id"),
        parent_frame=_name(data["parent_frame"], "camera.parent_frame"),
        position=_vector(data["position"], 3, "camera.position"),
        orientation_xyzw=orientation,
        width=_integer(data["width"], "camera.width", 1),
        height=_integer(data["height"], "camera.height", 1),
        fx=_positive(data["fx"], "camera.fx"),
        fy=_positive(data["fy"], "camera.fy"),
        cx=_number(data["cx"], "camera.cx"),
        cy=_number(data["cy"], "camera.cy"),
        min_depth=float(depth_range[0]),
        max_depth=float(depth_range[1]),
    )


def _parse_box(data: Any, index: int, classes: Sequence[str]) -> FakeBox:
    where = f"boxes[{index}]"
    _check_fields(data, where, ("name", "label", "track_id", "score", "center", "size"), ("view_quality", "color"))
    label = _name(data["label"], f"{where}.label")
    if label not in classes:
        raise ValueError(f"{where}.label {label!r} is not in classes")
    color = data.get("color")
    if color is not None and color not in crop.COLOR_NAMES:
        raise ValueError(f"{where}.color must be one of {list(crop.COLOR_NAMES)}, not {color!r}")
    size = _vector(data["size"], 3, f"{where}.size")
    if np.any(size <= 0):
        raise ValueError(f"{where}.size must be above 0, not {size.tolist()}")
    return FakeBox(
        name=_name(data["name"], f"{where}.name"),
        label=label,
        track_id=_integer(data["track_id"], f"{where}.track_id", 0),
        score=_number(data["score"], f"{where}.score", 0.0, 1.0),
        view_quality=_number(data.get("view_quality", 0.0), f"{where}.view_quality", 0.0, 1.0),
        color=color,
        center=_vector(data["center"], 3, f"{where}.center"),
        size=size,
    )


def _parse_phase(data: Any, index: int, names: Sequence[str]) -> FakePhase:
    where = f"sequence[{index}]"
    _check_fields(data, where, ("frames", "visible"), ("move",))
    visible = data["visible"]
    if isinstance(visible, str) or not isinstance(visible, Sequence) or not all(isinstance(v, str) for v in visible):
        raise ValueError(f"{where}.visible must be a list of box names, not {visible!r}")
    if len(set(visible)) != len(visible) or not set(visible) <= set(names):
        raise ValueError(f"{where}.visible must name each known box at most once, not {visible!r}")
    move = data.get("move", {})
    if not isinstance(move, Mapping) or not set(move) <= set(names):
        raise ValueError(f"{where}.move must map known box names to centres, not {move!r}")
    return FakePhase(
        frames=_integer(data["frames"], f"{where}.frames", 1),
        visible=tuple(visible),
        move={name: _vector(center, 3, f"{where}.move.{name}") for name, center in move.items()},
    )


def parse_scene(data: Any) -> FakeScene:
    """Return the scene that a parsed JSON document describes. Raises ValueError for anything else."""
    _check_fields(data, "the scene", ("format", "camera", "classes", "boxes", "sequence"))
    if data["format"] != 1:
        raise ValueError(f"the scene's format must be 1, not {data['format']!r}")
    classes = _check_classes(data["classes"], "classes")
    if not isinstance(data["boxes"], Sequence) or not data["boxes"]:
        raise ValueError("boxes must be a list with at least one box")
    boxes = tuple(_parse_box(b, i, classes) for i, b in enumerate(data["boxes"]))
    names = [b.name for b in boxes]
    if len(set(names)) != len(names):
        raise ValueError(f"boxes has duplicate names: {names}")
    tracks = [b.track_id for b in boxes if b.track_id != 0]
    if len(set(tracks)) != len(tracks):
        raise ValueError(f"boxes has duplicate track ids: {tracks}")
    if not isinstance(data["sequence"], Sequence) or not data["sequence"]:
        raise ValueError("sequence must be a list with at least one phase")
    phases = tuple(_parse_phase(p, i, names) for i, p in enumerate(data["sequence"]))
    return FakeScene(camera=_parse_camera(data["camera"]), classes=classes, boxes=boxes, phases=phases)


def load_scene(path: str | Path) -> FakeScene:
    """Return the scene in a JSON file. Raises OSError or ValueError."""
    with open(path, encoding="utf-8") as f:
        return parse_scene(json.load(f))


@dataclass(frozen=True, eq=False)
class RenderedBox:
    """What one phase's camera sees of one box."""

    box: FakeBox
    bounds: np.ndarray
    """x1, y1, x2, y2 of the pixels the box won; x2 and y2 are one past the last pixel."""
    points: np.ndarray
    """N x 3 surface points in the optical frame, in row-major pixel order."""
    crop_jpeg: array.array
    color_shares: np.ndarray | None


def _camera_rays(camera: FakeCamera) -> np.ndarray:
    """Return one ray per pixel in the parent frame, H x W x 3. Each has optical z 1, so t is depth."""
    u, v = np.meshgrid(np.arange(camera.width, dtype=np.float64), np.arange(camera.height, dtype=np.float64))
    optical = np.stack([(u - camera.cx) / camera.fx, (v - camera.cy) / camera.fy, np.ones_like(u)], axis=2)
    rays = optical @ camera.rotation().T
    # A ray parallel to a face would divide by zero. This tiny slope gives the same hits.
    rays[np.abs(rays) < 1e-12] = 1e-12
    return rays


def _entry_depths(camera: FakeCamera, rays: np.ndarray, center: np.ndarray, size: np.ndarray) -> np.ndarray:
    """Return the depth where each ray enters the box (the slab method), and inf where it misses."""
    low = (center - size / 2 - camera.position) / rays
    high = (center + size / 2 - camera.position) / rays
    entry = np.minimum(low, high).max(axis=2)
    leave = np.maximum(low, high).min(axis=2)
    return np.where((entry <= leave) & (entry > 0), entry, np.inf)


def _shares(color: str | None, pixels: int) -> np.ndarray | None:
    if color is None or pixels < MIN_COLOR_PIXELS:
        return None
    return np.eye(len(crop.COLOR_NAMES))[crop.COLOR_NAMES.index(color)]


def _encode_crop(image: np.ndarray, bounds: np.ndarray) -> array.array:
    x1, y1, x2, y2 = pad_rect(bounds, CROP_PADDING, image.shape[1], image.shape[0])
    pixels, _, _ = crop.scale_down(image[y1:y2, x1:x2], CROP_MAX_SIDE)
    data = array.array("B")
    data.frombytes(crop.encode_jpeg(pixels, crop.JPEG_QUALITY))
    return data


def render(camera: FakeCamera, placed: Sequence[tuple[FakeBox, np.ndarray]]) -> list[RenderedBox]:
    """Return what the camera sees of boxes at the given centres, in their given order.

    A box with no winning pixel is left out. Raises ValueError when the
    camera is inside a box, where no surface is visible.
    """
    if not placed:
        return []
    for box, center in placed:
        if np.all(np.abs(camera.position - center) < box.size / 2):
            raise ValueError(f"the camera is inside box {box.name!r}")
    rays = _camera_rays(camera)
    depths = np.stack([_entry_depths(camera, rays, center, box.size) for box, center in placed])
    winner = depths.argmin(axis=0)
    nearest = depths.min(axis=0)
    image = np.full((camera.height, camera.width, 3), _BACKGROUND_BGR, dtype=np.uint8)
    masks = []
    for i, (box, _) in enumerate(placed):
        mask = (winner == i) & np.isfinite(nearest)
        masks.append(mask)
        if box.color is None:
            image[mask] = _NO_COLOR_BGR
        else:
            r, g, b = PALETTE_RGB[crop.COLOR_NAMES.index(box.color)]
            image[mask] = (b, g, r)
    out = []
    for (box, _), mask in zip(placed, masks, strict=True):
        rows, cols = np.nonzero(mask)
        if rows.size == 0:
            continue
        bounds = np.array([cols.min(), rows.min(), cols.max() + 1, rows.max() + 1], dtype=np.float64)
        z = nearest[rows, cols]
        keep = (z >= camera.min_depth) & (z <= camera.max_depth)
        rows, cols, z = rows[keep], cols[keep], z[keep]
        points = np.column_stack([(cols - camera.cx) * z / camera.fx, (rows - camera.cy) * z / camera.fy, z])
        out.append(
            RenderedBox(
                box=box,
                bounds=bounds,
                points=points,
                crop_jpeg=_encode_crop(image, bounds),
                color_shares=_shares(box.color, int(mask.sum())),
            )
        )
    return out


def _render_phases(scene: FakeScene) -> list[list[RenderedBox]]:
    centers = {b.name: b.center for b in scene.boxes}
    out = []
    for phase in scene.phases:
        centers.update(phase.move)
        visible = set(phase.visible)
        out.append(render(scene.camera, [(b, centers[b.name]) for b in scene.boxes if b.name in visible]))
    return out


class FakeStream:
    """The arrays of a scene. The same index, stamp and stream id give an equal message."""

    def __init__(
        self,
        scene: FakeScene,
        *,
        stream_id: str | None = None,
        classes: Sequence[str] | None = None,
        crops_enabled: bool = True,
        embedding_model: str = "",
        embedding_dim: int = 512,
    ) -> None:
        if embedding_model and not embedding_model.startswith(FAKE_MODEL_PREFIX):
            raise ValueError(f"embedding_model must start with {FAKE_MODEL_PREFIX!r}, not {embedding_model!r}")
        if embedding_dim < 1:
            raise ValueError(f"embedding_dim must be at least 1, not {embedding_dim}")
        self._camera = scene.camera
        self._classes = list(_check_classes(scene.classes if classes is None else classes, "classes"))
        self._crops_enabled = crops_enabled
        self._embedding_model = embedding_model
        self._embedding_dim = embedding_dim
        self._stream_id = convert.new_stream_id() if stream_id is None else stream_id
        self._rendered = _render_phases(scene)
        self._frames = [i for i, phase in enumerate(scene.phases) for _ in range(phase.frames)]
        # Every distinct frame follows the contract, or nothing starts.
        for first_frame in sorted({self._frames.index(i) for i in range(len(scene.phases))}):
            self.array(first_frame, Time())

    @property
    def stream_id(self) -> str:
        """The id on every array of this stream."""
        return self._stream_id

    @property
    def frame_count(self) -> int:
        """The number of frames before the sequence restarts."""
        return len(self._frames)

    def array(self, index: int, stamp: Time) -> ObjectDetectionArray:
        """Return frame index of the endless sequence, stamped."""
        header = Header(stamp=stamp, frame_id=self._camera.frame_id)
        rendered = self._rendered[self._frames[index % len(self._frames)]]
        return convert.to_object_detection_array(
            header,
            self._camera.camera_info(header),
            self._classes,
            [self._detection(r, header) for r in rendered if r.box.label in self._classes],
            stream_id=self._stream_id,
        )

    def _detection(self, rendered: RenderedBox, header: Header) -> ObjectDetection:
        box = rendered.box
        return convert.to_object_detection(
            Detection(box=rendered.bounds, label=box.label, score=box.score, track_id=box.track_id),
            embedding=hash_vector(box.name, self._embedding_dim) if self._embedding_model else None,
            embedding_model=self._embedding_model,
            points=convert.to_point_cloud(header, rendered.points),
            crop=(
                CompressedImage(header=header, format=convert.JPEG_FORMAT, data=rendered.crop_jpeg)
                if self._crops_enabled
                else None
            ),
            view_quality=box.view_quality,
            color_shares=rendered.color_shares,
        )


class FakePublisher(Node):
    """Publishes a FakeStream at a fixed rate on the node's clock."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(NODE_NAME, **kwargs)
        read_once = ParameterDescriptor(read_only=True)
        topic = self.declare_parameter("objects_topic", "/walkie_vision/objects", read_once).value
        self.declare_parameter("classes", Parameter.Type.STRING_ARRAY, read_once)
        rate_hz = self.declare_parameter("rate_hz", 5.0, read_once).value
        scene_file = self.declare_parameter("scene_file", str(DEFAULT_SCENE), read_once).value
        self._loop = self.declare_parameter("loop", True, read_once).value
        crops_enabled = self.declare_parameter("crops_enabled", True, read_once).value
        embedding_model = self.declare_parameter("embedding_model", "", read_once).value
        embedding_dim = self.declare_parameter("embedding_dim", 512, read_once).value
        publish_tf = self.declare_parameter("publish_tf", True, read_once).value
        if not (math.isfinite(rate_hz) and rate_hz > 0):
            raise ValueError(f"rate_hz must be finite and above 0, not {rate_hz}")
        try:
            # None means the scene's own list.
            classes = list(self.get_parameter("classes").value)
        except ParameterUninitializedException:
            classes = None
        scene = load_scene(scene_file)
        self._stream = FakeStream(
            scene,
            classes=classes,
            crops_enabled=crops_enabled,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
        )
        self.finished = False
        self._index = 0
        self._last_ns = 0
        self._publisher = self.create_publisher(ObjectDetectionArray, topic, OBSERVATION_QOS)
        self._tf_publisher = None
        if publish_tf:
            self._tf_publisher = self.create_publisher(TFMessage, "/tf_static", TF_STATIC_QOS)
            self._tf_publisher.publish(scene.camera.transform(self.get_clock().now().to_msg()))
        # Steady time, so that the timer still fires under use_sim_time before /clock starts.
        self._timer = self.create_timer(1.0 / rate_hz, self.publish_next, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.get_logger().info(
            f"publishing synthetic objects, not perception: topic {topic}, stream_id {self._stream.stream_id}, "
            f"{self._stream.frame_count} frames at {rate_hz} Hz, loop {self._loop}, "
            f"rmw {rclpy.get_rmw_implementation_identifier()}"
        )

    @property
    def stream_id(self) -> str:
        """The id on every array this process publishes."""
        return self._stream.stream_id

    def publish_next(self) -> ObjectDetectionArray | None:
        """Publish the next frame. Returns None when the clock has not moved since the last one."""
        if self.finished:
            return None
        now = self.get_clock().now()
        if now.nanoseconds == 0 or now.nanoseconds == self._last_ns:
            return None
        self._last_ns = now.nanoseconds
        msg = self._stream.array(self._index, now.to_msg())
        self._publisher.publish(msg)
        self._index += 1
        if not self._loop and self._index >= self._stream.frame_count:
            self.finished = True
            self._timer.cancel()
            self.get_logger().info("the sequence has ended")
        return msg


def main(args: list[str] | None = None) -> int:
    """Run until Ctrl-C or the end of the sequence (exit 0), or a startup failure (exit 2)."""
    logger = get_logger(NODE_NAME)
    node: FakePublisher | None = None
    code = 0
    try:
        try:
            rclpy.init(args=args)
            node = FakePublisher()
        except (KeyboardInterrupt, ExternalShutdownException):
            raise
        except Exception as e:
            logger.fatal(f"could not start: {type(e).__name__}: {e}")
            code = EXIT_STARTUP
        else:
            while rclpy.ok() and not node.finished:
                try:
                    rclpy.spin_once(node, timeout_sec=0.1)
                except Exception:
                    # SIGTERM shuts the context down from another thread, so a
                    # timer the wait already picked can publish on a dead context.
                    if rclpy.ok():
                        raise
    except (KeyboardInterrupt, ExternalShutdownException):
        code = 0
    finally:
        # A second SIGINT must not cut the cleanup short. pixi run forwards
        # Ctrl-C to its child, and ros2 launch sends one of its own when its
        # stdin is not a terminal, so a node often gets two.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
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
