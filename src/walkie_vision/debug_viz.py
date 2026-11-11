"""Draw the object stream for a person to look at: an annotated image and 3D markers.

This node is a debugging aid, not part of the observation contract (ADR-0013).
It reads ObjectDetectionArray and the colour images the arrays came from, and
publishes two things that Foxglove and RViz show without walkie_vision_msgs:

- image_out_topic: sensor_msgs/CompressedImage, the colour image with each
  detection's box and label drawn on it, scaled so its longest side is at
  most max_side, as a JPEG. Its header is the array's header.
- markers_topic: visualization_msgs/MarkerArray in the array's optical frame
  and at its stamp. Per detection with points: a POINTS marker of its cloud
  and a TEXT_VIEW_FACING label 5 cm above the cloud, over its centroid.
  Detections without points get no marker. Each array first deletes the
  previous array's markers.

An array is drawn on the image with exactly its header stamp and frame, which
the node keeps in a short ring of recent images. When that image is gone, or
never came, the node still publishes the markers and counts the miss. Once an
array is handled, the ring drops every image up to its stamp: arrays arrive
in stamp order, so no later array asks for them.

The node loads no model and needs no GPU. It costs a second subscription to
the raw colour image and up to ring_size raw images of memory while the
vision node falls behind: about 110 MB at the default 10 with the robot's
2208 x 1242 bgra8 images. Leave it off on the robot unless someone is watching.
"""

from __future__ import annotations

import collections
import math
import signal
import sys
import zlib
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import ExternalShutdownException
from rclpy.logging import get_logger
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image, PointCloud2
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray
from walkie_vision_msgs.msg import ObjectDetection, ObjectDetectionArray

from .crop import encode_jpeg
from .image import image_to_bgr

NODE_NAME = "walkie_vision_debug_viz"
EXIT_STARTUP = 2
NAMESPACE = "detections"

# Eight colours that stay apart for most colour-blind viewers (Okabe-Ito,
# without black), as 0-255 RGB. A class takes the colour of its index in the
# array's classes, so a class keeps its colour while the class list holds.
PALETTE_RGB = (
    (230, 159, 0),
    (86, 180, 233),
    (0, 158, 115),
    (240, 228, 66),
    (0, 114, 178),
    (213, 94, 0),
    (204, 121, 167),
    (153, 153, 153),
)

# What the node subscribes to: the newest images only, and every array.
IMAGE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)
ARRAY_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)
# A late viewer gets nothing old: these are live views.
OUTPUT_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=2,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)


def class_color(label: str, classes: Sequence[str]) -> tuple[int, int, int]:
    """Return the 0-255 RGB colour of a label: by its index in classes, else by a hash of the label."""
    try:
        index = list(classes).index(label)
    except ValueError:
        index = zlib.crc32(label.encode("utf-8"))
    return PALETTE_RGB[index % len(PALETTE_RGB)]


def short_track(track_id: int) -> str:
    """Return a short form of a track id for a label: '#' and its last four digits, or '' when untracked."""
    return f"#{track_id % 10000:04d}" if track_id else ""


def label_text(det: ObjectDetection) -> str:
    """Return the text drawn next to a detection: label, score and short track id."""
    track = short_track(int(det.track_id))
    return f"{det.label} {det.score:.2f}" + (f" {track}" if track else "")


def cloud_xyz(cloud: PointCloud2) -> np.ndarray:
    """Return the x, y, z float32 fields of a cloud as an N x 3 float64 array.

    Reads the field offsets from the cloud, so any point_step works. Returns an
    empty (0, 3) array for an empty cloud. Raises ValueError for a cloud
    without float32 x, y and z, a big-endian cloud, or data shorter than its size.
    """
    count = int(cloud.width) * int(cloud.height)
    if count == 0:
        return np.zeros((0, 3))
    if cloud.is_bigendian:
        raise ValueError("a big-endian cloud is not supported")
    offsets = {}
    for field in cloud.fields:
        if field.name in ("x", "y", "z"):
            if field.datatype != field.FLOAT32 or field.count != 1:
                raise ValueError(f"field {field.name} is not one float32")
            offsets[field.name] = int(field.offset)
    if set(offsets) != {"x", "y", "z"}:
        raise ValueError(f"the cloud lacks x, y or z: it has {[f.name for f in cloud.fields]}")
    step = int(cloud.point_step)
    data = np.frombuffer(cloud.data, dtype=np.uint8)
    if step < 12 or len(data) < count * step:
        raise ValueError(f"the cloud has {len(data)} bytes, fewer than {count} points of {step} bytes")
    rows = data[: count * step].reshape(count, step)
    xyz = np.empty((count, 3))
    for column, name in enumerate(("x", "y", "z")):
        start = offsets[name]
        xyz[:, column] = rows[:, start : start + 4].copy().view("<f4").ravel()
    return xyz


def fit_scale(width: int, height: int, max_side: int) -> float:
    """Return the factor that shrinks an image so its longest side is at most max_side; never above 1."""
    longest = max(width, height)
    return min(1.0, max_side / longest) if longest > 0 else 1.0


def draw_detections(
    bgr: np.ndarray, detections: Sequence[ObjectDetection], classes: Sequence[str], scale: float = 1.0
) -> np.ndarray:
    """Return a copy of a BGR image, scaled by scale, with each detection's box and label drawn on it."""
    out = bgr if scale == 1.0 else cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    out = np.ascontiguousarray(out).copy()
    height, width = out.shape[:2]
    thickness = max(1, round(min(width, height) / 300))
    font_scale = max(0.4, min(width, height) / 900)
    for det in detections:
        r, g, b = class_color(det.label, classes)
        color = (b, g, r)
        cx, cy = det.bbox.center.position.x * scale, det.bbox.center.position.y * scale
        half_w, half_h = det.bbox.size_x * scale / 2, det.bbox.size_y * scale / 2
        x0, y0 = int(round(cx - half_w)), int(round(cy - half_h))
        x1, y1 = int(round(cx + half_w)), int(round(cy + half_h))
        cv2.rectangle(out, (x0, y0), (x1, y1), color, thickness)
        text = label_text(det)
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        # Above the box when there is room, else just inside its top edge.
        ty = y0 - 4 if y0 - th - base - 4 >= 0 else y0 + th + 4
        tx = min(max(0, x0), max(0, width - tw))
        cv2.rectangle(out, (tx, ty - th - base // 2), (tx + tw, ty + base), color, cv2.FILLED)
        text_color = (0, 0, 0) if (0.299 * r + 0.587 * g + 0.114 * b) > 140 else (255, 255, 255)
        cv2.putText(out, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)
    return out


def _rgba(rgb: tuple[int, int, int], alpha: float = 1.0) -> ColorRGBA:
    return ColorRGBA(r=rgb[0] / 255.0, g=rgb[1] / 255.0, b=rgb[2] / 255.0, a=alpha)


def build_markers(msg: ObjectDetectionArray, lifetime_s: float = 1.0, max_points: int = 500) -> MarkerArray:
    """Return the markers of one array: a delete-all, then points and a label per detection with points.

    Detections without points get no marker: their place is unknown. Each
    cloud is thinned to at most max_points evenly spaced points.
    """
    out = MarkerArray()
    clear = Marker()
    clear.header = msg.header
    clear.ns = NAMESPACE
    clear.action = Marker.DELETEALL
    out.markers.append(clear)
    secs = int(lifetime_s)
    lifetime = DurationMsg(sec=secs, nanosec=int(round((lifetime_s - secs) * 1e9)))
    for index, det in enumerate(msg.detections):
        try:
            xyz = cloud_xyz(det.points)
        except ValueError:
            continue
        if len(xyz) == 0:
            continue
        if len(xyz) > max_points:
            xyz = xyz[np.arange(max_points) * len(xyz) // max_points]
        rgb = class_color(det.label, msg.classes)
        points = Marker()
        points.header = msg.header
        points.ns = NAMESPACE
        points.id = 2 * index
        points.type = Marker.POINTS
        points.action = Marker.ADD
        points.pose.orientation.w = 1.0
        points.scale.x = points.scale.y = 0.01
        points.color = _rgba(rgb)
        points.lifetime = lifetime
        points.points = [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in xyz]
        out.markers.append(points)
        centroid = xyz.mean(axis=0)
        text = Marker()
        text.header = msg.header
        text.ns = NAMESPACE
        text.id = 2 * index + 1
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        # The optical frame's y points down: lift the label above the cloud.
        top = float(xyz[:, 1].min())
        text.pose.position = Point(x=float(centroid[0]), y=top - 0.05, z=float(centroid[2]))
        text.pose.orientation.w = 1.0
        text.scale.z = 0.08
        text.color = _rgba(rgb)
        text.lifetime = lifetime
        text.text = label_text(det)
        out.markers.append(text)
    return out


def _stamp_ns(stamp: Time) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class ImageRing:
    """The newest colour images, found again by exact stamp and frame.

    Keeps at most capacity images, and drops any image more than max_age_s
    older than the newest. An image with a stamp before the newest one's
    (the clock jumped back) empties the ring first.
    """

    def __init__(self, capacity: int, max_age_s: float) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be at least 1, not {capacity}")
        if not (max_age_s > 0 and math.isfinite(max_age_s)):
            raise ValueError(f"max_age_s must be positive and finite, not {max_age_s}")
        self._images: collections.OrderedDict[tuple[int, str], Image] = collections.OrderedDict()
        self._capacity = capacity
        self._max_age_ns = int(max_age_s * 1e9)
        self._newest_ns: int | None = None

    def __len__(self) -> int:
        return len(self._images)

    def add(self, msg: Image) -> None:
        stamp = _stamp_ns(msg.header.stamp)
        if self._newest_ns is not None and stamp < self._newest_ns:
            self._images.clear()
            self._newest_ns = None
        self._newest_ns = stamp if self._newest_ns is None else max(self._newest_ns, stamp)
        self._images[(stamp, msg.header.frame_id)] = msg
        self._images.move_to_end((stamp, msg.header.frame_id))
        while len(self._images) > self._capacity:
            self._images.popitem(last=False)
        oldest_kept = self._newest_ns - self._max_age_ns
        for key in [k for k in self._images if k[0] < oldest_kept]:
            del self._images[key]

    def find(self, header: Header) -> Image | None:
        """Return the image with exactly this stamp and frame, or None."""
        return self._images.get((_stamp_ns(header.stamp), header.frame_id))

    def drop_through(self, stamp: Time) -> None:
        """Drop every image stamped at or before stamp, in any frame."""
        last = _stamp_ns(stamp)
        for key in [k for k in self._images if k[0] <= last]:
            del self._images[key]


@dataclass(frozen=True)
class VizConfig:
    objects_topic: str = "/walkie_vision/objects"
    image_topic: str = "/zed_head/zed_node/rgb/color/rect/image"
    image_out_topic: str = "/walkie_vision/debug/objects_image/compressed"
    markers_topic: str = "/walkie_vision/debug/objects_markers"
    max_side: int = 960
    jpeg_quality: int = 80
    ring_size: int = 10
    ring_age_s: float = 3.0
    marker_lifetime_s: float = 1.0


class DebugViz(Node):
    def __init__(self, **node_kwargs) -> None:
        super().__init__(NODE_NAME, **node_kwargs)
        defaults = VizConfig()
        # One descriptor per parameter: rclpy writes the name and type into it.
        values = {
            name: self.declare_parameter(name, getattr(defaults, name), ParameterDescriptor(read_only=True)).value
            for name in VizConfig.__dataclass_fields__
        }
        cfg = VizConfig(**values)
        if cfg.max_side < 32:
            raise ValueError(f"max_side must be at least 32, not {cfg.max_side}")
        if not 1 <= cfg.jpeg_quality <= 100:
            raise ValueError(f"jpeg_quality must be in 1..100, not {cfg.jpeg_quality}")
        if not (math.isfinite(cfg.marker_lifetime_s) and 0 <= cfg.marker_lifetime_s < 2**31):
            raise ValueError(f"marker_lifetime_s must be finite and not negative, not {cfg.marker_lifetime_s}")
        self._cfg = cfg
        self._ring = ImageRing(cfg.ring_size, cfg.ring_age_s)
        self._drawn = 0
        self._missed = 0
        self._image_pub = self.create_publisher(CompressedImage, cfg.image_out_topic, OUTPUT_QOS)
        self._marker_pub = self.create_publisher(MarkerArray, cfg.markers_topic, OUTPUT_QOS)
        self.create_subscription(Image, cfg.image_topic, self._ring.add, IMAGE_QOS)
        self.create_subscription(ObjectDetectionArray, cfg.objects_topic, self._on_array, ARRAY_QOS)
        self.create_timer(30.0, self._report)
        self.get_logger().info(
            f"drawing {cfg.objects_topic} on {cfg.image_topic}; publishing {cfg.image_out_topic} "
            f"and {cfg.markers_topic}"
        )

    def _on_array(self, msg: ObjectDetectionArray) -> None:
        self._marker_pub.publish(build_markers(msg, self._cfg.marker_lifetime_s))
        image = self._ring.find(msg.header)
        self._ring.drop_through(msg.header.stamp)
        if image is None:
            self._missed += 1
            return
        try:
            bgr = image_to_bgr(image)
        except ValueError as e:
            self._missed += 1
            self.get_logger().warning(f"cannot draw on the image: {e}", throttle_duration_sec=10.0)
            return
        scale = fit_scale(bgr.shape[1], bgr.shape[0], self._cfg.max_side)
        drawn = draw_detections(bgr, msg.detections, msg.classes, scale)
        out = CompressedImage(header=msg.header, format="jpeg", data=encode_jpeg(drawn, self._cfg.jpeg_quality))
        self._image_pub.publish(out)
        self._drawn += 1

    def _report(self) -> None:
        self.get_logger().info(f"drawn {self._drawn} arrays; {self._missed} without their image")


def main(args: list[str] | None = None) -> int:
    """Run until Ctrl-C (exit 0) or a startup failure (exit 2)."""
    logger = get_logger(NODE_NAME)
    node: DebugViz | None = None
    code = 0
    try:
        try:
            rclpy.init(args=args)
            node = DebugViz()
        except (KeyboardInterrupt, ExternalShutdownException):
            raise
        except Exception as e:
            logger.fatal(f"could not start: {type(e).__name__}: {e}")
            code = EXIT_STARTUP
        else:
            rclpy.spin(node)
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
