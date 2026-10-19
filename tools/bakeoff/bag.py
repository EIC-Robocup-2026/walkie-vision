"""Read camera frames from a rosbag2 recording, and write small test recordings.

rosbag2_py and the message packages come from the pixi environment. They
are imported inside the functions, so the rest of the tool works without
them.
"""

from __future__ import annotations

import array
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

RAW_TYPE = "sensor_msgs/msg/Image"
COMPRESSED_TYPE = "sensor_msgs/msg/CompressedImage"
IMAGE_TYPES = (RAW_TYPE, COMPRESSED_TYPE)
DEFAULT_TOPIC = "/zed_head/zed_node/rgb/color/rect/image"
"""The ZED head camera's rectified colour image, as walkie-ros's yolo_node reads it."""
FRAME_COLUMNS = ("image", "bag", "topic", "index", "stamp_ns", "bag_time_ns")

_CHANNELS = {"bgra8": 4, "rgba8": 4, "bgr8": 3, "rgb8": 3, "mono8": 1}
_TO_BGR = {
    "bgra8": cv2.COLOR_BGRA2BGR,
    "rgba8": cv2.COLOR_RGBA2BGR,
    "rgb8": cv2.COLOR_RGB2BGR,
    "mono8": cv2.COLOR_GRAY2BGR,
}


class BagError(RuntimeError):
    """The recording cannot be read as asked."""


def image_to_bgr(msg) -> np.ndarray:
    """Convert a sensor_msgs/Image to an H x W x 3 BGR array.

    Rows may be padded: msg.step is the length of one row in bytes, and can
    be more than width x channels.
    """
    encoding = msg.encoding.lower()
    if encoding not in _CHANNELS:
        raise BagError(f"unsupported image encoding {msg.encoding!r}; supported: {', '.join(_CHANNELS)}")
    channels = _CHANNELS[encoding]
    row = msg.width * channels
    if msg.step < row:
        raise BagError(f"image step {msg.step} is shorter than width x channels ({row})")
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if data.size < msg.height * msg.step:
        raise BagError(f"image data has {data.size} bytes, expected {msg.height * msg.step}")
    pixels = data[: msg.height * msg.step].reshape(msg.height, msg.step)[:, :row]
    pixels = pixels.reshape(msg.height, msg.width, channels)
    if encoding == "bgr8":
        return pixels.copy()
    if encoding == "mono8":
        pixels = pixels[:, :, 0]
    return cv2.cvtColor(pixels, _TO_BGR[encoding])


def compressed_suffix(data: bytes) -> str:
    """Return the file suffix for compressed image bytes, from their first bytes."""
    head = bytes(data[:8])
    if head[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if head == b"\x89PNG\r\n\x1a\n":
        return ".png"
    raise BagError("compressed image is neither JPEG nor PNG (compressedDepth is not supported)")


def _open_reader(bag: Path):
    import rosbag2_py

    if not bag.exists():
        raise BagError(f"{bag}: not found")
    reader = rosbag2_py.SequentialReader()
    try:
        # An empty storage_id lets rosbag2 detect mcap or sqlite3 itself.
        reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id=""), rosbag2_py.ConverterOptions("", ""))
    except RuntimeError as e:
        raise BagError(f"{bag}: cannot open as a rosbag2 recording: {e}") from e
    return reader


@dataclass(frozen=True)
class ExtractResult:
    messages: int
    """Messages on the topic."""
    written: list[str]
    """File names written under the images directory."""


def extract(bag: Path, topic: str, every: int, images_dir: Path, frames_csv: Path, prefix: str | None = None) -> ExtractResult:
    """Save every `every`-th frame of `topic` into images_dir.

    Raw images are saved as PNG, so no detail is lost. Compressed images are
    saved byte for byte, as .jpg or .png. File names are
    <prefix>_<index>.<ext>, where index counts the messages on the topic and
    prefix defaults to the recording's name, so frames from several
    recordings can share one directory. frames.csv gets one row per file.
    """
    if every < 1:
        raise BagError("--every must be at least 1")
    bag = Path(bag)
    prefix = prefix or (bag.stem if bag.is_file() else bag.name)
    images_dir.mkdir(parents=True, exist_ok=True)
    # Only names this function writes can clash. A plain glob for prefix_*
    # would also match another recording whose name starts with prefix_.
    own = re.compile(re.escape(prefix) + r"_\d{6,}\.(png|jpg)")
    clash = sorted(p.name for p in images_dir.iterdir() if own.fullmatch(p.name))
    if clash:
        raise BagError(
            f"{images_dir} already has {len(clash)} file(s) named {prefix}_<index> (for example {clash[0]}); "
            "pass another --prefix, or delete them and their rows in frames.csv first"
        )
    listed = sorted(name for name in _listed_frames(frames_csv) if own.fullmatch(name))
    if listed:
        raise BagError(
            f"{frames_csv} already lists {len(listed)} frame(s) named {prefix}_<index> (for example {listed[0]}); "
            "pass another --prefix, or delete those rows first"
        )

    reader = _open_reader(bag)
    try:
        return _extract(reader, bag, topic, every, images_dir, frames_csv, prefix)
    finally:
        reader.close()


def _extract(reader, bag: Path, topic: str, every: int, images_dir: Path, frames_csv: Path, prefix: str) -> ExtractResult:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic not in types:
        cams = sorted(n for n, t in types.items() if t in IMAGE_TYPES)
        listed = ", ".join(f"{n} ({types[n]})" for n in cams) or "none"
        raise BagError(f"{bag}: no topic {topic}; image topics in this recording: {listed}")
    if types[topic] not in IMAGE_TYPES:
        raise BagError(f"{bag}: {topic} is a {types[topic]}, not one of {', '.join(IMAGE_TYPES)}")
    msg_type = get_message(types[topic])
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    written, rows, index = [], [], 0
    try:
        while reader.has_next():
            _, data, t_ns = reader.read_next()
            i, index = index, index + 1
            if i % every:
                continue
            msg = deserialize_message(data, msg_type)
            if types[topic] == RAW_TYPE:
                bgr = image_to_bgr(msg)
                name = f"{prefix}_{i:06d}.png"
                written.append(name)
                if not cv2.imwrite(str(images_dir / name), bgr):
                    raise BagError(f"cannot write {images_dir / name}")
            else:
                name = f"{prefix}_{i:06d}{compressed_suffix(msg.data)}"
                written.append(name)
                (images_dir / name).write_bytes(bytes(msg.data))
            stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            rows.append([name, bag.name, topic, i, stamp, t_ns])
    except BaseException:
        # Leave nothing behind, so that the same command can run again once
        # the problem is fixed.
        for name in written:
            (images_dir / name).unlink(missing_ok=True)
        raise
    new_file = not frames_csv.exists()
    with open(frames_csv, "a", newline="", encoding="utf-8") as f:
        out = csv.writer(f)
        if new_file:
            out.writerow(FRAME_COLUMNS)
        out.writerows(rows)
    return ExtractResult(messages=index, written=written)


def _listed_frames(frames_csv: Path) -> list[str]:
    """The image names in frames.csv, or none if it does not exist yet."""
    if not frames_csv.is_file():
        return []
    with open(frames_csv, newline="", encoding="utf-8") as f:
        return [row.get("image") or "" for row in csv.DictReader(f)]


def write_bag(path: Path, storage_id: str, messages: list[tuple[str, object, int]]) -> None:
    """Write (topic, message, time in ns) tuples to a new recording at path.

    Used by the smoke run and the tests. The message type of each topic is
    taken from its first message.
    """
    import rosbag2_py
    from rclpy.serialization import serialize_message

    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(path), storage_id=storage_id), rosbag2_py.ConverterOptions("", ""))
    try:
        created: set[str] = set()
        for topic, msg, t_ns in messages:
            if topic not in created:
                ros_type = f"{type(msg).__module__.split('.')[0]}/msg/{type(msg).__name__}"
                writer.create_topic(rosbag2_py.TopicMetadata(len(created), topic, ros_type, "cdr"))
                created.add(topic)
            writer.write(topic, serialize_message(msg), t_ns)
    finally:
        writer.close()  # writes metadata.yaml


def make_image_msg(bgr: np.ndarray, encoding: str, stamp_ns: int = 0, frame_id: str = "camera", pad: int = 0):
    """Build a sensor_msgs/Image from a BGR array, with `pad` extra bytes per row."""
    from sensor_msgs.msg import Image

    encoding = encoding.lower()
    if encoding == "bgr8":
        pixels = bgr
    elif encoding == "mono8":
        pixels = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)[:, :, None]
    else:
        code = {"bgra8": cv2.COLOR_BGR2BGRA, "rgba8": cv2.COLOR_BGR2RGBA, "rgb8": cv2.COLOR_BGR2RGB}[encoding]
        pixels = cv2.cvtColor(bgr, code)
    h, w = pixels.shape[:2]
    rows = pixels.reshape(h, -1)
    if pad:
        rows = np.concatenate([rows, np.zeros((h, pad), np.uint8)], axis=1)
    msg = Image()
    msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp_ns, 1_000_000_000)
    msg.header.frame_id = frame_id
    msg.height, msg.width, msg.encoding = h, w, encoding
    msg.step = rows.shape[1]
    # An array.array skips the per-byte type check that a bytes value gets.
    msg.data = array.array("B", np.ascontiguousarray(rows).tobytes())
    return msg


def make_compressed_msg(bgr: np.ndarray, fmt: str = "jpeg", stamp_ns: int = 0, frame_id: str = "camera"):
    """Build a sensor_msgs/CompressedImage from a BGR array, as JPEG or PNG."""
    from sensor_msgs.msg import CompressedImage

    ok, buf = cv2.imencode(".jpg" if fmt == "jpeg" else ".png", bgr)
    if not ok:
        raise BagError(f"cannot encode the image as {fmt}")
    msg = CompressedImage()
    msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp_ns, 1_000_000_000)
    msg.header.frame_id = frame_id
    msg.format = fmt
    msg.data = array.array("B", buf.tobytes())
    return msg
