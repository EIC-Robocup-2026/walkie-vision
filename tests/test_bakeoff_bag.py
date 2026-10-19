"""Frame extraction from rosbag2 recordings (tools/bakeoff/bag.py)."""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_TOOL = str(Path(__file__).resolve().parents[1] / "tools" / "bakeoff")
if _TOOL not in sys.path:
    sys.path.insert(0, _TOOL)

from bag import (  # noqa: E402
    BagError,
    extract,
    image_to_bgr,
    make_compressed_msg,
    make_image_msg,
    write_bag,
)

RAW = "/camera/image"
COMPRESSED = "/camera/image/compressed"


@pytest.fixture(scope="module")
def frame():
    # Odd sizes and random pixels catch row-padding and channel-order mistakes.
    return np.random.default_rng(7).integers(0, 256, (37, 53, 3), dtype=np.uint8)


@pytest.mark.parametrize("encoding", ["bgra8", "rgba8", "bgr8", "rgb8"])
@pytest.mark.parametrize("pad", [0, 5])
def test_image_to_bgr_round_trip(frame, encoding, pad):
    msg = make_image_msg(frame, encoding, pad=pad)
    assert msg.step == frame.shape[1] * len(encoding.rstrip("8")) + pad
    np.testing.assert_array_equal(image_to_bgr(msg), frame)


def test_image_to_bgr_mono(frame):
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out = image_to_bgr(make_image_msg(frame, "mono8", pad=3))
    for c in range(3):
        np.testing.assert_array_equal(out[:, :, c], grey)


def test_image_to_bgr_rejects_other_encodings(frame):
    msg = make_image_msg(frame, "bgr8")
    msg.encoding = "16UC1"
    with pytest.raises(BagError, match="unsupported image encoding"):
        image_to_bgr(msg)
    msg = make_image_msg(frame, "bgr8")
    msg.step = 10
    with pytest.raises(BagError, match="shorter than"):
        image_to_bgr(msg)


def _recording(path: Path, storage: str, frame: np.ndarray) -> Path:
    from std_msgs.msg import String

    messages = []
    for i, encoding in enumerate(["bgra8", "bgr8", "rgb8", "rgba8", "bgra8"]):
        messages.append((RAW, make_image_msg(frame, encoding, stamp_ns=1_000 + i, pad=2 * i), 10 + i))
    messages.append((COMPRESSED, make_compressed_msg(frame, "png", stamp_ns=2_000), 20))
    messages.append((COMPRESSED, make_compressed_msg(frame, "jpeg", stamp_ns=2_001), 21))
    note = String()
    note.data = "not an image"
    messages.append(("/note", note, 30))
    write_bag(path, storage, messages)
    return path


@pytest.mark.parametrize("storage", ["mcap", "sqlite3"])
def test_extract_raw_frames(tmp_path, frame, storage):
    bag = _recording(tmp_path / "rec", storage, frame)
    images, frames = tmp_path / "images", tmp_path / "frames.csv"
    result = extract(bag, RAW, 2, images, frames)
    assert result.messages == 5
    assert result.written == ["rec_000000.png", "rec_000002.png", "rec_000004.png"]
    for name in result.written:
        np.testing.assert_array_equal(cv2.imread(str(images / name)), frame)
    with open(frames, newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(r["image"], r["index"], r["stamp_ns"], r["bag_time_ns"]) for r in rows] == [
        ("rec_000000.png", "0", "1000", "10"),
        ("rec_000002.png", "2", "1002", "12"),
        ("rec_000004.png", "4", "1004", "14"),
    ]
    assert {r["topic"] for r in rows} == {RAW}


@pytest.mark.parametrize("storage", ["mcap", "sqlite3"])
def test_extract_compressed_frames_byte_for_byte(tmp_path, frame, storage):
    bag = _recording(tmp_path / "rec", storage, frame)
    result = extract(bag, COMPRESSED, 1, tmp_path / "images", tmp_path / "frames.csv", prefix="c")
    assert result.written == ["c_000000.png", "c_000001.jpg"]
    png = (tmp_path / "images" / "c_000000.png").read_bytes()
    assert png == bytes(make_compressed_msg(frame, "png").data)
    np.testing.assert_array_equal(cv2.imread(str(tmp_path / "images" / "c_000000.png")), frame)
    assert (tmp_path / "images" / "c_000001.jpg").read_bytes()[:3] == b"\xff\xd8\xff"


def test_extract_one_mcap_file(tmp_path, frame):
    bag = _recording(tmp_path / "rec", "mcap", frame)
    (mcap,) = bag.glob("*.mcap")
    result = extract(mcap, RAW, 5, tmp_path / "images", tmp_path / "frames.csv")
    assert result.written == [f"{mcap.stem}_000000.png"]


def test_extract_errors(tmp_path, frame):
    bag = _recording(tmp_path / "rec", "mcap", frame)
    images, frames = tmp_path / "images", tmp_path / "frames.csv"
    with pytest.raises(BagError, match="no topic /missing") as e:
        extract(bag, "/missing", 1, images, frames)
    assert f"{RAW} (sensor_msgs/msg/Image), {COMPRESSED} (sensor_msgs/msg/CompressedImage)" in str(e.value)
    assert "/note" not in str(e.value)
    with pytest.raises(BagError, match="std_msgs/msg/String"):
        extract(bag, "/note", 1, images, frames)
    with pytest.raises(BagError, match="at least 1"):
        extract(bag, RAW, 0, images, frames)
    with pytest.raises(BagError, match="not found"):
        extract(tmp_path / "nothing", RAW, 1, images, frames)
    extract(bag, RAW, 1, images, frames)
    with pytest.raises(BagError, match="already has 5 file"):
        extract(bag, RAW, 1, images, frames)
    # Another prefix can share the directory.
    assert len(extract(bag, RAW, 1, images, frames, prefix="again").written) == 5


def test_extract_prefix_clash_needs_the_same_frame_names(tmp_path, frame):
    bag = _recording(tmp_path / "rec", "mcap", frame)
    images, frames = tmp_path / "images", tmp_path / "frames.csv"
    extract(bag, RAW, 1, images, frames, prefix="rec_2")
    # rec_2_000000.png cannot be a frame of the prefix rec.
    assert len(extract(bag, RAW, 1, images, frames, prefix="rec").written) == 5
    # Frames whose files were deleted are still in frames.csv.
    for path in images.glob("rec_0*.png"):
        path.unlink()
    with pytest.raises(BagError, match="frames.csv already lists 5 frame"):
        extract(bag, RAW, 1, images, frames, prefix="rec")


def test_failed_extract_leaves_nothing_behind(tmp_path, frame):
    depth = make_image_msg(frame, "bgr8")
    depth.encoding = "16UC1"
    bag = tmp_path / "mixed"
    write_bag(bag, "mcap", [(RAW, make_image_msg(frame, "bgra8"), 1), (RAW, depth, 2)])
    images, frames = tmp_path / "images", tmp_path / "frames.csv"
    with pytest.raises(BagError, match="unsupported image encoding"):
        extract(bag, RAW, 1, images, frames)
    assert list(images.iterdir()) == []
    assert not frames.exists()
