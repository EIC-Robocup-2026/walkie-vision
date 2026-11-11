"""The fake object publisher: its geometry, its sequence and the wire.

The wire tests stream until they have what they need, instead of expecting a
given frame. The topic is RELIABLE but VOLATILE, so a frame published before
discovery is lost, and a BEST_EFFORT or slow subscriber can still miss later ones.
"""

import copy
import os
import subprocess
import sys
import time
import tomllib
import uuid
from pathlib import Path

import cv2
import numpy as np
import pytest
import rclpy
from builtin_interfaces.msg import Time
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSCompatibility,
    QoSProfile,
    ReliabilityPolicy,
    qos_check_compatible,
)
from rclpy.serialization import deserialize_message, serialize_message
from rclpy.time import Time as RosTime
from tf2_msgs.msg import TFMessage
from walkie_vision_msgs.msg import ObjectDetectionArray

from walkie_vision.convert import MAX_CLOUD_POINTS, bbox_to_box, check_object_detection_array
from walkie_vision.crop import COLOR_NAMES
from walkie_vision.fake_publisher import (
    DEFAULT_SCENE,
    OBSERVATION_QOS,
    PALETTE_RGB,
    FakePublisher,
    FakeStream,
    load_scene,
    parse_scene,
)
from walkie_vision.node import OUTPUT_QOS

REPO = Path(__file__).resolve().parents[1]
STREAM_ID = "6f1c1b52-8c1e-4a53-9f0e-2d3b7a6c9e10"
STAMP = Time(sec=1_792_400_000, nanosec=250_000_000)
TIMEOUT_S = 30.0

# walkie-database's subscription to the object arrays (_OBSERVATION_QOS in its node.py).
DATABASE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)
# A subscriber that asks for every array, such as a bag recorder.
RELIABLE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)

# Identity orientation: map axes are the optical axes, so depths and pixels can be worked out by hand.
CAMERA = {
    "frame_id": "test_optical_frame",
    "parent_frame": "map",
    "position": [0.0, 0.0, 0.0],
    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    "width": 100,
    "height": 100,
    "fx": 100.0,
    "fy": 100.0,
    "cx": 49.5,
    "cy": 49.5,
    "depth_range": [0.2, 3.0],
}
NEAR_BOX = {
    "name": "cup_1",
    "label": "cup",
    "track_id": 1,
    "score": 0.9,
    "view_quality": 0.75,
    "color": "red",
    "center": [0.0, 0.0, 2.0],
    "size": [0.4, 0.4, 0.2],
}


def _scene_data(*boxes, sequence=None, camera=None, classes=("cup", "bottle")) -> dict:
    return {
        "format": 1,
        "camera": camera or CAMERA,
        "classes": list(classes),
        "boxes": [copy.deepcopy(b) for b in boxes] or [copy.deepcopy(NEAR_BOX)],
        "sequence": sequence or [{"frames": 1, "visible": [b["name"] for b in boxes] or ["cup_1"]}],
    }


def _stream(*boxes, sequence=None, camera=None, **kwargs) -> FakeStream:
    kwargs.setdefault("stream_id", STREAM_ID)
    return FakeStream(parse_scene(_scene_data(*boxes, sequence=sequence, camera=camera)), **kwargs)


def _box(**changes) -> dict:
    return {**NEAR_BOX, **changes}


def _bottle(**changes) -> dict:
    return {**NEAR_BOX, "name": "bottle_1", "label": "bottle", "track_id": 2, "color": "blue", **changes}


def _xyz(detection) -> np.ndarray:
    return np.frombuffer(bytes(detection.points.data), dtype="<f4").reshape(-1, 3).astype(np.float64)


def _to_map(scene, xyz: np.ndarray) -> np.ndarray:
    return xyz @ scene.camera.rotation().T + scene.camera.position


def _assert_on_surface(points: np.ndarray, center, size) -> None:
    """Every point is on one of the box's faces, within float32 rounding."""
    offset = np.abs(points - np.asarray(center)) - np.asarray(size) / 2
    assert np.all(offset <= 1e-5)
    assert np.all(np.abs(offset).min(axis=1) <= 1e-5)


def test_a_known_box_has_the_known_box_points_and_pixels():
    # The front face is at z 1.9 and 0.2 m to each side, so it is
    # 0.2 * 100 / 1.9 = 10.53 px to each side of 49.5: pixels 39 to 60.
    msg = _stream().array(0, STAMP)
    (det,) = msg.detections
    np.testing.assert_array_equal(bbox_to_box(det.bbox), [39.0, 39.0, 61.0, 61.0])
    assert det.bbox.center.position.x == 50.0 and det.bbox.size_x == 22.0
    xyz = _xyz(det)
    assert det.points.width == 22 * 22 == len(xyz)
    np.testing.assert_allclose(xyz[:, 2], 1.9, atol=1e-6)
    # The first point is pixel (39, 39), and row-major order ends at (60, 60).
    np.testing.assert_allclose(xyz[0], [(39 - 49.5) * 1.9 / 100, (39 - 49.5) * 1.9 / 100, 1.9], atol=1e-6)
    np.testing.assert_allclose(xyz[-1], [(60 - 49.5) * 1.9 / 100, (60 - 49.5) * 1.9 / 100, 1.9], atol=1e-6)
    assert det.track_id == 1 and det.label == "cup"
    assert det.score == pytest.approx(0.9) and det.view_quality == pytest.approx(0.75)


def test_every_stamp_is_the_given_stamp():
    msg = _stream().array(0, STAMP)
    (det,) = msg.detections
    assert msg.header.stamp == STAMP and msg.header.frame_id == "test_optical_frame"
    assert msg.camera_info.header == msg.header == det.points.header == det.crop.header
    check_object_detection_array(msg)


def test_the_camera_info_describes_the_camera():
    info = _stream().array(0, STAMP).camera_info
    assert (info.width, info.height) == (100, 100)
    assert list(info.k) == [100.0, 0.0, 49.5, 0.0, 100.0, 49.5, 0.0, 0.0, 1.0]
    assert list(info.p) == [100.0, 0.0, 49.5, 0.0, 0.0, 100.0, 49.5, 0.0, 0.0, 0.0, 1.0, 0.0]


def test_the_fixture_points_are_on_their_own_boxes_and_never_on_the_other():
    scene = load_scene(DEFAULT_SCENE)
    msg = FakeStream(scene, stream_id=STREAM_ID).array(0, STAMP)
    assert [d.label for d in msg.detections] == ["cup", "bottle"]
    boxes = {b.label: b for b in scene.boxes}
    for det in msg.detections:
        assert 30 <= det.points.width <= MAX_CLOUD_POINTS
        own, other = boxes[det.label], boxes["bottle" if det.label == "cup" else "cup"]
        points = _to_map(scene, _xyz(det))
        _assert_on_surface(points, own.center, own.size)
        assert np.all(np.any(np.abs(points - other.center) > other.size / 2, axis=1))


def test_the_fixture_camera_looks_along_the_map_x_axis():
    scene = load_scene(DEFAULT_SCENE)
    np.testing.assert_allclose(scene.camera.rotation(), [[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    (cup, bottle) = FakeStream(scene, stream_id=STREAM_ID).array(0, STAMP).detections
    # The cup is at map y +0.2, on the camera's left; the bottle is at -0.25, on its right.
    assert cup.bbox.center.position.x < scene.camera.cx < bottle.bbox.center.position.x
    # Both are below the camera's height of 1 m, so below the image's centre.
    assert cup.bbox.center.position.y > scene.camera.cy and bottle.bbox.center.position.y > scene.camera.cy


def test_a_large_box_keeps_at_most_the_cap_of_points():
    # 60 x 60 px holds 3600 points, and the cut must not change the box.
    msg = _stream(_box(size=[1.0, 1.0, 0.2], center=[0.0, 0.0, 1.4])).array(0, STAMP)
    (det,) = msg.detections
    assert det.points.width == MAX_CLOUD_POINTS
    box = bbox_to_box(det.bbox)
    assert (box[2] - box[0]) * (box[3] - box[1]) > MAX_CLOUD_POINTS
    assert len(det.points.data) == 12 * MAX_CLOUD_POINTS


def test_a_small_box_is_detected_with_an_empty_cloud_and_no_shares():
    # 0.05 m at z 2 is 2.5 px to each side: 5 x 5 px, under both floors.
    (det,) = _stream(_box(size=[0.05, 0.05, 0.05])).array(0, STAMP).detections
    assert det.points.width == 0 and len(det.points.data) == 0 and len(det.points.fields) == 3
    assert len(det.color_shares) == 0


def test_a_box_outside_the_view_has_no_detection():
    msg = _stream(_box(center=[5.0, 0.0, 2.0])).array(0, STAMP)
    assert msg.detections == []
    check_object_detection_array(msg)


def test_a_box_behind_the_camera_has_no_detection():
    assert _stream(_box(center=[0.0, 0.0, -2.0])).array(0, STAMP).detections == []


def test_a_box_beyond_the_depth_range_is_seen_without_points():
    (det,) = _stream(_box(center=[0.0, 0.0, 4.0], size=[1.0, 1.0, 0.2])).array(0, STAMP).detections
    assert det.points.width == 0 and det.bbox.size_x > 0


def test_a_nearer_box_hides_the_pixels_behind_it():
    far = _bottle(center=[0.0, 0.0, 2.5], size=[1.0, 1.0, 0.2])
    (near_det, far_det) = _stream(_box(), far).array(0, STAMP).detections
    assert near_det.points.width == 22 * 22
    # The far face at z 2.4 is 0.5 * 100 / 2.4 = 20.8 px to each side: pixels 29 to 70, or 42 x 42.
    np.testing.assert_array_equal(bbox_to_box(far_det.bbox), [29.0, 29.0, 71.0, 71.0])
    assert far_det.points.width == 42 * 42 - 22 * 22
    np.testing.assert_allclose(_xyz(far_det)[:, 2], 2.4, atol=1e-5)
    np.testing.assert_allclose(_xyz(near_det)[:, 2], 1.9, atol=1e-5)


def test_a_fully_hidden_box_has_no_detection():
    far = _bottle(center=[0.0, 0.0, 2.5], size=[0.1, 0.1, 0.1])
    (det,) = _stream(_box(), far).array(0, STAMP).detections
    assert det.label == "cup"


def test_a_box_a_phase_leaves_out_does_not_hide_another():
    far = _bottle(center=[0.0, 0.0, 2.5], size=[0.1, 0.1, 0.1])
    sequence = [{"frames": 1, "visible": ["cup_1", "bottle_1"]}, {"frames": 1, "visible": ["bottle_1"]}]
    stream = _stream(_box(), far, sequence=sequence)
    assert [d.label for d in stream.array(0, STAMP).detections] == ["cup"]
    assert [d.label for d in stream.array(1, STAMP).detections] == ["bottle"]


def test_the_fixture_sequence_drops_out_empties_and_moves():
    scene = load_scene(DEFAULT_SCENE)
    stream = FakeStream(scene, stream_id=STREAM_ID)
    assert stream.frame_count == 30
    labels = [[d.label for d in stream.array(i, STAMP).detections] for i in range(stream.frame_count)]
    assert labels == [["cup", "bottle"]] * 10 + [["cup"]] * 5 + [[]] * 5 + [["cup", "bottle"]] * 10
    first = stream.array(0, STAMP).detections[0]
    moved = stream.array(20, STAMP).detections[0]
    assert moved.track_id == first.track_id
    # 0.15 m to the left at a depth of 1.45 m is about 54 px.
    assert first.bbox.center.position.x - moved.bbox.center.position.x == pytest.approx(54, abs=2)
    _assert_on_surface(_to_map(scene, _xyz(moved)), [1.5, 0.35, 0.95], scene.boxes[0].size)


def test_a_move_lasts_until_the_sequence_restarts():
    sequence = [
        {"frames": 1, "visible": ["cup_1"]},
        {"frames": 1, "visible": ["cup_1"], "move": {"cup_1": [0.3, 0.0, 2.0]}},
        {"frames": 1, "visible": ["cup_1"]},
    ]
    stream = _stream(_box(), sequence=sequence)
    x = [stream.array(i, STAMP).detections[0].bbox.center.position.x for i in range(4)]
    assert x[1] == x[2] and x[1] > x[0]
    assert x[3] == x[0]


def test_a_finite_sequence_repeats_exactly():
    stream = _stream(_box(), sequence=[{"frames": 2, "visible": ["cup_1"]}, {"frames": 1, "visible": []}])
    assert stream.frame_count == 3
    for i in range(3):
        assert stream.array(i, STAMP) == stream.array(i + 3, STAMP)


def test_an_empty_frame_is_a_valid_array_with_the_searched_classes():
    scene = load_scene(DEFAULT_SCENE)
    stream = FakeStream(scene, stream_id=STREAM_ID)
    msg = stream.array(15, STAMP)
    assert msg.detections == []
    assert list(msg.classes) == list(scene.classes)
    assert msg.stream_id == STREAM_ID and msg.header.stamp == STAMP
    check_object_detection_array(deserialize_message(serialize_message(msg), ObjectDetectionArray))


def test_every_frame_follows_the_contract_on_the_wire_format():
    stream = FakeStream(load_scene(DEFAULT_SCENE))
    for i in range(stream.frame_count):
        back = deserialize_message(serialize_message(stream.array(i, STAMP)), ObjectDetectionArray)
        check_object_detection_array(back)


def test_two_streams_give_equal_messages():
    # Message equality covers every field, with the cloud's and the crop's
    # bytes. Serialized bytes would not do: CDR leaves its padding bytes unset.
    first = FakeStream(load_scene(DEFAULT_SCENE), stream_id=STREAM_ID)
    second = FakeStream(load_scene(DEFAULT_SCENE), stream_id=STREAM_ID)
    for i in range(first.frame_count):
        one, other = first.array(i, STAMP), second.array(i, STAMP)
        assert one == other
        for a, b in zip(one.detections, other.detections, strict=True):
            assert bytes(a.points.data) == bytes(b.points.data)
            assert bytes(a.crop.data) == bytes(b.crop.data)


def test_a_new_stream_gets_a_new_id():
    scene = load_scene(DEFAULT_SCENE)
    assert FakeStream(scene).stream_id != FakeStream(scene).stream_id


def test_a_crop_is_the_padded_box_as_a_jpeg():
    (det,) = _stream().array(0, STAMP).detections
    assert det.crop.format == "jpeg" and bytes(det.crop.data[:2]) == b"\xff\xd8"
    pixels = cv2.imdecode(np.frombuffer(bytes(det.crop.data), dtype=np.uint8), cv2.IMREAD_COLOR)
    # 22 px with 15% on each side is 3.3 px: from floor(35.7) to ceil(64.3), 30 px. Never upscaled.
    assert pixels.shape == (30, 30, 3)
    r, g, b = PALETTE_RGB[COLOR_NAMES.index("red")]
    np.testing.assert_allclose(pixels[15, 15], [b, g, r], atol=6)


def test_a_crop_is_clipped_to_the_image():
    (det,) = _stream(_box(center=[-0.9, -0.9, 2.0])).array(0, STAMP).detections
    pixels = cv2.imdecode(np.frombuffer(bytes(det.crop.data), dtype=np.uint8), cv2.IMREAD_COLOR)
    box = bbox_to_box(det.bbox)
    assert box[0] == 0.0 and box[1] == 0.0
    assert pixels.shape[0] < (box[3] - box[1]) * 1.3 and pixels.shape[1] < (box[2] - box[0]) * 1.3


def test_a_crop_keeps_to_the_longest_side_limit():
    camera = {**CAMERA, "width": 1280, "height": 960, "fx": 600.0, "fy": 600.0, "cx": 639.5, "cy": 479.5}
    (det,) = _stream(_box(size=[2.0, 1.0, 0.2], center=[0.0, 0.0, 1.4]), camera=camera).array(0, STAMP).detections
    pixels = cv2.imdecode(np.frombuffer(bytes(det.crop.data), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert max(pixels.shape[:2]) == 512 and pixels.shape[1] > pixels.shape[0]


def test_crops_can_be_turned_off():
    (det,) = _stream(crops_enabled=False).array(0, STAMP).detections
    assert len(det.crop.data) == 0 and det.crop.header == _stream().array(0, STAMP).header
    assert det.points.width > 0


def test_color_shares_name_the_box_color():
    (det,) = _stream().array(0, STAMP).detections
    shares = np.asarray(det.color_shares)
    assert shares.shape == (12,) and shares[COLOR_NAMES.index("red")] == 1.0 and shares.sum() == 1.0


def test_a_box_without_a_color_has_no_shares_and_no_quality_by_default():
    box = _box()
    del box["color"], box["view_quality"]
    (det,) = _stream(box).array(0, STAMP).detections
    assert len(det.color_shares) == 0 and det.view_quality == 0.0


def test_there_are_no_vectors_by_default():
    for det in FakeStream(load_scene(DEFAULT_SCENE)).array(0, STAMP).detections:
        assert det.embedding_model == "" and len(det.embedding) == 0


def test_a_fake_model_gives_stable_unit_vectors_per_box():
    stream = FakeStream(load_scene(DEFAULT_SCENE), embedding_model="fake_clip", embedding_dim=8)
    (cup, bottle) = stream.array(0, STAMP).detections
    assert cup.embedding_model == bottle.embedding_model == "fake_clip"
    assert len(cup.embedding) == 8 and np.linalg.norm(np.asarray(cup.embedding)) == pytest.approx(1.0)
    assert list(cup.embedding) != list(bottle.embedding)
    assert list(stream.array(5, STAMP).detections[0].embedding) == list(cup.embedding)


@pytest.mark.parametrize("name", ["clip_vit_b16", "my_fake", "FAKE_clip"])
def test_a_real_looking_model_name_is_refused(name):
    with pytest.raises(ValueError, match="embedding_model"):
        FakeStream(load_scene(DEFAULT_SCENE), embedding_model=name)


def test_a_classes_list_narrows_the_detections():
    msg = FakeStream(load_scene(DEFAULT_SCENE), classes=["bottle", "shelf"]).array(0, STAMP)
    assert list(msg.classes) == ["bottle", "shelf"]
    assert [d.label for d in msg.detections] == ["bottle"]
    check_object_detection_array(msg)


@pytest.mark.parametrize("classes", [["person"], ["cup", "person"], [], ["cup", "cup"], ["cup", ""]])
def test_a_bad_classes_list_is_refused(classes):
    with pytest.raises(ValueError, match="classes"):
        FakeStream(load_scene(DEFAULT_SCENE), classes=classes)


def _change(**changes):
    def apply(data: dict) -> None:
        data.update(copy.deepcopy(changes))

    return apply


def _change_box(**changes):
    def apply(data: dict) -> None:
        data["boxes"][0].update(changes)

    return apply


def _change_camera(**changes):
    def apply(data: dict) -> None:
        data["camera"] = {**data["camera"], **changes}

    return apply


def _change_phase(**changes):
    def apply(data: dict) -> None:
        data["sequence"][0].update(changes)

    return apply


def _drop_box_field(name: str):
    def apply(data: dict) -> None:
        del data["boxes"][0][name]

    return apply


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(_change(format=2), id="format"),
        pytest.param(_change(extra=1), id="unknown-field"),
        pytest.param(_change(classes=[]), id="no-classes"),
        pytest.param(_change(classes=["cup", "person"]), id="person-class"),
        pytest.param(_change(boxes=[]), id="no-boxes"),
        pytest.param(_change(sequence=[]), id="no-phases"),
        pytest.param(_change_box(label="sofa"), id="label-not-a-class"),
        pytest.param(_change_box(label="person"), id="person-box"),
        pytest.param(_change_box(score=1.5), id="score-high"),
        pytest.param(_change_box(score=float("nan")), id="score-nan"),
        pytest.param(_change_box(score=True), id="score-bool"),
        pytest.param(_change_box(view_quality=-0.1), id="quality-low"),
        pytest.param(_change_box(color="teal"), id="color"),
        pytest.param(_change_box(track_id=-1), id="track-negative"),
        pytest.param(_change_box(track_id=1.5), id="track-float"),
        pytest.param(_change_box(size=[0.1, 0.0, 0.1]), id="size-zero"),
        pytest.param(_change_box(center=[0.0, 0.0]), id="center-short"),
        pytest.param(_change_box(center=[0.0, float("inf"), 1.0]), id="center-inf"),
        pytest.param(_drop_box_field("score"), id="box-missing-field"),
        pytest.param(_change_camera(width=0), id="camera-width"),
        pytest.param(_change_camera(fx=0.0), id="camera-fx"),
        pytest.param(_change_camera(orientation_xyzw=[0.0, 0.0, 0.0, 2.0]), id="camera-quaternion"),
        pytest.param(_change_camera(depth_range=[3.0, 0.2]), id="camera-depth-range"),
        pytest.param(_change_camera(frame_id=""), id="camera-frame"),
        pytest.param(_change_phase(frames=0), id="phase-frames"),
        pytest.param(_change_phase(visible=["nobody"]), id="phase-unknown-box"),
        pytest.param(_change_phase(visible=["cup_1", "cup_1"]), id="phase-duplicate-box"),
        pytest.param(_change_phase(visible="cup_1"), id="phase-visible-string"),
        pytest.param(_change_phase(move={"nobody": [0.0, 0.0, 1.0]}), id="phase-move-unknown"),
        pytest.param(_change_phase(move={"cup_1": [0.0, 0.0]}), id="phase-move-short"),
    ],
)
def test_a_bad_scene_is_refused(change):
    data = _scene_data()
    parse_scene(data)
    change(data)
    with pytest.raises(ValueError):
        parse_scene(data)


def test_duplicate_box_names_and_track_ids_are_refused():
    with pytest.raises(ValueError, match="names"):
        parse_scene(_scene_data(_box(), _bottle(name="cup_1")))
    with pytest.raises(ValueError, match="track ids"):
        parse_scene(_scene_data(_box(), _bottle(track_id=1)))


def test_a_camera_inside_a_box_is_refused():
    with pytest.raises(ValueError, match="inside"):
        _stream(_box(center=[0.0, 0.0, 0.0]))


def test_the_task_runs_on_fast_dds_in_a_private_domain():
    task = tomllib.loads((REPO / "pixi.toml").read_text())["tasks"]["fake-detections"]
    assert task["cmd"] == "python -m walkie_vision.fake_publisher"
    assert task["env"]["RMW_IMPLEMENTATION"] == "rmw_fastrtps_cpp"
    assert task["env"]["ROS_DOMAIN_ID"] == "77"
    assert task["env"]["ROS_AUTOMATIC_DISCOVERY_RANGE"] == "LOCALHOST"


def test_importing_the_publisher_loads_no_model_library():
    code = (
        "import sys; import walkie_vision.fake_publisher; "
        "bad = {'torch', 'torchvision', 'ultralytics', 'onnxruntime', 'open_clip', 'transformers', 'timm', 'insightface'};"
        "print(sorted(bad & set(sys.modules)))"
    )
    out = subprocess.run([sys.executable, "-P", "-c", code], capture_output=True, text=True, timeout=60, check=True)
    assert out.stdout.strip() == "[]"


# Wire tests.


@pytest.fixture
def ctx():
    context = rclpy.Context()
    rclpy.init(context=context, domain_id=77)
    yield context
    context.try_shutdown()


class _Wire:
    """A helper node with a collector on a unique topic, subscribed with the given profile."""

    def __init__(self, context, qos: QoSProfile) -> None:
        self.topic = f"/t_{uuid.uuid4().hex[:10]}/objects"
        self.node = rclpy.create_node(f"test_{uuid.uuid4().hex[:10]}", context=context)
        self.got: list[ObjectDetectionArray] = []
        self.node.create_subscription(ObjectDetectionArray, self.topic, self.got.append, qos)
        self.executor = SingleThreadedExecutor(context=context)
        self.executor.add_node(self.node)

    def spin_until(self, condition, timeout_s: float = TIMEOUT_S) -> None:
        deadline = time.monotonic() + timeout_s
        while not condition():
            assert time.monotonic() < deadline, f"timed out with {len(self.got)} arrays"
            self.executor.spin_once(timeout_sec=0.05)

    def close(self) -> None:
        self.executor.shutdown()
        self.node.destroy_node()


@pytest.fixture
def wire(ctx, request):
    """A collector with walkie-database's profile, or with the profile an indirect parametrize gives."""
    w = _Wire(ctx, getattr(request, "param", DATABASE_QOS))
    yield w
    w.close()


def _publisher(ctx, wire: _Wire, **overrides) -> FakePublisher:
    params = [Parameter("objects_topic", value=wire.topic), Parameter("rate_hz", value=100.0)]
    params += [Parameter(k, value=v) for k, v in overrides.items()]
    node = FakePublisher(context=ctx, parameter_overrides=params)
    wire.executor.add_node(node)
    return node


@pytest.mark.parametrize("wire", [DATABASE_QOS, RELIABLE_QOS], ids=["database", "reliable"], indirect=True)
def test_a_subscriber_sees_empty_and_nonempty_arrays(ctx, wire):
    node = _publisher(ctx, wire)
    try:
        wire.spin_until(lambda: {bool(m.detections) for m in wire.got} == {True, False})
    finally:
        node.destroy_node()
    for msg in wire.got:
        check_object_detection_array(msg)
        assert msg.stream_id == node.stream_id
        assert list(msg.classes) == list(load_scene(DEFAULT_SCENE).classes)
    stamps = [m.header.stamp.sec * 10**9 + m.header.stamp.nanosec for m in wire.got]
    assert stamps == sorted(stamps) and stamps[0] > 0
    full = next(m for m in wire.got if m.detections)
    assert all(d.points.width > 0 and len(d.crop.data) > 0 and len(d.embedding) == 0 for d in full.detections)


def test_a_subscriber_sees_fake_vectors_when_asked(ctx, wire):
    node = _publisher(ctx, wire, embedding_model="fake_clip", classes=["cup"])
    try:
        wire.spin_until(lambda: any(m.detections for m in wire.got))
    finally:
        node.destroy_node()
    det = next(m for m in wire.got if m.detections).detections[0]
    assert det.label == "cup" and det.embedding_model == "fake_clip" and len(det.embedding) == 512
    assert all(list(m.classes) == ["cup"] for m in wire.got)


def test_the_fake_publishes_with_the_vision_node_output_qos():
    # One profile for both publishers, so a consumer that matches the fake matches the real node.
    assert OBSERVATION_QOS == OUTPUT_QOS
    for subscriber in (DATABASE_QOS, RELIABLE_QOS):
        compatibility, reason = qos_check_compatible(OBSERVATION_QOS, subscriber)
        assert compatibility != QoSCompatibility.ERROR, reason


def test_the_publisher_offers_reliable_keep_last_5_volatile_and_no_people(ctx, wire):
    node = _publisher(ctx, wire)
    try:
        # The graph does not carry history or depth, so those come from the publisher itself.
        (publisher,) = [p for p in node.publishers if p.topic_name == wire.topic]
        assert publisher.qos_profile.history == HistoryPolicy.KEEP_LAST
        assert publisher.qos_profile.depth == 5
        (info,) = node.get_publishers_info_by_topic(wire.topic)
        assert info.qos_profile.reliability == ReliabilityPolicy.RELIABLE
        assert info.qos_profile.durability == DurabilityPolicy.VOLATILE
        published = dict(node.get_publisher_names_and_types_by_node(node.get_name(), node.get_namespace()))
        assert published[wire.topic] == ["walkie_vision_msgs/msg/ObjectDetectionArray"]
        assert not any("PersonDetection" in t for types in published.values() for t in types)
    finally:
        node.destroy_node()


def test_a_late_listener_gets_the_static_camera_transform(ctx, wire):
    node = _publisher(ctx, wire)
    got: list[TFMessage] = []
    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    try:
        wire.node.create_subscription(TFMessage, "/tf_static", got.append, qos)
        wire.spin_until(lambda: any(t.child_frame_id == "fake_camera_optical_frame" for m in got for t in m.transforms))
    finally:
        node.destroy_node()
    (transform,) = next(m for m in got if m.transforms[0].child_frame_id == "fake_camera_optical_frame").transforms
    assert transform.header.frame_id == "map"
    assert transform.transform.translation.z == 1.0
    r = transform.transform.rotation
    assert (r.x, r.y, r.z, r.w) == (-0.5, 0.5, -0.5, 0.5)


def test_tf_can_be_turned_off(ctx, wire):
    node = _publisher(ctx, wire, publish_tf=False)
    try:
        published = dict(node.get_publisher_names_and_types_by_node(node.get_name(), node.get_namespace()))
        assert "/tf_static" not in published
    finally:
        node.destroy_node()


class _Clock:
    def __init__(self) -> None:
        self.nanoseconds = 0

    def now(self) -> RosTime:
        return RosTime(nanoseconds=self.nanoseconds)


def test_a_clock_at_zero_or_paused_publishes_nothing_new(ctx, wire):
    node = FakePublisher(
        context=ctx, parameter_overrides=[Parameter("objects_topic", value=wire.topic), Parameter("rate_hz", value=1.0)]
    )
    clock = _Clock()
    node.get_clock = lambda: clock
    try:
        # Under sim time, before /clock starts, the clock reads 0.
        assert node.publish_next() is None
        clock.nanoseconds = 5_000_000_000
        assert node.publish_next().header.stamp == Time(sec=5)
        assert node.publish_next() is None
        clock.nanoseconds += 1
        assert node.publish_next().header.stamp == Time(sec=5, nanosec=1)
    finally:
        node.destroy_node()


def _env() -> dict:
    return dict(
        os.environ, ROS_DOMAIN_ID="77", RMW_IMPLEMENTATION="rmw_fastrtps_cpp", ROS_AUTOMATIC_DISCOVERY_RANGE="LOCALHOST"
    )


def _command(*params: str) -> list[str]:
    args = [arg for param in params for arg in ("-p", param)]
    return [sys.executable, "-P", "-m", "walkie_vision.fake_publisher", "--ros-args", *args]


def _stream_id_of_one_run(wire: _Wire, log: Path) -> str:
    wire.got.clear()
    with log.open("w") as out:
        process = subprocess.Popen(
            _command(f"objects_topic:={wire.topic}", "rate_hz:=50.0"), env=_env(), stdout=out, stderr=subprocess.STDOUT
        )
    try:
        wire.spin_until(lambda: len(wire.got) >= 3)
        stream_ids = {m.stream_id for m in wire.got}
        assert len(stream_ids) == 1, stream_ids
        process.terminate()
        assert process.wait(timeout=TIMEOUT_S) == 0, log.read_text()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    return stream_ids.pop()


def test_a_restart_gets_a_fresh_stream_id_and_the_same_track_ids(wire, tmp_path):
    first = _stream_id_of_one_run(wire, tmp_path / "first.log")
    second = _stream_id_of_one_run(wire, tmp_path / "second.log")
    assert first != second
    assert str(uuid.UUID(first)) == first and str(uuid.UUID(second)) == second
    assert first in (tmp_path / "first.log").read_text()
    assert "synthetic" in (tmp_path / "first.log").read_text()
    assert "rmw_fastrtps_cpp" in (tmp_path / "first.log").read_text()


_STOP_DURING_A_PUBLISH = """
import sys
import rclpy
from walkie_vision import fake_publisher

publish_next = fake_publisher.FakePublisher.publish_next

def stopped_first(self):
    rclpy.shutdown()
    return publish_next(self)

fake_publisher.FakePublisher.publish_next = stopped_first
sys.exit(fake_publisher.main())
"""


def test_a_stop_that_lands_during_a_publish_still_exits_zero():
    # SIGTERM shuts the context down from another thread, so a timer the wait
    # already picked can publish on a dead context. This forces that order.
    topic = f"/t_{uuid.uuid4().hex[:10]}/objects"
    out = subprocess.run(
        [sys.executable, "-P", "-c", _STOP_DURING_A_PUBLISH, "--ros-args", "-p", f"objects_topic:={topic}"],
        env=_env(),
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )
    assert out.returncode == 0, out.stdout + out.stderr


_CTRL_C_AND_A_LOST_SHUTDOWN_RACE = """
import sys
import rclpy
from walkie_vision import fake_publisher


def ctrl_c(node, timeout_sec=None):
    raise KeyboardInterrupt


def lose_the_race():
    context = rclpy.get_default_context()
    context.shutdown()
    context.shutdown()


rclpy.spin_once = ctrl_c
rclpy.try_shutdown = lose_the_race
sys.exit(fake_publisher.main())
"""


def test_ctrl_c_exits_0_when_the_signal_handler_shuts_rclpy_down_first():
    # After Ctrl-C, rclpy's signal handler shuts the context down from its own
    # thread a moment later, and can do it between try_shutdown's check and its
    # shutdown. The stand-in forces that order.
    topic = f"/t_{uuid.uuid4().hex[:10]}/objects"
    out = subprocess.run(
        [sys.executable, "-P", "-c", _CTRL_C_AND_A_LOST_SHUTDOWN_RACE, "--ros-args", "-p", f"objects_topic:={topic}"],
        env=_env(),
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "Traceback" not in out.stdout + out.stderr


def test_a_finite_run_ends_by_itself(tmp_path):
    topic = f"/t_{uuid.uuid4().hex[:10]}/objects"
    out = subprocess.run(
        _command(f"objects_topic:={topic}", "rate_hz:=200.0", "loop:=false"),
        env=_env(),
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "the sequence has ended" in out.stdout + out.stderr


@pytest.mark.parametrize(
    ("param", "reason"),
    [
        ("rate_hz:=0.0", "rate_hz"),
        ("embedding_model:=clip_vit_b16", "embedding_model"),
        ("classes:=[cup,person]", "person"),
        ("scene_file:=/nonexistent/boxes.json", "No such file"),
    ],
)
def test_a_bad_parameter_stops_the_start(param, reason):
    out = subprocess.run(_command(param), env=_env(), capture_output=True, text=True, timeout=TIMEOUT_S)
    assert out.returncode == 2, out.stdout + out.stderr
    assert reason in out.stdout + out.stderr
