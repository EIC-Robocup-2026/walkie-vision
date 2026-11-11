"""walkie_vision.debug_viz: the drawing, the markers, the image ring and the wire."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid

import cv2
import numpy as np
import pytest
import rclpy
from builtin_interfaces.msg import Time
from rcl_interfaces.msg import ParameterType
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from sensor_msgs.msg import CompressedImage, Image, PointCloud2, PointField
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from walkie_vision_msgs.msg import ObjectDetectionArray

from walkie_vision.convert import box_to_bbox, to_point_cloud
from walkie_vision.debug_viz import (
    ARRAY_QOS,
    IMAGE_QOS,
    NAMESPACE,
    PALETTE_RGB,
    DebugViz,
    ImageRing,
    build_markers,
    class_color,
    cloud_xyz,
    draw_detections,
    fit_scale,
    label_text,
    short_track,
)
from walkie_vision_msgs.msg import ObjectDetection

STAMP = Time(sec=1_792_400_000, nanosec=250_000_000)
FRAME = "zed_head_left_camera_frame_optical"
TIMEOUT_S = 20.0


def _header(stamp: Time = STAMP, frame: str = FRAME) -> Header:
    return Header(stamp=stamp, frame_id=frame)


def _image(stamp: Time = STAMP, frame: str = FRAME, width: int = 64, height: int = 48) -> Image:
    pixels = np.full((height, width, 3), 40, dtype=np.uint8)
    return Image(
        header=_header(stamp, frame),
        height=height,
        width=width,
        encoding="bgr8",
        step=width * 3,
        data=pixels.tobytes(),
    )


def _detection(label: str, box, points=None, track_id: int = 0, score: float = 0.5) -> ObjectDetection:
    det = ObjectDetection(label=label, score=score, track_id=track_id, bbox=box_to_bbox(box))
    det.points = to_point_cloud(_header(), points if points is not None else np.zeros((0, 3)))
    return det


def _cube(center, half: float = 0.05, n: int = 64) -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.asarray(center) + rng.uniform(-half, half, size=(n, 3))


def _array(detections, classes=("cup", "bottle")) -> ObjectDetectionArray:
    return ObjectDetectionArray(
        header=_header(), classes=list(classes), detections=list(detections), stream_id=str(uuid.uuid4())
    )


# Pure helpers.


def test_a_class_takes_the_colour_of_its_index_and_an_unknown_label_a_stable_one():
    assert class_color("cup", ["cup", "bottle"]) == PALETTE_RGB[0]
    assert class_color("bottle", ["cup", "bottle"]) == PALETTE_RGB[1]
    assert class_color("vase", ["cup"]) == class_color("vase", ["bottle", "cup"])
    assert class_color("vase", ["cup"]) in PALETTE_RGB


def test_a_label_names_the_class_the_score_and_the_short_track():
    assert short_track(0) == ""
    assert short_track(3414943733) == "#3733"
    assert label_text(_detection("cup", [0, 0, 10, 10], track_id=3414943733, score=0.837)) == "cup 0.84 #3733"
    assert label_text(_detection("cup", [0, 0, 10, 10])) == "cup 0.50"


def test_the_cloud_reader_reads_the_contract_layout():
    pts = _cube([0.1, -0.2, 1.5])
    xyz = cloud_xyz(to_point_cloud(_header(), pts))
    assert xyz.shape == pts.shape
    np.testing.assert_allclose(xyz, pts.astype(np.float32), rtol=0, atol=0)


def test_the_cloud_reader_honours_the_field_offsets_and_point_step():
    pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="<f4")
    # z, x, y order with 4 bytes of padding per point.
    rows = np.zeros((2, 16), dtype=np.uint8)
    rows[:, 0:4] = pts[:, 2:3].view(np.uint8)
    rows[:, 4:8] = pts[:, 0:1].view(np.uint8)
    rows[:, 8:12] = pts[:, 1:2].view(np.uint8)
    fields = [
        PointField(name="z", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="x", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    cloud = PointCloud2(height=1, width=2, fields=fields, point_step=16, row_step=32, data=rows.tobytes())
    np.testing.assert_array_equal(cloud_xyz(cloud), pts.astype(np.float64))


def test_the_cloud_reader_refuses_a_short_or_partial_cloud():
    assert cloud_xyz(PointCloud2()).shape == (0, 3)
    cloud = to_point_cloud(_header(), _cube([0, 0, 1]))
    cloud.data = bytes(cloud.data)[:-1]
    with pytest.raises(ValueError, match="fewer than"):
        cloud_xyz(cloud)
    partial = to_point_cloud(_header(), _cube([0, 0, 1]))
    partial.fields = partial.fields[:2]
    with pytest.raises(ValueError, match="lacks"):
        cloud_xyz(partial)


def test_fit_scale_only_shrinks():
    assert fit_scale(2208, 1242, 960) == pytest.approx(960 / 2208)
    assert fit_scale(640, 480, 960) == 1.0


def test_drawing_marks_the_box_in_the_class_colour_and_keeps_the_input():
    bgr = np.full((120, 160, 3), 40, dtype=np.uint8)
    before = bgr.copy()
    out = draw_detections(bgr, [_detection("bottle", [40, 50, 100, 110])], ["cup", "bottle"])
    np.testing.assert_array_equal(bgr, before)
    r, g, b = PALETTE_RGB[1]
    # The left edge of the box, below the label, is in the class colour.
    assert tuple(out[90, 40]) == (b, g, r)
    assert out.shape == bgr.shape


def test_drawing_scales_the_image_and_the_boxes_together():
    bgr = np.full((200, 400, 3), 40, dtype=np.uint8)
    out = draw_detections(bgr, [_detection("cup", [100, 100, 300, 180])], ["cup"], scale=0.5)
    assert out.shape == (100, 200, 3)
    r, g, b = PALETTE_RGB[0]
    assert tuple(out[75, 50]) == (b, g, r)


def test_markers_clear_first_then_points_and_a_label_per_placed_detection():
    placed = _detection("cup", [0, 0, 10, 10], _cube([0.1, -0.2, 1.5]), track_id=12)
    unplaced = _detection("bottle", [20, 20, 30, 30])
    markers = build_markers(_array([unplaced, placed])).markers
    assert markers[0].action == Marker.DELETEALL and markers[0].ns == NAMESPACE
    assert [m.type for m in markers[1:]] == [Marker.POINTS, Marker.TEXT_VIEW_FACING]
    points, text = markers[1:]
    assert points.header.frame_id == FRAME and points.header.stamp == STAMP
    assert len(points.points) == 64 and points.ns == text.ns == NAMESPACE
    assert points.id != text.id
    assert text.text == "cup 0.50 #0012"
    # The label sits above the cloud: smaller y in the optical frame.
    assert text.pose.position.y < min(p.y for p in points.points)
    assert (points.color.r, points.color.g, points.color.b) == tuple(c / 255.0 for c in PALETTE_RGB[0])


def test_markers_thin_a_big_cloud():
    det = _detection("cup", [0, 0, 10, 10], _cube([0, 0, 1.0], n=2000))
    (_, points, _) = build_markers(_array([det]), max_points=500).markers
    assert len(points.points) == 500


def test_an_empty_array_still_clears_the_old_markers():
    (clear,) = build_markers(_array([])).markers
    assert clear.action == Marker.DELETEALL


def test_the_ring_finds_an_image_by_exact_stamp_and_frame():
    ring = ImageRing(capacity=3, max_age_s=10.0)
    first = _image(Time(sec=10))
    ring.add(first)
    assert ring.find(_header(Time(sec=10))) is first
    assert ring.find(_header(Time(sec=10, nanosec=1))) is None
    assert ring.find(_header(Time(sec=10), frame="other")) is None


def test_the_ring_keeps_only_its_capacity_and_its_age():
    ring = ImageRing(capacity=2, max_age_s=0.5)
    for sec in (10, 11, 12):
        ring.add(_image(Time(sec=sec)))
    assert len(ring) == 1  # 11 is more than 0.5 s older than 12, and 10 fell out first
    assert ring.find(_header(Time(sec=12))) is not None
    ring = ImageRing(capacity=2, max_age_s=10.0)
    for sec in (10, 11, 12):
        ring.add(_image(Time(sec=sec)))
    assert ring.find(_header(Time(sec=10))) is None and len(ring) == 2


def test_a_clock_jump_back_empties_the_ring():
    ring = ImageRing(capacity=5, max_age_s=10.0)
    ring.add(_image(Time(sec=100)))
    ring.add(_image(Time(sec=5)))
    assert len(ring) == 1 and ring.find(_header(Time(sec=5))) is not None


def test_dropping_through_a_stamp_keeps_only_newer_images():
    ring = ImageRing(capacity=5, max_age_s=10.0)
    ring.add(_image(Time(sec=1)))
    ring.add(_image(Time(sec=2)))
    ring.add(_image(Time(sec=2), frame="other_frame"))
    ring.add(_image(Time(sec=3)))
    assert len(ring) == 4
    ring.drop_through(Time(sec=2))
    assert len(ring) == 1
    assert ring.find(_header(Time(sec=3))) is not None


def test_the_ring_refuses_a_bad_size():
    with pytest.raises(ValueError):
        ImageRing(capacity=0, max_age_s=1.0)
    with pytest.raises(ValueError):
        ImageRing(capacity=1, max_age_s=float("inf"))


# The wire.


@pytest.fixture
def ctx():
    context = rclpy.Context()
    # Not 77 or 78: the simulations on this machine use those.
    rclpy.init(context=context, domain_id=79)
    yield context
    context.try_shutdown()


def test_the_node_publishes_the_annotated_image_and_the_markers(ctx):
    base = f"/t_{uuid.uuid4().hex[:10]}"
    viz = DebugViz(
        context=ctx,
        parameter_overrides=[
            Parameter("objects_topic", value=f"{base}/objects"),
            Parameter("image_topic", value=f"{base}/image"),
            Parameter("image_out_topic", value=f"{base}/debug/objects_image/compressed"),
            Parameter("markers_topic", value=f"{base}/debug/objects_markers"),
            Parameter("max_side", value=32),
        ],
    )
    helper = rclpy.create_node(f"test_{uuid.uuid4().hex[:10]}", context=ctx)
    images: list[CompressedImage] = []
    markers: list[MarkerArray] = []
    helper.create_subscription(CompressedImage, f"{base}/debug/objects_image/compressed", images.append, 5)
    helper.create_subscription(MarkerArray, f"{base}/debug/objects_markers", markers.append, 5)
    image_pub = helper.create_publisher(Image, f"{base}/image", IMAGE_QOS)
    array_pub = helper.create_publisher(ObjectDetectionArray, f"{base}/objects", ARRAY_QOS)
    executor = SingleThreadedExecutor(context=ctx)
    executor.add_node(viz)
    executor.add_node(helper)
    array = _array([_detection("cup", [10, 10, 40, 30], _cube([0.0, 0.0, 1.0]))])
    try:
        deadline = time.monotonic() + TIMEOUT_S
        # Publish until both outputs arrive: discovery can drop the first sends.
        while not (images and markers):
            assert time.monotonic() < deadline, f"timed out: {len(images)} images, {len(markers)} marker arrays"
            image_pub.publish(_image())
            for _ in range(5):
                executor.spin_once(timeout_sec=0.02)
            array_pub.publish(array)
            for _ in range(5):
                executor.spin_once(timeout_sec=0.02)
    finally:
        executor.shutdown()
        helper.destroy_node()
        viz.destroy_node()
    out = images[0]
    assert out.format == "jpeg" and out.header.stamp == STAMP and out.header.frame_id == FRAME
    pixels = cv2.imdecode(np.frombuffer(bytes(out.data), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert pixels.shape == (24, 32, 3)  # 64 x 48 scaled to a longest side of 32
    assert markers[0].markers[0].action == Marker.DELETEALL
    kinds = [m.type for m in markers[0].markers[1:]]
    assert kinds == [Marker.POINTS, Marker.TEXT_VIEW_FACING]


def test_the_node_refuses_a_bad_quality(ctx):
    with pytest.raises(ValueError, match="jpeg_quality"):
        DebugViz(context=ctx, parameter_overrides=[Parameter("jpeg_quality", value=0)])


@pytest.mark.parametrize("lifetime", [float("nan"), float("inf"), -0.5])
def test_the_node_refuses_a_bad_marker_lifetime(ctx, lifetime):
    with pytest.raises(ValueError, match="marker_lifetime_s"):
        DebugViz(context=ctx, parameter_overrides=[Parameter("marker_lifetime_s", value=lifetime)])


def test_each_parameter_describes_itself(ctx):
    viz = DebugViz(context=ctx)
    try:
        for name, kind in [
            ("objects_topic", ParameterType.PARAMETER_STRING),
            ("max_side", ParameterType.PARAMETER_INTEGER),
            ("marker_lifetime_s", ParameterType.PARAMETER_DOUBLE),
        ]:
            described = viz.describe_parameter(name)
            assert (described.name, described.type, described.read_only) == (name, kind, True)
    finally:
        viz.destroy_node()


def test_a_handled_array_frees_its_image_and_older_ones(ctx):
    viz = DebugViz(context=ctx, parameter_overrides=[Parameter("max_side", value=32)])
    try:
        for sec in (1, 2, 3):
            viz._ring.add(_image(Time(sec=sec)))
        array = _array([_detection("cup", [10, 10, 40, 30])])
        array.header = _header(Time(sec=2))
        viz._on_array(array)
        assert len(viz._ring) == 1
        assert viz._ring.find(_header(Time(sec=3))) is not None
    finally:
        viz.destroy_node()


_CTRL_C_AND_A_LOST_SHUTDOWN_RACE = """
import sys
import rclpy
from walkie_vision import debug_viz


def ctrl_c(node):
    raise KeyboardInterrupt


def lose_the_race():
    context = rclpy.get_default_context()
    context.shutdown()
    context.shutdown()


rclpy.spin = ctrl_c
rclpy.try_shutdown = lose_the_race
sys.exit(debug_viz.main())
"""


def test_ctrl_c_exits_0_when_the_signal_handler_shuts_rclpy_down_first():
    # After Ctrl-C, rclpy's signal handler shuts the context down from its own
    # thread a moment later, and can do it between try_shutdown's check and its
    # shutdown. The stand-in forces that order.
    env = dict(
        os.environ, ROS_DOMAIN_ID="79", RMW_IMPLEMENTATION="rmw_fastrtps_cpp", ROS_AUTOMATIC_DISCOVERY_RANGE="LOCALHOST"
    )
    result = subprocess.run(
        [sys.executable, "-P", "-c", _CTRL_C_AND_A_LOST_SHUTDOWN_RACE, "--ros-args"],
        env=env,
        timeout=60,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "Traceback" not in output
