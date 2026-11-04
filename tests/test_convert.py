import array
import json
import uuid
from pathlib import Path

import numpy as np
import pytest
from ament_index_python.packages import get_package_share_directory
from rclpy.serialization import deserialize_message, serialize_message
from sensor_msgs.msg import CameraInfo, CompressedImage, PointCloud2, PointField
from std_msgs.msg import Header
from walkie_vision_msgs.msg import ObjectDetection, ObjectDetectionArray, PersonDetection, PersonDetectionArray
from walkie_vision_msgs.srv import DescribeCrop, DetectObjects, EmbedCrop, EmbedText, GetPersonCrop

from walkie_vision import crop
from walkie_vision.convert import (
    MAX_CLOUD_POINTS,
    MIN_CLOUD_POINTS,
    bbox_to_box,
    box_to_bbox,
    check_object_detection_array,
    empty_cloud,
    fill_embed_text_response,
    fill_person_crop_response,
    new_stream_id,
    pose_from_person_detection,
    to_object_detection,
    to_object_detection_array,
    to_person_detection,
    to_person_detection_array,
    to_point_cloud,
    unit_vector,
    vectors_from_response,
)
from walkie_vision.crop import COLOR_NAMES, crop_person
from walkie_vision.fakes import FakeEmbedder, FakeObjectDetector, figure_pose, paint_figure
from walkie_vision.providers import CLIP_VIT_B16, Detection, Face, PersonPose

# HD2K, the ZED 2i's resolution on the robot.
IMAGE = np.zeros((1242, 2208, 3), dtype=np.uint8)
STREAM_ID = "6f1c1b52-8c1e-4a53-9f0e-2d3b7a6c9e10"
JPEG = b"\xff\xd8\xff\xe0 a crop"


def _header() -> Header:
    header = Header(frame_id="camera_optical_frame")
    header.stamp.sec = 1_792_400_000
    header.stamp.nanosec = 250_000_000
    return header


def _camera_info() -> CameraInfo:
    info = CameraInfo(width=2208, height=1242, distortion_model="plumb_bob")
    info.header = _header()
    info.k = [1050.0, 0.0, 1104.0, 0.0, 1050.0, 621.0, 0.0, 0.0, 1.0]
    return info


def _pose(track_id: int = 2) -> PersonPose:
    keypoints = np.stack([np.arange(17) * 10.0 + 400, np.arange(17) * 20.0 + 100], axis=1)
    return PersonPose(
        box=np.array([380.0, 80.0, 620.0, 900.0]),
        score=0.93,
        keypoints=keypoints,
        keypoint_scores=np.linspace(0.1, 0.9, 17),
        track_id=track_id,
    )


# A message change must be deliberate: a node and walkie-database built from
# different definitions never match under rmw_zenoh, and nothing reports it
# (walkie-agent-ros ADR-0018). Change these together with the .msg files.
@pytest.mark.parametrize(
    ("msg_type", "fields"),
    [
        (
            ObjectDetection,
            {
                "label": "string",
                "score": "float",
                "track_id": "uint32",
                "bbox": "vision_msgs/BoundingBox2D",
                "embedding_model": "string",
                "embedding": "sequence<float>",
                "points": "sensor_msgs/PointCloud2",
                "crop": "sensor_msgs/CompressedImage",
                "view_quality": "float",
                "color_shares": "sequence<float>",
            },
        ),
        (
            ObjectDetectionArray,
            {
                "header": "std_msgs/Header",
                "camera_info": "sensor_msgs/CameraInfo",
                "classes": "sequence<string>",
                "detections": "sequence<walkie_vision_msgs/ObjectDetection>",
                "stream_id": "string",
            },
        ),
        (
            PersonDetection,
            {
                "score": "float",
                "track_id": "uint32",
                "bbox": "vision_msgs/BoundingBox2D",
                "keypoints": "float[34]",
                "keypoint_scores": "float[17]",
                "face_bbox": "vision_msgs/BoundingBox2D",
                "face_det_score": "float",
                "face_model": "string",
                "face_embedding": "sequence<float>",
                "appearance_model": "string",
                "appearance_embedding": "sequence<float>",
            },
        ),
        (
            PersonDetectionArray,
            {
                "header": "std_msgs/Header",
                "camera_info": "sensor_msgs/CameraInfo",
                "people": "sequence<walkie_vision_msgs/PersonDetection>",
            },
        ),
        (EmbedText.Request, {"texts": "sequence<string>"}),
        (
            EmbedText.Response,
            {
                "success": "boolean",
                "message": "string",
                "embedding_model": "string",
                "dim": "uint32",
                "vectors": "sequence<float>",
            },
        ),
        (
            GetPersonCrop.Request,
            {
                "track_id": "uint32",
                "padding": "float",
                "max_side": "uint16",
                "mask_others": "boolean",
                "min_keypoint_score": "float",
                "timeout_s": "float",
            },
        ),
        (
            GetPersonCrop.Response,
            {
                "status": "uint8",
                "message": "string",
                "header": "std_msgs/Header",
                "person": "walkie_vision_msgs/PersonDetection",
                "crop": "sensor_msgs/CompressedImage",
                "crop_keypoints": "sequence<float>",
                "masked_boxes": "sequence<float>",
                "max_overlap": "float",
                "torso_cover": "float",
                "color_names": "sequence<string>",
                "top_shares": "sequence<float>",
                "bottom_shares": "sequence<float>",
                "top_left": "float",
                "bottom_left": "float",
            },
        ),
        (
            DetectObjects.Request,
            {
                "classes": "sequence<string>",
                "max_image_age_s": "float",
                "timeout_s": "float",
            },
        ),
        (
            DetectObjects.Response,
            {
                "success": "boolean",
                "error_code": "string",
                "message": "string",
                "objects": "walkie_vision_msgs/ObjectDetectionArray",
            },
        ),
        (
            EmbedCrop.Request,
            {
                "header": "std_msgs/Header",
                "image_width": "uint32",
                "image_height": "uint32",
                "bbox": "vision_msgs/BoundingBox2D",
                "timeout_s": "float",
            },
        ),
        (
            EmbedCrop.Response,
            {
                "success": "boolean",
                "error_code": "string",
                "message": "string",
                "header": "std_msgs/Header",
                "crop": "sensor_msgs/CompressedImage",
                "embedding_model": "string",
                "embedding": "sequence<float>",
                "view_quality": "float",
            },
        ),
        (
            DescribeCrop.Request,
            {
                "crop": "sensor_msgs/CompressedImage",
                "color_shares": "sequence<float>",
                "timeout_s": "float",
            },
        ),
        (
            DescribeCrop.Response,
            {
                "success": "boolean",
                "error_code": "string",
                "message": "string",
                "caption": "string",
                "attributes_json": "string",
                "printed_text": "string",
                "caption_model": "string",
                "prompt_version": "string",
            },
        ),
    ],
)
def test_message_fields(msg_type, fields):
    # Field order is part of the wire format, and dict equality ignores it.
    assert list(msg_type.get_fields_and_field_types().items()) == list(fields.items())


# The hash that a peer's rmw compares. A change here is a wire contract change:
# walkie-database pins these messages by commit (ADR-0013).
@pytest.mark.parametrize(
    ("path", "rihs"),
    [
        ("msg/ObjectDetection", "RIHS01_6c81c3c4cb5323cfb2fb8704979706965b54c32d63245a2f35c59eec7e7a4416"),
        ("msg/ObjectDetectionArray", "RIHS01_785d36c74d98e32def95c792b51b867bde9cc69b730dec3b8b00e4b5adb5f51b"),
        ("msg/PersonDetection", "RIHS01_1b5c0af5d11a7954a285b093156e4b7d3369fc9767e510a6db89029d06f814c5"),
        ("msg/PersonDetectionArray", "RIHS01_22c693bc6a0b3a04fe91c64f8fbbd9b205f5a504a74cd9ff76a0c35f8bd407e0"),
        ("srv/DescribeCrop", "RIHS01_28fe63bde54bf4157e8a68a2cb3c4c197067ddcc2dcc64c1aabc2bc69129ead4"),
        ("srv/DetectObjects", "RIHS01_1a078f73087bc7a5473d880098ed6b27701c2968bd06193a48bec8515c4ff0a6"),
        ("srv/EmbedCrop", "RIHS01_61be523fc4cf3a76a3ff382fe74bfc77b8da650cc2ceb7871ee01b4ae451a841"),
        ("srv/EmbedText", "RIHS01_00dbadbc2a53cd59b500e85bc14639cbb0608dee2dbfaca33fcdf1ca112c17e0"),
        ("srv/GetPersonCrop", "RIHS01_e65042762a876ec78630be29f8af723cfdedcd53879285d49341f75a23ebd175"),
    ],
)
def test_type_hashes(path, rihs):
    described = Path(get_package_share_directory("walkie_vision_msgs")) / f"{path}.json"
    hashes = json.loads(described.read_text())["type_hashes"]
    assert {h["type_name"]: h["hash_string"] for h in hashes}[f"walkie_vision_msgs/{path}"] == rihs


def test_get_person_crop_request_defaults():
    request = GetPersonCrop.Request()
    assert request.track_id == 0
    assert request.padding == pytest.approx(0.10)
    assert request.max_side == 512
    assert request.mask_others is True
    assert request.min_keypoint_score == pytest.approx(0.5)
    assert request.timeout_s == pytest.approx(1.0)


def test_new_service_request_defaults():
    detect = DetectObjects.Request()
    assert list(detect.classes) == []
    assert detect.max_image_age_s == pytest.approx(0.3)
    assert detect.timeout_s == pytest.approx(4.0)
    assert EmbedCrop.Request().timeout_s == pytest.approx(1.0)
    assert DescribeCrop.Request().timeout_s == pytest.approx(4.0)


def test_color_order_is_the_one_the_message_names():
    assert list(COLOR_NAMES) == [
        "black", "white", "grey", "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "beige",
    ]  # fmt: skip


def test_box_round_trip():
    box = [12.25, 30.5, 100.75, 200.0]
    bbox = box_to_bbox(np.array(box))
    assert (bbox.center.position.x, bbox.center.position.y) == (56.5, 115.25)
    assert (bbox.size_x, bbox.size_y) == (88.5, 169.5)
    assert bbox.center.theta == 0.0
    np.testing.assert_allclose(bbox_to_box(bbox), box)


@pytest.mark.parametrize("box", [[10, 10, 5, 20], [0, 0, float("nan"), 5]])
def test_box_to_bbox_rejects_bad_boxes(box):
    with pytest.raises(ValueError):
        box_to_bbox(box)


def test_unit_vector():
    v = unit_vector([3.0, 4.0])
    assert v.dtype == np.float32
    np.testing.assert_allclose(v, [0.6, 0.8])


@pytest.mark.parametrize(
    "vector", [[], [0.0, 0.0], [1.0, float("nan")], [1.0, float("inf")], [[1.0, 0.0]]]
)
def test_unit_vector_rejects_bad_vectors(vector):
    with pytest.raises(ValueError):
        unit_vector(vector)


def test_object_detection_with_embedding():
    det = Detection(box=np.array([10.0, 20.0, 110.0, 220.0]), label="mug", score=0.75, track_id=7)
    crop = CompressedImage(format="jpeg", data=b"\xff\xd8\xff")
    msg = to_object_detection(det, embedding=np.eye(512)[1] * 2, embedding_model=CLIP_VIT_B16, crop=crop)
    assert (msg.label, msg.track_id, msg.embedding_model) == ("mug", 7, "clip_vit_b16")
    assert msg.score == pytest.approx(0.75)
    assert isinstance(msg.embedding, array.array) and msg.embedding.typecode == "f"
    assert list(msg.embedding) == list(np.eye(512)[1])
    assert msg.crop.format == "jpeg"
    # No depth image: the cloud stays empty.
    assert msg.points.width == 0 and len(msg.points.data) == 0


def test_object_detection_without_embedding():
    msg = to_object_detection(Detection(box=np.array([0.0, 0.0, 5.0, 5.0]), label="chair", score=0.5))
    assert msg.embedding_model == "" and len(msg.embedding) == 0
    assert msg.track_id == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"embedding": np.ones(4)},
        {"embedding_model": CLIP_VIT_B16},
        {"embedding": np.zeros(4), "embedding_model": CLIP_VIT_B16},
    ],
)
def test_object_detection_rejects_an_embedding_without_its_model(kwargs):
    with pytest.raises(ValueError):
        to_object_detection(Detection(box=np.array([0.0, 0.0, 5.0, 5.0]), label="mug", score=0.5), **kwargs)


def test_object_detection_rejects_a_negative_track_id():
    # Ultralytics reports -1 for an untracked box in some versions; 0 is the
    # message's value for that.
    with pytest.raises(ValueError):
        to_object_detection(Detection(box=np.array([0.0, 0.0, 5.0, 5.0]), label="mug", score=0.5, track_id=-1))


def test_empty_object_array_keeps_the_negative_evidence():
    msg = to_object_detection_array(_header(), _camera_info(), ["mug", "chair"], [], stream_id=STREAM_ID)
    assert list(msg.classes) == ["mug", "chair"]
    assert msg.detections == []
    assert msg.header.frame_id == "camera_optical_frame"
    assert msg.header.stamp.nanosec == 250_000_000
    assert (msg.camera_info.width, msg.camera_info.height) == (2208, 1242)
    assert msg.stream_id == STREAM_ID


def test_object_array_rejects_a_label_that_is_not_a_class():
    det = to_object_detection(Detection(box=np.array([0.0, 0.0, 5.0, 5.0]), label="mug", score=0.5))
    with pytest.raises(ValueError, match="mug"):
        to_object_detection_array(_header(), _camera_info(), ["chair"], [det], stream_id=STREAM_ID)


def _cloud_points(count: int = MIN_CLOUD_POINTS) -> np.ndarray:
    return np.column_stack([np.arange(count) * 0.25, np.arange(count) * 0.5, np.arange(count) * 0.125 + 1.0])


def _full_detection() -> ObjectDetection:
    """A detection with every optional part set, valid for _full_array()."""
    return to_object_detection(
        Detection(box=np.array([10.0, 20.0, 110.0, 220.0]), label="mug", score=0.75, track_id=7),
        embedding=np.eye(512)[0],
        embedding_model=CLIP_VIT_B16,
        points=to_point_cloud(_header(), _cloud_points()),
        crop=CompressedImage(header=_header(), format="jpeg", data=JPEG),
        view_quality=0.5,
        color_shares=np.eye(len(COLOR_NAMES))[3],
    )


def _full_array() -> ObjectDetectionArray:
    return to_object_detection_array(_header(), _camera_info(), ["mug", "chair"], [_full_detection()], stream_id=STREAM_ID)


def _set(path: str, value):
    """Return a change that sets one dotted field of an array's first detection."""

    def change(msg: ObjectDetectionArray) -> None:
        *parents, name = path.split(".")
        target = msg.detections[0]
        for parent in parents:
            target = getattr(target, parent)
        setattr(target, name, value)

    return change


def _set_array(name: str, value):
    return lambda msg: setattr(msg, name, value)


def _stamp_nanosec(target: str):
    def change(msg: ObjectDetectionArray) -> None:
        owner = msg if target == "camera_info" else msg.detections[0]
        getattr(owner, target).header.stamp.nanosec += 1

    return change


def _frame_id(target: str):
    def change(msg: ObjectDetectionArray) -> None:
        getattr(msg.detections[0], target).header.frame_id = "map"

    return change


def _point_field(index: int, **values):
    def change(msg: ObjectDetectionArray) -> None:
        for name, value in values.items():
            setattr(msg.detections[0].points.fields[index], name, value)

    return change


def _point_data(index: int, value: float):
    def change(msg: ObjectDetectionArray) -> None:
        xyz = np.frombuffer(bytes(msg.detections[0].points.data), dtype="<f4").reshape(-1, 3).copy()
        xyz[0, index] = value
        data = array.array("B")
        data.frombytes(xyz.tobytes())
        msg.detections[0].points.data = data

    return change


def _points_count(count: int):
    def change(msg: ObjectDetectionArray) -> None:
        points = msg.detections[0].points
        points.width = count
        points.row_step = 12 * count
        points.data = array.array("B", bytes(12 * count))

    return change


_NAN = float("nan")
_INF = float("inf")


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(_set_array("stream_id", ""), id="stream_id-empty"),
        pytest.param(_set_array("stream_id", "run-1"), id="stream_id-not-a-uuid"),
        pytest.param(_set_array("stream_id", STREAM_ID.upper()), id="stream_id-not-canonical"),
        pytest.param(_set_array("classes", []), id="classes-empty"),
        pytest.param(_set_array("classes", ["mug", "mug"]), id="classes-duplicate"),
        pytest.param(_set_array("classes", ["mug", ""]), id="classes-blank"),
        pytest.param(_stamp_nanosec("camera_info"), id="camera_info-stamp"),
        pytest.param(_set("label", "chair-leg"), id="label-not-a-class"),
        pytest.param(_set("score", _NAN), id="score-nan"),
        pytest.param(_set("score", _INF), id="score-inf"),
        pytest.param(_set("score", 1.01), id="score-high"),
        pytest.param(_set("score", -0.01), id="score-low"),
        pytest.param(_set("bbox.size_x", _NAN), id="bbox-nan"),
        pytest.param(_set("bbox.size_y", -1.0), id="bbox-negative"),
        pytest.param(_set("bbox.center.theta", 0.5), id="bbox-theta"),
        pytest.param(_set("embedding_model", ""), id="embedding-without-model"),
        pytest.param(_set("embedding", array.array("f")), id="model-without-embedding"),
        pytest.param(_set("embedding", array.array("f", [1.0])), id="embedding-wrong-length"),
        pytest.param(_set("embedding", array.array("f", [_NAN] + [0.0] * 511)), id="embedding-nan"),
        pytest.param(_set("embedding", array.array("f", [_INF] + [0.0] * 511)), id="embedding-inf"),
        pytest.param(_set("embedding", array.array("f", [0.5] + [0.0] * 511)), id="embedding-not-unit"),
        pytest.param(_set("embedding", array.array("f", [0.0] * 512)), id="embedding-zero"),
        pytest.param(_set("view_quality", _NAN), id="quality-nan"),
        pytest.param(_set("view_quality", _INF), id="quality-inf"),
        pytest.param(_set("view_quality", 1.01), id="quality-high"),
        pytest.param(_set("view_quality", -0.01), id="quality-low"),
        pytest.param(_set("color_shares", array.array("f", [1.0])), id="shares-wrong-length"),
        pytest.param(_set("color_shares", array.array("f", [0.5] + [0.0] * 11)), id="shares-sum"),
        pytest.param(_set("color_shares", array.array("f", [1.5, -0.5] + [0.0] * 10)), id="shares-negative"),
        pytest.param(_set("color_shares", array.array("f", [_NAN] + [0.0] * 11)), id="shares-nan"),
        pytest.param(_stamp_nanosec("points"), id="cloud-stamp"),
        pytest.param(_frame_id("points"), id="cloud-frame"),
        pytest.param(_point_field(0, name="u"), id="cloud-field-name"),
        pytest.param(_point_field(1, offset=8), id="cloud-field-offset"),
        pytest.param(_point_field(2, datatype=PointField.FLOAT64), id="cloud-field-type"),
        pytest.param(_point_field(0, count=2), id="cloud-field-count"),
        pytest.param(_set("points.fields", []), id="cloud-no-fields"),
        pytest.param(_set("points.is_bigendian", True), id="cloud-big-endian"),
        pytest.param(_set("points.is_dense", False), id="cloud-not-dense"),
        pytest.param(_set("points.height", 2), id="cloud-height"),
        pytest.param(_set("points.point_step", 16), id="cloud-point-step"),
        pytest.param(_set("points.row_step", 12), id="cloud-row-step"),
        pytest.param(_set("points.data", array.array("B", bytes(12))), id="cloud-short-data"),
        pytest.param(_points_count(MIN_CLOUD_POINTS - 1), id="cloud-under-the-floor"),
        pytest.param(_points_count(MAX_CLOUD_POINTS + 1), id="cloud-over-the-cap"),
        pytest.param(_point_data(0, _NAN), id="cloud-nan"),
        pytest.param(_point_data(1, _INF), id="cloud-inf"),
        pytest.param(_point_data(2, 0.0), id="cloud-behind-the-camera"),
        pytest.param(_stamp_nanosec("crop"), id="crop-stamp"),
        pytest.param(_frame_id("crop"), id="crop-frame"),
        pytest.param(_set("crop.format", "png"), id="crop-format"),
        pytest.param(_set("crop.data", array.array("B", b"\x89PNG")), id="crop-not-jpeg"),
    ],
)
def test_object_array_check_refuses(change):
    msg = _full_array()
    check_object_detection_array(msg)
    change(msg)
    with pytest.raises(ValueError):
        check_object_detection_array(msg)


@pytest.mark.parametrize(
    "stream_id", ["", "run-1", STREAM_ID.upper()], ids=["empty", "not-a-uuid", "not-canonical"]
)
def test_object_array_builder_refuses_a_bad_stream_id(stream_id):
    with pytest.raises(ValueError, match="stream_id"):
        to_object_detection_array(_header(), _camera_info(), ["mug"], [], stream_id=stream_id)


def test_object_array_builder_requires_a_stream_id():
    with pytest.raises(TypeError):
        to_object_detection_array(_header(), _camera_info(), ["mug"], [])


def test_object_array_builder_refuses_a_cloud_from_another_image():
    other = _header()
    other.stamp.sec += 1
    det = to_object_detection(
        Detection(box=np.array([0.0, 0.0, 5.0, 5.0]), label="mug", score=0.5),
        points=to_point_cloud(other, _cloud_points()),
    )
    with pytest.raises(ValueError, match="points.header"):
        to_object_detection_array(_header(), _camera_info(), ["mug"], [det], stream_id=STREAM_ID)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"view_quality": _NAN},
        {"view_quality": _INF},
        {"view_quality": 1.5},
        {"view_quality": -0.5},
        {"color_shares": [1.0]},
        {"color_shares": [0.5] + [0.0] * 11},
        {"color_shares": [1.5, -0.5] + [0.0] * 10},
        {"color_shares": [_NAN] + [0.0] * 11},
        {"embedding": np.ones(511), "embedding_model": CLIP_VIT_B16},
        {"embedding": np.ones(513), "embedding_model": CLIP_VIT_B16},
    ],
    ids=str,
)
def test_object_detection_refuses(kwargs):
    with pytest.raises(ValueError):
        to_object_detection(Detection(box=np.array([0.0, 0.0, 5.0, 5.0]), label="mug", score=0.5), **kwargs)


@pytest.mark.parametrize("score", [_NAN, _INF, -_INF, 1.01, -0.01])
def test_object_detection_refuses_a_score_outside_the_range(score):
    with pytest.raises(ValueError, match="score"):
        to_object_detection(Detection(box=np.array([0.0, 0.0, 5.0, 5.0]), label="mug", score=score))


def test_a_fake_vector_may_have_any_length_but_not_any_norm():
    # Only a real model's length is known here. A fake must still be a unit vector.
    det = Detection(box=np.array([0.0, 0.0, 5.0, 5.0]), label="mug", score=0.5)
    fake = to_object_detection(det, embedding=np.array([3.0, 4.0]), embedding_model="fake_embedder")
    assert list(fake.embedding) == pytest.approx([0.6, 0.8])
    msg = to_object_detection_array(_header(), _camera_info(), ["mug"], [fake], stream_id=STREAM_ID)
    msg.detections[0].embedding = array.array("f", [3.0, 4.0])
    with pytest.raises(ValueError, match="unit"):
        check_object_detection_array(msg)


def test_a_new_stream_id_is_a_canonical_uuid_and_differs_each_time():
    first, second = new_stream_id(), new_stream_id()
    assert first != second
    assert str(uuid.UUID(first)) == first


def _assert_cloud_layout(cloud: PointCloud2, width: int) -> None:
    assert [(f.name, f.offset, f.datatype, f.count) for f in cloud.fields] == [
        ("x", 0, PointField.FLOAT32, 1),
        ("y", 4, PointField.FLOAT32, 1),
        ("z", 8, PointField.FLOAT32, 1),
    ]
    assert (cloud.height, cloud.width) == (1, width)
    assert (cloud.point_step, cloud.row_step) == (12, 12 * width)
    assert cloud.is_bigendian is False and cloud.is_dense is True
    assert len(cloud.data) == 12 * width


def test_a_point_cloud_has_the_contract_layout_and_little_endian_values():
    points = _cloud_points(MIN_CLOUD_POINTS)
    cloud = to_point_cloud(_header(), points)
    _assert_cloud_layout(cloud, MIN_CLOUD_POINTS)
    assert cloud.header == _header()
    np.testing.assert_array_equal(np.frombuffer(bytes(cloud.data), dtype="<f4").reshape(-1, 3), points.astype("<f4"))


def test_an_empty_cloud_keeps_the_layout_and_the_header():
    cloud = empty_cloud(_header())
    _assert_cloud_layout(cloud, 0)
    assert cloud.header == _header()


@pytest.mark.parametrize("points", [np.zeros((0, 3)), [], _cloud_points(MIN_CLOUD_POINTS - 1)], ids=["none", "list", "29"])
def test_too_few_points_give_an_empty_cloud(points):
    cloud = to_point_cloud(_header(), points)
    _assert_cloud_layout(cloud, 0)
    assert cloud == empty_cloud(_header())


def test_the_cap_picks_evenly_spaced_points_in_input_order():
    count = 5000
    points = _cloud_points(count)
    cloud = to_point_cloud(_header(), points)
    _assert_cloud_layout(cloud, MAX_CLOUD_POINTS)
    xyz = np.frombuffer(bytes(cloud.data), dtype="<f4").reshape(-1, 3)
    # Index i of the output is floor(i * 5000 / 2000).
    np.testing.assert_array_equal(xyz, points[[(i * count) // MAX_CLOUD_POINTS for i in range(MAX_CLOUD_POINTS)]].astype("<f4"))
    assert to_point_cloud(_header(), points) == cloud


def test_exactly_the_cap_keeps_every_point():
    cloud = to_point_cloud(_header(), _cloud_points(MAX_CLOUD_POINTS))
    assert cloud.width == MAX_CLOUD_POINTS
    np.testing.assert_array_equal(
        np.frombuffer(bytes(cloud.data), dtype="<f4").reshape(-1, 3), _cloud_points(MAX_CLOUD_POINTS).astype("<f4")
    )


@pytest.mark.parametrize(
    "points",
    [
        np.zeros((40, 2)),
        np.zeros(120),
        np.full((40, 3), _NAN),
        np.full((40, 3), _INF),
        np.zeros((40, 3)),
        np.column_stack([np.zeros(40), np.zeros(40), -np.ones(40)]),
    ],
    ids=["two-columns", "flat", "nan", "inf", "at-the-camera", "behind-the-camera"],
)
def test_a_point_cloud_refuses_bad_points(points):
    with pytest.raises(ValueError):
        to_point_cloud(_header(), points)


def test_an_object_with_no_optional_parts_has_valid_empty_values():
    det = to_object_detection(Detection(box=np.array([0.0, 0.0, 5.0, 5.0]), label="mug", score=0.5))
    msg = to_object_detection_array(_header(), _camera_info(), ["mug"], [det], stream_id=STREAM_ID)
    back = deserialize_message(serialize_message(msg), ObjectDetectionArray)
    check_object_detection_array(back)
    (out,) = back.detections
    assert out.embedding_model == "" and len(out.embedding) == 0
    _assert_cloud_layout(out.points, 0)
    assert out.points.header == back.header
    assert len(out.crop.data) == 0 and out.crop.header == back.header
    assert out.view_quality == 0.0 and len(out.color_shares) == 0


def test_a_full_object_array_round_trip():
    msg = _full_array()
    back = deserialize_message(serialize_message(msg), ObjectDetectionArray)
    check_object_detection_array(back)
    assert back == msg
    assert back.stream_id == STREAM_ID
    (out,) = back.detections
    assert out.points.header == back.header == out.crop.header == back.camera_info.header
    _assert_cloud_layout(out.points, MIN_CLOUD_POINTS)
    assert bytes(out.crop.data) == JPEG and out.crop.format == "jpeg"
    assert np.linalg.norm(np.asarray(out.embedding)) == pytest.approx(1.0)
    assert out.embedding_model == CLIP_VIT_B16
    assert out.view_quality == pytest.approx(0.5)
    assert sum(out.color_shares) == pytest.approx(1.0)
    assert out.color_shares[COLOR_NAMES.index("red")] == pytest.approx(1.0)


def test_an_array_with_no_detections_round_trip():
    msg = to_object_detection_array(_header(), _camera_info(), ["mug"], [], stream_id=STREAM_ID)
    back = deserialize_message(serialize_message(msg), ObjectDetectionArray)
    check_object_detection_array(back)
    assert back == msg and list(back.detections) == []


def test_detect_objects_round_trip():
    request = DetectObjects.Request(classes=["mug", "chair"], max_image_age_s=0.25, timeout_s=3.0)
    assert deserialize_message(serialize_message(request), DetectObjects.Request) == request
    response = DetectObjects.Response(success=True, objects=_full_array())
    response.objects.detections[0].track_id = 0
    back = deserialize_message(serialize_message(response), DetectObjects.Response)
    assert back == response
    check_object_detection_array(back.objects)
    failed = DetectObjects.Response(error_code="BUSY", message="another request is running")
    assert deserialize_message(serialize_message(failed), DetectObjects.Response) == failed


def test_embed_crop_round_trip():
    request = EmbedCrop.Request(header=_header(), image_width=2208, image_height=1242, bbox=box_to_bbox([1.0, 2.0, 3.0, 4.0]))
    assert deserialize_message(serialize_message(request), EmbedCrop.Request) == request
    response = EmbedCrop.Response(
        success=True,
        header=_header(),
        crop=CompressedImage(header=_header(), format="jpeg", data=JPEG),
        embedding_model=CLIP_VIT_B16,
        embedding=array.array("f", np.eye(512, dtype=np.float32)[0]),
        view_quality=0.25,
    )
    assert deserialize_message(serialize_message(response), EmbedCrop.Response) == response
    failed = EmbedCrop.Response(error_code="STALE_IMAGE", message="the image has gone")
    assert deserialize_message(serialize_message(failed), EmbedCrop.Response) == failed


def test_describe_crop_round_trip():
    request = DescribeCrop.Request(
        crop=CompressedImage(header=_header(), format="jpeg", data=JPEG),
        color_shares=array.array("f", np.eye(len(COLOR_NAMES), dtype=np.float32)[3]),
    )
    assert deserialize_message(serialize_message(request), DescribeCrop.Request) == request
    disabled = DescribeCrop.Response(error_code="DISABLED", message="captions are off")
    assert deserialize_message(serialize_message(disabled), DescribeCrop.Response) == disabled
    filled = DescribeCrop.Response(
        success=True,
        caption="a red mug",
        attributes_json='{"colour": "red"}',
        printed_text="HOT",
        caption_model="some-weights",
        prompt_version="1",
    )
    assert deserialize_message(serialize_message(filled), DescribeCrop.Response) == filled


def test_fake_pipeline_serialises_and_deserialises():
    detector = FakeObjectDetector(
        [Detection(box=np.array([2000.0, 1000.0, 2400.0, 1300.0]), label="mug", score=0.8, track_id=1)],
        classes=["mug", "bottle"],
    )
    embedder = FakeEmbedder(model_name=CLIP_VIT_B16)
    found = detector.detect(IMAGE)
    crops = [IMAGE[int(d.box[1]) : int(d.box[3]), int(d.box[0]) : int(d.box[2])] for d in found]
    vectors = embedder.embed_images(crops)
    msg = to_object_detection_array(
        _header(),
        _camera_info(),
        detector.classes,
        [to_object_detection(d, v, embedder.model_name) for d, v in zip(found, vectors)],
        stream_id=STREAM_ID,
    )
    back = deserialize_message(serialize_message(msg), ObjectDetectionArray)
    assert list(back.classes) == ["mug", "bottle"]
    (det,) = back.detections
    # Clipped to the image's right and bottom edges.
    np.testing.assert_allclose(bbox_to_box(det.bbox), [2000, 1000, 2208, 1242])
    np.testing.assert_allclose(np.asarray(det.embedding), vectors[0], rtol=1e-6)
    assert back.camera_info.k[2] == 1104.0


def test_person_detection_with_face_and_appearance():
    face = Face(
        bbox=np.array([450.0, 90.0, 540.0, 200.0]),
        kps=np.zeros((5, 2)),
        det_score=0.82,
        embedding=np.full(512, 3.0, dtype=np.float32),
    )
    pose = _pose()
    msg = to_person_detection(
        pose, face=face, face_model="arcface_w600k_r50", appearance=np.ones(512), appearance_model="osnet_x1_0_msmt17"
    )
    assert msg.track_id == 2
    assert msg.keypoints.dtype == np.float32 and msg.keypoints.shape == (34,)
    # x, y of keypoint k are at 2k and 2k + 1.
    assert (msg.keypoints[2 * 5], msg.keypoints[2 * 5 + 1]) == (pose.keypoints[5, 0], pose.keypoints[5, 1])
    np.testing.assert_allclose(msg.keypoint_scores, pose.keypoint_scores, rtol=1e-6)
    assert msg.face_det_score == pytest.approx(0.82)
    assert msg.face_model == "arcface_w600k_r50"
    assert np.isclose(np.linalg.norm(np.asarray(msg.face_embedding)), 1.0)
    assert msg.appearance_model == "osnet_x1_0_msmt17" and len(msg.appearance_embedding) == 512
    back = deserialize_message(
        serialize_message(to_person_detection_array(_header(), _camera_info(), [msg])), PersonDetectionArray
    )
    np.testing.assert_allclose(bbox_to_box(back.people[0].face_bbox), face.bbox)


def test_person_detection_without_face():
    msg = to_person_detection(_pose(track_id=0))
    assert msg.face_det_score == 0.0
    assert msg.face_model == "" and len(msg.face_embedding) == 0
    assert msg.appearance_model == "" and len(msg.appearance_embedding) == 0
    assert msg.track_id == 0


def test_person_detection_with_a_face_but_no_embedding():
    face = Face(bbox=np.array([450.0, 90.0, 540.0, 200.0]), kps=np.zeros((5, 2)), det_score=0.6)
    msg = to_person_detection(_pose(), face=face)
    assert msg.face_det_score == pytest.approx(0.6)
    assert msg.face_model == "" and len(msg.face_embedding) == 0


def test_person_detection_rejects_bad_input():
    pose = _pose()
    pose.keypoints = pose.keypoints[:16]
    with pytest.raises(ValueError, match="17 COCO keypoints"):
        to_person_detection(pose)
    with pytest.raises(ValueError):
        to_person_detection(_pose(), face_model="arcface_w600k_r50")


def test_empty_person_array():
    msg = to_person_detection_array(_header(), _camera_info(), [])
    assert msg.people == [] and msg.header.frame_id == "camera_optical_frame"


def test_a_published_person_back_to_a_pose():
    pose = _pose(track_id=7)
    back = pose_from_person_detection(to_person_detection(pose))
    np.testing.assert_allclose(back.box, pose.box)
    np.testing.assert_allclose(back.keypoints, pose.keypoints)
    np.testing.assert_allclose(back.keypoint_scores, pose.keypoint_scores, rtol=1e-6)
    assert back.track_id == 7 and back.score == pytest.approx(0.93)


def _crop_scene():
    """A 2208 x 1242 image with the guest, track 7, and a person behind, track 8."""
    image = np.full((1242, 2208, 3), 60, dtype=np.uint8)
    guest = figure_pose((800.0, 200.0, 1100.0, 1000.0), track_id=7)
    behind = figure_pose((1050.0, 150.0, 1300.0, 900.0), track_id=8)
    paint_figure(image, guest, (0, 0, 255), (255, 0, 0))
    people = to_person_detection_array(_header(), _camera_info(), [to_person_detection(p) for p in (behind, guest)])
    return image, people


def test_person_crop_round_trip():
    image, people = _crop_scene()
    poses = [pose_from_person_detection(p) for p in people.people]
    found = crop_person(image, poses, 7)
    response = fill_person_crop_response(GetPersonCrop.Response(), people, found)
    back = deserialize_message(serialize_message(response), GetPersonCrop.Response)
    assert back.status == GetPersonCrop.Response.OK and back.message == ""
    assert back.header == people.header and back.crop.header == people.header
    assert back.person.track_id == 7
    assert back.crop.format == "jpeg" and bytes(back.crop.data) == found.jpeg
    assert max(found.pixels.shape[:2]) == 512
    np.testing.assert_allclose(np.asarray(back.crop_keypoints).reshape(17, 2), found.keypoints, rtol=1e-5)
    masked = np.asarray(back.masked_boxes).reshape(-1, 4)
    assert len(masked) == 2
    np.testing.assert_allclose(masked, found.masked, rtol=1e-5)
    assert list(back.color_names) == list(COLOR_NAMES)
    assert len(back.top_shares) == len(back.bottom_shares) == len(COLOR_NAMES)
    assert back.top_shares[COLOR_NAMES.index("red")] == pytest.approx(1.0)
    assert back.bottom_shares[COLOR_NAMES.index("blue")] == pytest.approx(1.0)
    assert back.top_left == pytest.approx(1.0) and back.bottom_left == pytest.approx(1.0)


def test_a_gated_region_gives_empty_shares_in_the_response():
    image, people = _crop_scene()
    poses = [pose_from_person_detection(p) for p in people.people]
    poses[1].keypoint_scores[13] = 0.1  # the guest's left knee
    found = crop_person(image, poses, 7)
    response = fill_person_crop_response(GetPersonCrop.Response(), people, found)
    assert response.status == GetPersonCrop.Response.OK
    assert len(response.top_shares) == len(COLOR_NAMES)
    assert len(response.bottom_shares) == 0 and response.bottom_left == 0.0


def test_an_occluded_person_crop_gives_the_person_and_the_overlaps():
    image, people = _crop_scene()
    # Its box ends 100 px lower than the guest's: nearer.
    nearer = to_person_detection(figure_pose((850.0, 300.0, 1150.0, 1100.0), track_id=9))
    people.people.append(nearer)
    found = crop_person(image, [pose_from_person_detection(p) for p in people.people], 7)
    response = fill_person_crop_response(GetPersonCrop.Response(), people, found)
    assert response.status == GetPersonCrop.Response.OCCLUDED and "nearer" in response.message
    assert response.header == people.header and response.person.track_id == 7
    assert response.max_overlap == pytest.approx(found.max_overlap) and response.max_overlap > 0.3
    assert response.torso_cover == pytest.approx(found.torso_cover)
    assert len(response.crop.data) == 0 and len(response.masked_boxes) == 0
    assert len(response.color_names) == 0 and len(response.top_shares) == 0


@pytest.mark.parametrize(
    ("track_id", "status"),
    [(9, GetPersonCrop.Response.NOT_IN_VIEW), (8, GetPersonCrop.Response.TOO_SMALL)],
)
def test_a_person_crop_without_a_crop(track_id, status, monkeypatch):
    image, people = _crop_scene()
    # Track 8 is 750 px tall: too small only when the floor is above that.
    monkeypatch.setattr(crop, "MIN_BOX_HEIGHT_PX", 800.0)
    found = crop_person(image, [pose_from_person_detection(p) for p in people.people], track_id)
    response = fill_person_crop_response(GetPersonCrop.Response(), people, found)
    assert response.status == status and response.message
    assert response.header == people.header
    assert response.person.track_id == (0 if status == GetPersonCrop.Response.NOT_IN_VIEW else 8)
    assert len(response.crop.data) == 0 and len(response.crop_keypoints) == 0
    assert len(response.color_names) == 0 and len(response.top_shares) == 0


def test_embed_text_round_trip():
    embedder = FakeEmbedder(model_name=CLIP_VIT_B16, dim=8)
    texts = ["a red mug", "a chair"]
    response = fill_embed_text_response(EmbedText.Response(), embedder, texts)
    assert response.success and response.message == ""
    assert (response.embedding_model, response.dim) == (CLIP_VIT_B16, 8)
    back = deserialize_message(serialize_message(response), EmbedText.Response)
    vectors = vectors_from_response(back)
    assert vectors.shape == (2, 8)
    np.testing.assert_allclose(vectors, embedder.embed_texts(texts), rtol=1e-6)


class _BrokenEmbedder(FakeEmbedder):
    def __init__(self, result=None, error=None):
        super().__init__(dim=8)
        self.result, self.error = result, error

    def embed_texts(self, texts):
        if self.error:
            raise self.error
        return self.result


@pytest.mark.parametrize(
    ("embedder", "texts", "reason"),
    [
        (FakeEmbedder(dim=8), [], "texts is empty"),
        (_BrokenEmbedder(error=RuntimeError("CUDA out of memory")), ["mug"], "RuntimeError: CUDA out of memory"),
        (_BrokenEmbedder(result=np.ones((2, 8))), ["mug"], "expected shape (1, 8)"),
        (_BrokenEmbedder(result=np.zeros((1, 8))), ["mug"], "zero"),
    ],
)
def test_embed_text_failures_answer_instead_of_raising(embedder, texts, reason):
    response = fill_embed_text_response(EmbedText.Response(), embedder, texts)
    assert not response.success
    assert reason in response.message
    assert response.embedding_model == "" and len(response.vectors) == 0
    with pytest.raises(ValueError, match="EmbedText failed"):
        vectors_from_response(response)
