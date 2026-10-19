import array

import numpy as np
import pytest
from rclpy.serialization import deserialize_message, serialize_message
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Header
from walkie_vision_msgs.msg import ObjectDetection, ObjectDetectionArray, PersonDetection, PersonDetectionArray
from walkie_vision_msgs.srv import EmbedText

from walkie_vision.convert import (
    bbox_to_box,
    box_to_bbox,
    fill_embed_text_response,
    to_object_detection,
    to_object_detection_array,
    to_person_detection,
    to_person_detection_array,
    unit_vector,
    vectors_from_response,
)
from walkie_vision.fakes import FakeEmbedder, FakeObjectDetector
from walkie_vision.providers import CLIP_VIT_B16, Detection, Face, PersonPose

# HD2K, the ZED 2i's resolution on the robot.
IMAGE = np.zeros((1242, 2208, 3), dtype=np.uint8)


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
            },
        ),
        (
            ObjectDetectionArray,
            {
                "header": "std_msgs/Header",
                "camera_info": "sensor_msgs/CameraInfo",
                "classes": "sequence<string>",
                "detections": "sequence<walkie_vision_msgs/ObjectDetection>",
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
    ],
)
def test_message_fields(msg_type, fields):
    assert msg_type.get_fields_and_field_types() == fields


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
    msg = to_object_detection(det, embedding=np.array([0.0, 2.0, 0.0]), embedding_model=CLIP_VIT_B16, crop=crop)
    assert (msg.label, msg.track_id, msg.embedding_model) == ("mug", 7, "clip_vit_b16")
    assert msg.score == pytest.approx(0.75)
    assert isinstance(msg.embedding, array.array) and msg.embedding.typecode == "f"
    assert list(msg.embedding) == [0.0, 1.0, 0.0]
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
    msg = to_object_detection_array(_header(), _camera_info(), ["mug", "chair"], [])
    assert list(msg.classes) == ["mug", "chair"]
    assert msg.detections == []
    assert msg.header.frame_id == "camera_optical_frame"
    assert msg.header.stamp.nanosec == 250_000_000
    assert (msg.camera_info.width, msg.camera_info.height) == (2208, 1242)


def test_object_array_rejects_a_label_that_is_not_a_class():
    det = to_object_detection(Detection(box=np.array([0.0, 0.0, 5.0, 5.0]), label="mug", score=0.5))
    with pytest.raises(ValueError, match="mug"):
        to_object_detection_array(_header(), _camera_info(), ["chair"], [det])


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
