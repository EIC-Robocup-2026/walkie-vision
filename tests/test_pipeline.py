"""VisionPipeline with the fakes: one image to its two arrays. No ROS graph.

The LatencyLog tests are here too: it is the node's, but needs no node.
"""

import dataclasses
import threading

import numpy as np
import pytest
from builtin_interfaces.msg import Time
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header

from walkie_vision import embeddings as embeddings_module
from walkie_vision import pipeline as pipeline_module
from walkie_vision.depth import RegisteredDepth, frame_laplacian, view_quality
from walkie_vision.fakes import (
    FakeEmbedder,
    FakeObjectDetector,
    FakePersonFaceRecognizer,
    FakePoseEstimator,
    FakeReIdentifier,
    box_mask,
)
from walkie_vision.node import LatencyLog
from walkie_vision.pipeline import Providers, TrackIdBase, VisionPipeline, dedupe_poses
from walkie_vision.providers import Detection, Face, PersonPose

CLASSES = ["chair", "sofa"]
IMAGE = np.random.default_rng(0).integers(0, 256, (48, 64, 3), dtype=np.uint8)
FRAME = "zed_head_left_camera_frame_optical"


def _header(sec: int = 12, nanosec: int = 345) -> Header:
    return Header(stamp=Time(sec=sec, nanosec=nanosec), frame_id=FRAME)


def _info() -> CameraInfo:
    info = CameraInfo()
    info.header.frame_id = "something_else"
    info.width, info.height = 64, 48
    info.k = [50.0, 0.0, 32.0, 0.0, 50.0, 24.0, 0.0, 0.0, 1.0]
    return info


def _pose(box, score=0.9, track_id=3) -> PersonPose:
    return PersonPose(
        box=np.array(box, dtype=np.float64),
        score=score,
        keypoints=np.full((17, 2), 10.0, dtype=np.float32),
        keypoint_scores=np.full(17, 0.8, dtype=np.float32),
        track_id=track_id,
    )


def _face(box) -> Face:
    return Face(bbox=np.array(box, dtype=np.float64), kps=np.zeros((5, 2)), det_score=0.8)


def _detection(box=(1.0, 2.0, 30.0, 40.0), label="chair", score=0.8, track_id=5) -> Detection:
    return Detection(box=np.array(box, dtype=np.float64), label=label, score=score, track_id=track_id)


BIG = (2.0, 2.0, 40.0, 46.0)  # 38 x 44 px
SMALL = (50.0, 2.0, 60.0, 12.0)  # 10 x 10 px


class _CountingEmbedder(FakeEmbedder):
    """Counts the images it embeds."""

    def __init__(self) -> None:
        super().__init__()
        self.images = 0

    def embed_images(self, images):
        self.images += len(images)
        return super().embed_images(images)


def _raises_if_called(*args, **kwargs):
    raise AssertionError("this work must not run")


def _providers(objects=None, people=None, faces=None, *, classes=CLASSES, reid=None, face_model=None):
    return Providers(
        objects=objects if objects is not None else FakeObjectDetector([_detection()], classes=classes),
        pose=people if people is not None else FakePoseEstimator([_pose(BIG), _pose(SMALL, track_id=0)]),
        faces=face_model if face_model is not None else FakePersonFaceRecognizer(faces or [_face((10.0, 5.0, 20.0, 15.0))]),
        reid=reid if reid is not None else FakeReIdentifier(),
    )


def _pipeline(providers=None, *, start_ns=0, **kwargs) -> VisionPipeline:
    return VisionPipeline(providers or _providers(), CLASSES, TrackIdBase(start_ns=start_ns), **kwargs)


def test_both_arrays_carry_the_image_header_and_its_camera_info():
    result = _pipeline().process(IMAGE, _header(), _info())
    for array in (result.people, result.objects):
        assert array.header.stamp.sec == 12 and array.header.stamp.nanosec == 345
        assert array.header.frame_id == FRAME
        assert array.camera_info.header.frame_id == FRAME
        assert array.camera_info.header.stamp.sec == 12
        assert list(array.camera_info.k) == list(_info().k)
        assert (array.camera_info.width, array.camera_info.height) == (64, 48)
    assert list(result.objects.classes) == CLASSES
    assert result.dropped == {}


def test_the_input_camera_info_is_not_changed():
    info = _info()
    _pipeline().process(IMAGE, _header(), info)
    assert info.header.frame_id == "something_else"


def test_classes_are_the_configured_list():
    assert _pipeline().classes == CLASSES


@pytest.mark.parametrize("classes", [[], ["chair", "chair"], [" chair"], [""]])
def test_bad_classes_raise(classes):
    with pytest.raises(ValueError):
        VisionPipeline(_providers(), classes, TrackIdBase(start_ns=0))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"people_min_score": 1.5},
        {"objects_min_score": -0.1},
        {"reid_min_crop_px": 0},
        {"people_dedupe_iou": 0.0},
        {"people_dedupe_iou": 1.5},
        {"depth_min_m": 0.0},
        {"depth_min_m": float("nan")},
        {"depth_min_m": 2.0, "depth_max_m": 1.0},
        {"depth_max_m": float("inf")},
    ],
)
def test_bad_settings_raise(kwargs):
    with pytest.raises(ValueError):
        _pipeline(**kwargs)


def test_track_ids_go_through_the_base_and_zero_stays_zero():
    result = _pipeline().process(IMAGE, _header(), _info())
    assert [p.track_id for p in result.people.people] == [3, 0]
    assert [d.track_id for d in result.objects.detections] == [5]
    base = TrackIdBase(start_ns=1_793_000_000_000 * 1_000_000)
    result = VisionPipeline(_providers(), CLASSES, base).process(IMAGE, _header(), _info())
    assert [p.track_id for p in result.people.people] == [base.map(3), 0]
    assert [d.track_id for d in result.objects.detections] == [base.map(5)]


def test_faces_from_the_fake_go_into_their_person():
    result = _pipeline().process(IMAGE, _header(), _info())
    with_face, without = result.people.people
    assert with_face.face_det_score > 0
    assert with_face.face_model == "fake_face"
    vector = np.asarray(with_face.face_embedding)
    assert vector.shape == (512,) and np.isclose(np.linalg.norm(vector), 1.0)
    bbox = with_face.face_bbox
    assert (bbox.center.position.x, bbox.center.position.y, bbox.size_x, bbox.size_y) == (15.0, 10.0, 10.0, 10.0)
    assert without.face_det_score == 0 and without.face_model == "" and list(without.face_embedding) == []


def test_appearance_only_for_people_at_least_reid_min_crop_px():
    result = _pipeline(reid_min_crop_px=16).process(IMAGE, _header(), _info())
    big, small = result.people.people
    assert big.appearance_model == "fake_reid" and len(big.appearance_embedding) == 512
    assert small.appearance_model == "" and list(small.appearance_embedding) == []
    result = _pipeline(reid_min_crop_px=10).process(IMAGE, _header(), _info())
    assert result.people.people[1].appearance_model == "fake_reid"


class _Recording:
    """Wraps a fake and records what it was called with."""

    def __init__(self, inner):
        self.inner = inner
        self.model_name = inner.model_name
        self.dim = inner.dim
        self.calls = []

    def faces_for(self, image, people):
        self.calls.append(list(people))
        return self.inner.faces_for(image, people)

    def embed(self, crops):
        self.calls.append(list(crops))
        return self.inner.embed(crops)


def test_score_floors_leave_out_people_and_objects_before_faces_and_reid():
    faces = _Recording(FakePersonFaceRecognizer([_face((10.0, 5.0, 20.0, 15.0))]))
    reid = _Recording(FakeReIdentifier())
    providers = _providers(
        objects=FakeObjectDetector([_detection(score=0.2, track_id=1), _detection(score=0.25, track_id=2)], classes=CLASSES),
        people=FakePoseEstimator([_pose(BIG, score=0.2, track_id=1), _pose(BIG, score=0.25, track_id=2)]),
        face_model=faces,
        reid=reid,
    )
    result = _pipeline(providers, people_min_score=0.25, objects_min_score=0.25).process(IMAGE, _header(), _info())
    assert [p.track_id for p in result.people.people] == [2]
    assert [d.track_id for d in result.objects.detections] == [2]
    assert [[p.track_id for p in call] for call in faces.calls] == [[2]]
    assert [len(call) for call in reid.calls] == [1]
    # Nothing bad was in the frame: a floor is not a drop.
    assert result.dropped == {}


# BIG is 38 x 44 px. GHOST covers its top 31 px: IoU 31/44 = 0.70.
GHOST = (2.0, 2.0, 40.0, 33.0)
# Beside BIG, overlapping it by 10 px: IoU 440 / 2640 = 0.17.
BESIDE = (30.0, 2.0, 62.0, 46.0)


@pytest.mark.parametrize(
    "boxes_scores, max_iou, kept",
    [
        # A partial box inside a higher-scoring one is left out, whichever comes first.
        ([(BIG, 0.9), (GHOST, 0.4)], 0.5, [0]),
        ([(GHOST, 0.4), (BIG, 0.9)], 0.5, [1]),
        # The higher score wins, also when it is the partial box.
        ([(BIG, 0.4), (GHOST, 0.9)], 0.5, [1]),
        # Two people side by side are both kept, in their input order.
        ([(BESIDE, 0.5), (BIG, 0.9)], 0.5, [0, 1]),
        # The threshold is exclusive: IoU 0.70 is kept at 0.75, and dropped at 0.65.
        ([(BIG, 0.9), (GHOST, 0.4)], 0.75, [0, 1]),
        ([(BIG, 0.9), (GHOST, 0.4)], 0.65, [0]),
        # On equal scores the earlier pose is kept.
        ([(BIG, 0.9), (BIG, 0.9)], 0.5, [0]),
        # 1.0 keeps every pose, even two equal boxes.
        ([(BIG, 0.9), (BIG, 0.8)], 1.0, [0, 1]),
        # A pose dropped as a duplicate does not drop a third one.
        ([(BIG, 0.9), (GHOST, 0.8), ((2.0, 2.0, 40.0, 20.0), 0.7)], 0.5, [0, 2]),
        ([], 0.5, []),
    ],
)
def test_dedupe_poses(boxes_scores, max_iou, kept):
    poses = [_pose(box, score=score, track_id=i + 1) for i, (box, score) in enumerate(boxes_scores)]
    assert [p.track_id for p in dedupe_poses(poses, max_iou)] == [i + 1 for i in kept]


def test_a_second_box_of_one_person_is_left_out_before_faces_and_reid():
    # The face is inside both boxes. The fake gives it to the first person
    # that contains it, so without the dedupe the ghost would take it.
    faces = _Recording(FakePersonFaceRecognizer([_face((10.0, 5.0, 20.0, 15.0))]))
    reid = _Recording(FakeReIdentifier())
    providers = _providers(
        people=FakePoseEstimator([_pose(GHOST, score=0.4, track_id=9), _pose(BIG, score=0.9, track_id=1)]),
        face_model=faces,
        reid=reid,
    )
    result = _pipeline(providers, people_dedupe_iou=0.5).process(IMAGE, _header(), _info())
    assert [p.track_id for p in result.people.people] == [1]
    assert result.people.people[0].face_det_score > 0
    assert [[p.track_id for p in call] for call in faces.calls] == [[1]]
    assert [len(call) for call in reid.calls] == [1]
    assert result.duplicates == 1
    # A second box is not a bad field.
    assert result.dropped == {}


def test_the_dedupe_is_off_by_default_and_the_floor_comes_first():
    people = FakePoseEstimator([_pose(BIG, score=0.9, track_id=1), _pose(GHOST, score=0.2, track_id=9)])
    result = _pipeline(_providers(people=people)).process(IMAGE, _header(), _info())
    assert [p.track_id for p in result.people.people] == [1, 9] and result.duplicates == 0
    # Under the floor, the ghost is not published and is not counted as a duplicate.
    result = _pipeline(_providers(people=people), people_min_score=0.25, people_dedupe_iou=0.5).process(
        IMAGE, _header(), _info()
    )
    assert [p.track_id for p in result.people.people] == [1] and result.duplicates == 0


def test_no_people_calls_neither_faces_nor_reid():
    faces = _Recording(FakePersonFaceRecognizer([]))
    reid = _Recording(FakeReIdentifier())
    providers = _providers(people=FakePoseEstimator([]), face_model=faces, reid=reid)
    result = _pipeline(providers).process(IMAGE, _header(), _info())
    assert list(result.people.people) == [] and faces.calls == [] and reid.calls == []
    assert len(result.objects.detections) == 1


class _Faces:
    """Returns fixed faces for the first people, in order."""

    model_name = "fake_face"
    dim = 512

    def __init__(self, faces):
        self.faces = faces

    def faces_for(self, image, people):
        return [self.faces[i] if i < len(self.faces) else None for i in range(len(people))]


def _embedded(box, embedding) -> Face:
    return Face(bbox=np.array(box, dtype=np.float64), kps=np.zeros((5, 2)), det_score=0.8, embedding=embedding)


@pytest.mark.parametrize(
    "bad",
    [
        _embedded((10.0, 5.0, 20.0, 15.0), np.full(512, np.nan, dtype=np.float32)),
        _embedded((10.0, 5.0, 20.0, 15.0), np.zeros(512, dtype=np.float32)),
        _embedded((10.0, 5.0, 20.0, 15.0), None),
        _embedded((10.0, 5.0, 20.0, 15.0), np.ones((2, 256), dtype=np.float32)),
        _embedded((20.0, 5.0, 10.0, 15.0), np.ones(512, dtype=np.float32)),
        _embedded((np.nan, 5.0, 20.0, 15.0), np.ones(512, dtype=np.float32)),
    ],
)
def test_a_bad_face_becomes_no_face_and_the_rest_is_published(bad):
    good = _embedded((25.0, 5.0, 35.0, 15.0), np.ones(512, dtype=np.float32))
    providers = _providers(people=FakePoseEstimator([_pose(BIG), _pose(BIG, track_id=4)]), face_model=_Faces([bad, good]))
    result = _pipeline(providers).process(IMAGE, _header(), _info())
    first, second = result.people.people
    assert first.face_det_score == 0 and first.face_model == "" and list(first.face_embedding) == []
    assert second.face_model == "fake_face" and len(second.face_embedding) == 512
    assert first.appearance_model == "fake_reid"
    assert result.dropped == {"face": 1}
    assert len(result.objects.detections) == 1


def test_a_face_with_a_score_that_is_not_finite_becomes_no_face():
    bad = _embedded((10.0, 5.0, 20.0, 15.0), np.ones(512, dtype=np.float32))
    bad.det_score = float("inf")
    providers = _providers(people=FakePoseEstimator([_pose(BIG)]), face_model=_Faces([bad]))
    result = _pipeline(providers).process(IMAGE, _header(), _info())
    assert result.people.people[0].face_det_score == 0 and result.dropped == {"face": 1}


class _Reid:
    """Returns fixed rows."""

    model_name = "fake_reid"
    dim = 512

    def __init__(self, rows):
        self.rows = rows

    def embed(self, crops):
        return self.rows[: len(crops)]


@pytest.mark.parametrize("bad_row", [np.full(512, np.nan, dtype=np.float32), np.zeros(512, dtype=np.float32)])
def test_a_bad_appearance_row_becomes_no_appearance(bad_row):
    rows = np.stack([bad_row, np.ones(512, dtype=np.float32)])
    providers = _providers(people=FakePoseEstimator([_pose(BIG), _pose(BIG, track_id=4)]), reid=_Reid(rows))
    result = _pipeline(providers).process(IMAGE, _header(), _info())
    first, second = result.people.people
    assert first.appearance_model == "" and list(first.appearance_embedding) == []
    assert second.appearance_model == "fake_reid"
    assert result.dropped == {"appearance": 1}


def test_a_pose_with_a_bad_value_is_left_out():
    bad = _pose(BIG, track_id=7)
    bad.keypoints[4, 1] = np.nan
    inverted = _pose((40.0, 2.0, 2.0, 46.0), track_id=8)
    bad_scores = _pose(BIG, track_id=9)
    bad_scores.keypoint_scores[0] = np.inf
    providers = _providers(people=FakePoseEstimator([bad, _pose(BIG, track_id=1), inverted, bad_scores]))
    result = _pipeline(providers).process(IMAGE, _header(), _info())
    assert [p.track_id for p in result.people.people] == [1]
    assert result.dropped == {"person": 3}
    assert len(result.objects.detections) == 1


class _Detector:
    """Returns fixed detections without clipping them."""

    def __init__(self, detections, classes=CLASSES):
        self.detections = detections
        self._classes = list(classes)
        self.set_calls = []

    @property
    def classes(self):
        return list(self._classes)

    def set_classes(self, classes):
        self.set_calls.append(list(classes))
        self._classes = list(classes)

    def detect(self, image):
        return list(self.detections)


def test_a_detection_with_a_bad_box_or_score_is_left_out():
    detector = _Detector(
        [
            _detection(box=(30.0, 2.0, 1.0, 40.0), track_id=1),
            _detection(box=(1.0, np.nan, 30.0, 40.0), track_id=2),
            _detection(score=float("nan"), track_id=3),
            _detection(track_id=4),
        ]
    )
    result = _pipeline(_providers(objects=detector)).process(IMAGE, _header(), _info())
    assert [d.track_id for d in result.objects.detections] == [4]
    assert result.dropped == {"object": 3}
    assert len(result.people.people) == 2


@pytest.mark.parametrize("score", [1.5, -0.1, float("inf")])
def test_a_detection_with_a_score_outside_the_range_is_left_out(score):
    detector = _Detector([_detection(score=score, track_id=1), _detection(track_id=2)])
    result = _pipeline(_providers(objects=detector)).process(IMAGE, _header(), _info())
    assert [d.track_id for d in result.objects.detections] == [2]
    assert result.dropped == {"object": 1}


def test_a_pipeline_keeps_one_stream_id_and_a_new_pipeline_gets_another():
    pipeline = _pipeline()
    first = pipeline.process(IMAGE, _header(), _info()).objects
    second = pipeline.process(IMAGE, _header(nanosec=346), _info()).objects
    assert first.stream_id == second.stream_id == pipeline.stream_id
    assert first.stream_id
    assert _pipeline().stream_id != pipeline.stream_id


def test_an_object_array_has_empty_parts_with_the_image_header():
    (det,) = _pipeline().process(IMAGE, _header(), _info()).objects.detections
    assert det.points.header == _header() and det.points.width == 0 and len(det.points.fields) == 3
    assert det.crop.header == _header() and len(det.crop.data) == 0
    assert det.view_quality == 0.0 and len(det.color_shares) == 0


# --- geometry ---------------------------------------------------------------------

IMAGE_H, IMAGE_W = IMAGE.shape[:2]
LEFT_BOX = (4.0, 8.0, 24.0, 28.0)  # 20 x 20 px, on the depth's near half.
RIGHT_BOX = (40.0, 8.0, 60.0, 28.0)  # 20 x 20 px, on the depth's far half.
ERODED_POINTS = 18 * 18


def _masked(box=LEFT_BOX, *, score=0.8, track_id=5, mask=None) -> Detection:
    if mask is None:
        mask = box_mask(IMAGE_H, IMAGE_W, box)
    return Detection(box=np.array(box, dtype=np.float64), label="chair", score=score, track_id=track_id, mask=mask)


def _plain_depth(value=1.0, height=IMAGE_H, width=IMAGE_W, fx=50.0, cx=32.0, cy=24.0) -> RegisteredDepth:
    return RegisteredDepth(np.full((height, width), value, np.float32), fx, fx, cx, cy)


def _geometry_pipeline(*detections, **kwargs) -> VisionPipeline:
    kwargs.setdefault("geometry_enabled", True)
    return _pipeline(_providers(objects=FakeObjectDetector(detections, classes=CLASSES)), **kwargs)


def _xyz(det) -> np.ndarray:
    return np.frombuffer(bytes(det.points.data), dtype="<f4").reshape(-1, 3)


def _forbid(monkeypatch, *names):
    def forbidden(*args, **kwargs):
        raise AssertionError("geometry work ran")

    for name in names:
        monkeypatch.setattr(pipeline_module, name, forbidden)


def test_geometry_off_ignores_a_mask_and_a_depth_and_runs_no_geometry_work(monkeypatch):
    _forbid(monkeypatch, "frame_laplacian", "lift_mask", "depth_cover", "view_quality", "resize_mask")
    result = _geometry_pipeline(_masked(), geometry_enabled=False).process(IMAGE, _header(), _info(), _plain_depth())
    (det,) = result.objects.detections
    assert det.points.width == 0 and len(det.points.data) == 0 and det.view_quality == 0.0


def test_geometry_fills_the_points_and_the_quality_of_a_detection():
    (det,) = _geometry_pipeline(_masked()).process(IMAGE, _header(), _info(), _plain_depth()).objects.detections
    assert det.points.header == _header()
    assert det.points.width == ERODED_POINTS and det.points.height == 1
    assert (_xyz(det)[:, 2] == 1.0).all()
    expected = view_quality(box_mask(IMAGE_H, IMAGE_W, LEFT_BOX), np.array(LEFT_BOX), frame_laplacian(IMAGE), 1.0)
    assert det.view_quality == pytest.approx(expected, abs=1e-6)
    # The new fields of the message stay empty. V5 and V6 fill them.
    assert len(det.embedding) == 0 and len(det.crop.data) == 0 and len(det.color_shares) == 0


def test_the_laplacian_runs_once_per_image(monkeypatch):
    calls = []

    def counted(image):
        calls.append(image.shape)
        return frame_laplacian(image)

    monkeypatch.setattr(pipeline_module, "frame_laplacian", counted)
    pipeline = _geometry_pipeline(_masked(LEFT_BOX), _masked(RIGHT_BOX, track_id=6))
    pipeline.process(IMAGE, _header(), _info(), _plain_depth())
    assert calls == [IMAGE.shape]
    pipeline.process(IMAGE, _header(), _info(), _plain_depth())
    assert len(calls) == 2


def test_no_mask_and_no_detection_run_no_geometry_work(monkeypatch):
    _forbid(monkeypatch, "frame_laplacian", "lift_mask")
    (det,) = _geometry_pipeline(_detection()).process(IMAGE, _header(), _info(), _plain_depth()).objects.detections
    assert det.points.width == 0 and det.view_quality == 0.0
    assert _geometry_pipeline().process(IMAGE, _header(), _info(), _plain_depth()).objects.detections == []


def test_each_detection_gets_the_points_of_its_own_mask_after_the_filters():
    metres = np.full((IMAGE_H, IMAGE_W), 2.0, np.float32)
    metres[:, :32] = 1.0
    depth = RegisteredDepth(metres, 50.0, 50.0, 32.0, 24.0)
    pipeline = _geometry_pipeline(
        # A row left out by the score floor, one by a bad box, and one without a mask.
        _masked(RIGHT_BOX, score=0.1, track_id=1),
        _masked(LEFT_BOX, track_id=2),
        Detection(box=np.array([float("nan"), 0.0, 5.0, 5.0]), label="chair", score=0.8, track_id=3),
        _detection(RIGHT_BOX, track_id=4),
        _masked(RIGHT_BOX, track_id=5),
        objects_min_score=0.25,
    )
    result = pipeline.process(IMAGE, _header(), _info(), depth)
    assert [d.track_id for d in result.objects.detections] == [2, 4, 5]
    left, plain, right = result.objects.detections
    assert (_xyz(left)[:, 2] == 1.0).all() and left.points.width == ERODED_POINTS
    assert plain.points.width == 0 and plain.view_quality == 0.0
    assert (_xyz(right)[:, 2] == 2.0).all() and right.points.width == ERODED_POINTS
    assert (_xyz(left)[:, 0] < 0).all() and (_xyz(right)[:, 0] > 0).all()


def test_without_depth_the_cloud_is_empty_and_the_quality_has_no_cover():
    (det,) = _geometry_pipeline(_masked()).process(IMAGE, _header(), _info(), None).objects.detections
    assert det.points.width == 0 and len(det.points.data) == 0 and det.points.header == _header()
    expected = view_quality(box_mask(IMAGE_H, IMAGE_W, LEFT_BOX), np.array(LEFT_BOX), frame_laplacian(IMAGE), 0.0)
    assert det.view_quality == pytest.approx(expected, abs=1e-6)
    assert det.view_quality > 0.0


def test_a_depth_out_of_range_gives_no_points_and_no_cover():
    # The test's depth is at 5 m, past the 3 m limit.
    (det,) = _geometry_pipeline(_masked()).process(IMAGE, _header(), _info(), _plain_depth(5.0)).objects.detections
    assert det.points.width == 0
    expected = view_quality(box_mask(IMAGE_H, IMAGE_W, LEFT_BOX), np.array(LEFT_BOX), frame_laplacian(IMAGE), 0.0)
    assert det.view_quality == pytest.approx(expected, abs=1e-6)


def test_a_small_mask_gives_no_points_but_keeps_its_quality():
    box = (4.0, 8.0, 9.0, 13.0)  # 25 px, under the point floor.
    (det,) = _geometry_pipeline(_masked(box)).process(IMAGE, _header(), _info(), _plain_depth()).objects.detections
    assert det.points.width == 0
    assert det.view_quality > 0.0


def test_the_configured_range_reaches_the_lift():
    pipeline = _geometry_pipeline(_masked(), depth_min_m=2.0, depth_max_m=2.5)
    (det,) = pipeline.process(IMAGE, _header(), _info(), _plain_depth(1.0)).objects.detections
    assert det.points.width == 0
    (det,) = pipeline.process(IMAGE, _header(), _info(), _plain_depth(2.2)).objects.detections
    assert det.points.width == ERODED_POINTS


def test_a_lower_resolution_raster_uses_its_own_intrinsics():
    # Half the size: fx' = fx / 2 and cx' = (cx + 0.5) / 2 - 0.5. The native box
    # 16..48 lands on raster columns 8..23, and a radius of 1 leaves 9..22.
    box = (16.0, 8.0, 48.0, 40.0)
    depth = RegisteredDepth(np.full((24, 32), 1.0, np.float32), 25.0, 25.0, 15.75, 11.75)
    (det,) = _geometry_pipeline(_masked(box)).process(IMAGE, _header(), _info(), depth).objects.detections
    xyz = _xyz(det)
    assert det.points.width == 14 * 14
    np.testing.assert_allclose(
        [xyz[:, 0].min(), xyz[:, 0].max()], [(9 - 15.75) / 25.0, (22 - 15.75) / 25.0], atol=1e-6
    )
    # The area and the sharpness use the native mask, and the cover the raster mask.
    expected = view_quality(box_mask(IMAGE_H, IMAGE_W, box), np.array(box), frame_laplacian(IMAGE), 1.0)
    assert det.view_quality == pytest.approx(expected, abs=1e-6)


def test_a_mask_of_another_size_fails_the_image():
    mask = np.zeros((IMAGE_H // 2, IMAGE_W // 2), bool)
    with pytest.raises(ValueError, match="mask"):
        _geometry_pipeline(_masked(mask=mask)).process(IMAGE, _header(), _info(), _plain_depth())


class _Raises:
    model_name = "fake_raises"
    dim = 512
    classes = CLASSES

    def set_classes(self, classes):
        pass

    def _raise(self, *args):
        raise RuntimeError("the model failed")

    detect = estimate = faces_for = embed = _raise


@pytest.mark.parametrize("stage", ["objects", "pose", "faces", "reid"])
def test_a_provider_that_raises_makes_process_raise(stage):
    providers = _providers()
    kwargs = {f: getattr(providers, f) for f in ("objects", "pose", "faces", "reid")}
    kwargs[stage] = _Raises()
    with pytest.raises(RuntimeError, match="the model failed"):
        _pipeline(Providers(**kwargs)).process(IMAGE, _header(), _info())


def test_a_face_model_with_the_wrong_length_raises():
    providers = _providers(face_model=_Faces([]))
    providers.faces.faces_for = lambda image, people: [None] * (len(people) + 1)
    with pytest.raises(ValueError, match="entries"):
        _pipeline(providers).process(IMAGE, _header(), _info())


@pytest.mark.parametrize("shape", [(2, 512), (1, 256), (512,)])
def test_a_reid_model_with_the_wrong_shape_raises(shape):
    providers = _providers(people=FakePoseEstimator([_pose(BIG)]), reid=_Reid(np.ones(shape, dtype=np.float32)))
    providers.reid.embed = lambda crops: np.ones(shape, dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        _pipeline(providers).process(IMAGE, _header(), _info())


def test_a_label_outside_classes_raises():
    detector = _Detector([_detection(label="table")])
    with pytest.raises(ValueError, match="table"):
        _pipeline(_providers(objects=detector)).process(IMAGE, _header(), _info())


# --- detect_on_demand ---------------------------------------------------


class _ForbiddenDetector(FakeObjectDetector):
    """A streaming detector that a test expects no request to reach."""

    def __init__(self) -> None:
        super().__init__([], classes=CLASSES)

    def set_classes(self, classes):
        raise AssertionError("the stream's classes must not change")

    def detect(self, image):
        raise AssertionError("the stream's detector must not run")


def _on_demand_pipeline(*detections, classes=("cup",), embedder=None, **kwargs) -> VisionPipeline:
    providers = dataclasses.replace(
        _providers(objects=_ForbiddenDetector()),
        on_demand=FakeObjectDetector(list(detections), classes=classes),
        embedder=embedder,
    )
    return _pipeline(providers, on_demand_enabled=True, allow_fake_embedder=True, **kwargs)


def test_on_demand_needs_its_own_detector():
    with pytest.raises(ValueError, match="needs an on-demand detector"):
        _pipeline(on_demand_enabled=True)
    providers = _providers()
    with pytest.raises(ValueError, match="must not be the streaming detector"):
        _pipeline(dataclasses.replace(providers, on_demand=providers.objects), on_demand_enabled=True)


def test_an_on_demand_array_is_its_own_and_every_track_id_is_zero():
    det = _detection(label="cup", track_id=9)
    pipeline = _on_demand_pipeline(det)
    result = pipeline.detect_on_demand(IMAGE, _header(), _info())
    array = result.objects
    assert list(array.classes) == ["cup"]
    assert array.header == _header() and array.camera_info.header == _header()
    assert array.stream_id == pipeline.stream_id
    assert [d.track_id for d in array.detections] == [0]
    assert pipeline.classes == CLASSES and pipeline.processed_images == 0


def test_an_on_demand_call_leaves_the_sampler_on_the_same_image():
    # A track that is due on the stream's first image stays due after any number of requests.
    embedder = FakeEmbedder()
    providers = dataclasses.replace(
        _providers(objects=FakeObjectDetector([_detection(BIG, track_id=1)], classes=CLASSES)),
        on_demand=FakeObjectDetector([_detection(BIG, label="cup")], classes=("cup",)),
        embedder=embedder,
    )
    # Offset 9 maps tracker id 1 to 10, so (track_id + 0) % 10 == 0.
    pipeline = _pipeline(
        providers, start_ns=9_000_000, on_demand_enabled=True, object_embeddings_enabled=True, allow_fake_embedder=True
    )
    for _ in range(3):
        pipeline.detect_on_demand(IMAGE, _header(), _info())
    assert pipeline.processed_images == 0
    [det] = pipeline.process(IMAGE, _header(), _info()).objects.detections
    assert len(det.embedding) == embedder.dim


def test_on_demand_keeps_the_highest_scores_up_to_the_cap_before_any_work():
    embedder = _CountingEmbedder()
    # Scores rise with the index, so the cap must sort before it cuts.
    many = [_detection(BIG, label="cup", score=(i + 1) / 100.0) for i in range(pipeline_module.ON_DEMAND_MAX_OBJECTS + 5)]
    pipeline = _on_demand_pipeline(*many, embedder=embedder, object_embeddings_enabled=True)
    array = pipeline.detect_on_demand(IMAGE, _header(), _info()).objects
    assert pipeline_module.ON_DEMAND_MAX_OBJECTS == 20
    assert len(array.detections) == 20
    assert [round(float(d.score), 2) for d in array.detections] == [round((25 - i) / 100.0, 2) for i in range(20)]
    assert embedder.images == 20


def test_on_demand_applies_the_published_floor_and_counts_a_bad_box():
    pipeline = _on_demand_pipeline(
        _detection(label="cup", score=0.9),
        _detection(label="cup", score=0.1),
        _detection((float("nan"), 0.0, 1.0, 1.0), label="cup"),
        objects_min_score=0.25,
    )
    result = pipeline.detect_on_demand(IMAGE, _header(), _info())
    assert [round(float(d.score), 2) for d in result.objects.detections] == [0.9]
    assert result.dropped == {"object": 1}


def test_on_demand_embeds_every_usable_box_with_no_sampler():
    embedder = FakeEmbedder()
    pipeline = _on_demand_pipeline(
        _detection(BIG, label="cup", score=0.9),
        _detection(BIG, label="cup", score=0.8),
        _detection(BIG, label="cup", score=0.7),
        _detection(BIG, label="cup", score=0.6),
        _detection(SMALL, label="cup", score=0.5),
        embedder=embedder,
        object_embeddings_enabled=True,
        object_crops_enabled=True,
    )
    detections = pipeline.detect_on_demand(IMAGE, _header(), _info()).objects.detections
    # Four large boxes, more than the stream's limit of three, and one under 32 px.
    assert [len(d.embedding) for d in detections] == [embedder.dim] * 4 + [0]
    assert [d.embedding_model for d in detections] == [embedder.model_name] * 4 + [""]
    assert [bool(len(d.crop.data)) for d in detections] == [True] * 4 + [False]


def test_on_demand_fills_points_and_colour_shares_with_geometry_and_masks():
    mask = box_mask(48, 64, BIG)
    det = Detection(box=np.array(BIG), label="cup", score=0.8, mask=mask)
    pipeline = _on_demand_pipeline(det, embedder=FakeEmbedder(), geometry_enabled=True, object_embeddings_enabled=True)
    [got] = pipeline.detect_on_demand(IMAGE, _header(), _info(), _plain_depth()).objects.detections
    assert got.points.width > 0 and got.points.header == _header()
    assert got.view_quality > 0.0
    assert len(got.color_shares) == 12


def test_on_demand_with_the_additions_off_does_no_added_work(monkeypatch):
    # Flags stay orthogonal: on-demand alone gives boxes, and nothing more.
    _forbid(monkeypatch, "frame_laplacian", "lift_mask", "depth_cover", "view_quality", "resize_mask")
    monkeypatch.setattr(embeddings_module, "encode_crop", _raises_if_called)
    monkeypatch.setattr(embeddings_module, "mask_color_shares", _raises_if_called)
    det = Detection(box=np.array(BIG), label="cup", score=0.8, mask=box_mask(48, 64, BIG))
    [got] = _on_demand_pipeline(det).detect_on_demand(IMAGE, _header(), _info(), _plain_depth()).objects.detections
    assert got.points.width == 0 and len(got.embedding) == 0 and len(got.crop.data) == 0 and len(got.color_shares) == 0


def test_on_demand_prepares_only_its_own_detector():
    pipeline = _on_demand_pipeline(_detection(label="bowl"))
    assert pipeline.on_demand_classes == ["cup"]
    pipeline.prepare_on_demand(["bowl"])
    assert pipeline.on_demand_classes == ["bowl"]
    assert pipeline.classes == CLASSES
    assert list(pipeline.detect_on_demand(IMAGE, _header(), _info()).objects.classes) == ["bowl"]


def test_a_successful_on_demand_array_can_be_empty():
    array = _on_demand_pipeline().detect_on_demand(IMAGE, _header(), _info()).objects
    assert len(array.detections) == 0 and list(array.classes) == ["cup"] and array.stream_id


def test_set_classes_is_called_once_when_the_classes_differ():
    detector = _Detector([], classes=["person"])
    pipeline = _pipeline(_providers(objects=detector))
    pipeline.process(IMAGE, _header(), _info())
    pipeline.process(IMAGE, _header(), _info())
    assert detector.set_calls == [CLASSES]


def test_set_classes_is_not_called_when_the_classes_match():
    detector = _Detector([], classes=CLASSES)
    _pipeline(_providers(objects=detector)).process(IMAGE, _header(), _info())
    assert detector.set_calls == []


def test_stage_times_are_measured():
    times = _pipeline().process(IMAGE, _header(), _info()).times
    for stage in ("objects", "pose", "faces", "reid", "build"):
        assert getattr(times, stage) >= 0.0


class _StageClock:
    """Stands in for the pipeline's time module: its clock moves only when a slowed call runs."""

    def __init__(self) -> None:
        self.ms = 0.0

    def perf_counter(self) -> float:
        return self.ms / 1000.0

    def time_ns(self) -> int:
        return 0

    def slowed(self, call, ms: float):
        def slow(*args, **kwargs):
            self.ms += ms
            return call(*args, **kwargs)

        return slow


class _MaskTimedDetector(FakeObjectDetector):
    """A detector that reports how long its masks took, as the real detectors do."""

    last_mask_ms = 2.0


@pytest.fixture
def stage_clock(monkeypatch) -> _StageClock:
    """A clock with lifting and JPEG encoding each slowed by a known time."""
    clock = _StageClock()
    monkeypatch.setattr(pipeline_module, "time", clock)
    monkeypatch.setattr(pipeline_module, "lift_mask", clock.slowed(pipeline_module.lift_mask, 5.0))
    monkeypatch.setattr(embeddings_module, "encode_crop", clock.slowed(embeddings_module.encode_crop, 3.0))
    return clock


def _slowed_detector(clock: _StageClock, *detections, classes) -> FakeObjectDetector:
    detector = _MaskTimedDetector(list(detections), classes=classes)
    detector.detect = clock.slowed(detector.detect, 9.0)
    return detector


def _slowed_embedder(clock: _StageClock) -> FakeEmbedder:
    embedder = FakeEmbedder()
    embedder.embed_images = clock.slowed(embedder.embed_images, 11.0)
    return embedder


def _assert_stages(times, *, lift: float, crop: float, embed: float) -> None:
    # 2 ms of the detector's 9 ms are its masks. Nothing else moved the clock.
    assert times.objects == pytest.approx(7.0) and times.mask == pytest.approx(2.0)
    assert times.lift == pytest.approx(lift)
    assert times.crop == pytest.approx(crop)
    assert times.embed == pytest.approx(embed)
    assert times.build == pytest.approx(0.0)


def test_each_added_stage_of_an_image_counts_once_and_build_leaves_them_out(stage_clock):
    # Offset 9 maps tracker id 1 to 10, which the first image samples.
    pipeline = _pipeline(
        dataclasses.replace(
            _providers(objects=_slowed_detector(stage_clock, _masked(BIG, track_id=1), classes=CLASSES)),
            embedder=_slowed_embedder(stage_clock),
        ),
        start_ns=9_000_000,
        geometry_enabled=True,
        object_crops_enabled=True,
        object_embeddings_enabled=True,
        allow_fake_embedder=True,
    )
    result = pipeline.process(IMAGE, _header(), _info(), _plain_depth())
    _assert_stages(result.times, lift=5.0, crop=3.0, embed=11.0)
    assert result.sampled == 1


def test_each_added_stage_of_a_detect_objects_call_counts_once(stage_clock):
    pipeline = _pipeline(
        dataclasses.replace(
            _providers(objects=_ForbiddenDetector()),
            on_demand=_slowed_detector(stage_clock, _masked(BIG), classes=("chair",)),
            embedder=_slowed_embedder(stage_clock),
        ),
        on_demand_enabled=True,
        geometry_enabled=True,
        object_crops_enabled=True,
        object_embeddings_enabled=True,
        allow_fake_embedder=True,
    )
    result = pipeline.detect_on_demand(IMAGE, _header(), _info(), _plain_depth())
    _assert_stages(result.times, lift=5.0, crop=3.0, embed=11.0)
    assert result.sampled == 1


def test_a_stage_whose_addition_is_off_stays_at_zero():
    result = _pipeline(_providers(objects=FakeObjectDetector([_masked(BIG, track_id=10)], classes=CLASSES))).process(
        IMAGE, _header(), _info(), _plain_depth()
    )
    times = result.times
    assert (times.mask, times.lift, times.crop, times.embed) == (0.0, 0.0, 0.0, 0.0)
    assert result.sampled == 0


def test_the_objects_stage_cannot_go_below_zero_when_the_masks_took_all_of_it():
    times = pipeline_module.StageTimes()
    pipeline_module._split_detector_time(times, _MaskTimedDetector([], classes=CLASSES), 1.5)
    assert (times.objects, times.mask) == (0.0, 2.0)


def test_sampled_counts_only_the_objects_that_got_a_sample():
    # Tracker ids 10 and 11 map to 10 and 11, and only the first is due on image 0.
    detections = [_masked(BIG, track_id=10), _masked(BIG, track_id=11)]
    pipeline = _pipeline(
        _providers(objects=FakeObjectDetector(detections, classes=CLASSES)), object_crops_enabled=True
    )
    assert pipeline.process(IMAGE, _header(), _info()).sampled == 1


# --- LatencyLog ---------------------------------------------------------


def _add(log: LatencyLog, value: float) -> None:
    log.add(**{stage: value for stage in LatencyLog.STAGES})


def test_latency_summary_format_and_percentiles():
    log = LatencyLog()
    for value in range(101):  # 0..100: p50 is 50 and p95 is 95
        _add(log, float(value))
    counts = {"received": 150, "replaced": 98, "skipped": 0, "failed": 0, "dropped": 0}
    line = log.summary(10.0, counts)
    stages = " ".join(f"{s} 50/95" for s in LatencyLog.STAGES if s != "age")
    assert line == (
        "processed 101 in 10.0 s (10.1 Hz); received 150, replaced 98, skipped 0, failed 0, dropped 0 "
        f"| ms p50/p95: {stages} | age 50/95"
    )


def test_latency_summary_without_samples():
    line = LatencyLog().summary(10.0, {"received": 0})
    assert line.startswith("processed 0 in 10.0 s (0.0 Hz); received 0 | ms p50/p95: wait -/-")
    assert line.endswith("| age -/-")


def test_latency_counts_are_the_change_since_the_previous_summary():
    log = LatencyLog()
    log.summary(1.0, {"received": 10, "failed": 1})
    _add(log, 1.0)
    line = log.summary(1.0, {"received": 25, "failed": 1})
    assert "processed 1 in 1.0 s" in line and "received 15, failed 0" in line
    assert log.last_processed == 1 and log.last_deltas == {"received": 15, "failed": 0}


def test_latency_add_rejects_unknown_stages():
    with pytest.raises(ValueError):
        LatencyLog().add(nope=1.0)


BASE_STAGES = ("wait", "convert", "objects", "pose", "faces", "reid", "build", "publish", "age")


def test_with_every_addition_off_the_latency_line_keeps_its_old_form():
    log = LatencyLog()
    log.add(**dict.fromkeys(BASE_STAGES, 1.0))
    assert log.summary(1.0, {}) == (
        "processed 1 in 1.0 s (1.0 Hz) "
        "| ms p50/p95: wait 1/1 convert 1/1 objects 1/1 pose 1/1 faces 1/1 reid 1/1 build 1/1 publish 1/1 | age 1/1"
    )


def test_an_added_stage_shows_in_its_place_once_it_has_samples():
    log = LatencyLog()
    log.add(**dict.fromkeys(BASE_STAGES, 1.0), depth=2.0, mask=3.0, lift=4.0, crop=5.0, embed=6.0)
    assert log.summary(1.0, {}) == (
        "processed 1 in 1.0 s (1.0 Hz) | ms p50/p95: wait 1/1 convert 1/1 depth 2/2 objects 1/1 mask 3/3 pose 1/1 "
        "faces 1/1 reid 1/1 lift 4/4 crop 5/5 embed 6/6 build 1/1 publish 1/1 | age 1/1"
    )


def test_a_service_job_shows_at_the_end_and_is_no_processed_image():
    log = LatencyLog()
    log.add_job(service=100.0)
    line = log.summary(1.0, {})
    assert line.startswith("processed 0 in 1.0 s")
    assert line.endswith("| age -/- | service 100/100")
    assert log.last_processed == 0

    log.add_job(service=200.0, prepare=150.0)
    assert log.summary(1.0, {}).endswith("| age -/- | service 200/200 prepare 150/150")
    assert log.summary(1.0, {}).endswith("| age -/-")


def test_latency_add_job_rejects_an_image_stage():
    with pytest.raises(ValueError):
        LatencyLog().add_job(wait=1.0)


def test_latency_samples_are_never_lost_or_counted_twice():
    log = LatencyLog()
    done = threading.Event()

    def worker():
        for i in range(1000):
            _add(log, float(i))
        done.set()

    totals = []
    thread = threading.Thread(target=worker)
    thread.start()
    while not done.is_set():
        log.summary(0.1, {})
        totals.append(log.last_processed)
    thread.join()
    log.summary(0.1, {})
    totals.append(log.last_processed)
    assert sum(totals) == 1000
