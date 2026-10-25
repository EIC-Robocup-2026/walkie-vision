"""VisionPipeline with the fakes: one image to its two arrays. No ROS graph.

The LatencyLog tests are here too: it is the node's, but needs no node.
"""

import threading

import numpy as np
import pytest
from builtin_interfaces.msg import Time
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header

from walkie_vision.fakes import FakeObjectDetector, FakePersonFaceRecognizer, FakePoseEstimator, FakeReIdentifier
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
