"""Tests of walkie_vision.yolo without a GPU or a model file.

The pure helpers are tested directly. The two tracker classes are tested with
stand-ins for Ultralytics' YOLOE and YOLO classes, so no model is built. The
last tests pin the ByteTrack behaviour that the vision node's contract relies
on (one-frame delay for new tracks, ids per tracker, the track buffer).
Re-run them and checks/c11_providers.py on any Ultralytics version bump.
"""

from __future__ import annotations

import os

# Before Ultralytics is imported: it reads this once, at import.
os.environ.setdefault("YOLO_OFFLINE", "true")

import types
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

byte_tracker = pytest.importorskip("ultralytics.trackers.byte_tracker")

import ultralytics  # noqa: E402
from ultralytics.engine.results import Boxes, Keypoints, Results  # noqa: E402
from ultralytics.utils import ops  # noqa: E402

from walkie_vision.providers import ObjectDetector, PoseEstimator  # noqa: E402
from walkie_vision.yolo import (  # noqa: E402
    MOBILECLIP_FILE,
    TRACKER_NAMES,
    YoloeDetector,
    YoloeTracker,
    YoloPoseTracker,
    detections_from_arrays,
    mask_to_native,
    poses_from_arrays,
    untracked_detections_from_arrays,
    validate_classes,
    validate_tracker,
)

SHAPE = (480, 640)  # height, width


def _keypoints(n: int, score: float = 0.9) -> np.ndarray:
    kp = np.zeros((n, 17, 3), dtype=np.float32)
    kp[:, :, 0] = np.linspace(100, 200, 17)
    kp[:, :, 1] = np.linspace(50, 400, 17)
    kp[:, :, 2] = score
    return kp


# --- detections_from_arrays -------------------------------------------------


def test_detections_without_track_ids_are_empty():
    xyxy = np.array([[10.0, 20.0, 110.0, 220.0]])
    assert detections_from_arrays(xyxy, np.array([0.9]), np.array([0]), None, ["chair"], SHAPE) == []


def test_detections_drop_untracked_rows():
    xyxy = np.array([[10.0, 20.0, 110.0, 220.0], [200.0, 20.0, 300.0, 220.0], [300.0, 20.0, 400.0, 220.0]])
    dets = detections_from_arrays(
        xyxy, np.array([0.9, 0.8, 0.7]), np.array([0, 0, 0]), np.array([0, 5, -1]), ["chair"], SHAPE
    )
    assert [d.track_id for d in dets] == [5]
    np.testing.assert_array_equal(dets[0].box, [200.0, 20.0, 300.0, 220.0])


def test_detection_boxes_are_clipped_and_empty_boxes_dropped():
    xyxy = np.array(
        [
            [-15.0, -5.0, 700.0, 500.0],  # past every edge
            [650.0, 10.0, 700.0, 50.0],  # fully right of the image
            [100.0, 100.0, 100.0, 150.0],  # zero width
        ]
    )
    dets = detections_from_arrays(
        xyxy, np.array([0.9, 0.9, 0.9]), np.array([0, 0, 0]), np.array([1, 2, 3]), ["chair"], SHAPE
    )
    assert len(dets) == 1
    assert dets[0].box.dtype == np.float64
    np.testing.assert_array_equal(dets[0].box, [0.0, 0.0, 640.0, 480.0])


@pytest.mark.parametrize("names", [["chair", "sofa"], {0: "chair", 1: "sofa"}])
def test_detection_labels_and_types(names):
    xyxy = np.array([[10.0, 20.0, 110.0, 220.0], [200.0, 20.0, 300.0, 220.0]], dtype=np.float32)
    dets = detections_from_arrays(
        xyxy, np.array([0.5, 0.25], dtype=np.float32), np.array([1, 0]), np.array([7, 8]), names, SHAPE
    )
    assert [d.label for d in dets] == ["sofa", "chair"]
    assert [d.track_id for d in dets] == [7, 8]
    assert all(type(d.track_id) is int and type(d.score) is float and d.mask is None for d in dets)
    assert dets[0].score == pytest.approx(0.5)


def test_detection_arrays_of_different_lengths_raise():
    with pytest.raises(ValueError):
        detections_from_arrays(
            np.zeros((2, 4)), np.array([0.9]), np.array([0, 0]), np.array([1, 2]), ["chair"], SHAPE
        )


def test_untracked_detections_have_track_id_zero_and_clipped_boxes():
    xyxy = np.array([[-5.0, 20.0, 700.0, 220.0], [650.0, 10.0, 700.0, 50.0]], dtype=np.float32)
    dets = untracked_detections_from_arrays(xyxy, np.array([0.5, 0.9]), np.array([1, 0]), ["chair", "sofa"], SHAPE)
    assert [(d.label, d.track_id) for d in dets] == [("sofa", 0)]
    np.testing.assert_array_equal(dets[0].box, [0.0, 20.0, 640.0, 220.0])
    assert type(dets[0].score) is float and dets[0].mask is None


def test_a_kept_untracked_box_has_the_mask_of_its_own_row():
    # Row 0 lies outside the image and is dropped: row 1 must keep its own mask.
    xyxy = np.array([[700.0, 0.0, 800.0, 10.0], [0.0, 0.0, 10.0, 10.0]])
    masks = np.zeros((2, 480, 640), dtype=np.float32)
    masks[0, 0:5, 0:5] = 1.0
    masks[1, 100:105, 100:105] = 1.0
    [det] = untracked_detections_from_arrays(xyxy, np.array([0.9, 0.8]), np.array([0, 0]), ["chair"], SHAPE, masks)
    assert det.mask.dtype == bool and det.mask.shape == SHAPE
    assert det.mask[102, 102] and not det.mask[2, 2]


def test_untracked_arrays_of_different_lengths_raise():
    with pytest.raises(ValueError):
        untracked_detections_from_arrays(np.zeros((2, 4)), np.array([0.9]), np.array([0, 0]), ["chair"], SHAPE)


# --- masks ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("image_shape", "on_model", "native"),
    [
        # A wide image: gain 0.5, and 12 rows of padding above and below.
        ((16, 64), (14, 16, 4, 10), (4, 8, 8, 20)),
        # A tall image: gain 0.5, and 12 columns of padding left and right.
        ((64, 16), (4, 10, 14, 16), (8, 20, 4, 8)),
    ],
)
def test_a_letterboxed_mask_lands_on_its_native_pixels(image_shape, on_model, native):
    top, bottom, left, right = on_model
    mask = np.zeros((32, 32), np.uint8)
    mask[top:bottom, left:right] = 1
    top, bottom, left, right = native
    expected = np.zeros(image_shape, bool)
    expected[top:bottom, left:right] = True
    result = mask_to_native(mask, image_shape)
    assert result.dtype == bool
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    ("image_shape", "padding"),
    [
        ((16, 64), (slice(0, 12), slice(None))),
        ((16, 64), (slice(20, 32), slice(None))),
        ((64, 16), (slice(None), slice(0, 12))),
        ((64, 16), (slice(None), slice(20, 32))),
    ],
)
def test_the_letterbox_padding_never_reaches_the_image(image_shape, padding):
    mask = np.zeros((32, 32), np.uint8)
    mask[padding] = 1
    assert not mask_to_native(mask, image_shape).any()


def test_a_native_mask_only_becomes_bool():
    mask = np.zeros(SHAPE, np.float32)
    mask[3, 5] = 1.0
    result = mask_to_native(mask, SHAPE)
    assert result.dtype == bool and result.shape == SHAPE and np.count_nonzero(result) == 1 and result[3, 5]


@pytest.mark.parametrize(
    ("image_shape", "model_shape"),
    [
        ((1242, 2208), (384, 640)),  # The ZED's size at the default input size.
        ((101, 211), (64, 128)),  # Odd padding, where the rounding shows.
    ],
)
def test_the_padding_rule_matches_the_installed_ultralytics(image_shape, model_shape):
    # An L shape, so that a flip or a shift would show. Ultralytics interpolates
    # and this code takes the nearest pixel, so only the edges may differ.
    height, width = model_shape
    mask = np.zeros(model_shape, np.float32)
    mask[height // 4 : 3 * height // 4, width // 8 : width // 4] = 1.0
    mask[height // 2 : 3 * height // 4, width // 8 : 3 * width // 4] = 1.0
    ours = mask_to_native(mask, image_shape)
    theirs = ops.scale_masks(torch.from_numpy(mask)[None, None], image_shape)[0, 0].numpy() > 0.5
    assert np.count_nonzero(ours & theirs) / np.count_nonzero(ours | theirs) > 0.95


@pytest.mark.parametrize(
    ("track_ids", "kept_ids", "kept_rows"),
    [
        ([0, 7, 9, 8], [7, 8], [1, 3]),
        ([5, 0, 9, 8], [5, 8], [0, 3]),
        ([5, 7, 9, 0], [5, 7], [0, 1]),
    ],
)
def test_a_kept_box_has_the_mask_of_its_own_row(track_ids, kept_ids, kept_rows):
    # Row 2 lies outside the image, and a track id of 0 drops one more row.
    xyxy = np.array([[0, 0, 10, 10], [20, 0, 30, 10], [700, 0, 800, 10], [40, 0, 50, 10]], dtype=np.float64)
    masks = np.zeros((4, *SHAPE), np.uint8)
    for row in range(4):
        masks[row, row, :] = 1
    dets = detections_from_arrays(
        xyxy, np.full(4, 0.9), np.zeros(4, int), np.array(track_ids), ["chair"], SHAPE, masks
    )
    assert [d.track_id for d in dets] == kept_ids
    assert [np.flatnonzero(d.mask.any(axis=1)).tolist() for d in dets] == [[row] for row in kept_rows]


def test_masks_of_another_row_count_raise():
    with pytest.raises(ValueError, match="masks"):
        detections_from_arrays(
            np.zeros((2, 4)), np.array([0.9, 0.8]), np.array([0, 0]), np.array([1, 2]), ["chair"], SHAPE,
            np.zeros((1, *SHAPE)),
        )


# --- poses_from_arrays ------------------------------------------------------


def test_poses_without_track_ids_are_empty():
    assert poses_from_arrays(np.array([[0.0, 0.0, 50.0, 50.0]]), np.array([0.9]), _keypoints(1), None, SHAPE) == []


def test_pose_keypoints_are_split_with_their_types():
    kp = _keypoints(2)
    kp[1, :, 2] = np.linspace(0.0, 1.0, 17)
    poses = poses_from_arrays(
        np.array([[90.0, 40.0, 210.0, 410.0], [300.0, 40.0, 400.0, 410.0]]),
        np.array([0.9, 0.6]),
        kp,
        np.array([3, 0]),
        SHAPE,
    )
    assert len(poses) == 1
    p = poses[0]
    assert p.track_id == 3 and type(p.track_id) is int and type(p.score) is float
    assert p.keypoints.shape == (17, 2) and p.keypoints.dtype == np.float32
    assert p.keypoint_scores.shape == (17,) and p.keypoint_scores.dtype == np.float32
    np.testing.assert_array_equal(p.keypoints, kp[0, :, :2])
    np.testing.assert_array_equal(p.keypoint_scores, kp[0, :, 2])
    assert p.box.dtype == np.float64


def test_pose_keypoints_are_clipped_and_never_zeroed():
    kp = _keypoints(1)
    kp[0, 0] = [-20.0, 30.0, 0.95]  # left of the image
    kp[0, 1] = [700.0, 500.0, 0.02]  # past the right and bottom edges, low score
    kp[0, 2] = [320.0, 240.0, 0.0]  # inside, score 0
    poses = poses_from_arrays(np.array([[-10.0, 0.0, 650.0, 480.0]]), np.array([0.9]), kp, np.array([1]), SHAPE)
    p = poses[0]
    np.testing.assert_array_equal(p.box, [0.0, 0.0, 640.0, 480.0])
    np.testing.assert_array_equal(p.keypoints[0], [0.0, 30.0])
    np.testing.assert_array_equal(p.keypoints[1], [640.0, 480.0])
    np.testing.assert_array_equal(p.keypoints[2], [320.0, 240.0])
    np.testing.assert_allclose(p.keypoint_scores[:3], [0.95, 0.02, 0.0])


def test_pose_boxes_with_no_area_are_dropped():
    poses = poses_from_arrays(
        np.array([[650.0, 0.0, 700.0, 100.0]]), np.array([0.9]), _keypoints(1), np.array([1]), SHAPE
    )
    assert poses == []


@pytest.mark.parametrize("shape", [(1, 17, 2), (1, 16, 3), (17, 3)])
def test_pose_keypoints_of_the_wrong_shape_raise(shape):
    with pytest.raises(ValueError):
        poses_from_arrays(np.zeros((1, 4)), np.array([0.9]), np.zeros(shape), np.array([1]), SHAPE)


# --- validate_classes and validate_tracker ----------------------------------


def test_validate_classes_returns_a_copy():
    given = ("chair", "sofa", "armchair", "stool")
    got = validate_classes(given)
    assert got == list(given) and isinstance(got, list)
    original = ["chair"]
    assert validate_classes(original) is not original


@pytest.mark.parametrize(
    "classes", [[], ["chair", "chair"], ["chair", ""], [" chair"], ["chair "], ["chair", "sofa", "sofa"], "chair"]
)
def test_validate_classes_rejects(classes):
    with pytest.raises(ValueError):
        validate_classes(classes)


def test_validate_tracker_accepts_names_and_absolute_yaml(tmp_path):
    for name in TRACKER_NAMES:
        assert validate_tracker(name) == name
    custom = tmp_path / "my_tracker.yaml"
    custom.write_text("tracker_type: bytetrack\n")
    assert validate_tracker(str(custom)) == str(custom)


def test_validate_tracker_rejects(tmp_path):
    not_yaml = tmp_path / "tracker.txt"
    not_yaml.write_text("x")
    for bad in ("ocsort.yaml", "trackers/bytetrack.yaml", str(tmp_path / "missing.yaml"), str(not_yaml), ""):
        with pytest.raises(ValueError):
            validate_tracker(bad)


# --- the tracker classes, with stand-ins for Ultralytics' models -----------


class _Tracker:
    def __init__(self) -> None:
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1


class _StubModel:
    """Records what the wrapper does with an Ultralytics model and returns fixed results."""

    def __init__(self, weights: str) -> None:
        self.calls: list[tuple] = [("init", weights, os.getcwd())]
        self.predictor = None
        self.results = None

    def set_classes(self, classes: list[str]) -> None:
        self.calls.append(("set_classes", list(classes), os.getcwd()))
        # Ultralytics drops the predictor, and with it the trackers.
        self.predictor = None

    def track(self, image: np.ndarray, **kwargs):
        self.calls.append(("track", image.shape, kwargs))
        if self.predictor is None:
            self.predictor = types.SimpleNamespace(trackers=[_Tracker()])
        if self.results is not None:
            return [self.results]
        return [types.SimpleNamespace(boxes=Boxes(torch.zeros((0, 6)), image.shape[:2]), names={}, keypoints=None)]

    def predict(self, image: np.ndarray, **kwargs):
        self.calls.append(("predict", image.shape, kwargs))
        if self.results is not None:
            return [self.results]
        return [types.SimpleNamespace(boxes=Boxes(torch.zeros((0, 6)), image.shape[:2]), names={}, masks=None)]


@pytest.fixture
def stub_ultralytics(monkeypatch):
    monkeypatch.setattr(ultralytics, "YOLOE", _StubModel)
    monkeypatch.setattr(ultralytics, "YOLO", _StubModel)


def _weights(tmp_path: Path, *names: str) -> Path:
    for name in names:
        (tmp_path / name).write_bytes(b"")
    return tmp_path / names[0]


def test_yoloe_checks_its_arguments_before_any_model(tmp_path, stub_ultralytics):
    with pytest.raises(ValueError):
        YoloeTracker(tmp_path / "yoloe.pt", ["chair", "chair"])
    with pytest.raises(ValueError):
        YoloeTracker(tmp_path / "yoloe.pt", ["chair"], tracker="relative.yaml")
    with pytest.raises(FileNotFoundError, match="fetch-models"):
        YoloeTracker(tmp_path / "yoloe.pt", ["chair"])
    weights = _weights(tmp_path, "yoloe.pt")
    with pytest.raises(FileNotFoundError, match=MOBILECLIP_FILE):
        YoloeTracker(weights, ["chair"])


def test_yoloe_loads_in_the_weights_folder_then_warms_up_and_resets(tmp_path, stub_ultralytics):
    weights = _weights(tmp_path, "yoloe.pt", MOBILECLIP_FILE)
    cwd = os.getcwd()
    tracker = YoloeTracker(weights, ("chair", "sofa"), device="cpu", imgsz=64)
    assert os.getcwd() == cwd
    assert isinstance(tracker, ObjectDetector)
    model = tracker._model
    kinds = [c[0] for c in model.calls]
    assert kinds == ["init", "set_classes", "track"]
    assert model.calls[0][1:] == (str(weights.resolve()), str(tmp_path))
    assert model.calls[1][1:] == (["chair", "sofa"], str(tmp_path))
    assert model.calls[2][1] == (64, 64, 3)
    assert model.calls[2][2] == {
        "persist": True,
        "tracker": "bytetrack.yaml",
        "conf": 0.1,
        "imgsz": 64,
        "agnostic_nms": True,
        "device": "cpu",
        "verbose": False,
    }
    # The warm-up used the tracker's first frame; it was reset after it.
    assert model.predictor.trackers[0].resets == 1
    assert tracker.classes == ["chair", "sofa"]
    tracker.classes.append("stool")
    assert tracker.classes == ["chair", "sofa"]


def test_yoloe_set_classes_warms_up_and_resets_again(tmp_path, stub_ultralytics):
    tracker = YoloeTracker(_weights(tmp_path, "yoloe.pt", MOBILECLIP_FILE), ["chair"], device="cpu", imgsz=32)
    tracker.set_classes(["sofa", "stool"])
    model = tracker._model
    assert [c[0] for c in model.calls[-2:]] == ["set_classes", "track"]
    assert model.calls[-2][2] == str(tmp_path)
    assert model.predictor.trackers[0].resets == 1
    assert tracker.classes == ["sofa", "stool"]
    with pytest.raises(ValueError):
        tracker.set_classes([])
    assert tracker.classes == ["sofa", "stool"]


def test_yoloe_detect_converts_tracked_boxes(tmp_path, stub_ultralytics):
    tracker = YoloeTracker(_weights(tmp_path, "yoloe.pt", MOBILECLIP_FILE), ["chair", "sofa"], device="cpu")
    image = np.zeros((480, 640, 3), np.uint8)
    # x1, y1, x2, y2, track id, score, class
    data = torch.tensor([[10.0, 20.0, 700.0, 220.0, 4.0, 0.8, 1.0], [5.0, 5.0, 50.0, 50.0, 9.0, 0.3, 0.0]])
    tracker._model.results = types.SimpleNamespace(boxes=Boxes(data, SHAPE), names={0: "chair", 1: "sofa"})
    dets = tracker.detect(image)
    assert [(d.label, d.track_id) for d in dets] == [("sofa", 4), ("chair", 9)]
    np.testing.assert_array_equal(dets[0].box, [10.0, 20.0, 640.0, 220.0])
    # Boxes without ids (nothing confirmed yet) give no detections.
    untracked = torch.tensor([[10.0, 20.0, 100.0, 220.0, 0.8, 1.0]])
    tracker._model.results = types.SimpleNamespace(boxes=Boxes(untracked, SHAPE), names={0: "chair", 1: "sofa"})
    assert tracker.detect(image) == []
    tracker._model.results = types.SimpleNamespace(boxes=None, names={})
    assert tracker.detect(image) == []


def test_the_detector_checks_its_arguments_before_any_model(tmp_path, stub_ultralytics):
    with pytest.raises(ValueError):
        YoloeDetector(tmp_path / "yoloe.pt", ["chair", "chair"])
    with pytest.raises(FileNotFoundError, match="fetch-models"):
        YoloeDetector(tmp_path / "yoloe.pt", ["chair"])
    with pytest.raises(FileNotFoundError, match=MOBILECLIP_FILE):
        YoloeDetector(_weights(tmp_path, "yoloe.pt"), ["chair"])


def test_the_detector_loads_in_the_weights_folder_and_warms_up_with_no_tracker(tmp_path, stub_ultralytics):
    weights = _weights(tmp_path, "yoloe.pt", MOBILECLIP_FILE)
    cwd = os.getcwd()
    detector = YoloeDetector(weights, ("chair", "sofa"), device="cpu", imgsz=64, conf=0.3, max_det=7)
    assert os.getcwd() == cwd
    assert isinstance(detector, ObjectDetector)
    calls = detector._model.calls
    assert [c[0] for c in calls] == ["init", "set_classes", "predict"]
    assert calls[0][1:] == (str(weights.resolve()), str(tmp_path))
    assert calls[1][1:] == (["chair", "sofa"], str(tmp_path))
    assert calls[2][1] == (64, 64, 3)
    assert calls[2][2] == {
        "conf": 0.3,
        "imgsz": 64,
        "agnostic_nms": True,
        "max_det": 7,
        "device": "cpu",
        "verbose": False,
    }
    assert detector.classes == ["chair", "sofa"]


def test_the_detector_gives_track_id_zero_and_masks_only_when_asked(tmp_path, stub_ultralytics):
    files = _weights(tmp_path, "yoloe.pt", MOBILECLIP_FILE)
    image = np.zeros(SHAPE + (3,), np.uint8)
    # x1, y1, x2, y2, score, class
    boxes = Boxes(torch.tensor([[10.0, 20.0, 700.0, 220.0, 0.8, 1.0]]), SHAPE)
    mask = torch.zeros((1, 480, 640))
    mask[0, 30:60, 30:60] = 1.0
    result = types.SimpleNamespace(boxes=boxes, names={0: "chair", 1: "sofa"}, masks=types.SimpleNamespace(data=mask))

    plain = YoloeDetector(files, ["chair", "sofa"], device="cpu")
    plain._model.results = result
    [det] = plain.detect(image)
    assert (det.label, det.track_id, det.mask) == ("sofa", 0, None)
    np.testing.assert_array_equal(det.box, [10.0, 20.0, 640.0, 220.0])

    masked = YoloeDetector(files, ["chair", "sofa"], device="cpu", keep_masks=True)
    masked._model.results = result
    [det] = masked.detect(image)
    assert det.track_id == 0 and det.mask.shape == SHAPE and det.mask[40, 40]

    plain._model.results = types.SimpleNamespace(boxes=None, names={}, masks=None)
    assert plain.detect(image) == []


def test_a_failed_set_classes_leaves_the_detector_with_no_list(tmp_path, stub_ultralytics, monkeypatch):
    detector = YoloeDetector(_weights(tmp_path, "yoloe.pt", MOBILECLIP_FILE), ["chair"], device="cpu")

    def broken(classes):
        raise RuntimeError("the text encoder failed")

    monkeypatch.setattr(detector._model, "set_classes", broken)
    with pytest.raises(RuntimeError):
        detector.set_classes(["sofa"])
    assert detector.classes == []


def test_a_failed_warm_up_leaves_the_detector_with_no_list(tmp_path, stub_ultralytics, monkeypatch):
    detector = YoloeDetector(_weights(tmp_path, "yoloe.pt", MOBILECLIP_FILE), ["chair"], device="cpu")

    def broken(image, **kwargs):
        raise RuntimeError("the model failed")

    monkeypatch.setattr(detector._model, "predict", broken)
    with pytest.raises(RuntimeError):
        detector.set_classes(["sofa"])
    assert detector.classes == []


def test_two_model_objects_keep_the_streams_classes_and_track_ids(tmp_path, stub_ultralytics):
    files = _weights(tmp_path, "yoloe.pt", MOBILECLIP_FILE)
    tracker = YoloeTracker(files, ["chair", "sofa"], device="cpu")
    detector = YoloeDetector(files, ["chair", "sofa"], device="cpu")
    assert tracker._model is not detector._model

    image = np.zeros(SHAPE + (3,), np.uint8)
    # x1, y1, x2, y2, track id, score, class
    tracks = torch.tensor([[10.0, 20.0, 100.0, 220.0, 4.0, 0.8, 1.0]])
    tracker._model.results = types.SimpleNamespace(boxes=Boxes(tracks, SHAPE), names={0: "chair", 1: "sofa"})
    before = [(d.label, d.track_id) for d in tracker.detect(image)]
    assert before == [("sofa", 4)]
    stream_calls = list(tracker._model.calls)
    stream_trackers = tracker._model.predictor.trackers
    resets = stream_trackers[0].resets

    detector.set_classes(["cup"])
    detector.detect(image)
    detector.set_classes(["bottle", "bowl"])
    detector.detect(image)

    # Nothing reached the stream's model, its tracker or its list.
    assert tracker._model.calls == stream_calls
    assert tracker._model.predictor.trackers is stream_trackers and stream_trackers[0].resets == resets
    assert tracker.classes == ["chair", "sofa"]
    assert [(d.label, d.track_id) for d in tracker.detect(image)] == before
    # And the detector never tracked.
    assert "track" not in [c[0] for c in detector._model.calls]
    assert detector.classes == ["bottle", "bowl"]


def _tracked_result(names: dict[int, str]) -> Results:
    """Return a real Results object after the steps of Ultralytics' tracker callback.

    A wide 16 x 64 image and masks at 32 x 32, with 12 rows of padding. The
    predictor found three rows. The tracker keeps rows 2 and 0, in that order.
    """
    image = np.zeros((16, 64, 3), np.uint8)
    boxes = torch.tensor(
        [[0.0, 0.0, 8.0, 8.0, 0.9, 0.0], [10.0, 0.0, 18.0, 8.0, 0.8, 1.0], [20.0, 0.0, 28.0, 8.0, 0.7, 1.0]]
    )
    masks = torch.zeros((3, 32, 32))
    for row in range(3):
        masks[row, 12 + 2 * row, 4:8] = 1.0
    tracks = torch.tensor([[20.0, 0.0, 28.0, 8.0, 7.0, 0.7, 1.0], [0.0, 0.0, 8.0, 8.0, 8.0, 0.9, 0.0]])
    result = Results(image, path="", names=names, boxes=boxes, masks=masks)[torch.tensor([2, 0])]
    result.update(boxes=tracks)
    return result


def test_yoloe_keeps_each_row_mask_through_the_tracker_filter(tmp_path, stub_ultralytics):
    tracker = YoloeTracker(
        _weights(tmp_path, "yoloe.pt", MOBILECLIP_FILE), ["chair", "sofa"], device="cpu", keep_masks=True
    )
    tracker._model.results = _tracked_result({0: "chair", 1: "sofa"})
    first, second = tracker.detect(np.zeros((16, 64, 3), np.uint8))
    assert (first.label, first.track_id, second.label, second.track_id) == ("sofa", 7, "chair", 8)
    # Row 2's mask is on model row 16, native rows 8 and 9. Row 0's is on model row 12, native rows 0 and 1.
    assert first.mask.shape == (16, 64) and first.mask.dtype == bool
    assert np.flatnonzero(first.mask.any(axis=1)).tolist() == [8, 9]
    assert np.flatnonzero(second.mask.any(axis=1)).tolist() == [0, 1]
    assert np.flatnonzero(first.mask.any(axis=0)).tolist() == list(range(8, 16))


@pytest.mark.parametrize("make", [YoloeTracker, YoloeDetector])
def test_only_kept_masks_report_a_time(tmp_path, stub_ultralytics, make):
    files = _weights(tmp_path, "yoloe.pt", MOBILECLIP_FILE)
    image = np.zeros((16, 64, 3), np.uint8)
    masked = make(files, ["chair", "sofa"], device="cpu", keep_masks=True)
    assert masked.last_mask_ms == 0.0
    masked._model.results = _tracked_result({0: "chair", 1: "sofa"})
    assert masked.detect(image)
    assert masked.last_mask_ms > 0.0

    # A later image with no box leaves no time behind from the one before.
    masked._model.results = types.SimpleNamespace(boxes=None, names={}, masks=None)
    assert masked.detect(image) == []
    assert masked.last_mask_ms == 0.0

    plain = make(files, ["chair", "sofa"], device="cpu")
    plain._model.results = _tracked_result({0: "chair", 1: "sofa"})
    assert plain.detect(image)
    assert plain.last_mask_ms == 0.0


class _MasksMustNotBeRead(types.SimpleNamespace):
    @property
    def masks(self):
        raise AssertionError("a tracker that was not asked for masks read them")


def test_yoloe_without_keep_masks_does_not_read_the_masks(tmp_path, stub_ultralytics):
    tracker = YoloeTracker(_weights(tmp_path, "yoloe.pt", MOBILECLIP_FILE), ["chair"], device="cpu")
    data = torch.tensor([[10.0, 20.0, 100.0, 220.0, 4.0, 0.8, 0.0]])
    tracker._model.results = _MasksMustNotBeRead(boxes=Boxes(data, SHAPE), names={0: "chair"})
    (det,) = tracker.detect(np.zeros((480, 640, 3), np.uint8))
    assert det.mask is None


def test_yoloe_with_keep_masks_accepts_a_result_without_masks(tmp_path, stub_ultralytics):
    tracker = YoloeTracker(
        _weights(tmp_path, "yoloe.pt", MOBILECLIP_FILE), ["chair"], device="cpu", keep_masks=True
    )
    data = torch.tensor([[10.0, 20.0, 100.0, 220.0, 4.0, 0.8, 0.0]])
    tracker._model.results = types.SimpleNamespace(boxes=Boxes(data, SHAPE), names={0: "chair"}, masks=None)
    (det,) = tracker.detect(np.zeros((480, 640, 3), np.uint8))
    assert det.mask is None


def test_pose_tracker_warms_up_resets_and_converts(tmp_path, stub_ultralytics):
    with pytest.raises(FileNotFoundError, match="fetch-models"):
        YoloPoseTracker(tmp_path / "pose.pt")
    with pytest.raises(ValueError):
        YoloPoseTracker(_weights(tmp_path, "pose.pt"), tracker="nope.yaml")
    tracker = YoloPoseTracker(_weights(tmp_path, "pose.pt"), device="cpu", imgsz=96)
    assert isinstance(tracker, PoseEstimator)
    model = tracker._model
    assert [c[0] for c in model.calls] == ["init", "track"]
    assert model.calls[1][1] == (96, 96, 3)
    assert model.calls[1][2] == {
        "persist": True,
        "tracker": "botsort.yaml",
        "conf": 0.1,
        "imgsz": 96,
        "device": "cpu",
        "verbose": False,
    }
    assert model.predictor.trackers[0].resets == 1

    image = np.zeros((480, 640, 3), np.uint8)
    data = torch.tensor([[90.0, 40.0, 210.0, 410.0, 2.0, 0.9, 0.0]])
    kp = Keypoints(torch.from_numpy(_keypoints(1)), SHAPE)
    model.results = types.SimpleNamespace(boxes=Boxes(data, SHAPE), names={0: "person"}, keypoints=kp)
    poses = tracker.estimate(image)
    assert len(poses) == 1 and poses[0].track_id == 2
    np.testing.assert_allclose(poses[0].keypoints, _keypoints(1)[0, :, :2])
    model.results = types.SimpleNamespace(boxes=Boxes(data, SHAPE), names={0: "person"}, keypoints=None)
    assert tracker.estimate(image) == []

    tracker.reset_tracks()
    assert model.predictor.trackers[0].resets == 2


def test_reset_tracks_without_a_predictor_does_nothing(tmp_path, stub_ultralytics):
    tracker = YoloPoseTracker(_weights(tmp_path, "pose.pt"), device="cpu", imgsz=32)
    tracker._model.predictor = None
    tracker.reset_tracks()


# --- ByteTrack facts the contract relies on ---------------------------------


def _bytetrack():
    cfg_file = Path(ultralytics.__file__).parent / "cfg" / "trackers" / "bytetrack.yaml"
    cfg = yaml.safe_load(cfg_file.read_text())
    return byte_tracker.BYTETracker(types.SimpleNamespace(**cfg))


def _frame(*rows: list[float]) -> Boxes:
    """Detections of one frame: x1, y1, x2, y2, conf, cls."""
    return Boxes(np.array(rows, dtype=np.float32).reshape(-1, 6), orig_shape=SHAPE)


def _ids(output: np.ndarray) -> list[int]:
    return sorted(int(i) for i in output[:, 4]) if len(output) else []


A = [100.0, 100.0, 200.0, 300.0, 0.9, 0.0]
B = [400.0, 100.0, 500.0, 300.0, 0.9, 0.0]


def test_bytetrack_settings_are_the_ones_the_contract_assumes():
    t = _bytetrack()
    assert t.args.track_buffer == 30 and t.max_frames_lost == 30
    assert t.args.new_track_thresh == 0.25 and t.args.track_high_thresh == 0.25
    assert t.args.track_low_thresh == 0.1


def test_bytetrack_activates_the_first_frame_and_delays_new_tracks_by_one_frame():
    t = _bytetrack()
    assert _ids(t.update(_frame(A))) == [1]  # first-frame activation
    assert _ids(t.update(_frame(A, B))) == [1]  # B is not confirmed yet
    assert _ids(t.update(_frame(A, B))) == [1, 2]


def test_bytetrack_ids_are_per_tracker():
    first, second = _bytetrack(), _bytetrack()
    first.update(_frame(A))
    first.update(_frame(A, B))
    assert _ids(first.update(_frame(A, B))) == [1, 2]
    assert _ids(second.update(_frame(B))) == [1]


def test_bytetrack_keeps_an_id_within_the_track_buffer():
    t = _bytetrack()
    for _ in range(3):
        assert _ids(t.update(_frame(A))) == [1]
    for _ in range(10):
        assert _ids(t.update(_frame())) == []
    assert _ids(t.update(_frame(A))) == [1]


def test_bytetrack_gives_a_new_id_after_the_track_buffer():
    t = _bytetrack()
    for _ in range(3):
        t.update(_frame(A))
    for _ in range(40):
        t.update(_frame())
    assert _ids(t.update(_frame(A))) == []  # a new, unconfirmed track
    assert _ids(t.update(_frame(A))) == [2]


def test_bytetrack_reset_restarts_ids_and_first_frame_activation():
    t = _bytetrack()
    t.update(_frame(A))
    t.update(_frame(A, B))
    t.reset()
    assert t.frame_id == 0
    assert _ids(t.update(_frame(B))) == [1]
