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

import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

byte_tracker = pytest.importorskip("ultralytics.trackers.byte_tracker")

import ultralytics  # noqa: E402
from ultralytics.engine.results import Boxes, Keypoints  # noqa: E402

from walkie_vision.providers import ObjectDetector, PoseEstimator  # noqa: E402
from walkie_vision.yolo import (  # noqa: E402
    MOBILECLIP_FILE,
    TRACKER_NAMES,
    YoloeTracker,
    YoloPoseTracker,
    detections_from_arrays,
    poses_from_arrays,
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


def test_importing_the_module_loads_nothing_heavy():
    # The node takes MOBILECLIP_FILE from walkie_vision.yolo before it
    # imports torch or Ultralytics. A fresh interpreter, because this one has
    # imported them already.
    code = (
        "import sys, walkie_vision.yolo; "
        "print(' '.join(m for m in ('torch', 'ultralytics', 'onnxruntime') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=60)
    assert out.stdout.strip() == ""
