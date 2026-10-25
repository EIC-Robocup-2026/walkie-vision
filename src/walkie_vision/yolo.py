"""YOLOE-26 with ByteTrack, and YOLO26 pose with a tracker, as walkie_vision.providers detectors.

YoloeTracker is an ObjectDetector that looks for text prompts. YoloPoseTracker
is a PoseEstimator; its default tracker is BoT-SORT, whose camera-motion
compensation keeps person ids while the robot turns its base.

Both keep their tracker between calls, and return only tracked boxes, each
with a non-zero track id. In Ultralytics 8.4 a tracker hides a new track
until it is confirmed on the next frame, so a new object or person appears
one processed frame late. The tracker's first frame after a reset is the
exception: it activates every track at once. That is why both classes reset
their tracker after the warm-up.

Ultralytics is imported only inside the constructors, so the pure helpers
below can be imported and tested without it.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .providers import COCO_KEYPOINTS, Detection, PersonPose

# YOLOE.set_classes() loads this text encoder by file name from the working
# directory. It must sit next to the YOLOE weights.
MOBILECLIP_FILE = "mobileclip2_b.ts"
# The tracker files that ship with Ultralytics, by name.
TRACKER_NAMES = ("bytetrack.yaml", "botsort.yaml")


def _clip_box(box: np.ndarray, width: int, height: int) -> np.ndarray | None:
    """Clip an x1, y1, x2, y2 box to the image as float64; None when no area is left."""
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float64)
    x1, x2 = np.clip([x1, x2], 0, width)
    y1, y2 = np.clip([y1, y2], 0, height)
    if x2 <= x1 or y2 <= y1:
        return None
    return np.array([x1, y1, x2, y2], dtype=np.float64)


def _check_rows(n: int, **arrays: np.ndarray | None) -> None:
    for name, array in arrays.items():
        if array is not None and len(array) != n:
            raise ValueError(f"{name} has {len(array)} rows, but there are {n} boxes")


def detections_from_arrays(
    xyxy: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    track_ids: np.ndarray | None,
    names: Sequence[str] | Mapping[int, str],
    image_shape: tuple[int, int],
) -> list[Detection]:
    """Turn a tracker's boxes into Detections.

    Only tracked boxes are returned: no track ids gives [], and a row with a
    track id of 0 or less is dropped. Boxes are clipped to the image, and a
    box with no area left is dropped. image_shape is (height, width).
    """
    if track_ids is None:
        return []
    xyxy = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    _check_rows(len(xyxy), scores=scores, class_ids=class_ids, track_ids=track_ids)
    height, width = image_shape[:2]
    found = []
    for box, score, class_id, track_id in zip(xyxy, scores, class_ids, track_ids):
        if int(track_id) <= 0:
            continue
        clipped = _clip_box(box, width, height)
        if clipped is None:
            continue
        found.append(
            Detection(
                box=clipped,
                label=names[int(class_id)],
                score=float(score),
                track_id=int(track_id),
                mask=None,
            )
        )
    return found


def poses_from_arrays(
    xyxy: np.ndarray,
    scores: np.ndarray,
    keypoints: np.ndarray,
    track_ids: np.ndarray | None,
    image_shape: tuple[int, int],
) -> list[PersonPose]:
    """Turn a pose tracker's boxes and keypoints into PersonPoses.

    keypoints has shape (n, 17, 3): x, y and score. The same rules as in
    detections_from_arrays apply to the boxes. Keypoints are clipped to the
    image too, and are never zeroed, whatever their score.
    """
    kp = np.asarray(keypoints)
    if kp.ndim != 3 or kp.shape[1:] != (len(COCO_KEYPOINTS), 3):
        raise ValueError(f"keypoints must have shape (n, {len(COCO_KEYPOINTS)}, 3), not {kp.shape}")
    if track_ids is None:
        return []
    xyxy = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    _check_rows(len(xyxy), scores=scores, keypoints=kp, track_ids=track_ids)
    height, width = image_shape[:2]
    found = []
    for box, score, points, track_id in zip(xyxy, scores, kp, track_ids):
        if int(track_id) <= 0:
            continue
        clipped = _clip_box(box, width, height)
        if clipped is None:
            continue
        xy = points[:, :2].astype(np.float32)
        xy[:, 0] = np.clip(xy[:, 0], 0, width)
        xy[:, 1] = np.clip(xy[:, 1], 0, height)
        found.append(
            PersonPose(
                box=clipped,
                score=float(score),
                keypoints=xy,
                keypoint_scores=points[:, 2].astype(np.float32),
                track_id=int(track_id),
            )
        )
    return found


def validate_classes(classes: Sequence[str]) -> list[str]:
    """Return a copy of the class names, or raise ValueError.

    The list must not be empty, and every name must be a non-empty string
    with no spaces around it and no duplicate.
    """
    if isinstance(classes, str):
        raise ValueError(f"classes must be a list of names, not the string {classes!r}")
    names = list(classes)
    if not names:
        raise ValueError("classes is empty")
    for name in names:
        if not isinstance(name, str) or not name or name != name.strip():
            raise ValueError(f"class name {name!r} is empty or has spaces around it")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"classes has duplicates: {duplicates}")
    return names


def validate_tracker(tracker: str) -> str:
    """Return the tracker if it is a known file name or an absolute path to a .yaml file, else raise ValueError."""
    if tracker in TRACKER_NAMES:
        return tracker
    path = Path(tracker)
    if path.is_absolute() and path.suffix == ".yaml" and path.is_file():
        return tracker
    raise ValueError(
        f"tracker {tracker!r} must be one of {', '.join(TRACKER_NAMES)} or an absolute path to an existing .yaml file"
    )


def _require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run `pixi run fetch-models`")


def _reset_trackers(model: Any) -> None:
    for tracker in getattr(getattr(model, "predictor", None), "trackers", []):
        tracker.reset()


class YoloeTracker:
    """YOLOE-26 on text prompts, with a tracker: a walkie_vision.providers.ObjectDetector.

    weights is the YOLOE checkpoint; MOBILECLIP_FILE must sit next to it.
    conf is the detector threshold fed to the tracker. 0.1 is Ultralytics'
    own default for track(): the tracker's second matching stage needs
    low-score boxes. The vision node applies its published floor later.
    agnostic_nms keeps one label per box, so an armchair does not come out
    as both chair and armchair. Masks are computed by the seg head and not
    returned.
    """

    def __init__(
        self,
        weights: str | Path,
        classes: Sequence[str],
        *,
        device: str = "cuda:0",
        imgsz: int = 640,
        conf: float = 0.1,
        agnostic_nms: bool = True,
        tracker: str = "bytetrack.yaml",
    ) -> None:
        validate_classes(classes)
        self._tracker = validate_tracker(tracker)
        self._weights = Path(weights).resolve()
        _require(self._weights)
        _require(self._weights.parent / MOBILECLIP_FILE)
        self._device = device
        self._imgsz = imgsz
        self._conf = conf
        self._agnostic_nms = agnostic_nms
        self._classes: list[str] = []

        from ultralytics import YOLOE

        # YOLOE.set_classes() loads its text encoder by file name, relative
        # to the working directory. Next to the weights it finds the file and
        # never downloads it.
        with contextlib.chdir(self._weights.parent):
            self._model = YOLOE(str(self._weights))
        self.set_classes(classes)

    @property
    def classes(self) -> list[str]:
        """The text prompts that detect() looks for, as given."""
        return list(self._classes)

    def set_classes(self, classes: Sequence[str]) -> None:
        """Look for these text prompts, then warm up and reset the tracker.

        Ultralytics drops the predictor, and with it the tracker and its ids,
        which start at 1 again. The vision node sets the classes once, at
        start.
        """
        names = validate_classes(classes)
        with contextlib.chdir(self._weights.parent):
            self._model.set_classes(list(names))
        self._classes = names
        self.detect(np.zeros((self._imgsz, self._imgsz, 3), np.uint8))
        self.reset_tracks()

    def reset_tracks(self) -> None:
        """Clear every track and restart the ids at 1.

        The warm-up used up the tracker's first frame. After the reset, the
        first real image activates its tracks at once.
        """
        _reset_trackers(self._model)

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Return the tracked objects in a BGR image."""
        r = self._model.track(
            image,
            persist=True,
            tracker=self._tracker,
            conf=self._conf,
            imgsz=self._imgsz,
            agnostic_nms=self._agnostic_nms,
            device=self._device,
            verbose=False,
        )[0]
        b = r.boxes
        if b is None or len(b) == 0:
            return []
        return detections_from_arrays(
            b.xyxy.cpu().numpy(),
            b.conf.cpu().numpy(),
            b.cls.int().cpu().numpy(),
            None if b.id is None else b.id.int().cpu().numpy(),
            r.names,
            image.shape[:2],
        )


class YoloPoseTracker:
    """YOLO26 pose with a tracker: a walkie_vision.providers.PoseEstimator.

    The default tracker is BoT-SORT (botsort.yaml): ByteTrack's two-stage
    matching plus global motion compensation from sparse optical flow, with
    no re-ID model. The robot turns its base while it follows one person;
    without the motion compensation a person's box can jump past the
    tracker's IoU gate and get a new id.
    """

    def __init__(
        self,
        weights: str | Path,
        *,
        device: str = "cuda:0",
        imgsz: int = 640,
        conf: float = 0.1,
        tracker: str = "botsort.yaml",
    ) -> None:
        self._tracker = validate_tracker(tracker)
        self._weights = Path(weights).resolve()
        _require(self._weights)
        self._device = device
        self._imgsz = imgsz
        self._conf = conf

        from ultralytics import YOLO

        self._model = YOLO(str(self._weights))
        self.estimate(np.zeros((imgsz, imgsz, 3), np.uint8))
        self.reset_tracks()

    def reset_tracks(self) -> None:
        """Clear every track and restart the ids at 1.

        The warm-up used up the tracker's first frame. After the reset, the
        first real image activates its tracks at once.
        """
        _reset_trackers(self._model)

    def estimate(self, image: np.ndarray) -> list[PersonPose]:
        """Return the tracked people in a BGR image, with their 17 COCO keypoints."""
        r = self._model.track(
            image,
            persist=True,
            tracker=self._tracker,
            conf=self._conf,
            imgsz=self._imgsz,
            device=self._device,
            verbose=False,
        )[0]
        b = r.boxes
        if b is None or len(b) == 0 or r.keypoints is None:
            return []
        return poses_from_arrays(
            b.xyxy.cpu().numpy(),
            b.conf.cpu().numpy(),
            r.keypoints.data.cpu().numpy(),
            None if b.id is None else b.id.int().cpu().numpy(),
            image.shape[:2],
        )
