"""YOLOE-26 with and without a tracker, as walkie_vision.providers object detectors.

YoloeTracker is an ObjectDetector that looks for text prompts. It keeps its
tracker between calls and returns only tracked boxes, one processed frame
late for a new object; walkie_vision.providers._ultralytics says why, and why
it resets its tracker after the warm-up.

YoloeDetector is a second, separate YOLOE for DetectObjects. It has no tracker,
and its classes change with the requests, so the stream's classes and track ids
never do.

Ultralytics is imported only inside the constructors, so the pure helpers
below can be imported and tested without it.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from ...boxes import resize_mask
from .._ultralytics import check_rows, clip_box, require, reset_trackers, validate_tracker
from .base import Detection

# YOLOE.set_classes() loads this text encoder by file name from the working
# directory. It must sit next to the YOLOE weights.
MOBILECLIP_FILE = "mobileclip2_b.ts"


def mask_to_native(mask: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Return a mask at the letterboxed inference size as a bool array the size of the image.

    Removes the padding by the rule of ultralytics.utils.ops.scale_masks, then
    resizes by nearest neighbour. A mask of the image's size only becomes bool.
    image_shape is (height, width).
    """
    height, width = image_shape[:2]
    binary = np.asarray(mask) > 0.5
    in_height, in_width = binary.shape
    if (in_height, in_width) == (height, width):
        return binary
    gain = min(in_height / height, in_width / width)
    pad_width, pad_height = (in_width - round(width * gain)) / 2, (in_height - round(height * gain)) / 2
    top, left = round(pad_height - 0.1), round(pad_width - 0.1)
    bottom, right = top + round(height * gain), left + round(width * gain)
    return resize_mask(binary[top:bottom, left:right], height, width)


def detections_from_arrays(
    xyxy: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    track_ids: np.ndarray | None,
    names: Sequence[str] | Mapping[int, str],
    image_shape: tuple[int, int],
    masks: np.ndarray | None = None,
) -> list[Detection]:
    """Turn a tracker's boxes into Detections.

    Only tracked boxes are returned: no track ids gives [], and a row with a
    track id of 0 or less is dropped. Boxes are clipped to the image, and a
    box with no area left is dropped. image_shape is (height, width).

    masks, when given, has one row per box at the inference size. A kept box
    takes the mask of its own row, so dropping a row cannot shift a mask to a
    neighbouring box.
    """
    if track_ids is None:
        return []
    xyxy = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    check_rows(len(xyxy), scores=scores, class_ids=class_ids, track_ids=track_ids, masks=masks)
    height, width = image_shape[:2]
    found = []
    for row, (box, score, class_id, track_id) in enumerate(zip(xyxy, scores, class_ids, track_ids)):
        if int(track_id) <= 0:
            continue
        clipped = clip_box(box, width, height)
        if clipped is None:
            continue
        found.append(
            Detection(
                box=clipped,
                label=names[int(class_id)],
                score=float(score),
                track_id=int(track_id),
                mask=None if masks is None else mask_to_native(masks[row], (height, width)),
            )
        )
    return found


def untracked_detections_from_arrays(
    xyxy: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    names: Sequence[str] | Mapping[int, str],
    image_shape: tuple[int, int],
    masks: np.ndarray | None = None,
) -> list[Detection]:
    """Turn the boxes of a detector with no tracker into Detections with track id 0.

    Otherwise the rules of detections_from_arrays apply: boxes are clipped, a
    box with no area left is dropped, and a kept box takes the mask of its own
    row.
    """
    xyxy = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    check_rows(len(xyxy), scores=scores, class_ids=class_ids, masks=masks)
    height, width = image_shape[:2]
    found = []
    for row, (box, score, class_id) in enumerate(zip(xyxy, scores, class_ids)):
        clipped = clip_box(box, width, height)
        if clipped is None:
            continue
        found.append(
            Detection(
                box=clipped,
                label=names[int(class_id)],
                score=float(score),
                track_id=0,
                mask=None if masks is None else mask_to_native(masks[row], (height, width)),
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


class YoloeTracker:
    """YOLOE-26 on text prompts, with a tracker: a walkie_vision.providers.ObjectDetector.

    weights is the YOLOE checkpoint; MOBILECLIP_FILE must sit next to it.
    conf is the detector threshold fed to the tracker. 0.1 is Ultralytics'
    own default for track(): the tracker's second matching stage needs
    low-score boxes. The vision node applies its published floor later.
    agnostic_nms keeps one label per box, so an armchair does not come out
    as both chair and armchair. The seg head always computes masks. Only with
    keep_masks are they copied to the CPU and returned, at the image's size.
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
        keep_masks: bool = False,
    ) -> None:
        validate_classes(classes)
        self._tracker = validate_tracker(tracker)
        self._weights = Path(weights).resolve()
        require(self._weights)
        require(self._weights.parent / MOBILECLIP_FILE)
        self._device = device
        self._imgsz = imgsz
        self._conf = conf
        self._agnostic_nms = agnostic_nms
        self._keep_masks = keep_masks
        self._classes: list[str] = []
        self.last_mask_ms = 0.0
        """How long the last detect() took after the model to copy its masks and boxes to the CPU and resize each mask.

        0 when it kept no mask. The box copy is small next to the masks.
        """

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
        reset_trackers(self._model)

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
        self.last_mask_ms = 0.0
        b = r.boxes
        if b is None or len(b) == 0:
            return []
        started = time.perf_counter()
        # Ultralytics filters a result's boxes and masks together, by row, so
        # these rows still line up after the tracker dropped some.
        masks = r.masks.data.cpu().numpy() if self._keep_masks and r.masks is not None else None
        found = detections_from_arrays(
            b.xyxy.cpu().numpy(),
            b.conf.cpu().numpy(),
            b.cls.int().cpu().numpy(),
            None if b.id is None else b.id.int().cpu().numpy(),
            r.names,
            image.shape[:2],
            masks,
        )
        if masks is not None:
            self.last_mask_ms = (time.perf_counter() - started) * 1000.0
        return found


class YoloeDetector:
    """YOLOE-26 on text prompts with no tracker: a walkie_vision.providers.ObjectDetector for DetectObjects.

    This is its own model object next to the stream's YoloeTracker. weights is
    the YOLOE checkpoint; MOBILECLIP_FILE must sit next to it. Every detection
    has track id 0. conf is the detector threshold: with no tracker, nothing
    needs the low-score boxes. max_det bounds the boxes and the masks that are
    copied to the CPU. Only with keep_masks are masks returned, at the image's
    size.

    The node builds this at start only with on_demand_enabled, and warms it up
    there, so a first request does not pay for the load.
    """

    def __init__(
        self,
        weights: str | Path,
        classes: Sequence[str],
        *,
        device: str = "cuda:0",
        imgsz: int = 640,
        conf: float = 0.25,
        agnostic_nms: bool = True,
        max_det: int = 20,
        keep_masks: bool = False,
    ) -> None:
        validate_classes(classes)
        self._weights = Path(weights).resolve()
        require(self._weights)
        require(self._weights.parent / MOBILECLIP_FILE)
        self._device = device
        self._imgsz = imgsz
        self._conf = conf
        self._agnostic_nms = agnostic_nms
        self._max_det = max_det
        self._keep_masks = keep_masks
        self._classes: list[str] = []
        self.last_mask_ms = 0.0
        """As YoloeTracker.last_mask_ms: the copy and resize after the model, 0 when no mask was kept."""

        from ultralytics import YOLOE

        with contextlib.chdir(self._weights.parent):
            self._model = YOLOE(str(self._weights))
        self.set_classes(classes)

    @property
    def classes(self) -> list[str]:
        """The text prompts that detect() looks for, as given. Empty after a failed set_classes()."""
        return list(self._classes)

    def set_classes(self, classes: Sequence[str]) -> None:
        """Look for these text prompts, then warm up, so the next detect() starts at once.

        A failure can leave the model between two lists. The classes then stay
        empty, so the next request prepares its own list again.
        """
        names = validate_classes(classes)
        self._classes = []
        with contextlib.chdir(self._weights.parent):
            self._model.set_classes(list(names))
        self.detect(np.zeros((self._imgsz, self._imgsz, 3), np.uint8))
        self._classes = names

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Return the objects in a BGR image, each with track id 0."""
        r = self._model.predict(
            image,
            conf=self._conf,
            imgsz=self._imgsz,
            agnostic_nms=self._agnostic_nms,
            max_det=self._max_det,
            device=self._device,
            verbose=False,
        )[0]
        self.last_mask_ms = 0.0
        b = r.boxes
        if b is None or len(b) == 0:
            return []
        started = time.perf_counter()
        masks = r.masks.data.cpu().numpy() if self._keep_masks and r.masks is not None else None
        found = untracked_detections_from_arrays(
            b.xyxy.cpu().numpy(),
            b.conf.cpu().numpy(),
            b.cls.int().cpu().numpy(),
            r.names,
            image.shape[:2],
            masks,
        )
        if masks is not None:
            self.last_mask_ms = (time.perf_counter() - started) * 1000.0
        return found
