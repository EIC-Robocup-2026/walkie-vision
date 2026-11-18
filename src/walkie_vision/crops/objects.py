"""Sampled object crops: which detections to sample, and their padded JPEG crops.

The vision node samples a few detections per image (select_samples). A
sampled detection gets a padded JPEG crop, a CLIP vector of the same padded
pixels and the colour shares of its mask (crops.colour). Every other
detection leaves those fields empty, and none of this work runs for it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .. import boxes
from ..providers import Detection
from . import jpeg

SAMPLE_PERIOD = 10
"""A track is eligible once in this many processed images."""
SAMPLE_MAX_OBJECTS = 3
"""The most detections sampled in one image."""
SAMPLE_MIN_SIDE_PX = 32
"""The shortest side a box may have to be sampled, before padding."""
CROP_PADDING = 0.15
"""How much a crop grows on each side, as a share of the box's width and height."""
CROP_MAX_SIDE = 512
"""The longest side of an encoded crop, in pixels. The embedding uses the crop before this reduction."""


def _short_side(box: np.ndarray, width: int, height: int) -> float:
    """The shorter side of a box inside a width x height image, in pixels."""
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float64)
    return float(min(min(x2, width) - max(x1, 0.0), min(y2, height) - max(y1, 0.0)))


def select_samples(detections: Sequence[Detection], counter: int, width: int, height: int) -> list[int]:
    """Return the indices of the detections to sample in one image, best first.

    detections carry their published track ids. counter is the number of
    images processed before this one. A detection is eligible when
    (track_id + counter) % SAMPLE_PERIOD is 0 and its box's shorter side, inside
    the image, is at least SAMPLE_MIN_SIDE_PX. The highest scores win, then the
    lower track id, then the earlier detection.
    """
    eligible = [
        i
        for i, det in enumerate(detections)
        if (int(det.track_id) + counter) % SAMPLE_PERIOD == 0 and _short_side(det.box, width, height) >= SAMPLE_MIN_SIDE_PX
    ]
    eligible.sort(key=lambda i: (-float(detections[i].score), int(detections[i].track_id), i))
    return eligible[:SAMPLE_MAX_OBJECTS]


def usable_samples(detections: Sequence[Detection], width: int, height: int) -> list[int]:
    """Return the indices of every detection whose box is large enough to sample, in order.

    For DetectObjects, which has no period and no limit of its own here: the
    caller asked for this one image.
    """
    return [i for i, det in enumerate(detections) if _short_side(det.box, width, height) >= SAMPLE_MIN_SIDE_PX]


def padded_crop(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    """The pixels of a box grown by CROP_PADDING on each side and clipped to the image. A view, not a copy.

    Raises ValueError when no pixel is left.
    """
    height, width = image.shape[:2]
    x1, y1, x2, y2 = boxes.pad_rect(box, CROP_PADDING, width, height)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"the box {np.asarray(box).tolist()} has no pixels in a {width}x{height} image")
    return image[y1:y2, x1:x2]


def encode_crop(pixels: np.ndarray) -> bytes:
    """Encode padded pixels as a JPEG, reduced to CROP_MAX_SIDE on the longest side, never enlarged."""
    reduced, _, _ = jpeg.scale_down(pixels, CROP_MAX_SIDE)
    return jpeg.encode_jpeg(reduced, jpeg.JPEG_QUALITY)
