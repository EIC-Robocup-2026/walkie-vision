"""The interface of an object detector, and the detection it returns."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class Detection:
    """One object that a detector found in an image."""

    box: np.ndarray
    """x1, y1, x2, y2 in pixels."""
    label: str
    """The class name that matched: one of the detector's classes."""
    score: float
    track_id: int = 0
    """The tracker's id, the same from image to image; 0 when not tracked."""
    mask: np.ndarray | None = None
    """This box's own pixels: a bool array the size of the image. None unless the detector was asked for masks."""


@runtime_checkable
class ObjectDetector(Protocol):
    """An open-vocabulary detector that looks for the classes it is given.

    A detector that tracks keeps its tracks between calls to detect() and
    sets Detection.track_id; set_classes() may reset them.
    """

    @property
    def classes(self) -> list[str]:
        """The class names that detect() looks for."""
        ...

    def set_classes(self, classes: Sequence[str]) -> None:
        """Look for these class names (text prompts) from the next image on."""
        ...

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Return every object of one of the classes in a BGR image."""
        ...
