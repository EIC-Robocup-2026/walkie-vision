"""The interface of a pose estimator, the person pose it returns, and the keypoint order."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

# The order of the 17 COCO body keypoints that YOLO26 pose returns, and that
# PersonDetection.keypoints uses.
COCO_KEYPOINTS = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


@dataclass
class PersonPose:
    """One person that a pose estimator found in an image."""

    box: np.ndarray
    """x1, y1, x2, y2 in pixels."""
    score: float
    keypoints: np.ndarray
    """Shape (17, 2): x, y in pixels, in COCO_KEYPOINTS order."""
    keypoint_scores: np.ndarray
    """Shape (17,): the confidence of each keypoint, from 0 to 1."""
    track_id: int = 0
    """The tracker's id, the same from image to image; 0 when not tracked."""


@runtime_checkable
class PoseEstimator(Protocol):
    """A model that finds people and their 17 COCO body keypoints."""

    def estimate(self, image: np.ndarray) -> list[PersonPose]:
        """Return every person in a BGR image."""
        ...
