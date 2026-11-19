"""The interfaces of a face recogniser, and the face they return.

PersonFaceRecognizer takes the people a pose estimator found, the one import
between two vision tasks' base modules.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from ..pose_estimation.base import PersonPose


@dataclass
class Face:
    """One detected face, in the pixel coordinates of the input image."""

    bbox: np.ndarray
    """x1, y1, x2, y2."""
    kps: np.ndarray
    """Five landmarks, shape (5, 2): eyes, nose tip, mouth corners."""
    det_score: float
    embedding: np.ndarray | None = None
    """The recogniser's embedding, shape (dim,), not normalised."""

    @property
    def normed_embedding(self) -> np.ndarray | None:
        """The embedding scaled to unit length, for cosine similarity."""
        if self.embedding is None:
            return None
        return self.embedding / np.linalg.norm(self.embedding)


@runtime_checkable
class FaceRecognizer(Protocol):
    """A model that finds faces and computes an identity embedding for each."""

    model_name: str
    dim: int

    def recognize(self, image: np.ndarray) -> list[Face]:
        """Return every face in a BGR image, each with its embedding."""
        ...


@runtime_checkable
class PersonFaceRecognizer(Protocol):
    """A face model that looks for each person's face where their pose puts the head.

    It returns one entry per person, in the order of people: the person's
    face, in pixels of the whole image and with its embedding, or None when
    no face was found for that person. No face is given to two people.
    """

    model_name: str
    dim: int

    def faces_for(self, image: np.ndarray, people: Sequence[PersonPose]) -> list[Face | None]:
        """Return the face of each person in a BGR image, or None, in the order of people."""
        ...
