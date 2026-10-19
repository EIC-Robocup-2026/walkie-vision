"""The interfaces of the vision models, and the results they return.

The ROS nodes (not written yet) will call each model only through one of
these Protocols. A test passes a fake from walkie_vision.fakes instead, and a
model can be replaced without a change to the node that calls it.

Conventions for every provider:

- An image is an H x W x 3 uint8 numpy array in BGR order, as cv2 and
  cv_bridge return it.
- A box is (x1, y1, x2, y2) in pixels of that image, as a float array.
- embed_* methods return a float32 array of shape (n, dim): one row per
  input, in the order of the inputs. The rows need not have unit length:
  walkie_vision.convert normalises every vector that goes into a message.
- model_name is the <model> part of walkie-database's vector table names
  (for example arcface_w600k_r50 for face_arcface_w600k_r50). It names the
  weights, not only the network, so that vectors from two sets of weights
  are never compared. A model with several output lengths includes the
  length in the name.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

# The model names of walkie-database's vector tables (its ADR-0007).
# room_clip_vit_b16 and object_clip_vit_b16 name only the network; there is
# one set of CLIP ViT-B/16 weights in use, OpenAI's. The face and appearance
# names follow walkie-database's migration 0002; before it, its tables are
# face_arcface and appearance_osnet.
CLIP_VIT_B16 = "clip_vit_b16"
ARCFACE_W600K_R50 = "arcface_w600k_r50"
OSNET_X1_0_MSMT17 = "osnet_x1_0_msmt17"

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
    """The object's pixels: a bool array the size of the image, or None."""


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


@runtime_checkable
class Embedder(Protocol):
    """A model that embeds images and texts into one vector space.

    An image's vector and a text's vector are comparable by cosine
    similarity, so that a text such as "a red mug" can find object images.
    """

    model_name: str
    dim: int

    def embed_images(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """Return one row per BGR image (or crop), shape (len(images), dim)."""
        ...

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Return one row per text, shape (len(texts), dim)."""
        ...


@runtime_checkable
class PoseEstimator(Protocol):
    """A model that finds people and their 17 COCO body keypoints."""

    def estimate(self, image: np.ndarray) -> list[PersonPose]:
        """Return every person in a BGR image."""
        ...


@runtime_checkable
class ReIdentifier(Protocol):
    """A model that embeds the appearance of a person crop.

    Two crops of the same person, for example from behind and from the side,
    give similar vectors, also when the face is not visible.
    """

    model_name: str
    dim: int

    def embed(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Return one row per BGR person crop, shape (len(crops), dim)."""
        ...


@runtime_checkable
class FaceRecognizer(Protocol):
    """A model that finds faces and computes an identity embedding for each."""

    model_name: str
    dim: int

    def recognize(self, image: np.ndarray) -> list[Face]:
        """Return every face in a BGR image, each with its embedding."""
        ...
