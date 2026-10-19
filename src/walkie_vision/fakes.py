"""Fakes of every provider in walkie_vision.providers, for tests without a GPU.

They load no model and return results that depend only on their input, so
a test gives the same result on every computer. The detectors return the
results they were given; the embedders return hash_vector() of the input
bytes, so equal inputs give equal vectors and different inputs give vectors
that are nearly orthogonal.

The fakes' model names start with fake_. No walkie-database vector table has
such a name, so a fake's vectors cannot be stored next to a real model's.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Sequence

import numpy as np

from .providers import Detection, Face, PersonPose


def hash_vector(data: bytes | str, dim: int) -> np.ndarray:
    """Return a unit-length float32 vector of length dim that depends only on data."""
    if isinstance(data, str):
        data = data.encode()
    seed = int.from_bytes(hashlib.sha256(data).digest()[:8], "little")
    vector = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return vector / np.linalg.norm(vector)


def _image_key(image: np.ndarray) -> bytes:
    # The shape is part of the key: two images with the same bytes but a
    # different shape are different images.
    return repr(image.shape).encode() + np.ascontiguousarray(image).tobytes()


def _clip_box(box: np.ndarray, image: np.ndarray) -> np.ndarray | None:
    """Clip an x1, y1, x2, y2 box to the image; None when nothing is left."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float64)
    x1, x2 = np.clip([x1, x2], 0, width)
    y1, y2 = np.clip([y1, y2], 0, height)
    if x2 <= x1 or y2 <= y1:
        return None
    return np.array([x1, y1, x2, y2])


class FakeObjectDetector:
    """Returns the given objects whose label is one of the current classes.

    Boxes are clipped to the image, and an object that lies outside the image
    is left out. Track ids are returned as given.
    """

    def __init__(self, objects: Sequence[Detection], classes: Sequence[str] = ()) -> None:
        self.objects = list(objects)
        self._classes = list(classes)

    @property
    def classes(self) -> list[str]:
        return list(self._classes)

    def set_classes(self, classes: Sequence[str]) -> None:
        self._classes = list(classes)

    def detect(self, image: np.ndarray) -> list[Detection]:
        found = []
        for obj in self.objects:
            if obj.label not in self._classes:
                continue
            box = _clip_box(obj.box, image)
            if box is None:
                continue
            det = copy.deepcopy(obj)
            det.box = box
            found.append(det)
        return found


class FakePoseEstimator:
    """Returns copies of the given people for every image."""

    def __init__(self, people: Sequence[PersonPose]) -> None:
        self.people = list(people)

    def estimate(self, image: np.ndarray) -> list[PersonPose]:
        return copy.deepcopy(self.people)


class FakeEmbedder:
    """Embeds images and texts as hash vectors of their bytes."""

    def __init__(self, model_name: str = "fake_embedder", dim: int = 512) -> None:
        self.model_name = model_name
        self.dim = dim

    def embed_images(self, images: Sequence[np.ndarray]) -> np.ndarray:
        return self._stack([b"image " + _image_key(image) for image in images])

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        return self._stack([b"text " + text.encode() for text in texts])

    def _stack(self, keys: list[bytes]) -> np.ndarray:
        if not keys:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([hash_vector(key, self.dim) for key in keys])


class FakeReIdentifier:
    """Embeds person crops as hash vectors of their bytes."""

    def __init__(self, model_name: str = "fake_reid", dim: int = 512) -> None:
        self.model_name = model_name
        self.dim = dim

    def embed(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([hash_vector(_image_key(crop), self.dim) for crop in crops])


class FakeFaceRecognizer:
    """Returns copies of the given faces for every image.

    A face given without an embedding gets the hash vector of its box, so the
    same face gives the same embedding in every image.
    """

    def __init__(self, faces: Sequence[Face], model_name: str = "fake_face", dim: int = 512) -> None:
        self.faces = list(faces)
        self.model_name = model_name
        self.dim = dim

    def recognize(self, image: np.ndarray) -> list[Face]:
        faces = copy.deepcopy(self.faces)
        for face in faces:
            if face.embedding is None:
                key = np.asarray(face.bbox, dtype=np.float64).tobytes()
                face.embedding = hash_vector(b"face " + key, self.dim)
        return faces
