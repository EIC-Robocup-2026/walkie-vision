"""The interface of an image and text embedder, and the check of its vector length."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from ..vector_tables import EMBEDDING_DIMS


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


def check_model_dim(embedder: Embedder) -> None:
    """Raise ValueError for a real model's name on a vector of another length."""
    expected = EMBEDDING_DIMS.get(embedder.model_name)
    if expected is not None and embedder.dim != expected:
        raise ValueError(f"{embedder.model_name} has {expected} dimensions, not {embedder.dim}")
