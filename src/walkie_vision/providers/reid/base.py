"""The interface of a re-identification model, which embeds the appearance of a person crop."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np


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
