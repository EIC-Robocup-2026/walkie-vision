"""The colour names and the colour shares of a set of pixels.

COLOR_NAMES is the one place the names and their order are written: the
person crop's shares, GetPersonCrop's colour_names and the object samples'
shares all follow it. It stays one literal assignment, since walkie-database
reads it from the source with ast.literal_eval.
"""

from __future__ import annotations

import cv2
import numpy as np

COLOR_NAMES = (
    "black",
    "white",
    "grey",
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "brown",
    "beige",
)
"""The colour names of the shares, in GetPersonCrop.srv's order: the enum answers for a colour, without unknown."""

# The colour bins, on OpenCV's HSV with the hue in degrees and saturation
# and value from 0 to 1. Black and the greys come first, then the hue
# families, split by saturation and value into pale (beige, pink) and dark
# (brown, purple) shades.
_BLACK_MAX_V = 0.22
_GREY_MAX_S = 0.18
_WHITE_MIN_V = 0.75
_PALE_MAX_S = 0.45
_PALE_MIN_V = 0.60
_BROWN_MAX_V = 0.65
_DARK_PINK_MAX_V = 0.55


def color_indices(pixels: np.ndarray) -> np.ndarray:
    """The index in COLOR_NAMES of each pixel of a uint8 BGR array, flattened."""
    bgr = np.ascontiguousarray(pixels, dtype=np.uint8).reshape(-1, 1, 3)
    if bgr.shape[0] == 0:
        return np.zeros(0, dtype=np.intp)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float64)
    h, s, v = hsv[:, 0] * 2.0, hsv[:, 1] / 255.0, hsv[:, 2] / 255.0
    red = (h < 15) | (h >= 345)
    orange = (h >= 15) & (h < 45)
    yellow = (h >= 45) & (h < 70)
    green = (h >= 70) & (h < 165)
    blue = (h >= 165) & (h < 260)
    purple = (h >= 260) & (h < 300)
    magenta = (h >= 300) & (h < 345)
    pale = (s < _PALE_MAX_S) & (v >= _PALE_MIN_V)
    dark = v < _BROWN_MAX_V
    index = {name: i for i, name in enumerate(COLOR_NAMES)}
    # np.select takes the first condition that holds.
    rules = [
        (v < _BLACK_MAX_V, "black"),
        ((s < _GREY_MAX_S) & (v >= _WHITE_MIN_V), "white"),
        (s < _GREY_MAX_S, "grey"),
        (red & pale, "pink"),
        (red, "red"),
        (orange & dark, "brown"),
        (orange & (s < _PALE_MAX_S), "beige"),
        (orange, "orange"),
        (yellow & (s < _PALE_MAX_S), "beige"),
        (yellow & dark, "brown"),
        (yellow, "yellow"),
        (green, "green"),
        (blue, "blue"),
        (purple, "purple"),
        (magenta & (v < _DARK_PINK_MAX_V), "purple"),
    ]
    return np.select([c for c, _ in rules], [index[n] for _, n in rules], default=index["pink"])


def color_shares(pixels: np.ndarray) -> np.ndarray:
    """The share of each colour in COLOR_NAMES among a non-empty set of uint8 BGR pixels; the shares sum to 1."""
    indices = color_indices(pixels)
    if indices.size == 0:
        raise ValueError("no pixels to count")
    return np.bincount(indices, minlength=len(COLOR_NAMES)).astype(np.float64) / indices.size
