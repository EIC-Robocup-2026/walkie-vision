"""The colour names and the colour shares of a set of pixels.

COLOR_NAMES is the one place the names and their order are written: the
person crop's shares, GetPersonCrop's colour_names and the object samples'
shares all follow it. It stays one literal assignment, since walkie-database
reads it from the source with ast.literal_eval.

Two recipes count the shares. A person's torso and thighs use the HSV colour
bins (color_shares). A sampled object's mask uses colour recipe v1, the
nearest Lab palette centre (mask_color_shares).
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


MIN_COLOR_PIXELS = 50
"""Fewer mask pixels than this give no colour shares."""

# Colour recipe v1: RGB, in the order of crops.colour.COLOR_NAMES. This is deterministic
# plumbing, not colour constancy. Its accuracy has not been measured.
PALETTE_RGB = (
    (0, 0, 0),
    (255, 255, 255),
    (128, 128, 128),
    (220, 35, 35),
    (245, 130, 25),
    (245, 220, 35),
    (45, 155, 65),
    (40, 90, 205),
    (135, 65, 175),
    (235, 145, 180),
    (115, 75, 45),
    (210, 190, 145),
)
_PALETTE_LAB = cv2.cvtColor(np.array(PALETTE_RGB, dtype=np.float32).reshape(-1, 1, 3) / 255.0, cv2.COLOR_RGB2Lab).reshape(-1, 3)


def nearest_palette_indices(lab: np.ndarray, centres: np.ndarray = _PALETTE_LAB) -> np.ndarray:
    """The index of the nearest Lab centre of each Lab pixel, by Euclidean distance. A tie takes the earlier centre.

    Compares one centre at a time, so a large mask does not need a pixels x
    centres array.
    """
    lab = np.asarray(lab, dtype=np.float32).reshape(-1, 3)
    best = np.full(lab.shape[0], np.inf, dtype=np.float32)
    indices = np.zeros(lab.shape[0], dtype=np.intp)
    for i, centre in enumerate(centres):
        distance = np.sum((lab - centre) ** 2, axis=1)
        nearer = distance < best
        best[nearer] = distance[nearer]
        indices[nearer] = i
    return indices


def mask_color_shares(image: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """The shares of the palette colours among a mask's pixels of a BGR image; None under MIN_COLOR_PIXELS.

    Uses the original mask and pixels: no padding, resizing or JPEG. OpenCV
    converts float32 sRGB to CIELAB. The shares follow crops.colour.COLOR_NAMES and sum
    to 1.
    """
    pixels = image[np.asarray(mask, dtype=bool)]
    count = pixels.shape[0]
    if count < MIN_COLOR_PIXELS:
        return None
    lab = cv2.cvtColor(pixels.reshape(-1, 1, 3).astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    counts = np.bincount(nearest_palette_indices(lab), minlength=len(PALETTE_RGB))
    return counts.astype(np.float64) / count
