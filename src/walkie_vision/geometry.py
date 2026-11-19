"""Object geometry from registered depth: lifting a mask to points and scoring a view (ADR-0014).

lift_mask() turns one object's mask into its points in the optical frame, and
view_quality() scores a view for best-view selection. Vision stops there: no
map transform, no clustering, no plane removal, no box fit.

Imports no model code, so the tests run it without a GPU.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .camera.depth import RegisteredDepth
from .convert import MAX_CLOUD_POINTS, MIN_CLOUD_POINTS, evenly_spaced_indices

# The erosion amount: a share of the mask's smaller extent, within these bounds.
_EROSION_SHARE = 0.05
_MIN_EROSION_PX = 1
_MAX_EROSION_PX = 5
# Three of these MADs make about three standard deviations of normal noise.
_MAD_TO_SIGMA = 1.4826
_MAD_WINDOW_SIGMAS = 3.0
_MIN_DEPTH_WINDOW_M = 0.01
"""The floor of the depth window, so that a zero MAD cannot reject small valid noise."""

# The view quality recipe. Initial values, not a calibrated probability.
_FULL_AREA_SHARE = 0.10
_FULL_SHARPNESS_VARIANCE = 1000.0
_MIN_SHARPNESS_PIXELS = 50


def erosion_radius(mask: np.ndarray) -> int:
    """Return the erosion amount for a bool mask in depth pixels; 0 for an empty mask.

    5% of the smaller extent of the mask, within 1 to 5. Python's round()
    sends a value that ends in .5 to the even side, so 2.5 gives 2.
    """
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0:
        return 0
    extent = min(int(rows[-1] - rows[0]) + 1, int(cols[-1] - cols[0]) + 1)
    return min(_MAX_EROSION_PX, max(_MIN_EROSION_PX, round(_EROSION_SHARE * extent)))


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    """Erode a bool mask with a square kernel. Outside the raster counts as not in the mask."""
    side = 2 * radius + 1
    kernel = np.ones((side, side), np.uint8)
    # The default border value would keep a mask that touches the edge.
    return cv2.erode(mask.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)


def _in_range(values: np.ndarray, min_m: float, max_m: float) -> np.ndarray:
    return np.isfinite(values) & (values >= min_m) & (values <= max_m)


def _surviving_pixels(
    mask: np.ndarray, metres: np.ndarray, min_m: float, max_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the rows, columns and depths of a mask's pixels that pass the depth rules, in row-major order.

    A depth must be finite, in range and near the median, within
    max(3 * 1.4826 * MAD, 1 cm).
    """
    rows, cols = np.nonzero(mask)
    z = metres[rows, cols]
    valid = _in_range(z, min_m, max_m)
    rows, cols, z = rows[valid], cols[valid], z[valid]
    if z.size == 0:
        return rows, cols, z
    median = np.median(z)
    mad = float(np.median(np.abs(z - median)))
    near = np.abs(z - median) <= max(_MAD_WINDOW_SIGMAS * _MAD_TO_SIGMA * mad, _MIN_DEPTH_WINDOW_M)
    return rows[near], cols[near], z[near]


def lift_mask(mask: np.ndarray, depth: RegisteredDepth, min_m: float, max_m: float) -> np.ndarray:
    """Return a mask's points as an N x 3 float32 array of x, y and z in the optical frame.

    mask is a bool array the size of the depth raster. The eroded mask comes
    first. When fewer than MIN_CLOUD_POINTS pixels survive it, the whole mask
    is tried by the same rules. When still fewer survive, the result is empty.
    At most MAX_CLOUD_POINTS pixels are kept, at evenly spaced indices in
    row-major order, so the same input gives the same points.
    """
    radius = erosion_radius(mask)
    if radius == 0:
        return np.empty((0, 3), np.float32)
    rows, cols, z = _surviving_pixels(_erode(mask, radius), depth.metres, min_m, max_m)
    if z.size < MIN_CLOUD_POINTS:
        rows, cols, z = _surviving_pixels(mask, depth.metres, min_m, max_m)
        if z.size < MIN_CLOUD_POINTS:
            return np.empty((0, 3), np.float32)
    chosen = evenly_spaced_indices(z.size, MAX_CLOUD_POINTS)
    rows, cols, z = rows[chosen], cols[chosen], z[chosen].astype(np.float64)
    x = (cols - depth.cx) * z / depth.fx
    y = (rows - depth.cy) * z / depth.fy
    return np.column_stack((x, y, z)).astype(np.float32)


def depth_cover(mask: np.ndarray, metres: np.ndarray, min_m: float, max_m: float) -> float:
    """Return the share of a mask's pixels with a valid depth in range, before erosion."""
    pixels = int(np.count_nonzero(mask))
    if pixels == 0:
        return 0.0
    return int(np.count_nonzero(_in_range(metres[mask], min_m, max_m))) / pixels


def frame_laplacian(image: np.ndarray) -> np.ndarray:
    """Return the Laplacian of a BGR image's grey values. Compute it once per image, not per detection."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(grey, cv2.CV_32F, ksize=3)


def view_quality(mask: np.ndarray, box: np.ndarray, laplacian: np.ndarray, cover: float) -> float:
    """Score one view for best-view selection, in [0, 1].

    The mean of four terms: the mask's area, how centred the box is, cover
    (depth_cover(), 0 without depth) and the mask's sharpness. mask is a bool
    array, mask and laplacian have the image's size, and box is x1, y1, x2, y2
    in its pixels. An empty mask scores 0.

    These are initial formulas, not a calibrated probability. Change them only
    on recordings, with a versioned recipe.
    """
    pixels = int(np.count_nonzero(mask))
    if pixels == 0:
        return 0.0
    height, width = mask.shape
    area = min(1.0, math.sqrt(pixels / (_FULL_AREA_SHARE * width * height)))
    x1, y1, x2, y2 = (float(v) for v in box)
    half_width, half_height = width / 2, height / 2
    off_centre = math.hypot(((x1 + x2) / 2 - half_width) / half_width, ((y1 + y2) / 2 - half_height) / half_height)
    centred = max(0.0, 1.0 - off_centre / math.sqrt(2.0))
    sharpness = 0.0
    if pixels >= _MIN_SHARPNESS_PIXELS:
        sharpness = min(1.0, float(np.var(laplacian[mask], dtype=np.float64)) / _FULL_SHARPNESS_VARIANCE)
    return min(1.0, max(0.0, 0.25 * (area + centred + cover + sharpness)))
