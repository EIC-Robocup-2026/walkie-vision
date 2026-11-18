"""Box and rectangle maths, and the mask resize, on numpy arrays alone.

This is a bottom-layer module: it imports nothing from walkie_vision, so the
person crop, the object crops, the services and the YOLOE provider all build
on it.

A rectangle is (x0, y0, x1, y1) in whole pixels, with x1 and y1 excluded.
A box is x1, y1, x2, y2 in float pixels, as in walkie_vision.providers.

face.heads.box_iou and pipeline._box_iou are separate copies of box_iou that
clamp differently, and stay where they are.
"""

from __future__ import annotations

import math

import numpy as np

Rect = tuple[int, int, int, int]


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two boxes; 0 when the union has no area."""
    inter = _overlap_area(a, b)
    union = box_area(a) + box_area(b) - inter
    return float(inter / union) if union > 0 else 0.0


def box_cover(box: np.ndarray, region: np.ndarray) -> float:
    """The share of region's area that box covers; 0 when region has no area."""
    area = box_area(region)
    return float(_overlap_area(box, region) / area) if area > 0 else 0.0


def outer_rect(box: np.ndarray) -> Rect:
    """The whole pixels that a box touches, also in part."""
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float64)
    return math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)


def pad_box(box: np.ndarray, padding: float) -> np.ndarray:
    """The box grown by padding times its width and height on each side, in float pixels, not clipped.

    A corner can overflow to infinity, with no warning. outer_rect() cannot
    take such a box, so a request's box is checked with this first.
    """
    x1, y1, x2, y2 = (float(v) for v in np.asarray(box, dtype=np.float64))
    grow = float(padding)
    dx, dy = grow * (x2 - x1), grow * (y2 - y1)
    return np.array([x1 - dx, y1 - dy, x2 + dx, y2 + dy])


def pad_rect(box: np.ndarray, padding: float, width: int, height: int) -> Rect:
    """The box grown by padding times its width and height on each side, clipped to the image."""
    x0, y0, x3, y3 = outer_rect(pad_box(box, padding))
    return max(0, x0), max(0, y0), min(width, x3), min(height, y3)


def intersect_rects(a: Rect, b: Rect) -> Rect | None:
    """The overlap of two rectangles; None when they do not overlap."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def subtract_rect(rect: Rect, hole: Rect) -> list[Rect]:
    """rect minus hole, as up to 4 rectangles: above, below, left of and right of the hole."""
    inner = intersect_rects(rect, hole)
    if inner is None:
        return [rect]
    x0, y0, x1, y1 = rect
    hx0, hy0, hx1, hy1 = inner
    out = []
    if hy0 > y0:
        out.append((x0, y0, x1, hy0))
    if hy1 < y1:
        out.append((x0, hy1, x1, y1))
    if hx0 > x0:
        out.append((x0, hy0, hx0, hy1))
    if hx1 < x1:
        out.append((hx1, hy0, x1, hy1))
    return out


def resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Return a 2-D mask at another size, by nearest neighbour on pixel centres.

    Output pixel i takes input pixel floor((i + 0.5) * input / output). An
    equal size returns the mask itself.
    """
    in_height, in_width = mask.shape
    if (in_height, in_width) == (height, width):
        return mask
    rows = ((np.arange(height) + 0.5) * in_height / height).astype(np.intp)
    cols = ((np.arange(width) + 0.5) * in_width / width).astype(np.intp)
    return mask[rows[:, None], cols]


def box_area(box: np.ndarray) -> float:
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float64)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _overlap_area(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = np.asarray(a, dtype=np.float64)
    bx1, by1, bx2, by2 = np.asarray(b, dtype=np.float64)
    return max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
