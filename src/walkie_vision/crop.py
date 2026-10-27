"""One person's crop for describing them, with the colour shares of their top and bottom.

The vision node answers GetPersonCrop with crop_person(), on the image it
has just processed and the people it found in it (ADR-0011). Everything
here is pure: numpy and OpenCV on one image, with no ROS and no model, so
the tests run it on synthetic figures.

crop_person() works in this order:

- It finds the track among the image's people, or answers NOT_IN_VIEW.
- It answers OCCLUDED only for a person in front. Box overlap alone cannot
  tell front from behind, so only a nearer person counts (is_nearer()): one
  whose box ends lower in the image, because their feet are closer to the
  camera. A nearer person is in the way when their box has an IoU above
  OCCLUDED_IOU with this person's box, or covers more than
  OCCLUDED_TORSO_COVER of this person's torso box. The second test catches
  a smaller person in front, whose IoU stays low. People behind are only
  masked.
- It answers TOO_SMALL for a box under MIN_BOX_HEIGHT_PX tall.
- It cuts the padded box out of the image, and greys every other person's
  box inside it, minus this person's own box (mask_rects()). So this
  person's pixels are never greyed. Another person's pixels inside this
  person's box, and people the detector missed, stay in the crop.
- It computes the colour shares of the torso and the thighs on the
  full-resolution pixels, before the crop is scaled and compressed. A
  region is gated, with no shares, when one of its four keypoints scores
  under min_keypoint_score. There is no fallback to a share of the box.
- It scales the crop down to max_side and encodes it as a JPEG.

A rectangle is (x0, y0, x1, y1) in whole pixels, with x1 and y1 excluded.
A box is x1, y1, x2, y2 in float pixels, as in walkie_vision.providers.

The thresholds are first values, not measured. They are to be tuned on
photos from the robot (docs/architecture/vision.md, "Person crops").
"""

from __future__ import annotations

import enum
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np

from .providers import COCO_KEYPOINTS, PersonPose

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

MIN_BOX_HEIGHT_PX = 160.0
"""A box under this many pixels tall is TOO_SMALL."""
NEARER_MARGIN = 0.03
"""Another person is nearer when their box ends lower by at least this share of the image height."""
BOTTOM_EDGE_PX = 1.0
"""A box that ends this close to the image's bottom edge is cut off by it."""
OCCLUDED_IOU = 0.3
"""OCCLUDED when a nearer person's box has an IoU above this with the person's box."""
OCCLUDED_TORSO_COVER = 0.3
"""OCCLUDED when a nearer person's box covers more than this share of the person's torso box."""
TORSO_SHRINK = 0.2
"""The torso polygon shrinks by this share towards its centre, away from the arms and the background."""
MIN_REGION_PIXELS = 50
"""A region with fewer pixels left gets no shares: a side view can make its polygon a sliver."""
MASK_GREY = 128
"""The value of every channel of a greyed pixel."""
JPEG_QUALITY = 90

_KEYPOINT = {name: i for i, name in enumerate(COCO_KEYPOINTS)}
TORSO = tuple(_KEYPOINT[n] for n in ("left_shoulder", "right_shoulder", "right_hip", "left_hip"))
"""The torso polygon's corners, in order around it."""
THIGHS = tuple(_KEYPOINT[n] for n in ("left_hip", "right_hip", "right_knee", "left_knee"))
"""The thighs polygon's corners, in order around it."""

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


class CropStatus(enum.IntEnum):
    """GetPersonCrop's statuses, with the values of the constants on GetPersonCrop.Response."""

    OK = 0
    NOT_IN_VIEW = 1
    OCCLUDED = 2
    TOO_SMALL = 3
    TIMEOUT = 4
    NO_IMAGE = 5
    FAILED = 6


@dataclass
class PersonCrop:
    """What crop_person() found. The crop fields are set only when status is OK."""

    status: CropStatus
    message: str = ""
    index: int = -1
    """The person's index among the image's people; -1 when the track is not there."""
    max_overlap: float = 0.0
    torso_cover: float = 0.0
    pixels: np.ndarray | None = None
    """The crop as sent: greyed and scaled, H x W x 3 uint8 BGR."""
    jpeg: bytes = b""
    keypoints: np.ndarray | None = None
    """Shape (17, 2): the person's keypoints in pixels of the crop."""
    masked: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    """Shape (n, 4): the greyed rectangles, x0, y0, x1, y1, in pixels of the crop."""
    top_shares: np.ndarray | None = None
    """One share per name in COLOR_NAMES, or None when the region is gated."""
    bottom_shares: np.ndarray | None = None
    top_left: float = 0.0
    bottom_left: float = 0.0


Rect = tuple[int, int, int, int]


def check_options(padding: float, max_side: int, min_keypoint_score: float) -> None:
    """Raise ValueError naming the first request option that is out of range."""
    if not (math.isfinite(padding) and 0.0 <= padding <= 1.0):
        raise ValueError(f"padding must be from 0 to 1, not {padding}")
    if max_side < 1:
        raise ValueError(f"max_side must be at least 1, not {max_side}")
    if not (math.isfinite(min_keypoint_score) and 0.0 <= min_keypoint_score <= 1.0):
        raise ValueError(f"min_keypoint_score must be from 0 to 1, not {min_keypoint_score}")


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two boxes; 0 when the union has no area."""
    inter = _overlap_area(a, b)
    union = _area(a) + _area(b) - inter
    return float(inter / union) if union > 0 else 0.0


def box_cover(box: np.ndarray, region: np.ndarray) -> float:
    """The share of region's area that box covers; 0 when region has no area."""
    area = _area(region)
    return float(_overlap_area(box, region) / area) if area > 0 else 0.0


def torso_box(person: PersonPose) -> np.ndarray | None:
    """The box around the shoulder and hip keypoints, clipped to the person's box; None when it has no area.

    The keypoints are used whatever their scores: a hidden keypoint's
    position is the pose model's guess, which is good enough for a box.
    """
    points = np.asarray(person.keypoints, dtype=np.float64)[list(TORSO)]
    x1, y1, x2, y2 = np.asarray(person.box, dtype=np.float64)
    box = np.array(
        [
            np.clip(points[:, 0].min(), x1, x2),
            np.clip(points[:, 1].min(), y1, y2),
            np.clip(points[:, 0].max(), x1, x2),
            np.clip(points[:, 1].max(), y1, y2),
        ]
    )
    return box if _area(box) > 0 else None


def is_nearer(other: PersonPose, person: PersonPose, image_height: int, min_keypoint_score: float) -> bool:
    """True when other stands nearer to the camera than person.

    - The person whose box ends lower in the image is nearer, by at least
      NEARER_MARGIN of the image height.
    - A box cut off by the image's bottom edge is nearer than one that is
      not.
    - When both are cut off, other is nearer only when one of person's
      torso keypoints inside other's box scores under min_keypoint_score:
      other hides it.
    """
    edge = image_height - BOTTOM_EDGE_PX
    other_bottom, own_bottom = float(other.box[3]), float(person.box[3])
    other_cut, own_cut = other_bottom >= edge, own_bottom >= edge
    if other_cut and own_cut:
        keypoints = np.asarray(person.keypoints, dtype=np.float64)
        scores = np.asarray(person.keypoint_scores, dtype=np.float64)
        return any(
            _contains(other.box, keypoints[i]) and scores[i] < min_keypoint_score for i in TORSO
        )
    if other_cut or own_cut:
        return other_cut
    return other_bottom - own_bottom >= NEARER_MARGIN * image_height


def occlusion(
    person: PersonPose, others: Sequence[PersonPose], image_height: int, min_keypoint_score: float
) -> tuple[float, float]:
    """Return max_overlap and torso_cover: the largest IoU with person's box and the largest torso cover, of the nearer people."""
    torso = torso_box(person)
    max_overlap = torso_cover = 0.0
    for other in others:
        if not is_nearer(other, person, image_height, min_keypoint_score):
            continue
        max_overlap = max(max_overlap, box_iou(other.box, person.box))
        if torso is not None:
            torso_cover = max(torso_cover, box_cover(other.box, torso))
    return max_overlap, torso_cover


def outer_rect(box: np.ndarray) -> Rect:
    """The whole pixels that a box touches, also in part."""
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float64)
    return math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)


def pad_rect(box: np.ndarray, padding: float, width: int, height: int) -> Rect:
    """The box grown by padding times its width and height on each side, clipped to the image."""
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float64)
    dx, dy = padding * (x2 - x1), padding * (y2 - y1)
    x0, y0, x3, y3 = outer_rect(np.array([x1 - dx, y1 - dy, x2 + dx, y2 + dy]))
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


def mask_rects(own_box: np.ndarray, other_boxes: Sequence[np.ndarray], crop_rect: Rect) -> list[Rect]:
    """The rectangles to grey, in image pixels: each other box inside crop_rect, minus own_box.

    own_box is taken as every pixel it touches, so none of its pixels is
    greyed. Rectangles of two other people may overlap.
    """
    own = outer_rect(own_box)
    out: list[Rect] = []
    for box in other_boxes:
        inside = intersect_rects(outer_rect(box), crop_rect)
        if inside is not None:
            out.extend(subtract_rect(inside, own))
    return out


def region_polygon(keypoints: np.ndarray, corners: Sequence[int], shrink: float = 0.0) -> np.ndarray:
    """The polygon through the given keypoints, shrunk by shrink towards its centre. Shape (len(corners), 2)."""
    points = np.asarray(keypoints, dtype=np.float64)[list(corners)]
    centre = points.mean(axis=0)
    return centre + (points - centre) * (1.0 - shrink)


def polygon_mask(polygon: np.ndarray, width: int, height: int) -> tuple[np.ndarray, int]:
    """Fill a polygon. Return its pixels inside a width x height image, and its pixel count in total.

    The total also counts the pixels outside the image, so the share left
    after clipping is the first count over the second.
    """
    polygon = np.asarray(polygon, dtype=np.float64)
    lo = np.floor(polygon.min(axis=0)).astype(int)
    hi = np.ceil(polygon.max(axis=0)).astype(int)
    canvas = np.zeros((hi[1] - lo[1] + 1, hi[0] - lo[0] + 1), dtype=np.uint8)
    cv2.fillPoly(canvas, [np.round(polygon - lo).astype(np.int32)], 1)
    total = int(np.count_nonzero(canvas))
    out = np.zeros((height, width), dtype=bool)
    inside = intersect_rects((0, 0, width, height), (int(lo[0]), int(lo[1]), int(hi[0]) + 1, int(hi[1]) + 1))
    if inside is not None:
        x0, y0, x1, y1 = inside
        out[y0:y1, x0:x1] = canvas[y0 - lo[1] : y1 - lo[1], x0 - lo[0] : x1 - lo[0]] > 0
    return out, total


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


def region_colors(
    pixels: np.ndarray, polygon: np.ndarray, box: Rect, masked: Sequence[Rect]
) -> tuple[np.ndarray | None, float]:
    """The colour shares of a region of a crop, and the share of its pixels left.

    All in pixels of the crop. The region is the polygon clipped to box,
    without the masked rectangles: greyed pixels are left out, not counted
    as grey. Returns (None, 0.0) when fewer than MIN_REGION_PIXELS are left.
    """
    height, width = pixels.shape[:2]
    inside, total = polygon_mask(polygon, width, height)
    keep = np.zeros((height, width), dtype=bool)
    x0, y0, x1, y1 = box
    keep[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)] = True
    for rx0, ry0, rx1, ry1 in masked:
        keep[max(0, ry0) : max(0, ry1), max(0, rx0) : max(0, rx1)] = False
    region = inside & keep
    count = int(np.count_nonzero(region))
    if total == 0 or count < MIN_REGION_PIXELS:
        return None, 0.0
    return color_shares(pixels[region]), count / total


def scale_down(pixels: np.ndarray, max_side: int) -> tuple[np.ndarray, float, float]:
    """Scale an image so that its longer side is at most max_side. Returns it with the x and y scales."""
    height, width = pixels.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return pixels, 1.0, 1.0
    scale = max_side / longest
    new_width, new_height = max(1, round(width * scale)), max(1, round(height * scale))
    out = cv2.resize(pixels, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return out, new_width / width, new_height / height


def encode_jpeg(pixels: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    """Encode a uint8 BGR image as a JPEG."""
    ok, data = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError(f"OpenCV could not encode a {pixels.shape} image as a JPEG")
    return data.tobytes()


def crop_person(
    image: np.ndarray,
    people: Sequence[PersonPose],
    track_id: int,
    *,
    padding: float = 0.10,
    max_side: int = 512,
    mask_others: bool = True,
    min_keypoint_score: float = 0.5,
) -> PersonCrop:
    """Crop the person with track_id out of a BGR image, given every person found in it.

    people are the image's people as published, with their published
    track ids. Raises ValueError for an option out of range (check_options).
    """
    check_options(padding, max_side, min_keypoint_score)
    height, width = image.shape[:2]
    index = next((i for i, p in enumerate(people) if int(p.track_id) == int(track_id)), -1)
    if index < 0:
        ids = sorted(int(p.track_id) for p in people)
        return PersonCrop(CropStatus.NOT_IN_VIEW, f"track {track_id} is not in the image, which has tracks {ids}")
    person = people[index]
    others = [p for i, p in enumerate(people) if i != index]
    max_overlap, torso_cover = occlusion(person, others, height, min_keypoint_score)
    found = {"index": index, "max_overlap": max_overlap, "torso_cover": torso_cover}

    if max_overlap > OCCLUDED_IOU or torso_cover > OCCLUDED_TORSO_COVER:
        return PersonCrop(
            CropStatus.OCCLUDED,
            f"a nearer person is in the way: IoU {max_overlap:.2f}, torso cover {torso_cover:.2f}",
            **found,
        )
    x1, y1, x2, y2 = np.asarray(person.box, dtype=np.float64)
    if y2 - y1 < MIN_BOX_HEIGHT_PX:
        # Rounded down, so a box just under the floor never reads as at it.
        return PersonCrop(
            CropStatus.TOO_SMALL, f"the box is {np.floor(y2 - y1):.0f} px tall, under {MIN_BOX_HEIGHT_PX:g}", **found
        )
    if x2 - x1 <= 0:
        return PersonCrop(CropStatus.TOO_SMALL, "the box has no width", **found)

    rect = pad_rect(person.box, padding, width, height)
    cx, cy = rect[0], rect[1]
    pixels = image[rect[1] : rect[3], rect[0] : rect[2]].copy()
    masked = [_shift(r, -cx, -cy) for r in mask_rects(person.box, [p.box for p in others], rect)] if mask_others else []
    for rx0, ry0, rx1, ry1 in masked:
        pixels[ry0:ry1, rx0:rx1] = MASK_GREY
    own = _shift(intersect_rects(outer_rect(person.box), rect) or (cx, cy, cx, cy), -cx, -cy)
    keypoints = np.asarray(person.keypoints, dtype=np.float64) - [cx, cy]
    scores = np.asarray(person.keypoint_scores, dtype=np.float64)

    def region(corners: Sequence[int], shrink: float) -> tuple[np.ndarray | None, float]:
        if any(scores[i] < min_keypoint_score for i in corners):
            return None, 0.0
        return region_colors(pixels, region_polygon(keypoints, corners, shrink), own, masked)

    top_shares, top_left = region(TORSO, TORSO_SHRINK)
    bottom_shares, bottom_left = region(THIGHS, 0.0)

    scaled, sx, sy = scale_down(pixels, max_side)
    return PersonCrop(
        CropStatus.OK,
        pixels=scaled,
        jpeg=encode_jpeg(scaled),
        keypoints=keypoints * [sx, sy],
        masked=np.array(masked, dtype=np.float64).reshape(-1, 4) * [sx, sy, sx, sy],
        top_shares=top_shares,
        bottom_shares=bottom_shares,
        top_left=top_left,
        bottom_left=bottom_left,
        **found,
    )


def _area(box: np.ndarray) -> float:
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float64)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _overlap_area(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = np.asarray(a, dtype=np.float64)
    bx1, by1, bx2, by2 = np.asarray(b, dtype=np.float64)
    return max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))


def _contains(box: np.ndarray, point: np.ndarray) -> bool:
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float64)
    return bool(x1 <= point[0] <= x2 and y1 <= point[1] <= y2)


def _shift(rect: Rect, dx: int, dy: int) -> Rect:
    return rect[0] + dx, rect[1] + dy, rect[2] + dx, rect[3] + dy
