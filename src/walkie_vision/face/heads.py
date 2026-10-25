"""Faces from head crops at native resolution, one per person.

SCRFD's 640 input shrinks a 2208 px wide image about 3.4 times. A face that
walkie-agent-ros already trusts, 0.0469 rad or about 50 px at fx 1066, would
be about 14 px there, the edge of what SCRFD detects. HeadFaceRecognizer
instead runs SCRFD on a square crop around each person's head, taken from
the full-resolution image, so the face stays at 1x or larger. The head comes
from the pose keypoints (nose, eyes and ears).

Each face is checked against its person: its centre must lie inside the
person's box and near the head keypoints. A face found for two people goes
to the nearer one. ArcFace then embeds every face in one batch.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..providers import ARCFACE_W600K_R50, Face, PersonPose
from . import face_align
from .analysis import FaceAnalyzer

HEAD_KEYPOINTS = (0, 1, 2, 3, 4)  # nose, eyes, ears (COCO order)
SHOULDERS = (5, 6)
# A face's centre may lie this far outside the person's box, as a fraction
# of the box's width and height on each side.
_BOX_MARGIN = 0.1
# A clipped crop narrower or shorter than this is not searched.
_MIN_CROP_SIDE = 8


@dataclass(frozen=True)
class HeadCropConfig:
    """How HeadFaceRecognizer crops each head and picks its face."""

    min_keypoint_score: float = 0.3
    """A keypoint counts only from this score."""
    head_spread_scale: float = 4.0
    """The crop side as a multiple of the head keypoints' spread."""
    shoulder_scale: float = 1.2
    """The crop side as a multiple of the shoulder width."""
    box_width_scale: float = 0.8
    """Without head spread or shoulders: the side as a fraction of the box width."""
    box_height_cap: float = 0.4
    """... but at most this fraction of the box height."""
    min_crop_px: int = 64
    """The smallest crop side."""
    input_sizes: tuple[int, ...] = (160, 256, 384, 640)
    """SCRFD input sizes. A crop runs at the smallest size that holds it."""
    max_people: int = 6
    """Only the largest people get a face search."""
    dedupe_iou: float = 0.5
    """Two people's faces that overlap more than this are one face."""
    max_face_offset: float = 0.25
    """A face's centre must be within this fraction of the crop side from the head."""

    def __post_init__(self) -> None:
        sizes = tuple(self.input_sizes)
        if not sizes:
            raise ValueError("input_sizes is empty")
        for s in sizes:
            if isinstance(s, bool) or not isinstance(s, numbers.Integral) or s <= 0 or s % 32:
                raise ValueError(f"input size {s!r} is not a positive multiple of 32 (SCRFD's strides are 8, 16, 32)")
        object.__setattr__(self, "input_sizes", tuple(int(s) for s in sizes))
        if self.max_people < 0:
            raise ValueError(f"max_people is {self.max_people}; it must be 0 or more")
        if not 0 < self.dedupe_iou <= 1:
            raise ValueError(f"dedupe_iou is {self.dedupe_iou}; it must be in (0, 1]")
        if not 0 < self.max_face_offset <= 1:
            raise ValueError(f"max_face_offset is {self.max_face_offset}; it must be in (0, 1]")


def head_crop(
    pose: PersonPose, image_shape: tuple[int, int], config: HeadCropConfig
) -> tuple[tuple[int, int, int, int], np.ndarray, float] | None:
    """Return the crop box (x1, y1, x2, y2, ints), the head centre (x, y) and the crop side, or None.

    None means the pose sees no head keypoint, or the crop has almost
    nothing left inside the image. The side is that of the square before it
    is clipped to the image.
    """
    kp = np.asarray(pose.keypoints, dtype=np.float64)
    scores = np.asarray(pose.keypoint_scores, dtype=np.float64)
    seen = [i for i in HEAD_KEYPOINTS if scores[i] >= config.min_keypoint_score]
    if not seen:
        return None
    head = kp[seen]
    centre = head.mean(axis=0)

    ref = 0.0
    if len(head) >= 2:
        spread = np.ptp(head, axis=0)
        ref = max(ref, config.head_spread_scale * float(spread.max()))
    if all(scores[i] >= config.min_keypoint_score for i in SHOULDERS):
        ref = max(ref, config.shoulder_scale * float(np.linalg.norm(kp[SHOULDERS[0]] - kp[SHOULDERS[1]])))
    if ref == 0:
        x1, y1, x2, y2 = np.asarray(pose.box, dtype=np.float64)
        ref = min(config.box_width_scale * (x2 - x1), config.box_height_cap * (y2 - y1))
    side = max(ref, float(config.min_crop_px))
    if not (np.isfinite(centre).all() and math.isfinite(side)):
        return None

    height, width = image_shape[:2]
    half = side / 2
    x1 = max(0, math.floor(centre[0] - half))
    y1 = max(0, math.floor(centre[1] - half))
    x2 = min(width, math.ceil(centre[0] + half))
    y2 = min(height, math.ceil(centre[1] + half))
    if x2 - x1 < _MIN_CROP_SIDE or y2 - y1 < _MIN_CROP_SIDE:
        return None
    return (x1, y1, x2, y2), centre, side


def input_size_for(width: int, height: int, sizes: Sequence[int]) -> int:
    """Return the smallest size that holds the crop, else the largest size."""
    need = max(width, height)
    ordered = sorted(sizes)
    for s in ordered:
        if s >= need:
            return s
    return ordered[-1]


def clip_faces(det: np.ndarray, kpss: np.ndarray, image_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Clip the face boxes (det[:, 0:4]) to the image and drop those with no area left.

    A dropped box takes its landmarks with it. The landmarks are not
    clipped: alignment needs where they really are.
    """
    height, width = image_shape[:2]
    det = np.array(det, copy=True)
    kpss = np.array(kpss, copy=True)
    det[:, [0, 2]] = np.clip(det[:, [0, 2]], 0, width)
    det[:, [1, 3]] = np.clip(det[:, [1, 3]], 0, height)
    keep = (det[:, 2] > det[:, 0]) & (det[:, 3] > det[:, 1])
    return det[keep], kpss[keep]


def pick_face(boxes: np.ndarray, head_centre: np.ndarray, person_box: np.ndarray, max_offset_px: float) -> int | None:
    """Return the row of the face that belongs to this person, or None.

    A face qualifies when its centre lies inside the person's box, grown by
    10% of its width and height on each side, and within max_offset_px of
    the head centre. The nearest one wins; on a tie, the higher score.
    Without the distance gate, a neighbour's face inside this person's crop
    and box would be given to this person whenever the neighbour has no box
    of their own.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    if len(boxes) == 0:
        return None
    x1, y1, x2, y2 = np.asarray(person_box, dtype=np.float64)
    mx, my = _BOX_MARGIN * (x2 - x1), _BOX_MARGIN * (y2 - y1)
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    inside = (cx >= x1 - mx) & (cx <= x2 + mx) & (cy >= y1 - my) & (cy <= y2 + my)
    head = np.asarray(head_centre, dtype=np.float64)
    dist = np.hypot(cx - head[0], cy - head[1])
    candidates = np.flatnonzero(inside & (dist <= max_offset_px))
    if len(candidates) == 0:
        return None
    best = min(candidates, key=lambda i: (dist[i], -boxes[i, 4]))
    return int(best)


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Return the intersection over union of two x1, y1, x2, y2 boxes."""
    ax1, ay1, ax2, ay2 = np.asarray(a, dtype=np.float64)[:4]
    bx1, by1, bx2, by2 = np.asarray(b, dtype=np.float64)[:4]
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return inter / union if union > 0 else 0.0


class HeadFaceRecognizer:
    """SCRFD on head crops plus ArcFace: a walkie_vision.providers.PersonFaceRecognizer.

    analyzer gives the two ONNX Runtime models (analyzer.detector and
    analyzer.recognizer); its det_thresh filters the faces. With warm_up,
    the constructor runs SCRFD once at every input size and ArcFace once at
    every batch size up to max_people, because the CUDA provider pays a
    large first-call cost for each new input shape.
    """

    model_name = ARCFACE_W600K_R50
    dim = 512

    def __init__(self, analyzer: FaceAnalyzer, config: HeadCropConfig = HeadCropConfig(), *, warm_up: bool = True) -> None:
        self.analyzer = analyzer
        self.config = config
        if warm_up:
            self._warm_up()

    def _warm_up(self) -> None:
        for s in self.config.input_sizes:
            self.analyzer.detector.detect(np.zeros((s, s, 3), np.uint8), input_size=(s, s))
        w, h = self.analyzer.recognizer.input_size
        blank = np.zeros((h, w, 3), np.uint8)
        for n in range(1, max(1, self.config.max_people) + 1):
            self.analyzer.recognizer.get_feat([blank] * n)

    def faces_for(self, image: np.ndarray, people: Sequence[PersonPose]) -> list[Face | None]:
        """Return the face of each person in a BGR image, or None, in the order of people."""
        cfg = self.config
        out: list[Face | None] = [None] * len(people)
        offsets: dict[int, float] = {}

        def area(i: int) -> float:
            x1, y1, x2, y2 = np.asarray(people[i].box, dtype=np.float64)
            return max(0.0, x2 - x1) * max(0.0, y2 - y1)

        # sorted() is stable, so people of equal size keep their order.
        selected = sorted(range(len(people)), key=lambda i: -area(i))[: cfg.max_people]
        for i in selected:
            crop_box = head_crop(people[i], image.shape[:2], cfg)
            if crop_box is None:
                continue
            (x1, y1, x2, y2), centre, side = crop_box
            crop = image[y1:y2, x1:x2]
            s = input_size_for(x2 - x1, y2 - y1, cfg.input_sizes)
            det, kpss = self.analyzer.detector.detect(crop, input_size=(s, s), max_num=0)
            if det is None or len(det) == 0:
                continue
            if kpss is None:
                raise RuntimeError("the face detector returns no landmarks, so faces cannot be aligned")
            det = np.array(det, dtype=np.float64)
            kpss = np.array(kpss, dtype=np.float64)
            det[:, 0:4] += (x1, y1, x1, y1)
            kpss += (x1, y1)
            det, kpss = clip_faces(det, kpss, image.shape[:2])
            j = pick_face(det, centre, people[i].box, cfg.max_face_offset * side)
            if j is None:
                continue
            face = Face(bbox=det[j, :4].copy(), kps=kpss[j].copy(), det_score=float(det[j, 4]))
            fc = np.array([(face.bbox[0] + face.bbox[2]) / 2, (face.bbox[1] + face.bbox[3]) / 2])
            out[i] = face
            offsets[i] = float(np.linalg.norm(fc - centre)) / side

        self._dedupe(out, offsets, selected)
        self._embed(image, out)
        return out

    def _dedupe(self, out: list[Face | None], offsets: dict[int, float], selected: list[int]) -> None:
        """Give a face that two people share to the one whose head is nearer, by normalised distance."""
        # Nearest first; on a tie, the larger person. A face that overlaps a
        # face already kept is the same face, seen from a second crop.
        rank = {i: k for k, i in enumerate(selected)}
        chosen = sorted(offsets, key=lambda i: (offsets[i], rank[i]))
        kept: list[np.ndarray] = []
        for i in chosen:
            face = out[i]
            if face is None:
                continue
            if any(box_iou(face.bbox, box) > self.config.dedupe_iou for box in kept):
                out[i] = None
            else:
                kept.append(face.bbox)

    def _embed(self, image: np.ndarray, out: list[Face | None]) -> None:
        """Set each face's ArcFace embedding, all in one batch."""
        size = self.analyzer.recognizer.input_size[0]
        aligned: list[np.ndarray] = []
        owners: list[int] = []
        for i, face in enumerate(out):
            if face is None:
                continue
            try:
                aligned.append(face_align.norm_crop(image, landmark=face.kps, image_size=size))
            except ValueError:
                # Degenerate landmarks: no similarity transform fits them.
                out[i] = None
                continue
            owners.append(i)
        if not aligned:
            return
        feats = np.asarray(self.analyzer.recognizer.get_feat(aligned))
        if feats.shape[0] != len(aligned):
            raise ValueError(f"the recogniser returned {feats.shape[0]} embeddings for {len(aligned)} faces")
        for k, i in enumerate(owners):
            face = out[i]
            assert face is not None
            face.embedding = feats[k].astype(np.float32)
