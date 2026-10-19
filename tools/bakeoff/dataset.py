"""The bake-off data set: objects, queries, and the reviewed manifest of crops.

A work directory holds:

    objects.csv     object_id,label,pair,note: one row per physical object
    queries.csv     query,object_ids: one row per query, ids separated by spaces
    images/         frames from `extract`, or any images
    frames.csv      where each extracted frame came from
    manifest.csv    crop_id,image,x1,y1,x2,y2,label,score,object_id: one row
                    per box, from `propose`, reviewed by hand
    review/         images with the proposed boxes drawn on them
    results/        the output of `run`

docs/guides/bakeoff.md describes each file and the review rules.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

OBJECT_COLUMNS = ("object_id", "label", "pair", "note")
QUERY_COLUMNS = ("query", "object_ids")
MANIFEST_COLUMNS = ("crop_id", "image", "x1", "y1", "x2", "y2", "label", "score", "object_id")
DROP = "-"
"""The object_id that drops a manifest row."""
SIZES = (32, 48, 64, 96)
BUCKETS: tuple[str | int, ...] = ("native", *SIZES)
""""native" is the crop as cut from the frame; a number is its longest side in pixels."""
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
_ID = re.compile(r"[A-Za-z0-9_.\-]+")


class DatasetError(ValueError):
    """A data set file is missing or wrong. The message says which row."""


@dataclass(frozen=True)
class Workdir:
    root: Path

    @property
    def objects(self) -> Path:
        return self.root / "objects.csv"

    @property
    def queries(self) -> Path:
        return self.root / "queries.csv"

    @property
    def images(self) -> Path:
        return self.root / "images"

    @property
    def frames(self) -> Path:
        return self.root / "frames.csv"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.csv"

    @property
    def review(self) -> Path:
        return self.root / "review"

    @property
    def results(self) -> Path:
        return self.root / "results"


@dataclass(frozen=True)
class Object:
    object_id: str
    label: str
    """The class name YOLOE is prompted with, for example "mug"."""
    pair: str = ""
    """Objects with the same non-empty pair value are compared as similar objects."""
    note: str = ""


@dataclass(frozen=True)
class Query:
    text: str
    object_ids: tuple[str, ...]
    """Every object that answers the query."""


@dataclass(frozen=True)
class Crop:
    crop_id: str
    image: str
    """File name under images/."""
    box: tuple[int, int, int, int]
    """x1, y1, x2, y2 in pixels; x2 and y2 are exclusive."""
    object_id: str
    label: str = ""
    score: float | None = None


def _read_rows(path: Path, required: tuple[str, ...]) -> list[tuple[int, dict[str, str]]]:
    """Return (line number, row) for each data row, with values stripped."""
    if not path.is_file():
        raise DatasetError(f"{path}: file not found")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise DatasetError(f"{path}: missing column(s) {', '.join(missing)}; expected {', '.join(required)}")
        rows = []
        for row in reader:
            values = {k: (v or "").strip() for k, v in row.items() if k is not None}
            if any(values.values()):
                rows.append((reader.line_num, values))
        return rows


def _raise_if(errors: list[str], path: Path) -> None:
    if errors:
        raise DatasetError(f"{path}:\n  " + "\n  ".join(errors))


def read_objects(path: Path) -> list[Object]:
    errors, objects, seen = [], [], set()
    for line, row in _read_rows(path, ("object_id", "label")):
        oid, label = row["object_id"], row["label"]
        if not _ID.fullmatch(oid) or oid == DROP:
            errors.append(f"line {line}: object_id {oid!r} must be letters, digits, '_', '.' or '-', and not '-' alone")
        elif oid in seen:
            errors.append(f"line {line}: object_id {oid!r} appears twice")
        if not label:
            errors.append(f"line {line}: label is empty")
        seen.add(oid)
        objects.append(Object(oid, label, row.get("pair", ""), row.get("note", "")))
    if not objects and not errors:
        errors.append("no objects")
    _raise_if(errors, path)
    return objects


def read_queries(path: Path, objects: list[Object]) -> list[Query]:
    known = {o.object_id for o in objects}
    errors, queries, seen = [], [], set()
    for line, row in _read_rows(path, QUERY_COLUMNS):
        text, ids = row["query"], tuple(dict.fromkeys(row["object_ids"].split()))
        if not text:
            errors.append(f"line {line}: query is empty")
        elif text in seen:
            errors.append(f"line {line}: query {text!r} appears twice; put all its objects on one row")
        if not ids:
            errors.append(f"line {line}: object_ids is empty")
        unknown = [i for i in ids if i not in known]
        if unknown:
            errors.append(f"line {line}: unknown object id(s) {', '.join(unknown)}")
        seen.add(text)
        queries.append(Query(text, ids))
    if not queries and not errors:
        errors.append("no queries")
    _raise_if(errors, path)
    return queries


def _int(value: str, name: str, line: int, errors: list[str]) -> int:
    try:
        return int(value)
    except ValueError:
        errors.append(f"line {line}: {name} {value!r} is not a whole number of pixels")
        return 0


def read_manifest(path: Path, objects: list[Object]) -> list[Crop]:
    """Return the kept crops, sorted by image and crop_id.

    Every row needs an object_id: a known id keeps the row, and "-" drops it.
    A blank object_id is an error, so that a row nobody reviewed cannot slip
    into the results.
    """
    known = {o.object_id for o in objects}
    errors, blank, crops, seen = [], [], [], set()
    for line, row in _read_rows(path, MANIFEST_COLUMNS):
        cid, oid = row["crop_id"], row["object_id"]
        if not cid:
            errors.append(f"line {line}: crop_id is empty")
        elif cid in seen:
            errors.append(f"line {line}: crop_id {cid!r} appears twice")
        seen.add(cid)
        if not oid:
            blank.append(f"line {line} ({cid}, proposed {row['label'] or '?'})")
            continue
        if oid == DROP:
            continue
        if oid not in known:
            errors.append(f"line {line}: unknown object_id {oid!r}")
            continue
        box = tuple(_int(row[k], k, line, errors) for k in ("x1", "y1", "x2", "y2"))
        if min(box) < 0 or box[2] <= box[0] or box[3] <= box[1]:
            errors.append(f"line {line}: box {box} needs 0 <= x1 < x2 and 0 <= y1 < y2")
        if not row["image"]:
            errors.append(f"line {line}: image is empty")
        score = row.get("score", "")
        try:
            score_value = float(score) if score else None
        except ValueError:
            errors.append(f"line {line}: score {score!r} is not a number")
            score_value = None
        crops.append(Crop(cid, row["image"], box, oid, row["label"], score_value))
    if blank:
        errors.append(
            f"{len(blank)} row(s) not reviewed: write an object_id, or {DROP} to drop the row:\n    "
            + "\n    ".join(blank)
        )
    if not crops and not errors:
        errors.append("no rows are kept")
    _raise_if(errors, path)
    return sorted(crops, key=lambda c: (c.image, c.crop_id))


def write_manifest(path: Path, crops: list[Crop]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(MANIFEST_COLUMNS)
        for c in crops:
            score = "" if c.score is None else f"{c.score:.3f}"
            w.writerow([c.crop_id, c.image, *c.box, c.label, score, c.object_id])


def write_objects(path: Path, objects: list[Object]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(OBJECT_COLUMNS)
        for o in objects:
            w.writerow([o.object_id, o.label, o.pair, o.note])


def write_queries(path: Path, queries: list[Query]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(QUERY_COLUMNS)
        for q in queries:
            w.writerow([q.text, " ".join(q.object_ids)])


def similar_pairs(objects: list[Object]) -> list[tuple[str, str, str]]:
    """Return (group, object A, object B) for every two objects in one pair group."""
    groups: dict[str, list[str]] = {}
    for o in objects:
        if o.pair:
            groups.setdefault(o.pair, []).append(o.object_id)
    return [(g, a, b) for g, ids in groups.items() for a, b in combinations(ids, 2)]


def check(objects: list[Object], queries: list[Query], crops: list[Crop]) -> list[str]:
    """Raise DatasetError for a query that cannot be scored; return warnings."""
    counts = Counter(c.object_id for c in crops)
    errors, warnings = [], []
    for q in queries:
        present = [i for i in q.object_ids if counts[i]]
        if not present:
            errors.append(f"query {q.text!r}: none of its objects ({' '.join(q.object_ids)}) has a crop")
        elif len(present) < len(q.object_ids):
            absent = [i for i in q.object_ids if not counts[i]]
            warnings.append(f"query {q.text!r}: {' '.join(absent)} has no crop; it is scored on the others")
    for o in objects:
        if not counts[o.object_id]:
            warnings.append(f"object {o.object_id} has no crop")
    for group, a, b in similar_pairs(objects):
        if counts[a] < 2 and counts[b] < 2:
            warnings.append(f"pair {group} ({a}, {b}): neither object has two crops, so within-object similarity is undefined")
    if errors:
        raise DatasetError("\n  ".join(["the queries cannot be scored:", *errors]))
    return warnings


def load_dataset(work: Workdir) -> tuple[list[Object], list[Query], list[Crop], list[str]]:
    objects = read_objects(work.objects)
    queries = read_queries(work.queries, objects)
    crops = read_manifest(work.manifest, objects)
    warnings = check(objects, queries, crops)
    return objects, queries, crops, warnings


def cut_crops(images_dir: Path, crops: list[Crop]) -> list[Image.Image]:
    """Cut every crop out of its image, as RGB. Each image is opened once.

    An image's EXIF orientation is applied first, as cv2.imread applies it
    for YOLOE and the review overlays, so that a box means the same pixels
    in every step. Frames from `extract` have no EXIF data.
    """
    out: list[Image.Image] = []
    cache: dict[str, Image.Image] = {}
    errors = []
    for c in crops:
        if c.image not in cache:
            path = images_dir / c.image
            if not path.is_file():
                errors.append(f"{c.crop_id}: image {path} not found")
                continue
            with Image.open(path) as im:
                cache.clear()  # crops are sorted by image, so one image at a time is enough
                cache[c.image] = ImageOps.exif_transpose(im).convert("RGB")
        image = cache[c.image]
        x1, y1, x2, y2 = c.box
        if x2 > image.width or y2 > image.height:
            errors.append(f"{c.crop_id}: box {c.box} is outside {c.image} ({image.width}x{image.height})")
            continue
        out.append(image.crop(c.box))
    if errors:
        raise DatasetError("\n  ".join(["cannot cut the crops:", *errors]))
    return out


def downscale(image: Image.Image, longest: int) -> Image.Image | None:
    """Shrink image so that its longest side is `longest` pixels.

    Returns None when the image is smaller than that: an image is never
    enlarged here. Each model's own preprocessing enlarges it afterwards,
    which is what happens to a small, far-away object on the robot.
    """
    w, h = image.size
    side = max(w, h)
    if side < longest:
        return None
    if side == longest:
        return image.copy()
    scale = longest / side
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    # INTER_AREA averages the source pixels, like a camera that sees the
    # object from further away; it does not alias.
    return Image.fromarray(cv2.resize(np.asarray(image), size, interpolation=cv2.INTER_AREA))


def make_buckets(native: list[Image.Image]) -> dict[str | int, tuple[list[int], list[Image.Image]]]:
    """Return, per bucket, the indices of the crops in it and their images.

    A crop is in bucket s only when its longest side is at least s pixels.
    """
    buckets: dict[str | int, tuple[list[int], list[Image.Image]]] = {"native": (list(range(len(native))), native)}
    for s in SIZES:
        idx, ims = [], []
        for i, im in enumerate(native):
            small = downscale(im, s)
            if small is not None:
                idx.append(i)
                ims.append(small)
        buckets[s] = (idx, ims)
    return buckets


def list_images(images_dir: Path) -> list[Path]:
    """The images in images_dir, sorted by name.

    Crop ids and review overlays are named after an image's stem, so two
    images that differ only in their suffix, such as a.png and a.jpg, are
    refused.
    """
    if not images_dir.is_dir():
        return []
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    stems = Counter(p.stem for p in images)
    twins = [p.name for p in images if stems[p.stem] > 1]
    if twins:
        raise DatasetError(
            f"{images_dir}: these images share a name and differ only in the suffix: {', '.join(twins)}; "
            "rename or remove one of each"
        )
    return images
