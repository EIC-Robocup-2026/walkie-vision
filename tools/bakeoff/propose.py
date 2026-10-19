"""Propose boxes with YOLOE-26 for a person to review.

YOLOE runs in text-prompt mode with the labels of objects.csv (or a labels
file). Each detection becomes one manifest row. The object_id column is
filled only when exactly one object has the detected label; every other row
is left blank for the reviewer. review/ gets a copy of each image with the
boxes and crop ids drawn on it.
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager
from pathlib import Path

import cv2

from dataset import Crop, DatasetError, Object, list_images, write_manifest
from walkie_vision.models import ULTRALYTICS_DIR, models_dir

WEIGHTS = "yoloe-26s-seg.pt"
TEXT_ENCODER = "mobileclip2_b.ts"


def read_labels(path: Path) -> list[str]:
    """One label per line; blank lines and lines starting with # are skipped."""
    if not path.is_file():
        raise DatasetError(f"{path}: file not found")
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            labels.append(line)
    labels = list(dict.fromkeys(labels))
    if not labels:
        raise DatasetError(f"{path}: no labels")
    return labels


@contextmanager
def _in_dir(path: Path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def detect(images: list[Path], labels: list[str], conf: float, device: str) -> list[list[tuple[tuple[int, int, int, int], str, float]]]:
    """Return, per image, (box, label, score) for each detection, best score first."""
    weights_dir = models_dir() / ULTRALYTICS_DIR
    missing = [f for f in (WEIGHTS, TEXT_ENCODER) if not (weights_dir / f).is_file()]
    if missing:
        raise FileNotFoundError(f"{', '.join(missing)} not in {weights_dir}: run `pixi run fetch-models`")
    # No DNS probe and no usage analytics: Ultralytics treats itself as offline.
    os.environ.setdefault("YOLO_OFFLINE", "1")
    import torch
    from ultralytics import YOLOE

    images = [p.resolve() for p in images]
    out = []
    # Ultralytics looks for weights in the working directory, and
    # set_classes() loads the text encoder by file name.
    with _in_dir(weights_dir):
        model = YOLOE(WEIGHTS)
        model.set_classes(labels)
        try:
            for path in images:
                r = model.predict(str(path), device=device, conf=conf, agnostic_nms=True, verbose=False)[0]
                h, w = r.orig_shape
                found = []
                for xyxy, cls, score in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                    x1, y1, x2, y2 = xyxy
                    box = (max(0, math.floor(x1)), max(0, math.floor(y1)), min(w, math.ceil(x2)), min(h, math.ceil(y2)))
                    if box[2] > box[0] and box[3] > box[1]:
                        found.append((box, r.names[int(cls)], float(score)))
                found.sort(key=lambda d: (-d[2], d[0]))
                out.append(found)
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return out


def propose(
    images_dir: Path,
    objects: list[Object],
    labels: list[str],
    out: Path,
    review_dir: Path,
    conf: float = 0.25,
    device: str = "0",
    force: bool = False,
) -> list[Crop]:
    if out.exists() and not force:
        raise DatasetError(f"{out} exists and may hold your review; pass --force to replace it")
    images = list_images(images_dir)
    if not images:
        raise DatasetError(f"no images in {images_dir}: run extract first")
    by_label: dict[str, list[str]] = {}
    for o in objects:
        by_label.setdefault(o.label, []).append(o.object_id)

    detections = detect(images, labels, conf, device)
    crops = []
    review_dir.mkdir(parents=True, exist_ok=True)
    for path, found in zip(images, detections):
        canvas = cv2.imread(str(path))
        for j, (box, label, score) in enumerate(found):
            ids = by_label.get(label, [])
            crop = Crop(f"{path.stem}_{j:02d}", path.name, box, ids[0] if len(ids) == 1 else "", label, score)
            crops.append(crop)
            _draw(canvas, crop)
        cv2.imwrite(str(review_dir / f"{path.stem}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    write_manifest(out, crops)
    return crops


def _draw(canvas, crop: Crop) -> None:
    x1, y1, x2, y2 = crop.box
    cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 255), 2)
    text = f"{crop.crop_id} {crop.label} {crop.score:.2f}"
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y = max(th + base, y1)
    cv2.rectangle(canvas, (x1, y - th - base), (x1 + tw, y), (0, 255, 255), -1)
    cv2.putText(canvas, text, (x1, y - base), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
