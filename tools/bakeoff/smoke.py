"""A tiny data set from the Ultralytics sample images, to check the whole tool.

The smoke run writes a recording with four frames (bus.jpg, zidane.jpg and
their mirror images, in four raw encodings), extracts it, runs YOLOE on it,
and then writes the reviewed manifest by hand: two views of each object,
the frame and its mirror image. Its numbers show that the tool works, not
which model is better.
"""

from __future__ import annotations

import concurrent.futures
import importlib.util
import multiprocessing
import shutil
from pathlib import Path

import cv2

from bag import extract, make_image_msg, write_bag
from dataset import Crop, Object, Query, Workdir, read_manifest, write_manifest, write_objects, write_queries

TOPIC = "/smoke/image"
MARKER = ".bakeoff-smoke"
"""Marks a directory the smoke run made, and may therefore delete."""

OBJECTS = [
    Object("bus", "bus", "", "blue electric city bus"),
    Object("man_cream_coat", "person", "", "man in a cream fur coat and sunglasses"),
    Object("man_black_coat", "person", "", "man in a black coat, red shirt and sunglasses"),
    Object("man_walking", "person", "", "man in black at the right edge, walking"),
    Object("bollard_centre", "bollard", "bollards", "grey bollard on the pavement, centre"),
    Object("bollard_right", "bollard", "bollards", "grey bollard on the pavement, right"),
    Object("no_parking_sign", "traffic sign", "", "round blue and red no-parking sign"),
    Object("man_bald", "person", "suits", "bald man in a dark suit, shouting"),
    Object("man_grey_hair", "person", "suits", "grey-haired man in a dark suit"),
    Object("tie_black", "necktie", "", "black tie"),
    Object("badge", "badge", "", "ID badge on a lanyard"),
]

QUERIES = [
    Query("a blue bus", ("bus",)),
    Query("a man in a cream fur coat", ("man_cream_coat",)),
    Query("a man in a black coat and a red shirt", ("man_black_coat",)),
    Query("a man wearing sunglasses", ("man_cream_coat", "man_black_coat")),
    Query("a person in dark clothes and green trainers", ("man_walking",)),
    Query("a short grey post on the pavement", ("bollard_centre", "bollard_right")),
    Query("a no parking sign", ("no_parking_sign",)),
    Query("a bald man shouting", ("man_bald",)),
    Query("an old man with grey hair", ("man_grey_hair",)),
    Query("a man in a dark suit", ("man_bald", "man_grey_hair")),
    Query("a black tie", ("tie_black",)),
    Query("an ID badge on a lanyard", ("badge",)),
]

# Boxes drawn by hand on the original images (x1, y1, x2, y2; x2 and y2 exclusive).
BOXES = {
    "bus.jpg": {
        "bus": (0, 232, 804, 737),
        "man_cream_coat": (52, 401, 243, 901),
        "man_black_coat": (225, 406, 341, 859),
        "man_walking": (672, 386, 810, 877),
        "bollard_centre": (426, 660, 461, 761),
        "bollard_right": (658, 626, 690, 716),
        "no_parking_sign": (0, 252, 33, 325),
    },
    "zidane.jpg": {
        "man_bald": (127, 204, 1077, 718),
        "man_grey_hair": (748, 44, 1152, 719),
        "tie_black": (435, 433, 528, 720),
        "badge": (1047, 481, 1137, 703),
    },
}

# (source image, mirrored, encoding). Every raw encoding the extractor reads
# in colour is used once.
FRAMES = [
    ("bus.jpg", False, "bgra8"),
    ("zidane.jpg", False, "bgr8"),
    ("bus.jpg", True, "rgb8"),
    ("zidane.jpg", True, "rgba8"),
]


def assets_dir() -> Path:
    """The sample images that ship with Ultralytics."""
    spec = importlib.util.find_spec("ultralytics")
    if spec is None or spec.origin is None:
        raise FileNotFoundError("ultralytics is not installed, so its sample images are missing")
    return Path(spec.origin).parent / "assets"


def default_workdir(repo: Path) -> Path:
    return repo / "bakeoff-data" / "smoke"


def reset_workdir(root: Path) -> None:
    """Empty a smoke directory, refusing any directory the smoke run did not make."""
    if root.exists():
        if not (root / MARKER).is_file():
            if any(root.iterdir()):
                raise FileExistsError(f"{root} exists and was not made by the smoke run; choose another --workdir")
        else:
            shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / MARKER).write_text("Made by `bakeoff.py run --smoke`; deleted and rewritten by the next smoke run.\n")


def propose_in_child(*args, **kwargs) -> list[Crop]:
    """Run propose.propose() in a new process and return its crops.

    Loading a model in Ultralytics changes the whole process: it sets
    CUBLAS_WORKSPACE_CONFIG=:4096:8 (ultralytics/engine/model.py). Every
    later cuBLAS call in the process then uses that workspace, which made
    each embedder's batch-1 times up to twice as long and its peak memory
    larger. In its own process, YOLOE cannot change the costs that the run
    measures afterwards, and its memory is freed when the process ends.
    """
    from propose import propose

    # spawn starts a fresh interpreter; fork would copy this process's CUDA state.
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
        return pool.submit(propose, *args, **kwargs).result()


def _mirror(box: tuple[int, int, int, int], width: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (width - x2, y1, width - x1, y2)


def build(work: Workdir, device: str, log=print) -> list[Crop]:
    """Write the smoke data set into work.root; return the reviewed crops."""
    reset_workdir(work.root)
    assets = assets_dir()
    messages, frame_boxes = [], []
    for i, (name, mirrored, encoding) in enumerate(FRAMES):
        bgr = cv2.imread(str(assets / name))
        if bgr is None:
            raise FileNotFoundError(f"cannot read {assets / name}")
        width = bgr.shape[1]
        boxes = BOXES[name]
        if mirrored:
            bgr = cv2.flip(bgr, 1)
            boxes = {k: _mirror(b, width) for k, b in boxes.items()}
        t_ns = (i + 1) * 100_000_000
        messages.append((TOPIC, make_image_msg(bgr, encoding, stamp_ns=t_ns), t_ns))
        frame_boxes.append(boxes)
    bag_path = work.root / "bag"
    write_bag(bag_path, "mcap", messages)
    result = extract(bag_path, TOPIC, 1, work.images, work.frames, prefix="smoke")
    log(f"extract: {len(result.written)} of {result.messages} frames from {bag_path}")

    write_objects(work.objects, OBJECTS)
    write_queries(work.queries, QUERIES)
    label = {o.object_id: o.label for o in OBJECTS}
    crops = []
    for image, boxes in zip(result.written, frame_boxes):
        stem = Path(image).stem
        for oid, box in boxes.items():
            crops.append(Crop(f"{stem}_{oid}", image, box, oid, label[oid], None))
    write_manifest(work.manifest, crops)

    try:
        proposed = propose_in_child(
            work.images,
            OBJECTS,
            list(dict.fromkeys(o.label for o in OBJECTS)),
            work.root / "manifest.proposed.csv",
            work.review,
            device=device,
        )
        matched = sum(1 for c in crops if any(_iou(c.box, p.box) >= 0.5 for p in proposed if p.image == c.image))
        log(f"propose: {len(proposed)} boxes; {matched} of the {len(crops)} hand-drawn boxes have one with IoU >= 0.5")
    except FileNotFoundError as e:
        log(f"propose: skipped ({e})")
    return read_manifest(work.manifest, OBJECTS)


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0
