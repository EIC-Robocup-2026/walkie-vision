"""Check 11: each of the vision node's model wrappers on the GPU.

YoloeTracker (YOLOE-26 with ByteTrack), YoloPoseTracker (YOLO26 pose with
BoT-SORT), HeadFaceRecognizer (SCRFD on head crops, then ArcFace) and
OSNetReIdentifier, loaded in the order the vision node loads them. The check
creates no ROS node: it only runs the models.

A tracker hides a new track until the next frame confirms it, except on its
first frame after a reset. So before each new still image the check resets
the tracker in use, runs it three times on that image and checks the third
result.

The camera pan shows why people use BoT-SORT: the robot turns its base at up
to 0.6 rad/s while it follows a person. A pan of 64 px per frame is 0.6 rad/s
at fx 1066 and 10 Hz, the node's default rate cap. YOLO26 pose has no NMS and
now and then gives one person a second, partial box, which the tracker keeps
as its own track. So the pan prints what the tracker returns, and asserts on
what the node publishes: the poses after its score floor and its dedupe
(walkie_vision.pipeline.dedupe_poses), with the node's default parameters.
"""

from __future__ import annotations

import os

# As the vision node's launch does: no online checks from Ultralytics.
os.environ.setdefault("YOLO_OFFLINE", "true")

import statistics
import time
from collections.abc import Callable, Sequence

import cv2
import numpy as np
import torch

from common import assets_dir, expect, finish, gpu_mem_of, timed_ms
from walkie_vision.models import BUFFALO_L_DIR, ULTRALYTICS_DIR, models_dir
from walkie_vision.node import VisionConfig
from walkie_vision.pipeline import dedupe_poses
from walkie_vision.providers import PersonPose
from walkie_vision.reid.identifier import OSNetReIdentifier
from walkie_vision.yolo import YoloeTracker, YoloPoseTracker

# The two faces of zidane.jpg, as checks/c08_face.py expects them (InsightFace
# 2.1 on the whole image).
ZIDANE_FACES = [[913, 96, 1057, 276], [545, 230, 664, 451]]
HD2K = (1242, 2208)  # height, width
FACE_TOLERANCE_PX = 15
NODE = VisionConfig(classes=("person",))  # the node's default parameters


def load(label: str, fn: Callable[[], object]) -> object:
    t0 = time.perf_counter()
    out = fn()
    print(f"{label}: loaded and warmed up in {time.perf_counter() - t0:.1f} s", flush=True)
    return out


def three(run: Callable[[np.ndarray], list], reset: Callable[[], None], image: np.ndarray) -> list[list]:
    """Reset the tracker, then run it three times on one image."""
    reset()
    return [run(image) for _ in range(3)]


def median_ms(fn: Callable[[], object], n: int = 10) -> float:
    return statistics.median(timed_ms(fn)[1] for _ in range(n))


def ids(results: Sequence) -> set[int]:
    return {r.track_id for r in results}


def keypoints_inside(poses: Sequence[PersonPose], shape: tuple[int, int]) -> bool:
    h, w = shape
    return all(
        bool(((p.keypoints[:, 0] >= 0) & (p.keypoints[:, 0] <= w) & (p.keypoints[:, 1] >= 0)
              & (p.keypoints[:, 1] <= h)).all())
        for p in poses
    )


def box_inside(box: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    x1, y1, x2, y2 = box
    return 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h


def match(faces: Sequence, want: Sequence[float]):
    """Return the found face nearest a wanted box (max abs difference), and that difference."""
    found = [f for f in faces if f is not None]
    if not found:
        return None, float("inf")
    best = min(found, key=lambda f: float(np.abs(np.asarray(f.bbox) - want).max()))
    return best, float(np.abs(np.asarray(best.bbox) - want).max())


def people_crops(image: np.ndarray, poses: Sequence[PersonPose]) -> list[np.ndarray]:
    h, w = image.shape[:2]
    crops = []
    for p in poses:
        x1, y1, x2, y2 = np.round(np.clip(p.box, 0, [w, h, w, h])).astype(int)
        if x2 - x1 >= 16 and y2 - y1 >= 16:
            crops.append(image[y1:y2, x1:x2])
    return crops


ultra = models_dir() / ULTRALYTICS_DIR
bus = cv2.imread(str(assets_dir() / "bus.jpg"))
zidane = cv2.imread(str(assets_dir() / "zidane.jpg"))
print("bus.jpg", bus.shape, "| zidane.jpg", zidane.shape)

# The vision node's order: the PyTorch models first, then the ONNX Runtime sessions.
yoloe = load("YoloeTracker", lambda: YoloeTracker(ultra / "yoloe-26s-seg.pt", ["person", "bus"]))
pose = load("YoloPoseTracker (botsort.yaml)", lambda: YoloPoseTracker(ultra / "yolo26s-pose.pt"))
reid = load("OSNetReIdentifier", lambda: OSNetReIdentifier())

import onnxruntime  # noqa: E402

from walkie_vision.face import FaceAnalyzer, HeadFaceRecognizer  # noqa: E402
from walkie_vision.face.heads import box_iou  # noqa: E402

# As build_providers() does: without it, ONNX Runtime warns on every crop
# smaller than 640 and every batch of faces.
onnxruntime.set_default_logger_severity(3)
analyzer = FaceAnalyzer(models_dir() / BUFFALO_L_DIR)
faces = load("HeadFaceRecognizer", lambda: HeadFaceRecognizer(analyzer))
for label, sess in (("detector", analyzer.detector.session), ("recognizer", analyzer.recognizer.session)):
    expect(sess.get_providers()[0] == "CUDAExecutionProvider", f"face {label} runs on CUDAExecutionProvider")

# 1. YOLOE on bus.jpg.
runs = three(yoloe.detect, yoloe.reset_tracks, bus)
labels = [d.label for d in runs[2]]
print("YOLOE on bus.jpg, per call:", [sorted((d.label, d.track_id) for d in r) for r in runs])
expect(labels.count("person") >= 4 and "bus" in labels, "YOLOE: the third call has at least 4 people and the bus")
expect(all(d.track_id > 0 for r in runs for d in r), "YOLOE: every track id is non-zero")
expect(len(runs[1]) > 0 and ids(runs[1]) == ids(runs[2]), "YOLOE: the ids of calls 2 and 3 are equal")
expect(all(d.label in yoloe.classes for r in runs for d in r), "YOLOE: every label is one of the classes")
expect(all(box_inside(d.box, bus.shape[:2]) for r in runs for d in r), "YOLOE: every box is inside the image")

# 2. Pose on bus.jpg.
runs = three(pose.estimate, pose.reset_tracks, bus)
bus_poses = runs[2]
print("pose on bus.jpg, ids per call:", [sorted(ids(r)) for r in runs])
expect(len(bus_poses) >= 4, f"pose: the third call has at least 4 people ({len(bus_poses)})")
expect(all(p.track_id > 0 for r in runs for p in r), "pose: every track id is non-zero")
expect(len(runs[1]) > 0 and ids(runs[1]) == ids(runs[2]), "pose: the ids of calls 2 and 3 are equal")
expect(all(keypoints_inside(r, bus.shape[:2]) for r in runs), "pose: every keypoint is inside the image")

# 3. Faces from head crops on zidane.jpg.
zidane_poses = three(pose.estimate, pose.reset_tracks, zidane)[2]
found = faces.faces_for(zidane, zidane_poses)
reference = analyzer.get(zidane)
print(f"zidane.jpg: {len(zidane_poses)} people; faces "
      f"{[None if f is None else (np.round(f.bbox).astype(int).tolist(), round(f.det_score, 3)) for f in found]}")
expect(len(found) == len(zidane_poses), "faces_for returns one entry per person")
for want in ZIDANE_FACES:
    face, diff = match(found, want)
    expect(diff <= FACE_TOLERANCE_PX, f"zidane.jpg: a head-crop face at {want} (within {FACE_TOLERANCE_PX} px; {diff:.1f})")
    ref, _ = match(reference, want)
    if face is not None and ref is not None:
        cos = float(face.normed_embedding @ ref.normed_embedding)
        print(f"  {want}: cosine with FaceAnalyzer.get on the whole image {cos:.4f}")
        expect(cos >= 0.9, f"zidane.jpg: the embedding at {want} matches the whole-image one (cosine >= 0.9)")
expect(all(box_inside(f.bbox, zidane.shape[:2]) for f in found if f is not None), "every face box is inside the image")

# 4. The HD2K case: half-size zidane.jpg on a black 2208 x 1242 canvas.
canvas = np.zeros((*HD2K, 3), np.uint8)
small = cv2.resize(zidane, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
ox, oy = 800, 500
canvas[oy : oy + small.shape[0], ox : ox + small.shape[1]] = small
canvas_poses = three(pose.estimate, pose.reset_tracks, canvas)[2]
canvas_faces = faces.faces_for(canvas, canvas_poses)
print(f"HD2K canvas: {len(canvas_poses)} people; faces "
      f"{[None if f is None else np.round(f.bbox).astype(int).tolist() for f in canvas_faces]}")
for want in ZIDANE_FACES:
    scaled = np.asarray(want) * 0.5 + (ox, oy, ox, oy)
    _, diff = match(canvas_faces, scaled)
    expect(diff <= FACE_TOLERANCE_PX,
           f"HD2K: a head-crop face at {np.round(scaled).astype(int).tolist()} (within {FACE_TOLERANCE_PX} px; {diff:.1f})")
whole = analyzer.get(canvas)
print(f"  for information: FaceAnalyzer.get on the whole canvas at 640 finds {len(whole)} faces")

# 5. OSNet on the people of bus.jpg.
crops = people_crops(bus, bus_poses)
vectors = reid.embed(crops)
print(f"OSNet: {len(crops)} crops -> {vectors.shape}")
expect(vectors.shape == (len(crops), 512) and len(crops) >= 4, "OSNet: one 512-value row per person crop")
expect(bool(np.isfinite(vectors).all()), "OSNet: every value is finite")

# 6. Camera pan.
tex = np.random.default_rng(0).integers(0, 256, (78, 240, 3), dtype=np.uint8)
strip = cv2.resize(tex, (3840, 1248), interpolation=cv2.INTER_NEAREST)[: HD2K[0]]
px, py = 1888, 600
strip[py : py + small.shape[0], px : px + small.shape[1]] = small
rect = (px, py, px + small.shape[1], py + small.shape[0])
HOLD = 3


def published(poses: Sequence[PersonPose]) -> list[PersonPose]:
    """The poses the vision node publishes: its score floor, then its dedupe."""
    kept = [p for p in poses if p.score >= NODE.people_min_score]
    return dedupe_poses(kept, NODE.people_dedupe_iou)


def inside_rect(poses: Sequence[PersonPose], x0: int) -> dict[int, PersonPose]:
    """The poses whose box centre lies inside the pasted people, by id."""
    inside: dict[int, PersonPose] = {}
    for p in poses:
        cx = (p.box[0] + p.box[2]) / 2 + x0
        cy = (p.box[1] + p.box[3]) / 2
        if rect[0] <= cx <= rect[2] and rect[1] <= cy <= rect[3]:
            inside[p.track_id] = p
    return inside


Frames = list[dict[int, PersonPose]]


def pan(tracker: YoloPoseTracker, step: int) -> tuple[Frames, Frames, float]:
    """Return, per frame, the pasted people from the tracker and as published, by id, and the median ms."""
    # 16 frames of 64 px at 10 Hz. A larger step is a lower frame rate at the
    # same turn rate, so the same pan takes fewer frames.
    frames = 16 * 64 // step
    offsets = [592] * HOLD + [592 + step * (k + 1) for k in range(frames)]
    assert offsets[-1] + HD2K[1] <= strip.shape[1], "the window leaves the strip"
    tracker.reset_tracks()
    raw, out, times = [], [], []
    for x0 in offsets:
        frame = np.ascontiguousarray(strip[:, x0 : x0 + HD2K[1]])
        poses, ms = timed_ms(tracker.estimate, frame)
        times.append(ms)
        raw.append(inside_rect(poses, x0))
        out.append(inside_rect(published(poses), x0))
    return raw, out, statistics.median(times)


def report(label: str, per_frame: Frames, ms: float | None = None) -> None:
    """Print the counts and ids; for an id missing from some frames, its scores and its
    largest IoU with a pose of the same frame (a high IoU means a second box of one person)."""
    counts = [len(f) for f in per_frame]
    frames = per_frame[HOLD - 1 :]
    distinct = set().union(*frames)
    always = set(frames[0]).intersection(*frames)
    others = {}
    for i in sorted(distinct - always):
        seen = [f for f in frames if i in f]
        iou = max((box_iou(f[i].box, q.box) for f in seen for j, q in f.items() if j != i), default=0.0)
        scores = [f[i].score for f in seen]
        others[i] = (f"in {len(seen)} frames, scores {min(scores):.2f}-{max(scores):.2f}, "
                     f"max IoU with another pose {iou:.2f}")
    timing = "" if ms is None else f"; pose median {ms:.1f} ms per HD2K frame"
    print(f"  {label}: people per frame {counts}; distinct ids from frame {HOLD} on: {len(distinct)} "
          f"{sorted(distinct)}; in every frame: {sorted(always)}{timing}")
    for i, text in others.items():
        print(f"    id {i}: {text}")


def report_pan(label: str, raw: Frames, out: Frames, ms: float) -> None:
    report(f"{label}, from the tracker", raw, ms)
    report(f"{label}, published (score floor {NODE.people_min_score:g}, dedupe IoU {NODE.people_dedupe_iou:g})", out)


print("camera pan, the two pasted people (frames 1-3 hold still, then the pan):")
raw, out, pan_ms = pan(pose, 64)
report_pan("botsort.yaml, 64 px per frame (0.6 rad/s at 10 Hz)", raw, out, pan_ms)
counts = [len(f) for f in out]
expect(all(c == 2 for c in counts[HOLD - 1 :]),
       "pan with BoT-SORT at 64 px: exactly 2 people published in every frame from frame 3")
expect(len({frozenset(s) for s in out[HOLD - 1 :]}) == 1,
       "pan with BoT-SORT at 64 px: their published ids never change")
report_pan("botsort.yaml, 128 px per frame (0.6 rad/s at 5 Hz)", *pan(pose, 128))

# Timings. Each image gets a fresh tracker: BoT-SORT's motion estimate fails on
# every later frame once the image size changes (it keeps the old frame), and
# falling back to no motion estimate would make it look cheaper than it is.
pose.reset_tracks()
yoloe.reset_tracks()
print("median ms over 10 calls (after a CUDA sync):")
print(f"  YOLOE detect, bus.jpg            {median_ms(lambda: yoloe.detect(bus)):7.1f}")
print(f"  pose estimate (botsort), bus.jpg {median_ms(lambda: pose.estimate(bus)):7.1f}")
pose.reset_tracks()
print(f"  pose estimate (botsort), HD2K    {median_ms(lambda: pose.estimate(canvas)):7.1f}")
print(f"  for information: the pan's median above is BoT-SORT on a textured HD2K image, {pan_ms:.1f} ms")
print(f"  faces_for, zidane.jpg, {sum(f is not None for f in found)} faces  "
      f"{median_ms(lambda: faces.faces_for(zidane, zidane_poses)):7.1f}")
print(f"  faces_for, HD2K canvas, {sum(f is not None for f in canvas_faces)} faces "
      f"{median_ms(lambda: faces.faces_for(canvas, canvas_poses)):7.1f}")
print(f"  OSNet embed, {len(crops)} crops of bus.jpg   {median_ms(lambda: reid.embed(crops)):7.1f}")

print("GPU memory with the node's four providers loaded:")
print(f"  nvidia-smi {gpu_mem_of()} | torch reserved {torch.cuda.memory_reserved() / 2**20:.0f} MiB"
      f" | torch max allocated {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")

# For information only: the same pans with ByteTrack, which has no motion compensation.
bytetrack = load("YoloPoseTracker (bytetrack.yaml)", lambda: YoloPoseTracker(ultra / "yolo26s-pose.pt",
                                                                               tracker="bytetrack.yaml"))
print("camera pan with ByteTrack, for information:")
report_pan("bytetrack.yaml, 64 px per frame", *pan(bytetrack, 64))
report_pan("bytetrack.yaml, 128 px per frame", *pan(bytetrack, 128))
bytetrack.reset_tracks()
print(f"  pose estimate (bytetrack), bus.jpg {median_ms(lambda: bytetrack.estimate(bus)):7.1f} ms median")
bytetrack.reset_tracks()
print(f"  pose estimate (bytetrack), HD2K    {median_ms(lambda: bytetrack.estimate(canvas)):7.1f} ms median")

finish()
