"""Check 12: the vision node's own start path and 20 HD2K frames in one process.

build_providers() from walkie_vision.node loads every model the way the node
does, and VisionPipeline runs them in turn on each frame: YOLOE, pose, SCRFD
and ArcFace on head crops, then OSNet. The input is zidane.jpg resized to the
ZED's HD2K (2208 x 1242) as a bgra8 sensor_msgs/Image, read back with
image_to_bgr() as the node reads its camera.

The check creates no ROS node and does not call rclpy.init(): it only builds
messages. It prints the start time, the stage latencies and the GPU memory,
which ADR-0010 cites. The latency target is checked on the laptop only.
"""

from __future__ import annotations

import os

# As the vision node's launch does: no online checks from Ultralytics.
os.environ.setdefault("YOLO_OFFLINE", "true")

import array
import logging
import math
import time

import cv2
import numpy as np
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
from walkie_vision_msgs.msg import ObjectDetectionArray, PersonDetectionArray

from common import assets_dir, expect, finish, gpu_mem_of
from walkie_vision.image import image_to_bgr
from walkie_vision.node import VisionConfig, build_providers
from walkie_vision.pipeline import FrameResult, TrackIdBase, VisionPipeline

WIDTH, HEIGHT = 2208, 1242
FRAMES = 20
FIRST_CHECKED = 3  # frames are numbered from 1; a new track shows from frame 2
FRAME_ID = "zed_head_left_camera_frame_optical"
STAGES = ("objects", "pose", "faces", "reid", "build")
TARGET_P95_MS = 300.0


def nonfinite(msg: object, path: str = "") -> list[str]:
    """Return the path of every float field in a message, at any depth, that is not finite."""
    bad = []
    for name, kind in msg.get_fields_and_field_types().items():
        value = getattr(msg, name)
        where = f"{path}.{name}" if path else name
        if kind in ("float", "double"):
            if not math.isfinite(value):
                bad.append(where)
        elif kind.startswith(("float[", "double[", "sequence<float>", "sequence<double>")):
            values = np.asarray(value, dtype=np.float64)
            if values.size and not np.isfinite(values).all():
                bad.append(where)
        elif hasattr(value, "get_fields_and_field_types"):
            bad += nonfinite(value, where)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if hasattr(item, "get_fields_and_field_types"):
                    bad += nonfinite(item, f"{where}[{i}]")
    return bad


def p50_p95(values: list[float]) -> tuple[float, float]:
    return float(np.percentile(values, 50)), float(np.percentile(values, 95))


logging.basicConfig(level=logging.INFO)
config = VisionConfig(classes=("person", "tie"))
t0 = time.perf_counter()
providers = build_providers(config, logging.getLogger("c12"))
start_s = time.perf_counter() - t0
print(f"build_providers: {start_s:.1f} s", flush=True)

import torch  # noqa: E402  (after the start, which imports it itself, as the node does)

print(f"  after the start: nvidia-smi {gpu_mem_of()} | torch reserved "
      f"{torch.cuda.memory_reserved() / 2**20:.0f} MiB", flush=True)

pipeline = VisionPipeline(
    providers,
    config.classes,
    TrackIdBase(),
    reid_min_crop_px=config.reid_min_crop_px,
    people_min_score=config.people_min_score,
    objects_min_score=config.objects_min_score,
    people_dedupe_iou=config.people_dedupe_iou,
)

zidane = cv2.imread(str(assets_dir() / "zidane.jpg"))
bgra = cv2.cvtColor(cv2.resize(zidane, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2BGRA)
image = Image()
image.header.frame_id = FRAME_ID
image.height, image.width = HEIGHT, WIDTH
image.encoding = "bgra8"
image.is_bigendian = 0
image.step = WIDTH * 4
image.data = array.array("B", bgra.tobytes())

info = CameraInfo()
info.header.frame_id = FRAME_ID
info.height, info.width = HEIGHT, WIDTH
info.distortion_model = "plumb_bob"
info.d = [0.0] * 5
info.k = [1066.0, 0.0, 1104.0, 0.0, 1066.0, 621.0, 0.0, 0.0, 1.0]
info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
info.p = [1066.0, 0.0, 1104.0, 0.0, 0.0, 1066.0, 621.0, 0.0, 0.0, 0.0, 1.0, 0.0]

results: list[FrameResult] = []
convert_ms: list[float] = []
process_ms: list[float] = []
for n in range(1, FRAMES + 1):
    ns = time.time_ns()
    image.header.stamp.sec, image.header.stamp.nanosec = ns // 10**9, ns % 10**9
    t = time.perf_counter()
    bgr = image_to_bgr(image)
    convert_ms.append((time.perf_counter() - t) * 1000.0)
    header = Header(frame_id=image.header.frame_id, stamp=image.header.stamp)
    t = time.perf_counter()
    result = pipeline.process(bgr, header, info)
    process_ms.append((time.perf_counter() - t) * 1000.0)
    results.append(result)
    people = result.people.people
    labels = [d.label for d in result.objects.detections]
    print(f"frame {n:2d}: {process_ms[-1]:6.1f} ms | people {len(people)} ids "
          f"{sorted(p.track_id for p in people)} face scores {[round(p.face_det_score, 2) for p in people]}"
          f" | objects {sorted(labels)} | dropped {dict(result.dropped)} | duplicates {result.duplicates}",
          flush=True)

expect(bgr.shape == (HEIGHT, WIDTH, 3), f"image_to_bgr gives {HEIGHT} x {WIDTH} x 3 ({bgr.shape})")
expect(all(isinstance(r.people, PersonDetectionArray) and isinstance(r.objects, ObjectDetectionArray)
           for r in results), "every frame returns both arrays")
bad = [(n, path) for n, r in enumerate(results, 1) for path in nonfinite(r.people) + nonfinite(r.objects)]
expect(not bad, f"every float in both arrays is finite{'' if not bad else f' (not: {bad[:5]})'}")
expect(all(r.people.header.frame_id == FRAME_ID and r.people.camera_info.header.frame_id == FRAME_ID
           and r.objects.header == r.people.header for r in results),
       "both arrays and their camera_info carry the image's header")

analyzer = providers.faces.analyzer
for label, sess in (("detector", analyzer.detector.session), ("recognizer", analyzer.recognizer.session)):
    expect(sess.get_providers()[0] == "CUDAExecutionProvider",
           f"after the loop the face {label} still runs on CUDAExecutionProvider")

checked = results[FIRST_CHECKED - 1 :]
people_per_frame = [r.people.people for r in checked]
expect(all(len(p) == 2 for p in people_per_frame),
       f"from frame {FIRST_CHECKED}: exactly 2 people in every frame ({[len(p) for p in people_per_frame]})")
id_sets = {frozenset(p.track_id for p in people) for people in people_per_frame}
expect(len(id_sets) == 1 and all(i != 0 for s in id_sets for i in s),
       f"from frame {FIRST_CHECKED}: non-zero track ids that do not change ({[sorted(s) for s in id_sets]})")
everyone = [p for people in people_per_frame for p in people]
expect(all(p.face_det_score >= 0.5 for p in everyone), f"from frame {FIRST_CHECKED}: every face_det_score >= 0.5")
expect(all(p.face_model == "arcface_w600k_r50" for p in everyone),
       f"from frame {FIRST_CHECKED}: every face_model is arcface_w600k_r50")
norms = [float(np.linalg.norm(np.asarray(p.face_embedding, dtype=np.float64))) for p in everyone]
expect(all(len(p.face_embedding) == 512 for p in everyone) and all(abs(v - 1.0) <= 1e-3 for v in norms),
       f"from frame {FIRST_CHECKED}: every face_embedding has 512 values and norm 1 "
       f"(norms {min(norms, default=0):.4f}-{max(norms, default=0):.4f})")
expect(all(len(p.appearance_embedding) == 512 for p in everyone),
       f"from frame {FIRST_CHECKED}: every appearance_embedding has 512 values")
person_objects = [sum(d.label == "person" for d in r.objects.detections) for r in checked]
expect(all(c >= 2 for c in person_objects),
       f"from frame {FIRST_CHECKED}: at least 2 objects labelled person ({person_objects})")
expect(all(not r.dropped for r in results), "FrameResult.dropped is empty in every frame")

print(f"latency over frames {FIRST_CHECKED} to {FRAMES}, ms (p50 / p95):")
for stage in STAGES:
    p50, p95 = p50_p95([getattr(r.times, stage) for r in checked])
    print(f"  {stage:9s} {p50:7.1f} / {p95:7.1f}")
p50, p95 = p50_p95(process_ms[FIRST_CHECKED - 1 :])
print(f"  {'process()':9s} {p50:7.1f} / {p95:7.1f}")
c50, c95 = p50_p95(convert_ms[FIRST_CHECKED - 1 :])
print(f"  for information, image_to_bgr: {c50:.1f} / {c95:.1f}")
print(f"  for information (the laptop, not the robot): the p95 of process() is "
      f"{'under' if p95 < TARGET_P95_MS else 'not under'} {TARGET_P95_MS:.0f} ms")

print("GPU memory after the loop:")
print(f"  torch reserved {torch.cuda.memory_reserved() / 2**20:.0f} MiB | torch max allocated "
      f"{torch.cuda.max_memory_allocated() / 2**20:.0f} MiB | nvidia-smi {gpu_mem_of()}")

finish()
