"""Check 5: YOLOE-26 open-vocabulary segmentation and YOLO26 pose on the GPU.

Usage: python c05_ultralytics.py [yoloe|pose|track]

The weights come from models/ultralytics (`pixi run fetch-models`). The check
runs from that directory: Ultralytics looks for a weights file in the working
directory first, and YOLOE.set_classes() loads its text encoder,
mobileclip2_b.ts, by file name. Elsewhere Ultralytics would download the files
again.
"""

from __future__ import annotations

import os
import statistics
import sys
import time

import torch

from common import assets_dir, expect, finish, gpu_mem_of
from walkie_vision.models import ULTRALYTICS_DIR, models_dir

which = sys.argv[1] if len(sys.argv) > 1 else "yoloe"
bus = str(assets_dir() / "bus.jpg")
os.chdir(models_dir() / ULTRALYTICS_DIR)
print("working directory", os.getcwd(), sorted(os.listdir()))


def bench(model, src, n=20, **kw):
    """Return the last result and the median end-to-end time in ms, after one warm-up."""
    t0 = time.perf_counter()
    model.predict(src, device=0, verbose=False, **kw)
    first = (time.perf_counter() - t0) * 1000
    times = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = model.predict(src, device=0, verbose=False, **kw)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return r[0], statistics.median(times), first


def names(r) -> list[str]:
    return [r.names[int(c)] for c in r.boxes.cls]


if which == "yoloe":
    from ultralytics import YOLOE

    model = YOLOE("yoloe-26s-seg.pt")
    print("weights", model.ckpt_path)
    household = ["cup", "bottle", "apple", "sponge", "chair"]
    t0 = time.perf_counter()
    model.set_classes(household)
    print(f"set_classes({household}) took {(time.perf_counter() - t0) * 1000:.0f} ms")
    model.set_classes(["person", "bus"])
    r32, med32, first = bench(model, bus)
    print(f"yoloe-26s-seg fp32 on bus.jpg: first call {first:.0f} ms, then median {med32:.1f} ms; "
          f"{names(r32)}, masks {None if r32.masks is None else tuple(r32.masks.data.shape)}; speed {r32.speed}")
    expect(names(r32).count("person") >= 4 and "bus" in names(r32), "fp32 finds at least 4 people and the bus")
    expect(r32.masks is not None and len(r32.masks) == len(r32.boxes), "fp32 returns one mask per box")

    # The predictor is built on the first predict() call and reused, so fp16
    # needs a fresh model object. Ultralytics 8.4 replaced half=True with
    # quantize=16 and logs a deprecation warning on every half=True call.
    model = YOLOE("yoloe-26s-seg.pt")
    model.set_classes(["person", "bus"])
    r16, med16, first = bench(model, bus, quantize=16)
    print(f"yoloe-26s-seg quantize=16: first call {first:.0f} ms, then median {med16:.1f} ms; {names(r16)}; "
          f"predictor fp16 flag {getattr(model.predictor.model, 'fp16', None)}")
    same = sorted(names(r16)) == sorted(names(r32))
    expect(same, "fp16 finds the same objects as fp32")
    if same and r16.masks is not None:
        # Each fp16 mask against its best-matching fp32 mask.
        m16 = r16.masks.data.float()[:, None] > 0.5
        m32 = r32.masks.data.float()[None] > 0.5
        inter = (m16 & m32).sum((2, 3)).float()
        union = (m16 | m32).sum((2, 3)).float().clamp(min=1)
        iou = (inter / union).max(dim=1).values.tolist()
        print("  mask IoU fp16 vs fp32:", [round(v, 3) for v in iou])
        expect(min(iou) > 0.9, "fp16 masks match fp32 (IoU > 0.9)")

elif which == "pose":
    from ultralytics import YOLO

    for label, kw in (("fp32", {}), ("quantize=16", {"quantize": 16})):
        # A fresh model for each precision; see the fp16 note under yoloe.
        model = YOLO("yolo26s-pose.pt")
        print("weights", model.ckpt_path)
        r, med, first = bench(model, bus, **kw)
        kp = r.keypoints
        shape = tuple(kp.data.shape) if kp is not None else None
        print(f"yolo26s-pose {label} on bus.jpg: first call {first:.0f} ms, then median {med:.1f} ms; "
              f"{len(r.boxes)} people, keypoints {shape}; predictor fp16 flag "
              f"{getattr(model.predictor.model, 'fp16', None)}; speed {r.speed}")
        expect(len(r.boxes) >= 4 and shape is not None and shape[1:] == (17, 3),
               f"{label} finds at least 4 people with 17 keypoints each")

elif which == "track":
    # Tracking needs lap; with YOLO_AUTOINSTALL=false a missing lap fails here.
    from ultralytics import YOLO

    model = YOLO("yolo26s-pose.pt")
    for _ in range(3):
        r = model.track(bus, device=0, persist=True, verbose=False, tracker="bytetrack.yaml")
    ids = None if r[0].boxes.id is None else r[0].boxes.id.int().tolist()
    print("track ids", ids)
    expect(ids is not None and len(ids) >= 4, "ByteTrack assigns an id to every person")

print("GPU memory (nvidia-smi):", gpu_mem_of(),
      f"| torch max allocated {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
finish()
