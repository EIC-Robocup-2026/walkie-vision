"""The memory task of check 12, on a recorded colour and depth pair.

`python checks/c12_vision_node.py --memory FIXTURE` runs this. It is an explicit
GPU check: run_checks.py and check-env do not start it, and it needs the model
files and a recording of your own. No recording ships with this repository.

The fixture is an .npz file with three arrays:

    color  H x W x 3 uint8, BGR
    depth  H x W float32, in metres, registered to the colour image
    k      the 9 values of the camera matrix

It prints the p50 and p95 of each stage, and the three DetectObjects cases with
the stages of each call. The budgets are printed for information and are never
a pass or a fail: a result has to be read against
docs/architecture/vision.md#unmeasured-budgets.

The fixture holds depth that is already read, so the depth stage, the pairing
and the node's queue are not measured here. The latency line on the robot
shows them.

Nothing here loads a model. c12_vision_node.py builds the providers, and the
tests call run_memory() with fakes.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn

import numpy as np
import yaml
from builtin_interfaces.msg import Time
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header

from walkie_vision.camera.depth import RegisteredDepth
from walkie_vision.convert import check_object_detection_array
from walkie_vision.node import VisionConfig
from walkie_vision.pipeline import OnDemandResult, Providers, TrackIdBase, VisionPipeline

MEMORY_TASK = Path(__file__).resolve().parents[1] / "config" / "tasks" / "memory.yaml"
FRAME_ID = "zed_head_left_camera_frame_optical"
FRAMES = 20
FIRST_CHECKED = 3  # frames are numbered from 1; a new track shows from frame 2
STAGES = ("objects", "mask", "pose", "faces", "reid", "lift", "crop", "embed", "build")
ON_DEMAND_STAGES = ("objects", "mask", "lift", "crop", "embed", "build")
"""The stages of a DetectObjects call: it runs no pose, face or reid model."""
NEW_CLASSES = ("mug", "laptop", "book", "remote", "plant")
"""A list of 5 classes that differs from the task's, as in the plan's report of about 1.5 s."""

GEOMETRY_TARGET_MS = 8.0
TARGET_HZ = 5.0
FIXTURE_REFUSED = 2


def p50_p95(values: list[float]) -> tuple[float, float]:
    return float(np.percentile(values, 50)), float(np.percentile(values, 95))


def _refuse(message: str) -> NoReturn:
    print(f"c12 --memory: {message}", file=sys.stderr, flush=True)
    sys.exit(FIXTURE_REFUSED)


def load_memory_fixture(path: Path) -> tuple[np.ndarray, RegisteredDepth, CameraInfo]:
    """Return a recording's image, depth and camera info. Exit with status 2 when it cannot be used."""
    try:
        with np.load(path) as data:
            color, metres, k = data["color"], data["depth"], data["k"]
    except Exception as e:  # numpy raises many types here, and every one means the same thing
        _refuse(f"cannot read the fixture {path}: {type(e).__name__}: {e}")
    if color.ndim != 3 or color.shape[2] != 3 or color.dtype != np.uint8:
        _refuse(f"color must be H x W x 3 uint8 BGR, not {color.shape} {color.dtype}")
    if metres.shape != color.shape[:2] or metres.dtype != np.float32:
        _refuse(f"depth must be float32 metres of the image's size {color.shape[:2]}, not {metres.shape} {metres.dtype}")
    if k.size != 9:
        _refuse(f"k must have 9 values, not {k.size}")
    try:
        k = np.asarray(k, dtype=np.float64).reshape(-1)
        depth = RegisteredDepth(metres, float(k[0]), float(k[4]), float(k[2]), float(k[5]))
    except ValueError as e:
        _refuse(f"k cannot be used: {e}")

    info = CameraInfo()
    info.header.frame_id = FRAME_ID
    info.height, info.width = color.shape[:2]
    info.distortion_model = "plumb_bob"
    info.d = [0.0] * 5
    info.k = k.tolist()
    info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    info.p = [k[0], 0.0, k[2], 0.0, 0.0, k[4], k[5], 0.0, 0.0, 0.0, 1.0, 0.0]
    return color, depth, info


def memory_config() -> VisionConfig:
    """Return the configuration of the memory task, from its parameter file."""
    params = yaml.safe_load(MEMORY_TASK.read_text())["walkie_vision"]["ros__parameters"]
    return VisionConfig(**{**params, "classes": tuple(params["classes"])})


def _header() -> Header:
    ns = time.time_ns()
    return Header(frame_id=FRAME_ID, stamp=Time(sec=ns // 10**9, nanosec=ns % 10**9))


def _on_demand_stages(result: OnDemandResult) -> str:
    """One DetectObjects call's stages. It embeds every usable box, not one in 10, so embed can dominate."""
    times = result.times
    split = " ".join(f"{stage} {getattr(times, stage):.1f}" for stage in ON_DEMAND_STAGES)
    return f"{split} | objects {len(result.objects.detections)}, sampled {result.sampled}"


def _array_problem(result) -> str | None:
    try:
        check_object_detection_array(result.objects)
    except ValueError as e:
        return str(e)
    return None


def run_memory(
    color: np.ndarray,
    depth: RegisteredDepth,
    info: CameraInfo,
    config: VisionConfig,
    providers: Providers,
    expect: Callable[[bool, str], bool],
    *,
    allow_fake_embedder: bool = False,
) -> None:
    """Run the stream and DetectObjects on one recorded pair, and report through expect()."""
    pipeline = VisionPipeline(
        providers,
        config.classes,
        TrackIdBase(),
        reid_min_crop_px=config.reid_min_crop_px,
        people_min_score=config.people_min_score,
        objects_min_score=config.objects_min_score,
        people_dedupe_iou=config.people_dedupe_iou,
        geometry_enabled=config.geometry_enabled,
        depth_min_m=config.depth_min_m,
        depth_max_m=config.depth_max_m,
        object_crops_enabled=config.object_crops_enabled,
        object_embeddings_enabled=config.object_embeddings_enabled,
        allow_fake_embedder=allow_fake_embedder,
        on_demand_enabled=config.on_demand_enabled,
    )
    stream_classes = pipeline.classes

    results = []
    process_ms: list[float] = []
    for n in range(1, FRAMES + 1):
        header = _header()
        t = time.perf_counter()
        result = pipeline.process(color, header, info, depth)
        process_ms.append((time.perf_counter() - t) * 1000.0)
        results.append(result)
        print(
            f"frame {n:2d}: {process_ms[-1]:6.1f} ms | objects {len(result.objects.detections)} "
            f"| sampled {result.sampled} | dropped {dict(result.dropped)}",
            flush=True,
        )

    checked = results[FIRST_CHECKED - 1 :]
    bad = [(n, p) for n, r in enumerate(results, 1) if (p := _array_problem(r))]
    expect(not bad, f"every object array follows the message rules{'' if not bad else f' (not: {bad[:3]})'}")
    objects = [d for r in checked for d in r.objects.detections]
    expect(bool(objects), f"from frame {FIRST_CHECKED}: the recording shows an object of the task's classes")
    expect(any(d.points.width > 0 for d in objects), f"from frame {FIRST_CHECKED}: an object has points")
    expect(sum(r.sampled for r in results) > 0, "the sampler took at least one object")

    print(f"latency over frames {FIRST_CHECKED} to {FRAMES}, ms (p50 / p95):")
    stage_p50 = {}
    for stage in STAGES:
        p50, p95 = p50_p95([getattr(r.times, stage) for r in checked])
        stage_p50[stage] = p50
        print(f"  {stage:9s} {p50:7.1f} / {p95:7.1f}")
    process_p50, process_p95 = p50_p95(process_ms[FIRST_CHECKED - 1 :])
    print(f"  {'process()':9s} {process_p50:7.1f} / {process_p95:7.1f}")

    def detect() -> tuple[OnDemandResult, float]:
        started = time.perf_counter()
        found = pipeline.detect_on_demand(color, _header(), info, depth)
        return found, (time.perf_counter() - started) * 1000.0

    first, first_ms = detect()
    same, same_ms = detect()
    started = time.perf_counter()
    pipeline.prepare_on_demand(NEW_CLASSES)
    prepare_ms = (time.perf_counter() - started) * 1000.0
    new, new_ms = detect()
    print("DetectObjects, ms:")
    print(f"  first call on the task's list {first_ms:7.1f} | {_on_demand_stages(first)}")
    print(f"  the same list again           {same_ms:7.1f} | {_on_demand_stages(same)}")
    print(f"  a new list of {len(NEW_CLASSES)}: prepare {prepare_ms:7.1f}, detect {new_ms:7.1f} | {_on_demand_stages(new)}")

    answers = (first, same, new)
    bad = [p for r in answers if (p := _array_problem(r))]
    expect(not bad, f"every DetectObjects array follows the message rules{'' if not bad else f' (not: {bad[:3]})'}")
    expect(
        all(d.track_id == 0 for r in answers for d in r.objects.detections), "every DetectObjects track id is 0"
    )
    expect(pipeline.on_demand_classes == list(NEW_CLASSES), "the on-demand detector holds the new list")
    after = pipeline.process(color, _header(), info, depth)
    expect(
        pipeline.classes == stream_classes and list(after.objects.classes) == stream_classes,
        "a new DetectObjects list leaves the stream's classes alone",
    )

    print("for information, never a pass or a fail here:")
    print(
        f"  mask + lift p50 {stage_p50['mask'] + stage_p50['lift']:.1f} ms; the engineering target is "
        f"{GEOMETRY_TARGET_MS:.0f} ms for all geometry at 5 detections and 2000 points "
        f"(this recording has {len(checked[-1].objects.detections)} detections)"
    )
    print(f"  process() p50 gives {1000.0 / max(process_p50, 1e-6):.1f} Hz at most; the target is {TARGET_HZ:.0f} Hz")
    print(f"  a new list stops the stream for {prepare_ms:.0f} ms; the plan reports about 1500 ms for 5 classes")
