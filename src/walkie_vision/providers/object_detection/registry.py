"""The object detectors the vision node can build, by provider name.

PROVIDERS maps each name to its ProviderSpec (walkie_vision.providers.spec).
The provider classes are imported only inside the build functions, so
importing this module loads no model library; yoloe.py is imported here for
MOBILECLIP_FILE, since it loads Ultralytics only in its constructors.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .._ultralytics import weights_path
from ..spec import LoadContext, ProviderSpec
from .base import ObjectDetector
from .yoloe import MOBILECLIP_FILE


@dataclass(frozen=True)
class YoloeOptions:
    """YOLOE-26's options: the weights file and the tracker's settings."""

    weights: str
    imgsz: int
    conf: float
    agnostic_nms: bool
    tracker: str


def _yoloe_files(options: YoloeOptions, models_dir: Path) -> tuple[Path, ...]:
    weights = weights_path(options.weights, models_dir)
    # YOLOE loads its text encoder by file name from the folder of its weights.
    return (weights, weights.parent / MOBILECLIP_FILE)


def _masks(context: LoadContext) -> dict[str, Any]:
    # Passed only when asked for, so a detector otherwise gets its own default.
    return {"keep_masks": True} if context.keep_masks else {}


def _build_yoloe(options: YoloeOptions, context: LoadContext, prepared: None) -> Callable[[], ObjectDetector]:
    from walkie_vision.providers.object_detection.yoloe import YoloeTracker

    weights = weights_path(options.weights, context.models_dir)
    return lambda: YoloeTracker(
        weights,
        list(context.classes),
        device=context.device,
        imgsz=options.imgsz,
        conf=options.conf,
        agnostic_nms=options.agnostic_nms,
        tracker=options.tracker,
        **_masks(context),
    )


def _build_yoloe_detector(
    options: YoloeOptions, context: LoadContext, conf: float, max_det: int
) -> Callable[[], ObjectDetector]:
    from walkie_vision.providers.object_detection.yoloe import YoloeDetector

    weights = weights_path(options.weights, context.models_dir)
    return lambda: YoloeDetector(
        weights,
        list(context.classes),
        device=context.device,
        imgsz=options.imgsz,
        conf=conf,
        agnostic_nms=options.agnostic_nms,
        max_det=max_det,
        **_masks(context),
    )


PROVIDERS: dict[str, ProviderSpec[Any]] = {
    "yoloe": ProviderSpec(
        options=YoloeOptions,
        files=_yoloe_files,
        prepare=None,
        build=_build_yoloe,
        build_detector=_build_yoloe_detector,
    ),
}
