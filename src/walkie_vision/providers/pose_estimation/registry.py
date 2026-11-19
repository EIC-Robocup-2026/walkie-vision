"""The pose estimators the vision node can build, by provider name.

PROVIDERS maps each name to its ProviderSpec (walkie_vision.providers.spec).
The provider classes are imported only inside the build functions, so
importing this module loads no model library.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .._ultralytics import weights_path
from ..spec import LoadContext, ProviderSpec
from .base import PoseEstimator


@dataclass(frozen=True)
class YoloPoseOptions:
    """YOLO26 pose's options: the weights file and the tracker's settings."""

    weights: str
    imgsz: int
    conf: float
    tracker: str


def _yolo_pose_files(options: YoloPoseOptions, models_dir: Path) -> tuple[Path, ...]:
    return (weights_path(options.weights, models_dir),)


def _build_yolo_pose(options: YoloPoseOptions, context: LoadContext, prepared: None) -> Callable[[], PoseEstimator]:
    from walkie_vision.providers.pose_estimation.yolo_pose import YoloPoseTracker

    weights = weights_path(options.weights, context.models_dir)
    return lambda: YoloPoseTracker(
        weights,
        device=context.device,
        imgsz=options.imgsz,
        conf=options.conf,
        tracker=options.tracker,
    )


PROVIDERS: dict[str, ProviderSpec[Any]] = {
    "yolo_pose": ProviderSpec(options=YoloPoseOptions, files=_yolo_pose_files, prepare=None, build=_build_yolo_pose),
}
