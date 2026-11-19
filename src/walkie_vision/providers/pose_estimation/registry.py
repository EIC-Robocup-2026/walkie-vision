"""The pose estimators the vision node can build, by provider name.

PROVIDERS maps each name to its ProviderSpec (walkie_vision.providers.spec),
and DEFAULT_PROVIDER is the one the vision node builds unless the parameter
pose_estimation.provider names another. The provider classes are imported
only inside the build functions, so importing this module loads no model
library.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .._ultralytics import check_options, weights_path
from ..spec import LoadContext, ProviderSpec
from .base import PoseEstimator

DEFAULT_PROVIDER = "yolo_pose"


@dataclass(frozen=True)
class YoloPoseOptions:
    """YOLO26 pose's options: the weights file and the tracker's settings.

    Each is the vision node's parameter pose_estimation.yolo_pose.<option>,
    and the defaults equal config/vision.yaml.
    """

    weights: str = "yolo26s-pose.pt"
    imgsz: int = 640
    conf: float = 0.1
    tracker: str = "botsort.yaml"

    def __post_init__(self) -> None:
        check_options("pose_estimation.yolo_pose", self.weights, self.imgsz, self.conf, self.tracker)


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
