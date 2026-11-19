"""The helpers that the Ultralytics providers share: YOLOE-26 in object_detection.yoloe and YOLO26 pose in pose_estimation.yolo_pose.

Both trackers keep their tracker between calls, and return only tracked boxes,
each with a non-zero track id. In Ultralytics 8.4 a tracker hides a new track
until it is confirmed on the next frame, so a new object or person appears
one processed frame late. The tracker's first frame after a reset is the
exception: it activates every track at once. That is why both classes reset
their tracker after the warm-up.

Nothing here imports Ultralytics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..weights import ULTRALYTICS_DIR

# The tracker files that ship with Ultralytics, by name.
TRACKER_NAMES = ("bytetrack.yaml", "botsort.yaml")


def clip_box(box: np.ndarray, width: int, height: int) -> np.ndarray | None:
    """Clip an x1, y1, x2, y2 box to the image as float64; None when no area is left."""
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float64)
    x1, x2 = np.clip([x1, x2], 0, width)
    y1, y2 = np.clip([y1, y2], 0, height)
    if x2 <= x1 or y2 <= y1:
        return None
    return np.array([x1, y1, x2, y2], dtype=np.float64)


def check_rows(n: int, **arrays: np.ndarray | None) -> None:
    for name, array in arrays.items():
        if array is not None and len(array) != n:
            raise ValueError(f"{name} has {len(array)} rows, but there are {n} boxes")


def validate_tracker(tracker: str) -> str:
    """Return the tracker if it is a known file name or an absolute path to a .yaml file, else raise ValueError."""
    if tracker in TRACKER_NAMES:
        return tracker
    path = Path(tracker)
    if path.is_absolute() and path.suffix == ".yaml" and path.is_file():
        return tracker
    raise ValueError(
        f"tracker {tracker!r} must be one of {', '.join(TRACKER_NAMES)} or an absolute path to an existing .yaml file"
    )


def weights_path(name: str, models_dir: Path) -> Path:
    """Return the path of Ultralytics weights: name when it is absolute, else name in models/ultralytics."""
    path = Path(name)
    return path if path.is_absolute() else (models_dir / ULTRALYTICS_DIR).absolute() / path


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run `pixi run fetch-models`")


def reset_trackers(model: Any) -> None:
    for tracker in getattr(getattr(model, "predictor", None), "trackers", []):
        tracker.reset()
