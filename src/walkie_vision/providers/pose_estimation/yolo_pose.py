"""YOLO26 pose with a tracker, as a walkie_vision.providers pose estimator.

YoloPoseTracker is a PoseEstimator; its default tracker is BoT-SORT, whose
camera-motion compensation keeps person ids while the robot turns its base.
It keeps its tracker between calls and returns only tracked people, one
processed frame late for a new person; walkie_vision.providers._ultralytics
says why, and why it resets its tracker after the warm-up.

Ultralytics is imported only inside the constructor, so poses_from_arrays
can be imported and tested without it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .._ultralytics import check_rows, clip_box, require, reset_trackers, validate_tracker
from .base import COCO_KEYPOINTS, PersonPose


def poses_from_arrays(
    xyxy: np.ndarray,
    scores: np.ndarray,
    keypoints: np.ndarray,
    track_ids: np.ndarray | None,
    image_shape: tuple[int, int],
) -> list[PersonPose]:
    """Turn a pose tracker's boxes and keypoints into PersonPoses.

    keypoints has shape (n, 17, 3): x, y and score. The same rules as in
    detections_from_arrays apply to the boxes. Keypoints are clipped to the
    image too, and are never zeroed, whatever their score.
    """
    kp = np.asarray(keypoints)
    if kp.ndim != 3 or kp.shape[1:] != (len(COCO_KEYPOINTS), 3):
        raise ValueError(f"keypoints must have shape (n, {len(COCO_KEYPOINTS)}, 3), not {kp.shape}")
    if track_ids is None:
        return []
    xyxy = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    check_rows(len(xyxy), scores=scores, keypoints=kp, track_ids=track_ids)
    height, width = image_shape[:2]
    found = []
    for box, score, points, track_id in zip(xyxy, scores, kp, track_ids):
        if int(track_id) <= 0:
            continue
        clipped = clip_box(box, width, height)
        if clipped is None:
            continue
        xy = points[:, :2].astype(np.float32)
        xy[:, 0] = np.clip(xy[:, 0], 0, width)
        xy[:, 1] = np.clip(xy[:, 1], 0, height)
        found.append(
            PersonPose(
                box=clipped,
                score=float(score),
                keypoints=xy,
                keypoint_scores=points[:, 2].astype(np.float32),
                track_id=int(track_id),
            )
        )
    return found


class YoloPoseTracker:
    """YOLO26 pose with a tracker: a walkie_vision.providers.PoseEstimator.

    The default tracker is BoT-SORT (botsort.yaml): ByteTrack's two-stage
    matching plus global motion compensation from sparse optical flow, with
    no re-ID model. The robot turns its base while it follows one person;
    without the motion compensation a person's box can jump past the
    tracker's IoU gate and get a new id.
    """

    def __init__(
        self,
        weights: str | Path,
        *,
        device: str = "cuda:0",
        imgsz: int = 640,
        conf: float = 0.1,
        tracker: str = "botsort.yaml",
    ) -> None:
        self._tracker = validate_tracker(tracker)
        self._weights = Path(weights).resolve()
        require(self._weights)
        self._device = device
        self._imgsz = imgsz
        self._conf = conf

        from ultralytics import YOLO

        self._model = YOLO(str(self._weights))
        self.estimate(np.zeros((imgsz, imgsz, 3), np.uint8))
        self.reset_tracks()

    def reset_tracks(self) -> None:
        """Clear every track and restart the ids at 1.

        The warm-up used up the tracker's first frame. After the reset, the
        first real image activates its tracks at once.
        """
        reset_trackers(self._model)

    def estimate(self, image: np.ndarray) -> list[PersonPose]:
        """Return the tracked people in a BGR image, with their 17 COCO keypoints."""
        r = self._model.track(
            image,
            persist=True,
            tracker=self._tracker,
            conf=self._conf,
            imgsz=self._imgsz,
            device=self._device,
            verbose=False,
        )[0]
        b = r.boxes
        if b is None or len(b) == 0 or r.keypoints is None:
            return []
        return poses_from_arrays(
            b.xyxy.cpu().numpy(),
            b.conf.cpu().numpy(),
            r.keypoints.data.cpu().numpy(),
            None if b.id is None else b.id.int().cpu().numpy(),
            image.shape[:2],
        )
