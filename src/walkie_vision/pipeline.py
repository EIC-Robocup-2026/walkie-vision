"""One image through every vision model, to one people array and one object array.

VisionPipeline runs the providers in turn on one BGR image: objects, pose,
the faces of the people it found, and their appearance. It turns the
results into the two arrays that the vision node publishes for that image,
with the image's header and a copy of its camera_info. It has no ROS node
and imports no model library, so the tests run it with the fakes.

With geometry_enabled, a detection with a mask also gets its points and its
view quality (walkie_vision.depth). Without it, masks are ignored and no
geometry work runs.

With object_crops_enabled or object_embeddings_enabled, up to three detections
per image are sampled (walkie_vision.embeddings). The two flags are separate,
and one sampler serves both. A sampled detection gets its crop, its vector or
both, and the colour shares of its mask. The time counts in the build stage.
With both off, no embedder runs, and no JPEG or colour share is computed.

With on_demand_enabled, detect_on_demand() runs a separate detector on one image
for DetectObjects. It returns one array to its caller, and it does not touch the
streaming detector, its classes, its track ids or the sampler's counter.

A single bad value from a model (a box or a vector that is not finite, or an
object score outside [0, 1]) leaves out that field or that detection, and is
counted in FrameResult.dropped. A provider that raises, or that returns the wrong
number or shape of results, makes process() raise: the caller then
publishes neither array for that image.

YOLO26 pose is end-to-end: it has no NMS, and it sometimes gives one person
a second, partial box, which the tracker then confirms as its own track.
process() leaves out a pose whose box overlaps a higher-scoring pose's box
by more than people_dedupe_iou (dedupe_poses), and counts it in
FrameResult.duplicates.
"""

from __future__ import annotations

import collections
import copy
import dataclasses
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from sensor_msgs.msg import CameraInfo, CompressedImage, PointCloud2
from std_msgs.msg import Header
from walkie_vision_msgs.msg import ObjectDetection, ObjectDetectionArray, PersonDetectionArray

from . import convert, embeddings
from .boxes import resize_mask
from .depth import RegisteredDepth, depth_cover, frame_laplacian, lift_mask, view_quality
from .providers import (
    FAKE_MODEL_PREFIX,
    Detection,
    Embedder,
    Face,
    ObjectDetector,
    PersonFaceRecognizer,
    PersonPose,
    PoseEstimator,
    ReIdentifier,
)

TRACK_ID_SPACE = 2**32 - 1
"""The number of non-zero uint32 values, which a published track id can take."""
ON_DEMAND_MAX_OBJECTS = 20
"""DetectObjects returns and works on at most this many detections, those with the highest scores."""


class TrackIdBase:
    """Maps a tracker's ids (1, 2, ...) to uint32 ids that a restart does not reuse.

    The offset is the Unix time in ms when the node started. A later run
    starts after the earlier one ended, so its offset is larger by at least
    the earlier run's lifetime in ms. A tracker issues far fewer than 1000
    ids a second, so the two ranges never meet. The space wraps after
    2^32 - 1 ms, about 49.7 days. A system clock that steps back further than
    the time between two starts could reuse ids.

    Wall time, not the ROS clock: sim time restarts at 0.
    """

    def __init__(self, start_ns: int | None = None) -> None:
        ns = time.time_ns() if start_ns is None else int(start_ns)
        self.offset: int = (ns // 1_000_000) % TRACK_ID_SPACE

    def map(self, local: int) -> int:
        """Return the published id of a tracker's id; 0 (not tracked) stays 0."""
        local = int(local)
        if local < 0:
            raise ValueError(f"a track id must be 0 or positive, not {local}")
        if local == 0:
            return 0
        return ((self.offset + local - 1) % TRACK_ID_SPACE) + 1


@dataclass(frozen=True)
class Providers:
    """The models the vision node runs.

    embedder is set only with object embeddings on, and on_demand only with
    on-demand detection on. on_demand is never the same object as objects.
    """

    objects: ObjectDetector
    pose: PoseEstimator
    faces: PersonFaceRecognizer
    reid: ReIdentifier
    embedder: Embedder | None = None
    on_demand: ObjectDetector | None = None


@dataclass(frozen=True)
class _Sample:
    """What sampling adds to one detection. All empty for a detection that was not sampled."""

    crop: CompressedImage | None = None
    embedding: np.ndarray | None = None
    embedding_model: str = ""
    color_shares: np.ndarray | None = None
    chosen: bool = False


@dataclass
class StageTimes:
    """How long each stage of one image took, in milliseconds.

    mask is the part of the detector's call that copies and resizes masks, and
    objects leaves it out. lift, crop and embed are parts of building the arrays,
    and build leaves them out. A stage whose addition is off stays 0.
    """

    objects: float = 0.0
    mask: float = 0.0
    pose: float = 0.0
    faces: float = 0.0
    reid: float = 0.0
    lift: float = 0.0
    crop: float = 0.0
    embed: float = 0.0
    build: float = 0.0


@dataclass
class FrameResult:
    """The two arrays for one image, the stage times, and what was left out."""

    people: PersonDetectionArray
    objects: ObjectDetectionArray
    times: StageTimes
    dropped: collections.Counter[str] = field(default_factory=collections.Counter)
    """Bad fields or detections left out of this image: object, person, face, appearance, embedding."""
    duplicates: int = 0
    """Poses left out of this image as a second box of a person already in it."""
    sampled: int = 0
    """Objects of this image chosen for a crop or a vector."""


@dataclass
class OnDemandResult:
    """One DetectObjects array, its stage times, and the bad fields or detections left out of it."""

    objects: ObjectDetectionArray
    times: StageTimes = field(default_factory=StageTimes)
    dropped: collections.Counter[str] = field(default_factory=collections.Counter)
    sampled: int = 0
    """Objects of this array chosen for a crop or a vector."""


def _check_classes(classes: Sequence[str]) -> list[str]:
    classes = list(classes)
    if not classes:
        raise ValueError("classes must not be empty")
    for c in classes:
        if not isinstance(c, str) or not c or c != c.strip():
            raise ValueError(f"every class must be a non-empty string without surrounding spaces, not {c!r}")
    if len(set(classes)) != len(classes):
        raise ValueError(f"classes has duplicates: {classes}")
    return classes


def _box_ok(box: np.ndarray) -> bool:
    """True for a finite x1, y1, x2, y2 box that is not inverted. Raises for a box that is not 4 values."""
    b = np.asarray(box, dtype=np.float64)
    if b.shape != (4,):
        raise ValueError(f"a box must have 4 values, not shape {b.shape}")
    return bool(np.all(np.isfinite(b)) and b[2] >= b[0] and b[3] >= b[1])


def _finite(value: float) -> bool:
    return bool(np.isfinite(float(value)))


def _vector_ok(vector: np.ndarray | None) -> bool:
    """True for a 1-D, non-empty, finite and non-zero vector."""
    if vector is None:
        return False
    v = np.asarray(vector, dtype=np.float64)
    return bool(v.ndim == 1 and v.size > 0 and np.all(np.isfinite(v)) and np.any(v != 0))


def _pose_ok(pose: PersonPose) -> bool:
    if not _box_ok(pose.box) or not _finite(pose.score):
        return False
    return bool(
        np.all(np.isfinite(np.asarray(pose.keypoints, dtype=np.float64)))
        and np.all(np.isfinite(np.asarray(pose.keypoint_scores, dtype=np.float64)))
    )


def _face_ok(face: Face) -> bool:
    return _box_ok(face.bbox) and _finite(face.det_score) and _vector_ok(face.embedding)


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two x1, y1, x2, y2 boxes; 0 when the union has no area."""
    ax1, ay1, ax2, ay2 = np.asarray(a, dtype=np.float64)
    bx1, by1, bx2, by2 = np.asarray(b, dtype=np.float64)
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / union) if union > 0 else 0.0


def dedupe_poses(poses: Sequence[PersonPose], max_iou: float) -> list[PersonPose]:
    """Return the poses without the second boxes of one person, in their input order.

    Going from the highest score down, a pose is left out when its box has
    an IoU above max_iou with the box of a pose already kept. On equal
    scores the earlier pose is kept. max_iou 1.0 keeps every pose.
    """
    order = sorted(range(len(poses)), key=lambda i: -float(poses[i].score))
    kept: list[int] = []
    for i in order:
        if all(_box_iou(poses[i].box, poses[j].box) <= max_iou for j in kept):
            kept.append(i)
    return [poses[i] for i in sorted(kept)]


class VisionPipeline:
    """Runs the providers on one image and builds its two arrays.

    The detector's classes are set once, here, when they differ from the
    configured list. Nothing else calls set_classes(): it costs about 1.5 s
    and restarts the detector's track ids.

    A fake embedder needs allow_fake_embedder, which only an injected set of
    providers gets. The real path can then never publish a fake's vectors.

    The on-demand detector's classes change with the requests, through
    prepare_on_demand(). It must be its own object, so a request can never reach
    the stream's detector.
    """

    def __init__(
        self,
        providers: Providers,
        classes: Sequence[str],
        track_ids: TrackIdBase,
        *,
        reid_min_crop_px: int = 16,
        people_min_score: float = 0.0,
        objects_min_score: float = 0.0,
        people_dedupe_iou: float = 1.0,
        stream_id: str | None = None,
        geometry_enabled: bool = False,
        depth_min_m: float = 0.2,
        depth_max_m: float = 3.0,
        object_crops_enabled: bool = False,
        object_embeddings_enabled: bool = False,
        allow_fake_embedder: bool = False,
        on_demand_enabled: bool = False,
    ) -> None:
        self._classes = _check_classes(classes)
        if on_demand_enabled:
            if providers.on_demand is None:
                raise ValueError("on_demand_enabled needs an on-demand detector")
            if providers.on_demand is providers.objects:
                raise ValueError("the on-demand detector must not be the streaming detector")
        if object_embeddings_enabled:
            if providers.embedder is None:
                raise ValueError("object_embeddings_enabled needs an embedder")
            if providers.embedder.model_name.startswith(FAKE_MODEL_PREFIX) and not allow_fake_embedder:
                raise ValueError(f"the fake embedder {providers.embedder.model_name} must not run with the real models")
            convert.check_model_dim(providers.embedder)
        if not 0.0 < depth_min_m < depth_max_m < float("inf"):
            raise ValueError(f"the depth range must be positive and finite, not {depth_min_m} to {depth_max_m}")
        for name, value in (("people_min_score", people_min_score), ("objects_min_score", objects_min_score)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], not {value}")
        if not 0.0 < people_dedupe_iou <= 1.0:
            raise ValueError(f"people_dedupe_iou must be in (0, 1], not {people_dedupe_iou}")
        if reid_min_crop_px < 1:
            raise ValueError(f"reid_min_crop_px must be at least 1, not {reid_min_crop_px}")
        self._providers = providers
        self._ids = track_ids
        self._reid_min_crop_px = int(reid_min_crop_px)
        self._people_min_score = float(people_min_score)
        self._objects_min_score = float(objects_min_score)
        self._people_dedupe_iou = float(people_dedupe_iou)
        self._stream_id = convert.new_stream_id() if stream_id is None else stream_id
        self._geometry_enabled = bool(geometry_enabled)
        self._depth_min_m = float(depth_min_m)
        self._depth_max_m = float(depth_max_m)
        self._crops_enabled = bool(object_crops_enabled)
        self._embeddings_enabled = bool(object_embeddings_enabled)
        self._processed_images = 0
        if list(providers.objects.classes) != self._classes:
            providers.objects.set_classes(list(self._classes))

    @property
    def classes(self) -> list[str]:
        """The detector's text prompts, as configured."""
        return list(self._classes)

    @property
    def stream_id(self) -> str:
        """The id on every object array of this pipeline. A new process gets a new one."""
        return self._stream_id

    @property
    def processed_images(self) -> int:
        """Images that process() has returned arrays for. The sampler counts on it."""
        return self._processed_images

    @property
    def on_demand_classes(self) -> list[str]:
        """The classes the on-demand detector looks for now: the last list it prepared."""
        return list(self._providers.on_demand.classes)

    def prepare_on_demand(self, classes: Sequence[str]) -> None:
        """Make the on-demand detector look for these classes. Can take about 1.5 s. Raises when it fails."""
        self._providers.on_demand.set_classes(_check_classes(classes))

    def detect_on_demand(
        self, image: np.ndarray, header: Header, camera_info: CameraInfo, depth: RegisteredDepth | None = None
    ) -> OnDemandResult:
        """Run the on-demand detector on one BGR image and return its array for the caller.

        The array has the image's header, a copy of camera_info, the detector's
        classes and this pipeline's stream id. Every track id is 0. Only the
        highest ON_DEMAND_MAX_OBJECTS scores are kept, before any later work.
        Each kept detection with a large enough box gets what the flags allow:
        its crop, its vector and its colour shares, with no 1-in-10 sampling.
        Raises when a provider raises or returns the wrong shape.
        """
        detector = self._providers.on_demand
        times = StageTimes()
        dropped: collections.Counter[str] = collections.Counter()
        t = time.perf_counter()
        found: list[Detection] = []
        for det in detector.detect(image):
            if not _box_ok(det.box) or not 0.0 <= float(det.score) <= 1.0:
                dropped["object"] += 1
            elif det.score >= self._objects_min_score:
                found.append(dataclasses.replace(det, track_id=0))
        found.sort(key=lambda det: -float(det.score))
        kept = found[:ON_DEMAND_MAX_OBJECTS]
        _split_detector_time(times, detector, _ms_since(t))

        t = time.perf_counter()
        info = copy.deepcopy(camera_info)
        info.header = header
        height, width = image.shape[:2]
        found_geometry = self._geometry(image, header, kept, depth, times)
        started = time.perf_counter()
        samples = self._fill_samples(image, header, kept, embeddings.usable_samples(kept, width, height), dropped, times)
        times.crop = max(0.0, _ms_since(started) - times.embed)
        objects = self._to_objects(kept, found_geometry, samples)
        array = convert.to_object_detection_array(header, info, detector.classes, objects, stream_id=self._stream_id)
        times.build = max(0.0, _ms_since(t) - times.lift - times.crop - times.embed)
        return OnDemandResult(
            objects=array, times=times, dropped=dropped, sampled=sum(sample.chosen for sample in samples)
        )

    def process(
        self, image: np.ndarray, header: Header, camera_info: CameraInfo, depth: RegisteredDepth | None = None
    ) -> FrameResult:
        """Run every model on one BGR image and return its two arrays.

        header is the image's own header; both arrays and the copy of
        camera_info get it. depth is the image's checked depth, if it has one.
        Raises when a provider raises or returns the wrong number or shape of
        results.
        """
        p = self._providers
        times = StageTimes()
        dropped: collections.Counter[str] = collections.Counter()

        t = time.perf_counter()
        detections: list[Detection] = []
        for det in p.objects.detect(image):
            if not _box_ok(det.box) or not 0.0 <= float(det.score) <= 1.0:
                dropped["object"] += 1
            elif det.score >= self._objects_min_score:
                detections.append(det)
        _split_detector_time(times, p.objects, _ms_since(t))

        t = time.perf_counter()
        # The floor and the dedupe come before faces and reid, so no work is
        # spent on people that are not published, and a second box of one
        # person cannot take that person's face.
        floored: list[PersonPose] = []
        for pose in p.pose.estimate(image):
            if not _pose_ok(pose):
                dropped["person"] += 1
            elif pose.score >= self._people_min_score:
                floored.append(pose)
        poses = dedupe_poses(floored, self._people_dedupe_iou)
        times.pose = _ms_since(t)

        t = time.perf_counter()
        faces: list[Face | None] = []
        if poses:
            faces = list(p.faces.faces_for(image, poses))
            if len(faces) != len(poses):
                raise ValueError(f"the face model returned {len(faces)} entries for {len(poses)} people")
            for i, face in enumerate(faces):
                if face is not None and not _face_ok(face):
                    faces[i] = None
                    dropped["face"] += 1
        times.faces = _ms_since(t)

        t = time.perf_counter()
        appearances = self._appearances(image, poses, dropped)
        times.reid = _ms_since(t)

        t = time.perf_counter()
        info = copy.deepcopy(camera_info)
        info.header = header
        people = [
            convert.to_person_detection(
                dataclasses.replace(pose, track_id=self._ids.map(pose.track_id)),
                face=face,
                face_model=p.faces.model_name if face is not None else "",
                appearance=vector,
                appearance_model=p.reid.model_name if vector is not None else "",
            )
            for pose, face, vector in zip(poses, faces, appearances, strict=True)
        ]
        published = [dataclasses.replace(det, track_id=self._ids.map(det.track_id)) for det in detections]
        found_geometry = self._geometry(image, header, detections, depth, times)
        samples = self._samples(image, header, published, dropped, times)
        objects = self._to_objects(published, found_geometry, samples)
        people_array = convert.to_person_detection_array(header, info, people)
        objects_array = convert.to_object_detection_array(
            header, info, self._classes, objects, stream_id=self._stream_id
        )
        times.build = max(0.0, _ms_since(t) - times.lift - times.crop - times.embed)
        self._processed_images += 1
        return FrameResult(
            people=people_array,
            objects=objects_array,
            times=times,
            dropped=dropped,
            duplicates=len(floored) - len(poses),
            sampled=sum(sample.chosen for sample in samples),
        )

    @staticmethod
    def _to_objects(
        detections: Sequence[Detection],
        geometry: Sequence[tuple[PointCloud2 | None, float]],
        samples: Sequence[_Sample],
    ) -> list[ObjectDetection]:
        """Return the message of each detection, with its geometry and its sample."""
        return [
            convert.to_object_detection(
                det,
                points=points,
                view_quality=quality,
                crop=sample.crop,
                embedding=sample.embedding,
                embedding_model=sample.embedding_model,
                color_shares=sample.color_shares,
            )
            for det, (points, quality), sample in zip(detections, geometry, samples, strict=True)
        ]

    def _geometry(
        self,
        image: np.ndarray,
        header: Header,
        detections: Sequence[Detection],
        depth: RegisteredDepth | None,
        times: StageTimes,
    ) -> list[tuple[PointCloud2 | None, float]]:
        """Return each detection's cloud and view quality; (None, 0) without a mask or with geometry off.

        Without depth, the clouds stay empty and the quality's depth term is
        0. Raises for a mask that does not have the image's size. Records the
        lift stage.
        """
        if not self._geometry_enabled or all(det.mask is None for det in detections):
            return [(None, 0.0)] * len(detections)
        started = time.perf_counter()
        height, width = image.shape[:2]
        laplacian = frame_laplacian(image)
        found: list[tuple[PointCloud2 | None, float]] = []
        for det in detections:
            if det.mask is None:
                found.append((None, 0.0))
                continue
            mask = _native_mask(det.mask, height, width)
            points, cover = None, 0.0
            if depth is not None:
                on_raster = resize_mask(mask, *depth.metres.shape)
                cover = depth_cover(on_raster, depth.metres, self._depth_min_m, self._depth_max_m)
                points = convert.to_point_cloud(header, lift_mask(on_raster, depth, self._depth_min_m, self._depth_max_m))
            found.append((points, view_quality(mask, det.box, laplacian, cover)))
        times.lift = _ms_since(started)
        return found

    def _samples(
        self,
        image: np.ndarray,
        header: Header,
        detections: Sequence[Detection],
        dropped: collections.Counter[str],
        times: StageTimes,
    ) -> list[_Sample]:
        """Return what sampling adds to each detection, which carries its published track id.

        The vector and the crop come from the same padded pixels. A bad vector
        leaves the crop in place and counts as dropped. Raises when the
        embedder raises or returns the wrong shape, or for a mask that does
        not have the image's size. Records the crop and embed stages.
        """
        if not (self._crops_enabled or self._embeddings_enabled):
            return [_Sample()] * len(detections)
        started = time.perf_counter()
        height, width = image.shape[:2]
        chosen = embeddings.select_samples(detections, self._processed_images, width, height)
        samples = self._fill_samples(image, header, detections, chosen, dropped, times)
        times.crop = max(0.0, _ms_since(started) - times.embed)
        return samples

    def _fill_samples(
        self,
        image: np.ndarray,
        header: Header,
        detections: Sequence[Detection],
        chosen: Sequence[int],
        dropped: collections.Counter[str],
        times: StageTimes,
    ) -> list[_Sample]:
        """Return what the flags add to the chosen detections, and nothing for the others.

        With crops and embeddings both off, that is nothing at all.
        """
        samples = [_Sample()] * len(detections)
        if not chosen or not (self._crops_enabled or self._embeddings_enabled):
            return samples
        height, width = image.shape[:2]
        sources = [embeddings.padded_crop(image, detections[i].box) for i in chosen]
        started = time.perf_counter()
        vectors = self._embed(sources, dropped) if self._embeddings_enabled else [None] * len(chosen)
        times.embed = _ms_since(started)
        model = self._providers.embedder.model_name if self._embeddings_enabled else ""
        for i, source, vector in zip(chosen, sources, vectors, strict=True):
            mask = detections[i].mask
            samples[i] = _Sample(
                crop=convert.to_jpeg_image(header, embeddings.encode_crop(source)) if self._crops_enabled else None,
                embedding=vector,
                embedding_model=model if vector is not None else "",
                color_shares=None
                if mask is None
                else embeddings.mask_color_shares(image, _native_mask(mask, height, width)),
                chosen=True,
            )
        return samples

    def _embed(self, sources: Sequence[np.ndarray], dropped: collections.Counter[str]) -> list[np.ndarray | None]:
        """Return one vector per image, or None for a row that is not finite or is all zero."""
        embedder = self._providers.embedder
        vectors = np.asarray(embedder.embed_images(sources))
        if vectors.shape != (len(sources), embedder.dim):
            raise ValueError(f"the embedder returned shape {vectors.shape} for {len(sources)} crops of dim {embedder.dim}")
        out: list[np.ndarray | None] = []
        for row in vectors:
            if _vector_ok(row):
                out.append(row)
            else:
                out.append(None)
                dropped["embedding"] += 1
        return out

    def _appearances(
        self, image: np.ndarray, poses: Sequence[PersonPose], dropped: collections.Counter[str]
    ) -> list[np.ndarray | None]:
        """Return each person's appearance vector, or None for a small box or a bad vector."""
        reid = self._providers.reid
        height, width = image.shape[:2]
        crops: list[np.ndarray] = []
        owners: list[int] = []
        for i, pose in enumerate(poses):
            x1, y1, x2, y2 = np.asarray(pose.box, dtype=np.float64)
            x1, x2 = (int(round(v)) for v in np.clip([x1, x2], 0, width))
            y1, y2 = (int(round(v)) for v in np.clip([y1, y2], 0, height))
            if x2 - x1 >= self._reid_min_crop_px and y2 - y1 >= self._reid_min_crop_px:
                crops.append(image[y1:y2, x1:x2])
                owners.append(i)
        out: list[np.ndarray | None] = [None] * len(poses)
        if not crops:
            return out
        vectors = np.asarray(reid.embed(crops))
        if vectors.shape != (len(crops), reid.dim):
            raise ValueError(f"the reid model returned shape {vectors.shape} for {len(crops)} crops of dim {reid.dim}")
        for row, owner in zip(vectors, owners, strict=True):
            if _vector_ok(row):
                out[owner] = row
            else:
                dropped["appearance"] += 1
        return out


def _native_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """A detection's mask as bool. Raises when it does not have the image's size."""
    array = np.asarray(mask, dtype=bool)
    if array.shape != (height, width):
        raise ValueError(f"a mask has shape {array.shape}, but the image is {height}x{width}")
    return array


def _ms_since(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _split_detector_time(times: StageTimes, detector: ObjectDetector, elapsed_ms: float) -> None:
    """Record a detector's call as the objects and mask stages.

    Only the real detectors measure their masks, and the ObjectDetector protocol
    does not ask for it, so a detector without last_mask_ms has a mask stage of 0.
    """
    times.mask = float(getattr(detector, "last_mask_ms", 0.0))
    times.objects = max(0.0, elapsed_ms - times.mask)
