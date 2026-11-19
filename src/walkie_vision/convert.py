"""Conversions from provider results (walkie_vision.providers) to walkie_vision_msgs.

The helpers check the rules that the message definitions state, so that a
node cannot publish a message that breaks them:

- Every embedding is normalised to unit length. A vector that is empty, not
  finite or zero raises ValueError instead.
- An embedding is never sent without the name of the model that made it.
- Every detection's label is one of the array's classes, because a class
  without a detection tells walkie-database that the object was not seen.
- An object array has a stream id, and its cloud, crop and camera_info stamps
  equal the array's stamp. check_object_detection_array() holds these rules
  (ADR-0013).

Unbounded float32[] fields are filled with array.array("f"): rosidl's Python
setters accept that type or a list, not a numpy array.
"""

from __future__ import annotations

import array
import math
import uuid
from collections.abc import Iterable, Sequence

import numpy as np
from sensor_msgs.msg import CameraInfo, CompressedImage, PointCloud2, PointField
from std_msgs.msg import Header
from vision_msgs.msg import BoundingBox2D
from walkie_vision_msgs.msg import (
    ObjectDetection,
    ObjectDetectionArray,
    PersonDetection,
    PersonDetectionArray,
)
from walkie_vision_msgs.srv import GetPersonCrop

from .crops.colour import COLOR_NAMES
from .crops.person import CropStatus, PersonCrop
from .providers import Detection, Face, PersonPose
from .providers.vector_tables import EMBEDDING_DIMS

MAX_CLOUD_POINTS = 2000
"""The most points that one object's cloud may hold."""
MIN_CLOUD_POINTS = 30
"""An object with fewer valid points than this gets an empty cloud."""
POINT_STEP = 12
"""The bytes of one point: x, y and z as float32."""
JPEG_FORMAT = "jpeg"
"""The value of CompressedImage.format for every object crop."""

_CLOUD_FIELDS = (("x", 0), ("y", 4), ("z", 8))
_JPEG_START = b"\xff\xd8"
# Leaves room for float32 rounding, and no more.
_SHARES_SUM_TOLERANCE = 1e-4
_UNIT_NORM_TOLERANCE = 1e-3


def box_to_bbox(box: Sequence[float] | np.ndarray) -> BoundingBox2D:
    """Convert an x1, y1, x2, y2 box in pixels to a BoundingBox2D."""
    x1, y1, x2, y2 = (float(v) for v in box)
    if not all(np.isfinite([x1, y1, x2, y2])) or x2 < x1 or y2 < y1:
        raise ValueError(f"not a box: {[x1, y1, x2, y2]}")
    bbox = BoundingBox2D()
    bbox.center.position.x = (x1 + x2) / 2
    bbox.center.position.y = (y1 + y2) / 2
    bbox.size_x = x2 - x1
    bbox.size_y = y2 - y1
    return bbox


def bbox_to_box(bbox: BoundingBox2D) -> np.ndarray:
    """Convert a BoundingBox2D back to an x1, y1, x2, y2 float array."""
    cx, cy = bbox.center.position.x, bbox.center.position.y
    half_w, half_h = bbox.size_x / 2, bbox.size_y / 2
    return np.array([cx - half_w, cy - half_h, cx + half_w, cy + half_h])


def unit_vector(vector: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return a 1-D vector scaled to unit length, as float32.

    Raises ValueError for a vector that is not 1-D, is empty, has a value
    that is not finite, or is zero.
    """
    v = np.asarray(vector, dtype=np.float64)
    if v.ndim != 1 or v.size == 0:
        raise ValueError(f"an embedding must be a non-empty 1-D vector, not shape {v.shape}")
    if not np.all(np.isfinite(v)):
        raise ValueError("an embedding has a value that is not finite")
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("an embedding is zero and has no direction")
    return (v / norm).astype(np.float32)


def float32_array(values: np.ndarray) -> array.array:
    out = array.array("f")
    out.frombytes(np.ascontiguousarray(values, dtype=np.float32).tobytes())
    return out


def _embedding(vector: np.ndarray | None, model: str, field: str) -> tuple[str, array.array]:
    """Return the (model, vector) pair for an embedding field, checked."""
    if vector is None:
        if model:
            raise ValueError(f"{field}: a model name was given without an embedding")
        return "", array.array("f")
    if not model:
        raise ValueError(f"{field}: an embedding was given without its model name")
    return model, float32_array(unit_vector(vector))


def _track_id(track_id: int) -> int:
    if track_id < 0:
        raise ValueError(f"track_id must be 0 (not tracked) or positive, not {track_id}")
    return int(track_id)


def new_stream_id() -> str:
    """Return the id of one publisher process. Call it once, when the process starts."""
    return str(uuid.uuid4())


def stamp_ns(header: Header) -> int:
    """Return a header's stamp in nanoseconds."""
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def empty_cloud(header: Header) -> PointCloud2:
    """Return the cloud of an object with no points: the contract's layout and header, and no data."""
    return _cloud(header, array.array("B"), 0)


def evenly_spaced_indices(count: int, limit: int) -> np.ndarray:
    """Return at most limit indices into count items, evenly spaced and in order. The same input gives the same indices."""
    if count <= limit:
        return np.arange(count)
    return np.arange(limit) * count // limit


def to_point_cloud(header: Header, points: Sequence[Sequence[float]] | np.ndarray) -> PointCloud2:
    """Build an object's cloud from an N x 3 array of x, y, z in metres, in the optical frame.

    Fewer than MIN_CLOUD_POINTS points give an empty cloud. More than
    MAX_CLOUD_POINTS are cut down to that many, at evenly spaced indices in
    their input order, so the same input gives the same cloud. Raises
    ValueError for another shape, a value that is not finite, or a point
    that is not in front of the camera.
    """
    xyz = np.asarray(points, dtype=np.float64)
    if xyz.size == 0:
        return empty_cloud(header)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), not {xyz.shape}")
    if not np.all(np.isfinite(xyz)):
        raise ValueError("points has a value that is not finite")
    if np.any(xyz[:, 2] <= 0):
        raise ValueError("points has a point that is not in front of the camera")
    count = len(xyz)
    if count < MIN_CLOUD_POINTS:
        return empty_cloud(header)
    xyz = xyz[evenly_spaced_indices(count, MAX_CLOUD_POINTS)]
    data = array.array("B")
    data.frombytes(xyz.astype("<f4").tobytes())
    return _cloud(header, data, len(xyz))


def _cloud(header: Header, data: array.array, width: int) -> PointCloud2:
    return PointCloud2(
        header=header,
        height=1,
        width=width,
        fields=[PointField(name=n, offset=o, datatype=PointField.FLOAT32, count=1) for n, o in _CLOUD_FIELDS],
        is_bigendian=False,
        point_step=POINT_STEP,
        row_step=POINT_STEP * width,
        data=data,
        is_dense=True,
    )


def _unit_interval(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:  # also false for NaN
        raise ValueError(f"{name} must be finite and in [0, 1], not {value}")
    return value


def _color_shares(shares: Sequence[float] | np.ndarray | None) -> array.array:
    """Return shares for a message, checked. None or an empty sequence means not sampled."""
    if shares is None:
        return array.array("f")
    values = np.asarray(shares, dtype=np.float64)
    if values.size == 0:
        return array.array("f")
    problem = _color_shares_problem(values)
    if problem is not None:
        raise ValueError(problem)
    return float32_array(values)


def _color_shares_problem(values: np.ndarray) -> str | None:
    if values.shape != (len(COLOR_NAMES),):
        return f"color_shares must be empty or hold {len(COLOR_NAMES)} shares, not shape {values.shape}"
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        return "color_shares must be finite and not negative"
    if abs(float(values.sum()) - 1.0) > _SHARES_SUM_TOLERANCE:
        return f"color_shares must sum to 1, not {float(values.sum())}"
    return None


def to_jpeg_image(header: Header, jpeg: bytes) -> CompressedImage:
    """A CompressedImage of JPEG bytes. The array assignment skips a per-byte check of the message's setter."""
    data = array.array("B")
    data.frombytes(jpeg)
    return CompressedImage(header=header, format=JPEG_FORMAT, data=data)


def to_object_detection(
    detection: Detection,
    embedding: np.ndarray | None = None,
    embedding_model: str = "",
    points: PointCloud2 | None = None,
    crop: CompressedImage | None = None,
    view_quality: float = 0.0,
    color_shares: Sequence[float] | np.ndarray | None = None,
) -> ObjectDetection:
    """Build an ObjectDetection from a detector result and its optional parts.

    Raises ValueError for a score or a view quality outside [0, 1], colour
    shares that break the message's rule, or a real model's vector of the
    wrong length. to_object_detection_array() checks the rest.
    """
    msg = ObjectDetection()
    msg.label = detection.label
    msg.score = _unit_interval(detection.score, "score")
    msg.track_id = _track_id(detection.track_id)
    msg.bbox = box_to_bbox(detection.box)
    msg.embedding_model, msg.embedding = _embedding(embedding, embedding_model, "embedding")
    expected = EMBEDDING_DIMS.get(msg.embedding_model)
    if expected is not None and len(msg.embedding) != expected:
        raise ValueError(f"embedding: {msg.embedding_model} has {expected} values, not {len(msg.embedding)}")
    if points is not None:
        msg.points = points
    if crop is not None:
        msg.crop = crop
    msg.view_quality = _unit_interval(view_quality, "view_quality")
    msg.color_shares = _color_shares(color_shares)
    return msg


def to_object_detection_array(
    header: Header,
    camera_info: CameraInfo,
    classes: Sequence[str],
    detections: Iterable[ObjectDetection],
    *,
    stream_id: str,
) -> ObjectDetectionArray:
    """Build the array for one image. Publish it also when detections is empty.

    A detection whose cloud was never set gets the empty cloud with this
    header, and an empty crop gets this header, so that every stamp in the
    array agrees. Raises ValueError for an array that
    check_object_detection_array() refuses.
    """
    msg = ObjectDetectionArray()
    msg.header = header
    msg.camera_info = camera_info
    msg.classes = list(classes)
    msg.detections = list(detections)
    msg.stream_id = stream_id
    for detection in msg.detections:
        if detection.points == PointCloud2():
            detection.points = empty_cloud(header)
        if len(detection.crop.data) == 0:
            detection.crop.header = header
    check_object_detection_array(msg)
    return msg


def check_object_detection_array(msg: ObjectDetectionArray) -> None:
    """Raise ValueError unless an array follows the rules in the message files.

    It reads a crop's first bytes but does not decode it, so it cannot check
    the crop's size.
    """
    problem = _array_problem(msg) or next(
        (f"detection {i}: {p}" for i, d in enumerate(msg.detections) if (p := _detection_problem(d, msg))), None
    )
    if problem is not None:
        raise ValueError(problem)


def _array_problem(msg: ObjectDetectionArray) -> str | None:
    if not _is_stream_id(msg.stream_id):
        return f"stream_id must be a UUID, not {msg.stream_id!r}"
    if msg.camera_info.header != msg.header:
        return "camera_info.header differs from the array's header"
    classes = list(msg.classes)
    if not classes or not all(classes) or len(set(classes)) != len(classes):
        return f"classes must be unique non-empty names, and at least one, not {classes}"
    return None


def _is_stream_id(stream_id: str) -> bool:
    try:
        return str(uuid.UUID(stream_id)) == stream_id
    except ValueError:
        return False


def _detection_problem(det: ObjectDetection, array_msg: ObjectDetectionArray) -> str | None:
    if det.label not in array_msg.classes:
        return f"the label {det.label!r} is not in classes: {list(array_msg.classes)}"
    if not 0.0 <= det.score <= 1.0:
        return f"score must be finite and in [0, 1], not {det.score}"
    if not 0.0 <= det.view_quality <= 1.0:
        return f"view_quality must be finite and in [0, 1], not {det.view_quality}"
    return (
        _bbox_problem(det.bbox)
        or _embedding_problem(det)
        or _cloud_problem(det.points, array_msg.header)
        or _crop_problem(det.crop, array_msg.header)
        or (_color_shares_problem(np.asarray(det.color_shares, dtype=np.float64)) if len(det.color_shares) else None)
    )


def _bbox_problem(bbox: BoundingBox2D) -> str | None:
    values = (bbox.center.position.x, bbox.center.position.y, bbox.size_x, bbox.size_y)
    if not all(math.isfinite(v) for v in values) or bbox.size_x < 0 or bbox.size_y < 0:
        return f"bbox must be finite with a size that is not negative, not {values}"
    if bbox.center.theta != 0.0:
        return f"bbox.center.theta must be 0, not {bbox.center.theta}"
    return None


def _embedding_problem(det: ObjectDetection) -> str | None:
    vector = np.asarray(det.embedding, dtype=np.float64)
    if bool(det.embedding_model) != (vector.size > 0):
        return "embedding and embedding_model must be both set or both empty"
    if vector.size == 0:
        return None
    expected = EMBEDDING_DIMS.get(det.embedding_model)
    if expected is not None and vector.size != expected:
        return f"embedding: {det.embedding_model} has {expected} values, not {vector.size}"
    if not np.all(np.isfinite(vector)):
        return "embedding has a value that is not finite"
    if abs(float(np.linalg.norm(vector)) - 1.0) > _UNIT_NORM_TOLERANCE:
        return f"embedding must have unit length, not {float(np.linalg.norm(vector))}"
    return None


def _cloud_problem(cloud: PointCloud2, header: Header) -> str | None:
    if cloud.header != header:
        return "points.header differs from the array's header"
    fields = [(f.name, f.offset, f.datatype, f.count) for f in cloud.fields]
    if fields != [(n, o, PointField.FLOAT32, 1) for n, o in _CLOUD_FIELDS]:
        return f"points must have float32 fields x, y, z at offsets 0, 4, 8, not {fields}"
    if cloud.is_bigendian or not cloud.is_dense:
        return "points must be little-endian and dense"
    if cloud.height != 1 or cloud.point_step != POINT_STEP or cloud.row_step != POINT_STEP * cloud.width:
        return (
            f"points must have height 1, point_step {POINT_STEP} and row_step {POINT_STEP} * width, "
            f"not {cloud.height}, {cloud.point_step} and {cloud.row_step}"
        )
    if len(cloud.data) != cloud.row_step:
        return f"points has {len(cloud.data)} bytes for {cloud.width} points"
    if cloud.width == 0:
        return None
    if not MIN_CLOUD_POINTS <= cloud.width <= MAX_CLOUD_POINTS:
        return f"points must be empty or hold {MIN_CLOUD_POINTS} to {MAX_CLOUD_POINTS} points, not {cloud.width}"
    xyz = np.frombuffer(bytes(cloud.data), dtype="<f4").reshape(-1, 3)
    if not np.all(np.isfinite(xyz)) or np.any(xyz[:, 2] <= 0):
        return "points has a point that is not finite or not in front of the camera"
    return None


def _crop_problem(crop: CompressedImage, header: Header) -> str | None:
    if len(crop.data) == 0:
        return None
    if crop.header != header:
        return "crop.header differs from the array's header"
    if crop.format != JPEG_FORMAT or bytes(crop.data[:2]) != _JPEG_START:
        return f"crop must be a JPEG with format {JPEG_FORMAT!r}, not format {crop.format!r}"
    return None


def to_person_detection(
    pose: PersonPose,
    face: Face | None = None,
    face_model: str = "",
    appearance: np.ndarray | None = None,
    appearance_model: str = "",
) -> PersonDetection:
    """Build a PersonDetection from a pose result, and the person's face and appearance embedding.

    Without a face, the face fields keep their defaults: a zero box, a
    face_det_score of 0 and an empty face_model and face_embedding.
    """
    keypoints = np.asarray(pose.keypoints, dtype=np.float32)
    scores = np.asarray(pose.keypoint_scores, dtype=np.float32)
    if keypoints.shape != (17, 2) or scores.shape != (17,):
        raise ValueError(
            f"expected 17 COCO keypoints, shapes (17, 2) and (17,), not {keypoints.shape} and {scores.shape}"
        )
    msg = PersonDetection()
    msg.score = float(pose.score)
    msg.track_id = _track_id(pose.track_id)
    msg.bbox = box_to_bbox(pose.box)
    msg.keypoints = keypoints.reshape(34)
    msg.keypoint_scores = scores
    if face is not None:
        msg.face_bbox = box_to_bbox(face.bbox)
        msg.face_det_score = float(face.det_score)
        msg.face_model, msg.face_embedding = _embedding(face.embedding, face_model, "face_embedding")
    elif face_model:
        raise ValueError("face_embedding: a model name was given without a face")
    msg.appearance_model, msg.appearance_embedding = _embedding(
        appearance, appearance_model, "appearance_embedding"
    )
    return msg


def to_person_detection_array(
    header: Header, camera_info: CameraInfo, people: Iterable[PersonDetection]
) -> PersonDetectionArray:
    """Build the array for one image. Publish it also when people is empty."""
    msg = PersonDetectionArray()
    msg.header = header
    msg.camera_info = camera_info
    msg.people = list(people)
    return msg


def pose_from_person_detection(msg: PersonDetection) -> PersonPose:
    """Return a published person as a PersonPose, with its published track id."""
    return PersonPose(
        box=bbox_to_box(msg.bbox),
        score=float(msg.score),
        keypoints=np.asarray(msg.keypoints, dtype=np.float64).reshape(17, 2),
        keypoint_scores=np.asarray(msg.keypoint_scores, dtype=np.float64),
        track_id=int(msg.track_id),
    )


def fill_person_crop_response(
    response: GetPersonCrop.Response, people: PersonDetectionArray, found: PersonCrop
) -> GetPersonCrop.Response:
    """Fill a GetPersonCrop response from walkie_vision.crops.person.crop_person()'s result.

    people is the published array of the image that found came from. The
    header is set for every status that found can have; the person and the
    overlaps once the track is found; the crop and the shares only for OK.
    """
    response.status = int(found.status)
    response.message = found.message
    response.header = people.header
    if found.index < 0:
        return response
    response.person = people.people[found.index]
    response.max_overlap = float(found.max_overlap)
    response.torso_cover = float(found.torso_cover)
    if found.status != CropStatus.OK:
        return response
    response.crop = CompressedImage(header=people.header, format="jpeg", data=found.jpeg)
    response.crop_keypoints = float32_array(np.asarray(found.keypoints).reshape(-1))
    response.masked_boxes = float32_array(np.asarray(found.masked).reshape(-1))
    response.color_names = list(COLOR_NAMES)
    if found.top_shares is not None:
        response.top_shares = float32_array(found.top_shares)
        response.top_left = float(found.top_left)
    if found.bottom_shares is not None:
        response.bottom_shares = float32_array(found.bottom_shares)
        response.bottom_left = float(found.bottom_left)
    return response
