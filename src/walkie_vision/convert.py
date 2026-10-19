"""Conversions from provider results (walkie_vision.providers) to walkie_vision_msgs.

The helpers check the rules that the message definitions state, so that a
node cannot publish a message that breaks them:

- Every embedding is normalised to unit length. A vector that is empty, not
  finite or zero raises ValueError instead.
- An embedding is never sent without the name of the model that made it.
- Every detection's label is one of the array's classes, because a class
  without a detection tells walkie-database that the object was not seen.

Unbounded float32[] fields are filled with array.array("f"): rosidl's Python
setters accept that type or a list, not a numpy array.
"""

from __future__ import annotations

import array
from collections.abc import Iterable, Sequence

import numpy as np
from sensor_msgs.msg import CameraInfo, CompressedImage, PointCloud2
from std_msgs.msg import Header
from vision_msgs.msg import BoundingBox2D
from walkie_vision_msgs.msg import (
    ObjectDetection,
    ObjectDetectionArray,
    PersonDetection,
    PersonDetectionArray,
)
from walkie_vision_msgs.srv import EmbedText

from .providers import Detection, Embedder, Face, PersonPose


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


def _float32_array(values: np.ndarray) -> array.array:
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
    return model, _float32_array(unit_vector(vector))


def _track_id(track_id: int) -> int:
    if track_id < 0:
        raise ValueError(f"track_id must be 0 (not tracked) or positive, not {track_id}")
    return int(track_id)


def to_object_detection(
    detection: Detection,
    embedding: np.ndarray | None = None,
    embedding_model: str = "",
    points: PointCloud2 | None = None,
    crop: CompressedImage | None = None,
) -> ObjectDetection:
    """Build an ObjectDetection from a detector result and its optional parts."""
    msg = ObjectDetection()
    msg.label = detection.label
    msg.score = float(detection.score)
    msg.track_id = _track_id(detection.track_id)
    msg.bbox = box_to_bbox(detection.box)
    msg.embedding_model, msg.embedding = _embedding(embedding, embedding_model, "embedding")
    if points is not None:
        msg.points = points
    if crop is not None:
        msg.crop = crop
    return msg


def to_object_detection_array(
    header: Header,
    camera_info: CameraInfo,
    classes: Sequence[str],
    detections: Iterable[ObjectDetection],
) -> ObjectDetectionArray:
    """Build the array for one image. Publish it also when detections is empty."""
    msg = ObjectDetectionArray()
    msg.header = header
    msg.camera_info = camera_info
    msg.classes = list(classes)
    msg.detections = list(detections)
    unknown = sorted({d.label for d in msg.detections} - set(msg.classes))
    if unknown:
        raise ValueError(f"detections with labels that are not in classes: {unknown}")
    return msg


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


def fill_embed_text_response(
    response: EmbedText.Response, embedder: Embedder, texts: Sequence[str]
) -> EmbedText.Response:
    """Embed texts with embedder and fill an EmbedText response.

    A request without texts, or an embedder that raises or returns the wrong
    shape, gives success=False and the reason in message, so that the
    service answers instead of failing.
    """
    texts = list(texts)
    if not texts:
        response.success = False
        response.message = "texts is empty"
        return response
    try:
        vectors = np.asarray(embedder.embed_texts(texts))
        if vectors.shape != (len(texts), embedder.dim):
            raise ValueError(f"expected shape {(len(texts), embedder.dim)}, got {vectors.shape}")
        rows = np.stack([unit_vector(v) for v in vectors])
    except Exception as e:  # the reason goes back to the caller
        response.success = False
        response.message = f"{type(e).__name__}: {e}"
        return response
    response.success = True
    response.message = ""
    response.embedding_model = embedder.model_name
    response.dim = embedder.dim
    response.vectors = _float32_array(rows.reshape(-1))
    return response


def vectors_from_response(response: EmbedText.Response) -> np.ndarray:
    """Return an EmbedText response's vectors as a float32 array of shape (n, dim)."""
    if not response.success:
        raise ValueError(f"EmbedText failed: {response.message}")
    vectors = np.asarray(response.vectors, dtype=np.float32)
    if response.dim == 0 or vectors.size % response.dim:
        raise ValueError(f"{vectors.size} values do not split into vectors of {response.dim}")
    return vectors.reshape(-1, response.dim)
