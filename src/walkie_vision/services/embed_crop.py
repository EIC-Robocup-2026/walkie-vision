"""EmbedCrop: the request checks, the crop's view quality and embedding, and the responses (ADR-0015).

The callback checks the request with embed_crop_problem() and leaves a job in
the slot of walkie_vision.services.jobs. The worker answers from the current
image only when crop_image_problem() finds that it is the image the request
names, and embed_crop_answer() crops, scores and embeds the box.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from builtin_interfaces.msg import Time
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from walkie_vision_msgs.srv import EmbedCrop

from .. import boxes, convert
from ..boxes import resize_mask
from ..camera.depth import RegisteredDepth
from ..convert import float32_array, stamp_ns, to_jpeg_image, unit_vector
from ..crops import objects
from ..geometry import depth_cover, frame_laplacian, view_quality
from ..providers import Embedder
from ..providers.image_embed.base import check_model_dim
from .jobs import MAX_TIMEOUT_S, bounded_message

MAX_STAMP_GAP_NS = 5_000_000
"""The largest difference between a request's stamp and the current image's."""
MAX_IMAGE_AGE_NS = 300_000_000
"""The oldest current image a crop may come from, in ROS time."""
MAX_FUTURE_NS = 5_000_000
"""How far ahead of ROS time a request's stamp may be."""


def stamp_problem(stamp: Time, name: str = "header.stamp") -> str | None:
    """Return why a time in a request is refused, or None. Call it before stamp_ns() reads the time.

    The generated type takes any uint32 nanosec, so stamp_ns() would read 0 s
    and 1e9 ns as 1 s. walkie-database's ingress refuses the same times. A
    zero stamp is refused too.

    Of the requests the node serves, only EmbedCrop's has a time, in
    header.stamp. DescribeCrop's crop has a stamp too, but that service checks
    nothing while it always answers DISABLED. A caption build must check the
    stamp here.
    """
    sec, nanosec = int(stamp.sec), int(stamp.nanosec)
    if sec < 0 or nanosec >= 1_000_000_000:
        return f"{name} must be a valid time after zero, not {sec} s and {nanosec} ns"
    if sec == 0 and nanosec == 0:
        return f"{name} is zero"
    return None


def embed_crop_problem(request: EmbedCrop.Request) -> str | None:
    """Return why an EmbedCrop request is refused before it sees an image, or None."""
    if not request.header.frame_id:
        return "header.frame_id is empty"
    problem = stamp_problem(request.header.stamp)
    if problem is not None:
        return problem
    width, height = int(request.image_width), int(request.image_height)
    if width == 0 or height == 0:
        return f"the image size must be positive, not {width}x{height}"
    timeout = float(request.timeout_s)
    if not (math.isfinite(timeout) and 0.0 < timeout <= MAX_TIMEOUT_S):
        return f"timeout_s must be more than 0 and at most {MAX_TIMEOUT_S:g}, not {timeout}"
    bbox = request.bbox
    values = (bbox.center.position.x, bbox.center.position.y, bbox.center.theta, bbox.size_x, bbox.size_y)
    if not all(math.isfinite(v) for v in values):
        return "bbox has a value that is not finite"
    if bbox.center.theta != 0.0:
        return f"bbox.center.theta must be 0, not {bbox.center.theta}"
    if bbox.size_x <= 0.0 or bbox.size_y <= 0.0:
        return f"bbox must have a positive area, not {bbox.size_x} by {bbox.size_y}"
    # Finite fields can still add up past the largest float. A corner that is
    # not finite cannot become a whole pixel, so it is refused here.
    box = convert.bbox_to_box(bbox)
    if not np.all(np.isfinite(box)):
        return "bbox has a corner that is not finite"
    if not np.all(np.isfinite(boxes.pad_box(box, objects.CROP_PADDING))):
        return "bbox's padded crop has a corner that is not finite"
    if boxes.intersect_rects(boxes.outer_rect(box), (0, 0, width, height)) is None:
        return f"bbox has no pixels in a {width}x{height} image"
    return None


def crop_image_problem(request: EmbedCrop.Request, image: Image, now_ns: int) -> str | None:
    """Return why the current image cannot answer a request, or None.

    The box belongs to the image the request names. On any other view it would
    cover something else, so this never widens to a near match.
    """
    if request.header.frame_id != image.header.frame_id:
        return f"the request is in frame {request.header.frame_id!r}, the current image in {image.header.frame_id!r}"
    if (request.image_width, request.image_height) != (image.width, image.height):
        return (
            f"the request is for a {request.image_width}x{request.image_height} image, "
            f"the current image is {image.width}x{image.height}"
        )
    request_ns, image_ns = stamp_ns(request.header), stamp_ns(image.header)
    if abs(request_ns - image_ns) > MAX_STAMP_GAP_NS:
        return f"the request's stamp is {(request_ns - image_ns) / 1e6:.1f} ms from the current image's"
    if now_ns - image_ns > MAX_IMAGE_AGE_NS:
        return f"the current image is {(now_ns - image_ns) / 1e6:.0f} ms old"
    if request_ns - now_ns > MAX_FUTURE_NS:
        return f"the request's stamp is {(request_ns - now_ns) / 1e6:.1f} ms ahead of the clock"
    return None


def rectangle_quality(
    image: np.ndarray, box: np.ndarray, depth: RegisteredDepth | None, depth_min_m: float, depth_max_m: float
) -> float:
    """Score a requested box as a view, with the rectangle as an approximate mask.

    The same four terms as a detection. The depth term is 0 without depth.
    """
    height, width = image.shape[:2]
    x1, y1, x2, y2 = boxes.intersect_rects(boxes.outer_rect(box), (0, 0, width, height)) or (0, 0, 0, 0)
    mask = np.zeros((height, width), dtype=bool)
    mask[y1:y2, x1:x2] = True
    cover = 0.0
    if depth is not None:
        cover = depth_cover(resize_mask(mask, *depth.metres.shape), depth.metres, depth_min_m, depth_max_m)
    return view_quality(mask, box, frame_laplacian(image), cover)


def embed_crop_answer(
    embedder: Embedder,
    image: np.ndarray,
    header: Any,
    box: np.ndarray,
    depth: RegisteredDepth | None,
    depth_min_m: float,
    depth_max_m: float,
) -> EmbedCrop.Response:
    """Crop and embed one box of one image. Raises when the box, the model or its vector fails."""
    pixels = objects.padded_crop(image, box)
    vectors = np.asarray(embedder.embed_images([pixels]))
    if vectors.shape != (1, embedder.dim):
        raise ValueError(f"the embedder returned shape {vectors.shape} for one crop of dim {embedder.dim}")
    return fill_embed_crop_response(
        EmbedCrop.Response(),
        header,
        objects.encode_crop(pixels),
        embedder,
        vectors[0],
        rectangle_quality(image, box, depth, depth_min_m, depth_max_m),
    )


def embed_crop_failure(code: str, message: str) -> EmbedCrop.Response:
    """An EmbedCrop failure, with every other field at its empty value."""
    return EmbedCrop.Response(success=False, error_code=code, message=bounded_message(message))


def fill_embed_crop_response(
    response: EmbedCrop.Response, header: Header, jpeg: bytes, embedder: Embedder, vector: np.ndarray, quality: float
) -> EmbedCrop.Response:
    """Fill a successful EmbedCrop response from the source image's header.

    Raises ValueError for a vector that is not a usable one of the model's
    length, or a quality outside [0, 1].
    """
    check_model_dim(embedder)
    unit = unit_vector(vector)
    if unit.size != embedder.dim:
        raise ValueError(f"expected {embedder.dim} dimensions, got {unit.size}")
    quality = float(quality)
    if not (math.isfinite(quality) and 0.0 <= quality <= 1.0):
        raise ValueError(f"view_quality must be finite in [0, 1], not {quality}")
    response.success = True
    response.error_code = ""
    response.message = ""
    response.header = header
    response.crop = to_jpeg_image(header, jpeg)
    response.embedding_model = embedder.model_name
    response.embedding = float32_array(unit)
    response.view_quality = quality
    return response
