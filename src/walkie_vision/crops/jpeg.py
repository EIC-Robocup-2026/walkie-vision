"""Scaling a crop down and encoding it as a JPEG, for every crop the node sends."""

from __future__ import annotations

import cv2
import numpy as np

JPEG_QUALITY = 90


def scale_down(pixels: np.ndarray, max_side: int) -> tuple[np.ndarray, float, float]:
    """Scale an image so that its longer side is at most max_side. Returns it with the x and y scales."""
    height, width = pixels.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return pixels, 1.0, 1.0
    scale = max_side / longest
    new_width, new_height = max(1, round(width * scale)), max(1, round(height * scale))
    out = cv2.resize(pixels, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return out, new_width / width, new_height / height


def encode_jpeg(pixels: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    """Encode a uint8 BGR image as a JPEG."""
    ok, data = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError(f"OpenCV could not encode a {pixels.shape} image as a JPEG")
    return data.tobytes()
