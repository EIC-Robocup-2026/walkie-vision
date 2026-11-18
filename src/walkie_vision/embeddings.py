"""Sampled object crops, their colour shares, and the interim CLIP provider.

The vision node samples a few detections per image (select_samples). A
sampled detection gets a padded JPEG crop, a CLIP vector of the same padded
pixels and the colour shares of its mask. Every other detection leaves those
fields empty, and none of this work runs for it.

ClipEmbedder is the interim text and image embedder. ADR-0008 stays Proposed
until the bake-off measures the alternatives. The model name stays
clip_vit_b16, because the weights, QuickGELU and the text template are the
ones in tools/bakeoff/embedders.py. A raw-token vector would not compare with
a templated one, so embed_texts() always applies the template.

The node only loads the weights from the Hugging Face cache, at the commit in
HUB_MODELS. `pixi run fetch-models` is the one thing that downloads them.

torch, open_clip and huggingface_hub are imported on use, so this module
loads without them and the fake publisher can share its constants.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import cv2
import numpy as np

from . import boxes
from .crops import jpeg
from .providers import CLIP_VIT_B16, Detection
from .weights import CLIP_WEIGHTS, HUB_MODELS

SAMPLE_PERIOD = 10
"""A track is eligible once in this many processed images."""
SAMPLE_MAX_OBJECTS = 3
"""The most detections sampled in one image."""
SAMPLE_MIN_SIDE_PX = 32
"""The shortest side a box may have to be sampled, before padding."""
CROP_PADDING = 0.15
"""How much a crop grows on each side, as a share of the box's width and height."""
CROP_MAX_SIDE = 512
"""The longest side of an encoded crop, in pixels. The embedding uses the crop before this reduction."""
MIN_COLOR_PIXELS = 50
"""Fewer mask pixels than this give no colour shares."""

CLIP_ARCHITECTURE = "ViT-B-16-quickgelu"
"""The OpenAI weights were trained with QuickGELU (see checks/c07_clip.py)."""
CLIP_DIM = 512
TEXT_TEMPLATE = "a photo of {}."

# Colour recipe v1: RGB, in the order of crops.colour.COLOR_NAMES. This is deterministic
# plumbing, not colour constancy. Its accuracy has not been measured.
PALETTE_RGB = (
    (0, 0, 0),
    (255, 255, 255),
    (128, 128, 128),
    (220, 35, 35),
    (245, 130, 25),
    (245, 220, 35),
    (45, 155, 65),
    (40, 90, 205),
    (135, 65, 175),
    (235, 145, 180),
    (115, 75, 45),
    (210, 190, 145),
)
_PALETTE_LAB = cv2.cvtColor(np.array(PALETTE_RGB, dtype=np.float32).reshape(-1, 1, 3) / 255.0, cv2.COLOR_RGB2Lab).reshape(-1, 3)


def _short_side(box: np.ndarray, width: int, height: int) -> float:
    """The shorter side of a box inside a width x height image, in pixels."""
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float64)
    return float(min(min(x2, width) - max(x1, 0.0), min(y2, height) - max(y1, 0.0)))


def select_samples(detections: Sequence[Detection], counter: int, width: int, height: int) -> list[int]:
    """Return the indices of the detections to sample in one image, best first.

    detections carry their published track ids. counter is the number of
    images processed before this one. A detection is eligible when
    (track_id + counter) % SAMPLE_PERIOD is 0 and its box's shorter side, inside
    the image, is at least SAMPLE_MIN_SIDE_PX. The highest scores win, then the
    lower track id, then the earlier detection.
    """
    eligible = [
        i
        for i, det in enumerate(detections)
        if (int(det.track_id) + counter) % SAMPLE_PERIOD == 0 and _short_side(det.box, width, height) >= SAMPLE_MIN_SIDE_PX
    ]
    eligible.sort(key=lambda i: (-float(detections[i].score), int(detections[i].track_id), i))
    return eligible[:SAMPLE_MAX_OBJECTS]


def usable_samples(detections: Sequence[Detection], width: int, height: int) -> list[int]:
    """Return the indices of every detection whose box is large enough to sample, in order.

    For DetectObjects, which has no period and no limit of its own here: the
    caller asked for this one image.
    """
    return [i for i, det in enumerate(detections) if _short_side(det.box, width, height) >= SAMPLE_MIN_SIDE_PX]


def padded_crop(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    """The pixels of a box grown by CROP_PADDING on each side and clipped to the image. A view, not a copy.

    Raises ValueError when no pixel is left.
    """
    height, width = image.shape[:2]
    x1, y1, x2, y2 = boxes.pad_rect(box, CROP_PADDING, width, height)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"the box {np.asarray(box).tolist()} has no pixels in a {width}x{height} image")
    return image[y1:y2, x1:x2]


def encode_crop(pixels: np.ndarray) -> bytes:
    """Encode padded pixels as a JPEG, reduced to CROP_MAX_SIDE on the longest side, never enlarged."""
    reduced, _, _ = jpeg.scale_down(pixels, CROP_MAX_SIDE)
    return jpeg.encode_jpeg(reduced, jpeg.JPEG_QUALITY)


def nearest_palette_indices(lab: np.ndarray, centres: np.ndarray = _PALETTE_LAB) -> np.ndarray:
    """The index of the nearest Lab centre of each Lab pixel, by Euclidean distance. A tie takes the earlier centre.

    Compares one centre at a time, so a large mask does not need a pixels x
    centres array.
    """
    lab = np.asarray(lab, dtype=np.float32).reshape(-1, 3)
    best = np.full(lab.shape[0], np.inf, dtype=np.float32)
    indices = np.zeros(lab.shape[0], dtype=np.intp)
    for i, centre in enumerate(centres):
        distance = np.sum((lab - centre) ** 2, axis=1)
        nearer = distance < best
        best[nearer] = distance[nearer]
        indices[nearer] = i
    return indices


def mask_color_shares(image: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """The shares of the palette colours among a mask's pixels of a BGR image; None under MIN_COLOR_PIXELS.

    Uses the original mask and pixels: no padding, resizing or JPEG. OpenCV
    converts float32 sRGB to CIELAB. The shares follow crops.colour.COLOR_NAMES and sum
    to 1.
    """
    pixels = image[np.asarray(mask, dtype=bool)]
    count = pixels.shape[0]
    if count < MIN_COLOR_PIXELS:
        return None
    lab = cv2.cvtColor(pixels.reshape(-1, 1, 3).astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    counts = np.bincount(nearest_palette_indices(lab), minlength=len(PALETTE_RGB))
    return counts.astype(np.float64) / count


def clip_checkpoint(download: Callable[..., Any] | None = None) -> str:
    """Return the pinned CLIP weights' path in the Hugging Face cache, without using the network.

    Raises FileNotFoundError saying what to run when they are not cached.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    hub = HUB_MODELS["clip"]
    try:
        path = (download or hf_hub_download)(hub.repo, CLIP_WEIGHTS, revision=hub.revision, local_files_only=True)
    except LocalEntryNotFoundError as e:
        raise FileNotFoundError(
            f"{CLIP_WEIGHTS} of {hub.repo} at {hub.revision} is not in the Hugging Face cache; "
            "run `pixi run fetch-models`"
        ) from e
    return str(path)


def _build_open_clip(checkpoint: str, device: str) -> tuple[Any, Any, Any]:
    """Load the model from a local file. Return it with its image preprocessing and its tokenizer."""
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_ARCHITECTURE, pretrained=checkpoint, device=device)
    model.eval()
    return model, preprocess, open_clip.get_tokenizer(CLIP_ARCHITECTURE)


class ClipEmbedder:
    """OpenAI CLIP ViT-B/16 through open_clip, as an Embedder.

    checkpoint is the path from clip_checkpoint(). build loads the model; the
    tests pass a stub, so no test loads a model. Each call runs under
    inference_mode, with float16 autocast on CUDA, and returns unit rows as
    float32. A row that a broken model leaves at zero or NaN stays that way, so
    the caller can refuse it.
    """

    model_name = CLIP_VIT_B16
    dim = CLIP_DIM

    def __init__(
        self,
        checkpoint: str,
        *,
        device: str = "cuda:0",
        build: Callable[[str, str], tuple[Any, Any, Any]] = _build_open_clip,
    ) -> None:
        self._device = device
        self._model, self._preprocess, self._tokenizer = build(checkpoint, device)

    def embed_images(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """Embed BGR images. CLIP's own preprocessing resizes each one to 224 px."""
        if len(images) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        import torch
        from PIL import Image

        batch = torch.stack([self._preprocess(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))) for im in images])
        return self._run(self._model.encode_image, batch)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed each text inside TEXT_TEMPLATE."""
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._run(self._model.encode_text, self._tokenizer([TEXT_TEMPLATE.format(t) for t in texts]))

    def _run(self, encode: Callable[[Any], Any], batch: Any) -> np.ndarray:
        import torch

        autocast = torch.autocast("cuda", dtype=torch.float16, enabled=self._device.startswith("cuda"))
        with torch.inference_mode(), autocast:
            vectors = encode(batch.to(self._device)).float()
            return torch.nn.functional.normalize(vectors, dim=-1).cpu().numpy()
