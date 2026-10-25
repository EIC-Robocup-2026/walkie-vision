"""OSNet x1.0 as a walkie_vision.providers.ReIdentifier.

OSNetReIdentifier resizes each BGR person crop to the checkpoint's input,
normalises it as in training, and embeds the crops in batches. The rows are
not normalised: walkie_vision.convert does that.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
import torch

from ..providers import OSNET_X1_0_MSMT17
from . import INPUT_SIZE, MEAN, STD, load_osnet


def osnet_input(crops: Sequence[np.ndarray]) -> np.ndarray:
    """Return the network input for BGR person crops: shape (n, 3, 256, 128), float32.

    Each crop is converted to RGB, resized to INPUT_SIZE, scaled to [0, 1]
    and normalised with MEAN and STD. An empty crop, or one that is not an
    H x W x 3 uint8 array, raises ValueError.
    """
    h, w = INPUT_SIZE
    mean = np.asarray(MEAN, dtype=np.float32)
    std = np.asarray(STD, dtype=np.float32)
    batch = np.empty((len(crops), 3, h, w), dtype=np.float32)
    for i, crop in enumerate(crops):
        crop = np.asarray(crop)
        if crop.dtype != np.uint8 or crop.ndim != 3 or crop.shape[2] != 3 or crop.shape[0] == 0 or crop.shape[1] == 0:
            raise ValueError(f"crop {i} must be a non-empty H x W x 3 uint8 array, not {crop.dtype} {crop.shape}")
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        # INPUT_SIZE is (height, width); cv2 takes (width, height).
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        x = (rgb.astype(np.float32) / 255.0 - mean) / std
        batch[i] = x.transpose(2, 0, 1)
    return batch


class OSNetReIdentifier:
    """OSNet x1.0 with the MSMT17 re-ID weights, on one torch device.

    model is for tests: a torch module that maps (n, 3, 256, 128) to (n, 512).
    Without it, the constructor loads OSNet from the Hugging Face cache at
    the pinned revision (the first run on a computer must be online) and
    warms it up with one batch of two black crops.
    """

    model_name = OSNET_X1_0_MSMT17
    dim = 512

    def __init__(self, device: str = "cuda:0", batch_size: int = 16, model: torch.nn.Module | None = None) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size is {batch_size}; it must be 1 or more")
        self._batch_size = batch_size
        self._device = torch.device(device)
        self._model = model if model is not None else load_osnet(device)
        if model is None:
            h, w = INPUT_SIZE
            self.embed([np.zeros((h, w, 3), np.uint8)] * 2)

    def embed(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Return one row per BGR person crop, shape (len(crops), 512), float32."""
        crops = list(crops)
        if not crops:
            return np.zeros((0, self.dim), dtype=np.float32)
        rows = []
        with torch.inference_mode():
            for start in range(0, len(crops), self._batch_size):
                x = torch.from_numpy(osnet_input(crops[start : start + self._batch_size])).to(self._device)
                rows.append(self._model(x).float().cpu().numpy())
        return np.concatenate(rows).astype(np.float32, copy=False)
