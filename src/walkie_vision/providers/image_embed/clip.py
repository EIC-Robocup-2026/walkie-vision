"""The interim CLIP provider.

The object sampler and the padded crops that CLIP embeds are in
crops.objects, and the objects' colour shares in crops.colour.

ClipEmbedder is the interim text and image embedder. ADR-0008 stays Proposed
until the bake-off measures the alternatives. The model name stays
clip_vit_b16, because the weights, QuickGELU and the text template are the
ones in tools/bakeoff/embedders.py. A raw-token vector would not compare with
a templated one, so embed_texts() always applies the template.

The node only loads the weights from the Hugging Face cache, at the commit in
HUB_MODELS. `pixi run fetch-models` is the one thing that downloads them.

torch, open_clip and huggingface_hub are imported on use, so this module
loads without them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import cv2
import numpy as np

from ...weights import CLIP_WEIGHTS, HUB_MODELS
from ..vector_tables import CLIP_VIT_B16

CLIP_ARCHITECTURE = "ViT-B-16-quickgelu"
"""The OpenAI weights were trained with QuickGELU (see checks/c07_clip.py)."""
CLIP_DIM = 512
TEXT_TEMPLATE = "a photo of {}."


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
