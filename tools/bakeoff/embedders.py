"""The candidate models, behind one interface.

Each candidate turns images and texts into unit-length float32 vectors.
prepare() only checks (or downloads) the model files, so that a missing
model fails the run before any model is loaded. load() reads the files from
the local Hugging Face cache, never from the network.

Every model is pinned to the commit in walkie_vision.models.HUB_MODELS.
"""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from walkie_vision.models import HUB_MODELS, HubModel


class ModelUnavailable(RuntimeError):
    """A model's files are not on this computer and may not be downloaded."""


def hub_snapshot(hub: HubModel, files: Sequence[str], allow_download: bool) -> Path:
    """Return the local snapshot directory of a pinned repository, with `files` in it.

    Each file is looked up in the Hugging Face cache first. A missing file is
    downloaded only when allow_download is true; otherwise ModelUnavailable
    says what is missing and how to get it.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.constants import HF_HUB_CACHE
    from huggingface_hub.errors import LocalEntryNotFoundError

    found, missing = {}, []
    for name in files:
        try:
            found[name] = Path(hf_hub_download(hub.repo, name, revision=hub.revision, local_files_only=True))
        except LocalEntryNotFoundError:
            missing.append(name)
    if missing:
        where = f"{hub.repo} at {hub.revision}"
        if not allow_download:
            raise ModelUnavailable(
                f"{where}: {', '.join(missing)} not in the Hugging Face cache ({HF_HUB_CACHE}). "
                "Run once without --offline, with internet access, to download it."
            )
        for name in missing:
            try:
                found[name] = Path(hf_hub_download(hub.repo, name, revision=hub.revision))
            except Exception as e:  # network, gated repository, missing file
                raise ModelUnavailable(f"{where}: cannot download {name}: {e}") from e
    first = files[0]
    if "/" in first:
        raise ValueError("the first file must be at the top of the repository")
    return found[first].parent


def _normalize(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


class Candidate(ABC):
    """One model configuration in the bake-off."""

    name: str
    hub: HubModel
    files: tuple[str, ...]

    def __init__(self) -> None:
        self.root: Path | None = None
        self.device = "cpu"
        self.dtype = "float32"

    def prepare(self, allow_download: bool) -> None:
        self.root = hub_snapshot(self.hub, self.files, allow_download)

    @abstractmethod
    def load(self, device: str) -> None: ...

    @abstractmethod
    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        """Return one unit-length float32 row per image."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Return one unit-length float32 row per query."""

    def info(self) -> dict:
        return {"repo": self.hub.repo, "revision": self.hub.revision, "licence": self.hub.licence, "dtype": self.dtype}

    def close(self) -> None:
        """Drop the model and give its GPU memory back."""
        for attr in ("model", "processor", "preprocess", "tokenizer"):
            if hasattr(self, attr):
                delattr(self, attr)
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ClipViTB16(Candidate):
    """OpenAI CLIP ViT-B/16 through open_clip: the model behind today's *_clip_vit_b16 tables."""

    name = "clip_vit_b16"
    hub = HUB_MODELS["clip"]
    files = ("open_clip_model.safetensors",)
    template = "a photo of {}."

    def load(self, device: str) -> None:
        import open_clip

        self.device = device
        # The OpenAI weights were trained with QuickGELU (see checks/c07_clip.py).
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16-quickgelu", pretrained=str(self.root / self.files[0]), device=device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-16-quickgelu")
        self.dtype = "float16 autocast" if device.startswith("cuda") else "float32"

    def _run(self, fn, x):
        import torch

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=self.device.startswith("cuda")):
            return _normalize(fn(x.to(self.device)).float().cpu())

    def embed_images(self, images):
        import torch

        return self._run(self.model.encode_image, torch.stack([self.preprocess(im) for im in images]))

    def embed_texts(self, texts):
        return self._run(self.model.encode_text, self.tokenizer([self.template.format(t) for t in texts]))

    def info(self):
        return {**super().info(), "template": self.template, "preprocess": "resize 224 + centre crop"}


class Siglip2B16(Candidate):
    """SigLIP 2 base, patch 16, 256 px, through transformers."""

    name = "siglip2_b16_256"
    hub = HUB_MODELS["siglip2"]
    files = (
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    template = "this is a photo of {}."

    def load(self, device: str) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self.device = device
        self.torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.dtype = str(self.torch_dtype).removeprefix("torch.")
        self.model = AutoModel.from_pretrained(self.root, dtype=self.torch_dtype).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(self.root)

    @staticmethod
    def _pooled(out):
        # transformers 5 returns a ModelOutput from get_*_features.
        return out if not hasattr(out, "pooler_output") else out.pooler_output

    def embed_images(self, images):
        import torch

        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.torch_dtype)
        with torch.inference_mode():
            return _normalize(self._pooled(self.model.get_image_features(**inputs)).float().cpu())

    def embed_texts(self, texts):
        import torch

        # SigLIP 2 was trained on lower-case text padded to 64 tokens. The
        # Gemma tokenizer does neither by itself.
        texts = [self.template.format(t).lower() for t in texts]
        inputs = self.processor(
            text=texts, padding="max_length", max_length=64, truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            return _normalize(self._pooled(self.model.get_text_features(**inputs)).float().cpu())

    def info(self):
        return {
            **super().info(),
            "template": self.template + " (lower case, padded to 64 tokens)",
            "preprocess": "resize to 256 x 256",
        }


class EmbeddingGemma2(Candidate):
    """EmbeddingGemma 2 through sentence-transformers, at one image soft-token budget."""

    hub = HUB_MODELS["embeddinggemma2"]
    files = (
        "modules.json",
        "config.json",
        "config_sentence_transformers.json",
        "sentence_bert_config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "processor_config.json",
        "preprocessor_config.json",
        "1_Pooling/config.json",
        "2_Normalize/config.json",
    )
    prompt_name = "Retrieval-query"
    patches_per_token = 9
    """The image processor pools 3 x 3 patches of 16 px into one soft token."""

    def __init__(self, budget: int) -> None:
        super().__init__()
        self.budget = budget
        self.name = f"eg2_t{budget}"
        self.budget_check: dict = {}

    def load(self, device: str) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        self.device = device
        # bf16 only where the GPU has it in hardware; never fp16.
        bf16 = device.startswith("cuda") and torch.cuda.is_bf16_supported(including_emulation=False)
        torch_dtype = torch.bfloat16 if bf16 else torch.float32
        self.dtype = str(torch_dtype).removeprefix("torch.")
        self.model = SentenceTransformer(
            str(self.root),
            device=device,
            model_kwargs={"dtype": torch_dtype},
            # The audio tower is not needed for text and images.
            config_kwargs={"audio_config": None},
        )
        self.model.eval()
        self.prompt = self.model.prompts[self.prompt_name]
        self._check_budget()

    def _kwargs(self) -> dict:
        return {"processing_kwargs": {"image": {"max_soft_tokens": self.budget}}}

    def _check_budget(self) -> None:
        """Make sure the soft-token budget reaches the image processor.

        The processor's default is 280 tokens. If the budget were ignored,
        pixel_values would have 280 x 9 patches instead of budget x 9.
        """
        features = self.model.preprocess([Image.new("RGB", (256, 192), (128, 64, 32))], **self._kwargs())
        patches = int(features["pixel_values"].shape[1])
        token_id = self.model[0].processor.image_token_id
        tokens = int((features["input_ids"] == token_id).sum())
        self.budget_check = {"patches": patches, "image_tokens_256x192": tokens}
        if patches != self.budget * self.patches_per_token or not 0 < tokens <= self.budget:
            raise RuntimeError(
                f"{self.name}: the soft-token budget did not take effect: {patches} patches "
                f"(expected {self.budget * self.patches_per_token}) and {tokens} image tokens for one image"
            )

    def embed_images(self, images):
        out = self.model.encode(
            images,
            batch_size=len(images),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            **self._kwargs(),
        )
        return _normalize(out)

    def embed_texts(self, texts):
        out = self.model.encode(
            list(texts),
            prompt_name=self.prompt_name,
            batch_size=len(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return _normalize(out)

    def info(self):
        return {
            **super().info(),
            "template": f"{self.prompt}{{}} (prompt {self.prompt_name}); images have no prompt",
            "preprocess": f"aspect kept, at most {self.budget} soft tokens",
            "max_soft_tokens": self.budget,
            "budget_check": self.budget_check,
        }


def all_candidates() -> list[Candidate]:
    """The four configurations, in report order."""
    return [ClipViTB16(), Siglip2B16(), EmbeddingGemma2(70), EmbeddingGemma2(140)]


CANDIDATE_NAMES = ("clip_vit_b16", "siglip2_b16_256", "eg2_t70", "eg2_t140")
