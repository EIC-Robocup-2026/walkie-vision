"""walkie_vision.embeddings: the CLIP provider, without a GPU or a model file.

The provider's tests inject a stub loader and a stub open_clip, so nothing
downloads or loads a model.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import torch
from huggingface_hub.errors import LocalEntryNotFoundError

from walkie_vision.embeddings import ClipEmbedder, clip_checkpoint

CLIP_REPO = "timm/vit_base_patch16_clip_224.openai"
CLIP_REVISION = "977e3dd0ec55ab8da155f2fbeb6b5f54948b6e3d"
WIDTH, HEIGHT = 64, 48
IMAGE = np.random.default_rng(0).integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)


# --- the CLIP provider, with stubs -----------------------------------------


def test_the_weights_load_from_the_cache_at_the_pinned_commit():
    calls = []

    def stub(repo, filename, *, revision, local_files_only):
        calls.append((repo, filename, revision, local_files_only))
        return "/cache/open_clip_model.safetensors"

    assert clip_checkpoint(stub) == "/cache/open_clip_model.safetensors"
    assert calls == [(CLIP_REPO, "open_clip_model.safetensors", CLIP_REVISION, True)]


def test_weights_that_are_not_cached_say_what_to_run():
    def stub(*args, **kwargs):
        raise LocalEntryNotFoundError("not cached")

    with pytest.raises(FileNotFoundError, match="pixi run fetch-models"):
        clip_checkpoint(stub)


def test_the_loader_uses_quickgelu_and_the_local_file(monkeypatch):
    calls = []

    class Model:
        evaluating = False

        def eval(self):
            self.evaluating = True

    model = Model()

    def create(name, *, pretrained, device):
        calls.append((name, pretrained, device))
        return model, "train transform", "preprocess"

    stub = types.SimpleNamespace(create_model_and_transforms=create, get_tokenizer=lambda name: f"tokenizer {name}")
    monkeypatch.setitem(sys.modules, "open_clip", stub)

    built = ClipEmbedder("/cache/weights.safetensors", device="cpu")

    assert calls == [("ViT-B-16-quickgelu", "/cache/weights.safetensors", "cpu")]
    assert model.evaluating
    assert built._preprocess == "preprocess"
    assert built._tokenizer == "tokenizer ViT-B-16-quickgelu"


class StubClip:
    """Stands in for the open_clip model, preprocessing and tokenizer."""

    def __init__(self, image_rows=None) -> None:
        self.pil_images = []
        self.tokenized: list[str] = []
        self.image_rows = image_rows

    def build(self, checkpoint: str, device: str):
        assert checkpoint == "/cache/weights.safetensors" and device == "cpu"
        return self, self.preprocess, self.tokenizer

    def preprocess(self, image):
        self.pil_images.append(image)
        return torch.zeros(3, 224, 224)

    def tokenizer(self, texts):
        self.tokenized.extend(texts)
        return torch.zeros(len(texts), 77, dtype=torch.long)

    def encode_image(self, batch):
        if self.image_rows is not None:
            return torch.tensor(self.image_rows, dtype=torch.float32)
        rows = torch.zeros(batch.shape[0], 512)
        rows[:, 0] = 3.0
        rows[:, 1] = 4.0
        return rows

    def encode_text(self, batch):
        return torch.ones(batch.shape[0], 512)


def _clip(stub: StubClip) -> ClipEmbedder:
    return ClipEmbedder("/cache/weights.safetensors", device="cpu", build=stub.build)


def test_the_provider_names_the_model_and_its_length():
    embedder = _clip(StubClip())
    assert embedder.model_name == "clip_vit_b16" and embedder.dim == 512


def test_clip_gets_rgb_pixels():
    stub = StubClip()
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[..., 0] = 200  # blue in BGR
    _clip(stub).embed_images([image])
    assert stub.pil_images[0].getpixel((0, 0)) == (0, 0, 200)


def test_image_rows_are_unit_length_float32():
    vectors = _clip(StubClip()).embed_images([IMAGE, IMAGE])
    assert vectors.shape == (2, 512) and vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-6)


def test_a_zero_row_stays_zero_for_the_caller_to_refuse():
    rows = np.zeros((1, 512), dtype=np.float32).tolist()
    vectors = _clip(StubClip(rows)).embed_images([IMAGE])
    assert not vectors.any()


def test_texts_always_use_the_template():
    stub = StubClip()
    vectors = _clip(stub).embed_texts(["red mug", "cup {with} braces"])
    assert stub.tokenized == ["a photo of red mug.", "a photo of cup {with} braces."]
    assert vectors.shape == (2, 512)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-6)


def test_empty_batches_do_not_reach_the_model():
    stub = StubClip()
    embedder = _clip(stub)
    assert embedder.embed_images([]).shape == (0, 512)
    assert embedder.embed_texts([]).shape == (0, 512)
    assert stub.pil_images == [] and stub.tokenized == []
