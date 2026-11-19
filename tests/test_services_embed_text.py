"""walkie_vision.services.embed_text: the request check, the response and the reader of its vectors. No ROS graph."""

from __future__ import annotations

import numpy as np
import pytest
from rclpy.serialization import deserialize_message, serialize_message
from walkie_vision_msgs.srv import EmbedText

from walkie_vision.fakes import FakeEmbedder
from walkie_vision.providers import CLIP_VIT_B16
from walkie_vision.services.embed_text import (
    MAX_TEXT_CHARS,
    MAX_TEXTS,
    embed_text_problem,
    fill_embed_text_response,
    vectors_from_response,
)

# --- the checks ------------------------------------------------------------


@pytest.mark.parametrize(
    ("texts", "reason"),
    [
        ([], "empty"),
        (["mug"] * (MAX_TEXTS + 1), "at most 16"),
        (["mug", "   "], "text 1 is blank"),
        (["mug", ""], "text 1 is blank"),
        (["x" * (MAX_TEXT_CHARS + 1)], "more than 512"),
    ],
)
def test_bad_texts_are_refused_without_repeating_them(texts, reason):
    problem = embed_text_problem(texts)
    assert reason in problem
    assert "xxx" not in problem and "mug" not in problem


def test_texts_at_the_caps_are_valid():
    assert embed_text_problem(["x" * MAX_TEXT_CHARS] * MAX_TEXTS) is None


# --- the response ----------------------------------------------------------


def test_embed_text_round_trip():
    embedder = FakeEmbedder(model_name=CLIP_VIT_B16, dim=8)
    texts = ["a red mug", "a chair"]
    response = fill_embed_text_response(EmbedText.Response(), embedder, texts)
    assert response.success and response.message == ""
    assert (response.embedding_model, response.dim) == (CLIP_VIT_B16, 8)
    back = deserialize_message(serialize_message(response), EmbedText.Response)
    vectors = vectors_from_response(back)
    assert vectors.shape == (2, 8)
    np.testing.assert_allclose(vectors, embedder.embed_texts(texts), rtol=1e-6)


class _BrokenEmbedder(FakeEmbedder):
    def __init__(self, result=None, error=None):
        super().__init__(dim=8)
        self.result, self.error = result, error

    def embed_texts(self, texts):
        if self.error:
            raise self.error
        return self.result


@pytest.mark.parametrize(
    ("embedder", "texts", "reason"),
    [
        (FakeEmbedder(dim=8), [], "texts is empty"),
        (_BrokenEmbedder(error=RuntimeError("CUDA out of memory")), ["mug"], "RuntimeError: CUDA out of memory"),
        (_BrokenEmbedder(result=np.ones((2, 8))), ["mug"], "expected shape (1, 8)"),
        (_BrokenEmbedder(result=np.zeros((1, 8))), ["mug"], "zero"),
    ],
)
def test_embed_text_failures_answer_instead_of_raising(embedder, texts, reason):
    response = fill_embed_text_response(EmbedText.Response(), embedder, texts)
    assert not response.success
    assert reason in response.message
    assert response.embedding_model == "" and len(response.vectors) == 0
    with pytest.raises(ValueError, match="EmbedText failed"):
        vectors_from_response(response)
