"""EmbedText: the request check, the response and its failure, and the reader of the vectors (ADR-0015).

The callback checks the request with embed_text_problem() and leaves a job in
the slot of walkie_vision.services.jobs; the worker fills the response with
fill_embed_text_response().
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from walkie_vision_msgs.srv import EmbedText

from ..convert import float32_array, unit_vector
from ..providers import Embedder
from .jobs import bounded_message

MAX_TEXTS = 16
MAX_TEXT_CHARS = 512
EMBED_TEXT_DEADLINE_S = 2.0
"""EmbedText's request has no timeout field. The caller allows 5 s."""


def embed_text_problem(texts: Sequence[str]) -> str | None:
    """Return why an EmbedText request is refused, or None. The reason never repeats a text."""
    if not texts:
        return "texts is empty"
    if len(texts) > MAX_TEXTS:
        return f"at most {MAX_TEXTS} texts, not {len(texts)}"
    for i, text in enumerate(texts):
        if not text.strip():
            return f"text {i} is blank"
        if len(text) > MAX_TEXT_CHARS:
            return f"text {i} has {len(text)} characters, more than {MAX_TEXT_CHARS}"
    return None


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
        response.message = bounded_message(f"{type(e).__name__}: {e}")
        return response
    response.success = True
    response.message = ""
    response.embedding_model = embedder.model_name
    response.dim = embedder.dim
    response.vectors = float32_array(rows.reshape(-1))
    return response


def embed_text_failure(code: str, message: str) -> EmbedText.Response:
    """An EmbedText failure. The type has no error code field, so the code starts the message."""
    return EmbedText.Response(success=False, message=bounded_message(f"{code}: {message}"))


def vectors_from_response(response: EmbedText.Response) -> np.ndarray:
    """Return an EmbedText response's vectors as a float32 array of shape (n, dim)."""
    if not response.success:
        raise ValueError(f"EmbedText failed: {response.message}")
    vectors = np.asarray(response.vectors, dtype=np.float32)
    if response.dim == 0 or vectors.size % response.dim:
        raise ValueError(f"{vectors.size} values do not split into vectors of {response.dim}")
    return vectors.reshape(-1, response.dim)
