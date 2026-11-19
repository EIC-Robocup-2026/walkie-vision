"""The image embedders the vision node can build, by provider name.

PROVIDERS maps each name to its ProviderSpec (walkie_vision.providers.spec),
and DEFAULT_PROVIDER is the one the vision node builds unless the parameter
image_embed.provider names another. The provider classes are imported only
inside the functions, so importing this module loads no model library.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..spec import LoadContext, Logger, ProviderSpec
from .base import Embedder

DEFAULT_PROVIDER = "clip"


@dataclass(frozen=True)
class ClipOptions:
    """CLIP's options. It has none: the weights are pinned in walkie_vision.weights."""


def _clip_files(options: ClipOptions, models_dir: Path) -> tuple[Path, ...]:
    # The weights are in the Hugging Face cache, which _prepare_clip checks.
    return ()


def _prepare_clip(options: ClipOptions, logger: Logger) -> str:
    from walkie_vision.providers.image_embed.clip import clip_checkpoint

    # Only from the cache: it never downloads here.
    return clip_checkpoint()


def _build_clip(options: ClipOptions, context: LoadContext, prepared: str) -> Callable[[], Embedder]:
    from walkie_vision.providers.image_embed.clip import ClipEmbedder

    return lambda: ClipEmbedder(prepared, device=context.device)


PROVIDERS: dict[str, ProviderSpec[Any]] = {
    "clip": ProviderSpec(options=ClipOptions, files=_clip_files, prepare=_prepare_clip, build=_build_clip),
}
