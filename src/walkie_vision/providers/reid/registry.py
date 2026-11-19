"""The re-identifiers the vision node can build, by provider name.

PROVIDERS maps each name to its ProviderSpec (walkie_vision.providers.spec).
The provider classes and huggingface_hub are imported only inside the
functions, so importing this module loads no model library.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...errors import StartupError
from ...weights import HUB_MODELS, OSNET_CHECKPOINT
from ..spec import LoadContext, Logger, ProviderSpec
from .base import ReIdentifier


@dataclass(frozen=True)
class OsnetOptions:
    """OSNet's options: how many person crops go through it at once."""

    batch_size: int


def _osnet_files(options: OsnetOptions, models_dir: Path) -> tuple[Path, ...]:
    # The checkpoint is in the Hugging Face cache, which _prepare_osnet checks.
    return ()


def _prepare_osnet(options: OsnetOptions, logger: Logger) -> None:
    import huggingface_hub

    hub = HUB_MODELS["osnet"]
    cached = huggingface_hub.try_to_load_from_cache(hub.repo, OSNET_CHECKPOINT, revision=hub.revision)
    if not isinstance(cached, str):
        logger.warning("OSNet is not in the Hugging Face cache; downloading it now, which needs the internet once")
        try:
            huggingface_hub.hf_hub_download(hub.repo, OSNET_CHECKPOINT, revision=hub.revision)
        except Exception as e:
            raise StartupError(
                f"OSNet is not in the Hugging Face cache and could not be downloaded: {e}. "
                "Run once with internet access; see docs/guides/vision-node.md"
            ) from e


def _build_osnet(options: OsnetOptions, context: LoadContext, prepared: None) -> Callable[[], ReIdentifier]:
    from walkie_vision.providers.reid.osnet import OSNetReIdentifier

    return lambda: OSNetReIdentifier(device=context.device, batch_size=options.batch_size)


PROVIDERS: dict[str, ProviderSpec[Any]] = {
    "osnet": ProviderSpec(options=OsnetOptions, files=_osnet_files, prepare=_prepare_osnet, build=_build_osnet),
}
