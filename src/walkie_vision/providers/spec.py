"""ProviderSpec: how the vision node checks, prepares and builds one provider.

Each vision task's registry.py maps a provider name to a ProviderSpec.
walkie_vision.loading.build_providers() runs the files of every spec first,
then every prepare, then every build, so a broken setup fails before torch,
Ultralytics or ONNX Runtime is imported. A spec imports its provider class
only inside build, so importing a registry loads no model library.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar

OptionsT = TypeVar("OptionsT")


class Logger(Protocol):
    """What a spec logs to: the logger that build_providers() was given."""

    def info(self, message: str) -> object: ...

    def warning(self, message: str) -> object: ...


@dataclass(frozen=True)
class LoadContext:
    """What every build gets besides its options."""

    models_dir: Path
    """The models/ folder, from walkie_vision.weights.models_dir()."""
    device: str
    classes: tuple[str, ...]
    """The node's object classes. Only object detection reads them."""
    keep_masks: bool
    """Whether anything reads the object masks. Only object detection reads it."""
    logger: Logger


@dataclass(frozen=True)
class ProviderSpec(Generic[OptionsT]):
    """One provider of a vision task, as its registry lists it.

    files(options, models_dir) returns the files the provider needs, checked
    before any model library is imported. prepare(options, logger), when set,
    checks the Hugging Face cache or downloads once, and its result reaches
    build as prepared. build(options, context, prepared) imports the provider
    class and returns a function that constructs the provider:
    build_providers() calls that function itself, so it can name the model in
    the log when the construction fails, while an import that fails raises as
    it is. build_detector(options, context, conf, max_det), when set, does the
    same for a second detector with no tracker, from the same weights, for
    DetectObjects; conf and max_det come from above the providers.
    """

    options: type[OptionsT]
    files: Callable[[OptionsT, Path], tuple[Path, ...]]
    prepare: Callable[[OptionsT, Logger], Any] | None
    build: Callable[[OptionsT, LoadContext, Any], Callable[[], Any]]
    build_detector: Callable[[OptionsT, LoadContext, float, int], Callable[[], Any]] | None = None
