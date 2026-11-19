"""Loading the vision node's models: build_providers() checks their files, then builds every provider.

Each vision task's registry (walkie_vision.providers.<task>.registry) says how
one of its providers is checked, prepared and built, and the config names the
provider and holds its options (walkie_vision.config). torch, Ultralytics and
ONNX Runtime are imported only while build_providers() runs, so importing
this module loads no model library. build_providers() logs to the logger it
is given (StartupLogger): the vision node (walkie_vision.node) passes its own
rclpy logger.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Protocol, TypeVar

from .config import REGISTRIES, VisionConfig
from .pipeline import ON_DEMAND_MAX_OBJECTS, Providers
from .providers.spec import LoadContext

_T = TypeVar("_T")


class StartupLogger(Protocol):
    """What build_providers() logs to: an rclpy logger or a logging.Logger."""

    def info(self, message: str) -> object: ...

    def warning(self, message: str) -> object: ...


def _load(logger: StartupLogger, name: str, build: Callable[[], _T]) -> _T:
    """Build one model, and name it in the log when it fails, before the error reaches main()."""
    try:
        return build()
    except Exception as e:
        logger.warning(f"the {name} model failed to load: {type(e).__name__}: {e}")
        raise


def build_providers(config: VisionConfig, logger: StartupLogger) -> Providers:
    """Load every model the node runs, after checking that their files are there.

    The order makes a broken setup fail fast: the model files first, then
    the OSNet checkpoint in the Hugging Face cache, both before torch or
    Ultralytics is imported. Then the PyTorch models, then the ONNX Runtime
    sessions. Raises FileNotFoundError naming every missing file, and
    StartupError (a RuntimeError) when OSNet is not cached and cannot be
    downloaded. A model that fails to load logs its name and raises its own
    error.

    CLIP is checked and loaded only with object_embeddings_enabled, and only
    from the cache: it never downloads here. The second YOLOE is built only with
    on_demand_enabled, and here at start, so a first request never loads a model.
    """
    started = time.perf_counter()
    from .weights import models_dir

    # CLIP is checked, prepared and built only with object embeddings on.
    tasks = [task for task in REGISTRIES if task != "image_embed" or config.object_embeddings_enabled]
    # The provider each task builds, by its key in the task's registry, and its options.
    specs = {task: REGISTRIES[task][getattr(config, f"{task}_provider")] for task in tasks}
    options = {task: getattr(config, task) for task in tasks}
    root = models_dir()

    needed = [path for task in tasks for path in specs[task].files(options[task], root)]
    missing = [str(p) for p in needed if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"model files are missing: {', '.join(missing)}; run `pixi run fetch-models`")

    prepared = {
        task: None if spec.prepare is None else spec.prepare(options[task], logger) for task, spec in specs.items()
    }

    context = LoadContext(
        models_dir=root,
        device=config.device,
        classes=config.classes,
        # Asked for only when geometry or sampling reads them: copying masks to
        # the CPU costs time.
        keep_masks=config.geometry_enabled or config.object_crops_enabled or config.object_embeddings_enabled,
        logger=logger,
    )

    def build(task: str, name: str) -> Any:
        return _load(logger, name, specs[task].build(options[task], context, prepared[task]))

    objects = build("object_detection", "objects")
    detection = specs["object_detection"]
    # With no tracker, nothing reads a box under the published floor.
    on_demand = (
        _load(
            logger,
            "on-demand",
            detection.build_detector(
                options["object_detection"], context, config.objects_min_score, ON_DEMAND_MAX_OBJECTS
            ),
        )
        if config.on_demand_enabled and detection.build_detector is not None
        else None
    )
    pose = build("pose_estimation", "pose")
    reid = build("reid", "OSNet")
    embedder = build("image_embed", "CLIP") if config.object_embeddings_enabled else None
    faces = build("face_recognition", "face")

    import torch

    if torch.cuda.is_initialized():
        # Return the blocks freed during the loads and warm-ups, such as the
        # text encoder that set_classes() drops, so ONNX Runtime can use them.
        torch.cuda.empty_cache()
    logger.info(f"loaded the models in {time.perf_counter() - started:.1f} s")
    return Providers(objects=objects, pose=pose, faces=faces, reid=reid, embedder=embedder, on_demand=on_demand)
