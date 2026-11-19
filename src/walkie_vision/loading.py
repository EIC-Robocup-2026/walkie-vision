"""Loading the vision node's models: build_providers() checks their files, then builds every provider.

Each vision task's registry (walkie_vision.providers.<task>.registry) says how
one of its providers is checked, prepared and built. torch, Ultralytics and
ONNX Runtime are imported only while build_providers() runs, so importing
this module loads no model library. build_providers() logs to the logger it
is given (StartupLogger): the vision node (walkie_vision.node) passes its own
rclpy logger.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Protocol, TypeVar

from .config import VisionConfig
from .pipeline import ON_DEMAND_MAX_OBJECTS, Providers
from .providers.face_recognition import registry as face_recognition_registry
from .providers.face_recognition.registry import BuffaloLOptions
from .providers.image_embed import registry as image_embed_registry
from .providers.image_embed.registry import ClipOptions
from .providers.object_detection import registry as object_detection_registry
from .providers.object_detection.registry import YoloeOptions
from .providers.pose_estimation import registry as pose_estimation_registry
from .providers.pose_estimation.registry import YoloPoseOptions
from .providers.reid import registry as reid_registry
from .providers.reid.registry import OsnetOptions
from .providers.spec import LoadContext, ProviderSpec

_T = TypeVar("_T")

# Each vision task's registry, in the order the files are checked and the
# Hugging Face cache is prepared.
_REGISTRIES: dict[str, dict[str, ProviderSpec[Any]]] = {
    "object_detection": object_detection_registry.PROVIDERS,
    "pose_estimation": pose_estimation_registry.PROVIDERS,
    "face_recognition": face_recognition_registry.PROVIDERS,
    "reid": reid_registry.PROVIDERS,
    "image_embed": image_embed_registry.PROVIDERS,
}
# The provider each vision task builds, by its key in the task's registry.
# The model parameters have flat names, so the choice is fixed here.
_PROVIDER_NAMES = {
    "object_detection": "yoloe",
    "pose_estimation": "yolo_pose",
    "face_recognition": "buffalo_l",
    "reid": "osnet",
    "image_embed": "clip",
}


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


def _options(config: VisionConfig) -> dict[str, object]:
    """Each vision task's options, from the node's flat parameters."""
    return {
        "object_detection": YoloeOptions(
            weights=config.objects_weights,
            imgsz=config.objects_imgsz,
            conf=config.objects_conf,
            agnostic_nms=config.objects_agnostic_nms,
            tracker=config.objects_tracker,
        ),
        "pose_estimation": YoloPoseOptions(
            weights=config.pose_weights,
            imgsz=config.pose_imgsz,
            conf=config.pose_conf,
            tracker=config.pose_tracker,
        ),
        "face_recognition": BuffaloLOptions(
            det_thresh=config.face_det_thresh,
            input_sizes=config.face_input_sizes,
            max_people=config.face_max_people,
            max_offset=config.face_max_offset,
        ),
        "reid": OsnetOptions(batch_size=config.reid_batch_size),
        "image_embed": ClipOptions(),
    }


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
    tasks = [task for task in _REGISTRIES if task != "image_embed" or config.object_embeddings_enabled]
    specs = {task: _REGISTRIES[task][_PROVIDER_NAMES[task]] for task in tasks}
    options = _options(config)
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
