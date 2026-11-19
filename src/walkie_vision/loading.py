"""Loading the vision node's models: build_providers() checks their files, then builds every provider.

torch, Ultralytics and ONNX Runtime are imported only inside build_providers(),
so importing this module loads no model library. build_providers() logs to the
logger it is given (StartupLogger): the vision node (walkie_vision.node) passes
its own rclpy logger.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TypeVar

from .config import VisionConfig
from .errors import StartupError
from .pipeline import ON_DEMAND_MAX_OBJECTS, Providers

_T = TypeVar("_T")

# The files build_providers() needs in models/buffalo_l, as named in
# walkie_vision.weights (DETECTOR_FILE and RECOGNIZER_FILE).
_FACE_FILES = ("det_10g.onnx", "w600k_r50.onnx")


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
    from .providers.image_embed.clip import ClipEmbedder, clip_checkpoint
    from .providers.object_detection.yoloe import MOBILECLIP_FILE
    from .weights import BUFFALO_L_DIR, HUB_MODELS, OSNET_CHECKPOINT, ULTRALYTICS_DIR, models_dir

    root = models_dir()
    ultralytics_dir = (root / ULTRALYTICS_DIR).absolute()
    face_dir = (root / BUFFALO_L_DIR).absolute()

    def weights(name: str) -> Path:
        path = Path(name)
        return path if path.is_absolute() else ultralytics_dir / path

    objects_path = weights(config.objects_weights)
    pose_path = weights(config.pose_weights)
    # YOLOE loads its text encoder by file name from the folder of its weights.
    needed = [objects_path, objects_path.parent / MOBILECLIP_FILE, pose_path, *(face_dir / f for f in _FACE_FILES)]
    missing = [str(p) for p in needed if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"model files are missing: {', '.join(missing)}; run `pixi run fetch-models`")

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
    clip_path = clip_checkpoint() if config.object_embeddings_enabled else None

    from walkie_vision.providers.object_detection.yoloe import YoloeDetector, YoloeTracker
    from walkie_vision.providers.pose_estimation.yolo_pose import YoloPoseTracker

    # Asked for only when geometry or sampling reads them: copying masks to
    # the CPU costs time.
    keep_masks = (
        {"keep_masks": True}
        if config.geometry_enabled or config.object_crops_enabled or config.object_embeddings_enabled
        else {}
    )
    objects = _load(
        logger,
        "objects",
        lambda: YoloeTracker(
            objects_path,
            list(config.classes),
            device=config.device,
            imgsz=config.objects_imgsz,
            conf=config.objects_conf,
            agnostic_nms=config.objects_agnostic_nms,
            tracker=config.objects_tracker,
            **keep_masks,
        ),
    )
    # With no tracker, nothing reads a box under the published floor.
    on_demand = (
        _load(
            logger,
            "on-demand",
            lambda: YoloeDetector(
                objects_path,
                list(config.classes),
                device=config.device,
                imgsz=config.objects_imgsz,
                conf=config.objects_min_score,
                agnostic_nms=config.objects_agnostic_nms,
                max_det=ON_DEMAND_MAX_OBJECTS,
                **keep_masks,
            ),
        )
        if config.on_demand_enabled
        else None
    )
    pose = _load(
        logger,
        "pose",
        lambda: YoloPoseTracker(
            pose_path,
            device=config.device,
            imgsz=config.pose_imgsz,
            conf=config.pose_conf,
            tracker=config.pose_tracker,
        ),
    )

    from walkie_vision.providers.reid.osnet import OSNetReIdentifier

    reid = _load(logger, "OSNet", lambda: OSNetReIdentifier(device=config.device, batch_size=config.reid_batch_size))
    embedder = None if clip_path is None else _load(logger, "CLIP", lambda: ClipEmbedder(clip_path, device=config.device))

    import onnxruntime

    from walkie_vision.providers.face_recognition.buffalo_l import FaceAnalyzer, HeadCropConfig, HeadFaceRecognizer

    # ERROR and above only, for every session created from here on. SCRFD's
    # file declares its output shapes for a 640 input and ArcFace's for a
    # batch of 1, so ONNX Runtime would warn on every smaller crop and every
    # batch of faces, burying the node's own warnings. A session takes the
    # level when it is created, so this must come before FaceAnalyzer.
    onnxruntime.set_default_logger_severity(3)
    logger.info("warming up the face models; the first run on this GPU can take minutes")
    faces = _load(
        logger,
        "face",
        lambda: HeadFaceRecognizer(
            FaceAnalyzer(face_dir, det_thresh=config.face_det_thresh),
            HeadCropConfig(
                input_sizes=tuple(config.face_input_sizes),
                max_people=config.face_max_people,
                max_face_offset=config.face_max_offset,
            ),
        ),
    )

    import torch

    if torch.cuda.is_initialized():
        # Return the blocks freed during the loads and warm-ups, such as the
        # text encoder that set_classes() drops, so ONNX Runtime can use them.
        torch.cuda.empty_cache()
    logger.info(f"loaded the models in {time.perf_counter() - started:.1f} s")
    return Providers(objects=objects, pose=pose, faces=faces, reid=reid, embedder=embedder, on_demand=on_demand)
