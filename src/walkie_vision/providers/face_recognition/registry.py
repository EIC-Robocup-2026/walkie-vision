"""The face recognisers the vision node can build, by provider name.

PROVIDERS maps each name to its ProviderSpec (walkie_vision.providers.spec).
The provider classes and ONNX Runtime are imported only inside the build
functions, so importing this module loads no model library. The buffalo_l file
names come from walkie_vision.weights, which buffalo_l.py reads them from too.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...weights import BUFFALO_L_DIR, DETECTOR_FILE, RECOGNIZER_FILE
from ..spec import LoadContext, ProviderSpec
from .base import PersonFaceRecognizer


@dataclass(frozen=True)
class BuffaloLOptions:
    """buffalo_l's options: the detection threshold and the head crops."""

    det_thresh: float
    input_sizes: tuple[int, ...]
    max_people: int
    max_offset: float


def _buffalo_l_dir(models_dir: Path) -> Path:
    return (models_dir / BUFFALO_L_DIR).absolute()


def _buffalo_l_files(options: BuffaloLOptions, models_dir: Path) -> tuple[Path, ...]:
    face_dir = _buffalo_l_dir(models_dir)
    return (face_dir / DETECTOR_FILE, face_dir / RECOGNIZER_FILE)


def _build_buffalo_l(
    options: BuffaloLOptions, context: LoadContext, prepared: None
) -> Callable[[], PersonFaceRecognizer]:
    import onnxruntime

    from walkie_vision.providers.face_recognition.buffalo_l import FaceAnalyzer, HeadCropConfig, HeadFaceRecognizer

    # ERROR and above only, for every session created from here on. SCRFD's
    # file declares its output shapes for a 640 input and ArcFace's for a
    # batch of 1, so ONNX Runtime would warn on every smaller crop and every
    # batch of faces, burying the node's own warnings. A session takes the
    # level when it is created, so this must come before FaceAnalyzer.
    onnxruntime.set_default_logger_severity(3)
    context.logger.info("warming up the face models; the first run on this GPU can take minutes")
    face_dir = _buffalo_l_dir(context.models_dir)
    return lambda: HeadFaceRecognizer(
        FaceAnalyzer(face_dir, det_thresh=options.det_thresh),
        HeadCropConfig(
            input_sizes=tuple(options.input_sizes),
            max_people=options.max_people,
            max_face_offset=options.max_offset,
        ),
    )


PROVIDERS: dict[str, ProviderSpec[Any]] = {
    "buffalo_l": ProviderSpec(options=BuffaloLOptions, files=_buffalo_l_files, prepare=None, build=_build_buffalo_l),
}
