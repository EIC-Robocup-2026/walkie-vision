"""The face recognisers the vision node can build, by provider name.

PROVIDERS maps each name to its ProviderSpec (walkie_vision.providers.spec),
and DEFAULT_PROVIDER is the one the vision node builds unless the parameter
face_recognition.provider names another. The provider classes and ONNX
Runtime are imported only inside the build functions, so importing this
module loads no model library. The buffalo_l file names come from
walkie_vision.weights, which buffalo_l.py reads them from too.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...weights import BUFFALO_L_DIR, DETECTOR_FILE, RECOGNIZER_FILE
from ..spec import LoadContext, ProviderSpec
from .base import PersonFaceRecognizer

DEFAULT_PROVIDER = "buffalo_l"


@dataclass(frozen=True)
class BuffaloLOptions:
    """buffalo_l's options: the detection threshold and the head crops.

    Each is the vision node's parameter face_recognition.buffalo_l.<option>,
    and the defaults equal config/vision.yaml.
    """

    det_thresh: float = 0.5
    input_sizes: tuple[int, ...] = (160, 256, 384, 640)
    max_people: int = 6
    max_offset: float = 0.25

    def __post_init__(self) -> None:
        # A list becomes a tuple, so the options stay hashable and immutable.
        object.__setattr__(self, "input_sizes", tuple(int(s) for s in self.input_sizes))
        for name in ("det_thresh", "max_offset"):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"face_recognition.buffalo_l.{name} must be in (0, 1], not {value}")
        if not self.input_sizes or any(s <= 0 or s % 32 for s in self.input_sizes):
            raise ValueError(
                "face_recognition.buffalo_l.input_sizes must be a non-empty list of positive multiples of 32, "
                f"not {list(self.input_sizes)}"
            )
        if self.max_people < 0:
            raise ValueError(f"face_recognition.buffalo_l.max_people must be at least 0, not {self.max_people}")


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
