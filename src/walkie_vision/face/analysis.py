"""Face detection and recognition with the buffalo_l models on ONNX Runtime.

FaceAnalyzer does what InsightFace's FaceAnalysis does for the two models the
robot uses: SCRFD-10GF (det_10g.onnx) finds faces and their five landmarks,
and ArcFace R50 (w600k_r50.onnx) turns each aligned face into a 512-value
embedding. The other buffalo_l models (68 and 106 point landmarks, gender and
age) are not loaded.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnxruntime as ort

from ..providers import ARCFACE_W600K_R50, Face
from .arcface import ArcFaceONNX
from .scrfd import SCRFD

DETECTOR_FILE = "det_10g.onnx"
RECOGNIZER_FILE = "w600k_r50.onnx"
DEFAULT_PROVIDERS = ("CUDAExecutionProvider", "CPUExecutionProvider")


class FaceAnalyzer:
    """SCRFD detection plus ArcFace embeddings on one set of providers.

    model_dir holds det_10g.onnx and w600k_r50.onnx (`pixi run fetch-models`
    puts them in models/buffalo_l). providers is passed to ONNX Runtime as
    is; the first one must be available, or the constructor raises instead of
    letting ONNX Runtime fall back to the next one without telling anyone.

    FaceAnalyzer is a walkie_vision.providers.FaceRecognizer.
    """

    model_name = ARCFACE_W600K_R50
    dim = 512

    def __init__(
        self,
        model_dir: str | Path,
        providers: Sequence[str] = DEFAULT_PROVIDERS,
        det_size: tuple[int, int] = (640, 640),
        det_thresh: float = 0.5,
        nms_thresh: float = 0.4,
    ) -> None:
        model_dir = Path(model_dir)
        self.providers = list(providers)
        if "CUDAExecutionProvider" in self.providers:
            # The CUDA provider library does not find the CUDA and cuDNN
            # wheels by itself; without this, ONNX Runtime falls back to the
            # CPU provider (see onnxruntime-gpu in pixi.toml).
            ort.preload_dlls()
        det_path = model_dir / DETECTOR_FILE
        rec_path = model_dir / RECOGNIZER_FILE
        self.detector = SCRFD(str(det_path), session=self._session(det_path))
        self.detector.prepare(input_size=det_size, det_thresh=det_thresh, nms_thresh=nms_thresh)
        self.recognizer = ArcFaceONNX(str(rec_path), session=self._session(rec_path))

    def _session(self, path: Path) -> ort.InferenceSession:
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing; run `pixi run fetch-models`")
        session = ort.InferenceSession(str(path), providers=self.providers)
        active = session.get_providers()
        if active[0] != self.providers[0]:
            raise RuntimeError(
                f"{path.name} runs on {active[0]}, not {self.providers[0]}; "
                f"available providers: {ort.get_available_providers()}"
            )
        return session

    def get(self, img: np.ndarray, max_num: int = 0) -> list[Face]:
        """Detect the faces in a BGR image and compute their embeddings.

        max_num > 0 keeps only that many faces, preferring large faces near
        the image centre.
        """
        bboxes, kpss = self.detector.detect(img, max_num=max_num, metric="default")
        faces = []
        for i in range(bboxes.shape[0]):
            face = Face(bbox=bboxes[i, 0:4], kps=kpss[i], det_score=float(bboxes[i, 4]))
            self.recognizer.get(img, face)
            faces.append(face)
        return faces

    def recognize(self, image: np.ndarray) -> list[Face]:
        """Detect every face in a BGR image and compute its embedding."""
        return self.get(image)
