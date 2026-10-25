"""Face detection and recognition, vendored from InsightFace.

scrfd.py, arcface.py and face_align.py are the parts of InsightFace's MIT
library code (python-package v0.7.3, commit 668e8d4) that the buffalo_l
detector and recogniser need, adapted to run on onnxruntime-gpu directly.
The InsightFace package itself is not installed; see pixi.toml for why.

The buffalo_l model files are a separate matter: InsightFace releases them
for non-commercial research only.

heads.py is this repository's own code: it runs the detector on a crop
around each person's head, from the pose keypoints, and gives each person
their face.
"""

from .analysis import Face, FaceAnalyzer
from .heads import HeadCropConfig, HeadFaceRecognizer

__all__ = ["Face", "FaceAnalyzer", "HeadCropConfig", "HeadFaceRecognizer"]
