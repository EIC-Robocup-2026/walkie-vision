"""The interfaces of the vision models, and the results they return.

The vision node (walkie_vision.node) calls each model only through one of
these Protocols. A test passes a fake from walkie_vision.fakes instead, and a
model can be replaced without a change to the node that calls it.

Conventions for every provider:

- An image is an H x W x 3 uint8 numpy array in BGR order, as cv2 and
  cv_bridge return it.
- A box is (x1, y1, x2, y2) in pixels of that image, as a float array.
- embed_* methods return a float32 array of shape (n, dim): one row per
  input, in the order of the inputs. The rows need not have unit length:
  walkie_vision.messages normalises every vector that goes into a message.
- model_name is the <model> part of walkie-database's vector table names
  (for example arcface_w600k_r50 for face_arcface_w600k_r50). It names the
  weights, not only the network, so that vectors from two sets of weights
  are never compared. A model with several output lengths includes the
  length in the name.

Each name lives in its vision task's base module (object_detection.base,
pose_estimation.base, face_recognition.base, reid.base, image_embed.base) or
in vector_tables, and this package re-exports it for callers outside
providers/. A module inside providers/ imports the module that defines the
name, never this package.
"""

from .face_recognition.base import Face, FaceRecognizer, PersonFaceRecognizer
from .image_embed.base import Embedder
from .object_detection.base import Detection, ObjectDetector
from .pose_estimation.base import COCO_KEYPOINTS, PersonPose, PoseEstimator
from .reid.base import ReIdentifier
from .vector_tables import ARCFACE_W600K_R50, CLIP_VIT_B16, FAKE_MODEL_PREFIX, OSNET_X1_0_MSMT17

__all__ = [
    "ARCFACE_W600K_R50",
    "CLIP_VIT_B16",
    "COCO_KEYPOINTS",
    "FAKE_MODEL_PREFIX",
    "OSNET_X1_0_MSMT17",
    "Detection",
    "Embedder",
    "Face",
    "FaceRecognizer",
    "ObjectDetector",
    "PersonFaceRecognizer",
    "PersonPose",
    "PoseEstimator",
    "ReIdentifier",
]
