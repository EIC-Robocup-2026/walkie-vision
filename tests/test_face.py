import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

from walkie_vision.face.face_align import _similarity_transform, arcface_dst, estimate_norm, norm_crop


def test_estimate_norm_matches_scikit_image():
    # Expected matrices from scikit-image 0.26.0's SimilarityTransform, given
    # the same points as float64. InsightFace passes float32 points, which
    # scikit-image then solves in float32: up to 2e-7 relative difference.
    cases = [
        (
            [[950.1, 170.3], [1010.6, 168.2], [982.4, 205.9], [957.8, 238.1], [1004.9, 236.4]],
            {
                112: [[0.5966460438957117, -0.010183628182937879, -527.3038515654046],
                      [0.010183628182937873, 0.5966460438957116, -59.675517449849394]],
                128: [[0.5966460438957117, -0.010183628182937879, -519.3038515654046],
                      [0.010183628182937873, 0.5966460438957116, -59.675517449849394]],
            },
        ),
        (
            [[600.5, 300.25], [640.0, 310.0], [615.75, 345.5], [598.0, 380.0], [632.5, 388.0]],
            {
                112: [[0.5717784804183389, 0.06698224287368694, -320.05341344899637],
                      [-0.06698224287368695, 0.5717784804183389, -83.86836223615168]],
                128: [[0.571778480418339, 0.06698224287368695, -312.05341344899637],
                      [-0.06698224287368697, 0.571778480418339, -83.86836223615174]],
            },
        ),
    ]
    for lmk, expected in cases:
        lmk = np.array(lmk, dtype=np.float32)
        for size, m in expected.items():
            np.testing.assert_allclose(estimate_norm(lmk, size), m, rtol=1e-9, atol=1e-9)


def test_similarity_transform_recovers_a_known_transform():
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 500, size=(5, 2))
    angle, scale, shift = np.deg2rad(23.0), 1.7, np.array([-40.0, 12.5])
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    dst = scale * src @ rot.T + shift
    m = _similarity_transform(src, dst)
    np.testing.assert_allclose(m[:, :2], scale * rot, atol=1e-9)
    np.testing.assert_allclose(m[:, 2], shift, atol=1e-7)


def test_similarity_transform_rejects_degenerate_landmarks():
    with pytest.raises(ValueError):
        _similarity_transform(np.ones((5, 2)), np.random.default_rng(0).uniform(size=(5, 2)))


def test_norm_crop_puts_the_landmarks_on_the_template():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    lmk = np.array([[300, 200], [360, 198], [330, 235], [305, 265], [355, 263]], dtype=np.float32)
    crop = norm_crop(img, lmk)
    assert crop.shape == (112, 112, 3)
    moved = cv2.transform(lmk[None], estimate_norm(lmk))[0]
    # A similarity transform cannot fit five points exactly; within 3 pixels.
    assert np.abs(moved - arcface_dst).max() < 3.0


def _models_dir() -> Path | None:
    from walkie_vision.models import BUFFALO_L_DIR, models_dir

    try:
        path = models_dir() / BUFFALO_L_DIR
    except RuntimeError:
        return None
    return path if (path / "det_10g.onnx").is_file() else None


@pytest.mark.skipif(_models_dir() is None, reason="buffalo_l is missing; run `pixi run fetch-models`")
def test_face_analyzer_on_zidane_cpu():
    from walkie_vision.face import FaceAnalyzer
    from walkie_vision.providers import ARCFACE_W600K_R50, FaceRecognizer

    assets = Path(importlib.util.find_spec("ultralytics").origin).parent / "assets"
    img = cv2.imread(str(assets / "zidane.jpg"))
    analyzer = FaceAnalyzer(_models_dir(), providers=["CPUExecutionProvider"])
    assert isinstance(analyzer, FaceRecognizer)
    assert analyzer.model_name == ARCFACE_W600K_R50
    faces = sorted(analyzer.recognize(img), key=lambda f: -f.det_score)
    # InsightFace 2.1's FaceAnalysis finds the same two faces in this image:
    # [913, 96, 1057, 276] (score 0.871) and [545, 230, 664, 451] (0.682).
    assert len(faces) == 2
    np.testing.assert_allclose(faces[0].bbox, [913, 96, 1057, 276], atol=2)
    np.testing.assert_allclose(faces[1].bbox, [545, 230, 664, 451], atol=2)
    for f in faces:
        assert f.kps.shape == (5, 2)
        assert f.normed_embedding.shape == (512,)
        assert np.isclose(np.linalg.norm(f.normed_embedding), 1.0)
    # Two different people.
    assert float(faces[0].normed_embedding @ faces[1].normed_embedding) < 0.3
