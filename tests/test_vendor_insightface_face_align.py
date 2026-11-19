"""Tests of walkie_vision._vendor.insightface.face_align."""

import cv2
import numpy as np
import pytest

from walkie_vision._vendor.insightface.face_align import _similarity_transform, arcface_dst, estimate_norm, norm_crop


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
