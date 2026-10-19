# Vendored from InsightFace, python-package/insightface/utils/face_align.py,
# at commit 668e8d474267bdd2c9769c2cbbab3fe4d1db15a6 (python-package v0.7.3):
# https://github.com/deepinsight/insightface/blob/668e8d474267bdd2c9769c2cbbab3fe4d1db15a6/python-package/insightface/utils/face_align.py
#
# MIT License
#
# Copyright (c) 2022 Jiankang Deng and Jia Guo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Changes from the original:
# - Only arcface_dst, estimate_norm and norm_crop are kept.
# - skimage.transform.SimilarityTransform is replaced by _similarity_transform
#   below, so that scikit-image is not needed. Both implement Umeyama's
#   least-squares method; tests/test_face.py compares them.
# - The unused `mode` argument is removed.
"""ArcFace face alignment: warp a face to the 112x112 template from 5 landmarks."""

from __future__ import annotations

import cv2
import numpy as np

# Landmark positions in a 112x112 ArcFace crop: left eye, right eye, nose tip,
# left and right mouth corners.
arcface_dst = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)


def _similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Return the 2x3 similarity transform that maps src onto dst.

    A similarity transform is a rotation, a uniform scale and a translation.
    This is the least-squares solution of S. Umeyama, "Least-squares
    estimation of transformation parameters between two point patterns",
    IEEE TPAMI 13(4), 1991; the equation numbers below are the paper's.

    Raises ValueError when the source points are degenerate (all equal).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n, dim = src.shape
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    # Covariance of the centred point sets (38).
    cov = dst_c.T @ src_c / n
    u, sv, vt = np.linalg.svd(cov)
    rank = np.linalg.matrix_rank(cov)
    if rank < dim - 1:
        raise ValueError("cannot align: the landmarks are degenerate")
    # Pick the sign that gives a rotation, not a reflection (39), (43).
    d = np.ones(dim)
    if rank == dim:
        if np.linalg.det(cov) < 0:
            d[-1] = -1.0
    elif np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[-1] = -1.0
    rot = u @ np.diag(d) @ vt  # (40)
    # Variance of the source points (36) and the scale (42).
    src_var = (src_c**2).sum() / n
    scale = float(sv @ d) / src_var
    trans = dst_mean - scale * rot @ src_mean  # (41)
    m = np.empty((dim, dim + 1))
    m[:, :dim] = scale * rot
    m[:, dim] = trans
    return m


def estimate_norm(lmk, image_size=112):
    assert lmk.shape == (5, 2)
    assert image_size%112==0 or image_size%128==0
    if image_size%112==0:
        ratio = float(image_size)/112.0
        diff_x = 0
    else:
        ratio = float(image_size)/128.0
        diff_x = 8.0*ratio
    dst = arcface_dst * ratio
    dst[:,0] += diff_x
    M = _similarity_transform(lmk, dst)
    return M


def norm_crop(img, landmark, image_size=112):
    M = estimate_norm(landmark, image_size)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped
