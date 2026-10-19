# Vendored from InsightFace, python-package/insightface/model_zoo/arcface_onnx.py,
# at commit 668e8d474267bdd2c9769c2cbbab3fe4d1db15a6 (python-package v0.7.3):
# https://github.com/deepinsight/insightface/blob/668e8d474267bdd2c9769c2cbbab3fe4d1db15a6/python-package/insightface/model_zoo/arcface_onnx.py
#
# Original header:
# @Organization  : insightface.ai
# @Author        : Jia Guo
# @Time          : 2021-05-04
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
# - The caller creates the ONNX Runtime session and passes it in, for the same
#   reason as in scrfd.py.
# - prepare() is removed (it only switched the session to the CPU provider).
# - Commented-out code is removed, and a comment on the normalisation check is
#   added.
"""ArcFace face embeddings (w600k_r50.onnx in buffalo_l) on an ONNX Runtime session."""

from __future__ import division

import cv2
import numpy as np
import onnx

from . import face_align

__all__ = [
    'ArcFaceONNX',
]


class ArcFaceONNX:
    def __init__(self, model_file=None, session=None):
        assert model_file is not None
        self.model_file = model_file
        self.session = session
        if self.session is None:
            raise ValueError('ArcFaceONNX needs an onnxruntime.InferenceSession with explicit providers')
        self.taskname = 'recognition'
        # Models converted from MXNet normalise inside the graph (a Sub and a
        # Mul node near the input); the others expect (x - 127.5) / 127.5.
        find_sub = False
        find_mul = False
        model = onnx.load(self.model_file)
        graph = model.graph
        for nid, node in enumerate(graph.node[:8]):
            if node.name.startswith('Sub') or node.name.startswith('_minus'):
                find_sub = True
            if node.name.startswith('Mul') or node.name.startswith('_mul'):
                find_mul = True
        if find_sub and find_mul:
            #mxnet arcface model
            input_mean = 0.0
            input_std = 1.0
        else:
            input_mean = 127.5
            input_std = 127.5
        self.input_mean = input_mean
        self.input_std = input_std
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        input_name = input_cfg.name
        self.input_size = tuple(input_shape[2:4][::-1])
        self.input_shape = input_shape
        outputs = self.session.get_outputs()
        output_names = []
        for out in outputs:
            output_names.append(out.name)
        self.input_name = input_name
        self.output_names = output_names
        assert len(self.output_names)==1
        self.output_shape = outputs[0].shape

    def get(self, img, face):
        aimg = face_align.norm_crop(img, landmark=face.kps, image_size=self.input_size[0])
        face.embedding = self.get_feat(aimg).flatten()
        return face.embedding

    def compute_sim(self, feat1, feat2):
        from numpy.linalg import norm
        feat1 = feat1.ravel()
        feat2 = feat2.ravel()
        sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
        return sim

    def get_feat(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        input_size = self.input_size

        blob = cv2.dnn.blobFromImages(imgs, 1.0 / self.input_std, input_size,
                                      (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
        return net_out

    def forward(self, batch_data):
        blob = (batch_data - self.input_mean) / self.input_std
        net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
        return net_out
