"""Check 8: face detection and recognition (walkie_vision.face, buffalo_l) on the GPU.

The expected boxes are what the InsightFace 2.1 package finds on the CPU in
the same images. The CUDA embeddings are compared with the CPU ones.
"""

from __future__ import annotations

import statistics
import time

import cv2
import numpy as np
import onnxruntime as ort

from common import assets_dir, expect, finish, gpu_mem_of, mapped_libs
from walkie_vision.face import FaceAnalyzer
from walkie_vision.models import BUFFALO_L_DIR, models_dir

EXPECTED = {
    "zidane.jpg": [[913, 96, 1057, 276], [545, 230, 664, 451]],
    "bus.jpg": [[271, 420, 307, 472], [115, 420, 155, 470]],
}

model_dir = models_dir() / BUFFALO_L_DIR
print("onnxruntime", ort.__version__, "| models", model_dir)
t0 = time.perf_counter()
gpu = FaceAnalyzer(model_dir)
print(f"CUDA sessions created in {(time.perf_counter() - t0) * 1000:.0f} ms")
for label, sess in (("detector", gpu.detector.session), ("recognizer", gpu.recognizer.session)):
    expect(sess.get_providers()[0] == "CUDAExecutionProvider", f"{label} runs on CUDAExecutionProvider")
cpu = FaceAnalyzer(model_dir, providers=["CPUExecutionProvider"])

for name, want in EXPECTED.items():
    img = cv2.imread(str(assets_dir() / name))
    t0 = time.perf_counter()
    faces = gpu.get(img)
    first = (time.perf_counter() - t0) * 1000
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        faces = gpu.get(img)
        times.append((time.perf_counter() - t0) * 1000)
    print(f"{name}: {len(faces)} faces; first call {first:.0f} ms, then median {statistics.median(times):.1f} ms "
          "(detection and one embedding per face)")
    expect(len(faces) == len(want), f"{name}: {len(want)} faces found")
    ref = cpu.get(img)
    for box in want:
        f = min(faces, key=lambda f: np.abs(f.bbox - box).max())
        r = min(ref, key=lambda f: np.abs(f.bbox - box).max())
        cos = float(f.normed_embedding @ r.normed_embedding)
        print(f"  bbox {np.round(f.bbox).astype(int).tolist()} score {f.det_score:.3f}; "
              f"cosine with the CPU embedding {cos:.6f}")
        expect(np.abs(f.bbox - box).max() <= 2, f"{name}: a face at {box} (within 2 px)")
        expect(cos > 0.99, f"{name}: CUDA and CPU embeddings agree (cosine > 0.99)")

print("libcudnn mapped:", mapped_libs("libcudnn.so"))
print("GPU memory (nvidia-smi):", gpu_mem_of())
finish()
