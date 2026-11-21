"""Check 13: the Fast-FoundationStereo ONNX model (walkie_vision.stereo) on the GPU.

The input is bus.jpg resized to 1104 x 621, the size the stereo node runs at
on the ZED's HD2K images with its default scale of 0.5. The right image is
the same picture moved 12 px to the left, so the disparity is 12 px wherever
both images show the same point. On the robot's RTX 3080 Ti Laptop GPU
(2026-11-21) the interior median was 12.00 px, from 11.98 to 12.02 px at the
5th and 95th percentiles, with 819 ms per call after a first call of 1.6 s
and 3286 MiB of GPU memory. The check has not run on the RTX 2060 yet.
"""

from __future__ import annotations

import statistics
import time

import cv2
import numpy as np

from common import assets_dir, expect, finish, gpu_mem_of
from walkie_vision.models import FFS_ONNX, models_dir
from walkie_vision.stereo import centre_padding, load_model, model_input, unpad

SIZE = (1104, 621)
SHIFT_PX = 12
TOLERANCE_PX = 1.0
# The rows and columns left out of the median: the left edge that the right
# image does not show, the edge copies on the right, and the padding.
MARGIN_PX = 64

path = models_dir() / FFS_ONNX
t0 = time.perf_counter()
model = load_model(path)
print(f"{path.name}: session created in {time.perf_counter() - t0:.1f} s on {model.providers}")
expect(model.providers[0] == "CUDAExecutionProvider", "the stereo model runs on CUDAExecutionProvider")

left = cv2.resize(cv2.imread(str(assets_dir() / "bus.jpg")), SIZE, interpolation=cv2.INTER_AREA)
right = cv2.copyMakeBorder(left[:, SHIFT_PX:], 0, 0, 0, SHIFT_PX, cv2.BORDER_REPLICATE)
padding = centre_padding(*SIZE)
left_in, right_in = model_input(left, SIZE, padding), model_input(right, SIZE, padding)
print(f"input {left_in.shape[3]} x {left_in.shape[2]} after padding {padding}")

t0 = time.perf_counter()
disparity = model(left_in, right_in)
first = (time.perf_counter() - t0) * 1000
times = []
for _ in range(5):
    t0 = time.perf_counter()
    disparity = model(left_in, right_in)
    times.append((time.perf_counter() - t0) * 1000)
print(f"first call {first:.0f} ms, then median {statistics.median(times):.0f} ms of {len(times)} calls")

disparity = unpad(disparity, padding)
expect(disparity.shape == (SIZE[1], SIZE[0]), f"the disparity is {SIZE[0]} x {SIZE[1]} after unpadding")
expect(bool(np.isfinite(disparity).all()), "every disparity is finite")
interior = disparity[MARGIN_PX:-MARGIN_PX, MARGIN_PX:-MARGIN_PX]
median = float(np.median(interior))
print(f"interior disparity: median {median:.2f} px, 5th to 95th percentile "
      f"{np.percentile(interior, 5):.2f} to {np.percentile(interior, 95):.2f} px")
expect(abs(median - SHIFT_PX) <= TOLERANCE_PX, f"the interior median is within {TOLERANCE_PX} px of {SHIFT_PX} px")

print("GPU memory (nvidia-smi):", gpu_mem_of())
finish()
