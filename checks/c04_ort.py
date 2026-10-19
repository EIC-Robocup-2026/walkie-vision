"""Check 4: onnxruntime-gpu runs a session on CUDAExecutionProvider.

Usage: python c04_ort.py [preload|torch-first]
  preload      call onnxruntime.preload_dlls() first, as walkie_vision.face does
  torch-first  import torch first, as in a process that also runs PyTorch models

Without either, ONNX Runtime does not find the CUDA and cuDNN libraries and
falls back to the CPU provider without raising (see pixi.toml).

On a GPU that ONNX Runtime has no machine code for (an RTX 50-series GPU),
the first session run includes the driver's PTX compilation; the timings and
the size of the driver's compute cache show it.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

mode = sys.argv[1] if len(sys.argv) > 1 else "preload"
if mode == "torch-first":
    import torch  # noqa: F401

import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

from common import expect, finish, gpu_mem_of, mapped_libs

cache = Path(os.environ.get("CUDA_CACHE_PATH") or Path.home() / ".nv" / "ComputeCache")


def cache_mib() -> float:
    if not cache.is_dir():
        return 0.0
    return sum(f.stat().st_size for f in cache.rglob("*") if f.is_file()) / 2**20


print("mode", mode)
print("onnxruntime", ort.__version__, ort.__file__)
from onnxruntime.capi import build_and_package_info

print("built with CUDA", build_and_package_info.cuda_version)
print("available providers", ort.get_available_providers())
for var in ("CUDA_CACHE_DISABLE", "CUDA_CACHE_MAXSIZE", "CUDA_CACHE_PATH", "CUDA_FORCE_PTX_JIT", "CUDA_MODULE_LOADING"):
    print(f"{var} = {os.environ.get(var)}")
print(f"driver compute cache {cache}: {cache_mib():.0f} MiB before")
if mode == "preload":
    ort.preload_dlls()
    print("after preload_dlls, libcudnn mapped:", mapped_libs("libcudnn.so"))

# A small Conv + Relu graph: Conv on the CUDA provider goes through cuDNN.
w = np.random.default_rng(0).standard_normal((16, 3, 3, 3)).astype(np.float32)
graph = helper.make_graph(
    [helper.make_node("Conv", ["x", "w"], ["c"], pads=[1, 1, 1, 1]), helper.make_node("Relu", ["c"], ["y"])],
    "conv_relu",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 224, 224])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 16, 224, 224])],
    initializer=[helper.make_tensor("w", TensorProto.FLOAT, w.shape, w.flatten())],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 9
onnx.checker.check_model(model)

so = ort.SessionOptions()
so.log_severity_level = 3
t0 = time.perf_counter()
sess = ort.InferenceSession(
    model.SerializeToString(), sess_options=so,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
t_create = (time.perf_counter() - t0) * 1000
print("session providers", sess.get_providers())
expect(sess.get_providers()[0] == "CUDAExecutionProvider", "the session runs on CUDAExecutionProvider")
x = np.random.default_rng(1).standard_normal((1, 3, 224, 224)).astype(np.float32)
t0 = time.perf_counter()
y = sess.run(None, {"x": x})[0]
t_first = (time.perf_counter() - t0) * 1000
t0 = time.perf_counter()
for _ in range(20):
    y = sess.run(None, {"x": x})[0]
t_run = (time.perf_counter() - t0) * 1000 / 20
print(f"session create {t_create:.0f} ms, first run {t_first:.0f} ms, then {t_run:.2f} ms per run")
cpu = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
diff = float(np.abs(cpu - y).max())
print("max abs diff CUDA vs CPU", diff)
expect(diff < 1e-3, "CUDA and CPU results agree within 1e-3")
print(f"driver compute cache: {cache_mib():.0f} MiB after")
print("libcudnn mapped:", mapped_libs("libcudnn.so"))
print("libcudart / libcublas mapped:", mapped_libs("libcudart", "libcublas.so"))
print("GPU memory (nvidia-smi):", gpu_mem_of())

finish()
