"""Check 3: torch and torchvision on the GPU.

torch's arch list, a matmul in fp32 and fp16, and torchvision.ops.nms on CUDA,
which fails with "no kernel image is available" when torchvision has no code
for the GPU.
"""

from __future__ import annotations

import time

import torch
import torchvision

from common import expect, finish, gpu_mem_of, mapped_libs

print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| cuDNN", torch.backends.cudnn.version())
print("torchvision", torchvision.__version__)
expect(torch.cuda.is_available(), "torch sees a CUDA GPU")
arch = torch.cuda.get_arch_list()
print("arch list", arch)
expect({"sm_75", "sm_120"} <= set(arch), "torch was built for sm_75 and sm_120")
cap = torch.cuda.get_device_capability(0)
print("device", torch.cuda.get_device_name(0), "capability", cap,
      f"memory {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
expect(f"sm_{cap[0]}{cap[1]}" in arch, f"torch was built for this GPU (sm_{cap[0]}{cap[1]})")
bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
print("bf16 in hardware:", bf16, "(Turing has none; use fp16 or fp32 there)")

torch.zeros(1, device="cuda")
torch.cuda.synchronize()
time.sleep(0.5)
print("GPU memory, CUDA context only (nvidia-smi):", gpu_mem_of())

a = torch.randn(4096, 4096, device="cuda")
b = torch.randn(4096, 4096, device="cuda")
ref = a[:64].double() @ b[:, :64].double()
for dtype, tol in ((torch.float32, 0.05), (torch.float16, 1.0)):
    x, y = a.to(dtype), b.to(dtype)
    c = x @ y
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        c = x @ y
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 10
    err = (c[:64, :64].double() - ref).abs().max().item()
    print(f"matmul 4096x4096 {dtype}: {dt * 1000:.1f} ms ({2 * 4096**3 / dt / 1e12:.1f} TFLOPS), max abs error vs fp64 {err:.3g}")
    expect(err < tol, f"matmul {dtype} matches fp64 within {tol}")
del a, b, c, x, y

# Overlapping random boxes, so that NMS has work to do.
g = torch.Generator().manual_seed(0)
xy = torch.rand(2000, 2, generator=g) * 600
wh = torch.rand(2000, 2, generator=g) * 100 + 10
boxes = torch.cat([xy, xy + wh], dim=1)
scores = torch.rand(2000, generator=g)
keep_cpu = torchvision.ops.nms(boxes, scores, 0.5)
try:
    keep_gpu = torchvision.ops.nms(boxes.cuda(), scores.cuda(), 0.5)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        keep_gpu = torchvision.ops.nms(boxes.cuda(), scores.cuda(), 0.5)
    torch.cuda.synchronize()
    print(f"torchvision.ops.nms on CUDA: {len(keep_gpu)} of 2000 boxes kept, {(time.perf_counter() - t0) * 50:.2f} ms")
    expect(torch.equal(keep_gpu.cpu(), keep_cpu), "torchvision.ops.nms gives the same boxes on CUDA as on the CPU")
except RuntimeError as e:
    expect(False, f"torchvision.ops.nms runs on CUDA: {e}")

print("GPU memory at the end (nvidia-smi):", gpu_mem_of(),
      f"| torch max allocated {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
print("libcudnn mapped:", mapped_libs("libcudnn.so"))
# torchvision's wheel carries its own copy of libcudart next to torch's.
print("libcudart mapped:", mapped_libs("libcudart"))

finish()
