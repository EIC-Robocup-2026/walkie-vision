"""Check 1: which GPU architectures torch, torchvision and onnxruntime-gpu can run on.

A CUDA library carries machine code (SASS) for some GPU architectures and,
optionally, PTX. SASS built for sm_XY runs on a GPU of the same major version
whose minor version is Y or higher (sm_70 code runs on an sm_75 GPU). SASS
with an "a" suffix (sm_90a) runs only on that exact architecture. PTX for
compute_XY is compiled by the driver, on first use, for any GPU of XY or
newer; that works, but the first run is slow.

This check lists the architectures with cuobjdump and works out, for the
RTX 2060 (sm_75), an RTX 50-series GPU (sm_120) and every GPU in this
computer, whether each library has SASS, needs PTX JIT, or cannot run.
It does not need a GPU.
"""

from __future__ import annotations

import collections
import importlib.util
import re
import shutil
import subprocess
from pathlib import Path

from common import expect, finish

ENTRY = re.compile(r"^(ELF|PTX) file\s+\d+: .*\.sm_(\d+)([af]?)\.(?:cubin|ptx)$")


def package_dir(name: str) -> Path:
    return Path(importlib.util.find_spec(name).origin).parent


def arches(path: Path) -> tuple[collections.Counter, collections.Counter]:
    """Return the SASS and PTX architectures in a library, with entry counts."""
    sass, ptx = collections.Counter(), collections.Counter()
    for flag, counter in (("--list-elf", sass), ("--list-ptx", ptx)):
        out = subprocess.run(["cuobjdump", flag, str(path)], capture_output=True, text=True, check=False)
        for line in out.stdout.splitlines():
            m = ENTRY.match(line.strip())
            if m:
                counter[(int(m.group(2)), m.group(3))] += 1
    return sass, ptx


def split(arch: int) -> tuple[int, int]:
    """sm_120 -> (12, 0); sm_75 -> (7, 5)."""
    return divmod(arch, 10)


def runs_on(arch: int, suffix: str, cap: tuple[int, int], is_ptx: bool) -> bool:
    """Whether code for sm_<arch><suffix> runs on a GPU of compute capability cap."""
    major, minor = split(arch)
    if suffix == "a":  # architecture-specific: that GPU only
        return (major, minor) == cap
    if suffix == "f" or not is_ptx:  # family-specific code, or plain SASS: same major
        return major == cap[0] and minor <= cap[1]
    return (major, minor) <= cap  # plain PTX: any newer GPU


def support(sass, ptx, cap: tuple[int, int]) -> str:
    """Return "SASS", "PTX JIT" or "none" for a GPU of compute capability cap."""
    if any(runs_on(arch, suffix, cap, is_ptx=False) for arch, suffix in sass):
        return "SASS"
    if any(runs_on(arch, suffix, cap, is_ptx=True) for arch, suffix in ptx):
        return "PTX JIT"
    return "none"


def fmt(counter, prefix: str) -> str:
    return " ".join(f"{prefix}{a}{s}({n})" for (a, s), n in sorted(counter.items())) or "none"


def local_gpus() -> list[tuple[str, tuple[int, int]]]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
        capture_output=True, text=True, check=False,
    ).stdout
    gpus = []
    for line in out.splitlines():
        name, cap = (p.strip() for p in line.rsplit(",", 1))
        major, minor = cap.split(".")
        gpus.append((name, (int(major), int(minor))))
    return gpus


expect(shutil.which("cuobjdump") is not None, "cuobjdump is on PATH")
print("cuobjdump:", shutil.which("cuobjdump"))

torch_lib = package_dir("torch") / "lib"
libraries = [
    # (label, path, what an RTX 50-series GPU may use: SASS only, or PTX JIT too)
    ("torch", torch_lib / "libtorch_cuda.so", {"SASS"}),
    ("torch linalg", torch_lib / "libtorch_cuda_linalg.so", {"SASS"}),
    ("torchvision", package_dir("torchvision") / "_C.so", {"SASS"}),
    ("onnxruntime-gpu", package_dir("onnxruntime") / "capi" / "libonnxruntime_providers_cuda.so", {"SASS", "PTX JIT"}),
]
targets = [("RTX 2060", (7, 5)), ("RTX 50-series", (12, 0))]
gpus = local_gpus()
print("GPUs in this computer:", gpus or "none")

for label, path, sm120_ok in libraries:
    expect(path.is_file(), f"{label}: {path.name} exists")
    sass, ptx = arches(path)
    print(f"{label}: {path}")
    print(f"  SASS: {fmt(sass, 'sm_')}")
    print(f"  PTX:  {fmt(ptx, 'compute_')}")
    for name, cap in targets:
        how = support(sass, ptx, cap)
        allowed = {"SASS"} if cap == (7, 5) else sm120_ok
        expect(how in allowed, f"{label} on {name} (sm_{cap[0]}{cap[1]}): {how}")
    for name, cap in gpus:
        how = support(sass, ptx, cap)
        expect(how != "none", f"{label} on this computer's {name} (sm_{cap[0]}{cap[1]}): {how}")

finish()
