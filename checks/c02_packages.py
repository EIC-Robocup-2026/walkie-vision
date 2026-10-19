"""Check 2: the Python environment is self-contained and no package overwrote another.

conda packages and PyPI wheels are installed into one site-packages. Two
distributions that install the same file (for example onnxruntime and
onnxruntime-gpu, which both install the `onnxruntime` module) leave whichever
was installed last, so this check looks for files that more than one
distribution claims with different contents.
"""

from __future__ import annotations

import collections
import importlib.metadata as md
import os
import sys
from pathlib import Path

from common import expect, finish, foreign_paths, mapped_libs

prefix = Path(os.environ.get("CONDA_PREFIX", sys.prefix)).resolve()
repo = Path(__file__).resolve().parents[1]
print("python", sys.version.split()[0], sys.executable)
for var in ("PYTHONPATH", "AMENT_PREFIX_PATH", "LD_LIBRARY_PATH", "CUDA_CACHE_MAXSIZE", "CUDA_CACHE_PATH"):
    print(f"{var} = {os.environ.get(var)}")

expect(os.environ.get("RMW_IMPLEMENTATION") == "rmw_zenoh_cpp", "RMW_IMPLEMENTATION is rmw_zenoh_cpp")
expect(os.environ.get("YOLO_AUTOINSTALL") == "false", "YOLO_AUTOINSTALL is false")

# sys.path entries outside the environment would mean leakage from a system
# ROS install or the user site. foreign_paths() lists what is expected.
foreign = foreign_paths(sys.path, prefix, repo)
expect(not foreign, f"sys.path has no entries outside the environment: {foreign or 'none'}")

# Files that more than one distribution installed. Identical copies, such as
# the empty nvidia/__init__.py that every NVIDIA wheel ships, are harmless.
owners: dict[Path, list[tuple[str, str | None]]] = collections.defaultdict(list)
count = 0
for dist in md.distributions():
    count += 1
    # dist.metadata parses the METADATA file on every access, so read it once.
    label = f"{dist.metadata['Name']}=={dist.version}"
    for f in dist.files or []:
        if "__pycache__" in f.parts or f.parts[0].endswith((".dist-info", ".egg-info")):
            continue
        digest = f.hash.value if f.hash else None
        owners[Path(dist.locate_file(f)).resolve()].append((label, digest))
shared = {p: o for p, o in owners.items() if len({name for name, _ in o}) > 1}
clashes = {str(p): o for p, o in shared.items() if None in {d for _, d in o} or len({d for _, d in o}) > 1}
print(f"{count} distributions, {len(owners)} files, {len(shared)} installed identically by more than one")
for path, who in list(clashes.items())[:20]:
    print(f"  {path}: {[name for name, _ in who]}")
expect(not clashes, f"no two distributions install different copies of one file ({len(clashes)} do)")


def version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


for name in ["numpy", "opencv-python", "protobuf", "onnxruntime-gpu", "onnx", "setuptools", "pillow",
             "huggingface-hub", "transformers", "sentence-transformers", "torch", "torchvision",
             "ultralytics", "open-clip-torch", "timm", "scipy", "lap", "clip-anytorch", "walkie-vision"]:
    v = version(name)
    installer = (md.distribution(name).read_text("INSTALLER") or "?").strip() if v else "-"
    print(f"  {name:24s} {v or 'not installed':16s} installer={installer}")

# Packages that must not be here: the CPU-only onnxruntime would replace the
# CUDA provider, and InsightFace (which needs it) is vendored instead.
for name in ("onnxruntime", "insightface", "scikit-image", "opencv-python-headless", "opencv-contrib-python"):
    expect(version(name) is None, f"{name} is not installed")
expect((version("onnxruntime-gpu") or "").startswith("1.26."), f"onnxruntime-gpu is 1.26.x ({version('onnxruntime-gpu')})")
expect(version("onnx") is not None, f"onnx is installed ({version('onnx')})")

import cv2
import numpy

print("numpy", numpy.__version__, numpy.__file__)
print("cv2", cv2.__version__, cv2.__file__)
expect(Path(cv2.__file__).resolve().is_relative_to(prefix), "cv2 comes from the environment")
libstdcxx = mapped_libs("libstdc++")
print("libstdc++ mapped:", libstdcxx)
expect(all(Path(p).resolve().is_relative_to(prefix) for p in libstdcxx), "libstdc++ comes from the environment")

finish()
