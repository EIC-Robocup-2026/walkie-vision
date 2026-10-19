"""Shared helpers for the environment checks in this directory."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

_failures: list[str] = []


def expect(ok: bool, what: str) -> bool:
    """Print one result and remember a failure for finish()."""
    print(f"{'PASS' if ok else 'FAIL'}  {what}", flush=True)
    if not ok:
        _failures.append(what)
    return ok


def finish() -> None:
    """Exit with status 1 if any expect() failed."""
    if _failures:
        print(f"{len(_failures)} failed:", *_failures, sep="\n  ")
        sys.exit(1)
    print("all passed")


def gpu_mem_of(pid: int | None = None) -> str:
    """Return the GPU memory nvidia-smi reports for one process."""
    pid = pid or os.getpid()
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
        capture_output=True, text=True, check=False,
    ).stdout
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if parts and parts[0] == str(pid):
            return parts[1]
    return "not listed"


def mapped_libs(*needles: str) -> list[str]:
    """Return the shared objects mapped into this process whose path matches."""
    paths = set()
    with open("/proc/self/maps") as f:
        for line in f:
            fields = line.split()
            if len(fields) >= 6 and fields[5].startswith("/"):
                if any(n in fields[5] for n in needles):
                    paths.add(fields[5])
    return sorted(paths)


def assets_dir() -> Path:
    """Return the sample images that ship with Ultralytics (bus.jpg, zidane.jpg)."""
    import importlib.util

    spec = importlib.util.find_spec("ultralytics")
    return Path(spec.origin).parent / "assets"


def foreign_paths(paths: list[str], prefix: Path, repo: Path) -> list[str]:
    """Return the sys.path entries that come from neither the environment nor this repository.

    Allowed are the environment (prefix), the editable install of src/, the
    checks directory, and the message package that `pixi run build-msgs`
    installs under ros2_ws/install, which scripts/activate.sh adds to
    PYTHONPATH. Anything else is leakage, for example from a system ROS
    install or the user site.
    """
    allowed = {repo / "src", repo / "checks"}
    overlay = repo / "ros2_ws" / "install"
    foreign = []
    for p in paths:
        if not p or p.endswith(".zip"):
            continue
        path = Path(p).resolve()
        if not (path.is_relative_to(prefix) or path in allowed or path.is_relative_to(overlay)):
            foreign.append(p)
    return foreign


def timed_ms(fn, *args, **kwargs):
    """Call fn once; return (result, wall time in ms), after a CUDA sync if torch is loaded."""
    torch = sys.modules.get("torch")
    sync = torch.cuda.synchronize if torch is not None and torch.cuda.is_initialized() else (lambda: None)
    sync()
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    sync()
    return out, (time.perf_counter() - t0) * 1000.0
