"""Run the environment checks, each in its own process, and summarise them.

Usage: python checks/run_checks.py [prefix ...]
  prefix  run only the checks whose name starts with it, for example c03 or c05_pose

Each check writes its full output to checks/output/<name>.log. This script
prints one line per check, writes the same lines and the computer's GPU and
driver to checks/output/summary.txt, and exits with status 1 if any check
failed. The first run downloads the Hugging Face models into the Hugging Face
cache.
"""

from __future__ import annotations

import datetime
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
# On an RTX 50-series GPU the first ONNX Runtime run includes the driver's PTX
# compilation, which can take many minutes (see onnxruntime-gpu in pixi.toml).
TIMEOUT_S = 3600

CHECKS = [
    ("c01_fatbins", ["c01_fatbins.py"]),
    ("c02_packages", ["c02_packages.py"]),
    ("c03_torch", ["c03_torch.py"]),
    ("c04_ort_preload", ["c04_ort.py", "preload"]),
    ("c04_ort_torch_first", ["c04_ort.py", "torch-first"]),
    ("c05_yoloe", ["c05_ultralytics.py", "yoloe"]),
    ("c05_pose", ["c05_ultralytics.py", "pose"]),
    ("c05_track", ["c05_ultralytics.py", "track"]),
    ("c06_siglip2", ["c06_transformers.py", "siglip2"]),
    ("c06_florence2", ["c06_transformers.py", "florence2"]),
    ("c06_embeddinggemma2", ["c06_transformers.py", "embeddinggemma2"]),
    ("c07_clip", ["c07_clip.py"]),
    ("c08_face", ["c08_face.py"]),
    ("c09_osnet", ["c09_osnet.py"]),
    ("c10_ros_first", ["c10_ros.py", "ros-first"]),
    ("c10_ml_first", ["c10_ros.py", "ml-first"]),
    ("c11_providers", ["c11_providers.py"]),
    ("c12_vision_node", ["c12_vision_node.py"]),
]


def system_info() -> list[str]:
    lines = [
        f"date: {datetime.datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"host: {platform.node()}, kernel {platform.release()}",
    ]
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, check=False,
    )
    gpus = smi.stdout.strip().splitlines()
    lines += [f"GPU: {g}" for g in gpus] or ["GPU: nvidia-smi lists none"]
    return lines


def run(args: list[str], log: Path) -> int | str:
    """Run one check with its output in log; return its exit code or "timeout"."""
    with open(log, "w") as f:
        f.write(f"$ python {' '.join(args)}\n")
        f.flush()
        # Its own process group, so that a timeout or Ctrl-C also stops the
        # processes it started (the Zenoh router in c10).
        proc = subprocess.Popen(
            [sys.executable, str(HERE / args[0]), *args[1:]],
            cwd=HERE, stdout=f, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            return proc.wait(timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            return "timeout"
        except KeyboardInterrupt:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait()
            raise


def main(argv: list[str]) -> int:
    selected = [c for c in CHECKS if not argv or any(c[0].startswith(p) for p in argv)]
    if not selected:
        print(f"no check matches {argv}; the checks are: {' '.join(n for n, _ in CHECKS)}")
        return 2
    OUTPUT.mkdir(exist_ok=True)
    header = system_info()
    print(*header, sep="\n")
    lines = []
    failed = []
    for name, args in selected:
        print(f"{name:22s} ", end="", flush=True)
        log = OUTPUT / f"{name}.log"
        t0 = time.monotonic()
        code = run(args, log)
        status = "PASS" if code == 0 else "FAIL"
        line = f"{status}  {time.monotonic() - t0:6.1f} s  exit {code}  {log.relative_to(HERE.parent)}"
        print(line, flush=True)
        lines.append(f"{name:22s} {line}")
        if code != 0:
            failed.append(name)
    verdict = f"{len(selected) - len(failed)} of {len(selected)} checks passed"
    if failed:
        verdict += f"; failed: {' '.join(failed)}"
    print(verdict)
    (OUTPUT / "summary.txt").write_text("\n".join(header + lines + [verdict]) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
