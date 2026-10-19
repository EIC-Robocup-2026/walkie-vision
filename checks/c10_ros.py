"""Check 10: rclpy and the ML stack in one process, over Zenoh.

Usage: python c10_ros.py [ros-first|ml-first]
  ros-first  import rclpy, then torch, ultralytics, onnxruntime, transformers, cv2
  ml-first   the same imports in the reverse order

The check starts its own Zenoh router on localhost, on the port in
WALKIE_VISION_CHECK_ZENOH_PORT or a free port, and points every process it
starts at that router. So it needs no running router, and it does not join
the robot's ROS graph on the default port 7447. It then creates a node and
checks that a second process sees it (`ros2 node list --no-daemon`) and
receives its messages (`ros2 topic echo --no-daemon --once`). --no-daemon
keeps the ROS 2 CLI from leaving a daemon running.
"""

from __future__ import annotations

import importlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

order = sys.argv[1] if len(sys.argv) > 1 else "ros-first"
mods = ["rclpy", "torch", "ultralytics", "onnxruntime", "transformers", "cv2"]
if order == "ml-first":
    mods.reverse()
print("import order:", mods)
for m in mods:
    t0 = time.perf_counter()
    mod = importlib.import_module(m)
    print(f"  import {m:13s} {(time.perf_counter() - t0) * 1000:7.0f} ms  {getattr(mod, '__version__', '')}")

import cv2
import numpy as np
import rclpy
import torch
from rclpy.node import Node
from rclpy.serialization import deserialize_message, serialize_message
from rclpy.utilities import get_rmw_implementation_identifier
from sensor_msgs.msg import Image
from std_msgs.msg import String

from common import expect, finish, mapped_libs

prefix = Path(os.environ["CONDA_PREFIX"])
config_dir = prefix / "share" / "rmw_zenoh_cpp" / "config"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def write_config(src: Path, dst: Path, old: str, new: str) -> None:
    """Copy a default rmw_zenoh config with one endpoint replaced.

    The endpoint must appear exactly once outside comments, so that a change
    to the default file fails here instead of leaving port 7447 in place.
    """
    lines = src.read_text().splitlines(keepends=True)
    hits = [i for i, line in enumerate(lines) if old in line and not line.lstrip().startswith("//")]
    if len(hits) != 1:
        raise RuntimeError(f"{src.name}: expected {old} once outside comments, found it {len(hits)} times")
    lines[hits[0]] = lines[hits[0]].replace(old, new)
    dst.write_text("".join(lines))


def wait_for_port(port: int, router: subprocess.Popen, timeout: float = 15.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end and router.poll() is None:
        try:
            socket.create_connection(("localhost", port), timeout=0.2).close()
            return True
        except OSError:
            time.sleep(0.1)
    return False


port = int(os.environ.get("WALKIE_VISION_CHECK_ZENOH_PORT") or free_port())
expect(port != 7447, f"the check router uses its own port ({port}), not 7447")
# The generated configs and the router's log go next to the check logs.
tmp = Path(__file__).resolve().parent / "output" / f"c10_{order}_zenoh"
tmp.mkdir(parents=True, exist_ok=True)
router_config = tmp / "router.json5"
session_config = tmp / "session.json5"
write_config(config_dir / "DEFAULT_RMW_ZENOH_ROUTER_CONFIG.json5", router_config,
             '"tcp/[::]:7447"', f'"tcp/localhost:{port}"')
write_config(config_dir / "DEFAULT_RMW_ZENOH_SESSION_CONFIG.json5", session_config,
             '"tcp/localhost:7447"', f'"tcp/localhost:{port}"')
# Set before rclpy.init(): rmw_zenoh reads it when the context starts. The
# processes started below inherit it.
os.environ["ZENOH_SESSION_CONFIG_URI"] = str(session_config)
print("RMW_IMPLEMENTATION =", os.environ.get("RMW_IMPLEMENTATION"), "| ROS_DOMAIN_ID =", os.environ.get("ROS_DOMAIN_ID"))
print("ZENOH_SESSION_CONFIG_URI =", session_config)

router_log = open(tmp / "rmw_zenohd.log", "w")
router = subprocess.Popen(
    [str(prefix / "lib" / "rmw_zenoh_cpp" / "rmw_zenohd")],
    env={**os.environ, "ZENOH_ROUTER_CONFIG_URI": str(router_config)},
    stdout=router_log, stderr=subprocess.STDOUT,
)
print(f"router pid {router.pid}, listening on tcp/localhost:{port}")
node = None
try:
    if not expect(wait_for_port(port, router), "the check router accepts connections"):
        raise SystemExit(1)

    # GPU work in the same process, so that CUDA and the middleware are both live.
    x = torch.randn(1024, 1024, device="cuda")
    expect(bool(torch.isfinite(x @ x).all()), "torch runs on the GPU next to rclpy")

    rclpy.init()
    node = Node("walkie_vision_check")
    rmw = get_rmw_implementation_identifier()
    expect(rmw == "rmw_zenoh_cpp", f"rclpy uses rmw_zenoh_cpp ({rmw})")
    libs = mapped_libs("librmw_zenoh_cpp", "libzenohc")
    print("mapped Zenoh libraries:", libs)
    # Which conda package the loaded librmw_zenoh_cpp.so belongs to. In
    # RoboStack, ros-jazzy-rmw-zenoh-cpp is a metapackage, and the library
    # comes from another package (ros2-rmw-zenoh-cpp), so search them all.
    owners = []
    for lib in [p for p in libs if "librmw_zenoh_cpp" in p]:
        rel = str(Path(lib).resolve().relative_to(prefix.resolve()))
        for meta in (prefix / "conda-meta").glob("*rmw-zenoh*.json"):
            data = json.loads(meta.read_text())
            if rel in data.get("files", []):
                owners.append(data)
                print(f"  {rel} belongs to {data['name']}=={data['version']} ({data['build']})")
    expect(len(owners) == 1, f"one conda package installed the loaded librmw_zenoh_cpp ({len(owners)} did)")
    for data in owners:
        major_minor_patch = tuple(int(p) for p in data["version"].split(".")[:3])
        expect(major_minor_patch >= (0, 2, 11), f"rmw_zenoh_cpp is 0.2.11 or newer ({data['version']})")

    out = subprocess.run(["ros2", "node", "list", "--no-daemon"], capture_output=True, text=True, timeout=120)
    # Zenoh writes its log lines to stdout too: with the default configs of
    # rmw_zenoh_cpp 0.2.11 it warns about two deprecated settings. Node names
    # are the lines that start with a slash.
    nodes = [line.strip() for line in out.stdout.splitlines() if line.startswith("/")]
    other = [line for line in out.stdout.splitlines() if line.strip() and not line.startswith("/")]
    print("ros2 node list:", nodes, f"| {len(other)} other lines on stdout | stderr:", out.stderr.strip()[-300:])
    expect("/walkie_vision_check" in nodes, "a second process sees the node")

    topic = "/walkie_vision_check/chatter"
    pub = node.create_publisher(String, topic, 10)
    echo = subprocess.Popen(
        ["ros2", "topic", "echo", "--no-daemon", "--once", topic, "std_msgs/msg/String"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    stop = threading.Event()

    def publish() -> None:
        i = 0
        while not stop.is_set():
            pub.publish(String(data=f"walkie-vision check {i}"))
            i += 1
            time.sleep(0.1)

    t0 = time.monotonic()
    publisher = threading.Thread(target=publish, daemon=True)
    publisher.start()
    try:
        stdout, stderr = echo.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        echo.kill()
        stdout, stderr = echo.communicate()
    stop.set()
    publisher.join()
    received = [line.strip() for line in stdout.splitlines() if line.startswith("data:")]
    print(f"ros2 topic echo exited {echo.returncode} after {time.monotonic() - t0:.1f} s: {received}")
    expect(echo.returncode == 0 and any("walkie-vision check" in r for r in received),
           "a second process receives the messages")

    # A camera frame through serialisation, without cv_bridge.
    h, w = 720, 1280
    src = np.random.default_rng(0).integers(0, 255, (h, w, 4), dtype=np.uint8)
    img = Image(height=h, width=w, encoding="bgra8", is_bigendian=0, step=w * 4)
    img.data = src.tobytes()
    ser = serialize_message(img)
    t1 = time.perf_counter()
    rx = deserialize_message(ser, Image)
    t_deser = (time.perf_counter() - t1) * 1000

    def image_to_numpy(m: Image) -> np.ndarray:
        """A view of a bgra8 sensor_msgs/Image's data, without a copy."""
        rows = np.frombuffer(m.data, dtype=np.uint8).reshape(m.height, m.step)  # honours row padding
        return rows[:, : m.width * 4].reshape(m.height, m.width, 4)

    view = image_to_numpy(rx)
    expect(bool(np.array_equal(view, src)) and not view.flags.owndata, "an Image message maps to numpy without a copy")
    t1 = time.perf_counter()
    for _ in range(50):
        gpu = torch.from_numpy(cv2.cvtColor(image_to_numpy(rx), cv2.COLOR_BGRA2RGB)).cuda()
    torch.cuda.synchronize()
    t_gpu = (time.perf_counter() - t1) * 1000 / 50
    print(f"Image bgra8 {w}x{h} ({len(ser) / 1e6:.1f} MB): deserialise {t_deser:.2f} ms, "
          f"to RGB and onto the GPU {t_gpu:.2f} ms")
finally:
    if node is not None:
        node.destroy_node()
    rclpy.try_shutdown()
    router.terminate()
    try:
        router.wait(timeout=10)
    except subprocess.TimeoutExpired:
        router.kill()
        router.wait()
    router_log.close()
    print(f"router stopped (exit code {router.returncode}); its log: {tmp / 'rmw_zenohd.log'}")

finish()
