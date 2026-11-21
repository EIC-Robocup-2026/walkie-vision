"""walkie_vision.stereo_depth: the config, the pairing, the outputs, the report and the exit codes, with a fake model."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import rclpy
import yaml
from builtin_interfaces.msg import Time
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header

from walkie_vision import models
from walkie_vision.node import MAX_PAIR_WINDOW_S
from walkie_vision.stereo_depth import (
    AGE_WARN_S,
    INPUT_QOS,
    StereoConfig,
    StereoDepthNode,
    StereoPairer,
    report_lines,
)

REPO = Path(__file__).resolve().parents[1]
TIMEOUT_S = 20.0
WIDTH, HEIGHT = 64, 48
LEFT_FRAME = "zed_head_left_camera_frame_optical"
RIGHT_FRAME = "zed_head_right_camera_frame_optical"
K = [50.0, 0.0, 32.0, 0.0, 50.0, 24.0, 0.0, 0.0, 1.0]
SCALED_K = [25.0, 0.0, 15.75, 0.0, 25.0, 11.75, 0.0, 0.0, 1.0]
LEFT_P = [50.0, 0.0, 32.0, 0.0, 0.0, 50.0, 24.0, 0.0, 0.0, 0.0, 1.0, 0.0]
RIGHT_P = [50.0, 0.0, 32.0, -5.0, 0.0, 50.0, 24.0, 0.0, 0.0, 0.0, 1.0, 0.0]  # a baseline of 0.1 m
INPUTS = ("left_image_topic", "left_camera_info_topic", "right_image_topic", "right_camera_info_topic")
OUTPUTS = ("image_topic", "camera_info_topic", "depth_topic", "depth_camera_info_topic")


def constant_model(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """2.5 px everywhere: 1.0 m at scale 0.5 with these intrinsics."""
    return np.full(left.shape[2:], 2.5, np.float32)


def _image(sec: int, frame: str = LEFT_FRAME, value: int = 40) -> Image:
    return Image(
        header=Header(stamp=Time(sec=sec), frame_id=frame),
        height=HEIGHT,
        width=WIDTH,
        encoding="bgr8",
        step=WIDTH * 3,
        data=np.full((HEIGHT, WIDTH, 3), value, np.uint8).tobytes(),
    )


def _info(frame: str = LEFT_FRAME, p=LEFT_P) -> CameraInfo:
    info = CameraInfo()
    info.header.frame_id = frame
    info.width, info.height = WIDTH, HEIGHT
    info.d = [0.0] * 5
    info.k = K
    info.p = list(p)
    return info


def _topics() -> dict[str, str]:
    base = f"/t_{uuid.uuid4().hex[:10]}"
    return {name: f"{base}/{name.removesuffix('_topic')}" for name in (*INPUTS, *OUTPUTS)}


@pytest.fixture
def ctx():
    context = rclpy.Context()
    # Not 77 or 78: the simulations on this machine use those.
    rclpy.init(context=context, domain_id=79)
    yield context
    context.try_shutdown()


@pytest.fixture
def make_node(ctx):
    made: list[StereoDepthNode] = []

    def make(model=constant_model, **params) -> StereoDepthNode:
        overrides = [Parameter(k, value=v) for k, v in {**_topics(), **params}.items()]
        node = StereoDepthNode(model=model, context=ctx, parameter_overrides=overrides)
        made.append(node)
        return node

    yield make
    for node in made:
        node.destroy_node()


class _Recorder:
    def __init__(self, name: str, log: list) -> None:
        self.name, self.log = name, log

    def publish(self, msg) -> None:
        self.log.append((self.name, msg))


def _record(node: StereoDepthNode) -> list:
    log: list = []
    node._info_pub = _Recorder("camera_info", log)
    node._depth_info_pub = _Recorder("depth_camera_info", log)
    node._depth_pub = _Recorder("depth", log)
    node._image_pub = _Recorder("image", log)
    return log


def _hold_infos(node: StereoDepthNode) -> None:
    node._left_info, node._right_info = _info(), _info(RIGHT_FRAME, RIGHT_P)


# The config.


def test_the_config_defaults_read_the_zed_and_publish_under_walkie_vision_stereo():
    assert StereoConfig() == StereoConfig(
        left_image_topic="/zed_head/zed_node/rgb/color/rect/image",
        left_camera_info_topic="/zed_head/zed_node/rgb/color/rect/camera_info",
        right_image_topic="/zed_head/zed_node/right/color/rect/image",
        right_camera_info_topic="/zed_head/zed_node/right/color/rect/camera_info",
        image_topic="/walkie_vision/stereo/image",
        camera_info_topic="/walkie_vision/stereo/camera_info",
        depth_topic="/walkie_vision/stereo/depth",
        depth_camera_info_topic="/walkie_vision/stereo/depth/camera_info",
        model="ffs/c_fast_foundationstereo_bp2_dynbatch_dynhw.onnx",
        scale_x=0.5,
        scale_y=0.5,
        max_rate_hz=0.0,
    )
    assert StereoConfig().model == models.FFS_ONNX


def test_the_stereo_model_is_pinned_to_a_commit():
    (item,) = [d for d in models.DOWNLOADS if d.dest == models.FFS_ONNX]
    assert re.search(r"/resolve/[0-9a-f]{40}/", item.url)
    assert re.fullmatch(r"[0-9a-f]{64}", item.sha256)
    assert item.size == 103_228_473


def _vision_parameters(name: str) -> dict:
    path = REPO / "config" / "tasks" / f"{name}.yaml"
    return yaml.safe_load(path.read_text())["walkie_vision"]["ros__parameters"]


def test_memory_ffs_reads_what_the_stereo_node_publishes():
    ffs, memory = _vision_parameters("memory_ffs"), _vision_parameters("memory")
    cfg = StereoConfig()
    for name in OUTPUTS:
        assert ffs.pop(name) == getattr(cfg, name), name
    assert ffs.pop("pair_wait_s") == MAX_PAIR_WINDOW_S
    # Over the 1.3 to 1.5 s these arrays are expected to be (ADR-0017), so
    # the stale warning is not on for every array.
    assert ffs.pop("stale_warn_s") == 2.0
    assert ffs == memory


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("scale_x", 0.0, "scale_x"),
        ("scale_y", 1.5, "scale_y"),
        ("scale_x", float("nan"), "scale_x"),
        ("max_rate_hz", -1.0, "max_rate_hz"),
        ("max_rate_hz", float("inf"), "max_rate_hz"),
        ("model", "ffs/model.pt", "model"),
        ("model", "", "model"),
        ("depth_topic", "relative/depth", "depth_topic"),
        ("depth_topic", "/walkie_vision/stereo/image", "distinct"),
    ],
)
def test_the_config_refuses_bad_values(field, value, match):
    with pytest.raises(ValueError, match=match):
        StereoConfig(**{field: value})


def test_the_node_checks_parameters_before_loading_the_model(ctx):
    overrides = [Parameter("scale_x", value=0.0), Parameter("model", value="/nonexistent/x.onnx")]
    with pytest.raises(ValueError, match="scale_x"):
        StereoDepthNode(context=ctx, parameter_overrides=overrides)


def test_a_missing_model_file_names_fetch_models(ctx):
    with pytest.raises(FileNotFoundError, match="fetch-models"):
        StereoDepthNode(context=ctx, parameter_overrides=[Parameter("model", value="/nonexistent/x.onnx")])


def test_each_parameter_is_declared_read_only(make_node):
    node = make_node()
    for name in StereoConfig.__dataclass_fields__:
        described = node.describe_parameter(name)
        assert described.name == name and described.read_only, name


# The pairing.


def test_the_pairer_pairs_equal_stamps_in_either_order():
    pairer = StereoPairer()
    left, right = _image(1), _image(1, RIGHT_FRAME)
    assert pairer.offer_left(left) is None
    assert pairer.offer_right(right) == (left, right)
    left, right = _image(2), _image(2, RIGHT_FRAME)
    assert pairer.offer_right(right) is None
    pair = pairer.offer_left(left)
    assert pair[0] is left and pair[1] is right
    assert not pairer.counts


def test_the_pairer_keeps_only_the_newest_and_counts_what_it_drops():
    pairer = StereoPairer()
    left1, left2, right2 = _image(1), _image(2), _image(2, RIGHT_FRAME)
    assert pairer.offer_left(left1) is None
    assert pairer.offer_left(left2) is None
    assert pairer.counts["unmatched"] == 1
    assert pairer.offer_right(right2) == (left2, right2)
    right3, left4, right4 = _image(3, RIGHT_FRAME), _image(4), _image(4, RIGHT_FRAME)
    assert pairer.offer_right(right3) is None
    assert pairer.offer_left(left4) is None
    assert pairer.counts["unmatched"] == 2
    assert pairer.offer_right(_image(3, RIGHT_FRAME)) is None  # late: older than the held left
    assert pairer.counts["unmatched"] == 3
    assert pairer.offer_right(right4) == (left4, right4)


def test_the_pairer_never_pairs_a_zero_stamp():
    counts: Counter = Counter()
    pairer = StereoPairer(counts)
    assert pairer.offer_left(_image(0)) is None
    assert pairer.offer_right(_image(0, RIGHT_FRAME)) is None
    assert counts["zero_stamp"] == 2


# The outputs.


def test_the_node_publishes_both_infos_and_depth_and_relays_the_left_image(ctx, make_node):
    node = make_node()
    topics = {name: getattr(node.config, name) for name in (*INPUTS, *OUTPUTS)}
    helper = rclpy.create_node(f"test_{uuid.uuid4().hex[:10]}", context=ctx)
    got: dict[str, list] = {name: [] for name in OUTPUTS}
    for name, msg_type in (
        ("image_topic", Image),
        ("camera_info_topic", CameraInfo),
        ("depth_topic", Image),
        ("depth_camera_info_topic", CameraInfo),
    ):
        helper.create_subscription(msg_type, topics[name], got[name].append, 5)
    pubs = {
        name: helper.create_publisher(Image if "image" in name else CameraInfo, topics[name], INPUT_QOS)
        for name in INPUTS
    }
    executor = SingleThreadedExecutor(context=ctx)
    executor.add_node(node)
    executor.add_node(helper)
    sent: dict[int, Image] = {}
    try:
        deadline = time.monotonic() + TIMEOUT_S
        sec = 1
        # Publish until every output arrives: discovery can drop the first sends.
        while not all(got.values()):
            assert time.monotonic() < deadline, f"timed out: {({k: len(v) for k, v in got.items()})}, {node.counts}"
            sent[sec] = _image(sec, value=sec)
            for name, msg in (
                ("left_camera_info_topic", _info()),
                ("right_camera_info_topic", _info(RIGHT_FRAME, RIGHT_P)),
                ("left_image_topic", sent[sec]),
                ("right_image_topic", _image(sec, RIGHT_FRAME, value=sec)),
            ):
                pubs[name].publish(msg)
                for _ in range(3):
                    executor.spin_once(timeout_sec=0.01)
            sec += 1
        for _ in range(10):
            executor.spin_once(timeout_sec=0.01)
    finally:
        executor.shutdown()
        helper.destroy_node()
    # Reliable, but keep-last 1: take the first stamp with all four outputs.
    stamps = [{m.header.stamp.sec for m in msgs} for msgs in got.values()]
    sec = min(set.intersection(*stamps))
    (image,) = [m for m in got["image_topic"] if m.header.stamp.sec == sec]
    left = sent[sec]
    assert image.header == left.header
    assert (image.encoding, image.width, image.height, image.step) == (left.encoding, WIDTH, HEIGHT, left.step)
    assert bytes(image.data) == bytes(left.data)
    (info,) = [m for m in got["camera_info_topic"] if m.header == left.header]
    assert list(info.k) == K and (info.width, info.height) == (WIDTH, HEIGHT)
    (depth,) = [m for m in got["depth_topic"] if m.header == left.header]
    assert (depth.encoding, depth.width, depth.height) == ("32FC1", 32, 24)
    metres = np.frombuffer(bytes(depth.data), dtype="<f4").reshape(24, 32)
    np.testing.assert_allclose(metres[:, 3:], 1.0, rtol=1e-6)
    assert np.isnan(metres[:, :3]).all()
    (depth_info,) = [m for m in got["depth_camera_info_topic"] if m.header == left.header]
    assert (depth_info.width, depth_info.height) == (32, 24)
    np.testing.assert_allclose(depth_info.k, SCALED_K)


def test_the_outputs_go_out_infos_then_depth_then_image(make_node):
    node = make_node()
    _hold_infos(node)
    log = _record(node)
    left = _image(5)
    node._process(left, _image(5, RIGHT_FRAME))
    assert [name for name, _ in log] == ["camera_info", "depth_camera_info", "depth", "image"]
    assert log[-1][1] is left
    assert all(msg.header == left.header for _, msg in log)
    assert node.counts["published"] == 1


def test_the_node_logs_the_nearest_depth_once(make_node, capfd):
    node = make_node()
    _hold_infos(node)
    _record(node)
    for sec in (1, 2):
        node._process(_image(sec), _image(sec, RIGHT_FRAME))
    assert node.counts["published"] == 2
    # 25 px * 0.1 m / 184 px at the depth's size.
    assert "".join(capfd.readouterr()).count("depth starts at 0.01 m at scale_x 0.5") == 1


def test_a_refused_pair_is_counted_by_reason_and_publishes_nothing(make_node):
    calls = []
    node = make_node(model=lambda left, right: calls.append(1))
    log = _record(node)
    node._process(_image(5), _image(5, RIGHT_FRAME))
    assert node.counts["refused_camera_info"] == 1
    assert not log and not calls


def test_three_failures_in_a_row_raise_and_a_success_resets_the_run(make_node):
    outcomes = iter(["fail", "fail", "ok", "fail", "fail", "fail"])

    def flaky(left, right):
        if next(outcomes) == "fail":
            raise RuntimeError("cuda failure")
        return constant_model(left, right)

    node = make_node(model=flaky)
    _hold_infos(node)
    _record(node)
    for sec in range(1, 6):
        node._process(_image(sec), _image(sec, RIGHT_FRAME))
    assert node.counts["failed"] == 4 and node.counts["published"] == 1
    with pytest.raises(RuntimeError, match="cuda failure"):
        node._process(_image(6), _image(6, RIGHT_FRAME))


def test_max_rate_hz_skips_a_pair_that_comes_too_soon(make_node):
    node = make_node(max_rate_hz=1.0)
    _hold_infos(node)
    _record(node)
    for sec, now in ((1, 0.0), (2, 0.5), (3, 1.0)):
        node._monotonic = lambda now=now: now
        node._process(_image(sec), _image(sec, RIGHT_FRAME))
    assert node.counts["published"] == 2
    assert node.counts["rate_skipped"] == 1


# The report.


def _report(delta, ages=(), cfg=None):
    times = {"prepare": [10.0], "model": [800.0], "depth": [5.0]} if ages else {}
    return report_lines(delta, times, list(ages), [0.9] * len(ages), cfg or StereoConfig())


def test_the_report_hints_at_publish_left_right_when_only_left_images_arrive():
    lines = _report({"left": 5})
    assert any(level == "warning" and "publish_left_right" in text for level, text in lines)
    assert not any("publish_left_right" in text for _, text in _report({"left": 5, "right": 5}))


def test_the_report_warns_when_depth_is_older_than_age_warn_s():
    assert AGE_WARN_S < 1.6
    old = _report({"left": 2, "right": 2, "published": 2}, ages=[1.6, 1.7])
    assert any(level == "warning" and "1.65 s old" in text for level, text in old)
    # About the age a working node publishes at on the robot: no warning.
    normal = _report({"left": 1, "right": 1, "published": 1}, ages=[0.9])
    assert [level for level, _ in normal] == ["info"]
    assert "model 800" in normal[0][1] and "age s p50 0.90" in normal[0][1] and "valid p50 90%" in normal[0][1]


def test_the_report_warns_when_no_left_and_right_images_share_a_stamp():
    lines = _report({"left": 300, "right": 300, "unmatched": 299})
    assert any(level == "warning" and "share a stamp" in text for level, text in lines)
    pairing = _report({"left": 150, "right": 150, "unmatched": 280, "published": 10}, ages=[0.9] * 10)
    assert not any("share a stamp" in text for _, text in pairing)
    refusing = _report({"left": 150, "right": 150, "unmatched": 280, "refused_camera_info": 10})
    assert not any("share a stamp" in text for _, text in refusing)


def test_the_node_logs_a_report_with_info_and_warning_lines(make_node, capfd):
    node = make_node()
    node.counts["left"] += 5
    # rclpy refuses a second severity at one call site, so an info line then
    # a warning must not come from the same call.
    node._report()
    node.counts["left"] += 5
    node._report()
    out = "".join(capfd.readouterr())
    assert out.count("left 5 right 0") == 2
    assert out.count("publish_left_right") == 2
    assert node._reported["left"] == 10


def test_the_report_line_counts_refusals_by_reason():
    ((level, text), *_) = _report({"left": 3, "right": 3, "refused_frame": 1, "refused_camera_info": 2})
    assert level == "info"
    assert "refused 3 (camera_info 2, frame 1)" in text


# The process and its exit codes.


_STAND_IN_MODEL = """
import sys
import rclpy
from walkie_vision import stereo_depth


class FakeModel:
    providers = ["CPUExecutionProvider"]

    def __call__(self, left, right):
        return None


stereo_depth.load_model = lambda path: FakeModel()
"""

_CTRL_C_AND_A_LOST_SHUTDOWN_RACE = _STAND_IN_MODEL + """

def ctrl_c(node):
    raise KeyboardInterrupt


def lose_the_race():
    context = rclpy.get_default_context()
    context.shutdown()
    context.shutdown()


rclpy.spin = ctrl_c
rclpy.try_shutdown = lose_the_race
sys.exit(stereo_depth.main())
"""

_A_SECOND_CTRL_C_DURING_THE_CLEANUP = _STAND_IN_MODEL + """
import os
import signal
import time

shut_down = rclpy.try_shutdown


def ctrl_c(node):
    raise KeyboardInterrupt


def ctrl_c_again():
    os.kill(os.getpid(), signal.SIGINT)
    time.sleep(0.2)
    shut_down()


rclpy.spin = ctrl_c
rclpy.try_shutdown = ctrl_c_again
sys.exit(stereo_depth.main())
"""

_AN_ERROR_FROM_SPIN = _STAND_IN_MODEL + """

def broken(node):
    raise RuntimeError("cuda failure")


rclpy.spin = broken
sys.exit(stereo_depth.main())
"""


def _env() -> dict[str, str]:
    return dict(
        os.environ, ROS_DOMAIN_ID="79", RMW_IMPLEMENTATION="rmw_fastrtps_cpp", ROS_AUTOMATIC_DISCOVERY_RANGE="LOCALHOST"
    )


def _run_stand_in(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-P", "-c", script, "--ros-args", "-p", "model:=/nonexistent/x.onnx"],
        env=_env(),
        timeout=60,
        capture_output=True,
        text=True,
    )


def test_ctrl_c_exits_0_when_the_signal_handler_shuts_rclpy_down_first():
    # After Ctrl-C, rclpy's signal handler shuts the context down from its own
    # thread a moment later, and can do it between try_shutdown's check and its
    # shutdown. The stand-in forces that order.
    result = _run_stand_in(_CTRL_C_AND_A_LOST_SHUTDOWN_RACE)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "Traceback" not in output


def test_a_second_ctrl_c_does_not_cut_the_cleanup_short():
    # pixi run forwards Ctrl-C to its child, and ros2 launch sends one of its
    # own when its stdin is not a terminal, so a node often gets two SIGINTs.
    # The stand-in sends the second during the cleanup.
    result = _run_stand_in(_A_SECOND_CTRL_C_DURING_THE_CLEANUP)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "Traceback" not in output


def test_a_missing_model_exits_2():
    result = subprocess.run(
        [sys.executable, "-P", "-m", "walkie_vision.stereo_depth", "--ros-args", "-p", "model:=/nonexistent/x.onnx"],
        env=_env(),
        timeout=60,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 2, output
    assert "fetch-models" in output
    assert "Traceback" not in output


def test_an_error_from_spin_exits_1():
    result = _run_stand_in(_AN_ERROR_FROM_SPIN)
    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    assert "cuda failure" in output
