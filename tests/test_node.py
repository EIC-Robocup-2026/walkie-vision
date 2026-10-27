"""The vision node on Fast DDS, with fake providers.

Every test uses its own context on domain 77 and its own topic names, so
tests cannot hear each other. Images are at most 64 x 48, so Fast DDS's
shared memory carries them.
"""

from __future__ import annotations

import copy
import dataclasses
import gc
import os
import subprocess
import sys
import threading
import time
import uuid
import weakref
from pathlib import Path

import cv2
import numpy as np
import pytest
import rclpy
import yaml
from builtin_interfaces.msg import Time
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from walkie_vision_msgs.msg import ObjectDetectionArray, PersonDetectionArray
from walkie_vision_msgs.srv import GetPersonCrop

from walkie_vision import crop
from walkie_vision import node as node_module
from walkie_vision.convert import bbox_to_box
from walkie_vision.fakes import (
    FakeObjectDetector,
    FakePersonFaceRecognizer,
    FakePoseEstimator,
    FakeReIdentifier,
    figure_pose,
)
from walkie_vision.node import IMAGE_QOS, NODE_NAME, CropJob, CropMailbox, VisionConfig, VisionNode
from walkie_vision.pipeline import Providers, TrackIdBase
from walkie_vision.providers import Detection, Face, PersonPose

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config"
WIDTH, HEIGHT = 64, 48
FRAME = "zed_head_left_camera_frame_optical"
K = [50.0, 0.0, 32.0, 0.0, 50.0, 24.0, 0.0, 0.0, 1.0]
CLASSES = ["chair", "sofa"]
DEADLINE_S = 10.0

# walkie-agent-ros's VisionFeed subscribes with this profile.
AGENT_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)
RELIABLE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=50,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)


LEFT = (2.0, 2.0, 40.0, 46.0)
RIGHT = (44.0, 2.0, 62.0, 46.0)  # 18 x 44 px, apart from LEFT, so the dedupe keeps both


def _pose(track_id: int = 3, box=LEFT, score: float = 0.9) -> PersonPose:
    return PersonPose(
        box=np.array(box, dtype=np.float64),
        score=score,
        keypoints=np.full((17, 2), 10.0, dtype=np.float32),
        keypoint_scores=np.full(17, 0.8, dtype=np.float32),
        track_id=track_id,
    )


def fake_providers(*, objects=None, pose=None) -> Providers:
    return Providers(
        objects=objects
        or FakeObjectDetector(
            [Detection(box=np.array([1.0, 2.0, 30.0, 40.0]), label="chair", score=0.8, track_id=5)], classes=CLASSES
        ),
        pose=pose or FakePoseEstimator([_pose(3), _pose(0, RIGHT)]),
        faces=FakePersonFaceRecognizer(
            [Face(bbox=np.array([10.0, 5.0, 20.0, 15.0]), kps=np.zeros((5, 2)), det_score=0.8)]
        ),
        reid=FakeReIdentifier(),
    )


class BlockingPose:
    """A pose fake whose first call waits until released."""

    def __init__(self) -> None:
        self.inner = FakePoseEstimator([_pose(3)])
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def estimate(self, image):
        self.calls += 1
        if self.calls == 1:
            self.entered.set()
            self.release.wait(DEADLINE_S)
        return self.inner.estimate(image)


class RaisingDetector(FakeObjectDetector):
    """Raises on images whose first byte is in `values`, or on every image if `values` is None.

    The test fills each image with its stamp.
    """

    def __init__(self, values) -> None:
        super().__init__([], classes=CLASSES)
        self.values = None if values is None else set(values)

    def detect(self, image):
        if self.values is None or int(image[0, 0, 0]) in self.values:
            raise RuntimeError("the detector failed")
        return super().detect(image)


@pytest.fixture
def ctx():
    context = rclpy.Context()
    rclpy.init(context=context, domain_id=77)
    yield context
    context.try_shutdown()


class Harness:
    """A VisionNode on unique topics, a node that feeds it, collectors of its arrays and a crop client.

    The executor is main()'s: a GetPersonCrop callback that waits must not
    stop the image callback.
    """

    def __init__(self, ctx, providers: Providers, overrides, track_ids: TrackIdBase | None) -> None:
        prefix = f"/t_{uuid.uuid4().hex[:10]}"
        self.topics = {
            "image_topic": f"{prefix}/image",
            "camera_info_topic": f"{prefix}/camera_info",
            "people_topic": f"{prefix}/people",
            "objects_topic": f"{prefix}/objects",
            "person_crop_service": f"{prefix}/get_person_crop",
        }
        params = [Parameter("classes", value=CLASSES), Parameter("max_rate_hz", value=0.0)]
        params += [Parameter(k, value=v) for k, v in self.topics.items()]
        params += [Parameter(k, value=v) for k, v in overrides.items()]
        self.node = VisionNode(
            providers, track_ids=track_ids or TrackIdBase(start_ns=0), context=ctx, parameter_overrides=params
        )
        self.helper = rclpy.create_node(f"test_{uuid.uuid4().hex[:10]}", context=ctx)
        self.executor = MultiThreadedExecutor(num_threads=node_module.EXECUTOR_THREADS, context=ctx)
        self.executor.add_node(self.node)
        self.executor.add_node(self.helper)
        self.image_pub = self.helper.create_publisher(Image, self.topics["image_topic"], IMAGE_QOS)
        self.info_pub = self.helper.create_publisher(CameraInfo, self.topics["camera_info_topic"], IMAGE_QOS)
        self.crop_client = self.helper.create_client(GetPersonCrop, self.topics["person_crop_service"])
        self.got: dict[str, list] = {"people agent": [], "people reliable": [], "objects agent": [], "objects reliable": []}
        self.subs = []
        for kind, msg_type in (("people", PersonDetectionArray), ("objects", ObjectDetectionArray)):
            for name, qos in (("agent", AGENT_QOS), ("reliable", RELIABLE_QOS)):
                key = f"{kind} {name}"
                self.subs.append(
                    self.helper.create_subscription(msg_type, self.topics[f"{kind}_topic"], self.got[key].append, qos)
                )
        self.closed = False

    def spin_until(self, condition, timeout: float = DEADLINE_S) -> bool:
        deadline = time.monotonic() + timeout
        while not condition() and time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=0.05)
        return condition()

    def spin_for(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=0.05)

    def wait_matched(self) -> None:
        pubs, subs = [self.image_pub, self.info_pub], self.subs
        assert self.spin_until(
            lambda: all(p.get_subscription_count() >= 1 for p in pubs)
            and all(s.get_publisher_count() >= 1 for s in subs)
            and self.crop_client.service_is_ready()
        ), "discovery did not finish"

    def request_crop(self, track_id: int, **fields):
        """Send a GetPersonCrop request and return its future."""
        return self.crop_client.call_async(GetPersonCrop.Request(track_id=track_id, **fields))

    def answer(self, future) -> GetPersonCrop.Response:
        assert self.spin_until(future.done), "no GetPersonCrop answer"
        return future.result()

    def camera_info(self, sec: int, width: int = WIDTH, height: int = HEIGHT) -> CameraInfo:
        info = CameraInfo()
        info.header.stamp = Time(sec=sec)
        info.header.frame_id = FRAME
        info.width, info.height = width, height
        info.k = K
        return info

    def image(self, sec: int) -> Image:
        msg = Image()
        msg.header.stamp = Time(sec=sec)
        msg.header.frame_id = FRAME
        msg.height, msg.width = HEIGHT, WIDTH
        msg.encoding = "bgr8"
        msg.step = WIDTH * 3
        # Filled with the stamp, so a fake can tell the images apart.
        msg.data = np.full((HEIGHT, WIDTH, 3), sec % 256, dtype=np.uint8).tobytes()
        return msg

    def publish_image(self, sec: int) -> None:
        """Publish one image and spin until the node has received it."""
        before = self.node.frames_received
        self.image_pub.publish(self.image(sec))
        assert self.spin_until(lambda: self.node.frames_received > before), f"image {sec} did not arrive"

    def publish_frame(self, sec: int, info: CameraInfo | None = None) -> None:
        """Publish a camera_info until the node holds one, then the image.

        BEST_EFFORT and VOLATILE: one early camera_info can be lost.
        """
        info = info or self.camera_info(sec)
        deadline = time.monotonic() + 0.5
        while True:
            self.info_pub.publish(info)
            self.spin_for(0.05)
            with self.node._camera_info_lock:
                held = self.node._camera_info
            if (held is not None and held.header.stamp.sec == sec) or time.monotonic() > deadline:
                break
        self.publish_image(sec)

    def stamps(self, key: str) -> list[int]:
        return [m.header.stamp.sec for m in self.got[key]]

    def close(self) -> bool:
        if self.closed:
            return True
        self.closed = True
        stopped = self.node.close()
        self.helper.destroy_node()
        self.executor.shutdown()
        return stopped


@pytest.fixture
def harness(ctx):
    made: list[Harness] = []

    def make(providers: Providers | None = None, track_ids: TrackIdBase | None = None, **overrides) -> Harness:
        h = Harness(ctx, providers or fake_providers(), overrides, track_ids)
        made.append(h)
        return h

    yield make
    for h in made:
        h.close()


# --- 1. one pair per image ---------------------------------------------


def test_one_people_and_one_objects_array_per_image(harness):
    h = harness()
    h.wait_matched()
    for sec in (1, 2, 3):
        h.publish_frame(sec)
        h.spin_for(0.2)
    assert h.spin_until(lambda: len(h.got["objects reliable"]) >= 3 and len(h.got["people reliable"]) >= 3)
    h.spin_for(0.2)
    assert sorted(h.stamps("objects reliable")) == [1, 2, 3]
    for qos in ("agent", "reliable"):
        objects, people = h.got[f"objects {qos}"], h.got[f"people {qos}"]
        assert objects, f"no objects array with the {qos} QoS"
        for array in objects:
            matches = [p for p in people if p.header.stamp == array.header.stamp]
            assert len(matches) == 1
            assert matches[0].header.frame_id == array.header.frame_id == FRAME
            for msg in (array, matches[0]):
                assert list(msg.camera_info.k) == K
                assert msg.camera_info.header == msg.header
    people = h.got["people reliable"][0]
    assert [p.track_id for p in people.people] == [3, 0]
    assert people.people[0].face_model == "fake_face" and people.people[0].appearance_model == "fake_reid"
    assert list(h.got["objects reliable"][0].classes) == CLASSES
    assert h.node.frames_processed == 3


# --- 2 and 3. camera_info ------------------------------------------------


def test_no_camera_info_publishes_nothing(harness):
    h = harness()
    h.wait_matched()
    for sec in (1, 2):
        h.publish_image(sec)
        assert h.spin_until(lambda: h.node.skipped["camera_info"] >= sec)
    h.spin_for(0.3)
    assert all(not v for v in h.got.values())
    assert h.node.frames_processed == 0


def test_a_camera_info_of_another_size_publishes_nothing(harness):
    h = harness()
    h.wait_matched()
    h.publish_frame(1, info=h.camera_info(1, width=WIDTH * 2, height=HEIGHT * 2))
    assert h.spin_until(lambda: h.node.skipped["camera_info"] >= 1)
    h.spin_for(0.3)
    assert all(not v for v in h.got.values())


# --- 4. newest image only -----------------------------------------------


def test_only_the_newest_image_is_processed(harness):
    pose = BlockingPose()
    h = harness(fake_providers(pose=pose))
    h.wait_matched()
    h.publish_frame(1)
    assert h.spin_until(pose.entered.is_set), "the worker did not start on image 1"
    for sec in (2, 3, 4, 5):
        h.publish_image(sec)
    pose.release.set()
    assert h.spin_until(lambda: 5 in h.stamps("objects reliable") and 5 in h.stamps("people reliable"))
    h.spin_for(0.3)
    assert set(h.stamps("objects reliable")) == {1, 5}
    assert set(h.stamps("people reliable")) == {1, 5}
    assert h.node.frames_replaced >= 1


# --- 5. failures ---------------------------------------------------------


def test_a_failed_image_publishes_neither_array(harness):
    h = harness(fake_providers(objects=RaisingDetector({2})))
    h.wait_matched()
    h.publish_frame(1)
    assert h.spin_until(lambda: 1 in h.stamps("objects reliable"))
    h.publish_frame(2)
    assert h.spin_until(lambda: h.node.frames_failed == 1)
    assert h.node.consecutive_failures == 1
    h.publish_frame(3)
    assert h.spin_until(lambda: 3 in h.stamps("objects reliable") and 3 in h.stamps("people reliable"))
    h.spin_for(0.2)
    assert sorted(h.stamps("objects reliable")) == [1, 3]
    assert sorted(h.stamps("people reliable")) == [1, 3]
    assert h.node.consecutive_failures == 0 and h.node.fatal is None


def test_too_many_failures_in_a_row_are_fatal(harness):
    h = harness(fake_providers(objects=RaisingDetector(None)), max_consecutive_failures=2)
    h.wait_matched()
    h.publish_frame(1)
    assert h.spin_until(lambda: h.node.frames_failed == 1)
    assert h.node.fatal is None
    h.publish_frame(2)
    assert h.spin_until(lambda: h.node.fatal is not None)
    assert h.node.fatal.startswith("2 images in a row failed: RuntimeError: the detector failed")
    assert all(not v for v in h.got.values())


# --- 6. worker health -----------------------------------------------------


def test_a_worker_that_dies_is_fatal(harness, monkeypatch):
    def broken(msg):
        raise RuntimeError("conversion bug")

    monkeypatch.setattr(node_module, "image_to_bgr", broken)
    h = harness()
    h.wait_matched()
    h.publish_frame(1)
    assert h.spin_until(lambda: h.node.fatal is not None)
    fatal = h.node.fatal
    assert "worker" in fatal and "RuntimeError: conversion bug" in fatal
    h.node._worker.join(timeout=DEADLINE_S)
    assert not h.node._worker.is_alive()
    h.node.check_health()
    assert h.node.fatal == fatal


def test_a_worker_that_stops_is_noticed(harness):
    h = harness()
    # The thread ends without setting fatal itself.
    h.node._stop.set()
    h.node._mailbox.close()
    h.node._worker.join(timeout=DEADLINE_S)
    h.node._stop.clear()
    h.node.check_health()
    assert h.node.fatal == "the worker thread stopped"


def test_an_image_that_hangs_is_fatal(harness):
    pose = BlockingPose()
    h = harness(fake_providers(pose=pose), process_timeout_s=0.3)
    h.wait_matched()
    h.publish_frame(1)
    assert h.spin_until(pose.entered.is_set)
    h.node.check_health()
    assert h.node.fatal is None
    h.spin_for(0.5)
    h.node.check_health()
    assert h.node.fatal is not None and "more than 0.3 s" in h.node.fatal
    pose.release.set()
    assert h.close() is True


# --- 7. parameters ---------------------------------------------------------


def _fake_node(ctx, **kwargs) -> VisionNode:
    return VisionNode(fake_providers(), track_ids=TrackIdBase(start_ns=0), context=ctx, **kwargs)


def test_missing_classes_fail_at_construction(ctx):
    with pytest.raises(ValueError, match="classes"):
        _fake_node(ctx, parameter_overrides=[Parameter("max_rate_hz", value=0.0)])


def test_an_empty_classes_list_fails_at_construction(ctx):
    with pytest.raises(ValueError, match="classes"):
        _fake_node(ctx, parameter_overrides=[Parameter("classes", value=[])])


@pytest.mark.parametrize(
    "name, value",
    [
        ("objects_conf", 1.5),
        ("reid_batch_size", 0),
        ("face_input_sizes", [100]),
        ("device", "gpu"),
        ("people_topic", "no_slash"),
        ("person_crop_service", "no_slash"),
        ("objects_tracker", "relative/tracker.yaml"),
        ("objects_imgsz", 100),
        ("face_det_thresh", 0.0),
        ("people_dedupe_iou", 0.0),
        ("max_rate_hz", -1.0),
        # The wrong type: an integer for a double parameter.
        ("process_timeout_s", 60),
    ],
)
def test_a_bad_parameter_fails_at_construction(ctx, name, value):
    overrides = [Parameter("classes", value=CLASSES), Parameter(name, value=value)]
    with pytest.raises(ValueError, match=name):
        _fake_node(ctx, parameter_overrides=overrides)


def test_an_integer_for_a_double_in_a_params_file_fails(ctx, tmp_path):
    params = tmp_path / "params.yaml"
    params.write_text(yaml.safe_dump({NODE_NAME: {"ros__parameters": {"classes": CLASSES, "max_rate_hz": 10}}}))
    with pytest.raises(ValueError, match="max_rate_hz"):
        _fake_node(ctx, cli_args=["--ros-args", "--params-file", str(params)])


def _yaml_parameters(path: Path) -> dict:
    return yaml.safe_load(path.read_text())[NODE_NAME]["ros__parameters"]


def _as_config_value(value):
    return tuple(value) if isinstance(value, list) else value


def test_the_real_parameter_files(ctx):
    files = [CONFIG / "vision.yaml", CONFIG / "tasks" / "hri.yaml"]
    args = ["--ros-args"] + [a for f in files for a in ("--params-file", str(f))]
    node = _fake_node(ctx, cli_args=args)
    try:
        for f in files:
            for key, value in _yaml_parameters(f).items():
                assert getattr(node.config, key) == _as_config_value(value), key
        assert node.config.classes == ("chair", "sofa", "armchair", "stool")
        assert node.get_parameter("use_sim_time").value is False
        assert node.config.people_topic == "/walkie_vision/people"
        assert node.config.person_crop_service == "/walkie_vision/get_person_crop"
    finally:
        assert node.close()

    node = _fake_node(ctx, cli_args=args + ["--params-file", str(CONFIG / "sim.yaml")])
    try:
        assert node.get_parameter("use_sim_time").value is True
        assert node.config.people_topic == "/walkie_vision/sim/people"
        assert node.config.objects_topic == "/walkie_vision/sim/objects"
        assert node.config.person_crop_service == "/walkie_vision/sim/get_person_crop"
        assert node.config.classes == ("chair", "sofa", "armchair", "stool")
    finally:
        assert node.close()


def test_the_config_defaults_equal_vision_yaml():
    values = _yaml_parameters(CONFIG / "vision.yaml")
    for f in dataclasses.fields(VisionConfig):
        if f.name != "classes":
            assert f.default == _as_config_value(values[f.name]), f.name


def test_every_key_in_the_parameter_files_is_a_config_field():
    names = {f.name for f in dataclasses.fields(VisionConfig)}
    vision = set(_yaml_parameters(CONFIG / "vision.yaml"))
    assert vision == names - {"classes"}
    assert "classes" not in vision
    assert set(_yaml_parameters(CONFIG / "sim.yaml")) <= names | {"use_sim_time"}
    for task in (CONFIG / "tasks").glob("*.yaml"):
        assert set(_yaml_parameters(task)) <= names, task.name
        assert "classes" in _yaml_parameters(task), task.name


def test_every_double_in_the_parameter_files_has_a_decimal_point():
    # rclpy rejects an integer for a declared double.
    doubles = {f.name for f in dataclasses.fields(VisionConfig) if isinstance(f.default, float)}
    for path in [CONFIG / "vision.yaml", CONFIG / "sim.yaml", *(CONFIG / "tasks").glob("*.yaml")]:
        for key, value in _yaml_parameters(path).items():
            if key in doubles:
                assert isinstance(value, float), f"{path.name}: {key}"


def test_classes_cannot_change_at_runtime(ctx):
    node = _fake_node(ctx, parameter_overrides=[Parameter("classes", value=CLASSES)])
    try:
        result = node.set_parameters([Parameter("classes", value=["table"])])[0]
        assert not result.successful
        assert node.config.classes == tuple(CLASSES)
        assert list(node.get_parameter("classes").value) == CLASSES
    finally:
        assert node.close()


# --- 8. track ids -----------------------------------------------------------


@pytest.mark.parametrize("start_ms", [0, 1_793_000_000_000])
def test_track_ids_are_mapped_through_the_base(harness, start_ms):
    base = TrackIdBase(start_ns=start_ms * 1_000_000)
    h = harness(track_ids=base)
    h.wait_matched()
    h.publish_frame(1)
    assert h.spin_until(lambda: h.got["people reliable"] and h.got["objects reliable"])
    people = h.got["people reliable"][0].people
    assert [p.track_id for p in people] == [base.map(3), 0]
    assert [d.track_id for d in h.got["objects reliable"][0].detections] == [base.map(5)]
    if start_ms == 0:
        assert [p.track_id for p in people] == [3, 0]


def test_a_second_box_of_one_person_is_not_published(harness):
    # A partial box inside person 3's, as YOLO26 pose sometimes adds one.
    ghost = _pose(4, (2.0, 2.0, 40.0, 33.0), score=0.4)
    h = harness(providers=fake_providers(pose=FakePoseEstimator([_pose(3), ghost])))
    h.wait_matched()
    h.publish_frame(1)
    assert h.spin_until(lambda: h.got["people reliable"])
    assert [p.track_id for p in h.got["people reliable"][0].people] == [3]
    assert h.node.duplicates == 1
    assert sum(h.node.dropped.values()) == 0


# --- 9. other publishers ------------------------------------------------------


def test_other_publishers_on_the_output_topics(harness):
    h = harness()
    h.wait_matched()
    assert h.node.other_publishers() == []
    intruder = h.helper.create_publisher(PersonDetectionArray, h.topics["people_topic"], AGENT_QOS)
    assert h.spin_until(lambda: h.node.other_publishers() == [h.topics["people_topic"]])
    h.helper.destroy_publisher(intruder)
    assert h.spin_until(lambda: h.node.other_publishers() == [])


# --- 10. the latency timer ---------------------------------------------------


def test_the_latency_summary_runs_on_wall_time(harness):
    # Sim time without a /clock: the node's clock stands still at 0.
    h = harness(use_sim_time=True, latency_log_period_s=0.3)
    assert h.node.get_clock().now().nanoseconds == 0
    assert h.spin_until(lambda: h.node.summaries_logged >= 2, timeout=2.0)


# --- 11. start failures, in a subprocess ---------------------------------------


def _run_node(tmp_path: Path, *ros_args: str) -> subprocess.CompletedProcess:
    env = dict(
        os.environ,
        ROS_DOMAIN_ID="77",
        RMW_IMPLEMENTATION="rmw_fastrtps_cpp",
        ROS_AUTOMATIC_DISCOVERY_RANGE="LOCALHOST",
        WALKIE_VISION_MODELS=str(tmp_path),
    )
    return subprocess.run(
        [sys.executable, "-P", "-m", "walkie_vision.node", "--ros-args", *ros_args],
        env=env,
        timeout=60,
        capture_output=True,
        text=True,
    )


class _FatalLog:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def fatal(self, message: str) -> None:
        self.lines.append(message)


def _raised(error: BaseException) -> BaseException:
    try:
        raise error
    except BaseException as e:
        return e


@pytest.mark.parametrize(
    "error, traceback_logged",
    [
        (node_module.StartupError("OSNet is not in the Hugging Face cache"), False),
        (FileNotFoundError("model files are missing"), False),
        (ValueError("parameter classes"), False),
        (RuntimeError("CUDA out of memory"), True),
    ],
)
def test_a_startup_failure_logs_a_traceback_only_when_it_does_not_explain_itself(error, traceback_logged):
    log = _FatalLog()
    node_module._log_startup_failure(log, _raised(error))
    assert len(log.lines) == 1
    assert log.lines[0].startswith(f"cannot start: {type(error).__name__}: {error}")
    assert ("Traceback" in log.lines[0]) == traceback_logged


def test_a_missing_classes_list_exits_2(tmp_path):
    result = _run_node(tmp_path)
    output = result.stdout + result.stderr
    assert result.returncode == node_module.EXIT_STARTUP, output
    assert "classes" in output
    assert "Traceback" not in output


def test_missing_model_files_exit_2_before_any_model_loads(tmp_path):
    result = _run_node(tmp_path, "-p", "classes:=[chair]")
    output = result.stdout + result.stderr
    assert result.returncode == node_module.EXIT_STARTUP, output
    assert "pixi run fetch-models" in output
    assert "Traceback" not in output


# --- 12. person crops ------------------------------------------------------------

OK = GetPersonCrop.Response.OK


class PosesByImage:
    """A pose fake that finds the people given for each image's stamp, and nobody in other images.

    It keeps a weak reference to every image it gets, so a test can check
    that the node drops them.
    """

    def __init__(self, people_by_sec: dict[int, list[PersonPose]]) -> None:
        self.people_by_sec = people_by_sec
        self.images: list[weakref.ref] = []

    def estimate(self, image):
        self.images.append(weakref.ref(image))
        return copy.deepcopy(self.people_by_sec.get(int(image[0, 0, 0]), []))


@pytest.fixture
def small_boxes(monkeypatch):
    # The test images are 48 px tall; a real crop needs a 160 px box.
    monkeypatch.setattr(crop, "MIN_BOX_HEIGHT_PX", 10.0)


def test_the_crop_service_has_its_own_callback_group(harness):
    h = harness()
    group = h.node._crop_service.callback_group
    assert isinstance(group, MutuallyExclusiveCallbackGroup)
    assert group is not h.node.default_callback_group
    assert node_module.EXECUTOR_THREADS == 2


def test_a_person_crop_comes_from_the_next_image(harness, small_boxes):
    # Track 3 moves from the right to the left between images 1 and 2.
    pose = PosesByImage({1: [figure_pose(RIGHT, track_id=3)], 2: [figure_pose(LEFT, track_id=3)]})
    h = harness(fake_providers(pose=pose))
    h.wait_matched()
    h.publish_frame(1)
    assert h.spin_until(lambda: 1 in h.stamps("people reliable"))
    future = h.request_crop(3)
    assert h.spin_until(lambda: h.node.crop_requests == 1)
    h.publish_frame(2)
    response = h.answer(future)
    assert response.status == OK, response.message
    assert response.message == ""
    assert response.header.stamp.sec == 2 and response.header.frame_id == FRAME
    assert h.spin_until(lambda: 2 in h.stamps("people reliable"))
    (published,) = [m for m in h.got["people reliable"] if m.header.stamp.sec == 2]
    assert response.person == published.people[0]
    np.testing.assert_allclose(bbox_to_box(response.person.bbox), LEFT)
    assert response.crop.header == response.header and response.crop.format == "jpeg"
    # LEFT padded by 10% and clipped to the 64 x 48 image.
    decoded = cv2.imdecode(np.frombuffer(bytes(response.crop.data), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (48, 44, 3)
    assert len(response.crop_keypoints) == 34 and len(response.masked_boxes) == 0
    assert list(response.color_names) == list(crop.COLOR_NAMES)
    # Image 2 is filled with 2: black.
    assert response.top_shares[crop.COLOR_NAMES.index("black")] == pytest.approx(1.0)
    assert response.bottom_shares[crop.COLOR_NAMES.index("black")] == pytest.approx(1.0)


def test_a_request_during_an_image_is_answered_from_that_image(harness, small_boxes):
    pose = BlockingPose()
    h = harness(fake_providers(pose=pose))
    h.wait_matched()
    h.publish_frame(1)
    assert h.spin_until(pose.entered.is_set)
    future = h.request_crop(3)
    assert h.spin_until(lambda: h.node.crop_requests == 1)
    pose.release.set()
    response = h.answer(future)
    assert response.status == OK, response.message
    assert response.header.stamp.sec == 1


def test_a_track_that_is_not_in_the_next_image_is_not_in_view(harness):
    h = harness()
    h.wait_matched()
    future = h.request_crop(9)
    assert h.spin_until(lambda: h.node.crop_requests == 1)
    h.publish_frame(1)
    response = h.answer(future)
    assert response.status == GetPersonCrop.Response.NOT_IN_VIEW
    assert response.header.stamp.sec == 1
    assert response.person.track_id == 0 and len(response.crop.data) == 0


def test_a_person_under_160_px_is_too_small(harness):
    h = harness()
    h.wait_matched()
    future = h.request_crop(3)
    assert h.spin_until(lambda: h.node.crop_requests == 1)
    h.publish_frame(1)
    response = h.answer(future)
    assert response.status == GetPersonCrop.Response.TOO_SMALL
    assert response.person.track_id == 3 and response.header.stamp.sec == 1
    assert len(response.crop.data) == 0 and len(response.top_shares) == 0


def test_no_image_within_the_timeout_is_no_image(harness):
    h = harness()
    h.wait_matched()
    started = time.monotonic()
    response = h.answer(h.request_crop(3, timeout_s=0.3))
    assert response.status == GetPersonCrop.Response.NO_IMAGE
    assert h.topics["image_topic"] in response.message
    assert time.monotonic() - started >= 0.3
    assert response.header.stamp.sec == 0 and response.person.track_id == 0


def test_the_image_callback_runs_while_a_request_waits_and_then_times_out(harness):
    pose = BlockingPose()
    h = harness(fake_providers(pose=pose))
    h.wait_matched()
    h.publish_frame(1)
    assert h.spin_until(pose.entered.is_set)
    future = h.request_crop(3, timeout_s=2.0)
    assert h.spin_until(lambda: h.node.crop_requests == 1)
    received = h.node.frames_received
    for sec in (2, 3):
        h.publish_image(sec)
    assert h.node.frames_received == received + 2
    assert not future.done()
    response = h.answer(future)
    assert response.status == GetPersonCrop.Response.TIMEOUT
    assert response.header.stamp.sec == 0
    # The request is withdrawn before the worker, still in image 1, takes
    # it, so image 1 does not answer it later.
    assert h.node._crops.take() is None
    pose.release.set()
    assert h.spin_until(lambda: 1 in h.stamps("people reliable"))


def test_no_frame_is_kept_after_an_answer(harness, small_boxes):
    pose = PosesByImage({1: [figure_pose(LEFT, track_id=3)]})
    h = harness(fake_providers(pose=pose))
    h.wait_matched()
    future = h.request_crop(3)
    assert h.spin_until(lambda: h.node.crop_requests == 1)
    h.publish_frame(1)
    assert h.answer(future).status == OK

    def dropped() -> bool:
        gc.collect()
        return all(ref() is None for ref in pose.images)

    assert pose.images and h.spin_until(dropped), "the node still holds an image"
    assert h.node._crops.take() is None


@pytest.mark.parametrize(
    "fields, reason",
    [
        ({"track_id": 0}, "track_id"),
        ({"timeout_s": 0.0}, "timeout_s"),
        ({"timeout_s": 6.0}, "timeout_s"),
        ({"padding": 1.5}, "padding"),
        ({"max_side": 0}, "max_side"),
        ({"min_keypoint_score": -0.5}, "min_keypoint_score"),
    ],
)
def test_a_bad_request_fails_at_once(harness, fields, reason):
    h = harness()
    h.wait_matched()
    response = h.answer(h.request_crop(**{"track_id": 3, **fields}))
    assert response.status == GetPersonCrop.Response.FAILED
    assert reason in response.message
    assert h.node.crop_requests == 0


def test_closing_the_mailbox_answers_the_waiting_job():
    mailbox = CropMailbox()
    job = CropJob(GetPersonCrop.Request(track_id=3), frames_received=0)
    assert mailbox.put(job)
    mailbox.close()
    answer = job.wait(0.0)
    assert answer.status == GetPersonCrop.Response.FAILED and "shutting down" in answer.message
    assert not mailbox.put(CropJob(GetPersonCrop.Request(track_id=3), frames_received=0))


def test_a_job_given_up_takes_no_answer():
    job = CropJob(GetPersonCrop.Request(track_id=3), frames_received=0)
    assert job.wait(0.0) is None
    assert not job.finish(GetPersonCrop.Response(status=OK))
    assert job.wait(0.0) is None
