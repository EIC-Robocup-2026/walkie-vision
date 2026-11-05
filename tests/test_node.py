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
from collections.abc import Callable
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
from rclpy.time import Time as RosTime
from rosgraph_msgs.msg import Clock as ClockMsg
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
from walkie_vision_msgs.msg import ObjectDetectionArray, PersonDetectionArray
from walkie_vision_msgs.srv import DescribeCrop, DetectObjects, EmbedCrop, EmbedText, GetPersonCrop

from walkie_vision import crop, embeddings, requests
from walkie_vision import node as node_module
from walkie_vision import pipeline as pipeline_module
from walkie_vision.convert import (
    MAX_MESSAGE_CHARS,
    bbox_to_box,
    box_to_bbox,
    check_object_detection_array,
    embed_text_failure,
    vectors_from_response,
)
from walkie_vision.depth import RgbdBuffer, frame_laplacian, view_quality
from walkie_vision.fakes import (
    FakeEmbedder,
    FakeObjectDetector,
    FakePersonFaceRecognizer,
    FakePoseEstimator,
    FakeReIdentifier,
    box_mask,
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
MEMORY_CLASSES = ("cup", "bottle", "bowl", "apple", "banana", "backpack", "chair", "table", "shelf", "cabinet", "sofa")
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


def fake_providers_with_embedder(objects=None, embedder=None, **kwargs) -> Providers:
    return dataclasses.replace(fake_providers(objects=objects, **kwargs), embedder=embedder or FakeEmbedder())


class _SetClock:
    """Stands in for the node's ROS clock."""

    ros_time_is_active = False
    """As on system time, so the services keep every image whatever its stamp."""

    def __init__(self, ns: int) -> None:
        self.ns = ns

    def now(self) -> RosTime:
        return RosTime(nanoseconds=self.ns)


class BlockingEmbedder(FakeEmbedder):
    """An embedder whose first service call waits until released, as a stuck model would."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def _block(self) -> None:
        self.calls += 1
        if self.calls == 1:
            self.entered.set()
            self.release.wait(DEADLINE_S)

    def embed_texts(self, texts):
        self._block()
        return super().embed_texts(texts)

    def embed_images(self, images):
        self._block()
        return super().embed_images(images)


class RaisingEmbedder(FakeEmbedder):
    """An embedder that fails every call."""

    def embed_texts(self, texts):
        raise RuntimeError("CUDA out of memory")

    def embed_images(self, images):
        raise RuntimeError("CUDA out of memory")


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
            "embed_text_service": f"{prefix}/embed_text",
            "embed_crop_service": f"{prefix}/embed_crop",
            "describe_crop_service": f"{prefix}/describe_crop",
            "detect_objects_service": f"{prefix}/detect_objects",
            "depth_topic": f"{prefix}/depth",
            "depth_camera_info_topic": f"{prefix}/depth_camera_info",
        }
        self.topics.update({k: v for k, v in overrides.items() if k in self.topics})
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
        self.depth_pub = self.helper.create_publisher(Image, self.topics["depth_topic"], IMAGE_QOS)
        self.depth_info_pub = self.helper.create_publisher(CameraInfo, self.topics["depth_camera_info_topic"], IMAGE_QOS)
        self.crop_client = self.helper.create_client(GetPersonCrop, self.topics["person_crop_service"])
        self._service_clients: dict = {}
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
        if self.node.config.geometry_enabled:
            pubs += [self.depth_pub, self.depth_info_pub]
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

    def call(self, service_type, request):
        """Send a request to one of the bounded services and return its future."""
        name = {
            EmbedText: "embed_text_service",
            EmbedCrop: "embed_crop_service",
            DescribeCrop: "describe_crop_service",
            DetectObjects: "detect_objects_service",
        }
        client = self._service_clients.get(service_type)
        if client is None:
            client = self.helper.create_client(service_type, self.topics[name[service_type]])
            self._service_clients[service_type] = client
        assert self.spin_until(client.service_is_ready), "the service did not appear"
        return client.call_async(request)

    def set_clock(self, ns: int) -> None:
        """Hold the node's ROS clock at ns, so a test sets an image's age exactly."""
        clock = _SetClock(ns)
        self.node.get_clock = lambda: clock

    def publish_msg(self, msg: Image) -> None:
        """Publish one prepared image and spin until the node has received it."""
        before = self.node.frames_received
        self.image_pub.publish(msg)
        assert self.spin_until(lambda: self.node.frames_received > before), "the image did not arrive"

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

    def hold_infos(self, sec: int = 1, depth_info: CameraInfo | None = None) -> None:
        """Publish both camera_infos until the node holds them. BEST_EFFORT: an early one can be lost."""
        color_info, depth_info = self.camera_info(sec), depth_info or self.camera_info(sec)

        def held() -> bool:
            with self.node._camera_info_lock:
                return self.node._camera_info is not None and self.node._depth_camera_info is not None

        deadline = time.monotonic() + DEADLINE_S
        while not held() and time.monotonic() < deadline:
            self.info_pub.publish(color_info)
            self.depth_info_pub.publish(depth_info)
            self.spin_for(0.05)
        assert held(), "the camera_infos did not arrive"

    def freeze_pair_clock(self) -> None:
        """Stop the pairing's wall clocks, the stream's and the services', so a slow test machine cannot run out the wait."""
        self.node._mailbox._clock = lambda: 0.0
        if self.node._current is not None:
            self.node._current._clock = lambda: 0.0

    def publish_depth(self, msg: Image) -> None:
        """Publish one depth image and spin until the node has received it."""
        before = self.node.depth_received
        self.depth_pub.publish(msg)
        assert self.spin_until(lambda: self.node.depth_received > before), "the depth image did not arrive"

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
        ("embed_text_service", "no_slash"),
        ("embed_crop_service", "no_slash"),
        ("describe_crop_service", "no_slash"),
        ("captions_enabled", True),
        ("objects_tracker", "relative/tracker.yaml"),
        ("objects_imgsz", 100),
        ("face_det_thresh", 0.0),
        ("people_dedupe_iou", 0.0),
        ("max_rate_hz", -1.0),
        # The wrong type: an integer for a double parameter.
        ("process_timeout_s", 60),
        ("depth_topic", "no_slash"),
        ("depth_camera_info_topic", "no_slash"),
        ("max_pair_gap_s", -0.001),
        ("max_pair_gap_s", 0.2),
        ("max_pair_gap_s", float("nan")),
        ("pair_wait_s", 0.0),
        ("pair_wait_s", 0.2),
        ("require_depth", True),
        ("depth_min_m", 0.0),
        ("depth_min_m", float("nan")),
        ("depth_min_m", 5.0),
        ("depth_max_m", 0.1),
        ("depth_max_m", float("inf")),
        ("detect_objects_service", "no_slash"),
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
        assert node.config.embed_text_service == "/walkie_vision/embed_text"
        assert node.config.embed_crop_service == "/walkie_vision/embed_crop"
        assert node.config.describe_crop_service == "/walkie_vision/describe_crop"
        assert node.config.detect_objects_service == "/walkie_vision/detect_objects"
        assert node.config.captions_enabled is False
    finally:
        assert node.close()

    node = _fake_node(ctx, cli_args=args + ["--params-file", str(CONFIG / "sim.yaml")])
    try:
        assert node.get_parameter("use_sim_time").value is True
        assert node.config.people_topic == "/walkie_vision/sim/people"
        assert node.config.objects_topic == "/walkie_vision/sim/objects"
        assert node.config.person_crop_service == "/walkie_vision/sim/get_person_crop"
        assert node.config.embed_text_service == "/walkie_vision/sim/embed_text"
        assert node.config.embed_crop_service == "/walkie_vision/sim/embed_crop"
        assert node.config.describe_crop_service == "/walkie_vision/sim/describe_crop"
        assert node.config.detect_objects_service == "/walkie_vision/sim/detect_objects"
        assert node.config.classes == ("chair", "sofa", "armchair", "stool")
    finally:
        assert node.close()


def test_the_memory_task_turns_on_object_memory_without_a_person_class(ctx):
    files = [CONFIG / "vision.yaml", CONFIG / "tasks" / "memory.yaml"]
    args = ["--ros-args"] + [a for f in files for a in ("--params-file", str(f))]
    node = VisionNode(
        dataclasses.replace(
            fake_providers_with_embedder(FakeObjectDetector([], classes=list(MEMORY_CLASSES))),
            on_demand=FakeObjectDetector([]),
        ),
        track_ids=TrackIdBase(start_ns=0),
        context=ctx,
        cli_args=args,
    )
    try:
        assert node.config.classes == MEMORY_CLASSES
        assert "person" not in node.config.classes
        assert node.config.geometry_enabled and node.config.require_depth
        assert node.config.object_crops_enabled and node.config.object_embeddings_enabled
        assert node.config.on_demand_enabled
    finally:
        assert node.close()


def test_the_hri_task_keeps_every_object_addition_off():
    for path in (CONFIG / "vision.yaml", CONFIG / "tasks" / "hri.yaml"):
        values = _yaml_parameters(path)
        for flag in (
            "geometry_enabled",
            "require_depth",
            "object_crops_enabled",
            "object_embeddings_enabled",
            "on_demand_enabled",
        ):
            assert not values.get(flag, False), f"{path.name}: {flag}"


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
    assert node_module.EXECUTOR_THREADS == 4


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


def test_a_sampled_object_reaches_the_array(harness):
    # The first image has counter 0, so track id 10 is eligible. The second has counter 1.
    objects = FakeObjectDetector(
        [Detection(box=np.array(LEFT), label="chair", score=0.8, track_id=10, mask=box_mask(HEIGHT, WIDTH, LEFT))],
        classes=CLASSES,
    )
    h = harness(fake_providers_with_embedder(objects), object_crops_enabled=True, object_embeddings_enabled=True)
    h.wait_matched()
    h.publish_frame(1)
    assert h.spin_until(lambda: 1 in h.stamps("objects reliable"))
    h.publish_frame(2)
    assert h.spin_until(lambda: 2 in h.stamps("objects reliable"))
    first, second = (a.detections[0] for a in sorted(h.got["objects reliable"], key=lambda a: a.header.stamp.sec))
    assert first.crop.format == "jpeg" and bytes(first.crop.data[:2]) == b"\xff\xd8"
    assert first.embedding_model == "fake_embedder" and len(first.embedding) == 512
    assert len(first.color_shares) == 12
    assert first.crop.header.stamp.sec == 1
    assert not second.crop.data and not second.embedding and not second.color_shares and second.embedding_model == ""


def test_the_real_path_refuses_a_fake_embedder(ctx, monkeypatch):
    monkeypatch.setattr(node_module, "build_providers", lambda config, logger: fake_providers_with_embedder())
    params = [Parameter("classes", value=CLASSES), Parameter("object_embeddings_enabled", value=True)]
    with pytest.raises(ValueError, match="must not run with the real models"):
        VisionNode(track_ids=TrackIdBase(start_ns=0), context=ctx, parameter_overrides=params)


def test_object_embeddings_need_an_embedder(ctx):
    params = [Parameter("classes", value=CLASSES), Parameter("object_embeddings_enabled", value=True)]
    with pytest.raises(ValueError, match="needs an embedder"):
        VisionNode(fake_providers(), track_ids=TrackIdBase(start_ns=0), context=ctx, parameter_overrides=params)


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


# --- depth pairing --------------------------------------------------------------


def _depth_msg(sec: int, value: float = 1.0, *, encoding: str = "32FC1", frame: str = FRAME) -> Image:
    msg = Image()
    msg.header.stamp = Time(sec=sec)
    msg.header.frame_id = frame
    msg.height, msg.width = HEIGHT, WIDTH
    msg.encoding = encoding
    array = np.full((HEIGHT, WIDTH), value, dtype=np.float32 if encoding == "32FC1" else np.uint16)
    msg.step = WIDTH * array.dtype.itemsize
    msg.data = array.tobytes()
    return msg


def _subscribed(h: Harness) -> list[str]:
    return sorted(s.topic_name for s in h.node.subscriptions)


def test_the_default_config_adds_no_depth_subscription_or_buffer(harness):
    h = harness()
    assert h.node.config.geometry_enabled is False
    assert _subscribed(h) == sorted([h.topics["image_topic"], h.topics["camera_info_topic"]])
    assert isinstance(h.node._mailbox, node_module.NewestFrame)
    h.wait_matched()
    assert h.depth_pub.get_subscription_count() == 0
    h.publish_frame(1)
    assert h.spin_until(lambda: len(h.got["objects reliable"]) >= 1)
    h.node._on_latency_timer()
    assert list(h.node.latency.last_deltas) == ["received", "replaced", "skipped", "failed", "dropped", "duplicates"]


def test_geometry_subscribes_to_depth_and_its_own_camera_info(harness):
    h = harness(geometry_enabled=True)
    assert _subscribed(h) == sorted(
        [h.topics["image_topic"], h.topics["camera_info_topic"], h.topics["depth_topic"], h.topics["depth_camera_info_topic"]]
    )
    assert isinstance(h.node._mailbox, RgbdBuffer)


def test_one_camera_info_subscription_when_the_topics_coincide(harness):
    shared = f"/t_{uuid.uuid4().hex[:10]}/camera_info"
    h = harness(geometry_enabled=True, camera_info_topic=shared, depth_camera_info_topic=shared)
    assert _subscribed(h) == sorted([h.topics["image_topic"], shared, h.topics["depth_topic"]])
    h.wait_matched()
    h.freeze_pair_clock()
    h.publish_frame(1)
    h.publish_depth(_depth_msg(1))
    assert h.spin_until(lambda: h.node.depth_paired == 1)
    assert h.spin_until(lambda: len(h.got["objects reliable"]) == 1)
    assert not h.node.no_depth


@pytest.mark.parametrize("depth_first", [True, False])
def test_a_paired_image_is_published_in_either_arrival_order(harness, depth_first):
    h = harness(geometry_enabled=True, require_depth=True)
    h.wait_matched()
    h.hold_infos()
    h.freeze_pair_clock()
    if depth_first:
        h.publish_depth(_depth_msg(1))
        h.publish_image(1)
    else:
        h.publish_image(1)
        h.publish_depth(_depth_msg(1))
    assert h.spin_until(lambda: len(h.got["objects reliable"]) == 1 and len(h.got["people reliable"]) == 1)
    assert h.node.depth_paired == 1
    assert not h.node.skipped
    h.node._on_latency_timer()
    assert h.node.latency.last_deltas["depth"] == 1
    assert h.node.latency.last_deltas["paired"] == 1
    assert h.node.latency.last_deltas["no_depth"] == 0


def test_require_depth_skips_and_counts_an_unpaired_image(harness):
    h = harness(geometry_enabled=True, require_depth=True)
    h.wait_matched()
    h.hold_infos()
    h.publish_image(1)
    assert h.spin_until(lambda: h.node.skipped["unpaired"] == 1)
    h.spin_for(0.2)
    assert all(not v for v in h.got.values())
    assert h.node.frames_processed == 0


def test_without_require_depth_an_unpaired_image_is_published(harness):
    h = harness(geometry_enabled=True)
    h.wait_matched()
    h.hold_infos()
    h.publish_image(1)
    assert h.spin_until(lambda: len(h.got["objects reliable"]) == 1)
    assert h.node.no_depth["unpaired"] == 1
    assert not h.node.skipped
    assert len(h.got["objects reliable"][0].detections) == 1


@pytest.mark.parametrize("value", [float("nan"), 50.0, -1.0, 0.0])
def test_a_paired_image_with_invalid_depth_values_is_still_published(harness, value):
    h = harness(geometry_enabled=True, require_depth=True)
    h.wait_matched()
    h.hold_infos()
    h.freeze_pair_clock()
    h.publish_depth(_depth_msg(1, value))
    h.publish_image(1)
    assert h.spin_until(lambda: len(h.got["objects reliable"]) == 1)
    assert h.node.depth_paired == 1
    assert not h.node.skipped
    cloud = h.got["objects reliable"][0].detections[0].points
    assert cloud.width == 0 and len(cloud.data) == 0


@pytest.mark.parametrize(
    ("depth_msg", "depth_info_change", "reason"),
    [
        (_depth_msg(1, encoding="16UC1", frame="other"), {}, "depth_frame"),
        (_depth_msg(1, encoding="16UC1"), {"frame": "other"}, "depth_camera_info"),
        (_depth_msg(1, encoding="16UC1"), {"size": (WIDTH * 2, HEIGHT * 2)}, "depth_camera_info"),
        (_depth_msg(1, encoding="16UC1"), {"k": [90.0, 0.0, 32.0, 0.0, 50.0, 24.0, 0.0, 0.0, 1.0]}, "depth_calibration"),
        (_depth_msg(1, encoding="16UC1"), {"d": [0.1, 0.0, 0.0, 0.0, 0.0]}, "depth_calibration"),
    ],
)
@pytest.mark.parametrize("require_depth", [True, False])
def test_a_refused_pair_follows_require_depth(harness, depth_msg, depth_info_change, reason, require_depth):
    h = harness(geometry_enabled=True, require_depth=require_depth)
    h.wait_matched()
    info = h.camera_info(1)
    if "frame" in depth_info_change:
        info.header.frame_id = depth_info_change["frame"]
    if "size" in depth_info_change:
        info.width, info.height = depth_info_change["size"]
    if "k" in depth_info_change:
        info.k = depth_info_change["k"]
    if "d" in depth_info_change:
        info.d = depth_info_change["d"]
    h.hold_infos(depth_info=info)
    h.freeze_pair_clock()
    h.publish_depth(depth_msg)
    h.publish_image(1)
    if require_depth:
        assert h.spin_until(lambda: h.node.skipped[reason] == 1)
        h.spin_for(0.2)
        assert all(not v for v in h.got.values())
    else:
        assert h.spin_until(lambda: len(h.got["objects reliable"]) == 1)
        assert h.node.no_depth[reason] == 1
    assert h.node.depth_paired == 0


def test_a_wrong_encoding_is_counted_with_its_reason(harness):
    h = harness(geometry_enabled=True, require_depth=True)
    h.wait_matched()
    h.hold_infos()
    h.freeze_pair_clock()
    bad = _depth_msg(1)
    bad.encoding = "mono8"
    h.publish_depth(bad)
    h.publish_image(1)
    assert h.spin_until(lambda: h.node.skipped["depth_encoding"] == 1)


def test_a_paused_sim_clock_does_not_hang_the_pairing(harness):
    # Sim time without a /clock: the node's clock stands still at 0, and the wait still ends.
    h = harness(geometry_enabled=True, use_sim_time=True)
    assert h.node.get_clock().now().nanoseconds == 0
    h.wait_matched()
    h.hold_infos()
    h.publish_image(1)
    assert h.spin_until(lambda: len(h.got["objects reliable"]) == 1)
    assert h.node.no_depth["unpaired"] == 1


def test_a_depth_is_not_paired_with_a_second_image(harness):
    h = harness(geometry_enabled=True)
    h.wait_matched()
    h.hold_infos()
    h.freeze_pair_clock()
    h.publish_depth(_depth_msg(1))
    h.publish_image(1)
    assert h.spin_until(lambda: h.node.depth_paired == 1)
    assert h.spin_until(lambda: len(h.got["objects reliable"]) == 1)
    # The depth was used up, so the next image with the same stamp has none.
    # The clock runs again here, so that its wait can end.
    h.node._mailbox._clock = time.monotonic
    h.publish_image(1)
    assert h.spin_until(lambda: h.node.no_depth["unpaired"] == 1)
    assert h.node.depth_paired == 1


# --- points and view quality --------------------------------------------------------

MASKED_BOX = (8.0, 8.0, 40.0, 40.0)  # 32 x 32 px, so a radius of 2.


def _masked_providers() -> Providers:
    detection = Detection(
        box=np.array(MASKED_BOX), label="chair", score=0.8, track_id=5, mask=box_mask(HEIGHT, WIDTH, MASKED_BOX)
    )
    return fake_providers(objects=FakeObjectDetector([detection], classes=CLASSES))


def _published_object(h: Harness):
    assert h.spin_until(lambda: len(h.got["objects reliable"]) == 1)
    array = h.got["objects reliable"][0]
    (det,) = array.detections
    return array, det


def test_a_paired_image_publishes_points_in_the_optical_frame(harness):
    h = harness(_masked_providers(), geometry_enabled=True, require_depth=True)
    h.wait_matched()
    h.hold_infos()
    h.freeze_pair_clock()
    h.publish_depth(_depth_msg(1, 1.5))
    h.publish_image(1)
    array, det = _published_object(h)
    assert det.points.header == array.header
    assert det.points.width == 28 * 28 and det.points.height == 1
    xyz = np.frombuffer(bytes(det.points.data), dtype="<f4").reshape(-1, 3)
    # K has fx = fy = 50 at (32, 24). Erosion leaves pixels 10 to 37.
    np.testing.assert_array_equal(xyz[:, 2], 1.5)
    np.testing.assert_allclose(
        [xyz[:, 0].min(), xyz[:, 0].max(), xyz[:, 1].min(), xyz[:, 1].max()],
        [(10 - 32) * 1.5 / 50, (37 - 32) * 1.5 / 50, (10 - 24) * 1.5 / 50, (37 - 24) * 1.5 / 50],
        atol=1e-5,
    )
    assert 0.0 < det.view_quality <= 1.0


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"k": [0.0] * 9}, "depth_camera_info"),
        ({"d": [0.1, 0.0, 0.0, 0.0, 0.0]}, "depth_calibration"),
    ],
)
def test_a_refused_calibration_publishes_no_points(harness, change, reason):
    h = harness(_masked_providers(), geometry_enabled=True)
    h.wait_matched()
    info = h.camera_info(1)
    for name, value in change.items():
        setattr(info, name, value)
    h.hold_infos(depth_info=info)
    h.freeze_pair_clock()
    h.publish_depth(_depth_msg(1))
    h.publish_image(1)
    _, det = _published_object(h)
    assert h.node.no_depth[reason] == 1
    assert det.points.width == 0 and len(det.points.data) == 0
    # The mask still scores, with no depth term.
    assert 0.0 < det.view_quality < 0.75


def test_a_mask_without_geometry_stays_out_of_the_message(harness):
    h = harness(_masked_providers())
    h.wait_matched()
    h.publish_frame(1)
    _, det = _published_object(h)
    assert det.points.width == 0 and len(det.points.data) == 0 and det.view_quality == 0.0


# --- 9. the bounded services (ADR-0015) ----------------------------------


NO_OBJECTS = FakeObjectDetector([], classes=CLASSES)
"""So the stream never calls the embedder, and only the service job does."""
SECOND_NS = 1_000_000_000
BOX = (10.0, 10.0, 30.0, 30.0)


def _service_harness(harness, embedder=None, **overrides) -> Harness:
    return harness(
        fake_providers_with_embedder(objects=NO_OBJECTS, embedder=embedder), object_embeddings_enabled=True, **overrides
    )


def _text_request(*texts: str) -> EmbedText.Request:
    return EmbedText.Request(texts=list(texts or ("a red mug",)))


def _crop_request(
    sec: int = 1,
    *,
    nanosec: int = 0,
    frame: str = FRAME,
    width: int = WIDTH,
    height: int = HEIGHT,
    timeout_s: float = 5.0,
):
    return EmbedCrop.Request(
        header=Header(stamp=Time(sec=sec, nanosec=nanosec), frame_id=frame),
        image_width=width,
        image_height=height,
        bbox=box_to_bbox(BOX),
        timeout_s=timeout_s,
    )


def _slot_free(h: Harness) -> bool:
    return h.node._jobs._job is None


def _box_mask() -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[10:30, 10:30] = True
    return mask


def test_embed_text_answers_before_the_first_image(harness):
    embedder = FakeEmbedder()
    h = _service_harness(harness, embedder)
    texts = ["a red mug", "a chair"]
    response = h.answer(h.call(EmbedText, _text_request(*texts)))
    assert h.node.frames_received == 0
    assert response.success and response.message == ""
    assert response.embedding_model == embedder.model_name and response.dim == embedder.dim
    np.testing.assert_allclose(vectors_from_response(response), embedder.embed_texts(texts), rtol=1e-6)


def test_embed_text_wakes_a_worker_on_the_depth_buffer(harness):
    h = _service_harness(harness, geometry_enabled=True)
    assert isinstance(h.node._mailbox, RgbdBuffer)
    assert h.answer(h.call(EmbedText, _text_request())).success


@pytest.mark.parametrize(
    ("texts", "reason"),
    [
        ([], "empty"),
        (["mug"] * 17, "at most 16"),
        (["mug", "   "], "text 1 is blank"),
        (["x" * 513], "more than 512"),
    ],
)
def test_embed_text_refuses_a_bad_request_before_the_worker(harness, texts, reason):
    h = _service_harness(harness)
    response = h.answer(h.call(EmbedText, EmbedText.Request(texts=texts)))
    assert not response.success and response.message.startswith(requests.INVALID) and reason in response.message
    assert response.embedding_model == "" and response.dim == 0 and len(response.vectors) == 0
    assert h.node.service_jobs == 0


def test_embed_text_takes_a_request_at_the_caps(harness):
    h = _service_harness(harness)
    response = h.answer(h.call(EmbedText, EmbedText.Request(texts=["x" * 512] * 16)))
    assert response.success and vectors_from_response(response).shape == (16, response.dim)


def test_embed_crop_refuses_a_bad_request_before_the_worker(harness):
    h = _service_harness(harness)
    request = _crop_request()
    request.bbox.center.position.x = float("nan")
    response = h.answer(h.call(EmbedCrop, request))
    assert not response.success and response.error_code == requests.INVALID
    assert h.node.service_jobs == 0


@pytest.mark.parametrize(
    ("x", "size_x"),
    [(1.7e308, 1.7e308), (0.895e308, 1.79e308)],
    ids=["corner", "padded crop"],
)
def test_embed_crop_refuses_a_finite_box_that_overflows_and_keeps_serving(harness, x, size_x):
    embedder = BlockingEmbedder()
    embedder.release.set()
    h = _service_harness(harness, embedder)
    h.set_clock(SECOND_NS + 100_000_000)
    h.wait_matched()
    h.publish_image(1)
    request = _crop_request()
    # Each field is finite. A corner, or the padded crop around the box, is not.
    request.bbox.center.position.x, request.bbox.size_x = x, size_x
    response = h.answer(h.call(EmbedCrop, request))
    assert not response.success and response.error_code == requests.INVALID and "bbox" in response.message
    assert h.node.service_jobs == 0 and embedder.calls == 0
    # The executor still runs: the next request gets its answer.
    assert h.answer(h.call(EmbedCrop, _crop_request())).success


def test_embed_crop_refuses_a_stamp_that_is_not_a_valid_time_before_the_image_match(harness):
    embedder = BlockingEmbedder()
    embedder.release.set()
    h = _service_harness(harness, embedder)
    h.set_clock(SECOND_NS + 100_000_000)
    h.wait_matched()
    h.publish_image(1)
    # 0 s and 1e9 ns would add up to the current image's 1 s.
    response = h.answer(h.call(EmbedCrop, _crop_request(0, nanosec=1_000_000_000)))
    assert not response.success and response.error_code == requests.INVALID and "header.stamp" in response.message
    assert h.node.service_jobs == 0 and embedder.calls == 0
    assert len(response.embedding) == 0 and response.header.frame_id == ""


def test_a_second_request_gets_busy_and_nothing_queues(harness, monkeypatch):
    monkeypatch.setattr(requests, "EMBED_TEXT_DEADLINE_S", 30.0)
    embedder = BlockingEmbedder()
    h = _service_harness(harness, embedder)
    try:
        first = h.call(EmbedText, _text_request())
        assert h.spin_until(embedder.entered.is_set)
        busy_text = h.answer(h.call(EmbedText, _text_request()))
        assert not busy_text.success and busy_text.message.startswith(requests.BUSY)
        busy_crop = h.answer(h.call(EmbedCrop, _crop_request()))
        assert not busy_crop.success and busy_crop.error_code == requests.BUSY
        assert h.node.service_busy == 2
    finally:
        embedder.release.set()
    assert h.answer(first).success
    assert h.spin_until(lambda: _slot_free(h))
    # Neither refused request ran later.
    assert embedder.calls == 1 and h.node.service_jobs == 1
    assert h.answer(h.call(EmbedText, _text_request())).success
    assert embedder.calls == 2


def test_embed_text_stops_at_its_deadline_and_discards_the_late_result(harness, monkeypatch):
    monkeypatch.setattr(requests, "EMBED_TEXT_DEADLINE_S", 0.2)
    embedder = BlockingEmbedder()
    h = _service_harness(harness, embedder)
    try:
        response = h.answer(h.call(EmbedText, _text_request()))
        assert not response.success and response.message.startswith(requests.TIMEOUT)
        assert response.embedding_model == "" and len(response.vectors) == 0
        # The model still runs, so the slot stays taken.
        assert h.answer(h.call(EmbedText, _text_request())).message.startswith(requests.BUSY)
    finally:
        embedder.release.set()
    assert h.spin_until(lambda: h.node.late_results == 1)
    assert h.spin_until(lambda: _slot_free(h))
    monkeypatch.setattr(requests, "EMBED_TEXT_DEADLINE_S", 30.0)
    assert h.answer(h.call(EmbedText, _text_request())).success


def test_a_paused_clock_cannot_make_a_crop_request_hang(harness):
    # Sim time without a /clock: the node's clock stands still at 0.
    embedder = BlockingEmbedder()
    h = _service_harness(harness, embedder, use_sim_time=True)
    h.wait_matched()
    msg = h.image(1)
    msg.header.stamp = Time(nanosec=1000)
    h.publish_msg(msg)
    try:
        response = h.answer(h.call(EmbedCrop, _crop_request(0, nanosec=1000, timeout_s=0.2)))
        assert embedder.entered.is_set()
        assert not response.success and response.error_code == requests.TIMEOUT
        assert len(response.embedding) == 0 and len(response.crop.data) == 0
    finally:
        embedder.release.set()
    assert h.spin_until(lambda: h.node.late_results == 1)
    assert h.spin_until(lambda: _slot_free(h))
    assert h.answer(h.call(EmbedCrop, _crop_request(0, nanosec=1000))).success


def test_a_job_given_up_before_the_worker_reaches_it_runs_no_model(harness, monkeypatch):
    monkeypatch.setattr(requests, "EMBED_TEXT_DEADLINE_S", 0.2)
    embedder, pose = BlockingEmbedder(), BlockingPose()
    h = harness(
        fake_providers_with_embedder(objects=NO_OBJECTS, embedder=embedder, pose=pose), object_embeddings_enabled=True
    )
    h.wait_matched()
    h.publish_frame(1)
    assert h.spin_until(pose.entered.is_set)
    try:
        response = h.answer(h.call(EmbedText, _text_request()))
        assert response.message.startswith(requests.TIMEOUT)
        # The worker never took it, so the slot is free again.
        assert _slot_free(h)
    finally:
        pose.release.set()
    assert h.spin_until(lambda: 1 in h.stamps("people reliable"))
    assert embedder.calls == 0 and h.node.late_results == 0


def test_a_job_that_ends_after_its_deadline_counts_as_late_before_its_caller_gives_up(harness):
    now = [100.0]

    class SlowEmbedder(FakeEmbedder):
        def embed_texts(self, texts):
            now[0] = 100.2  # the model ran past the deadline
            return super().embed_texts(texts)

    h = _service_harness(harness, SlowEmbedder())
    job = requests.ServiceJob(_text_request(), 100.1, embed_text_failure, clock=lambda: now[0])
    # The worker's own loop never sees this job. The caller has not woken yet.
    h.node._run_job(job)
    assert h.node.service_jobs == 1 and h.node.late_results == 1
    assert job.wait() is None


def test_closing_the_slot_answers_the_running_job_and_later_requests(harness, monkeypatch):
    monkeypatch.setattr(requests, "EMBED_TEXT_DEADLINE_S", 30.0)
    embedder = BlockingEmbedder()
    h = _service_harness(harness, embedder)
    try:
        future = h.call(EmbedText, _text_request())
        assert h.spin_until(embedder.entered.is_set)
        h.node._jobs.close()
        response = h.answer(future)
        assert not response.success and response.message.startswith(requests.MODEL_UNAVAILABLE)
        assert "shutting down" in response.message
        refused = h.answer(h.call(EmbedCrop, _crop_request()))
        assert refused.error_code == requests.MODEL_UNAVAILABLE and "shutting down" in refused.message
    finally:
        embedder.release.set()


def test_closing_the_node_stops_with_a_job_running(harness, monkeypatch):
    monkeypatch.setattr(requests, "EMBED_TEXT_DEADLINE_S", 30.0)
    embedder = BlockingEmbedder()
    h = _service_harness(harness, embedder)
    h.call(EmbedText, _text_request())
    assert h.spin_until(embedder.entered.is_set)
    job = h.node._jobs._job
    release = threading.Timer(0.2, embedder.release.set)
    release.start()
    try:
        assert h.close()
    finally:
        release.join()
    assert job.wait().message.startswith(requests.MODEL_UNAVAILABLE)
    # The model's answer came after the refusal, and was dropped.
    assert h.node.late_results == 1


@pytest.mark.parametrize(
    ("fields", "reason"),
    [
        ({"nanosec": 5_000_001}, "from the current image"),
        ({"frame": "other_frame"}, "frame"),
        ({"width": WIDTH + 2}, "image is"),
        ({"height": HEIGHT + 2}, "image is"),
    ],
)
def test_embed_crop_refuses_a_request_that_does_not_name_the_current_image(harness, fields, reason):
    embedder = BlockingEmbedder()
    embedder.release.set()
    h = _service_harness(harness, embedder)
    h.set_clock(SECOND_NS + 100_000_000)
    h.wait_matched()
    h.publish_image(1)
    response = h.answer(h.call(EmbedCrop, _crop_request(**fields)))
    assert not response.success and response.error_code == requests.STALE_IMAGE and reason in response.message
    assert embedder.calls == 0
    assert len(response.embedding) == 0 and len(response.crop.data) == 0 and response.header.frame_id == ""


def test_embed_crop_never_crops_a_later_view_with_an_old_box(harness):
    embedder = BlockingEmbedder()
    embedder.release.set()
    h = _service_harness(harness, embedder)
    h.set_clock(2 * SECOND_NS)
    h.wait_matched()
    h.publish_image(1)
    h.publish_image(2)
    response = h.answer(h.call(EmbedCrop, _crop_request(1)))
    assert response.error_code == requests.STALE_IMAGE and embedder.calls == 0


@pytest.mark.parametrize(
    ("now_ns", "code"),
    [
        (SECOND_NS + requests.MAX_IMAGE_AGE_NS, ""),
        (SECOND_NS + requests.MAX_IMAGE_AGE_NS + 1, requests.STALE_IMAGE),
        (SECOND_NS - requests.MAX_FUTURE_NS - 1, requests.STALE_IMAGE),
    ],
)
def test_embed_crop_holds_the_image_age_in_ros_time(harness, now_ns, code):
    h = _service_harness(harness)
    h.set_clock(now_ns)
    h.wait_matched()
    h.publish_image(1)
    assert h.answer(h.call(EmbedCrop, _crop_request())).error_code == code


def test_embed_crop_without_an_image_answers_no_image_at_once(harness):
    h = _service_harness(harness)
    response = h.answer(h.call(EmbedCrop, _crop_request()))
    assert not response.success and response.error_code == requests.NO_IMAGE
    assert h.node.frames_received == 0


def test_embed_crop_answers_from_the_current_image(harness):
    embedder = FakeEmbedder()
    h = _service_harness(harness, embedder)
    h.set_clock(SECOND_NS + 100_000_000)
    h.wait_matched()
    msg = h.image(1)
    # A pattern, so the crop's pixels say where they came from.
    array = np.random.default_rng(1).integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    msg.data = array.tobytes()
    h.publish_msg(msg)
    # Within 5 ms, so the response shows the image's own stamp and not this one.
    response = h.answer(h.call(EmbedCrop, _crop_request(nanosec=4_000_000)))
    assert response.success and response.error_code == "" and response.message == ""
    assert response.header.stamp == Time(sec=1) and response.header.frame_id == FRAME
    assert response.crop.header == response.header and response.crop.format == "jpeg"
    pixels = embeddings.padded_crop(array, np.array(BOX))
    assert pixels.shape == (26, 26, 3)
    decoded = cv2.imdecode(np.frombuffer(bytes(response.crop.data), np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == pixels.shape
    # The same padded pixels give the crop and the vector.
    assert np.abs(decoded.astype(int) - pixels.astype(int)).mean() < np.abs(pixels.astype(int) - 128).mean()
    assert response.embedding_model == embedder.model_name
    np.testing.assert_allclose(response.embedding, embedder.embed_images([pixels])[0], rtol=1e-6)
    expected = view_quality(_box_mask(), np.array(BOX), frame_laplacian(array), 0.0)
    assert response.view_quality == pytest.approx(expected)


def test_embed_crop_scores_depth_only_for_a_paired_current_image(harness):
    h = _service_harness(harness, geometry_enabled=True)
    h.wait_matched()
    h.hold_infos()
    h.freeze_pair_clock()
    h.set_clock(SECOND_NS + 100_000_000)
    h.publish_depth(_depth_msg(1, 1.5))
    h.publish_image(1)
    plain = np.full((HEIGHT, WIDTH, 3), 1, dtype=np.uint8)
    paired = h.answer(h.call(EmbedCrop, _crop_request()))
    expected = view_quality(_box_mask(), np.array(BOX), frame_laplacian(plain), 1.0)
    assert paired.success and paired.view_quality == pytest.approx(expected)
    # Image 2 has no depth of its own, and depth 1 is too far from it to count.
    assert h.spin_until(lambda: _slot_free(h))
    h.set_clock(2 * SECOND_NS + 100_000_000)
    h.publish_image(2)
    unpaired = h.answer(h.call(EmbedCrop, _crop_request(2)))
    plain = np.full((HEIGHT, WIDTH, 3), 2, dtype=np.uint8)
    expected = view_quality(_box_mask(), np.array(BOX), frame_laplacian(plain), 0.0)
    assert unpaired.success and unpaired.view_quality == pytest.approx(expected)
    assert paired.view_quality > unpaired.view_quality


def test_a_refused_depth_leaves_only_the_depth_term_out_of_a_crop(harness):
    h = _service_harness(harness, geometry_enabled=True)
    h.wait_matched()
    h.hold_infos()
    h.freeze_pair_clock()
    h.set_clock(SECOND_NS + 100_000_000)
    h.publish_depth(_depth_msg(1, 1.5, frame="other_frame"))
    h.publish_image(1)
    response = h.answer(h.call(EmbedCrop, _crop_request()))
    plain = np.full((HEIGHT, WIDTH, 3), 1, dtype=np.uint8)
    expected = view_quality(_box_mask(), np.array(BOX), frame_laplacian(plain), 0.0)
    assert response.success and response.view_quality == pytest.approx(expected)


def test_embed_crop_scores_no_depth_from_before_a_clock_reset(harness):
    h = _service_harness(harness, geometry_enabled=True)
    h.wait_matched()
    h.hold_infos(11)
    h.set_clock(11 * SECOND_NS + 100_000_000)
    # The last depth of the previous episode, at 10 s, then a colour image at 11 s.
    h.publish_depth(_depth_msg(10, 1.5))
    h.publish_image(11)
    assert h.spin_until(lambda: h.node.no_depth["unpaired"] >= 1)
    # /clock goes back, and a new colour image comes at 10 s. No new depth came.
    h.set_clock(10 * SECOND_NS + 100_000_000)
    h.publish_image(10)
    assert h.spin_until(lambda: h.node.no_depth["unpaired"] >= 2)
    response = h.answer(h.call(EmbedCrop, _crop_request(10)))
    plain = np.full((HEIGHT, WIDTH, 3), 10, dtype=np.uint8)
    expected = view_quality(_box_mask(), np.array(BOX), frame_laplacian(plain), 0.0)
    assert response.success and response.header.stamp == Time(sec=10)
    assert response.view_quality == pytest.approx(expected)


def test_a_service_failure_leaves_the_stream_alone(harness):
    h = _service_harness(harness, RaisingEmbedder())
    h.set_clock(SECOND_NS + 100_000_000)
    h.wait_matched()
    h.publish_frame(1)
    assert h.spin_until(lambda: 1 in h.stamps("people reliable") and 1 in h.stamps("objects reliable"))
    response = h.answer(h.call(EmbedCrop, _crop_request()))
    assert not response.success and response.error_code == requests.INFERENCE_FAILED
    assert "CUDA out of memory" in response.message and len(response.embedding) == 0
    assert h.spin_until(lambda: _slot_free(h))
    text = h.answer(h.call(EmbedText, _text_request()))
    assert not text.success and text.message.startswith(f"{requests.INFERENCE_FAILED}: ")
    assert "CUDA out of memory" in text.message
    assert h.node.frames_failed == 0 and h.node.consecutive_failures == 0 and h.node.fatal is None
    h.publish_frame(2)
    assert h.spin_until(lambda: 2 in h.stamps("people reliable") and 2 in h.stamps("objects reliable"))
    assert h.node.frames_failed == 0


def test_a_bad_vector_answers_inference_failed(harness):
    class ZeroEmbedder(FakeEmbedder):
        def embed_images(self, images):
            return np.zeros((len(images), self.dim), dtype=np.float32)

    h = _service_harness(harness, ZeroEmbedder())
    h.set_clock(SECOND_NS + 100_000_000)
    h.wait_matched()
    h.publish_image(1)
    response = h.answer(h.call(EmbedCrop, _crop_request()))
    assert not response.success and response.error_code == requests.INFERENCE_FAILED and "zero" in response.message


def test_the_newest_frame_runs_before_the_next_job(harness, monkeypatch):
    monkeypatch.setattr(requests, "EMBED_TEXT_DEADLINE_S", 30.0)
    embedder, pose = BlockingEmbedder(), BlockingPose()
    h = harness(
        fake_providers_with_embedder(objects=NO_OBJECTS, embedder=embedder, pose=pose), object_embeddings_enabled=True
    )
    h.wait_matched()
    try:
        first = h.call(EmbedText, _text_request())
        assert h.spin_until(embedder.entered.is_set)
        # The input callbacks keep running while the model runs.
        h.publish_frame(7)
        h.publish_frame(8)
        assert h.node.frames_received == 2
        embedder.release.set()
        assert h.answer(first).success
        assert h.spin_until(pose.entered.is_set)
        # The job has its answer, and the slot stays taken through the newest frame.
        assert h.answer(h.call(EmbedText, _text_request())).message.startswith(requests.BUSY)
    finally:
        embedder.release.set()
        pose.release.set()
    assert h.spin_until(lambda: 8 in h.stamps("people reliable"))
    assert 7 not in h.stamps("people reliable")
    assert h.spin_until(lambda: _slot_free(h))
    assert h.answer(h.call(EmbedText, _text_request())).success


def test_only_the_newest_image_stays_for_a_crop(harness):
    h = _service_harness(harness)
    h.wait_matched()
    h.publish_image(1)
    h.publish_image(2)
    colour, depth = h.node._current.read()
    assert colour.header.stamp.sec == 2 and depth is None


def test_no_decoded_image_is_kept_after_a_crop_answer(harness):
    kept = []

    class Recording(FakeEmbedder):
        def embed_images(self, images):
            kept.extend(weakref.ref(i) for i in images)
            return super().embed_images(images)

    h = _service_harness(harness, Recording())
    h.set_clock(SECOND_NS + 100_000_000)
    h.wait_matched()
    h.publish_image(1)
    assert h.answer(h.call(EmbedCrop, _crop_request())).success

    def dropped() -> bool:
        gc.collect()
        return all(ref() is None for ref in kept)

    assert kept and h.spin_until(dropped), "the node still holds a crop"


def test_with_embeddings_off_no_image_stays_and_the_services_answer_disabled(harness):
    h = harness()
    assert h.node._current is None
    text = h.answer(h.call(EmbedText, _text_request()))
    assert not text.success and text.message.startswith(requests.DISABLED)
    response = h.answer(h.call(EmbedCrop, _crop_request()))
    assert not response.success and response.error_code == requests.DISABLED
    assert h.node.service_jobs == 0


@pytest.mark.parametrize("enabled", [False, True])
def test_describe_crop_answers_disabled_with_no_worker(harness, enabled):
    h = _service_harness(harness) if enabled else harness()
    response = h.answer(h.call(DescribeCrop, DescribeCrop.Request()))
    assert not response.success and response.error_code == requests.DISABLED and response.message
    assert len(response.message) <= MAX_MESSAGE_CHARS
    assert response.caption == "" and response.attributes_json == "" and response.printed_text == ""
    assert response.caption_model == "" and response.prompt_version == ""
    assert h.node.service_jobs == 0


# --- DetectObjects ---------------------------------------------------------


class OnDemandDetector(FakeObjectDetector):
    """An on-demand detector that a test can stop or fail in either of its two steps.

    It records each list it prepares, and the first byte of each image it sees,
    which is the image's stamp.
    """

    def __init__(self, objects=(), classes=("cup",)) -> None:
        super().__init__(objects, classes=classes)
        self.prepared: list[list[str]] = []
        self.images: list[int] = []
        self.block_prepare = self.block_detect = False
        self.fail_prepare = self.fail_detect = False
        self.during_detect = None
        self.entered = threading.Event()
        self.release = threading.Event()

    def _block(self) -> None:
        self.entered.set()
        self.release.wait(DEADLINE_S)

    def set_classes(self, classes) -> None:
        self.prepared.append(list(classes))
        if self.block_prepare:
            self._block()
        if self.fail_prepare:
            self._classes = []
            raise RuntimeError("the text encoder failed")
        super().set_classes(classes)

    def detect(self, image):
        self.images.append(int(image[0, 0, 0]))
        if self.block_detect:
            self._block()
        if self.fail_detect:
            raise RuntimeError("the detector failed")
        if self.during_detect is not None:
            self.during_detect()
        return super().detect(image)


STREAM_OBJECTS = FakeObjectDetector(
    [Detection(box=np.array(LEFT), label="chair", score=0.8, track_id=5)], classes=CLASSES
)
CUP = Detection(box=np.array(BOX), label="cup", score=0.8, track_id=9)
BIG_CUP = Detection(box=np.array(LEFT), label="cup", score=0.8, track_id=9)


def _detect_request(*classes: str, max_image_age_s: float = 0.3, timeout_s: float = 4.0) -> DetectObjects.Request:
    return DetectObjects.Request(
        classes=list(classes or ("cup",)), max_image_age_s=max_image_age_s, timeout_s=timeout_s
    )


def _on_demand_harness(harness, on_demand=None, *, objects=NO_OBJECTS, sec=1, **overrides) -> Harness:
    providers = dataclasses.replace(
        fake_providers_with_embedder(objects=objects), on_demand=on_demand or OnDemandDetector([CUP])
    )
    h = harness(providers, on_demand_enabled=True, **overrides)
    h.set_clock(sec * SECOND_NS + 100_000_000)
    h.wait_matched()
    return h


def _assert_failed(response, code: str) -> None:
    assert not response.success and response.error_code == code and response.message
    assert len(response.message) <= MAX_MESSAGE_CHARS
    assert len(response.objects.detections) == 0 and response.objects.stream_id == ""


def test_detect_objects_answers_disabled_by_default_with_no_worker(harness):
    h = harness()
    assert h.node._current is None
    _assert_failed(h.answer(h.call(DetectObjects, _detect_request())), requests.DISABLED)
    assert h.node.service_jobs == 0


def test_on_demand_needs_its_detector_at_start(ctx):
    overrides = [Parameter("classes", value=CLASSES), Parameter("on_demand_enabled", value=True)]
    with pytest.raises(ValueError, match="on-demand detector"):
        _fake_node(ctx, parameter_overrides=overrides)


def test_on_demand_alone_keeps_the_embedding_services_disabled(harness):
    h = _on_demand_harness(harness)
    assert h.answer(h.call(EmbedText, _text_request())).message.startswith(requests.DISABLED)
    assert h.answer(h.call(EmbedCrop, _crop_request())).error_code == requests.DISABLED
    assert h.node.service_jobs == 0


@pytest.mark.parametrize(
    ("request_", "reason"),
    [
        (DetectObjects.Request(classes=[], max_image_age_s=0.3, timeout_s=4.0), "empty"),
        (_detect_request(*[f"class {i}" for i in range(33)]), "at most 32"),
        (_detect_request("cup", "cup"), "duplicates"),
        (_detect_request(max_image_age_s=0.0), "max_image_age_s"),
        (_detect_request(timeout_s=5.5), "timeout_s"),
    ],
)
def test_detect_objects_refuses_a_bad_request_before_the_worker(harness, request_, reason):
    detector = OnDemandDetector()
    h = _on_demand_harness(harness, detector)
    response = h.answer(h.call(DetectObjects, request_))
    _assert_failed(response, requests.INVALID)
    assert reason in response.message
    assert h.node.service_jobs == 0 and detector.prepared == [] and detector.images == []


def test_detect_objects_without_an_image_answers_no_image(harness):
    detector = OnDemandDetector([CUP])
    h = _on_demand_harness(harness, detector)
    _assert_failed(h.answer(h.call(DetectObjects, _detect_request())), requests.NO_IMAGE)
    assert detector.images == []


def test_detect_objects_without_a_camera_info_answers_no_image(harness):
    detector = OnDemandDetector([CUP])
    h = _on_demand_harness(harness, detector)
    h.publish_image(1)
    _assert_failed(h.answer(h.call(DetectObjects, _detect_request())), requests.NO_IMAGE)
    assert detector.images == []


def test_detect_objects_requiring_depth_refuses_an_image_with_no_pair(harness):
    detector = OnDemandDetector([CUP])
    h = _on_demand_harness(harness, detector, geometry_enabled=True, require_depth=True)
    h.hold_infos()
    h.publish_image(1)
    response = h.answer(h.call(DetectObjects, _detect_request()))
    _assert_failed(response, requests.NO_IMAGE)
    assert "depth" in response.message
    assert detector.images == []


@pytest.mark.parametrize(
    ("now_ns", "success"),
    [
        (SECOND_NS + 250_000_000, True),
        (SECOND_NS + 250_000_001, False),
        (SECOND_NS - 250_000_001, False),  # a stamp ahead of the clock, as after a sim restart
    ],
)
def test_detect_objects_holds_the_image_age_in_ros_time(harness, now_ns, success):
    detector = OnDemandDetector([CUP])
    h = _on_demand_harness(harness, detector)
    h.publish_frame(1)
    h.set_clock(now_ns)
    # 0.25 is exact in the request's float32, so the limit falls on this nanosecond.
    response = h.answer(h.call(DetectObjects, _detect_request(max_image_age_s=0.25)))
    if success:
        assert response.success
    else:
        _assert_failed(response, requests.STALE_IMAGE)
        assert detector.images == []


def test_an_image_that_went_old_during_detection_is_not_labelled_current(harness):
    detector = OnDemandDetector([CUP])
    h = _on_demand_harness(harness, detector)
    h.publish_frame(1)
    detector.during_detect = lambda: h.set_clock(3 * SECOND_NS)
    response = h.answer(h.call(DetectObjects, _detect_request()))
    _assert_failed(response, requests.STALE_IMAGE)
    assert "after detection" in response.message
    assert detector.images == [1]


def test_detect_objects_returns_its_own_array_and_publishes_nothing(harness):
    h = _on_demand_harness(
        harness,
        OnDemandDetector([BIG_CUP]),
        objects=STREAM_OBJECTS,
        object_embeddings_enabled=True,
        object_crops_enabled=True,
    )
    h.publish_frame(1)
    assert h.spin_until(lambda: h.stamps("objects reliable") == [1] and _slot_free(h))
    published = h.got["objects reliable"][0]
    response = h.answer(h.call(DetectObjects, _detect_request()))
    assert response.success and response.error_code == "" and response.message == ""
    array = response.objects
    assert array.header.stamp == Time(sec=1) and array.header.frame_id == FRAME
    assert array.camera_info.header == array.header
    assert list(array.classes) == ["cup"]
    assert array.stream_id == published.stream_id == h.node.pipeline.stream_id
    [det] = array.detections
    assert det.label == "cup" and det.track_id == 0
    assert len(det.embedding) == FakeEmbedder().dim and det.embedding_model == FakeEmbedder().model_name
    assert det.crop.format == "jpeg" and len(det.crop.data) > 0
    check_object_detection_array(array)
    # The answer went to the caller alone.
    h.spin_for(0.3)
    assert h.stamps("objects reliable") == [1] and h.stamps("people reliable") == [1]
    assert h.node.pipeline.processed_images == 1


def test_detect_objects_with_nothing_found_is_a_success(harness):
    h = _on_demand_harness(harness, OnDemandDetector([]))
    h.publish_frame(1)
    response = h.answer(h.call(DetectObjects, _detect_request()))
    assert response.success and len(response.objects.detections) == 0
    assert list(response.objects.classes) == ["cup"] and response.objects.stream_id


def test_detect_objects_returns_at_most_the_cap(harness):
    # Scores rise with the index and all pass the floor, so the cap must keep the last ones.
    many = [Detection(box=np.array(LEFT), label="cup", score=0.5 + i / 100.0) for i in range(25)]
    h = _on_demand_harness(harness, OnDemandDetector(many))
    h.publish_frame(1)
    response = h.answer(h.call(DetectObjects, _detect_request()))
    assert response.success
    scores = [round(float(d.score), 2) for d in response.objects.detections]
    assert scores == [round(0.5 + (24 - i) / 100.0, 2) for i in range(20)]


def _masked_cup() -> Detection:
    return Detection(box=np.array(MASKED_BOX), label="cup", score=0.8, mask=box_mask(HEIGHT, WIDTH, MASKED_BOX))


def test_detect_objects_fills_points_from_a_matching_pair(harness):
    h = _on_demand_harness(harness, OnDemandDetector([_masked_cup()]), geometry_enabled=True)
    h.hold_infos()
    h.freeze_pair_clock()
    h.publish_depth(_depth_msg(1, 1.5))
    h.publish_image(1)
    response = h.answer(h.call(DetectObjects, _detect_request()))
    assert response.success
    [det] = response.objects.detections
    assert det.points.header == response.objects.header
    assert det.points.width == 28 * 28 and det.view_quality > 0.0
    check_object_detection_array(response.objects)


def test_detect_objects_prefers_a_current_pair_and_falls_back_to_the_newest_image(harness):
    h = _on_demand_harness(harness, OnDemandDetector([_masked_cup()]), geometry_enabled=True)
    h.hold_infos()
    h.freeze_pair_clock()
    h.publish_depth(_depth_msg(1, 1.5))
    h.publish_image(1)
    # A newer colour image with no depth of its own, while the pair is still current.
    newer = h.image(1)
    newer.header.stamp = Time(sec=1, nanosec=50_000_000)
    h.publish_msg(newer)
    paired = h.answer(h.call(DetectObjects, _detect_request()))
    assert paired.objects.header.stamp == Time(sec=1) and paired.objects.detections[0].points.width > 0
    assert h.spin_until(lambda: _slot_free(h))
    # The pair has gone old, so the newest image answers, with no points.
    h.set_clock(2 * SECOND_NS + 100_000_000)
    h.publish_image(2)
    plain = h.answer(h.call(DetectObjects, _detect_request()))
    assert plain.objects.header.stamp == Time(sec=2) and plain.objects.detections[0].points.width == 0


def test_detect_objects_requiring_depth_refuses_a_depth_from_before_a_clock_reset(harness):
    detector = OnDemandDetector([_masked_cup()])
    h = _on_demand_harness(harness, detector, geometry_enabled=True, require_depth=True, sec=11)
    h.hold_infos(11)
    # The last depth of the previous episode, at 10 s, then a colour image at 11 s.
    h.publish_depth(_depth_msg(10, 2.0))
    h.publish_image(11)
    assert h.spin_until(lambda: h.node.skipped["unpaired"] >= 1)
    # /clock goes back, and a new colour image comes at 10 s. No new depth came.
    h.set_clock(10 * SECOND_NS + 100_000_000)
    h.publish_image(10)
    assert h.spin_until(lambda: h.node.skipped["unpaired"] >= 2)
    response = h.answer(h.call(DetectObjects, _detect_request()))
    _assert_failed(response, requests.NO_IMAGE)
    assert "depth" in response.message
    assert detector.images == []


def _clock_ticker(h: Harness):
    """Return tick(ns), which publishes ns on /clock and spins until the node's ROS clock reads it.

    The node must run on sim time. A jump back is handled after the clock is
    set, so a test waits on clock_jumps before it relies on the clear.
    """
    pub = h.helper.create_publisher(ClockMsg, "/clock", 10)
    assert h.spin_until(lambda: pub.get_subscription_count() >= 1), "the node does not subscribe to /clock"

    def tick(ns: int) -> None:
        pub.publish(ClockMsg(clock=Time(sec=ns // SECOND_NS, nanosec=ns % SECOND_NS)))
        assert h.spin_until(lambda: h.node.get_clock().now().nanoseconds == ns), f"/clock did not reach {ns}"

    return tick


def _sim_services_harness(harness) -> tuple[Harness, Callable[[int], None]]:
    """EmbedCrop and DetectObjects with depth, on sim time, with a /clock that the test moves."""
    providers = dataclasses.replace(
        fake_providers_with_embedder(objects=NO_OBJECTS), on_demand=OnDemandDetector([_masked_cup()])
    )
    h = harness(
        providers, object_embeddings_enabled=True, on_demand_enabled=True, geometry_enabled=True, use_sim_time=True
    )
    h.wait_matched()
    h.hold_infos(10)
    return h, _clock_ticker(h)


def test_a_clock_jump_back_leaves_the_services_no_image_from_before_it(harness):
    h, tick = _sim_services_harness(harness)
    h.node._current._clock = lambda: 0.0
    # The previous run's last pair, at 10 s, current by its stamps and by its arrivals.
    tick(10 * SECOND_NS + 100_000_000)
    h.publish_depth(_depth_msg(10, 2.0))
    h.publish_image(10)
    assert h.node._current.read_pair() is not None
    # The sim restarts and /clock comes back up to the old stamps. The camera sends nothing.
    tick(0)
    assert h.spin_until(lambda: h.node.clock_jumps == 1)
    tick(10 * SECOND_NS + 100_000_000)
    _assert_failed(h.answer(h.call(DetectObjects, _detect_request())), requests.NO_IMAGE)
    assert h.spin_until(lambda: _slot_free(h))
    crop = h.answer(h.call(EmbedCrop, _crop_request(10)))
    assert not crop.success and crop.error_code == requests.NO_IMAGE
    assert h.node.clock_jumps == 1


@pytest.mark.parametrize("clock_back", [True, False], ids=["clock-back", "clock-on"])
def test_a_clock_jump_back_keeps_an_old_depth_from_the_services_new_image(harness, clock_back):
    h, tick = _sim_services_harness(harness)
    # Every arrival at once, so only the jump can keep the old depth from pairing.
    h.node._current._clock = lambda: 0.0
    tick((11 if clock_back else 10) * SECOND_NS + 100_000_000)
    # The last depth of the run, at 10 s, then a colour image at 11 s.
    h.publish_depth(_depth_msg(10, 2.0))
    h.publish_image(11)
    # The sim restarts: /clock comes back to 10.1 s, and a new colour image has
    # the old depth's stamp. In the control the clock runs on instead.
    tick(10 * SECOND_NS + (100_000_000 if clock_back else 150_000_000))
    assert h.spin_until(lambda: h.node.clock_jumps == (1 if clock_back else 0))
    h.publish_image(10)
    detect = h.answer(h.call(DetectObjects, _detect_request()))
    assert detect.success and detect.objects.header.stamp == Time(sec=10)
    assert (detect.objects.detections[0].points.width > 0) is not clock_back
    assert h.spin_until(lambda: _slot_free(h))
    crop = h.answer(h.call(EmbedCrop, _crop_request(10)))
    plain = np.full((HEIGHT, WIDTH, 3), 10, dtype=np.uint8)
    no_depth = view_quality(_box_mask(), np.array(BOX), frame_laplacian(plain), 0.0)
    assert crop.success
    assert (crop.view_quality == pytest.approx(no_depth)) is clock_back
    assert h.node.clock_jumps == (1 if clock_back else 0)


@pytest.mark.parametrize("clock_back", [True, False], ids=["clock-back", "clock-on"])
def test_a_clock_jump_back_drops_the_depth_the_stream_holds(harness, clock_back):
    h = harness(geometry_enabled=True, use_sim_time=True)
    h.wait_matched()
    h.hold_infos(10)
    tick = _clock_ticker(h)
    now = [100.0]
    h.node._mailbox._clock = lambda: now[0]
    tick((11 if clock_back else 10) * SECOND_NS + 100_000_000)
    # The last depth of the run, at 10 s, held for an image that has not come.
    h.publish_depth(_depth_msg(10))
    tick(10 * SECOND_NS + (100_000_000 if clock_back else 150_000_000))
    assert h.spin_until(lambda: h.node.clock_jumps == (1 if clock_back else 0))
    # Within the wait by the wall clock, a new image has the old depth's stamp.
    h.publish_image(10)
    now[0] += 0.1
    assert h.spin_until(lambda: len(h.got["objects reliable"]) == 1)
    assert h.node.depth_paired == (0 if clock_back else 1)
    assert h.node.no_depth["unpaired"] == (1 if clock_back else 0)


@pytest.mark.parametrize(
    ("held", "clock_back"),
    [("waiting", True), ("ready", True), ("newest", True), ("ready", False), ("newest", False)],
    ids=["waiting-clock-back", "ready-clock-back", "newest-clock-back", "ready-clock-on", "newest-clock-on"],
)
def test_a_clock_jump_back_drops_the_frame_the_worker_has_not_taken(harness, held, clock_back):
    pose = BlockingPose()
    geometry = held != "newest"
    h = harness(fake_providers(objects=_due_object(), pose=pose), geometry_enabled=geometry, use_sim_time=True)
    h.wait_matched()
    tick = _clock_ticker(h)
    tick(10 * SECOND_NS + 100_000_000)
    # The worker is busy with an earlier frame, and has not taken the next one.
    if geometry:
        h.hold_infos(10)
        h.publish_image(1)
    else:
        h.publish_frame(1)
    assert h.spin_until(pose.entered.is_set)
    now = [100.0]
    try:
        if geometry:
            # Stopped, so the image keeps waiting for a nearer depth.
            h.node._mailbox._clock = lambda: now[0]
        h.publish_image(10)
        if geometry:
            depth = _depth_msg(10, 2.0)
            if held == "waiting":
                # Just before the image, so the image holds it and waits for a nearer one.
                depth.header.stamp = Time(sec=9, nanosec=999_000_000)
            h.publish_depth(depth)
            with h.node._mailbox._cond:
                waiting = h.node._mailbox._colour
                assert (waiting is not None and waiting.candidate is not None) if held == "waiting" else waiting is None
                assert (h.node._mailbox._ready is not None) is (held == "ready")
        tick(0 if clock_back else 10 * SECOND_NS + 150_000_000)
        assert h.spin_until(lambda: h.node.clock_jumps == (1 if clock_back else 0))
        assert h.node.frames_processed == 0
    finally:
        pose.release.set()
    # Without the jump the worker goes on to the frame at 10 s. With it, nothing waits.
    processed = 1 if clock_back else 2
    assert h.spin_until(lambda: h.node.frames_processed == processed)
    h.spin_for(0.2)
    assert h.node.frames_processed == processed
    # A frame after the jump comes through.
    h.publish_image(11)
    now[0] += 1.0
    assert h.spin_until(lambda: 11 in h.stamps("objects reliable"))
    assert h.stamps("objects reliable") == ([1, 11] if clock_back else [1, 10, 11])
    if not clock_back and geometry:
        old = h.got["objects reliable"][1]
        assert old.detections[0].points.width > 0


@pytest.mark.parametrize(("back_to_ns", "kept"), [(0, False), (9 * SECOND_NS + 200_000_000, True)])
def test_the_services_keep_no_image_stamped_over_a_second_ahead_of_the_clock(harness, back_to_ns, kept):
    h, tick = _sim_services_harness(harness)
    tick(10 * SECOND_NS + 100_000_000)
    # The previous run's last capture, delivered only after the sim restarted.
    old_colour, old_depth = h.image(10), _depth_msg(10, 2.0)
    tick(back_to_ns)
    assert h.spin_until(lambda: h.node.clock_jumps == 1)
    now = [100.0]
    h.node._current._clock = lambda: now[0]
    h.publish_depth(old_depth)
    h.publish_msg(old_colour)
    assert (h.node._current.read_pair() is not None) is kept
    assert h.node.ahead_of_clock == (0 if kept else 2)
    # The camera sends nothing more, for as long as it takes /clock to reach the old stamps.
    now[0] += 1_000_000.0
    tick(10 * SECOND_NS + 100_000_000)
    detect = h.answer(h.call(DetectObjects, _detect_request()))
    assert h.spin_until(lambda: _slot_free(h))
    crop = h.answer(h.call(EmbedCrop, _crop_request(10)))
    if kept:
        assert detect.success and detect.objects.detections[0].points.width > 0
        assert crop.success
    else:
        _assert_failed(detect, requests.NO_IMAGE)
        assert not crop.success and crop.error_code == requests.NO_IMAGE


def test_on_system_time_the_services_keep_an_image_stamped_ahead_of_the_clock(harness):
    h = harness(fake_providers_with_embedder(objects=NO_OBJECTS), object_embeddings_enabled=True)
    h.wait_matched()
    ahead = h.image(int(time.time()) + 60)
    h.publish_msg(ahead)
    assert h.node._current.read()[0] is not None
    assert h.node.ahead_of_clock == 0


@pytest.mark.parametrize(("depth_after_s", "paired"), [(0.0625, True), (0.07, False)])
def test_detect_objects_pairs_a_later_depth_only_within_pair_wait_s(harness, depth_after_s, paired):
    h = _on_demand_harness(
        harness, OnDemandDetector([_masked_cup()]), geometry_enabled=True, require_depth=True, pair_wait_s=0.0625
    )
    h.hold_infos()
    now = [100.0]
    h.node._current._clock = lambda: now[0]
    h.publish_image(1)
    assert h.spin_until(lambda: h.node._current.read()[0] is not None)
    now[0] += depth_after_s
    h.publish_depth(_depth_msg(1, 1.5))
    assert h.spin_until(lambda: h.node._current.read()[1] is not None)
    response = h.answer(h.call(DetectObjects, _detect_request()))
    if paired:
        assert response.success and response.objects.detections[0].points.width > 0
    else:
        _assert_failed(response, requests.NO_IMAGE)
        assert "depth" in response.message


def test_the_newest_image_is_chosen_after_new_classes_are_ready(harness):
    detector = OnDemandDetector([Detection(box=np.array(BOX), label="bowl", score=0.8)])
    detector.block_prepare = True
    h = _on_demand_harness(harness, detector)
    h.publish_frame(1)
    try:
        future = h.call(DetectObjects, _detect_request("bowl"))
        assert h.spin_until(detector.entered.is_set)
        # The preparation takes long enough to make the first image old.
        h.set_clock(2 * SECOND_NS + 100_000_000)
        h.publish_image(2)
    finally:
        detector.release.set()
    response = h.answer(future)
    assert response.success and response.objects.header.stamp == Time(sec=2)
    assert detector.prepared == [["bowl"]] and detector.images == [2]
    assert list(response.objects.classes) == ["bowl"]
    assert h.node.classes_prepared == 1


def test_the_same_list_prepares_nothing_and_a_new_order_prepares_again(harness):
    detector = OnDemandDetector([CUP], classes=("cup", "bowl"))
    h = _on_demand_harness(harness, detector)
    now = [100.0]
    h.node._class_gate = requests.ClassPrepareGate(clock=lambda: now[0])
    h.publish_frame(1)
    for _ in range(3):
        assert h.answer(h.call(DetectObjects, _detect_request("cup", "bowl"))).success
        assert h.spin_until(lambda: _slot_free(h))
    assert detector.prepared == [] and h.node.classes_prepared == 0
    assert h.answer(h.call(DetectObjects, _detect_request("bowl", "cup"))).success
    assert detector.prepared == [["bowl", "cup"]] and h.node.classes_prepared == 1


def test_a_new_list_within_the_period_is_refused_with_no_preparation(harness):
    detector = OnDemandDetector([CUP])
    h = _on_demand_harness(harness, detector)
    now = [100.0]
    h.node._class_gate = requests.ClassPrepareGate(2.0, clock=lambda: now[0])
    h.publish_frame(1)
    assert h.answer(h.call(DetectObjects, _detect_request("bowl"))).success
    assert h.spin_until(lambda: _slot_free(h))
    now[0] = 101.9
    busy_before = h.node.service_busy
    response = h.answer(h.call(DetectObjects, _detect_request("plate")))
    _assert_failed(response, requests.BUSY)
    assert "at most once every 2 s" in response.message
    assert h.node.service_busy == busy_before + 1
    assert detector.prepared == [["bowl"]]
    # The list in place still answers within the period.
    assert h.spin_until(lambda: _slot_free(h))
    assert h.answer(h.call(DetectObjects, _detect_request("bowl"))).success
    assert h.spin_until(lambda: _slot_free(h))
    now[0] = 102.0
    assert h.answer(h.call(DetectObjects, _detect_request("plate"))).success
    assert detector.prepared == [["bowl"], ["plate"]]


def test_a_second_request_gets_busy_across_the_services(harness):
    detector = OnDemandDetector([CUP])
    detector.block_detect = True
    h = _on_demand_harness(harness, detector, object_embeddings_enabled=True)
    h.publish_frame(1)
    try:
        first = h.call(DetectObjects, _detect_request())
        assert h.spin_until(detector.entered.is_set)
        _assert_failed(h.answer(h.call(DetectObjects, _detect_request())), requests.BUSY)
        assert h.answer(h.call(EmbedText, _text_request())).message.startswith(requests.BUSY)
        assert h.node.service_busy == 2
    finally:
        detector.release.set()
    assert h.answer(first).success
    assert h.spin_until(lambda: _slot_free(h))
    # Neither refused request ran later.
    assert detector.images == [1] and h.node.service_jobs == 1


def test_detect_objects_stops_at_its_deadline_and_the_stream_goes_on(harness):
    detector = OnDemandDetector([CUP])
    detector.block_detect = True
    h = _on_demand_harness(harness, detector, objects=STREAM_OBJECTS)
    h.publish_frame(1)
    assert h.spin_until(lambda: h.stamps("objects reliable") == [1] and _slot_free(h))
    try:
        future = h.call(DetectObjects, _detect_request(timeout_s=0.2))
        assert h.spin_until(detector.entered.is_set)
        h.publish_image(2)
        _assert_failed(h.answer(future), requests.TIMEOUT)
        # The model still runs, so the slot stays taken and image 2 waits.
        assert h.stamps("objects reliable") == [1]
    finally:
        detector.release.set()
    assert h.spin_until(lambda: h.node.late_results == 1)
    assert h.spin_until(lambda: 2 in h.stamps("objects reliable"))
    assert h.spin_until(lambda: _slot_free(h))
    assert [d.track_id for d in h.got["objects reliable"][-1].detections] == [
        d.track_id for d in h.got["objects reliable"][0].detections
    ]


def test_a_deadline_during_the_preparation_runs_no_detector(harness):
    detector = OnDemandDetector([CUP])
    detector.block_prepare = True
    h = _on_demand_harness(harness, detector)
    h.publish_frame(1)
    try:
        future = h.call(DetectObjects, _detect_request("bowl", timeout_s=0.2))
        assert h.spin_until(detector.entered.is_set)
        _assert_failed(h.answer(future), requests.TIMEOUT)
    finally:
        detector.release.set()
    assert h.spin_until(lambda: h.node.late_results == 1)
    assert h.spin_until(lambda: _slot_free(h))
    assert detector.images == []
    assert h.node.classes_prepared == 1 and h.node.pipeline.on_demand_classes == ["bowl"]


def test_a_failed_preparation_answers_model_unavailable_and_prepares_again(harness):
    detector = OnDemandDetector([CUP])
    detector.fail_prepare = True
    h = _on_demand_harness(harness, detector, objects=STREAM_OBJECTS)
    now = [100.0]
    h.node._class_gate = requests.ClassPrepareGate(2.0, clock=lambda: now[0])
    h.publish_frame(1)
    assert h.spin_until(lambda: h.stamps("objects reliable") == [1] and _slot_free(h))
    response = h.answer(h.call(DetectObjects, _detect_request("bowl")))
    _assert_failed(response, requests.MODEL_UNAVAILABLE)
    assert "text encoder" in response.message
    assert h.node.classes_prepared == 0 and detector.images == []
    assert h.spin_until(lambda: _slot_free(h))
    # A failure also takes a place in the period, and the list stays empty.
    _assert_failed(h.answer(h.call(DetectObjects, _detect_request("bowl"))), requests.BUSY)
    assert h.spin_until(lambda: _slot_free(h))
    now[0] = 102.0
    detector.fail_prepare = False
    assert h.answer(h.call(DetectObjects, _detect_request("cup"))).success
    assert detector.prepared == [["bowl"], ["cup"]]
    assert h.node.frames_failed == 0


def test_a_failed_detection_answers_inference_failed_and_leaves_the_stream_alone(harness):
    detector = OnDemandDetector([CUP])
    detector.fail_detect = True
    h = _on_demand_harness(harness, detector, objects=STREAM_OBJECTS)
    h.publish_frame(1)
    assert h.spin_until(lambda: h.stamps("objects reliable") == [1] and _slot_free(h))
    response = h.answer(h.call(DetectObjects, _detect_request()))
    _assert_failed(response, requests.INFERENCE_FAILED)
    assert "the detector failed" in response.message
    assert h.spin_until(lambda: _slot_free(h))
    assert h.node.frames_failed == 0
    h.publish_image(2)
    assert h.spin_until(lambda: 2 in h.stamps("objects reliable"))


def test_a_new_list_leaves_the_streaming_detector_alone(harness):
    stream = FakeObjectDetector(STREAM_OBJECTS.objects, classes=CLASSES)
    detector = OnDemandDetector([Detection(box=np.array(BOX), label="bowl", score=0.8)])
    h = _on_demand_harness(harness, detector, objects=stream)
    h.publish_frame(1)
    assert h.spin_until(lambda: h.stamps("objects reliable") == [1] and _slot_free(h))
    assert h.answer(h.call(DetectObjects, _detect_request("bowl"))).success
    assert h.spin_until(lambda: _slot_free(h))
    h.publish_image(2)
    assert h.spin_until(lambda: h.stamps("objects reliable") == [1, 2])
    first, second = h.got["objects reliable"]
    assert list(first.classes) == list(second.classes) == CLASSES
    assert [d.track_id for d in first.detections] == [d.track_id for d in second.detections]
    assert stream.classes == CLASSES


# --- the added work and the latency line ------------------------------------


ADDED_STREAM_WORK = (
    (pipeline_module, ("frame_laplacian", "lift_mask", "depth_cover", "view_quality", "resize_mask")),
    (embeddings, ("select_samples", "padded_crop", "encode_crop", "mask_color_shares")),
    (node_module, ("read_registered_depth",)),
)
ADDED_WORDS = {
    # Stages.
    "depth",
    "mask",
    "lift",
    "crop",
    "embed",
    "service",
    "prepare",
    # Counts.
    "paired",
    "no_depth",
    "unpaired",
    "sampled",
    "jobs",
    "busy",
    "late",
    "prepared",
}


@pytest.fixture
def added_work(monkeypatch) -> list[str]:
    """The names of the added functions that ran. Each one still does its work."""
    ran: list[str] = []

    def spied(name: str, real):
        def spy(*args, **kwargs):
            ran.append(name)
            return real(*args, **kwargs)

        return spy

    for module, names in ADDED_STREAM_WORK:
        for name in names:
            monkeypatch.setattr(module, name, spied(name, getattr(module, name)))
    return ran


def _due_object() -> FakeObjectDetector:
    """A masked object that the sampler takes on the first image, whose counter is 0."""
    detection = Detection(box=np.array(LEFT), label="chair", score=0.8, track_id=10, mask=box_mask(HEIGHT, WIDTH, LEFT))
    return FakeObjectDetector([detection], classes=CLASSES)


def _latency_words(h: Harness) -> set[str]:
    """Return the words of the latency line, once every processed image has its samples."""
    processed = h.node.frames_processed
    assert h.spin_until(lambda: h.node.latency._processed == processed)
    return set(h.node.latency.summary(1.0, h.node._latency_counts()).replace(",", " ").replace(";", " ").split())


def test_the_default_config_adds_no_work_to_the_stream(harness, added_work):
    # The detection is due for a sample and carries a mask, and a depth image is on the
    # wire, so only the flags stand between this image and all of the added work.
    h = harness(fake_providers(objects=_due_object()))
    assert _subscribed(h) == sorted([h.topics["image_topic"], h.topics["camera_info_topic"]])
    h.wait_matched()
    h.depth_pub.publish(_depth_msg(1))
    h.publish_frame(1)
    assert h.spin_until(lambda: h.stamps("objects reliable") == [1])
    array = h.got["objects reliable"][0]
    # The new fields are present at their empty values, and stream_id is still set.
    assert array.stream_id == h.node.pipeline.stream_id != ""
    (det,) = array.detections
    assert det.points.width == 0 and det.view_quality == 0.0
    assert len(det.crop.data) == 0 and len(det.embedding) == 0 and len(det.color_shares) == 0
    words = _latency_words(h)
    assert added_work == []
    assert h.node.frames_failed == 0
    assert h.node.depth_received == 0 and h.node.sampled == 0 and h.node._current is None
    assert h.node.pipeline._providers.embedder is None and h.node.pipeline._providers.on_demand is None
    assert not words & ADDED_WORDS


def test_with_every_addition_on_each_spy_and_the_latency_line_see_it(harness, added_work):
    # The control for the test above: a spy that cannot fire would pass that test too.
    bowl = dataclasses.replace(_masked_cup(), label="bowl")
    h = _on_demand_harness(
        harness,
        OnDemandDetector([bowl]),
        objects=_due_object(),
        geometry_enabled=True,
        object_crops_enabled=True,
        object_embeddings_enabled=True,
    )
    h.hold_infos()
    h.freeze_pair_clock()
    h.publish_depth(_depth_msg(1, 1.5))
    h.publish_image(1)
    assert h.spin_until(lambda: h.stamps("objects reliable") == [1])
    (det,) = h.got["objects reliable"][0].detections
    assert det.points.width > 0 and len(det.crop.data) > 0 and len(det.embedding) > 0
    assert set(added_work) == {name for _, names in ADDED_STREAM_WORK for name in names}

    response = h.answer(h.call(DetectObjects, _detect_request("bowl")))
    assert response.success and response.objects.detections[0].points.width > 0
    assert h.spin_until(lambda: _slot_free(h))
    assert ADDED_WORDS <= _latency_words(h)
    deltas = h.node.latency.last_deltas
    assert (deltas["depth"], deltas["paired"], deltas["no_depth"], deltas["unpaired"]) == (1, 1, 0, 0)
    assert (deltas["sampled"], deltas["jobs"], deltas["prepared"]) == (2, 1, 1)


@pytest.mark.parametrize("require_depth", [False, True])
def test_the_latency_line_counts_an_unpaired_image_whether_it_was_published_or_not(harness, require_depth):
    h = harness(geometry_enabled=True, require_depth=require_depth)
    h.wait_matched()
    h.hold_infos()
    h.publish_image(1)
    assert h.spin_until(lambda: h.node.skipped["unpaired"] + h.node.no_depth["unpaired"] == 1)
    assert h.node._latency_counts()["unpaired"] == 1


def test_the_crop_stage_shows_with_crops_alone_and_the_embed_stage_does_not(harness):
    h = harness(fake_providers(objects=_due_object()), object_crops_enabled=True)
    h.wait_matched()
    h.publish_frame(1)
    assert h.spin_until(lambda: h.stamps("objects reliable") == [1])
    words = _latency_words(h)
    assert {"mask", "crop", "sampled"} <= words
    assert not words & {"depth", "lift", "embed", "paired", "unpaired", "jobs", "prepared"}
    assert h.node.latency.last_deltas["sampled"] == 1


def test_a_failed_preparation_shows_its_time_and_counts_no_list(harness):
    detector = OnDemandDetector([CUP])
    detector.fail_prepare = True
    h = _on_demand_harness(harness, detector)
    h.publish_frame(1)
    assert h.spin_until(lambda: h.stamps("objects reliable") == [1] and _slot_free(h))
    _assert_failed(h.answer(h.call(DetectObjects, _detect_request("bowl"))), requests.MODEL_UNAVAILABLE)
    assert h.spin_until(lambda: _slot_free(h))
    assert {"service", "prepare"} <= _latency_words(h)
    assert h.node.latency.last_deltas["prepared"] == 0


def test_a_list_that_needs_no_preparation_shows_no_prepare_time(harness):
    h = _on_demand_harness(harness)
    h.publish_frame(1)
    assert h.spin_until(lambda: h.stamps("objects reliable") == [1] and _slot_free(h))
    assert h.answer(h.call(DetectObjects, _detect_request())).success
    assert h.spin_until(lambda: _slot_free(h))
    words = _latency_words(h)
    assert "service" in words and "prepare" not in words
