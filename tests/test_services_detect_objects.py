"""walkie_vision.services.detect_objects: the class gate, the request check and the image age. No ROS graph."""

from __future__ import annotations

import pytest
from builtin_interfaces.msg import Time
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from walkie_vision_msgs.srv import DetectObjects

from walkie_vision.services.detect_objects import ClassPrepareGate, detect_objects_problem, image_age_problem

FRAME = "camera_optical"
WIDTH, HEIGHT = 64, 48
STAMP_NS = 10_000_000_000


def _stamp(ns: int) -> Time:
    return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


def _image(ns: int = STAMP_NS, frame: str = FRAME, width: int = WIDTH, height: int = HEIGHT) -> Image:
    return Image(header=Header(stamp=_stamp(ns), frame_id=frame), width=width, height=height)


# --- the class gate --------------------------------------------------------


def test_the_class_gate_runs_from_the_end_of_a_preparation():
    now = [100.0]
    gate = ClassPrepareGate(2.0, clock=lambda: now[0])
    assert gate.open()
    now[0] = 101.5  # the preparation took 1.5 s
    gate.mark()
    assert not gate.open()
    now[0] = 103.4
    assert not gate.open()
    now[0] = 103.5
    assert gate.open()


# --- the checks ------------------------------------------------------------


def _detect_request(**fields) -> DetectObjects.Request:
    request = DetectObjects.Request(classes=["cup"])
    for name, value in fields.items():
        setattr(request, name, value)
    return request


def test_a_detect_objects_request_has_the_defaults_of_its_type():
    request = DetectObjects.Request(classes=["cup"])
    assert request.max_image_age_s == pytest.approx(0.3) and request.timeout_s == pytest.approx(4.0)
    assert detect_objects_problem(request) is None


@pytest.mark.parametrize(
    ("fields", "reason"),
    [
        ({"classes": []}, "empty"),
        ({"classes": [f"c{i}" for i in range(33)]}, "at most 32"),
        ({"classes": ["cup", ""]}, "class 1 is empty"),
        ({"classes": [" cup"]}, "class 0 is empty or has spaces"),
        ({"classes": ["   "]}, "class 0 is empty or has spaces"),
        ({"classes": ["x" * 81]}, "more than 80"),
        ({"classes": ["cup", "cup"]}, "duplicates"),
        ({"max_image_age_s": 0.0}, "max_image_age_s"),
        ({"max_image_age_s": -0.1}, "max_image_age_s"),
        ({"max_image_age_s": 1.5}, "max_image_age_s"),
        ({"max_image_age_s": float("nan")}, "max_image_age_s"),
        ({"max_image_age_s": float("inf")}, "max_image_age_s"),
        ({"timeout_s": 0.0}, "timeout_s"),
        ({"timeout_s": 5.5}, "timeout_s"),
        ({"timeout_s": float("nan")}, "timeout_s"),
    ],
)
def test_a_bad_detect_objects_request_is_refused(fields, reason):
    assert reason in detect_objects_problem(_detect_request(**fields))


def test_a_detect_objects_request_at_the_caps_is_valid():
    request = _detect_request(
        classes=[f"{i}".ljust(80, "x") for i in range(32)], max_image_age_s=1.0, timeout_s=5.0
    )
    assert detect_objects_problem(request) is None


def test_a_detect_objects_refusal_does_not_repeat_a_class():
    assert "secret" not in detect_objects_problem(_detect_request(classes=["secret" * 20]))


@pytest.mark.parametrize(
    ("age_ns", "current"),
    [
        (0, True),
        (300_000_000, True),
        (300_000_001, False),
        (-300_000_000, True),  # /clock can lag an image
        (-300_000_001, False),  # a stamp from before a sim restart
    ],
)
def test_the_image_age_counts_both_ways(age_ns, current):
    problem = image_age_problem(_image(STAMP_NS), STAMP_NS + age_ns, 0.3)
    assert (problem is None) is current
