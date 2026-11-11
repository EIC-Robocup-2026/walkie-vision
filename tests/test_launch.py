"""launch/vision.launch.py: the command, the arguments and the respawn."""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
import threading
import time
from pathlib import Path

import pytest
import yaml
from launch import LaunchContext, LaunchDescription, LaunchService
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, Shutdown, TimerAction
from launch.event_handlers import OnProcessStart
from launch.utilities import perform_substitutions

from walkie_vision.node import VisionConfig

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config"
GUARD_S = 15.0


@pytest.fixture(scope="module")
def vision_launch():
    spec = importlib.util.spec_from_file_location("vision_launch", REPO / "launch" / "vision.launch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_command_runs_the_node_with_the_task_file_last(vision_launch):
    cmd = vision_launch.node_command("hri", False)
    assert cmd[:5] == [sys.executable, "-P", "-m", "walkie_vision.node", "--ros-args"]
    assert cmd[5:] == [
        "--params-file",
        str(CONFIG / "vision.yaml"),
        "--params-file",
        str(CONFIG / "tasks" / "hri.yaml"),
    ]


def test_sim_adds_the_sim_file_last(vision_launch):
    cmd = vision_launch.node_command("hri", True)
    assert cmd[-4:] == ["--params-file", str(CONFIG / "tasks" / "hri.yaml"), "--params-file", str(CONFIG / "sim.yaml")]


def test_the_memory_task_runs_in_sim_with_the_sim_file_last(vision_launch):
    assert "memory" in vision_launch.task_names()
    cmd = vision_launch.node_command("memory", True)
    assert cmd[-4:] == [
        "--params-file",
        str(CONFIG / "tasks" / "memory.yaml"),
        "--params-file",
        str(CONFIG / "sim.yaml"),
    ]


def test_the_sim_file_moves_every_service_and_topic_of_this_node(vision_launch):
    # Each output must have its own sim name, or a sim run would answer for the
    # real robot's endpoints. A new service fails here until sim.yaml names it.
    sim = yaml.safe_load((CONFIG / "sim.yaml").read_text())["walkie_vision"]["ros__parameters"]
    assert sim["use_sim_time"] is True
    fields = {f.name: f.default for f in dataclasses.fields(VisionConfig)}
    services = {name: value for name, value in fields.items() if name.endswith("_service")}
    assert set(services) >= {
        "person_crop_service",
        "embed_text_service",
        "embed_crop_service",
        "describe_crop_service",
        "detect_objects_service",
    }
    for name, value in {**services, "people_topic": fields["people_topic"], "objects_topic": fields["objects_topic"]}.items():
        assert value.startswith("/walkie_vision/"), name
        assert sim[name] == value.replace("/walkie_vision/", "/walkie_vision/sim/", 1), name


def test_an_unknown_task_names_the_known_ones(vision_launch):
    assert "hri" in vision_launch.task_names()
    with pytest.raises(ValueError, match="unknown task 'cooking'.*hri"):
        vision_launch.node_command("cooking", False)


def test_every_parameter_file_in_the_command_exists(vision_launch):
    cmd = vision_launch.node_command("hri", True)
    files = [Path(cmd[i + 1]) for i, arg in enumerate(cmd) if arg == "--params-file"]
    assert len(files) == 3 and all(f.is_file() for f in files)


def test_the_launch_arguments(vision_launch):
    description = vision_launch.generate_launch_description()
    declared = {e.name: e.default_value for e in description.entities if isinstance(e, DeclareLaunchArgument)}
    assert set(declared) == {"task", "sim", "respawn_delay", "respawn_max_retries", "viz"}
    defaults = {name: "".join(s.text for s in value) for name, value in declared.items()}
    assert defaults == {
        "task": "hri",
        "sim": "false",
        "respawn_delay": "3.0",
        "respawn_max_retries": "-1",
        "viz": "false",
    }


def test_the_visualizer_draws_the_sim_topics_under_the_sim_namespace(vision_launch):
    cmd = vision_launch.viz_command("memory", True)
    assert cmd[:5] == [sys.executable, "-P", "-m", "walkie_vision.debug_viz", "--ros-args"]
    params = dict(cmd[i + 1].split(":=", 1) for i, arg in enumerate(cmd) if arg == "-p")
    assert params == {
        "objects_topic": "/walkie_vision/sim/objects",
        "image_topic": "/zed_head/zed_node/rgb/color/rect/image",
        "image_out_topic": "/walkie_vision/sim/debug/objects_image/compressed",
        "markers_topic": "/walkie_vision/sim/debug/objects_markers",
        "use_sim_time": "true",
    }


def test_the_visualizer_on_the_robot_uses_the_robot_topics_and_wall_time(vision_launch):
    cmd = vision_launch.viz_command("hri", False)
    params = dict(cmd[i + 1].split(":=", 1) for i, arg in enumerate(cmd) if arg == "-p")
    assert params["objects_topic"] == "/walkie_vision/objects"
    assert params["image_out_topic"] == "/walkie_vision/debug/objects_image/compressed"
    assert params["markers_topic"] == "/walkie_vision/debug/objects_markers"
    assert "use_sim_time" not in params


def test_viz_adds_the_visualizer_process(vision_launch, tmp_path, monkeypatch):
    (tmp_path / "models" / "ultralytics").mkdir(parents=True)
    monkeypatch.setenv("WALKIE_VISION_MODELS", str(tmp_path / "models"))
    context = LaunchContext()
    context.launch_configurations.update(
        task="memory", sim="true", respawn_delay="3.0", respawn_max_retries="-1", viz="true"
    )
    processes = vision_launch._launch(context)
    assert len(processes) == 2
    cmd = [perform_substitutions(context, part) for part in processes[1].cmd]
    assert cmd == vision_launch.viz_command("memory", True)


def test_a_relative_models_dir_reaches_the_node_as_an_absolute_path(vision_launch, tmp_path, monkeypatch):
    # The node runs in models/ultralytics/, where a relative path would point elsewhere.
    (tmp_path / "models" / "ultralytics").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WALKIE_VISION_MODELS", "models")
    context = LaunchContext()
    context.launch_configurations.update(
        task="hri", sim="false", respawn_delay="3.0", respawn_max_retries="-1", viz="false"
    )
    [process] = vision_launch._launch(context)
    assert perform_substitutions(context, process.cwd) == str(tmp_path / "models" / "ultralytics")
    env = {perform_substitutions(context, k): perform_substitutions(context, v) for k, v in process.additional_env}
    assert env == {"YOLO_OFFLINE": "true", "WALKIE_VISION_MODELS": str(tmp_path / "models")}


def _run(vision_launch, tmp_path, *, max_retries: int, shutdown_at: int | None) -> tuple[int, float]:
    """Run a process that always exits 3 under the node's respawn settings.

    Returns the number of starts and the seconds the launch ran.
    """
    process = vision_launch.vision_process(
        [sys.executable, "-c", "raise SystemExit(3)"], tmp_path, 0.2, respawn_max_retries=max_retries
    )
    starts = 0

    def on_start(event, context):
        nonlocal starts
        starts += 1
        if shutdown_at is not None and starts >= shutdown_at:
            return [Shutdown(reason="the test saw enough starts")]
        return None

    entities = [process, RegisterEventHandler(OnProcessStart(target_action=process, on_start=on_start))]
    if shutdown_at is not None:
        # A pending timer keeps the launch alive, so it is only a guard
        # when the test ends the launch itself.
        entities.append(TimerAction(period=GUARD_S, actions=[Shutdown(reason="the guard fired")]))
    service = LaunchService()
    service.include_launch_description(LaunchDescription(entities))
    guard = threading.Timer(GUARD_S, service.shutdown)
    guard.start()
    started = time.monotonic()
    try:
        service.run()
    finally:
        guard.cancel()
    return starts, time.monotonic() - started


def test_the_node_starts_again_after_it_exits(vision_launch, tmp_path):
    starts, elapsed = _run(vision_launch, tmp_path, max_retries=-1, shutdown_at=2)
    assert starts >= 2
    assert elapsed < GUARD_S


def test_max_retries_caps_the_restarts(vision_launch, tmp_path):
    starts, elapsed = _run(vision_launch, tmp_path, max_retries=1, shutdown_at=None)
    assert starts == 2
    # The launch ended by itself once no restart was left.
    assert elapsed < GUARD_S
