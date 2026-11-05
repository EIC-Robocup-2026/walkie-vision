"""The test workflow keeps the restrictions of the local recipe. This reads the file; it does not run it."""

from __future__ import annotations

from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "test.yml"
RESTRICTIONS = (
    "RMW_IMPLEMENTATION=rmw_fastrtps_cpp",
    "ROS_DOMAIN_ID=77",
    "ROS_LOCALHOST_ONLY=1",
    "ROS_PYTHON_CHECK_FIELDS=1",
    "CUDA_VISIBLE_DEVICES=",
    "HF_HUB_OFFLINE=1",
    "TRANSFORMERS_OFFLINE=1",
    "YOLO_AUTOINSTALL=false",
)


def _steps() -> list[dict]:
    return yaml.safe_load(WORKFLOW.read_text())["jobs"]["test"]["steps"]


def test_the_tests_run_after_the_build_in_a_separate_pixi_run():
    runs = [step["run"] for step in _steps() if "run" in step]
    assert len(runs) == 2
    assert "build-msgs" in runs[0] and "pytest" not in runs[0]
    assert "pytest" in runs[1] and "build-msgs" not in runs[1]
    assert all("--locked" in run for run in runs)


def test_the_test_step_sets_the_restrictions_after_pixi_run():
    run = next(step["run"] for step in _steps() if "pytest" in step.get("run", ""))
    words = run.split()
    assert words[:4] == ["pixi", "run", "--locked", "env"]
    assert set(RESTRICTIONS) <= set(words[4:])


def test_the_job_environment_keeps_the_gpu_hidden_during_the_build():
    env = yaml.safe_load(WORKFLOW.read_text())["jobs"]["test"]["env"]
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["RMW_IMPLEMENTATION"] == "rmw_fastrtps_cpp"
    assert env["HF_HUB_OFFLINE"] == "1"


def test_the_workflow_only_reads_the_repository():
    assert yaml.safe_load(WORKFLOW.read_text())["permissions"] == {"contents": "read"}
