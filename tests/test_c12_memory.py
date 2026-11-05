"""The memory mode of check 12, with fake providers and a synthetic fixture. No model loads."""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from sensor_msgs.msg import CameraInfo

from walkie_vision.depth import RegisteredDepth
from walkie_vision.fakes import (
    FakeEmbedder,
    FakeObjectDetector,
    FakePersonFaceRecognizer,
    FakePoseEstimator,
    FakeReIdentifier,
    box_mask,
)
from walkie_vision.pipeline import Providers
from walkie_vision.providers import Detection

REPO = Path(__file__).resolve().parents[1]
CHECK = REPO / "checks" / "c12_vision_node.py"

# checks/ is not a package, so the module is loaded from its file under its own name.
_spec = importlib.util.spec_from_file_location("walkie_vision_checks_c12_memory", REPO / "checks" / "c12_memory.py")
c12_memory = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c12_memory)

HEIGHT, WIDTH = 48, 64
BOX = (2.0, 2.0, 40.0, 46.0)
K = np.array([50.0, 0.0, 32.0, 0.0, 50.0, 24.0, 0.0, 0.0, 1.0])
COLOR = np.random.default_rng(0).integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)
DEPTH = np.full((HEIGHT, WIDTH), 1.5, np.float32)


def _save(path: Path, *, color=COLOR, depth=DEPTH, k=K, **more) -> Path:
    arrays = {"color": color, "depth": depth, "k": k, **more}
    np.savez(path, **{name: value for name, value in arrays.items() if value is not None})
    return path


def _run_check(fixture: Path, models: Path) -> subprocess.CompletedProcess:
    """Run the check as a person would, with an empty models folder so that no model can load."""
    return subprocess.run(
        [sys.executable, str(CHECK), "--memory", str(fixture)],
        env=dict(os.environ, WALKIE_VISION_MODELS=str(models)),
        cwd=models,
        timeout=120,
        capture_output=True,
        text=True,
    )


def _masked(label: str, track_id: int) -> Detection:
    return Detection(
        box=np.array(BOX), label=label, score=0.8, track_id=track_id, mask=box_mask(HEIGHT, WIDTH, BOX)
    )


def _fake_providers(config) -> Providers:
    return Providers(
        objects=FakeObjectDetector([_masked("chair", 10)], classes=config.classes),
        pose=FakePoseEstimator([]),
        faces=FakePersonFaceRecognizer([]),
        reid=FakeReIdentifier(),
        embedder=FakeEmbedder(),
        # One object under the task's list, and one under the check's new list.
        on_demand=FakeObjectDetector([_masked("chair", 0), _masked("mug", 0)], classes=config.classes),
    )


def test_a_good_fixture_gives_its_image_depth_and_camera(tmp_path):
    color, depth, info = c12_memory.load_memory_fixture(_save(tmp_path / "pair.npz"))
    np.testing.assert_array_equal(color, COLOR)
    assert isinstance(depth, RegisteredDepth)
    assert (depth.fx, depth.fy, depth.cx, depth.cy) == (50.0, 50.0, 32.0, 24.0)
    assert (info.height, info.width) == (HEIGHT, WIDTH)
    assert list(info.k) == K.tolist()


@pytest.mark.parametrize(
    ("fixture", "says"),
    [
        (None, "cannot read"),
        ({"depth": None}, "cannot read"),
        ({"color": COLOR[:, :, 0]}, "color must be"),
        ({"color": COLOR.astype(np.float32)}, "color must be"),
        ({"depth": DEPTH[:10]}, "depth must be"),
        ({"depth": DEPTH.astype(np.float64)}, "depth must be"),
        ({"k": K[:4]}, "k must have 9"),
        ({"k": np.zeros(9)}, "k cannot be used"),
    ],
)
def test_a_bad_fixture_is_refused_with_status_2(tmp_path, capsys, fixture, says):
    path = tmp_path / "pair.npz"
    if fixture is not None:
        _save(path, **fixture)
    with pytest.raises(SystemExit) as refused:
        c12_memory.load_memory_fixture(path)
    assert refused.value.code == 2
    assert says in capsys.readouterr().err


def test_the_memory_config_turns_on_every_addition():
    config = c12_memory.memory_config()
    assert config.geometry_enabled and config.require_depth
    assert config.object_crops_enabled and config.object_embeddings_enabled and config.on_demand_enabled
    assert "cup" in config.classes and "person" not in config.classes


def _run_with_fakes(config, providers) -> list[tuple[bool, str]]:
    results: list[tuple[bool, str]] = []

    def expect(ok: bool, what: str) -> bool:
        results.append((ok, what))
        return ok

    depth = RegisteredDepth(DEPTH, 50.0, 50.0, 32.0, 24.0)
    info = CameraInfo(height=HEIGHT, width=WIDTH, k=K.tolist())
    c12_memory.run_memory(COLOR, depth, info, config, providers, expect, allow_fake_embedder=True)
    return results


def test_run_memory_passes_with_fakes_and_prints_every_stage(capsys):
    config = c12_memory.memory_config()
    providers = _fake_providers(config)
    results = _run_with_fakes(config, providers)
    assert results and all(ok for ok, _ in results), results
    printed = capsys.readouterr().out
    for stage in c12_memory.STAGES:
        assert f"  {stage}" in printed
    assert "a new list of 5" in printed and "never a pass or a fail" in printed
    calls = [line for line in printed.splitlines() if line.startswith(("  first call on", "  the same list again", "  a new list of"))]
    assert len(calls) == 3
    for line in calls:
        assert all(f" {stage} " in line for stage in c12_memory.ON_DEMAND_STAGES), line
        # chair under the task's list, mug under the new one: each call embeds its one box.
        assert "objects 1, sampled 1" in line, line
    assert providers.objects.classes == list(config.classes)


def test_run_memory_fails_a_recording_that_shows_nothing():
    config = c12_memory.memory_config()
    providers = dataclasses.replace(_fake_providers(config), objects=FakeObjectDetector([], classes=config.classes))
    failed = [what for ok, what in _run_with_fakes(config, providers) if not ok]
    assert any("shows an object" in what for what in failed)


def test_the_script_refuses_a_missing_fixture_before_any_model(tmp_path):
    done = _run_check(tmp_path / "missing.npz", tmp_path)
    assert done.returncode == 2, done.stderr
    assert "cannot read the fixture" in done.stderr
    assert "build_providers" not in done.stdout


def test_the_script_reaches_the_model_check_with_a_good_fixture(tmp_path):
    # The models folder is empty, so the start refuses before torch or Ultralytics load.
    done = _run_check(_save(tmp_path / "pair.npz"), tmp_path)
    assert done.returncode == 1, done.stderr
    assert "FileNotFoundError" in done.stderr and "fetch-models" in done.stderr
