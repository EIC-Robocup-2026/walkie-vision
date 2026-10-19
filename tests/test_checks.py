"""Helpers of the check-env checks (checks/common.py)."""

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# checks/ is not a package, and "common" is too generic a name to put on
# sys.path, so the module is loaded from its file under its own name.
_spec = importlib.util.spec_from_file_location("walkie_vision_checks_common", REPO / "checks" / "common.py")
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)


def test_foreign_paths_allows_the_environment_and_this_repository(tmp_path):
    prefix = tmp_path / "env"
    paths = [
        "",
        str(prefix / "lib" / "python3.12"),
        str(prefix / "lib" / "python312.zip"),
        str(REPO / "src"),
        str(REPO / "checks"),
        # Added by scripts/activate.sh once `pixi run build-msgs` has run.
        str(REPO / "ros2_ws" / "install" / "walkie_vision_msgs" / "lib" / "python3.12" / "site-packages"),
    ]
    assert common.foreign_paths(paths, prefix, REPO) == []


def test_foreign_paths_lists_everything_else(tmp_path):
    prefix = tmp_path / "env"
    paths = [
        "/opt/ros/jazzy/lib/python3.12/site-packages",
        str(tmp_path / ".local" / "lib" / "python3.12" / "site-packages"),
        str(REPO / "ros2_ws" / "src"),
        str(REPO / "tools"),
    ]
    assert common.foreign_paths(paths, prefix, REPO) == paths
