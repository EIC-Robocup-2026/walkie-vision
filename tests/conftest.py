import importlib.util
import os

import pytest

# Tests run on Fast DDS, although pixi.toml sets rmw_zenoh_cpp for every
# `pixi run`: under rmw_zenoh, nodes find each other only through a router,
# and CI has none. This must run before anything imports rclpy: importing it
# loads rcl, which reads RMW_IMPLEMENTATION and loads that middleware, so a
# change after the import has no effect. Keep every import that may pull in
# rclpy below these lines.
os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
# Keep test traffic on this machine and off any domain a real robot might use.
os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"


def pytest_configure(config):
    # The message package is built separately (`pixi run build-msgs`), and
    # scripts/activate.sh adds it to the environment only once it exists.
    # Without it, every test module that imports walkie_vision.convert would
    # fail with an ImportError that does not say what to do.
    if importlib.util.find_spec("walkie_vision_msgs") is None:
        raise pytest.UsageError(
            "walkie_vision_msgs is not built: run `pixi run build-msgs` once, then run the tests again"
        )
