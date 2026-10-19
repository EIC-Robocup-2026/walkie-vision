from rclpy.utilities import get_rmw_implementation_identifier


def test_tests_use_fast_dds():
    # conftest.py sets RMW_IMPLEMENTATION before rclpy loads the middleware.
    # If an import moved above that line, rclpy would load rmw_zenoh_cpp from
    # pixi.toml instead, and tests that start nodes would wait for a router.
    assert get_rmw_implementation_identifier() == "rmw_fastrtps_cpp"
