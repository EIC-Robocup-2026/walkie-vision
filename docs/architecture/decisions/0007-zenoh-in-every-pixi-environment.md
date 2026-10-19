# ADR-0007: Use rmw_zenoh_cpp in every Pixi environment

<!-- --8<-- [start:body] -->
**Status:** Accepted · **Date:** 2026-10-19

**Context**

walkie-ros, the robot, walkie-agent-ros and walkie-database run on the Zenoh
middleware, `rmw_zenoh_cpp`. Each computer runs one Zenoh router,
`rmw_zenohd`, on port 7447. Fast DDS and Zenoh do not talk to each other,
and a mismatch raises no error: the nodes simply never see each other.

`rmw_zenoh_cpp` 0.2.10 can block forever in `rmw_wait`
([ros2/rmw_zenoh#1032](https://github.com/ros2/rmw_zenoh/issues/1032)).
0.2.11 fixes it. RoboStack's 0.2.11 build belongs to the same stack build as
this environment's `ros-base`.

walkie-agent-ros (its ADR-0033) and walkie-database (its ADR-0014) made the
same decision.

**Decision**

- `pixi.toml` depends on `ros-jazzy-rmw-zenoh-cpp` `>=0.2.11,<0.3` from
  RoboStack. It also provides the router.
- `[activation.env]` sets `RMW_IMPLEMENTATION=rmw_zenoh_cpp` for every
  `pixi run` and `pixi shell`, and replaces a value exported in the calling
  shell. The `docs` environment leaves out the default feature, so
  `[feature.docs.activation.env]` sets the same value.
- There is no Zenoh configuration in this repository. Nodes connect to the
  router at `tcp/localhost:7447`, rmw_zenoh's default: the robot's router,
  the one walkie-ros's simulation uses, or one started with
  `pixi run ros2 run rmw_zenoh_cpp rmw_zenohd`.
- Tests run on Fast DDS. `tests/conftest.py` sets `rmw_fastrtps_cpp` before
  anything imports rclpy, so `pixi run test` needs no router.
- `check-env`'s ROS check (`c10`) starts a private router on a free port,
  never on 7447, so it never joins the robot's ROS graph.

**Consequences**

- The vision nodes reach walkie-ros, walkie-database and the agent without
  any export in the terminal. `ROS_DOMAIN_ID` must still match theirs.
- Exporting another middleware in the shell does not change what `pixi run`
  uses. Use `pixi run env RMW_IMPLEMENTATION=rmw_fastrtps_cpp <command>` for
  one command on Fast DDS.
- Every node and `ros2` command on Zenoh prints `zenoh_config` warnings
  about deprecated settings. They come from rmw_zenoh's own default
  configuration and are harmless.
- The version must match walkie-ros's. Move the upper bound only together
  with walkie-ros.
- A Zenoh-only problem does not show up in `pixi run test`. It shows up in
  `check-env`, or when the nodes run with a router.

**See also:** [Getting Started](/getting-started.md#7-talk-to-other-nodes),
[Step 1: ROS 2 and the middleware](/setup/01-vision-environment.md#step-1-ros-2-and-the-middleware)
<!-- --8<-- [end:body] -->
