# ADR-0003: Keep the messages in a msg-only package in this repository

<!-- --8<-- [start:body] -->
**Status:** Accepted · **Date:** 2026-10-19

**Context**

walkie-vision publishes detections with embeddings, the name of the model
that made each embedding, body keypoints, faces and 3D points.
`vision_msgs/Detection2DArray`, which walkie-ros's `yolo_node` publishes,
has no place for most of these, so new message types are needed.

They could live in walkie-ros, in walkie-database, in a repository of their
own, or here. Under rmw_zenoh, a publisher and a subscriber built from
different definitions of a type never connect, and nothing reports it
(walkie-agent-ros ADR-0017 and ADR-0018). So every repository must build
the messages from the same commit.

**Decision**

- The messages and the service are in `walkie_vision_msgs`, in
  `ros2_ws/src/` of this repository. The package holds only interface
  files: no nodes and no Python code.
- They use only `std_msgs`, `sensor_msgs` and `vision_msgs`. Boxes are
  `vision_msgs/BoundingBox2D`, in float pixels.
- `pixi run build-msgs` builds the package with colcon, and
  `scripts/activate.sh` adds the build to every `pixi run` and `pixi shell`.
- `tests/test_convert.py` pins every field name and type.

**Consequences**

- A message change and the code that fills it land in one commit, in the
  repository that publishes it.
- walkie-database and walkie-agent-ros must build `walkie_vision_msgs` in
  their own workspace, from a pinned commit of this repository, and need
  `ros-jazzy-vision-msgs`. A clone of this repository in their
  `ros2_ws/src/` shows colcon only the message package.
- They do not need PyTorch or a GPU to build it.
- After every message change, every repository that uses it must rebuild it
  from the new commit. A node built from the old definition gets no
  messages, and prints no error.
- walkie-database's object memory design (Proposed) sketches an
  `ObjectDetection` of its own, with a header in every detection, and a
  `/walkie_database/detections` topic. It must move to these messages and
  to this repository's topic names.

**See also:** [Add the Message Package](/setup/02-message-package.md),
[Messages](/architecture/vision.md#messages)
<!-- --8<-- [end:body] -->
