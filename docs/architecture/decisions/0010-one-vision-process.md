# ADR-0010: Run every vision model in one process

<!-- --8<-- [start:body] -->
**Status:** Accepted · **Date:** 2026-10-26

**Context**

[ADR-0004](/architecture/decisions/0004-one-environment-two-processes.md) put
the face models in a second process, so that a crash in ONNX Runtime would
not stop object detection. It left open how that process gets images and who
puts its faces into `PersonDetection`.

- Consumers pair the people and object arrays of one image by exact stamp,
  and expect one people array per image, with the faces in it. A face
  process would need a merge step keyed by stamp, and a way to avoid a
  second, partial people array.
- A face process would need the image too: the raw 11 MB HD2K image, or
  crops sent over ROS.
- SCRFD needs faces at native resolution. At fx 1066, walkie-agent-ros
  trusts a face from 0.0469 rad, about 50 px. On SCRFD's 640 input of a
  2208 px image, that face is about 14 px. Crops around the head keep it at
  full size, but the head position comes from the pose model.
- The check `c12` loads the models the way the node does, PyTorch models
  first and then the ONNX Runtime sessions, and runs 20 HD2K frames with
  every model in turn. Both ONNX Runtime sessions stay on CUDA and every
  output is finite. `c11` checks each model wrapper on its own.

**Decision**

- All models share one Pixi environment, the default one. (This keeps that
  part of ADR-0004.)
- YOLOE-26 and YOLO26 pose with tracking, SCRFD and ArcFace on
  `onnxruntime-gpu`, and OSNet run in one process, the vision node
  (`walkie_vision.node`). The `EmbedText` server and the `DetectObjects`
  service join it when they are built.
- The vision node is the only subscriber to the raw camera image.
- SCRFD runs on a crop around each person's head, from the pose keypoints,
  at native resolution. Each face goes into its person's `PersonDetection`
  before the array is published.
- The node runs under `ros2 launch` with respawn. Track ids start from the
  wall-clock time at start, so a restarted node does not reuse them.

**Consequences**

- The image is received once, and there is one CUDA context.
- There is no merge step and no partial people array.
- A crash in ONNX Runtime's native code now stops objects and people too.
  The launch starts the node again, but nothing is published while the
  models load, so consumers see stale data for several seconds.
- The first start on a GPU that ONNX Runtime has no machine code for, such
  as an RTX 50, compiles its kernels and can take minutes. Later starts are
  fast while the driver's cache in `~/.nv/ComputeCache` is kept.
- All models must fit in one process's GPU memory, next to whatever else
  runs on the GPU. On the robot that includes the ZED SDK's depth. `c12`
  measures the node on the laptop; the robot is not measured yet.
- The `device` parameter moves only the PyTorch models. ONNX Runtime uses
  GPU 0.
- Without the id base, a restart would reuse ids from 1, and walkie-agent-ros
  could join two people into one track. See
  [Track ids](/architecture/vision.md#contract-track-ids).
- One lock file, and PyTorch and ONNX Runtime must agree on a CUDA version,
  as before ([ADR-0005](/architecture/decisions/0005-cuda-12-sm75-sm120.md)).

**See also:** [Vision](/architecture/vision.md#processes),
[Contract](/architecture/vision.md#contract),
[Run the vision node](/guides/vision-node.md)
<!-- --8<-- [end:body] -->
