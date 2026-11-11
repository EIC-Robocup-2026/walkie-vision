# ADR-0016: Let an optional debug visualizer subscribe to the raw image

<!-- --8<-- [start:body] -->
**Status:** Accepted · **Date:** 2026-11-12

**Context**

[ADR-0010](/architecture/decisions/0010-one-vision-process.md) makes the
vision node the only subscriber to the raw camera image. An HD2K image in
`bgra8` is about 11 MB, 15 times a second, so a second subscriber costs a
second copy of every frame.

- People testing the memory pipeline in simulation want to see what the node
  detected: the boxes on the image and the points in 3D, in Foxglove or
  RViz.
- `ObjectDetectionArray` carries no full image
  ([ADR-0013](/architecture/decisions/0013-memory-observation-contract.md)),
  and a viewer without `walkie_vision_msgs`, such as a stock
  `foxglove_bridge`, cannot decode it.
- Drawing inside the vision node would add decoding, drawing and JPEG
  encoding to the process that runs the models, for a view that nobody
  needs during a run.

**Decision**

- A separate process, `walkie_vision.debug_viz`, may subscribe to the raw
  colour image and to the object arrays. It publishes a JPEG
  `sensor_msgs/CompressedImage` and a `visualization_msgs/MarkerArray`,
  standard types only.
- It runs only when asked: `viz:=true` on the launch file, off by default.
  It loads no model.
- It pairs an array with its image by exact stamp and frame, keeps at most
  `ring_size` images (10 by default), and drops every image up to an array's
  stamp once it has handled that array.
- No consumer may depend on its topics. The vision node stays the only
  subscriber to the raw image that the observation contract relies on, and
  ADR-0010 holds for every other process.

**Consequences**

- With `viz:=true` the raw image is received twice. On the robot that is a
  second 11 MB copy 15 times a second, and up to about 110 MB of frames
  while the vision node falls behind. It is for simulation, and for someone
  watching the robot, not for a scored run.
- The debug topics can change without a contract change.
- The statements that the vision node is the only subscriber to the raw
  image, in ADR-0010, ADR-0011 and [Vision](/architecture/vision.md#processes),
  now carry this exception.

**See also:** [See the detections](/guides/vision-node.md#debug-visualizer),
[ADR-0010](/architecture/decisions/0010-one-vision-process.md),
[ADR-0013](/architecture/decisions/0013-memory-observation-contract.md).
<!-- --8<-- [end:body] -->
