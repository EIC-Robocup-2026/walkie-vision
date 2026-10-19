# ADR-0002: Serve the models over ROS 2 only, statelessly

<!-- --8<-- [start:body] -->
**Status:** Accepted · **Date:** 2026-10-19

**Context**

The vision models ran in walkie-ai-server, a Flask HTTP server that also ran
speech-to-text and text-to-speech on the same GPU. It serves one request at
a time (`app.py`, `threaded=False`): in a new thread, the first CUDA kernel
launch costs about 800 ms, so a request took about 820 ms with threads and
about 25 ms without. With one worker, a streaming text-to-speech reply
(`api/routes/tts.py`, `synthesize_stream`) holds the server until the audio
ends, and every vision request waits behind it.

HTTP also loses what a robot needs around an image:

- **TF.** A box or a point cloud means something only in a frame, at a time.
  A ROS message carries both in its `header`, so any node can transform it
  with TF.
- **rosbag.** ROS topics can be recorded and replayed. A recording from the
  robot can be replayed on a laptop, for tests and for choosing a model.
- **QoS.** A subscriber can keep only the newest camera image and drop the
  rest, instead of queueing requests behind a slow model.

The RoboCup@Home 2026 rulebook (revision 2025-12-01) says in the Restaurant
test (section 5.5) that wireless and external computing "can't be
guaranteed. Assume unavailability." Section 3.4.1 requires robots to operate
safely when external computing is unavailable.

walkie-database already keeps the robot's memory. Callers send it vectors
with the name of the model that made them (walkie-database ADR-0006).

**Decision**

- walkie-vision talks to the rest of the robot only through ROS 2 topics and
  services. It runs no HTTP server.
- It is stateless. It publishes what it sees in each image and keeps
  nothing between images, except the tracker's ids within a stream. It
  writes nothing to disk. walkie-database is the only memory.
- Every model on the robot's critical path runs on the robot's own GPU. A
  cloud model may only be an extra that the robot can do without.
- Speech-to-text and text-to-speech stay in walkie-agent-ros. GraspNet,
  `yolo_node` and `obb3d` stay in walkie-ros.

**Consequences**

- A vision request never waits for speech: they run in other processes, in
  other repositories.
- Every consumer must be a ROS node, and must build this repository's
  messages ([ADR-0003](/architecture/decisions/0003-messages-in-this-repository.md)).
  A script that only speaks HTTP needs a ROS environment.
- Detections can be recorded with `ros2 bag` and replayed.
- A restart of walkie-vision loses only the track ids. walkie-database must
  match a new track to what it already knows, which it has to do anyway
  when a track breaks.
- The models must fit the robot's GPU next to everything else on it
  ([ADR-0008](/architecture/decisions/0008-model-choices.md)).

**See also:** [Vision](/architecture/vision.md),
[the Restaurant constraint](/architecture/vision.md#the-restaurant-constraint)
<!-- --8<-- [end:body] -->
