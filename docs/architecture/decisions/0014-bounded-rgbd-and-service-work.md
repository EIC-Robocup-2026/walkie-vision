# ADR-0014: Keep only bounded RGB-D input and service work

<!-- --8<-- [start:body] -->
**Status:** Accepted · **Date:** 2026-11-05

**Context**

[ADR-0002](/architecture/decisions/0002-ros2-only-stateless.md) says that the
node keeps nothing between images, except the tracker's ids. Its input so far is
one mailbox with the newest image, so that held.

Object memory needs three things that do not fit those words:

- **A point needs depth from the same capture as its colour pixel.** Depth and
  colour arrive on two topics, in either order, so the node must hold one until
  the other comes.
- **Sampling needs a count.** Choosing which objects get a crop in an image
  needs a number that survives from one image to the next.
- **Services need the model worker.** The worker runs every model, and it also
  runs the stream. A service request must wait for it without letting the
  requests pile up.

The obvious way to pair the two topics is `message_filters`. Its synchronizers
keep a queue per topic and work only on stamps. With a queue there is more to
hold, and with no bound on wall time, a colour image can wait as long as the
sim's clock stays paused.

**Decision**

- **Newest input only.** With `geometry_enabled`, the node holds the newest
  colour image, the newest depth image and one finished bundle. A new one
  replaces the old one. There is no queue and no older image.
- **Pairing is bounded twice.** The stamps may differ by `max_pair_gap_s` (5 ms)
  and an exact stamp wins. An image waits at most `pair_wait_s` (60 ms) of wall
  time. Then `require_depth` decides: skip and count the image, or publish it
  with empty clouds. The wait uses wall time, so a paused or reset sim clock
  cannot hang the worker. A depth image that waited longer than `pair_wait_s`
  is dropped, and a depth image is used once.
- **Registered depth only.** The depth must be rectified, in the colour image's
  optical frame, and describe the same rays. Anything else is refused. The node
  reads no TF. Registration is the camera's work, in walkie-ros.
- **Off means off.** Without `geometry_enabled`, the node does not subscribe to
  depth, and the input stays the one-slot mailbox.
- **One processed-image counter.** Crop sampling may keep one counter of
  processed images. It holds no image.
- **One worker, one pending service job.** The worker runs every model call,
  also for services, between images. At most one service job waits. Another
  request gets a BUSY answer instead of a thread that waits behind it. The input
  callbacks keep running. This is not a second GPU process:
  [ADR-0010](/architecture/decisions/0010-one-vision-process.md) stays in force.
- **Still no history.** Nothing here is an image history. A request cannot fetch
  an old image, and no handle to an image exists.

This supersedes only the input-buffer and "keeps nothing between images" wording
of ADR-0002. The rest of ADR-0002 stands: ROS 2 only, nothing on disk, and
walkie-database is the only memory.

**Consequences**

- Up to three large messages stay in memory at a time. At HD2K each one is about
  11 MB. A restart still loses only the track ids and that counter.
- An image with a depth just before it waits for the whole 60 ms, in case an
  exact depth still comes. On a stream with inexact stamps, that adds up to
  60 ms per image. An exact stamp costs nothing. An image without depth also
  waits the full 60 ms before it goes to the worker.
- Depth that arrives more than 60 ms before or after its image does not pair.
  The default was 20 ms until 2026-11-22. That day, on the robot with HD2K
  colour and depth through the team router and a heavily loaded computer, the
  memory task paired 0 to 1 images in 10 s, against 13 to 19 earlier the same
  day. The stamps of the two topics match exactly, so the cause is the wall-time
  gap between their arrival; that gap is still not measured directly. A wait
  near the frame interval does not help, because the next image replaces the
  waiting one, so 60 ms stays under the 66 to 90 ms between frames at 15 to
  11 Hz.
- A second CameraInfo subscription exists only when the depth has its own topic.
- The buffer is in use now, with `geometry_enabled`. The counter and the
  scheduler arrive with the slices that need them; this ADR does not change then.

**See also:** [Depth pairing](/architecture/vision.md#depth-pairing),
[ADR-0013](/architecture/decisions/0013-memory-observation-contract.md),
[ADR-0010](/architecture/decisions/0010-one-vision-process.md).
<!-- --8<-- [end:body] -->
