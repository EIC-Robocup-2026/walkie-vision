# ADR-0015: Schedule bounded service work on the one model worker

<!-- --8<-- [start:body] -->
**Status:** Accepted · **Date:** 2026-11-05

**Context**

[ADR-0014](/architecture/decisions/0014-bounded-rgbd-and-service-work.md) says
that the worker runs every model call, that at most one service job waits, and
that another request gets a BUSY answer. It leaves the mechanism open.
[ADR-0010](/architecture/decisions/0010-one-vision-process.md) keeps every model
in one process, and that stays in force. This ADR explains the exceptions to the
worker's loop. It adds no GPU process.

Object memory now needs two services that run a model:

- `EmbedText` must work before the first image, so a search can run with no
  camera. A worker that sleeps until an image arrives would never serve it.
- `EmbedCrop` embeds a box that walkie-database projected into a named image. On
  any other view, the same box covers something else.

Several things constrain the mechanism:

- A callback that waits for the worker needs an executor thread. Two requests
  and `GetPersonCrop` can all wait at once, and the input callbacks must keep
  running. [ADR-0011](/architecture/decisions/0011-serve-one-person-crop.md) spins
  two threads, which is not enough.
- A burst of requests must not stop the stream that walkie-agent-ros and
  walkie-database read.
- A paused `/clock` in the sim must not hang a request.
- `EmbedCrop` can name an image the worker has already processed, because the
  stamp comes from a published array. The mailbox has given that image away by
  then.

**Decision**

- **The worker wakes for a job.** A request wakes the worker, as an image does.
  With no image at all, a job still runs.
- **One slot across the services.** `EmbedText` and `EmbedCrop` share one job slot
  in `walkie_vision.requests`. A request that finds it taken gets `BUSY` at once.
  Nothing queues. `GetPersonCrop` keeps its own mailbox, because the worker answers
  it from the frame it is processing.
- **The frame comes before the next job.** After a job, the worker takes the
  newest image, if there is one, and only then frees the slot. A stream of requests
  cannot stop the stream of images. The rate cap still applies to that image.
- **One reentrant group, four executor threads.** The three new services share a
  `ReentrantCallbackGroup`, so a second request can run its callback and get `BUSY`
  while the first waits. `main()` spins a `MultiThreadedExecutor(num_threads=4)`.
  This replaces the two threads of ADR-0011.
- **Two clocks.** Deadlines use monotonic time. Image age uses ROS time. A paused
  clock can make an image look young, but it cannot make a request hang.
- **A late result is discarded.** A callback that passes its deadline answers
  `TIMEOUT`. A result that comes after that is counted and dropped. A job the
  worker has not taken leaves the slot at once. A job it has taken holds the slot
  until the worker comes back.
- **The watchdog covers jobs.** A job that runs longer than `process_timeout_s`
  stops the node, like an image that does.
- **A service failure stays with the service.** A failed job answers its caller.
  It does not count as a failed image, and the arrays keep flowing.
- **`EmbedCrop` answers from the newest colour and the newest depth only.** With
  `object_embeddings_enabled`, the input callbacks also keep a reference to the
  newest of each. A new one replaces the old one. There is no history and no way
  to fetch an older image.
- **The gates are exact.** The frame and the size must equal the current image's.
  The stamps may differ by at most 5 ms. The image may be at most 300 ms old in
  ROS time. The request's stamp may be at most 5 ms ahead of ROS time. A request
  that fails one of them gets `STALE_IMAGE`, and never a near match. The worker
  checks them when it starts the job and does not check again after inference. The
  response's header shows the image that was used.
- **Endpoints stay up when off.** The services always exist. With
  `object_embeddings_enabled` off, `EmbedText` and `EmbedCrop` answer `DISABLED`.
  `DescribeCrop` always answers `DISABLED`, and `captions_enabled` must be false,
  because this build has no caption model.

This supersedes two points of ADR-0011: "Two executor threads", and, with
`object_embeddings_enabled` on, the consequence that the node holds no image
beyond the one it answers from. The rest of ADR-0011 stands.

It also adds one consequence to ADR-0010, whose decision is unchanged: the single
worker now shares its GPU time between the stream and service jobs. No second GPU
process exists.

**Consequences**

- A service job delays the next image by its own length. The latency of a service
  request and the image rate while requests arrive are not measured on the robot.
  This ADR does not claim that 5 Hz holds under service load.
- The stamp gate is strict. At 15 Hz a new image replaces the current one every
  67 ms, so a request succeeds only while the image it names is still the newest.
  `EmbedCrop` may refuse most proposals under load, and the candidate stays an
  honest unknown. A bounded image history or proposal batching needs a measurement
  and a new ADR first.
- With `object_embeddings_enabled`, up to two more large messages stay alive
  between images, about 11 MB each at HD2K. They are references to messages the
  callbacks already hold, so they add no copy. HRI keeps nothing extra.
- A job that a timed-out callback leaves on the worker keeps the slot. Other
  requests get `BUSY` until it ends.
- `EmbedText`'s type has no error code field, so its failures carry the code at
  the start of `message`. A client must parse that prefix to tell `BUSY` from
  `DISABLED`.
- CLIP has no warm-up, so the first sampled image or service job can be slow.
- Four threads apply to every task, HRI included. An idle thread costs little, but
  the number is now a limit: more waiting services need more threads.

**See also:** [Threads](/architecture/vision.md#threads),
[DetectObjects, EmbedCrop and DescribeCrop](/architecture/vision.md#detectobjects-embedcrop-and-describecrop),
[ADR-0014](/architecture/decisions/0014-bounded-rgbd-and-service-work.md),
[ADR-0010](/architecture/decisions/0010-one-vision-process.md),
[ADR-0011](/architecture/decisions/0011-serve-one-person-crop.md).
<!-- --8<-- [end:body] -->
