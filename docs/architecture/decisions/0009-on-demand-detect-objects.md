# ADR-0009: Add an on-demand DetectObjects service for detecting by name

<!-- --8<-- [start:body] -->
**Status:** Proposed · **Date:** 2026-10-22

**Context**

walkie-agent-ros proposed a self-evolving agent
(`docs/architecture/self-evolving-agent.md` at walkie-agent-ros commit
91ffe25). walkie-agent-ros plans to withdraw that page and has no replacement
page yet, so this reference names a commit. The design finds an object in
steps: it asks walkie-database's memory first, then detects the object by its
noun in what the head camera sees now, then ranks the detections by their
embeddings against the spoken description. Its `detect_objects` tool needs to
say "look for `mug` now" and get detections back.

Nothing in walkie-vision's design answers that. The vision node
([ADR-0010](/architecture/decisions/0010-one-vision-process.md))
runs YOLOE-26 on every processed image, with a list of text prompts set
per task, and publishes the result on `/walkie_vision/objects`. A noun that
is not in the list is never detected. We considered three ways to close the
gap:

- **Re-prompt the streaming detector.** Call `set_classes()` with the
  caller's nouns. This costs too much:
    - `set_classes()` with 5 names took 1.5 s on the RTX 2060
      ([Measured on the RTX 2060](/setup/01-vision-environment.md#measured-on-the-rtx-2060)).
      In the installed Ultralytics 8.4.182, `YOLOE.set_classes()` loads the
      text encoder again on every call that changes the list.
    - It may reset the track ids (`ObjectDetector` in
      `walkie_vision.providers`). In Ultralytics 8.4.182 it drops the
      predictor, and the trackers live on the predictor (read in the code,
      not tested).
    - It changes the `classes` field of the published arrays. That field
      is negative evidence: a class in `classes` without a detection was not
      seen in that image (`ObjectDetectionArray.msg`). While the caller's
      nouns replace the task's list, walkie-database would read every
      task class as "not seen" when it was only "not looked for".
    - Restoring the old list afterwards pays the 1.5 s again and still
      resets the tracks. Two callers at once would fight over one list.
- **Return a frame handle.** A `capture_frame` call would keep the image in
  walkie-vision and give the caller an id to detect on later. That keeps
  state between requests, which
  [ADR-0002](/architecture/decisions/0002-ros2-only-stateless.md) rules
  out. Keeping the raw frame in the agent instead conflicts with
  walkie-agent-ros ADR-0028 (Proposed), which keeps raw camera images out of
  the agent and allows only one compressed snapshot on request. An HD2K
  image in `bgra8` is about 11 MB.
- **Wait on the topic.** The caller subscribes to `/walkie_vision/objects`
  and waits for its noun. This works only for nouns already in the task's
  list.

**Decision**

- Add a service type `DetectObjects` to `walkie_vision_msgs`, served by the
  vision node at `/walkie_vision/detect_objects`, and at
  `/walkie_vision/sim/detect_objects` with `sim:=true`. `on_demand_enabled`
  turns it on. Off, it answers `DISABLED` and the second model does not load.
- The request names the classes to look for. The response holds one
  `ObjectDetectionArray` for one image. Its `classes` are the request's names.
  The type:

    ```text title="walkie_vision_msgs/srv/DetectObjects.srv"
    string[] classes
    float32 max_image_age_s 0.3
    float32 timeout_s 4.0
    ---
    bool success
    string error_code
    string message
    ObjectDetectionArray objects
    ```

- It answers at most 20 objects, the highest scores above `objects_min_score`.
  A detection has `points` with `geometry_enabled`, and a crop, colour shares
  or vector when its flag is on and its box is at least 32 px on its short side.
  The stream's 1-in-10 sampler and its limit of three do not apply.
- The service reads the newest image, depth image and `camera_info` that the
  node holds. It adds no subscriber to the camera. It prefers the newest
  matching pair that is within `max_image_age_s` to a newer image with no depth.
  A pair matches by the stream's bounds in
  [ADR-0014](/architecture/decisions/0014-bounded-rgbd-and-service-work.md),
  wall time included, so a depth from before a sim restart never pairs with a
  new image.
  It keeps no image and no detection after it answers, and returns no id for
  the image. Only the on-demand model's last class list stays loaded. The
  array's `header` gives the image's stamp and frame, as on the topic.
- It uses a second YOLOE instance of its own, without a tracker, built and
  warmed up at start. The streaming detector's classes, its tracks and
  `/walkie_vision/objects` do not change. On-demand detections have
  `track_id` 0.
- It runs on the one model worker and shares the one job slot with `EmbedText`
  and `EmbedCrop` ([ADR-0015](/architecture/decisions/0015-bounded-service-scheduling.md)).
  So a request with a new list stops the stream while `set_classes()` runs.
  The node prepares a new list at most once every 2 s, counted from the end of
  the last preparation, and answers `BUSY` to another new list before that. The
  same ordered list prepares nothing.
- It picks the image after the classes are ready, so a slow preparation does
  not age it. It checks the image's age again before it answers.
- It answers only the caller. Its results are not published on
  `/walkie_vision/objects`.
- It answers `success` false, with a code in `error_code` and the reason in
  `message`, instead of failing. A list that cannot be prepared answers
  `MODEL_UNAVAILABLE`, and a failed detection answers `INFERENCE_FAILED`.

This clarifies ADR-0002 without changing it. Holding the newest image
until the next one replaces it is an input buffer, not memory: no request
can refer to an earlier image. walkie-ros's `ob_pose_service_cpp` does the
same with its latest depth image, for `/get_3d_poses`.

**Consequences**

- The streaming detector's classes and tracks, and walkie-database's negative
  evidence, are not affected by on-demand requests.
- A request with a new list of names pays the on-demand instance's
  `set_classes()` cost, 1.5 s for 5 names on the RTX 2060. A request with
  the same list as the previous one skips the text encoder (Ultralytics
  8.4.182). The agent's timeout for `detect_objects` must allow for it.
- The stream publishes no array during that time, and no empty array stands in
  for the gap. The 5 Hz target does not hold through a new list. A task that
  needs it must not use new lists, or must change this decision after a
  measurement.
- A second YOLOE instance needs GPU memory, and it loads at start with the
  task's classes, so the start takes longer. How much of either is not
  measured. The YOLOE-26s check used 362 MiB in its process. 98 MiB of that is
  the CUDA context, which a second instance in the same process shares.
- The response reuses `ObjectDetectionArray`, so `pipeline.detect_on_demand`
  fills it with the stream's own builders, and shares the stream's `stream_id`.
  `tests/test_convert.py` pins the `.srv`, and a change needs a rebuild in
  every repository that uses the package
  ([ADR-0003](/architecture/decisions/0003-messages-in-this-repository.md)).
- Fakes test the service: `tests/test_yolo.py` proves with two model objects
  that the stream's classes, trackers and track ids do not change, and
  `tests/test_node.py` covers each answer, the cap, the deadline and the image
  choice. No real model has run behind it.
- walkie-agent-ros keeps the response in its own handle store and shows
  its LLM only labels, scores and handle ids. walkie-vision does not need
  to know about handles.
- In simulation, walkie-agent-ros's `oracle_perception` node publishes
  detections from the scenario spec in the same message types.
  Whether it, or something else, also answers `DetectObjects` and
  `EmbedText` there is not decided ([open questions](/architecture/vision.md#open-questions)).
- The service is built. This ADR becomes Accepted once its latency and GPU
  memory are measured on the robot.

**See also:** [On-demand DetectObjects](/architecture/vision.md#on-demand-detectobjects),
[Messages](/architecture/vision.md#messages),
[ADR-0008](/architecture/decisions/0008-model-choices.md)
<!-- --8<-- [end:body] -->
