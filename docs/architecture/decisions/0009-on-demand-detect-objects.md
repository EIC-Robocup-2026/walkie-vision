# ADR-0009: Add an on-demand DetectObjects service for detecting by name

<!-- --8<-- [start:body] -->
**Status:** Proposed · **Date:** 2026-10-22

**Context**

walkie-agent-ros proposes a self-evolving agent (walkie-agent-ros
`docs/architecture/self-evolving-agent.md`, Proposed). It finds an object
in steps: it asks walkie-database's memory first, then detects the object
by its noun in what the head camera sees now, then ranks the detections by
their embeddings against the spoken description. Its `detect_objects` tool
needs to say "look for `mug` now" and get detections back.

Nothing in walkie-vision's design answers that. The planned torch process
([ADR-0004](/architecture/decisions/0004-one-environment-two-processes.md))
will run YOLOE-26 on every processed image, with a list of text prompts set
per task, and publish the result on `/walkie_vision/objects`. A noun that
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
  torch process at `/walkie_vision/detect_objects` (proposed name).
- The request names the classes to look for: the nouns, at least one. The
  response says whether it worked, why not, and holds one
  `ObjectDetectionArray` for the newest image. Its `classes` are the
  request's names. Every detection carries its box, its embedding from the
  object embedder, its points and, when the topic sends one, its crop, as
  on the topic. A sketch, with the exact `.srv` left to the implementation:

    ```text title="walkie_vision_msgs/srv/DetectObjects.srv (sketch)"
    string[] classes
    ---
    bool success
    string message
    ObjectDetectionArray objects
    ```

- The service runs on the newest RGB image, depth image and `camera_info`
  that the torch process will hold. It adds no subscriber to the camera.
  It keeps no image and no detection after it answers, and returns no id
  for the image. Only the on-demand model's last class list stays loaded,
  as a cache. The array's `header` gives the image's stamp and frame, as on
  the topic.
- It uses a second YOLOE instance of its own, without a tracker. The
  streaming detector's classes, its tracks and `/walkie_vision/objects` do
  not change. On-demand detections have `track_id` 0.
- It answers only the caller. Its results are not published on
  `/walkie_vision/objects`.
- It answers `success` false, with the reason in `message`, when
  `classes` is empty, when no image has arrived yet, or when the model
  fails. It answers instead of failing, as `fill_embed_text_response` does
  for `EmbedText`.

This clarifies ADR-0002 without changing it. Holding the newest image
until the next one replaces it is an input buffer, not memory: no request
can refer to an earlier image. walkie-ros's `ob_pose_service_cpp` does the
same with its latest depth image, for `/get_3d_poses`.

**Consequences**

- The streaming detector and walkie-database's negative evidence are not
  affected by on-demand requests.
- A request with a new list of names pays the on-demand instance's
  `set_classes()` cost, 1.5 s for 5 names on the RTX 2060. A request with
  the same list as the previous one skips the text encoder (Ultralytics
  8.4.182). The agent's timeout for `detect_objects` must allow for it.
- A second YOLOE instance needs GPU memory. How much is not measured. The
  YOLOE-26s check used 362 MiB in its process. 98 MiB of that is the CUDA
  context, which a second instance in the same process shares.
- The response reuses `ObjectDetectionArray`, so
  `to_object_detection_array` in `walkie_vision.convert` fills it, and
  `tests/test_convert.py` already pins its fields. The new `.srv` still
  needs its own pins, and a rebuild in every repository that uses the
  package ([ADR-0003](/architecture/decisions/0003-messages-in-this-repository.md)).
- walkie-agent-ros keeps the response in its own handle store and shows
  its LLM only labels, scores and handle ids. walkie-vision does not need
  to know about handles.
- In simulation, walkie-agent-ros's planned `oracle_perception` node
  publishes detections from the scenario spec in the same message types.
  Whether it, or something else, also answers `DetectObjects` and
  `EmbedText` there is not decided ([open questions](/architecture/vision.md#open-questions)).
- This ADR becomes Accepted once the service is built and its latency and
  GPU memory are measured on the robot.

**See also:** [On-demand DetectObjects](/architecture/vision.md#proposed-on-demand-detectobjects-for-the-self-evolving-agent),
[Messages](/architecture/vision.md#messages),
[ADR-0008](/architecture/decisions/0008-model-choices.md)
<!-- --8<-- [end:body] -->
