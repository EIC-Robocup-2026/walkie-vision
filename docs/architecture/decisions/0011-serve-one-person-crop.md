# ADR-0011: Serve one person crop on request

<!-- --8<-- [start:body] -->
**Status:** Accepted · **Date:** 2026-10-27

**Context**

In the HRI task, walkie-agent-ros describes the first guest to the second
guest. Its planned `describe_person` atomic asks a cloud vision-language
model (VLM) for strict enum answers about one person, such as the top's
colour and type, and cross-checks the colours against pixel colours
measured on the robot. When the network is down, it uses the pixel colours
alone. It needs, for one tracked person, from one image:

- a small JPEG crop of that person, with other people hidden where that is
  possible;
- the share of each colour in the person's top and bottom;
- the person's detection in that image, with the image's stamp, so it can
  check that the track id still names the same person.

Several things constrain where and how this runs:

- walkie-agent-ros ADR-0028 (Proposed) keeps raw camera images out of the
  agent: when a model needs to see something, a robot-side service returns
  one compressed snapshot on request. walkie-agent-ros ADR-0034 puts the
  vision models in walkie-vision. The vision node is
  the only subscriber to the raw camera image
  ([ADR-0010](/architecture/decisions/0010-one-vision-process.md)), and it
  keeps nothing between requests
  ([ADR-0002](/architecture/decisions/0002-ros2-only-stateless.md)).
- Colours measured on the full-resolution image, before scaling and JPEG
  compression, are more accurate than colours measured on the crop.
- Box overlap alone cannot tell a person in front from a person behind. In
  walkie-agent-ros's HRI scenario, a bystander stands behind the guest's
  spot in the door view. With the guest at rest, the bystander's box covers
  about 37% of the guest's torso width, at its full height. While the guest
  paces, the cover stays above 30% for about half of the cycle. A plain
  overlap gate would mark most views as occluded.
- The node spun a `SingleThreadedExecutor`. A service callback that waits
  for the worker would stop the image callback, so the worker would never
  get the image it waits for.
- This is the node's first service. `EmbedText` is defined but not served,
  and `DetectObjects`
  ([ADR-0009](/architecture/decisions/0009-on-demand-detect-objects.md)) is
  Proposed.

walkie-vision's open question 6 asks for a different service: one annotated
full image with numbered boxes for a VLM, with text regions masked at
Competition level so that a sign cannot instruct the robot
([Open questions](/architecture/vision.md#open-questions)).

**Decision**

- Add a service type `GetPersonCrop` to `walkie_vision_msgs`. The vision
  node serves it on `person_crop_service`: `/walkie_vision/get_person_crop`,
  or `/walkie_vision/sim/get_person_crop` with `config/sim.yaml`.
- **From the next image.** The request names a published track id. The
  service callback leaves it in a one-slot mailbox and waits up to
  `timeout_s` of wall time, at most 5 s. The worker takes it once the
  people of the next image are found, publishes that image's two arrays
  first, and then answers from that image and the people it published for
  it. The crop, the keypoints, the colour shares and the person all come
  from one image, and the response carries that image's header. The service
  thread runs no model and touches no image.
- **Two executor threads.** `main()` spins a
  `MultiThreadedExecutor(num_threads=2)`. The service has its own
  `MutuallyExclusiveCallbackGroup`, so the image, `camera_info` and timer
  callbacks keep running while a request waits. Those three stay in the
  node's default group and still run one at a time. Requests are answered
  one at a time.
- **Gates.** The statuses are OK, `NOT_IN_VIEW`, `OCCLUDED`, `TOO_SMALL`
  (a box under 160 px tall), `TIMEOUT` (images arrived, none was processed
  in time), `NO_IMAGE` (no image arrived) and `FAILED` (a refused request,
  a shutdown, or a crop that raised).
- **Only a nearer person occludes.** Another person is nearer when their
  box ends at least 3% of the image height lower, because their feet are
  closer to the camera, or when only their box is cut off by the image's
  bottom edge. When both boxes are cut off, the other person is nearer only
  if they hide one of the guest's torso keypoints. `OCCLUDED` when a nearer
  person's box has an IoU above 0.3 with the guest's box, or covers more
  than 30% of the guest's torso box. The second test catches a smaller
  person in front, whose IoU stays low. People behind are only masked.
- **The mask spares the guest.** Every other detected person's box inside
  the crop, minus the guest's own box, is filled with grey, and
  `masked_boxes` lists the rectangles. The guest's own pixels are never
  greyed. Another person's pixels inside the guest's box, and people the
  detector missed, stay in the crop.
- **Colour shares on the robot.** The torso (the polygon through the
  shoulders and the hips, shrunk by 20%) and the thighs (hips to knees),
  clipped to the guest's box and without greyed pixels, are binned in HSV
  into the 12 colours of the agent's schema. A region whose four keypoints
  do not all score at least `min_keypoint_score` gets no shares. There is
  no fallback to a share of the box.
- **The crop** is the box padded on each side by 10% of its size, scaled
  down to at most 512 px, and encoded as a JPEG at quality 90. The request
  can change the padding and the size.
- Nothing is kept after the answer.

**Consequences**

- The agent gets one compressed snapshot of one person on request, as
  walkie-agent-ros ADR-0028 allows. No image is published on a topic, and
  the node holds no image beyond the one it answers from.
- On a requested image, the worker also spends the crop's time, after the
  arrays are published. The next image waits that long. The latency, the
  JPEG size and the image rate during requests are not measured on the
  robot yet.
- A guest standing in front of a bystander stays describable, but the
  bystander's pixels inside the guest's box are sent with the crop.
  Keypoint silhouettes could tighten the mask later.
- The nearer test reads only box bottoms and keypoint scores. It assumes
  people stand on one floor in front of the camera. Its thresholds, the
  160 px floor and the HSV bins are first values, to be tuned on photos
  from the robot.
- This repository sends nothing off the robot. walkie-agent-ros sends the
  crop to the cloud VLM. That VLM is External Computing: it is declared to
  the Technical Committee one month ahead, with all External Computing
  (`ExternalDevices.tex:18`), and shown at Robot Inspection in addition.
- Open question 6 stays open. `GetPersonCrop` does not answer it: it sends
  one masked crop for enum answers, not an annotated full image with text
  masking.
- The new `.srv` needs a rebuild in every repository that uses the package
  ([ADR-0003](/architecture/decisions/0003-messages-in-this-repository.md)).
  In simulation, walkie-agent-ros's track ids come from its
  `oracle_perception`, so the node serves under the sim name, and the agent
  side must answer `/walkie_vision/get_person_crop` itself.

**See also:** [Person crops](/architecture/vision.md#person-crops),
[GetPersonCrop](/architecture/vision.md#getpersoncrop),
[Threads](/architecture/vision.md#threads)
<!-- --8<-- [end:body] -->
