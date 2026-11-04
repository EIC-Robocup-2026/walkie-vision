# ADR-0013: Freeze the memory observation contract

<!-- --8<-- [start:body] -->
**Status:** Accepted · **Date:** 2026-11-05

**Context**

walkie-database will build object memory from the object arrays this node
publishes. It also needs on-demand detection, one object crop's embedding and
text embeddings from this repository. Its builders pin a commit of
`walkie_vision_msgs` and build against it, and a peer built from a different
definition never matches
([ADR-0003](/architecture/decisions/0003-messages-in-this-repository.md)). So
the types have to settle before the code that fills them exists.

A track id alone does not identify a track for a long-lived memory. The id
base ([Track ids](/architecture/vision.md#contract-track-ids)) survives a
quick restart, but it wraps after about 49.7 days and it is not a run
identifier. A fake publisher reuses the same small ids after every restart.

**Decision**

- `ObjectDetection` gains `view_quality` and `color_shares`.
  `ObjectDetectionArray` gains `stream_id`. The service types `DetectObjects`,
  `EmbedCrop` and `DescribeCrop` are new. The people messages,
  `GetPersonCrop` and the fields of `EmbedText` do not change.
- `stream_id` is a UUID that a publisher process makes once, when it starts.
  An empty one is invalid. A track's identity is the source topic, the
  `stream_id`, the kind and the `track_id`. The id is not a handle on a frame
  and cannot fetch an old image.
- An object's points are x, y and z as little-endian float32 metres in the
  optical frame, height 1, dense, at most 2000 points. Fewer than 30 valid
  points give an empty cloud, which still has the fields and the header. The
  array, its `camera_info`, each cloud and each crop carry the same header.
- A sampled crop is a JPEG at quality 90, with 15% padding on each side,
  clipped to the image, and at most 512 px on its longest side. `view_quality`
  is finite in [0, 1]. `color_shares` is empty or holds 12 shares that sum to
  1, in the order of `walkie_vision.crop.COLOR_NAMES`.
- An unsampled field keeps its documented empty value. Nothing uses NaN as a
  sentinel. No map pose, persistent object id, mask or person record travels
  in these messages.
- `convert.check_object_detection_array` holds these rules, and
  `to_object_detection_array` runs it, so a producer cannot publish a
  malformed array. A real model's vector has a known length (`clip_vit_b16`:
  512). A fake's length is free, but it must still be a unit vector.

**Consequences**

- The type hashes of the two object messages change on purpose. Every peer
  must rebuild the messages from a commit at or after this one.
  `tests/test_convert.py` pins each hash, so a later change has to be
  deliberate.
- Each array costs a few more bytes on the wire for the UUID and the empty
  fields. An empty cloud also carries its three field descriptions.
- The real node fills none of the new data yet. Its detections carry empty
  clouds and crops, and zero quality, until later slices add them.
- `DetectObjects`, `EmbedCrop` and `DescribeCrop` are defined, and no node
  serves them yet.
- walkie-agent-ros's `oracle_perception` publishes arrays without a
  `stream_id`. A consumer that holds the rule must refuse them until that
  publisher sets one.
- This ADR records the contract. Later changes mark which parts run; they do
  not change the decision. A change to a field needs a new ADR that supersedes
  this one.

**Synthetic geometry and perception**

`walkie_vision.fake_publisher` publishes arrays under this contract, so that
consumers can be tested without a camera or a GPU. Its boxes, clouds, crops and
colour shares come from analytic boxes in a synthetic picture. A passing run
shows the layout, the stamps, the stream ids and the QoS working. It shows
nothing about detection quality, depth noise, masks or embeddings. Its vectors
are off by default and are hashes under a `fake_` model name, never labelled as
CLIP. Only a run with the real models can show perception working
([The fake publisher](/guides/vision-node.md#fake-publisher)).

**See also:** [Messages](/architecture/vision.md#messages),
[ADR-0003](/architecture/decisions/0003-messages-in-this-repository.md),
[ADR-0009](/architecture/decisions/0009-on-demand-detect-objects.md).
<!-- --8<-- [end:body] -->
