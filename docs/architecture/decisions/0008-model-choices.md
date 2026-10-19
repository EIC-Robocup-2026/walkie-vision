# ADR-0008: Model choices

<!-- --8<-- [start:body] -->
**Status:** Proposed · **Date:** 2026-10-19

**Context**

Each vision task needs one model on the robot. Every candidate below passed
`pixi run check-env` on an RTX 2060 (6 GB): it loads from a pinned commit or
a checked file, runs on the GPU, and gives the expected answer on two test
images. The robot's GPU has not been measured yet.

walkie-database stores the object and room vectors of one model per table
(walkie-database ADR-0007, Proposed). Its tables today hold CLIP ViT-B/16
vectors (`object_clip_vit_b16`, `room_clip_vit_b16`). A new embedder means a
new table and a migration there.

**Decision**

| Task                        | Proposed model                                     | Alternatives and why                                                     |
| --------------------------- | -------------------------------------------------- | ------------------------------------------------------------------------ |
| Object and room embedding, and text search | SigLIP 2 B/16 at 256 px (`google/siglip2-base-patch16-256`), 768 values | EmbeddingGemma 2 (`google/embeddinggemma-2`) is the challenger: one 768-value space for text and images, but 2.2 GB of GPU memory in fp32 and 0.9 s for two images on the laptop. CLIP ViT-B/16 is the baseline, because walkie-database's tables hold its vectors today. |
| Image captions              | Florence-2 base (`florence-community/Florence-2-base`) | 0.18 s per caption, 0.8 GB, MIT. Built into transformers 5, so no remote code runs. |
| Object detection and segmentation | YOLOE-26 s or m, in text-prompt mode, with ByteTrack | Open vocabulary: the class names are text prompts, set per task. The s model is measured; m if the GPU has room. |
| Body pose                   | YOLO26s-pose, or m if the GPU has room             | The same library and tracker as the detector, 17 COCO keypoints.         |
| Faces                       | InsightFace `buffalo_l`: SCRFD 10G and ArcFace `w600k_r50` | 512-value face embeddings ([ADR-0006](/architecture/decisions/0006-face-models-without-insightface.md)). |
| Appearance (re-identification) | OSNet x1.0, MSMT17 `combineall` weights        | 512 values, 13 ms for 6 crops, MIT. Recognises a person whose face is not visible. |

The object embedder is chosen by the
[embedding bake-off](/guides/bakeoff.md), on images recorded on the robot.
The model sizes are chosen after the GPU memory of both processes has been
measured on the robot.

**Consequences**

- Until the bake-off decides, CLIP ViT-B/16 stays the embedder that
  walkie-database can store.
- Moving to SigLIP 2 or EmbeddingGemma 2 needs a new vector table in
  walkie-database, and either new vectors for every stored object or a fresh
  memory. The model name in every message stops a vector from reaching the
  wrong table.
- `buffalo_l` is for non-commercial research only, and Ultralytics (YOLOE,
  YOLO26) is AGPL-3.0. Both must be acceptable before the repository is
  published under a licence.
- The m models are not in `pixi run fetch-models` yet.
- This ADR becomes Accepted, with the measured numbers, once the bake-off
  and the robot measurement are done.

**See also:** [Measured on the RTX 2060](/setup/01-vision-environment.md#measured-on-the-rtx-2060),
[Licences](/setup/01-vision-environment.md#licences)
<!-- --8<-- [end:body] -->
