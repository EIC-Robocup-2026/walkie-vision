# ADR-0004: Run the models in one Pixi environment and two processes

<!-- --8<-- [start:body] -->
**Status:** Accepted · **Date:** 2026-10-19

**Context**

The head camera, a ZED 2i, publishes HD2K images (2208 x 1242) at 15 Hz
(walkie-ros `bringup/config/camera/zed2i.yaml`). In `bgra8` one image is about 11 MB, so the raw
stream is about 165 MB/s. Every process that subscribes to it receives and
deserialises its own copy.

Every process that uses the GPU creates its own CUDA context. That costs
about 0.1 GB on its own, and up to about 0.75 GB once cuDNN and cuBLAS are
loaded.

Most models run on PyTorch. The face models, InsightFace's `buffalo_l`, are
ONNX files that run on ONNX Runtime. Both runtimes load the same CUDA
libraries and work in one process, in either import order (`check-env`,
`c04` and `c10`). A crash inside ONNX Runtime's native code ends the whole
process it runs in, though.

The environment is about 9.6 GB, most of it CUDA libraries and PyTorch.

**Decision**

- All models share one Pixi environment, the default one.
- They run in at most two processes:

| Process       | Models                                                                                               |
| ------------- | ---------------------------------------------------------------------------------------------------- |
| Torch process | YOLOE-26 segmentation with tracking, in text-prompt mode; the object and room embedder with the `EmbedText` service; YOLO26 pose; OSNet |
| Face process  | SCRFD and ArcFace (`buffalo_l`) on `onnxruntime-gpu`                                                  |

- The torch process is the only subscriber to the raw camera image.

**Consequences**

- The 11 MB image is received once, and every PyTorch model works on the same
  copy on the GPU.
- Two CUDA contexts instead of one per model.
- A crash in ONNX Runtime does not stop object detection.
- One lock file to update, and PyTorch and ONNX Runtime must agree on a CUDA
  version ([ADR-0005](/architecture/decisions/0005-cuda-12-sm75-sm120.md)).
- Both processes share one GPU, so their memory together must fit, next to
  whatever else runs on it.
- The face process needs images without subscribing to the raw stream. How
  it gets them (crops from the torch process, or a smaller image), and who
  puts its faces into `PersonDetection`, is still open.

**See also:** [Vision](/architecture/vision.md#processes),
[Build the Vision Environment](/setup/01-vision-environment.md)
<!-- --8<-- [end:body] -->
