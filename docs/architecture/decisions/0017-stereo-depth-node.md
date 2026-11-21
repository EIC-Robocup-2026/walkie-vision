# ADR-0017: Compute depth with Fast-FoundationStereo in a separate, experimental node

<!-- --8<-- [start:body] -->
**Status:** Proposed · **Date:** 2026-11-21

**Context**

The memory task lifts each object's mask to points with the ZED's depth
([Object points](/architecture/vision.md#contract-object-geometry)). On
`scene1`, a recording of a bottle on a table about 0.7 m from the robot's
head camera, the ZED's depth modes blur the bottle's edges along the line of
sight. `stereo/bottle_box.py` in the robot's experiment folder lifts the
bottle's mask with `walkie_vision.depth.lift_mask` from each source and
measures the points across and along the line of sight, 2nd to 98th
percentile. The bottle's true size was not measured. Its visible surface
should span about half its width along the line of sight, so a smaller
depth spread means less blur:

| Depth source                                   | Frames | Fill | Width   | Depth spread | Width SD |
| ---------------------------------------------- | ------ | ---- | ------- | ------------ | -------- |
| This node's processing, scale 0.5 x 0.5        | 14     | 1.00 | 55.7 mm | 30.8 mm      | 0.2 mm   |
| This node's processing, scale 0.5 x 0.333      | 14     | 1.00 | 56.2 mm | 60.4 mm      | 0.1 mm   |
| Fast-FoundationStereo ONNX, scale 1            | 8      | 1.00 | 56.9 mm | 101.4 mm     | 0.2 mm   |
| ZED `NEURAL_LIGHT`                             | 70     | 0.83 | 58.5 mm | 69.0 mm      | 2.2 mm   |
| ZED `NEURAL`                                   | 70     | 0.96 | 58.0 mm | 121.7 mm     | 2.8 mm   |
| ZED `NEURAL_PLUS`                              | 70     | 0.85 | 56.3 mm | 61.4 mm      | 2.5 mm   |
| ZED `ULTRA`                                    | 70     | 0.71 | 57.2 mm | 57.6 mm      | 2.8 mm   |

- NVIDIA publishes an ONNX export of Fast-FoundationStereo's commercial
  checkpoint in an ungated Hugging Face repository,
  `nvidia/c-fast-foundationstereo`, under the NVIDIA Open Model Agreement.
  The research repository's code is under a research licence.
- On the robot's RTX 3080 Ti Laptop GPU, with the robot's other processes
  running, one pair at 1104 x 621 takes 841 ms at the median, and 558 ms at
  1104 x 414. Creating the CUDA session takes about 27 s. Check 13's process
  held 3286 MiB of GPU memory.
- The vision node pairs depth with colour within at most 0.1 s of wall time,
  and holds only the newest image
  ([ADR-0014](/architecture/decisions/0014-bounded-rgbd-and-service-work.md)).
  The ZED publishes its right image only with `video.publish_left_right:
  true`, which is off on the robot.

**Decision**

- A separate process, `walkie_vision.stereo_depth`, computes the depth. It is
  experimental: started by hand with `pixi run stereo-depth`, never by
  `launch/vision.launch.py`.
- It pairs the ZED's left and right images by exact stamp and runs the ONNX
  file on CUDA, at half size by default. It publishes, each with the left
  image's header: the left `camera_info`, that `camera_info` scaled to the
  depth by the pixel-centre rule, the depth as 32FC1 metres with NaN where
  invalid, and last the left image itself, unchanged.
- `task:=memory_ffs` points the vision node at those four topics. The vision
  node does not change: the relayed image and its depth carry one stamp, so
  they pair at once.
- `pixi run fetch-models` downloads the ONNX file from a commit URL and checks
  its SHA-256. No code, comment or constant comes from the research
  repository.

Rejected:

- **Running the model in the vision node.** About 0.8 s per pair on the one
  model worker would stop objects and people for that long
  ([ADR-0015](/architecture/decisions/0015-bounded-service-scheduling.md)).
- **Publishing only the depth.** By the time it is published, the camera's
  image is 0.8 s newer. Pairing them would need a wait over 0.8 s and a
  buffer of images, which ADR-0014 rules out.
- **The ZED's depth modes.** They stay the memory task's default. The table
  shows their blur.
- **The TensorRT provider.** The environment has no `libnvinfer.so.10`.
- **The research repository's PyTorch checkpoint.** Its licence allows
  research only, and at scale 0.5 it gave the same depth spread (31.0 mm) as
  the ONNX file.

**Consequences**

- Under `memory_ffs` the relayed image is about 0.9 s old when the vision node
  receives it, so its arrays are expected to be 1.3 to 1.5 s old: over the 1 s
  that consumers accept ([Freshness](/architecture/vision.md#contract-freshness)).
  walkie-database drops them unless a trial raises its limit. `DetectObjects`
  and `EmbedCrop` accept images up to 1 s old, so they answer `STALE_IMAGE`.
- The stereo node is a second subscriber to the raw left image, and the only
  one to the right image: two 11 MB streams at up to 15 Hz. It also holds a
  second CUDA context. ADR-0010's one copy of the image and one CUDA context
  do not hold while it runs.
- A scale of 0.333 on y saves about 280 ms per pair, but doubles the bottle's
  depth spread. Measure on more objects before using it to save time.
- Pairing needs the ZED's left and right images to carry equal stamps. That
  is not yet checked on the robot.
- The model file adds 103 MB to `fetch-models` on every computer, whether or
  not it runs the node.

**See also:** [Stereo depth](/guides/vision-node.md#stereo-depth),
[Depth pairing](/architecture/vision.md#depth-pairing),
[ADR-0010](/architecture/decisions/0010-one-vision-process.md),
[ADR-0014](/architecture/decisions/0014-bounded-rgbd-and-service-work.md).
<!-- --8<-- [end:body] -->
