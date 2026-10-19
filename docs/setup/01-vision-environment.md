# 1. Build the Vision Environment

**Goal:** one Pixi environment that holds ROS 2 Jazzy on Zenoh together with
every vision model, on both the development laptops' GPU and the robot's, so
a Python process can run the models on the GPU and publish their results as
ROS messages.

## Background

walkie-vision runs the robot's vision models in at most two processes
([ADR-0004](../architecture/decisions/0004-one-environment-two-processes.md)):

| Process       | Models                                                                                  | Runtime      |
| ------------- | --------------------------------------------------------------------------------------- | ------------ |
| Torch process | YOLOE-26 detection and segmentation with tracking, the object image and text embedder, YOLO26 pose, OSNet re-identification | PyTorch      |
| Face process  | SCRFD face detection and ArcFace recognition, from InsightFace's `buffalo_l`             | ONNX Runtime |

Both processes use this one environment. The face process is separate so
that a crash in ONNX Runtime does not stop detection.

The environment has to work on two kinds of GPU:

| Computer            | GPU            | Compute capability | Driver        |
| ------------------- | -------------- | ------------------ | ------------- |
| Development laptops | RTX 2060, 6 GB | `sm_75`            | 595.91        |
| Robot               | RTX 5090 (not yet checked) | `sm_120` | Not yet known |

The robot's GPU is not verified: the only evidence is a comment in
walkie-ros's `yolo_node.py` that names an RTX 5090, and its robot Docker
image, which builds for `TORCH_CUDA_ARCH_LIST=12.0`. See
[On the robot](#on-the-robot).

ROS packages come from conda (RoboStack). The ML packages come from PyPI,
because their GPU builds are published there and on the PyTorch index.
Pixi installs both into the same environment.

## Step 1: ROS 2 and the middleware

```toml title="pixi.toml"
[dependencies]
python = "3.12.*"
ros-jazzy-ros-base = ">=0.11.0,<0.12"
ros-jazzy-rmw-zenoh-cpp = ">=0.2.11,<0.3"
ros-jazzy-vision-msgs = ">=4.1.1,<5"

[activation.env]
RMW_IMPLEMENTATION = "rmw_zenoh_cpp"
```

| Package                   | Why                                                                                                   |
| ------------------------- | ----------------------------------------------------------------------------------------------------- |
| `python = "3.12.*"`       | RoboStack Jazzy is built for Python 3.12.                                                             |
| `ros-jazzy-rmw-zenoh-cpp` | Zenoh is the middleware of every Walkie repository. 0.2.10 and older can block forever in `rmw_wait` ([ros2/rmw_zenoh#1032](https://github.com/ros2/rmw_zenoh/issues/1032)). RoboStack's 0.2.11 comes from the same stack build as `ros-base`, so no source build is needed. It also provides the router, `rmw_zenohd`. |
| `ros-jazzy-vision-msgs`   | `BoundingBox2D`, the box type in this repository's messages ([Add the Message Package](02-message-package.md#step-1-design-the-messages)). |

`[activation.env]` sets `RMW_IMPLEMENTATION` for every `pixi run` and
`pixi shell`, and replaces a value exported in the terminal
([ADR-0007](../architecture/decisions/0007-zenoh-in-every-pixi-environment.md)).
To try another middleware for one command:

```bash
pixi run env RMW_IMPLEMENTATION=rmw_fastrtps_cpp ros2 topic list
```

There is no Zenoh configuration file in this repository. Nodes connect to the
router at `tcp/localhost:7447`, rmw_zenoh's default. On the robot, walkie-ros
starts the router. On a laptop without walkie-ros, start one in its own
terminal:

```bash
pixi run ros2 run rmw_zenoh_cpp rmw_zenohd
```

## Step 2: PyTorch for both GPUs

```toml title="pixi.toml"
[pypi-dependencies]
torch = { version = "==2.11.0", index = "https://download.pytorch.org/whl/cu128" }
torchvision = { version = "==0.26.0", index = "https://download.pytorch.org/whl/cu128" }
```

A PyTorch wheel only runs on the GPUs it was compiled for, and so does
torchvision, which has its own CUDA operators such as `nms`. `cuobjdump`
lists the machine code (SASS) and the portable code (PTX) inside each
library. These are the counts in the installed wheels:

| Library                                    | SASS                                              | PTX          |
| ------------------------------------------ | ------------------------------------------------- | ------------ |
| `torch/lib/libtorch_cuda.so`               | `sm_75` to `sm_120`, 444 kernels each              | none         |
| `torch/lib/libtorch_cuda_linalg.so`        | `sm_50`, `sm_60`, `sm_70`, `sm_80`, `sm_86`, `sm_90`, `sm_100`, `sm_120`: no `sm_75` | none |
| `torchvision/_C.so`                        | `sm_70` to `sm_120`, `sm_75` included             | none         |
| `onnxruntime/capi/libonnxruntime_providers_cuda.so` | `sm_60` to `sm_90a`, 230 modules each    | `compute_90` |

`libtorch_cuda_linalg.so` has no `sm_75` code, but `sm_70` code runs on
`sm_75`: the same major version, and a lower minor one.
`torch.cuda.get_arch_list()` prints
`['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']`.

Why this index and these versions:

- **CUDA 12.8 index.** It is the newest CUDA 12 index whose torchvision
  wheels have `sm_120` code. The CUDA 12.9 index has newer torch wheels with
  `sm_120`, but its torchvision wheels (0.26.0 to 0.28.0) stop at `sm_90`
  and have no PTX, so `torchvision.ops.nms` fails on an RTX 50-series GPU.
  The 12.8 index stops at torch 2.11.
- **CUDA 12, not 13.** The driver decides which builds can run:

| Build      | NVIDIA driver it needs                                                       | Works with every RTX 50-series driver? |
| ---------- | ---------------------------------------------------------------------------- | -------------------------------------- |
| CUDA 12.8  | R570 or newer on an RTX 50-series GPU                                        | Yes                                    |
| CUDA 13.x  | R580 or newer                                                                | No: R570 and R575 drivers cannot run it |

The robot's driver is not known yet, so the environment stays on CUDA 12
([ADR-0005](../architecture/decisions/0005-cuda-12-sm75-sm120.md)).
Move `torch` and `torchvision` together: each torchvision release matches
one torch release. Run `pixi run check-env` before moving to another index.

The CUDA runtime and cuDNN come with PyTorch as `nvidia-*-cu12` wheels. No
system CUDA installation is needed.

## Step 3: ONNX Runtime on the GPU

```toml title="pixi.toml"
[pypi-dependencies]
onnxruntime-gpu = { version = ">=1.26.0,<1.27", extras = ["cuda", "cudnn"] }
onnx = ">=1.23.2,<2"
```

**Version.** 1.26 is the last `onnxruntime-gpu` release on PyPI built for
CUDA 12. 1.27 and newer are CUDA 13 builds (see Step 2). 1.26 is built with
CUDA 12.8, as PyTorch is, and its `cuda` and `cudnn` extras resolve to the
same `nvidia-*-cu12` wheels that PyTorch uses, so there is one copy of each
library.

**Finding the CUDA libraries.** The ONNX Runtime CUDA provider cannot find
the `nvidia-*-cu12` libraries by itself. Without help, it logs an error and
runs the session on the CPU, without raising:

```text
Failed to create CUDAExecutionProvider. Require cuDNN 9.* and CUDA 12.*.
```

Do one of these before the first `InferenceSession`:

| Option                       | When it applies                                    |
| ---------------------------- | -------------------------------------------------- |
| `onnxruntime.preload_dlls()` | Any process. Loads CUDA and cuDNN from the wheels. |
| `import torch`               | Processes that use PyTorch anyway.                 |

`walkie_vision.face.FaceAnalyzer` calls `preload_dlls()` itself when CUDA
is requested, and raises if a session does not run on the first provider it
asked for. In your own code, always check where a session runs:

```python
assert session.get_providers()[0] == "CUDAExecutionProvider"
```

**On an RTX 50-series GPU, the first run compiles.** The CUDA provider has
machine code up to `sm_90a` only, which runs on `sm_90` alone. On `sm_120`,
the driver compiles each module from the `compute_90` PTX the first time a
session uses it, and keeps the result in `~/.nv/ComputeCache`.

- Compiling all 230 modules with `ptxas` 12.8, one after another on one core
  of the laptop's i7-10750H, took 22 minutes. The largest modules took 45 to
  63 seconds each. This is an upper bound: the driver compiles only the
  modules a session uses, on first use, and the face models use a subset.
- The 230 compiled modules take 0.33 GiB. The cache's default limit,
  `CUDA_CACHE_MAXSIZE`, is 1 GiB, so it is not raised.
- A driver update empties the cache. Run `pixi run check-env` once after
  install and after every driver update, to fill it before a test.
  `checks/run_checks.py` allows 3600 seconds per check for this.
- On the RTX 2060 nothing is compiled: the cache stayed at 1 MiB.

If the first run turns out too slow on the robot, the options are ONNX
Runtime's TensorRT provider (compiled into 1.26, but the TensorRT libraries
are not installed), ONNX Runtime 1.27 or newer on CUDA 13 with an R580
driver, or running the two face models through PyTorch.

## Step 4: The face models, without the InsightFace package

`src/walkie_vision/face/` vendors the three files of InsightFace's MIT
library code that the `buffalo_l` detector and recogniser need: `scrfd.py`,
`arcface.py` and `face_align.py`, from GitHub commit `668e8d4` (v0.7.3).
Each file keeps the licence text, the source URL and a list of changes.
`FaceAnalyzer` in `analysis.py` is new code that replaces InsightFace's
`FaceAnalysis` for the detector and the recogniser only
([ADR-0006](../architecture/decisions/0006-face-models-without-insightface.md)).

Why not `pip install insightface`:

- It requires `onnxruntime`, the CPU-only package. That package installs the
  same `onnxruntime` Python module as `onnxruntime-gpu`, and can replace the
  CUDA provider without an error. Pixi 0.80 treats the lock file as out of
  date when that requirement is removed with a dependency override, so
  `pixi install --locked` fails.
- It pulls in scikit-image, only for one similarity transform, and other
  packages the robot does not use.
- The PyPI 2.1 wheel has no licence file. The GitHub tree at a fixed commit
  is clearly MIT.

The vendored code gives the same results. On the CPU, against the
InsightFace 2.1 package, on Ultralytics' `zidane.jpg` and `bus.jpg`: the same
faces, boxes within 1.5e-5 px, identical scores, and embeddings with a cosine
of 0.9999998 or more. `tests/test_face.py` checks the alignment against
matrices computed with scikit-image, and the detector on `zidane.jpg` when
the models are present.

The models come from InsightFace's v0.7 GitHub release.
`pixi run fetch-models` downloads `buffalo_l.zip` (289 MB), checks it and
each of its five files against a SHA-256 in `src/walkie_vision/models.py`,
and extracts them to `models/buffalo_l/`. GitHub publishes no digest for
this old release asset. The checksum is that of the file InsightFace's own
downloader fetched on 2026-10-19.

## Step 5: Ultralytics

```toml title="pixi.toml"
[dependencies]
setuptools = ">=77.0.3,<82"

[activation.env]
YOLO_AUTOINSTALL = "false"

[pypi-dependencies]
opencv-python = ">=5.0.0.93,<6"
ultralytics = ">=8.4.182,<8.5"
lap = ">=0.5.12,<0.6"
clip-anytorch = ">=2.6.0,<3"
```

By default, Ultralytics installs missing packages with pip while it runs.
That would change the environment behind Pixi's back, so
`YOLO_AUTOINSTALL=false` turns it off. Every package it asked for is
declared instead:

| Ultralytics asked for                         | Needed by                                    | Declared as                                                     |
| --------------------------------------------- | -------------------------------------------- | --------------------------------------------------------------- |
| `git+https://github.com/ultralytics/CLIP.git` | YOLOE `set_classes()` (`clip.clip.tokenize`) | `clip-anytorch`, a PyPI release of the same OpenAI CLIP package |
| `lap`                                         | `model.track()` with ByteTrack               | `lap`                                                           |

`clip-anytorch` imports `pkg_resources`, which setuptools removed in 82.
That is the reason for `setuptools<82`. PyTorch needs 77.0.3 or newer.
`opencv-python` comes from PyPI, not conda-forge, so that the process has
one copy of `cv2`.

**Half precision.** In 8.4, `half=True` is deprecated in favour of
`quantize=16`, which behaves the same. Ultralytics builds its predictor on
the first `predict()` call and reuses it, so the precision of an existing
model object cannot be changed later. Create a new model object to switch.

**Weights.** `pixi run fetch-models` downloads `yoloe-26s-seg.pt`,
`yolo26s-pose.pt` and `mobileclip2_b.ts`, the text encoder that
`YOLOE.set_classes()` loads, from the `ultralytics/assets` v8.4.0 release
into `models/ultralytics/`, and checks the SHA-256 digests GitHub publishes
for them. Ultralytics looks for weights in the current directory first, and
`set_classes()` loads `mobileclip2_b.ts` by bare file name. So the checks
change to `models/ultralytics/` before they load a model, and the ROS node
must do the same. Otherwise Ultralytics downloads the files again, into the
current directory or into its own `weights_dir`.

!!! note "Usage statistics"
    Ultralytics sends anonymous usage events after `predict()` when `sync`
    is true in `~/.config/Ultralytics/settings.json` and the computer is
    online. Set `YOLO_OFFLINE=true` in the robot's runtime environment, or
    run `yolo settings sync=False` once per user.

## Step 6: Hugging Face models

```toml title="pixi.toml"
[pypi-dependencies]
transformers = ">=5.20.1,<6"
sentence-transformers = ">=6.1.0,<7"
huggingface-hub = ">=1.31.0,<2"
open-clip-torch = ">=3.3.0,<4"
```

EmbeddingGemma 2 needs transformers 5.18 or newer and sentence-transformers
6.1 or newer. sentence-transformers 6.1 requires `huggingface_hub` below 2.0,
so raise both together.

`HUB_MODELS` in `src/walkie_vision/models.py` pins each repository to one
commit. Pass that `revision` whenever a model is loaded, so that a new
upload cannot change the results without a change in this repository.

| Model                                | Load it with                                                         | Notes |
| ------------------------------------ | -------------------------------------------------------------------- | ----- |
| `google/siglip2-base-patch16-256`    | `AutoModel`, `AutoProcessor`                                         | `get_image_features()` and `get_text_features()` return a `BaseModelOutputWithPooling`. Use `.pooler_output` (768 values). fp16 works. |
| `florence-community/Florence-2-base` | `AutoModelForImageTextToText`, `AutoProcessor`                       | transformers 5 has Florence-2 built in, so no `trust_remote_code`. `microsoft/Florence-2-base` does not load on transformers 5. |
| `google/embeddinggemma-2`            | `SentenceTransformer(..., config_kwargs={"audio_config": None})`     | Text and images share one 768-value space. bfloat16 only on a GPU that has it in hardware (`torch.cuda.is_bf16_supported(including_emulation=False)`), otherwise float32: on the RTX 2060, float32. Encode images as `[{"image": pil_image}]`. |
| CLIP ViT-B/16, OpenAI weights        | `open_clip.create_model_and_transforms("ViT-B-16-quickgelu", ...)` with `timm/vit_base_patch16_clip_224.openai` | The plain `ViT-B-16` config uses GELU and does not match the OpenAI weights. |
| OSNet x1.0, MSMT17 `combineall`      | `walkie_vision.reid.load_osnet()`                                    | The network code is vendored from deep-person-reid (MIT). The loader uses `torch.load(..., weights_only=True)`, drops `classifier.*`, and checks that nothing else is missing. Output is 512 values. |

None of these models is gated, so no Hugging Face login is needed.
`google/embeddinggemma-300m` is gated (manual approval); use
`google/embeddinggemma-2`.

## Where the models are stored

| Models                    | Location                         | Fetched by                          | Size on the laptop |
| ------------------------- | -------------------------------- | ----------------------------------- | ------------------ |
| InsightFace `buffalo_l`   | `models/buffalo_l/`              | `pixi run fetch-models`             | 326 MB             |
| Ultralytics weights       | `models/ultralytics/`            | `pixi run fetch-models`             | 295 MB             |
| Hugging Face models       | `~/.cache/huggingface/hub`       | The first load, for example in `check-env` | SigLIP 2 1.5 GB, EmbeddingGemma 2 1.5 GB, CLIP 571 MB, Florence-2 447 MB, OSNet 17 MB |

`WALKIE_VISION_MODELS` moves `models/` elsewhere. Git ignores `models/`, and
also every `*.onnx`, `*.pt`, `*.pth` and `*.ts` file, so a model file cannot
be committed by accident. The Hugging Face cache stays at its default
location, so every repository on a computer shares one copy.

## Step 7: Check it

```toml title="pixi.toml"
[dependencies]
cuda-cuobjdump = "12.8.*"

[tasks.check-env]
cmd = "python checks/run_checks.py"
depends-on = ["fetch-models"]
```

```bash
pixi install
pixi run check-env
```

`checks/run_checks.py` runs each check in its own process, writes its full
output to `checks/output/`, and writes the GPU, the driver and one line per
check to `checks/output/summary.txt`:

| Check          | What it checks                                                                 |
| -------------- | ------------------------------------------------------------------------------ |
| `c01_fatbins`  | The GPU architectures in every CUDA library, with `cuobjdump`, against this GPU |
| `c02_packages` | No CPU-only `onnxruntime`, no second OpenCV, and no file installed by two packages |
| `c03_torch`    | PyTorch on the GPU, fp32 and fp16, and `torchvision.ops.nms` against the CPU    |
| `c04_*`        | An ONNX Runtime session on the CUDA provider, after `preload_dlls()` and after `import torch` |
| `c05_*`        | YOLOE-26 with text prompts, YOLO26 pose, and tracking                          |
| `c06_*`        | SigLIP 2, Florence-2 and EmbeddingGemma 2                                      |
| `c07_clip`     | CLIP ViT-B/16                                                                  |
| `c08_face`     | SCRFD and ArcFace on CUDA, against the boxes InsightFace finds and the CPU embeddings |
| `c09_osnet`    | OSNet on person crops                                                          |
| `c10_*`        | rclpy on Zenoh next to every ML library, imported in both orders, through a private router |

`cuda-cuobjdump` is pinned to 12.8 because a 13.x `cuobjdump` pulls in
conda's `cuda-version` 13, which declares that it needs a CUDA 13 driver.

The environment is about 9.6 GB. The CUDA libraries and PyTorch are most of
it.

## Measured on the RTX 2060

All 16 checks passed on the laptop (RTX 2060, driver 595.91). Each check
runs one model in its own process. The time is the median after warm-up
unless noted. The memory is what `nvidia-smi` reports for the process at the
end of the check, CUDA context included (98 MiB on its own).

| Model                                  | Time                                          | GPU memory |
| -------------------------------------- | --------------------------------------------- | ---------- |
| YOLOE-26s-seg, `bus.jpg`               | fp32 18.5 ms end to end (12.1 ms inference); `quantize=16` 17.4 ms. The first call takes 0.65 s, or 2.5 s with `quantize=16`. `set_classes()` with 5 names takes 1.5 s | 362 MiB |
| YOLO26s-pose, `bus.jpg`                | fp32 17.2 ms; `quantize=16` 15.2 ms           | 272 MiB    |
| SigLIP 2, 2 images                     | fp16 12.1 ms; fp32 25.8 ms                    | 878 MiB (fp16) |
| Florence-2 base, one caption           | fp16 179 ms; 2.1 s for the first call         | 786 MiB    |
| EmbeddingGemma 2                       | fp32: 3 texts 242 ms, 2 images 878 ms (single calls) | 2180 MiB |
| CLIP ViT-B/16, 2 images                | fp16 autocast 54.3 ms                         | 750 MiB    |
| SCRFD + ArcFace                        | 27.7 ms (`zidane.jpg`) and 26.1 ms (`bus.jpg`), detection plus one embedding per face. Creating both sessions takes 0.6 s | 550 MiB |
| OSNet x1.0, 6 crops of 256 x 128       | 12.5 ms; 0.81 s for the first call            | 226 MiB    |
| rclpy on Zenoh                         | A 1280 x 720 `bgra8` image (3.7 MB): deserialise 2.1 ms, then to RGB on the GPU 0.6 ms | |

The ZED camera publishes HD2K images, 2208 x 1242: about 11 MB each in
`bgra8`, three times the image in the last row.

The results are right, too: SigLIP 2, EmbeddingGemma 2 and CLIP pick the
right caption for both images, OSNet scores the same person 0.93 to 0.97 and
different people 0.37 to 0.41, and ArcFace's CUDA and CPU embeddings have a
cosine of 1.000000.

!!! warning "GPU memory on the laptop is tight"
    Every model in the table together, with the CUDA context counted once,
    adds up to about 5.2 GiB, too close to the RTX 2060's 6 GB. Only one
    object embedder runs on the robot, so the real total is lower: with
    SigLIP 2, roughly 2.6 GiB over the two processes. Both figures are sums
    of single-model measurements. Measure the real total on the robot, with
    the processes as they will run.

The RTX 2060 has no native bfloat16 and gains little from fp16 at these
sizes. Measure again on the robot's GPU before choosing a precision.

## Licences

This repository has no licence of its own yet. The models and libraries it
uses have these:

| What                                                         | Licence                          |
| ------------------------------------------------------------ | -------------------------------- |
| InsightFace `buffalo_l` models (SCRFD, ArcFace)              | **Non-commercial research only** |
| Ultralytics: the library, YOLOE-26, YOLO26 and `mobileclip2_b.ts` | **AGPL-3.0**                |
| The vendored InsightFace code in `src/walkie_vision/face/`   | MIT                              |
| SigLIP 2                                                     | Apache-2.0                       |
| EmbeddingGemma 2                                             | Apache-2.0                       |
| Florence-2                                                   | MIT                              |
| OSNet, the vendored code and the weights                     | MIT                              |
| CLIP ViT-B/16, OpenAI's weights                              | MIT (the timm mirror's model card says Apache-2.0) |

!!! danger "Check before you publish or share"
    `buffalo_l` may be used for non-commercial research only. AGPL-3.0
    requires that anyone who uses a modified Ultralytics over a network can
    get its source. The team must confirm that both are acceptable for
    RoboCup and for this repository's own licence before it is published.

## On the robot

These steps are not done yet. Each one needs the robot.

- [ ] Record the GPU model, its memory and the NVIDIA driver version. The
      driver must be R570 or newer. `pixi run check-env` writes all three to
      `checks/output/summary.txt`.
- [ ] Run `pixi run check-env` once, online, after the install. It
      downloads the Hugging Face models, and the ONNX Runtime checks (`c04`,
      `c08`) fill the driver's compile cache. Record how long `c04` and
      `c08` take on the first run and on the second.
- [ ] Run it again after every NVIDIA driver update, which empties the
      cache.
- [ ] If the first run is too slow, choose one of the options at the end of
      [Step 3](#step-3-onnx-runtime-on-the-gpu).
- [ ] Measure the GPU memory of both processes with every model loaded.

## Known harmless warnings

| Warning                                                                                  | Source                                                       |
| ---------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `routing.router.peers_failover_brokering` is deprecated, and the same for `routing.peer.mode` and `routing.peer.linkstate` | rmw_zenoh 0.2.11's built-in configuration, read by a newer Zenoh library |
| `pkg_resources is deprecated as an API`                                                  | `clip-anytorch`, on every YOLOE run                          |
| `bos_token_id must be None or an integer within the vocabulary`, and the same for `eos_token_id` | Loading SigLIP 2. The results are correct.             |
| `You are sending unauthenticated requests to the HF Hub`                                 | No Hugging Face token. Downloads still work.                 |
