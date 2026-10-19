# Getting Started

This page gets the project running on your machine. Every command is meant to
be copied and pasted. If you want to know _how_ the project was built step by
step, read the [Setup Log](setup/index.md) after this.

## 1. Install Pixi

Pixi is the tool that installs ROS 2, PyTorch, ONNX Runtime and everything
else for this project. Install it once per computer:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Close and reopen your terminal, then check it works:

```bash
pixi --version
```

!!! note "Requirements"
    You need Linux on an x86_64 (normal Intel/AMD) computer, with an NVIDIA
    GPU and a driver that supports CUDA 12.8: version 570 or newer. Check
    with `nvidia-smi`. You do **not** need to install ROS 2, CUDA or cuDNN
    yourself. Pixi does that for you.

## 2. Get the code

```bash
git clone https://github.com/EIC-Robocup-2026/walkie-vision.git
cd walkie-vision
pixi install
```

!!! warning "Not on GitHub yet"
    The repository exists only on a team laptop for now. Until it is
    published, get a copy from a team member.

`pixi install` downloads everything listed in `pixi.toml` into a hidden
`.pixi/` folder inside the project. Nothing is installed system-wide. The
environment is about 9.6 GB, most of it CUDA libraries and PyTorch, so the
first install takes several minutes.

## 3. Download the models

```bash
pixi run fetch-models
```

This downloads about 600 MB into `models/`: InsightFace's `buffalo_l` face
models and three Ultralytics files. It checks every file against its
SHA-256, and skips a file that is already there, so it also passes offline
once it has run. Git ignores `models/`.

The Hugging Face models (SigLIP 2, Florence-2, EmbeddingGemma 2, OSNet and
CLIP, about 4 GB) are not downloaded here. They download into
`~/.cache/huggingface` the first time they are used, in the next step.

## 4. Build the messages

```bash
pixi run build-msgs
```

This builds the `walkie_vision_msgs` package in `ros2_ws/` with colcon. Every
later `pixi run` and `pixi shell` can then import it. Run it again after you
change a file in `ros2_ws/src/walkie_vision_msgs/`.

## 5. Check this computer

```bash
pixi run check-env
```

`check-env` runs `fetch-models` first, then 16 checks, each in its own
process: the GPU architectures compiled into PyTorch and ONNX Runtime, every
model on the GPU, and rclpy next to the models. It prints one line per
check:

```text
c01_fatbins            PASS     8.8 s  exit 0  checks/output/c01_fatbins.log
...
16 of 16 checks passed
```

The full output of each check, with its timings and GPU memory, is in
`checks/output/`. `checks/output/summary.txt` also records the GPU and the
driver version. To run only some checks, give the start of their names:
`pixi run python checks/run_checks.py c05 c08`.

!!! warning "The first run on an RTX 50-series GPU is slow"
    ONNX Runtime has no machine code for these GPUs, so the driver compiles
    it the first time, and keeps the result in `~/.nv/ComputeCache`. This
    can take many minutes. Run `check-env` once after you install, and again
    after every driver update, which empties the cache. See
    [Build the Vision Environment](setup/01-vision-environment.md#step-3-onnx-runtime-on-the-gpu).

## 6. Run the tests

```bash
pixi run test
```

The tests use fake models, so they need no GPU and no model files. They run
on Fast DDS, so they need no Zenoh router either. All tests should pass. If
you skipped step 4, they stop at once and tell you to run `build-msgs`.

## 7. Talk to other nodes

Every `pixi run` uses the Zenoh middleware, like walkie-ros and the robot
([ADR-0007](architecture/decisions/0007-zenoh-in-every-pixi-environment.md)).
Under Zenoh, nodes find each other only through a **Zenoh router**, one per
computer. On the robot, or next to walkie-ros's simulation, a router is
already running: use it. Otherwise, start one in its own terminal:

```bash title="Terminal 1: Zenoh router"
pixi run ros2 run rmw_zenoh_cpp rmw_zenohd
```

```bash title="Terminal 2"
pixi run ros2 topic list
```

There are no vision nodes to run yet. Until then, this is how to see the
robot's topics, such as the camera images, from this environment.

!!! tip "Use the same domain in every terminal"
    Nodes on different `ROS_DOMAIN_ID`s never see each other. Use the
    domain of the nodes you want to reach: `export ROS_DOMAIN_ID=23` on the
    robot. For walkie-ros's simulation, use the domain it was started with.

`check-env` needs no router. Its ROS check starts a private router on a free
port, so it never joins the robot's ROS graph.

## 8. Open the docs locally

```bash
pixi run docs
```

Open <http://127.0.0.1:8000> in your browser. The page reloads by itself when
you edit a file in `docs/`. Press ++ctrl+c++ to stop.

## Next

- [Setup Log](setup/index.md): how this repository was built, one step at a
  time.
- [Vision](architecture/vision.md): the processes, topics and messages.
- [Writing Docs](setup/03-documentation.md#writing-a-new-page): how to add
  your own page.
