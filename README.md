# Walkie Vision

The vision models of Walkie, the EIC RoboCup 2026 team's robot: object
detection, segmentation and tracking (YOLOE-26), object and text embeddings
(SigLIP 2, EmbeddingGemma 2, CLIP), body pose (YOLO26), faces (SCRFD and
ArcFace) and person re-identification (OSNet). They run on the robot's GPU
and talk to the rest of the robot over ROS 2 only. They remember nothing:
walkie-database keeps the memory. [Pixi](https://pixi.sh) installs all of
it, ROS 2 Jazzy, PyTorch and ONNX Runtime included.

The vision node (`pixi run vision`) publishes objects and people, and
serves one person's crop on request (`GetPersonCrop`). With
`object_embeddings_enabled`, it also serves `EmbedText` and `EmbedCrop`.

**Documentation:** <https://eic-robocup-2026.github.io/walkie-vision/>

- [Getting Started](https://eic-robocup-2026.github.io/walkie-vision/getting-started/):
  install, check the GPU and run the tests, step by step.
- [Setup Log](https://eic-robocup-2026.github.io/walkie-vision/setup/):
  how this repository was built, with explanations.
- [Architecture](https://eic-robocup-2026.github.io/walkie-vision/architecture/):
  the vision node, its contract, the messages, and the decisions.
- [Run the vision node](https://eic-robocup-2026.github.io/walkie-vision/guides/vision-node/):
  start it on the robot or in simulation, read its log, and troubleshoot.

## Quick start

You need Linux on x86_64 and an NVIDIA GPU with driver 570 or newer.

```bash
curl -fsSL https://pixi.sh/install.sh | sh  # once per computer, then reopen the terminal
git clone https://github.com/EIC-Robocup-2026/walkie-vision.git
cd walkie-vision
pixi install           # about 9.6 GB
pixi run fetch-models  # about 600 MB into models/, and CLIP into the Hugging Face cache
pixi run build-msgs    # builds walkie_vision_msgs in ros2_ws/
pixi run test
```

Check every model on this computer's GPU. The Hugging Face models (about
4 GB) download on the first run:

```bash
pixi run check-env
```

Every `pixi run` uses the Zenoh middleware, like walkie-ros and the robot.
Nodes see each other only through a Zenoh router on the same computer. The
robot and walkie-ros's simulation already run one. Elsewhere, start it
first, in its own terminal:

```bash
pixi run ros2 run rmw_zenoh_cpp rmw_zenohd
```

## Tasks

| Command                 | What it does                                                                 |
| ----------------------- | ---------------------------------------------------------------------------- |
| `pixi run fetch-models` | Download the face and Ultralytics model files into `models/` and check their SHA-256. Fetch the CLIP weights into the Hugging Face cache. |
| `pixi run build-msgs`   | Build the `walkie_vision_msgs` package in `ros2_ws/` with colcon.            |
| `pixi run vision`       | Run the vision node, restarted if it exits. `task:=hri` or `task:=memory`, `sim:=true`. The memory task also serves `DetectObjects`; its cost is not measured yet. |
| `pixi run fake-detections` | Publish synthetic object arrays with no camera, GPU or model, on Fast DDS in domain 77. |
| `pixi run test`         | Run the tests. They use fake models and need no GPU, no model files and no router. `.github/workflows/test.yml` runs them from a clean checkout. |
| `pixi run check-env`    | Run the 18 checks: the GPU builds, every model, the vision node's models and rclpy on this computer. Logs in `checks/output/`. The memory task's check is separate and needs a recording: `python checks/c12_vision_node.py --memory pair.npz`. |
| `pixi run bakeoff`      | Compare the object and room embedders on labelled images.                   |
| `pixi run docs`         | Serve the documentation at <http://127.0.0.1:8000>.                          |
| `pixi run docs-build`   | Build the documentation, failing on any warning.                            |

`pixi task list` shows every task.

## Repository layout

| Path                                | What it holds                                                    |
| ----------------------------------- | ---------------------------------------------------------------- |
| `ros2_ws/src/walkie_vision_msgs/`   | The messages and the services, built with colcon                 |
| `src/walkie_vision/`                | The vision package, laid out by concept (below)                  |
| `config/`                           | The vision node's parameters, and one file per task with its classes |
| `launch/`                           | The launch file that runs the vision node and restarts it        |
| `checks/`                           | The environment checks: `check-env`, and the memory task's explicit mode of check 12 |
| `tools/`                            | Standalone programs and scored runs. The embedding bake-off is in `tools/bakeoff/` |
| `scripts/`                          | Shell helpers for the environment. `scripts/activate.sh` adds the message build to every `pixi run` and `pixi shell` |
| `tests/`                            | The tests, with fakes and no GPU                                 |
| `.github/workflows/`                | The test workflow and the documentation workflow                 |
| `models/`                           | The model files. Not in Git.                                     |
| `docs/`                             | The documentation site (MkDocs)                                  |

A new file goes by what it is: a shell helper for the environment in
`scripts/`, a standalone program or a scored run in `tools/`, and an
environment check in `checks/`. This repository has no `evals/` folder.

### The vision package

The package's modules form layers, top first. A module may import only from
a lower layer, and never from its own layer, except that the modules in
layer 4 share a few names. import-linter checks these rules in the tests,
with the contracts in `pyproject.toml`
([ADR-0017](docs/architecture/decisions/0017-package-layout.md)).

| Layer | Path in `src/walkie_vision/` | What goes there | Who may import it |
| ----- | ---------------------------- | --------------- | ----------------- |
| 1 | `node.py`                   | The vision node: threads, publishing and health checks | Nothing. The launch file runs it |
| 1 | `debug_viz.py`              | The optional debug viewer (`viz:=true`) | Nothing. The launch file runs it |
| 2 | `loading.py`                | Builds every model the node runs from the registries, after checking their files | Layer 1 |
| 2 | `fake_publisher.py`         | Synthetic object arrays from a scene file, for consumers' tests | Layer 1. Consumers run it as a program |
| 3 | `pipeline.py`               | One image through every provider to the two arrays, without a ROS node | Layers 1 and 2 |
| 3 | `services/`                 | Bounded service work: the job, its one slot, and each service's request checks and responses | Layers 1 and 2 |
| 3 | `fakes.py`                  | A fake for each model interface, for tests without a GPU | Layers 1 and 2, and the tests |
| 4 | `messages.py`               | Model results to messages, with checks, and the QoS profiles | Layers 1 to 3, and layer 4 |
| 4 | `camera/`                   | Camera images to BGR arrays, the `camera_info` checks, registered depth, and colour and depth paired within bounds | Layers 1 to 3, and layer 4 |
| 4 | `crops/`                    | One person's crop for `GetPersonCrop`, the object sampler, padded crops, colour shares and JPEG encoding | Layers 1 to 3, and layer 4 |
| 4 | `geometry.py`               | Object points and view quality from registered depth. Off by default | Layers 1 to 3, and layer 4 |
| 4 | `config.py`                 | The vision node's parameters, with their checks | Layers 1 to 3, and layer 4 |
| 5 | `providers/`                | The model interfaces and result types (`base.py` per vision task), one provider module per model under `providers/<vision task>/`, and each vision task's `registry.py` | Layers 1 to 4. A module inside `providers/` imports the module that defines a name, never the package |
| 6 | `boxes.py`                  | Box and mask maths | Every layer above |
| 6 | `weights.py`                | Every model file and Hugging Face commit, with checksums and licences. `fetch-models` runs it | Every layer above |
| 6 | `errors.py`                 | `StartupError` | Every layer above |
| none | `_vendor/`               | Vendored MIT code: SCRFD, ArcFace and the alignment from InsightFace, OSNet from deep-person-reid | Only its wrapper: `providers/face_recognition/buffalo_l.py` for InsightFace, `providers/reid/osnet.py` for OSNet |

The providers and the vendored code import no ROS package. The vision tasks
are `object_detection`, `pose_estimation`, `face_recognition`, `reid` and
`image_embed`. Each provider is named in its task's `registry.py`, and the
parameter `<vision task>.provider` picks one.

## Licence

Everything in this repository, code and documentation, is licensed under the
GNU Affero General Public License, version 3 only (`AGPL-3.0-only`): see
[LICENSE](LICENSE). The vendored files in `src/walkie_vision/_vendor/` keep
their MIT licence. AGPL-3.0 matches
Ultralytics, which the vision node depends on. The models it downloads keep
their own licences: see [Licences](https://eic-robocup-2026.github.io/walkie-vision/setup/01-vision-environment/#licences).
