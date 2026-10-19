# Walkie Vision

The vision models of Walkie, the EIC RoboCup 2026 team's robot: object
detection, segmentation and tracking (YOLOE-26), object and text embeddings
(SigLIP 2, EmbeddingGemma 2, CLIP), body pose (YOLO26), faces (SCRFD and
ArcFace) and person re-identification (OSNet). They run on the robot's GPU
and talk to the rest of the robot over ROS 2 only. They remember nothing:
walkie-database keeps the memory. [Pixi](https://pixi.sh) installs all of
it, ROS 2 Jazzy, PyTorch and ONNX Runtime included.

The environment, the messages, the model interfaces and their tests are
done. The ROS nodes are not written yet.

**Documentation:** <https://eic-robocup-2026.github.io/walkie-vision/>

- [Getting Started](https://eic-robocup-2026.github.io/walkie-vision/getting-started/):
  install, check the GPU and run the tests, step by step.
- [Setup Log](https://eic-robocup-2026.github.io/walkie-vision/setup/):
  how this repository was built, with explanations.
- [Architecture](https://eic-robocup-2026.github.io/walkie-vision/architecture/):
  the processes, the messages, and the decisions behind them.

## Quick start

You need Linux on x86_64 and an NVIDIA GPU with driver 570 or newer.

```bash
curl -fsSL https://pixi.sh/install.sh | sh  # once per computer, then reopen the terminal
git clone https://github.com/EIC-Robocup-2026/walkie-vision.git
cd walkie-vision
pixi install           # about 9.6 GB
pixi run fetch-models  # about 600 MB into models/
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
| `pixi run fetch-models` | Download the face and Ultralytics model files into `models/` and check their SHA-256. |
| `pixi run build-msgs`   | Build the `walkie_vision_msgs` package in `ros2_ws/` with colcon.            |
| `pixi run test`         | Run the tests. They use fake models and need no GPU, no model files and no router. |
| `pixi run check-env`    | Check the GPU builds, every model and rclpy on this computer. Logs in `checks/output/`. |
| `pixi run bakeoff`      | Compare the object and room embedders on labelled images.                   |
| `pixi run docs`         | Serve the documentation at <http://127.0.0.1:8000>.                          |
| `pixi run docs-build`   | Build the documentation, failing on any warning.                            |

`pixi task list` shows every task.

## Repository layout

| Path                                | What it holds                                                    |
| ----------------------------------- | ---------------------------------------------------------------- |
| `ros2_ws/src/walkie_vision_msgs/`   | The messages and the service, built with colcon                  |
| `src/walkie_vision/providers.py`    | The interfaces every model implements, and their result types    |
| `src/walkie_vision/convert.py`      | Model results to messages, with checks                           |
| `src/walkie_vision/fakes.py`        | Fake models for tests                                            |
| `src/walkie_vision/face/`           | SCRFD and ArcFace on ONNX Runtime (vendored InsightFace code, MIT) |
| `src/walkie_vision/reid/`           | OSNet re-identification (vendored deep-person-reid code, MIT)    |
| `src/walkie_vision/models.py`       | Every model file and Hugging Face commit, with checksums. `fetch-models` runs it. |
| `checks/`                           | The `check-env` checks                                           |
| `tools/bakeoff/`                    | The embedding bake-off                                           |
| `tests/`                            | Tests for the messages, the conversions, the fakes and the face code |
| `scripts/activate.sh`               | Adds the message build to every `pixi run` and `pixi shell`      |
| `models/`                           | The model files. Not in Git.                                     |
| `docs/`                             | The documentation site (MkDocs)                                  |

The repository has no licence yet. The models it downloads have their own:
see [Licences](https://eic-robocup-2026.github.io/walkie-vision/setup/01-vision-environment/#licences).
