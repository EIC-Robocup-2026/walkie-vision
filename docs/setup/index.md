# Setup Log

This section records every step used to build this repository, in order.
Each page explains **what** we ran and **why**, so you can repeat the same
steps in your own project, or when you add or replace a model.

| Step                                                      | What it does                                                                 |
| --------------------------------------------------------- | ---------------------------------------------------------------------------- |
| [1. Build the vision environment](01-vision-environment.md) | Puts ROS 2 on Zenoh, PyTorch, ONNX Runtime and every model in one Pixi environment, for both GPUs |
| [2. Add the message package](02-message-package.md)       | Defines the messages the models publish, and builds them with colcon         |
| [3. Add the documentation site](03-documentation.md)      | Adds this website, built with MkDocs Material                                |
