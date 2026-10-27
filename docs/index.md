# Walkie Vision

Walkie's vision models, developed by the EIC RoboCup 2026 team. They run on
the robot's GPU and talk to the rest of the robot over ROS 2 only:

- **objects**: open-vocabulary detection, segmentation and tracking
  (YOLOE-26), with an image embedding for every object,
- **people**: body pose (YOLO26 pose), faces (SCRFD and ArcFace) and
  appearance for re-identification (OSNet), and
- **text search**: a service that embeds a text, such as "a red mug", into
  the same space as the object images.

walkie-vision remembers nothing. It publishes what it sees in every camera
image, and [walkie-database](https://github.com/EIC-Robocup-2026/walkie-database)
keeps the memory.

!!! note "What runs today"
    The vision node runs: it publishes objects and people from the head
    camera, and serves one person's crop on request
    ([Run the vision node](guides/vision-node.md)). The text search service
    (`EmbedText`) and the object embeddings are not built yet.

This site is also a learning guide. It shows how the project was set up, step
by step, so new members can learn how it works and change it safely.

## For next generations

1. [Getting Started](getting-started.md): install the tools, check the GPU
   and run the tests on your computer.
2. [Setup Log](setup/index.md): every step used to build this repository,
   with explanations.
3. [Architecture](architecture/index.md): how the vision node runs the
   models, its contract, the messages, and the decisions behind them.
4. [Embedding bake-off](guides/bakeoff.md): how to choose the model that
   embeds object images, with images recorded on the robot.
