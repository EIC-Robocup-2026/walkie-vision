# walkie-vision

Robot-side ROS 2 repository for Walkie's vision models. The models run on the
robot's GPU and publish over ROS 2 only. The repository is stateless:
walkie-database is the only memory. Read `docs/architecture/vision.md` and
its ADRs before changing the design.

## Commands

```bash
pixi run fetch-models   # model files into models/, checked by SHA-256
pixi run build-msgs     # build ros2_ws/src/walkie_vision_msgs with colcon
pixi run test           # pytest; fakes only, no GPU, no models, Fast DDS
pixi run check-env      # every model on this GPU; logs in checks/output/
pixi run python checks/c12_vision_node.py --memory pair.npz  # the memory task on a recording; GPU, never in check-env
pixi run vision         # the vision node (launch/vision.launch.py); needs a router
pixi run fake-detections  # synthetic object arrays, no models; Fast DDS, domain 77
pixi run docs-build     # mkdocs build --strict; must pass with no warnings
pixi install --locked   # must succeed: pixi.lock is committed
```

Run `build-msgs` once before `test`. A new `pixi run` is needed after a
build, because `scripts/activate.sh` sources the overlay on activation.

## Layout

- `ros2_ws/src/walkie_vision_msgs/`: the messages and the services only.
- `src/walkie_vision/providers/`: model interfaces and result types, one
  `base.py` per vision task, and the vector model names in `vector_tables.py`.
- `src/walkie_vision/messages.py`: results to messages, with validation, and
  the QoS profiles.
- `src/walkie_vision/node.py`, `pipeline.py`: the vision node,
  one process for every model (ADR-0010). `providers/object_detection/yoloe.py`,
  `providers/pose_estimation/yolo_pose.py` and
  `providers/face_recognition/buffalo_l.py` are the detector, pose and
  head-crop face providers, including the second detector behind
  `DetectObjects` (ADR-0009). Off by default.
- `src/walkie_vision/camera/`: the camera input. `camera/image.py` turns images
  into BGR arrays. `camera/depth.py` reads registered depth, and
  `camera/mailboxes.py` pairs colour with it, bounded by wall time (ADR-0014);
  the depth pairing is off by default.
- `src/walkie_vision/geometry.py`: lifts masks to points and scores views. Off
  by default.
- `src/walkie_vision/crops/objects.py`, `crops/colour.py`: the object sampler,
  padded crops and colour shares. `src/walkie_vision/embeddings.py`: the
  interim CLIP provider. Off by default.
- `src/walkie_vision/services/`: bounded service work on the one model worker
  (ADR-0015): the job and its slot (`services/jobs.py`), and each service's
  request checks and responses (`embed_text.py`, `embed_crop.py`,
  `detect_objects.py`, `get_person_crop.py`).
- `config/`: the node's parameters (`vision.yaml`, `sim.yaml`) and each task's
  classes and flags (`tasks/hri.yaml`, `tasks/memory.yaml`).
  `launch/vision.launch.py` starts the node and restarts it when it exits.
- `src/walkie_vision/fakes.py`: deterministic fakes for tests.
- `src/walkie_vision/fake_publisher.py`, `config/fakes/`: the fake object
  publisher and its scene, for consumers' tests. Synthetic geometry, not
  perception (ADR-0013).
- `src/walkie_vision/_vendor/`: vendored model code (InsightFace in
  `_vendor/insightface/`, deep-person-reid in `_vendor/torchreid/`), wrapped
  by `providers/face_recognition/buffalo_l.py` and `providers/reid/osnet.py`.
  Keep the licence headers and the change lists up to date.
- `src/walkie_vision/weights.py`: every downloaded file and Hugging Face
  commit, with checksums and licences.
- `checks/`: the `check-env` checks, one process each. `c12_memory.py` is the
  memory task's explicit mode of check 12; `check-env` does not start it.
- `.github/workflows/`: `test.yml` runs the fake-model tests from a clean
  checkout, and `docs.yml` publishes the site.
- `tools/bakeoff/`, `tests/test_bakeoff*.py`, `docs/guides/bakeoff.md`: the
  embedding bake-off.

## Rules

- Tests run on Fast DDS (`tests/conftest.py`) and use the fakes. They must
  not need a GPU, model files or a Zenoh router.
- Never commit model files, `.pixi/`, `checks/output/`, `site/`,
  `bakeoff-data/` (bake-off recordings may show people) or the colcon
  `build/`, `install/` and `log/` folders. `.gitignore` covers them.
- Never start a Zenoh router on port 7447 in a test or check. `c10` uses a
  private router on a free port.
- Never run `pixi run vision` on a computer joined to someone else's router
  while their vision node runs: consumers would get two arrays per image
  (`docs/guides/vision-node.md`).
- Use only ungated Hugging Face models, pinned to a commit in `HUB_MODELS`,
  and pass that `revision` when loading. Never log in to Hugging Face.
- A message change needs: the `.msg` comments, `messages.py`, the field pins
  in `tests/test_messages.py`, the digests in `tests/test_golden_messages.py`,
  and a rebuild in every consuming repository
  (walkie-database, walkie-agent-ros). Under rmw_zenoh, a mismatched
  definition fails silently.
- A significant decision gets an ADR in `docs/architecture/decisions/`,
  embedded in `docs/architecture/vision.md`. Do not rewrite an accepted
  ADR: supersede it (`docs/architecture/index.md`).
- Links inside an ADR start with `/` (resolved from `docs/`).
- Move `torch` and `torchvision` together, keep CUDA 12 builds, and run
  `check-env` before changing the PyTorch index or `onnxruntime-gpu`.
- Ultralytics loads weights from the working directory. Run YOLOE code from
  `models/ultralytics/`.
- Docs, code comments and commit messages use plain, short sentences.
