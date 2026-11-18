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
- `src/walkie_vision/providers.py`: model interfaces and result types.
- `src/walkie_vision/convert.py`: results to messages, with validation.
- `src/walkie_vision/node.py`, `pipeline.py`, `image.py`: the vision node,
  one process for every model (ADR-0010). `yolo.py` and `face/heads.py` are
  the detector, pose and head-crop face providers, including the second
  detector behind `DetectObjects` (ADR-0009). Off by default.
- `src/walkie_vision/depth.py`: pairs colour with registered depth, bounded by
  wall time (ADR-0014), lifts masks to points and scores views. Off by default.
- `src/walkie_vision/crops/objects.py`, `crops/colour.py`: the object sampler,
  padded crops and colour shares. `src/walkie_vision/embeddings.py`: the
  interim CLIP provider. Off by default.
- `src/walkie_vision/requests.py`: bounded service work on the one model worker
  (ADR-0015): the job, its slot, the current image and the request checks.
- `config/`: the node's parameters (`vision.yaml`, `sim.yaml`) and each task's
  classes and flags (`tasks/hri.yaml`, `tasks/memory.yaml`).
  `launch/vision.launch.py` starts the node and restarts it when it exits.
- `src/walkie_vision/fakes.py`: deterministic fakes for tests.
- `src/walkie_vision/fake_publisher.py`, `config/fakes/`: the fake object
  publisher and its scene, for consumers' tests. Synthetic geometry, not
  perception (ADR-0013).
- `src/walkie_vision/face/`, `src/walkie_vision/reid/`: vendored model code.
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
- A message change needs: the `.msg` comments, `convert.py`, the field pins
  in `tests/test_convert.py`, and a rebuild in every consuming repository
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
