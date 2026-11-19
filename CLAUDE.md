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

`src/walkie_vision/` is laid out by concept, in layers listed top first
(ADR-0017). A module imports only from a lower layer; the contracts in
`pyproject.toml` check this in `tests/test_import_contracts.py`.

1. `node.py` (the vision node, one process for every model, ADR-0010) and
   `debug_viz.py` (the optional viewer). Nothing imports them.
2. `loading.py` (builds the providers from the registries after checking
   their files) and `fake_publisher.py` with `config/fakes/` (synthetic
   object arrays for consumers' tests, ADR-0013).
3. `pipeline.py` (one image to the two arrays, no ROS node), `services/`
   (bounded service work on the one model worker, ADR-0015) and `fakes.py`
   (deterministic fakes for tests).
4. `messages.py` (results to messages, with checks, and the QoS profiles),
   `camera/` (BGR conversion, registered depth, colour and depth pairing,
   ADR-0014), `crops/` (person crops, the object sampler, colour shares),
   `geometry.py` (points and view quality from depth) and `config.py` (the
   node's parameters). These may share names with each other.
5. `providers/`: model interfaces and result types (`base.py` per vision
   task), one module per model under `providers/<vision task>/`, and each
   task's `registry.py`, which maps provider names to a `ProviderSpec`
   (`spec.py`). A provider module never imports its registry, and a module
   inside `providers/` never imports the `providers` package itself.
6. `boxes.py` (box and mask maths), `weights.py` (every model file and
   Hugging Face commit, with checksums and licences) and `errors.py`.

Outside the layers, `_vendor/` holds vendored model code (InsightFace,
deep-person-reid). Only its wrapper imports it:
`providers/face_recognition/buffalo_l.py` or `providers/reid/osnet.py`. Keep
the licence headers and the change lists up to date. The providers and the
vendored code import no ROS package.

Outside the package:

- `ros2_ws/src/walkie_vision_msgs/`: the messages and the services only.
- `config/`: the node's parameters (`vision.yaml`, `sim.yaml`) and each task's
  classes and flags (`tasks/hri.yaml`, `tasks/memory.yaml`).
  `launch/vision.launch.py` starts the node and restarts it when it exits.
- `checks/`: the environment checks, one process each. `c12_memory.py` is the
  memory task's explicit mode of check 12; `check-env` does not start it.
- `tools/`: standalone programs and scored runs. The embedding bake-off is in
  `tools/bakeoff/`, with `tests/test_bakeoff*.py` and
  `docs/guides/bakeoff.md`.
- `scripts/`: shell helpers for the environment (`scripts/activate.sh`).
- `.github/workflows/`: `test.yml` runs the fake-model tests from a clean
  checkout, and `docs.yml` publishes the site.
- This repository has no `evals/` folder.

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
- A new top-level module in `src/walkie_vision/` gets its place in the
  `layers-point-down` contract in `pyproject.toml`; the tests fail until it
  has one. A new provider goes under `providers/<vision task>/`, with its
  entry in that task's `registry.py`, named like its module.
- No shims: a moved module leaves nothing at its old path, and
  `tests/test_old_homes.py` checks that nothing still names it there.
- A significant decision gets an ADR in `docs/architecture/decisions/`,
  embedded in `docs/architecture/vision.md`. Do not rewrite an accepted
  ADR: supersede it (`docs/architecture/index.md`).
- Links inside an ADR start with `/` (resolved from `docs/`).
- Move `torch` and `torchvision` together, keep CUDA 12 builds, and run
  `check-env` before changing the PyTorch index or `onnxruntime-gpu`.
- Ultralytics loads weights from the working directory. Run YOLOE code from
  `models/ultralytics/`.
- Docs, code comments and commit messages use plain, short sentences.
