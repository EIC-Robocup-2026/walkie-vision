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
pixi run docs-build     # mkdocs build --strict; must pass with no warnings
pixi install --locked   # must succeed: pixi.lock is committed
```

Run `build-msgs` once before `test`. A new `pixi run` is needed after a
build, because `scripts/activate.sh` sources the overlay on activation.

## Layout

- `ros2_ws/src/walkie_vision_msgs/`: the messages and the service only.
- `src/walkie_vision/providers.py`: model interfaces and result types.
- `src/walkie_vision/convert.py`: results to messages, with validation.
- `src/walkie_vision/fakes.py`: deterministic fakes for tests.
- `src/walkie_vision/face/`, `src/walkie_vision/reid/`: vendored model code.
  Keep the licence headers and the change lists up to date.
- `src/walkie_vision/models.py`: every downloaded file and Hugging Face
  commit, with checksums and licences.
- `checks/`: the `check-env` checks, one process each.
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
