# ADR-0017: Lay out the vision package by concept, with its import rules checked

<!-- --8<-- [start:body] -->
**Status:** Accepted · **Date:** 2026-11-19

**Context**

The package `walkie_vision` had grown flat: fourteen top-level modules beside
two packages, `face/` and `reid/`. The file names said which vendor a module
wrapped, not what job it did.

- `models.py` held no models but the table of model files. `providers.py`
  held no providers but their interfaces. `yolo.py` held the object
  detector, the pose model and the helpers both share.
- `node.py`, 1,609 lines, held the node, its parameters and their checks,
  the model loading, the GetPersonCrop work and the QoS profiles.
- Adding or switching a model meant editing `node.py` in several places
  (the parameters, their checks, two file tables that copied other modules,
  and `build_providers`), and also `config/vision.yaml` and `models.py`.
- Model code reached ROS by accident: `yolo.py` imported `resize_mask` from
  `image.py`, which imports `sensor_msgs`.
- No import rule was enforced. Two subprocess tests each checked that one
  module loads no model library.
- Teammates will add providers, such as a second embedder. They need to
  know where a provider goes and what it may import.

**Decision**

The package is laid out by concept. Each step was a commit of its own on
the branch `refactor/package-layout`, cut at ad7c833. Six guard commits came
first: they pin the message digests, the node's declared parameters and
endpoints, the light imports, the import contracts, the module names that
the scripts and docs use, and what other repositories read from this one.
Sixteen pure moves followed, then the contracts, then the registries and the
parameter names. After each step the full test suite passed, and a snapshot
of the node's behaviour stayed the same, except for the `fetch-models` task's
module and the parameter rename.

The concepts, in the layers the contracts check, top first:

| Path in `src/walkie_vision/` | What it holds |
| --- | --- |
| `node.py`, `debug_viz.py` | The vision node, and the optional debug viewer ([ADR-0016](/architecture/decisions/0016-debug-visualizer-subscribes-to-the-image.md)) |
| `loading.py`, `fake_publisher.py` | `build_providers`, which builds the models from the registries; the fake object publisher |
| `pipeline.py`, `services/`, `fakes.py` | One image through every provider; the bounded service work; the fake providers |
| `messages.py`, `camera/`, `crops/`, `geometry.py`, `config.py` | Messages and QoS; the camera input; the person and object crops and colour shares; depth geometry; the node's parameters |
| `providers/` | One folder per vision task: its interface (`base.py`), its registry (`registry.py`) and its providers. Also `spec.py`, `vector_tables.py` and `_ultralytics.py` |
| `boxes.py`, `weights.py`, `errors.py` | Box and mask maths; every model file with its checksum; `StartupError` |
| `_vendor/` | Vendored code, outside the layers: `insightface/` and `torchreid/` |

Where each old module went:

| Old module | New home |
| --- | --- |
| `node.py` | `node.py`. `StartupError` went to `errors.py`, `VisionConfig` to `config.py`, `build_providers` to `loading.py`, the QoS profiles to `messages.py`, `NewestFrame` to `camera/mailboxes.py`, and the GetPersonCrop job, mailbox and request check to `services/get_person_crop.py`. |
| `models.py` | `weights.py` |
| `convert.py` | `messages.py`. The service response builders went to `services/`. |
| `requests.py` | `services/`: `jobs.py`, `embed_text.py`, `detect_objects.py`, `embed_crop.py`. `CurrentImage` went to `camera/mailboxes.py`. |
| `image.py` | `camera/image.py`. `resize_mask` went to `boxes.py`. |
| `depth.py` | `camera/depth.py`. The depth maths went to `geometry.py`, `RgbdBuffer` to `camera/mailboxes.py`, `stamp_ns` to `messages.py`. |
| `crop.py` | `crops/person.py`. The boxes went to `boxes.py`, the colour names and shares to `crops/colour.py`, the JPEG encoding to `crops/jpeg.py`. |
| `embeddings.py` | The sampler and the padded crops went to `crops/objects.py`, the Lab palette shares to `crops/colour.py`, and CLIP to `providers/image_embed/clip.py`. |
| `providers.py` | The `providers/` package: `vector_tables.py` and one `base.py` per vision task. `providers/__init__.py` re-exports the same 14 names. |
| `yolo.py` | `providers/object_detection/yoloe.py`, `providers/pose_estimation/yolo_pose.py`, and the shared helpers in `providers/_ultralytics.py` |
| `face/analysis.py`, `face/heads.py` | `providers/face_recognition/buffalo_l.py` |
| `face/scrfd.py`, `face/arcface.py`, `face/face_align.py` | `_vendor/insightface/` |
| `reid/__init__.py`, `reid/identifier.py` | `providers/reid/osnet.py` |
| `reid/osnet.py` | `_vendor/torchreid/osnet.py` |
| `pipeline.py`, `fakes.py`, `fake_publisher.py`, `debug_viz.py` | Kept. Their imports changed. |

Names whose home is not obvious. Each keeps every import pointing down:

- `resize_mask` is in `boxes.py`, not `camera/image.py`: `camera/image.py`
  imports `sensor_msgs`, and YOLOE needs the function below the providers.
- `stamp_ns` is in `messages.py`. `DepthRefused` is in `camera/depth.py`.
- `EMBEDDING_DIMS` is in `providers/vector_tables.py`, and
  `check_model_dim` in `providers/image_embed/base.py`. In
  `vector_tables.py` its `Embedder` annotation would make a cycle.
- `DETECTOR_FILE` and `RECOGNIZER_FILE` are in `weights.py`, which already
  keys the buffalo_l download and its checksum by these names. A registry
  cannot hold a name its provider imports: the pair would be a cycle.
- `clip_checkpoint` is in `providers/image_embed/clip.py`, and
  `MOBILECLIP_FILE` in `providers/object_detection/yoloe.py`.
- `image_age_problem` and `ClassPrepareGate` are in
  `services/detect_objects.py`. `vectors_from_response` is in
  `services/embed_text.py`. `bounded_message` and `MAX_MESSAGE_CHARS` are in
  `services/jobs.py`. `CropJob` and `CropMailbox` are in
  `services/get_person_crop.py`.
- `face/heads.py` merged into `buffalo_l.py`, its own `box_iou` included:
  the three `box_iou` copies clamp differently, so they stay apart.
- Six private names became public when they crossed a module:
  `messages.float32_array`, `boxes.box_area`, and `_ultralytics`'s
  `clip_box`, `check_rows`, `require` and `reset_trackers`.
- Two copied tables are gone: `TRACKER_FILES` (now `TRACKER_NAMES` in
  `_ultralytics.py`) and the face file table (now the two names in
  `weights.py`). `CLIP_DIM` stays: `check_model_dim` compares it with
  walkie-database's width in `EMBEDDING_DIMS`.

The registries. `providers/spec.py` defines `ProviderSpec`: a provider's
options dataclass, the files it needs, an optional `prepare` (the Hugging
Face cache check or download) and `build`. Only YOLOE sets the optional
`build_detector`, for the second detector behind `DetectObjects`. Each
vision task's `registry.py` holds `PROVIDERS`, from provider name to spec,
and `DEFAULT_PROVIDER`. A registry imports its provider modules only inside
its `prepare`, `build` and `build_detector` functions. The one exception is
object detection's registry: it imports `yoloe.py` at module level for the
constant `MOBILECLIP_FILE`, and `yoloe.py` loads Ultralytics only in its
constructors. So importing a registry loads no model library. A provider
module never imports its registry. `build_providers` checks every file
first and names all the missing ones, then prepares, then builds, in the
same order as before.

The parameters. The 14 model parameters are named
`<vision task>.<provider>.<option>`: `objects_weights` became
`object_detection.yoloe.weights`. Each vision task gets
`<vision task>.provider`. Thresholds that belong to no provider, such as
`objects_min_score`, stay flat. `tests/test_parameter_snapshot.py` holds the
rename table and checks that each old value lands at its new name.

The rules, checked by import-linter 2.16 inside pytest from
`tests/test_import_contracts.py`. The contracts are in `pyproject.toml`:

- `model-code-knows-no-ros`: `_vendor`, `boxes`, `crops`, `errors`,
  `fakes`, `providers` and `weights` import no ROS package.
- `providers-know-no-ros-or-the-node`: `providers` and `_vendor` import no
  ROS package and none of the modules above the providers.
- `insightface-wrapped-once` and `torchreid-wrapped-once`: only
  `providers.face_recognition.buffalo_l` imports `_vendor.insightface`, and
  only `providers.reid.osnet` imports `_vendor.torchreid`.
- `vendored-code-stands-alone`: `_vendor` imports nothing else from
  `walkie_vision`.
- `layers-point-down`: the layers in the first table. An import points only
  to a lower layer. The modules of a layer import none of each other, except
  in the middle layer, whose modules share a few names. `exhaustive = true`
  makes every top-level module part of a layer.
- `providers-modules-skip-the-package-init`: a module inside `providers/`
  imports the module that defines a name, never the package.
- `no-sibling-cycles`: no two modules of one package import each other,
  directly or through others.

Three more rules are tests:

- Light imports. `tests/test_import_light.py` imports every module in its
  own fresh interpreter. Only the three modules in `HEAVY_AT_IMPORT` may
  load a model library: `providers.face_recognition.buffalo_l`,
  `providers.reid.osnet` and `_vendor.torchreid.osnet`. grimp counts imports
  inside functions, so this rule cannot be a contract.
- `__init__.py` files. Each holds one docstring and nothing else, except
  `providers/__init__.py`, which re-exports the names of its base modules
  and `vector_tables` (decision RL-8). `tests/test_import_contracts.py`
  checks it.
- No shims. Nothing stays at an old path: no module, no re-export and no
  alias (decision RL-5). `tests/test_old_homes.py` fails when code, a build
  file or a live docs page names an old home.

Test files are named after the module they test:
`tests/test_crops_person.py` tests `crops/person.py`.

**Consequences**

- Importing an old path fails at once with `ModuleNotFoundError` or
  `ImportError`.
- walkie-database and walkie-agent-ros pin this repository at 64811fe,
  which keeps the old paths. Nothing changes for them until someone bumps
  that pin. Whoever bumps it updates what imports the old paths there:
  walkie-database's `tools/contract_peer.py` imports `walkie_vision.convert`
  and `walkie_vision.requests`, and walkie-agent-ros's
  `tests/test_oracle_perception.py` names `walkie_vision.convert` in its
  docstrings.
- A script outside the repositories that imports `walkie_vision.image`
  breaks once the layout reaches `main`. It needs
  `from walkie_vision.camera.image import image_to_bgr`.
- rclpy ignores an override for a parameter the node does not declare. A
  flat model parameter left in a parameter file outside this repository
  now falls back to its default, with no error.
  `tests/test_parameter_snapshot.py` checks only this repository's files.
- `pixi.lock` now holds import-linter 2.16 and grimp 3.20. In a checkout
  whose environment was installed before, the first `pixi run` without
  `--as-is` installs them. Pick that moment: not while a simulation runs
  from that checkout.
- Adding a top-level module means naming it in the `layers-point-down`
  contract, or the test fails.
- Adding a provider means a module under `providers/<vision task>/`, named
  after its key, an options dataclass and a `ProviderSpec` in that task's
  `registry.py`, and its files in `weights.py`. `config.py` types each
  task's options field with today's options dataclass, so a second provider
  widens that annotation.
- Real model loading is not covered by the tests: they use fakes and stub
  models. It needs a run on the GPU.

**Alternatives**

- **Keep the flat layout.** No moves and no broken paths, but a reader
  still cannot tell where a model is chosen, and the rules stay unchecked.
- **Shims for a release.** The old paths would keep working for a while,
  at the cost of a second name for every module, which is the confusion
  the layout removes.
- **A registry in each move step.** It would mix a rewrite into the moves,
  which the snapshot could then no longer check as pure moves.
- **A second object detection spec for `DetectObjects`.** A key such as
  `yoloe_on_demand` could be chosen as the tracker by mistake.
- **Packages that could replace our code**, rejected:
  - `insightface`: it requires the CPU `onnxruntime`, which installs the
    same module and can shadow `onnxruntime-gpu`
    ([ADR-0006](/architecture/decisions/0006-face-models-without-insightface.md)).
  - `message_filters`: its `ApproximateTimeSynchronizer` does not match the
    pairing rules of
    [ADR-0014](/architecture/decisions/0014-bounded-rgbd-and-service-work.md).
  - `supervision`: heavy for a few helpers, and sharing one `box_iou`
    would change behaviour.
  - `sensor_msgs_py.point_cloud2`: walkie-agent-ros copies our cloud byte
    layout, and the saving is small.
  - `pydantic` and `cv_bridge`: the checks are interleaved with
    `declare_parameter` and their errors are tested, and `cv_bridge` adds a
    compiled ROS dependency for a few lines of numpy.
  - Entry points or `pluggy`: every provider lives in this repository.
    Entry points need a reinstall and hide the list of providers from a
    reader.

**See also:** [Package layout](/architecture/vision.md#package-layout),
[ADR-0010](/architecture/decisions/0010-one-vision-process.md),
[ADR-0012](/architecture/decisions/0012-license-agpl-3-0-only.md) for the
vendored code, which now lives in `src/walkie_vision/_vendor/`, and the
decisions RL-5 (no shims), RL-8 (one vocabulary, and docstring-only
`__init__.py` files) and RL-12 to RL-16 (the choices this layout made) in
walkie-agent-ros's Mahoraga demo decisions.
<!-- --8<-- [end:body] -->
