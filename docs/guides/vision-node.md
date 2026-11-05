# Run the vision node

The vision node runs every model on the head camera's newest image and
publishes one `PersonDetectionArray` and one `ObjectDetectionArray` for
it. This page shows how to start it, what its log lines mean, and what to
check when it does not publish. Why it is built this way is in
[Vision](../architecture/vision.md#the-vision-node), and what it promises
its consumers is in the [Contract](../architecture/vision.md#contract).

## Before the first run

Do these once per computer, in the project folder:

1. Download the model files into `models/`, and the CLIP weights into the
   Hugging Face cache:

    ```bash
    pixi run fetch-models
    ```

    The node loads CLIP from the cache only, and only with
    `object_embeddings_enabled`. It never downloads them. Without them it
    stops at start and says what to run.

2. Build the messages:

    ```bash
    pixi run build-msgs
    ```

3. Put OSNet's checkpoint into the Hugging Face cache. This needs the
   internet once:

    ```bash
    pixi run python -c "from walkie_vision.reid import load_osnet; load_osnet('cpu')"
    ```

    The node checks the cache when it starts. Offline and without the
    checkpoint, it stops within about a second and says so.

## Run it

The node needs a Zenoh router, like every `pixi run`
([Getting Started](../getting-started.md#7-talk-to-other-nodes)).

=== "On the robot"

    The robot's router is already running. Use the robot's domain:

    ```bash
    export ROS_DOMAIN_ID=23
    pixi run vision
    ```

    The node reads the ZED's image and `camera_info`, publishes on
    `/walkie_vision/people` and `/walkie_vision/objects`, and serves
    `/walkie_vision/get_person_crop`.

=== "In simulation"

    Use the domain the simulation was started with, and add `sim:=true`:

    ```bash
    pixi run vision sim:=true
    ```

    The node then uses sim time, publishes on `/walkie_vision/sim/people`
    and `/walkie_vision/sim/objects`, and serves
    `/walkie_vision/sim/get_person_crop`. In simulation,
    walkie-agent-ros's `oracle_perception` owns the normal topics, so the
    two never mix. The crop service moves too, because the agent's track
    ids there come from `oracle_perception`. The agent does not read the
    sim topics: they are for looking at the node's output on rendered
    frames. MuJoCo has no person bodies, so expect no people.

!!! warning "One vision node per ROS graph"
    Do not start the node on a computer that is joined to someone else's
    router, such as the robot's, while their vision node runs. Consumers
    would get two sets of arrays for every image. The node warns when
    another node publishes on its topics.

`pixi run vision` runs `ros2 launch launch/vision.launch.py`. The launch
starts the node again whenever it exits. Press ++ctrl+c++ to stop it; that
does not restart it.

### The memory task

For object memory, use the `memory` task:

```bash
pixi run vision task:=memory
pixi run vision task:=memory sim:=true
```

It looks for household objects, and turns on depth, sampled crops, CLIP
vectors ([Sampled crops and vectors](../architecture/vision.md#contract-object-samples))
and `DetectObjects` ([On-demand DetectObjects](../architecture/vision.md#on-demand-detectobjects)).
`sim:=true` chooses the clock, the topics and the services. It brings no fakes
or oracle data.

It loads a second YOLOE and CLIP on top of the HRI task's models, and a new
`DetectObjects` list can stop the stream for about 1.5 s. Neither its rate nor
its GPU memory is measured yet: see
[Unmeasured budgets](../architecture/vision.md#unmeasured-budgets).

### The embedding services

With `object_embeddings_enabled`, the node serves two services on its one model
worker ([Bounded services](../architecture/vision.md#bounded-services)):

| Service         | Robot                       | `sim:=true`                     | What it does                          |
| --------------- | --------------------------- | ------------------------------- | ------------------------------------- |
| `EmbedText`     | `/walkie_vision/embed_text` | `/walkie_vision/sim/embed_text` | Embeds up to 16 texts. Works with no camera |
| `EmbedCrop`     | `/walkie_vision/embed_crop` | `/walkie_vision/sim/embed_crop` | Crops and embeds one box of the current image |

`/walkie_vision/describe_crop`, or `/walkie_vision/sim/describe_crop`, always
answers `DISABLED`. Without the flag, the other two answer `DISABLED` as well, so
HRI loads no CLIP.

Treat `BUSY` and `STALE_IMAGE` as normal answers. `BUSY` means another job holds
the worker, and nothing queues. `STALE_IMAGE` means that the box's image is no
longer the current one. The node has no older image, so ask with a newer image's
stamp, or carry on without the vector.

### The DetectObjects service

With `on_demand_enabled`, which the memory task turns on, the node serves
`DetectObjects` at `/walkie_vision/detect_objects`, or
`/walkie_vision/sim/detect_objects` with `sim:=true`. A second YOLOE, with no
tracker, looks for the request's classes in the newest image. Only the caller
gets the answer: nothing is published on the objects topic.

- Ask for 1 to 32 unique classes of at most 80 characters each. `max_image_age_s`
  is over 0 and at most 1 s, and `timeout_s` over 0 and at most 5 s.
- The answer holds at most 20 detections, those with the highest scores, each
  with `track_id` 0. It has points with `geometry_enabled`, and every box of at
  least 32 px on its short side gets the crop or the vector that its flags allow.
- Its depth must pair with the image the way the stream's does: stamps within
  `max_pair_gap_s`, and the two received within `pair_wait_s` of wall time of
  each other. A depth received later than that counts as none: no points, or
  `NO_IMAGE` under `require_depth`. A jump back of `/clock`, as at a sim
  restart, drops every image the services hold, so no image or depth from the
  old run answers. Until the next image arrives, `DetectObjects` and
  `EmbedCrop` answer `NO_IMAGE`.
- A new list of classes is prepared at most once every 2 s, and the stream stops
  while it is prepared: about 1.5 s for 5 classes on the RTX 2060. The same list
  again prepares nothing, so keep the list stable where you can.
- Without the flag it answers `DISABLED`, and the second model does not load.

[On-demand DetectObjects](../architecture/vision.md#on-demand-detectobjects)
has the details, and [Bounded services](../architecture/vision.md#bounded-services)
the error codes.

### The first start on a new GPU

On a GPU that ONNX Runtime has no machine code for, such as an RTX 50, the
first start compiles the face models' kernels. It can take minutes. The log
says "warming up the face models; the first run on this GPU can take
minutes" before it starts. The driver keeps the compiled kernels in
`~/.nv/ComputeCache`, so later starts are fast. Keep that folder between
runs. In a container, mount it as a volume. A driver update empties it.

## Launch arguments

| Argument              | Default | What it does                                                         |
| --------------------- | ------- | -------------------------------------------------------------------- |
| `task`                | `hri`   | The task whose classes the detector looks for: a file in `config/tasks/` |
| `sim`                 | `false` | Use sim time, and publish and serve on the sim names (`config/sim.yaml`) |
| `respawn_delay`       | `3.0`   | Seconds before the node starts again after it exits                  |
| `respawn_max_retries` | `-1`    | How many times it starts again. -1 means no limit                    |

Launch arguments use `:=`, for example
`pixi run vision task:=hri respawn_max_retries:=5`.

The node's own parameters, such as the topics, the model files and the
score floors, are in `config/vision.yaml`, with a comment on each. They
are read-only: change the file and start the node again. The full table is
in [Parameters](../architecture/vision.md#parameters).

## Add a task

The detector's text prompts are fixed for a whole run, because changing
them costs about 1.5 s and can reset the track ids. Each task has its own
list:

1. Copy `config/tasks/hri.yaml` to `config/tasks/<task>.yaml`.
2. Change `classes`. Use the exact words the consumer looks for. For HRI,
   walkie-agent-ros counts `chair`, `sofa`, `armchair` and `stool` as
   seats, and nothing else. A task can also turn on capability flags, as
   `config/tasks/memory.yaml` does.
3. Run `pixi run vision task:=<task>`.
4. Run `pixi run test`. It checks that every task file holds `classes` and
   only parameters the node knows.

## Read the log

When the models have loaded, the node logs one line with what it runs:

```text
classes ['chair', 'sofa', 'armchair', 'stool']; image /zed_head/zed_node/rgb/color/rect/image, camera_info /zed_head/zed_node/depth/camera_info; publishing /walkie_vision/people and /walkie_vision/objects; serving /walkie_vision/get_person_crop; trackers: objects bytetrack.yaml, people botsort.yaml; track id offset 1234567890
```

The track id offset comes from the clock at start. It differs on every
start, so a restarted node never reuses an id.

Every `GetPersonCrop` request logs one line when it is answered, with the
status, the time from the request to the answer, and for OK the JPEG's
size:

```text
person crop for track 1234567893: OK in 84 ms, 41 KB
```

Any status but OK ends with the reason, for example
`NOT_IN_VIEW in 70 ms; track 1234567893 is not in the image, which has tracks [1234567891]`.
[Person crops](../architecture/vision.md#person-crops) explains each
status.

Every 10 s it logs the latency line:

```text
processed 52 in 10.0 s (5.2 Hz); received 150, replaced 98, skipped 0, failed 0, dropped 0, duplicates 0 | ms p50/p95: wait 3/9 convert 6/8 objects 21/30 pose 18/25 faces 12/40 reid 9/15 build 1/2 publish 1/2 | age 95/160
```

- **The rate.** The target is at least 5 Hz. The node caps itself at
  10 Hz, so it leaves GPU time for the ZED's depth.
- **`age`** is the time from the image's stamp to the publish, in ms. The
  target is under 300 at p95. walkie-agent-ros drops arrays older than
  1 s.
- **`replaced`** counts images that a newer image replaced before the node
  got to them. It is normal: the camera sends 15 images a second and the
  node processes only the newest.
- **`skipped`**, **`failed`** and **`dropped`** should stay at 0. The
  [Troubleshooting](#troubleshooting) table says what each one means.
- **`duplicates`** counts second boxes of one person that the node left
  out. The pose model gives these now and then, so a few are normal.
- **`jobs`**, **`busy`** and **`late`** appear with
  `object_embeddings_enabled` or `on_demand_enabled`. They count the service
  jobs that ran, the requests refused with `BUSY`, and the answers that came
  after their deadline and were discarded.
- **`depth`**, **`paired`**, **`no_depth`**, **`unpaired`**, **`sampled`** and
  **`prepared`** appear with the addition they belong to.
- **The stages** show where the time goes. An addition that is off adds no
  stage. [The latency line](../architecture/vision.md#the-latency-line) explains
  each one.

A model that fails to load logs a warning with its name before the node exits:

```text
the on-demand model failed to load: RuntimeError: CUDA out of memory
```

## Troubleshooting

| Symptom                                    | Cause                                                                                                   |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| Nothing published                          | `camera_info` missing or the wrong size, an unsupported encoding, or no router or the wrong domain     |
| The agent ignores the arrays               | The age is over 1 s; see the latency line                                                               |
| A respawn loop, with FATAL each time       | Missing models, OSNet not cached and offline, or a bad parameter (exit 2, one line); a model that fails to load, such as GPU out of memory (exit 2, with a traceback) |
| Restarts after running a while             | 30 failed images in a row, a hung GPU call, or the worker stopped (exit 1)                              |
| The first start takes minutes              | ONNX Runtime compiling kernels; keep `~/.nv/ComputeCache`                                               |
| "another node also publishes"              | `oracle_perception` or a second vision node on the same topics                                          |
| People get new ids while the robot turns   | `pose_tracker` is not `botsort.yaml`                                                                    |
| One person published twice                 | Two boxes with an IoU of 0.5 or less; lower `people_dedupe_iou`, which can merge real people who overlap |
| GPU out of memory                          | Other GPU users, such as the ZED SDK's depth                                                            |
| Person crops answer `NO_IMAGE`             | No image arrives on `image_topic`: the camera, the router or the domain                                 |
| Person crops answer `TIMEOUT`              | Images arrive but none is processed in `timeout_s`: skipped or failed images, see the latency line      |
| `no_depth` or skipped images with `geometry_enabled` | No depth within the stamp gap and the wait, or a pair that was refused. The warning names the reason ([Depth pairing](../architecture/vision.md#depth-pairing)) |
| `depth 0` in the latency line              | No depth arrives on `depth_topic`: the camera, the router or the domain                                 |
| Empty `points` with `paired` above 0       | Under 30 valid pixels in the mask: a small object, or depth outside `depth_min_m` to `depth_max_m` ([Object points](../architecture/vision.md#contract-object-geometry)) |
| "`open_clip_model.safetensors` ... is not in the Hugging Face cache" at start | `object_embeddings_enabled` is on and the weights were never fetched: run `pixi run fetch-models` once with the internet |
| Most objects have no crop or vector        | The normal rate: a track is sampled once in ten images, and at most three objects per image ([Sampled crops and vectors](../architecture/vision.md#contract-object-samples)) |
| `EmbedText` or `EmbedCrop` answers `DISABLED` | `object_embeddings_enabled` is off, as it is for HRI. `DescribeCrop` always answers this |
| `EmbedText` or `EmbedCrop` answers `BUSY`  | Another service job holds the worker, or its image is still running. Try again later; nothing queues ([Bounded services](../architecture/vision.md#bounded-services)) |
| `EmbedCrop` answers `STALE_IMAGE`          | The image the box belongs to is no longer the newest, or it is over 300 ms old. Often normal under load; `busy` and the image rate in the latency line show the load |
| `late` above 0 in the latency line         | A service job finished after its caller's deadline, and its answer was discarded |
| `DetectObjects` answers `NO_IMAGE` for depth, or has no points, while depth arrives | No pair within `max_image_age_s`: no depth within the stamp gap of an image, or none received within `pair_wait_s` of it, as on the stream ([Depth pairing](../architecture/vision.md#depth-pairing)) |
| `EmbedCrop` scores no depth term while depth arrives | The newest image and the newest depth did not pair, by the same bounds. Each service checks its own input: `EmbedCrop` takes only the newest of each, and `DetectObjects` may take a pair one image older |
| `DetectObjects` and `EmbedCrop` answer `NO_IMAGE` just after a sim restart, with a warning that the ROS clock jumped | Expected. A jump back of `/clock` drops the images the node holds, and the next camera image ends it ([Bounded services](../architecture/vision.md#bounded-services)) |
| `DetectObjects` answers `BUSY` for a new list | Another job holds the worker, or the last new list was prepared under 2 s ago. The same list does not wait for that ([Bounded services](../architecture/vision.md#bounded-services)) |
| The rate falls during `DetectObjects`      | Expected. A job runs between images, and a new list stops the stream for its whole preparation: see `service` and `prepare` in the latency line |

When images are skipped, the node warns every 5 s with the reason, for
example "camera_info is 640x360 but the image is 2208x1242". The camera
topics are the parameters `image_topic`, `camera_info_topic`, `depth_topic`
and `depth_camera_info_topic` in `config/vision.yaml`.

## Test it

Without a GPU, model files or a router:

```bash
pixi run test
```

The node's tests run it with fake models on Fast DDS. On a GPU, check the
real models, one check at a time:

```bash
pixi run python checks/run_checks.py c11 c12
```

`c11` checks each model wrapper on its own. `c12` loads the models the way
the node does and runs 20 HD2K images through all of them, with the
latency and the GPU memory. Their logs are in `checks/output/`.

The memory task has its own mode of `c12`. `check-env` never starts it, because it
needs a recording of your own: an `.npz` file with a registered colour and depth
pair, which `checks/c12_memory.py` describes.

```bash
pixi run python checks/c12_vision_node.py --memory pair.npz
```

It prints every stage and the first, same-list and new-list cases of
`DetectObjects`, with the budgets next to them for information. It decides no
budget: read the numbers against
[Unmeasured budgets](../architecture/vision.md#unmeasured-budgets). It has not yet
run with real models. A fixture that cannot be used exits with status 2, before
any model loads.

### The test workflow

`.github/workflows/test.yml` builds the messages and runs these tests from a clean
checkout, with the same restrictions as the local run: Fast DDS on localhost,
domain 77, no GPU, and no model downloads. It installs the whole default
environment, about 9.6 GB. It has not run on GitHub yet, so the runner's disk
and cache limits are not checked.

### The pair check with walkie-database {#pair-check}

The test workflow cannot check this repository against walkie-database, because a
candidate pair of commits may not be published yet. walkie-database owns the local
gate, `tools/pair_check.py`, which its own preparation work adds. Until that lands,
the command below fails because the file is missing. The gate builds both sets of
messages separately, compares every type hash, and runs the real wire exchange
between the two repositories with fake models, in domain 78 on Fast DDS. Run it
from walkie-database, on clean commits of both preparation worktrees, here at
`/var/tmp/wv-prep` and `/var/tmp/wd-prep`:

```bash
wv_commit=$(git -C /var/tmp/wv-prep rev-parse HEAD)
wd_commit=$(git -C /var/tmp/wd-prep rev-parse HEAD)
cd /var/tmp/wd-prep
env RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_DOMAIN_ID=78 ROS_LOCALHOST_ONLY=1 \
  CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python3 tools/pair_check.py --vision-root /var/tmp/wv-prep --database-root /var/tmp/wd-prep \
  --vision-commit "$wv_commit" --database-commit "$wd_commit" --domain 78 --build \
  --report /var/tmp/wa-impl-prep/db-vision/build-reports/pair.json
```

On a shared machine, put `taskset -c <cpus>` before `env`, so that the builds
leave other work alone. The command runs `python3` from the host, not from a Pixi
environment, because the tool starts each repository's own Pixi environment
itself. The result is the JSON report, which holds each check, the type hashes and
the import paths. This is the local stand-in for CI of an unpublished pair. It does
not show that either repository's remote CI passed, and it does not stand in for
the agent or walkie-ros rebuilding against these types.

## Run the fake publisher {#fake-publisher}

`walkie_vision.fake_publisher` publishes `ObjectDetectionArray` messages from
a scene file, with no camera, GPU or model. A consumer such as walkie-database
can test its handling of the contract against it.

```bash
pixi run fake-detections
```

The task sets Fast DDS, this computer only and domain 77 itself, so neither
activation nor your shell can change them. That keeps synthetic objects off a
robot's graph. For another domain, set the variables after `pixi run`:

```bash
pixi run env RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_DOMAIN_ID=78 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
  python -m walkie_vision.fake_publisher
```

!!! warning "This is synthetic geometry, not perception"
    A run shows that the layout, stamps, stream ids and QoS reach a consumer.
    It says nothing about detection quality, depth noise, masks or
    embeddings ([ADR-0013](../architecture/decisions/0013-memory-observation-contract.md)).
    Never run it in a domain where the vision node or a robot's TF is running.

| Parameter         | Default                  | What it does |
| ----------------- | ------------------------ | ------------ |
| `objects_topic`   | `/walkie_vision/objects` | The topic of the arrays |
| `classes`         | The scene's `classes`    | The array's `classes`. A box with another label gets no detection. `person` is refused. |
| `rate_hz`         | 5.0                      | Frames per second, on a steady timer. Stamps come from the node's clock. |
| `scene_file`      | `config/fakes/boxes.json` | The scene |
| `loop`            | `true`                   | Restart the sequence at its end. `false` exits with code 0 after the last frame. |
| `crops_enabled`   | `true`                   | Fill `crop` |
| `embedding_model` | Empty                    | Empty gives no vectors. A name must start with `fake_`, so nothing reads a fake vector as a real model's. |
| `embedding_dim`   | 512                      | The length of a fake vector |
| `publish_tf`      | `true`                   | Publish the camera's static transform on `/tf_static` |

A bad parameter or scene stops the start with exit code 2. A new process makes
a new `stream_id`, which is in its first log line. It publishes no people.

Its arrays have the vision node's output profile: RELIABLE, VOLATILE,
KEEP_LAST 5. RELIABLE and BEST_EFFORT subscribers both match it, such as
walkie-database's BEST_EFFORT, KEEP_LAST 1. A frame published before a
subscriber is discovered is lost, and a BEST_EFFORT or slow subscriber can
still miss frames. A consumer test must wait for the frames it needs, not count
them.

The scene file has a fixed pinhole `camera` and its pose in `parent_frame`,
the `classes`, axis-aligned `boxes` in that frame, and a `sequence` of phases:

```json title="config/fakes/boxes.json (part)"
"sequence": [
  {"frames": 10, "visible": ["cup_1", "bottle_1"]},
  {"frames": 5, "visible": ["cup_1"]},
  {"frames": 5, "visible": []},
  {"frames": 10, "visible": ["cup_1", "bottle_1"], "move": {"cup_1": [1.5, 0.35, 0.95]}}
]
```

- A phase repeats one frame `frames` times. A box it leaves out is not in the
  picture at all, so it hides nothing.
- `visible: []` gives a valid array with no detections.
- `move` sets a box's centre in `parent_frame`. It lasts until the sequence
  restarts.
- A box's `score` and `view_quality` are the scene's values, with no formula
  behind them. `color` paints the box and fills its `color_shares`.

Each box's detection comes from the pixels it wins in a ray-cast picture, so
nearer boxes hide farther ones. Its cloud is its surface at those pixels. With
the same stamp and `stream_id`, a frame is always the same message.
