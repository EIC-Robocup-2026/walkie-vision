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

It looks for household objects, and turns on depth, sampled crops and CLIP
vectors ([Sampled crops and vectors](../architecture/vision.md#contract-object-samples)).
`sim:=true` chooses the clock, the topics and the services. It brings no fakes
or oracle data.

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
  `object_embeddings_enabled`. They count the service jobs that ran, the
  requests refused with `BUSY`, and the answers that came after their deadline
  and were discarded.
- **The stages** show where the time goes. [The latency line](../architecture/vision.md#the-latency-line)
  explains each one.

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

Its arrays are BEST_EFFORT, VOLATILE, KEEP_LAST 1, so a RELIABLE subscriber
does not match them, and a slow subscriber can miss frames. A consumer test
must wait for the frames it needs, not count them.

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
