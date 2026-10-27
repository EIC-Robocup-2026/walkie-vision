# Run the vision node

The vision node runs every model on the head camera's newest image and
publishes one `PersonDetectionArray` and one `ObjectDetectionArray` for
it. This page shows how to start it, what its log lines mean, and what to
check when it does not publish. Why it is built this way is in
[Vision](../architecture/vision.md#the-vision-node), and what it promises
its consumers is in the [Contract](../architecture/vision.md#contract).

## Before the first run

Do these once per computer, in the project folder:

1. Download the model files into `models/`:

    ```bash
    pixi run fetch-models
    ```

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
   seats, and nothing else.
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

When images are skipped, the node warns every 5 s with the reason, for
example "camera_info is 640x360 but the image is 2208x1242". The camera
topics are the parameters `image_topic` and `camera_info_topic` in
`config/vision.yaml`.

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
