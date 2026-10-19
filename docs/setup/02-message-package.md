# 2. Add the Message Package

**Goal:** define the ROS messages that the vision models publish, build them
in this repository with colcon, and make them easy for walkie-database and
walkie-agent-ros to build too.

## Background

A ROS message type is defined in a `.msg` file and compiled into Python and
C++ code by `rosidl`. The package that holds the files must be built with
colcon before any node can import the types. Every node that publishes or
subscribes to a type must be built from the **same definition**: under
rmw_zenoh, a publisher and a subscriber whose definitions differ never
connect, and nothing reports why (walkie-agent-ros ADR-0017 and ADR-0018).

The messages live in this repository, in a package that holds only messages
([ADR-0003](../architecture/decisions/0003-messages-in-this-repository.md)).
walkie-vision publishes them, so a change to a message and the code that
fills it land in one commit.

## Step 1: Design the messages

The package `walkie_vision_msgs` has four messages and one service:

| Type                   | What it is                                                                                     |
| ---------------------- | ---------------------------------------------------------------------------------------------- |
| `ObjectDetectionArray` | Every object found in one camera image: the image's `header` and `camera_info`, the `classes` the detector looked for, and the detections |
| `ObjectDetection`      | One object: `label`, `score`, `track_id`, `bbox`, an image embedding with its model name, its 3D `points`, and an optional compressed `crop` |
| `PersonDetectionArray` | Every person found in one camera image, with the image's `header` and `camera_info`             |
| `PersonDetection`      | One person: body `bbox` and 17 keypoints, the face box and face embedding, and an appearance embedding |
| `EmbedText` (service)  | Embeds texts with the object embedder, so that a text can be searched against stored object vectors |

Every field is commented in its `.msg` file. The rules behind them:

- **One array per image, also when it is empty.** `classes` lists the text
  prompts the detector used. A class without a detection tells
  walkie-database that no such object was visible, which is how it learns
  that an object has gone.
- **Boxes are `vision_msgs/BoundingBox2D`.** Its centre and size are float
  pixels, so sub-pixel boxes survive, and walkie-ros's `yolo_node` uses the
  same type. `sensor_msgs/RegionOfInterest` was not used: it holds integer
  pixels, and its `do_rectify` flag refers to a camera calibration, not to a
  detector's output. `center.theta` is always 0.
- **Every embedding names its model.** `embedding_model`, `face_model` and
  `appearance_model` are the `<model>` part of walkie-database's vector
  table names, for example `clip_vit_b16` for the `object_clip_vit_b16`
  table. walkie-database refuses a vector from a model it has no table for,
  or with the wrong length (walkie-database ADR-0006), so no separate length
  field is needed: the length is the array's length. A name without a vector,
  or a vector without a name, is an error.
- **Vectors have unit length.** The code that fills a message normalises
  them, so no consumer has to.
- **`track_id` 0 means "not tracked".** Ultralytics' tracker starts its ids
  at 1.
- **Keypoints are a fixed `float32[34]`**, x and y of the 17 COCO keypoints
  in COCO order, with `float32[17]` scores. A fixed length cannot be filled
  wrongly.
- **A person without a face** has `face_det_score` 0 and empty `face_model`
  and `face_embedding`.

## Step 2: Create the package

The package is in `ros2_ws/src/walkie_vision_msgs/`, the same layout as
walkie-agent-ros's `ros2_ws`:

```text
ros2_ws/
└── src/
    └── walkie_vision_msgs/
        ├── CMakeLists.txt
        ├── package.xml
        ├── msg/
        │   ├── ObjectDetection.msg
        │   ├── ObjectDetectionArray.msg
        │   ├── PersonDetection.msg
        │   └── PersonDetectionArray.msg
        └── srv/
            └── EmbedText.srv
```

`package.xml` declares the packages the messages use, and joins the
`rosidl_interface_packages` group, which every message package must:

```xml title="ros2_ws/src/walkie_vision_msgs/package.xml (part)"
<buildtool_depend>ament_cmake</buildtool_depend>
<buildtool_depend>rosidl_default_generators</buildtool_depend>

<depend>std_msgs</depend>
<depend>sensor_msgs</depend>
<depend>vision_msgs</depend>

<exec_depend>rosidl_default_runtime</exec_depend>

<member_of_group>rosidl_interface_packages</member_of_group>
```

`CMakeLists.txt` lists every file for `rosidl_generate_interfaces`. A new
`.msg` file must be added there too:

```cmake title="ros2_ws/src/walkie_vision_msgs/CMakeLists.txt (part)"
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ObjectDetection.msg"
  "msg/ObjectDetectionArray.msg"
  "msg/PersonDetection.msg"
  "msg/PersonDetectionArray.msg"
  "srv/EmbedText.srv"
  DEPENDENCIES std_msgs sensor_msgs vision_msgs
)
```

The `<license>` is `TODO`, because the repository has no licence yet
([Licences](01-vision-environment.md#licences)).

## Step 3: Build it with colcon

The build tools come from conda, so no system ROS is needed:

```toml title="pixi.toml"
[dependencies]
colcon-common-extensions = "*"
ros-jazzy-ament-cmake = "*"
ros-jazzy-rosidl-default-generators = "*"
c-compiler = "*"
cxx-compiler = "*"
cmake = "*"
pkg-config = "*"
make = "*"

[tasks.build-msgs]
cmd = "colcon build --symlink-install --cmake-args -DPython3_EXECUTABLE=$CONDA_PREFIX/bin/python"
cwd = "ros2_ws"
```

```bash
pixi run build-msgs
```

| Part                     | Why                                                                                         |
| ------------------------ | ------------------------------------------------------------------------------------------- |
| The compilers and CMake  | Without them in the environment, CMake falls back to whatever is on the system, such as an apt ROS install. |
| `cwd = "ros2_ws"`        | colcon writes `build/`, `install/` and `log/` next to `src/`, so they stay inside `ros2_ws/`. Git ignores all three. |
| `-DPython3_EXECUTABLE`   | A user-level `~/.colcon/defaults.yaml`, often set up for an apt ROS install, could otherwise point CMake at the system Python. |
| `--symlink-install`      | The same command as walkie-agent-ros's `build-ros2`.                                        |

## Step 4: Use the build in every `pixi run`

```toml title="pixi.toml"
[activation]
scripts = ["scripts/activate.sh"]
```

```bash title="scripts/activate.sh"
if [[ -f "${PIXI_PROJECT_ROOT}/ros2_ws/install/local_setup.sh" ]]; then
    source "${PIXI_PROJECT_ROOT}/ros2_ws/install/local_setup.sh"
fi
```

Pixi runs the script when it activates the environment, so every
`pixi run` and `pixi shell` after a build can import `walkie_vision_msgs`.
The shell that ran the build cannot: open a new `pixi shell` after the first
build.

Pixi 0.80 does not show the script's output, so a missing build would
otherwise only show up as an `ImportError`. Instead, `tests/conftest.py`
stops the tests at once with:

```text
walkie_vision_msgs is not built: run `pixi run build-msgs` once, then run the tests again
```

`test` does not depend on `build-msgs`. Pixi computes the environment
before it runs a dependency, so the first test run after a build would still
miss the overlay.

## Step 5: Fill the messages from the models

`src/walkie_vision/convert.py` turns the models' results into messages, and
checks them on the way:

| Function                                      | What it does                                                         |
| --------------------------------------------- | -------------------------------------------------------------------- |
| `to_object_detection`, `to_object_detection_array` | One `ObjectDetection` per detector result. Refuses a label that is not in `classes`. |
| `to_person_detection`, `to_person_detection_array` | One `PersonDetection` per pose result, with an optional face and appearance embedding. |
| `fill_embed_text_response`                    | Answers an `EmbedText` request. An embedder error gives `success = False` and the error in `message`, so the service never crashes. |
| `vectors_from_response`                       | The client side: the response's vectors as an `(n, dim)` array.      |
| `box_to_bbox`, `bbox_to_box`, `unit_vector`   | The conversions they share. `box_to_bbox` refuses a box that is not finite or is inverted, and `unit_vector` a vector that is empty, not finite or zero. |

The models themselves are behind small interfaces in
`src/walkie_vision/providers.py`, and `src/walkie_vision/fakes.py` has a
fake for each one. The tests fill every message from the fakes, serialise
it and read it back, so they run without a GPU or model files.

!!! info "Arrays in rosidl's Python code"
    An unbounded `float32[]` field, such as `embedding`, takes an
    `array.array("f")` or a list of floats, not a numpy array, and returns an
    `array.array("f")`. A fixed `float32[34]` field, such as `keypoints`,
    takes and returns a numpy `float32` array. `convert.py` does the
    conversion.

`tests/test_convert.py` also pins every field name and type. If a message
changes, that test fails until it is updated on purpose: see
[Changing a message](#changing-a-message).

## How other repositories build the package

walkie-database and walkie-agent-ros must build `walkie_vision_msgs` in their
own workspace, from the same commit as the walkie-vision they talk to.
Their environment needs `ros-jazzy-vision-msgs` and the same build tools as
in Step 3.

Either clone this repository into their `ros2_ws/src/`, or list it in their
`.repos` file, pinned to a commit:

```yaml title="ros2_ws.repos (in the other repository)"
repositories:
  ros2_ws/src/walkie-vision:
    type: git
    url: https://github.com/EIC-Robocup-2026/walkie-vision.git
    version: <commit>
```

colcon then finds only `walkie_vision_msgs` in the clone. It skips hidden
folders, such as a `.pixi/` environment inside the clone, and it does not
take this repository's `pyproject.toml` for a package. To build only the
messages:

```bash
colcon build --symlink-install --packages-select walkie_vision_msgs
```

## Changing a message

1. Edit the `.msg` or `.srv` file, and its comments.
2. Update `convert.py`, and the field pins in `tests/test_convert.py`.
3. Run `pixi run build-msgs`, then `pixi run test`.
4. Rebuild the package in every repository that uses it, from the new
   commit. A node still built from the old definition gets no messages, and
   prints no error.
