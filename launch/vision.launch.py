"""Run the vision node, and start it again whenever it exits.

    pixi run vision                 # the HRI task's classes
    pixi run vision task:=hri
    pixi run vision sim:=true       # sim time, and the sim output topics

walkie-vision is a Pixi project, not an ament package, so there is no
executable for launch_ros to look up. The node runs as a plain process:
`python -P -m walkie_vision.node` with the parameter files in config/.

respawn starts the process again whenever it exits while the launch runs:
after a fatal failure (exit 1) or a startup failure (exit 2). Ctrl-C stops
the launch and does not respawn. Jazzy's launch has no growing delay;
respawn_max_retries caps the restarts when wanted.
"""

from __future__ import annotations

import sys
from pathlib import Path

from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config"
TASKS = CONFIG / "tasks"


def task_names() -> list[str]:
    """The tasks that have a parameter file in config/tasks/."""
    return sorted(p.stem for p in TASKS.glob("*.yaml"))


def node_command(task: str, sim: bool) -> list[str]:
    """Return the command that runs the node with the parameter files of a task.

    Later files override earlier ones: vision.yaml, then the task's file,
    then sim.yaml in simulation.
    """
    if task not in task_names():
        raise ValueError(f"unknown task {task!r}; config/tasks/ has: {', '.join(task_names())}")
    # -P leaves the working directory off sys.path, so a stray .py file among
    # the Ultralytics downloads cannot shadow a module. Unlike -I, it keeps
    # PYTHONPATH, which the message overlay needs.
    cmd = [sys.executable, "-P", "-m", "walkie_vision.node", "--ros-args"]
    cmd += ["--params-file", str(CONFIG / "vision.yaml"), "--params-file", str(TASKS / f"{task}.yaml")]
    if sim:
        cmd += ["--params-file", str(CONFIG / "sim.yaml")]
    return cmd


def vision_process(
    cmd: list[str],
    cwd: Path,
    respawn_delay: float,
    respawn_max_retries: int = -1,
    *,
    models: Path | None = None,
) -> ExecuteProcess:
    """Return the node's process, restarted respawn_delay seconds after it exits.

    respawn_max_retries < 0 restarts it without limit. models, when given,
    is passed to the node as WALKIE_VISION_MODELS.
    """
    env = {"YOLO_OFFLINE": "true"}  # stops Ultralytics' online checks; it reads this at import
    if models is not None:
        env["WALKIE_VISION_MODELS"] = str(models)
    return ExecuteProcess(
        cmd=cmd,
        cwd=str(cwd),
        name="walkie_vision",
        output="screen",
        emulate_tty=True,
        respawn=True,
        respawn_delay=respawn_delay,
        respawn_max_retries=respawn_max_retries,
        additional_env=env,
    )


def _bool(name: str, text: str) -> bool:
    value = text.strip().lower()
    if value in ("true", "1", "yes"):
        return True
    if value in ("false", "0", "no"):
        return False
    raise ValueError(f"{name} must be true or false, not {text!r}")


def _launch(context: LaunchContext) -> list[ExecuteProcess]:
    from walkie_vision.models import ULTRALYTICS_DIR, models_dir

    task = LaunchConfiguration("task").perform(context)
    sim = _bool("sim", LaunchConfiguration("sim").perform(context))
    delay = float(LaunchConfiguration("respawn_delay").perform(context))
    retries = int(LaunchConfiguration("respawn_max_retries").perform(context))
    # The node runs in another working directory, so a relative
    # WALKIE_VISION_MODELS would point somewhere else there. It gets the
    # absolute path instead.
    models = models_dir().resolve()
    # The working directory: any Ultralytics download lands there, in a
    # folder that Git ignores.
    cwd = models / ULTRALYTICS_DIR
    if not cwd.is_dir():
        raise RuntimeError(f"{cwd} does not exist; run pixi run fetch-models")
    return [vision_process(node_command(task, sim), cwd, delay, retries, models=models)]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("task", default_value="hri", description="A task with a file in config/tasks/"),
            DeclareLaunchArgument(
                "sim", default_value="false", description="Use sim time and publish on the sim topics"
            ),
            DeclareLaunchArgument(
                "respawn_delay", default_value="3.0", description="Seconds before the node starts again"
            ),
            DeclareLaunchArgument(
                "respawn_max_retries", default_value="-1", description="How many restarts; -1 means no limit"
            ),
            OpaqueFunction(function=_launch),
        ]
    )
