"""Values other repositories read from walkie-vision, checked from this repository alone.

walkie-database and walkie-agent-ros read these without importing the package
at a fixed module path, so a move inside the package keeps them working only
while the values, the log lines and a few file paths stay the same:

- COLOR_NAMES: walkie-database's tests/test_cross_repo_contract.py reads it from
  this repository's source with ast, as exactly one literal module-level
  assignment anywhere under the walkie_vision package, and compares its length
  with the colour shares the database accepts.
- the vector model names (clip_vit_b16, arcface_w600k_r50, osnet_x1_0_msmt17,
  the fake_ prefix and the fakes' own names) and the CLIP vector length:
  walkie-database stores vectors under these names, and walkie-agent-ros
  copies them.
- the object cloud layout (POINT_STEP and the x, y, z fields) and hash_vector's
  values: walkie-agent-ros's oracle perception copies both, and its
  tests/test_track_vectors.py pins the two hash_vector values below.
- src/walkie_vision/fakes.py, debug_viz.py and fake_publisher.py, and the task
  files: walkie-agent-ros's scripts/chula-sim.sh checks for debug_viz.py and
  config/tasks/<task>.yaml, its comments cite fakes.py, and fake_publisher.py
  finds its default scene from its own path.
- the log lines `loaded the models in` (INFO) and `cannot start:` (FATAL):
  chula-sim.sh waits for one of them before it starts the agent.
- the node name walkie_vision: the parameter files the launch passes
  (config/vision.yaml, config/sim.yaml, config/tasks/*.yaml) are keyed by it,
  and the node's log lines carry it.

The tests find a definition by reading every module of the package with ast
(defining_module), not by importing a fixed path, so they hold while the
package's modules move. Only the paths that other repositories name are
imported or checked directly: walkie_vision.node, walkie_vision.providers,
walkie_vision.fakes and walkie_vision.fake_publisher.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sensor_msgs.msg import PointField
from std_msgs.msg import Header

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PACKAGE = "walkie_vision"
PACKAGE_DIR = SRC / PACKAGE

# The logger calls whose lines a node at the default level writes to its log.
LOGGED_LEVELS = {"info", "warning", "warn", "error", "fatal"}
# walkie-database's contract test skips these folders when it looks for the package.
SKIPPED_BY_DATABASE = {".git", ".pixi", "build", "install"}

COLOR_NAMES = (
    "black", "white", "grey", "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "beige",
)  # fmt: skip
VECTOR_MODEL_NAMES = {
    "CLIP_VIT_B16": "clip_vit_b16",
    "ARCFACE_W600K_R50": "arcface_w600k_r50",
    "OSNET_X1_0_MSMT17": "osnet_x1_0_msmt17",
    "FAKE_MODEL_PREFIX": "fake_",
}
FAKE_MODEL_NAMES = {
    "FakeEmbedder": "fake_embedder",
    "FakeReIdentifier": "fake_reid",
    "FakeFaceRecognizer": "fake_face",
    "FakePersonFaceRecognizer": "fake_face",
}
CLOUD_FIELDS = [("x", 0, PointField.FLOAT32, 1), ("y", 4, PointField.FLOAT32, 1), ("z", 8, PointField.FLOAT32, 1)]
# The first four values of hash_vector(data, 8), as walkie-agent-ros's
# tests/test_track_vectors.py pins them.
HASH_VECTORS = {
    "guest sam": [-0.2260013222694397, 0.31027519702911377, 0.2287740558385849, -0.33814337849617004],
    b"face 1": [-0.24494804441928864, -0.41149866580963135, 0.19372032582759857, -0.055320821702480316],
}
FILES_READ_ELSEWHERE = (
    "src/walkie_vision/fakes.py",
    "src/walkie_vision/debug_viz.py",
    "src/walkie_vision/fake_publisher.py",
    "config/tasks/hri.yaml",
    "config/tasks/memory.yaml",
)
MODELS_LOADED = "loaded the models in"
CANNOT_START = "cannot start:"


def python_files(package_dir: Path) -> list[Path]:
    return sorted(package_dir.rglob("*.py"))


def module_name(path: Path, package_dir: Path) -> str:
    """The dotted name of the module in *path*, a file under *package_dir*."""
    parts = path.relative_to(package_dir.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def defined_names(statement: ast.stmt) -> set[str]:
    """The names a module-level statement defines: assignments, functions and classes, not imports."""
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {statement.name}
    if isinstance(statement, ast.Assign):
        targets = [
            element
            for target in statement.targets
            for element in (target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target])
        ]
    elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
        targets = [statement.target]
    else:
        return set()
    return {target.id for target in targets if isinstance(target, ast.Name)}


def definitions(package_dir: Path, name: str) -> list[Path]:
    """Every file under *package_dir* that defines *name* at module level."""
    return [
        path
        for path in python_files(package_dir)
        if any(name in defined_names(statement) for statement in ast.parse(path.read_text(), str(path)).body)
    ]


def defining_module(name: str) -> Any:
    """The value of *name* from the one module of the package that defines it, wherever that module is."""
    found = definitions(PACKAGE_DIR, name)
    assert len(found) == 1, (
        f"{name} is read from the one module that defines it, and {len(found)} modules of "
        f"{PACKAGE_DIR.relative_to(ROOT)} define it: {[str(path.relative_to(ROOT)) for path in found]}"
    )
    module = importlib.import_module(module_name(found[0], PACKAGE_DIR))
    assert Path(module.__file__).resolve() == found[0].resolve(), (
        f"{module.__name__} was imported from {module.__file__}, not from {found[0]}"
    )
    return getattr(module, name)


def literal_definitions(package_dir: Path, name: str) -> list[tuple[Path, Any]]:
    """Each module-level assignment of *name* under *package_dir*, read as walkie-database reads it.

    A copy of the search in walkie-database's module_constant: every Assign and
    AnnAssign with a value, in any .py file, its value read with
    ast.literal_eval. A value that is not a literal (an alias such as
    `COLOR_NAMES = crop.COLOR_NAMES`, or a computed tuple) raises there; here it
    fails the test with the file and line.
    """
    found = []
    for path in python_files(package_dir):
        for statement in ast.parse(path.read_text(), filename=str(path)).body:
            if isinstance(statement, ast.Assign):
                targets = statement.targets
            elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
                targets = [statement.target]
            else:
                continue
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                try:
                    found.append((path, ast.literal_eval(statement.value)))
                    continue
                except (ValueError, TypeError):
                    pass
                pytest.fail(
                    f"{path.relative_to(package_dir.parent)}:{statement.lineno} assigns {name} a value that "
                    "is not a literal, and walkie-database reads it with ast.literal_eval: keep one literal "
                    "assignment and import it everywhere else"
                )
    return found


def package_folders(root: Path) -> list[Path]:
    """Every walkie_vision folder with an __init__.py under *root*, skipping what walkie-database skips."""
    found = []
    for folder, subfolders, files in os.walk(root):
        subfolders[:] = sorted(name for name in subfolders if name not in SKIPPED_BY_DATABASE)
        if Path(folder).name == PACKAGE and "__init__.py" in files:
            found.append(Path(folder))
    return found


def leading_text(value: ast.expr) -> str:
    """The text a string expression starts with: a string, an f-string or a sum that starts with one."""
    if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
        return leading_text(value.left)
    if isinstance(value, ast.JoinedStr) and value.values:
        value = value.values[0]
    return value.value if isinstance(value, ast.Constant) and isinstance(value.value, str) else ""


class _LoggedLines(ast.NodeVisitor):
    """The lines of the logger calls at info or above whose message starts with a prefix.

    The message is the call's first argument: a string, an f-string, or a name
    that the same function (or the module) assigned one to before the call, as
    in `message = f"cannot start: ..."` followed by `logger.fatal(message)`.
    """

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.scopes: list[dict[str, str]] = [{}]
        self.lines: list[int] = []

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.scopes.append({})
        self.generic_visit(node)
        self.scopes.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node: ast.Assign) -> None:
        self.generic_visit(node)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.scopes[-1][target.id] = leading_text(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.generic_visit(node)
        if isinstance(node.target, ast.Name) and node.value is not None:
            self.scopes[-1][node.target.id] = leading_text(node.value)

    def visit_Call(self, node: ast.Call) -> None:
        self.generic_visit(node)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in LOGGED_LEVELS
            and node.args
            and self.message(node.args[0]).startswith(self.prefix)
        ):
            self.lines.append(node.lineno)

    def message(self, argument: ast.expr) -> str:
        if isinstance(argument, ast.Name):
            return next((scope[argument.id] for scope in reversed(self.scopes) if argument.id in scope), "")
        return leading_text(argument)


def logged_lines_starting_with(package_dir: Path, prefix: str) -> list[str]:
    """Where *package_dir* passes a line that starts with *prefix* to a logger call at info or above."""
    found = []
    for path in python_files(package_dir):
        visitor = _LoggedLines(prefix)
        visitor.visit(ast.parse(path.read_text(), filename=str(path)))
        found += [f"{path.relative_to(package_dir.parent)}:{line}" for line in visitor.lines]
    return found


def test_the_contract_is_read_from_this_tree():
    spec = importlib.util.find_spec(PACKAGE)
    assert spec is not None and spec.origin is not None
    assert Path(spec.origin).is_relative_to(SRC)


def test_walkie_database_finds_one_walkie_vision_package_in_this_repository():
    assert package_folders(ROOT) == [PACKAGE_DIR], (
        "walkie-database reads COLOR_NAMES from the one walkie_vision folder with an __init__.py "
        "in its clone of this repository, and fails when there is another one"
    )


def test_colour_names_are_one_literal_that_walkie_database_can_read():
    found = literal_definitions(PACKAGE_DIR, "COLOR_NAMES")

    assert [path for path, _ in found] == definitions(PACKAGE_DIR, "COLOR_NAMES")
    assert len(found) == 1, (
        "walkie-database reads COLOR_NAMES from exactly one literal module-level assignment "
        f"under {PACKAGE_DIR.relative_to(ROOT)}, and it is assigned {len(found)} times: "
        f"{[str(path.relative_to(ROOT)) for path, _ in found]}"
    )
    assert found[0][1] == COLOR_NAMES, (
        "GetPersonCrop's colour shares and walkie-database's COLOR_SHARES follow these names in this order"
    )
    assert defining_module("COLOR_NAMES") == COLOR_NAMES


@pytest.mark.parametrize("name", sorted(VECTOR_MODEL_NAMES))
def test_the_stored_vector_model_names_keep_their_values(name):
    import walkie_vision.providers

    value = VECTOR_MODEL_NAMES[name]
    assert defining_module(name) == value, (
        f"walkie-database stores vectors under {value!r} and walkie-agent-ros copies it, so {name} "
        "must keep its value"
    )
    assert getattr(walkie_vision.providers, name) == value


def test_the_clip_vector_length_is_the_one_the_database_stores():
    assert defining_module("EMBEDDING_DIMS") == {"clip_vit_b16": 512}


def test_the_fakes_name_their_vectors_as_walkie_agent_ros_copies_them():
    import walkie_vision.fakes

    defaults = {
        fake: inspect.signature(getattr(walkie_vision.fakes, fake)).parameters["model_name"].default
        for fake in FAKE_MODEL_NAMES
    }

    assert defaults == FAKE_MODEL_NAMES
    assert all(name.startswith(VECTOR_MODEL_NAMES["FAKE_MODEL_PREFIX"]) for name in defaults.values())


def test_object_clouds_keep_the_point_layout_walkie_agent_ros_copies():
    header = Header(frame_id="camera_color_optical_frame")
    points = np.column_stack([np.linspace(-0.5, 0.5, 40), np.linspace(-0.2, 0.2, 40), np.linspace(1.0, 2.0, 40)])

    cloud = defining_module("to_point_cloud")(header, points)
    empty = defining_module("empty_cloud")(header)

    assert defining_module("POINT_STEP") == 12
    for built in (cloud, empty):
        assert [(f.name, f.offset, f.datatype, f.count) for f in built.fields] == CLOUD_FIELDS
        assert (built.height, built.point_step, built.row_step, built.is_bigendian) == (1, 12, 12 * built.width, False)
    assert cloud.width == len(points)
    assert bytes(cloud.data) == points.astype("<f4").tobytes()
    assert (empty.width, bytes(empty.data)) == (0, b"")


@pytest.mark.parametrize("data", list(HASH_VECTORS), ids=repr)
def test_hash_vector_gives_the_values_walkie_agent_ros_pins(data):
    vector = defining_module("hash_vector")(data, 8)

    assert vector.dtype == np.float32
    assert vector[:4].tolist() == pytest.approx(HASH_VECTORS[data])


@pytest.mark.parametrize("path", FILES_READ_ELSEWHERE)
def test_the_files_other_repositories_name_exist(path):
    assert (ROOT / path).is_file(), f"walkie-agent-ros names {path}, so it must stay where it is"


def test_the_fake_publisher_finds_its_default_scene():
    from walkie_vision.fake_publisher import DEFAULT_SCENE

    assert DEFAULT_SCENE == ROOT / "config" / "fakes" / "boxes.json"
    assert DEFAULT_SCENE.is_file()


@pytest.mark.parametrize("prefix", [MODELS_LOADED, CANNOT_START])
def test_one_call_logs_each_line_that_chula_sim_waits_for(prefix):
    logged = logged_lines_starting_with(PACKAGE_DIR, prefix)

    assert len(logged) == 1, (
        f"walkie-agent-ros's scripts/chula-sim.sh waits for a log line that starts with {prefix!r}, "
        f"so exactly one logger call at info or above must log it, and these do: {logged}"
    )


def test_the_node_is_named_walkie_vision():
    from walkie_vision.node import NODE_NAME

    assert NODE_NAME == "walkie_vision", (
        "the parameter files in config/ (vision.yaml, sim.yaml, tasks/*.yaml) are keyed by the node name "
        "walkie_vision, and the node's log lines carry it; a rename makes the node ignore its task and sim "
        "parameters"
    )


def test_a_second_definition_is_counted_and_an_import_is_not(tmp_path):
    package = tmp_path / "probe"
    package.mkdir()
    (package / "__init__.py").write_text('"""Probe."""\n')
    (package / "a.py").write_text("NAME = ('one',)\n")
    (package / "b.py").write_text("def NAME():\n    pass\n")
    (package / "c.py").write_text("from .a import NAME\nOTHER = NAME\n")
    (package / "d.py").write_text("NAME: tuple = ('two',)\n")

    assert definitions(package, "NAME") == [package / "a.py", package / "b.py", package / "d.py"]
    assert literal_definitions(package, "NAME") == [(package / "a.py", ("one",)), (package / "d.py", ("two",))]
    assert module_name(package / "__init__.py", package) == "probe"
    assert module_name(package / "a.py", package) == "probe.a"


def test_a_value_that_is_not_a_literal_fails_with_its_file_and_line(tmp_path):
    package = tmp_path / "probe"
    package.mkdir()
    (package / "a.py").write_text("NAME = ('one',)\n")
    (package / "b.py").write_text("from . import a\n\nNAME = a.NAME\n")

    with pytest.raises(pytest.fail.Exception, match=r"probe/b\.py:3 assigns NAME a value that is not a literal"):
        literal_definitions(package, "NAME")


def test_a_log_line_is_found_through_the_name_it_is_assigned_to(tmp_path):
    package = tmp_path / "probe"
    package.mkdir()
    (package / "a.py").write_text(
        "def start(logger, error):\n"
        "    message = f'cannot start: {error}'\n"
        "    message += '!'\n"
        "    logger.fatal(message)\n"
        "\n"
        "\n"
        "def ready(logger):\n"
        "    logger.info(f'loaded the models in {1.0:.1f} s')\n"
        "    logger.debug('loaded the models in no time')\n"
        "    logger.warning('loaded ' + 'the models in')\n"
        "    message = 'cannot start: elsewhere'\n"
        "\n"
        "\n"
        "def later(logger, message):\n"
        "    logger.error(message)\n"
    )

    assert logged_lines_starting_with(package, CANNOT_START) == ["probe/a.py:4"]
    assert logged_lines_starting_with(package, MODELS_LOADED) == ["probe/a.py:8"]
