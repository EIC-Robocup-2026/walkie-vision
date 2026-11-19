"""The package keeps the import rules in pyproject.toml's [tool.importlinter] section.

import-linter runs in this process through its Python API, so the check needs no
lint-imports executable and works in any environment that has the package. It builds
its graph from the walkie_vision that the import system finds, which PYTHONPATH can
point at another checkout, so the first test pins it to this tree.

Two rules that the contracts do not express are checked here directly:

- The layers contract names every top-level module exactly once. Its exhaustive
  option rejects a module that is in no layer; this test also rejects a module named
  twice, or a name that is no module.
- Every __init__.py holds exactly one statement, a non-empty docstring, except the
  packages in REEXPORTS. Python runs a package's __init__ before any module in it,
  so code there runs on every import of any of its modules. A package in REEXPORTS
  holds its docstring, from-imports of names from the modules REEXPORTS lists, and
  __all__ naming exactly those names, and nothing else. Every folder that holds
  Python code has an __init__.py: grimp leaves a folder without one out of the
  graph, so no contract would see its modules.
"""

import ast
import importlib.util
import pkgutil
import re
import tomllib
from collections.abc import Iterable, Mapping
from pathlib import Path

import grimp
from importlinter import api  # noqa: F401  Importing it configures import-linter's adapters.
from importlinter.application import output, use_cases

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PYPROJECT = ROOT / "pyproject.toml"
PACKAGE = "walkie_vision"
PACKAGE_DIR = SRC / PACKAGE

# The external packages that count as ROS: the client library, the generated
# interfaces and their messages, TF, launch and the build tools.
ROS_PACKAGE = re.compile(
    r"rclpy|.*_msgs|.*_srvs|.*_interfaces|rosidl_.*|tf2_.*|message_filters|launch.*|ament_.*"
)

# A layer of the layers contract lists its modules split by "|" (independent)
# or ":" (may import each other).
LAYER_SEPARATOR = re.compile(r"[|:]")

# The packages whose __init__.py re-exports names, and the modules of the package
# (relative to it) it may take them from. walkie_vision.providers re-exports the
# names of the five base modules and vector_tables, so walkie_vision.providers.Detection
# keeps working; importing it never loads a registry or a provider module.
REEXPORTS = {
    "walkie_vision.providers": frozenset(
        {
            "face_recognition.base",
            "image_embed.base",
            "object_detection.base",
            "pose_estimation.base",
            "reid.base",
            "vector_tables",
        }
    ),
}


def lint(config: Path) -> tuple[bool, str]:
    """Check every contract in config; return whether all were kept, and the report."""
    with output.console.capture() as captured:
        kept = use_cases.lint_imports(str(config), cache_dir=None, no_logo=True)
    return kept, captured.get()


def contracts() -> list[dict]:
    """The contracts in pyproject.toml."""
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["tool"]["importlinter"]["contracts"]


def contract(contract_id: str) -> dict:
    """The contract with this id in pyproject.toml."""
    (found,) = [entry for entry in contracts() if entry["id"] == contract_id]
    return found


def layer_names(layers: Iterable[str]) -> list[str]:
    """Every module name in a layers contract's layers, in order, repeats kept."""
    return [name.strip() for layer in layers for name in LAYER_SEPARATOR.split(layer)]


def is_docstring_only(body: list[ast.stmt]) -> bool:
    """Whether a module body is exactly one statement, a non-empty string docstring."""
    return (
        len(body) == 1
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
        and bool(body[0].value.value.strip())
    )


def is_string_list(node: ast.expr) -> bool:
    """Whether an expression is a list or a tuple of string literals."""
    return isinstance(node, ast.List | ast.Tuple) and all(
        isinstance(element, ast.Constant) and isinstance(element.value, str) for element in node.elts
    )


def reexport_problems(path: str, body: list[ast.stmt], modules: Iterable[str]) -> list[str]:
    """Where a re-exporting __init__.py breaks its rule.

    It holds a non-empty docstring, then from-imports of names from the listed
    modules of its own package (each name under its own name, no wildcard), then
    __all__ as a literal list of exactly the imported names, and nothing else.
    """
    allowed = set(modules)
    problems = []
    statements = body
    if body and is_docstring_only(body[:1]):
        statements = body[1:]
    else:
        problems.append(f"{path} must start with a non-empty docstring")
    imported: list[str] = []
    exported: list[str] | None = None
    for statement in statements:
        where = f"{path}:{statement.lineno}"
        if isinstance(statement, ast.ImportFrom) and statement.level == 1 and statement.module in allowed:
            for alias in statement.names:
                if alias.name == "*" or alias.asname not in (None, alias.name):
                    problems.append(f"{where} must import each name under its own name: {ast.unparse(statement)}")
                else:
                    imported.append(alias.name)
        elif (
            exported is None
            and isinstance(statement, ast.Assign)
            and [ast.unparse(target) for target in statement.targets] == ["__all__"]
            and is_string_list(statement.value)
        ):
            exported = [element.value for element in statement.value.elts]
        else:
            problems.append(
                f"{where} may only import names from .{', .'.join(sorted(allowed))} and list them in __all__:"
                f" {ast.unparse(statement)}"
            )
    if exported is None:
        problems.append(f"{path} must list the names it re-exports in __all__")
    elif sorted(exported) != sorted(imported) or len(set(exported)) != len(exported):
        problems.append(f"{path}: __all__ must list each name it imports once, and no other")
    return problems


def init_problems(
    package_dir: Path,
    package: str,
    reexports: Mapping[str, Iterable[str]] | None = None,
) -> list[str]:
    """Every folder with Python code but no __init__.py, every __init__.py that breaks the rule,
    and every stale REEXPORTS entry.

    Python imports a folder without __init__.py as a namespace package, but grimp
    leaves it out of the graph, so the contracts would skip it.
    """
    reexports = reexports or {}
    packages = set()
    sources = [source.relative_to(package_dir).parts for source in package_dir.rglob("*.py")]
    folders = {package_dir.joinpath(*parts[:depth]) for parts in sources for depth in range(len(parts))}
    problems = [
        f"{folder.relative_to(package_dir).as_posix()}/ holds Python files but no __init__.py,"
        " so grimp and the contracts skip it"
        for folder in sorted(folders)
        if not (folder / "__init__.py").is_file()
    ]
    for init in sorted(package_dir.rglob("__init__.py")):
        name = ".".join((package, *init.parent.relative_to(package_dir).parts))
        packages.add(name)
        path = init.relative_to(package_dir).as_posix()
        body = ast.parse(init.read_text(encoding="utf-8"), filename=str(init)).body
        if name in reexports:
            problems += reexport_problems(path, body, reexports[name])
        elif not is_docstring_only(body):
            problems.append(f"{path} must hold exactly one statement, a non-empty docstring")
    problems += [f"{name} is in REEXPORTS but is not a package" for name in sorted(set(reexports) - packages)]
    return problems


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_the_contracts_check_this_tree():
    spec = importlib.util.find_spec(PACKAGE)
    assert spec is not None and spec.origin is not None
    assert Path(spec.origin).is_relative_to(SRC)


def test_the_import_contracts_hold():
    kept, report = lint(PYPROJECT)
    assert kept, report
    assert f"{len(contracts())} kept, 0 broken" in report, report


def test_the_ros_contract_forbids_every_ros_package_the_code_imports():
    graph = grimp.build_graph(PACKAGE, include_external_packages=True, cache_dir=None)
    ros = {
        module
        for module in graph.modules
        if module != PACKAGE and not module.startswith(PACKAGE + ".") and ROS_PACKAGE.fullmatch(module)
    }
    assert "rclpy" in ros
    missing = sorted(ros - set(contract("model-code-knows-no-ros")["forbidden_modules"]))
    assert not missing, f"ROS packages the model code may still import: {missing}"


def test_a_broken_contract_fails_with_the_chain(tmp_path, monkeypatch):
    package = tmp_path / "contract_probe"
    _write(package / "__init__.py", '"""A probe package."""\n')
    _write(package / "model.py", "from contract_probe import ros_side\n")
    _write(package / "ros_side.py", "import json\n")
    config = tmp_path / "pyproject.toml"
    config.write_text(
        "[tool.importlinter]\n"
        'root_package = "contract_probe"\n'
        "[[tool.importlinter.contracts]]\n"
        'name = "The model code stays off the ROS side"\n'
        'type = "forbidden"\n'
        'source_modules = ["contract_probe.model"]\n'
        'forbidden_modules = ["contract_probe.ros_side"]\n'
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    kept, report = lint(config)

    assert not kept
    assert "contract_probe.model -> contract_probe.ros_side" in report


def test_the_layers_name_every_top_level_module_exactly_once():
    layers = contract("layers-point-down")
    on_disk = sorted(
        info.name for info in pkgutil.iter_modules([str(PACKAGE_DIR)]) if info.name not in layers["exhaustive_ignores"]
    )
    names = layer_names(layers["layers"])
    assert sorted(names) == on_disk, (
        f"in no layer: {sorted(set(on_disk) - set(names))};"
        f" not a module: {sorted(set(names) - set(on_disk))};"
        f" named twice: {sorted({name for name in names if names.count(name) > 1})}"
    )


def test_an_upward_or_sideways_import_breaks_the_layers_with_its_chain(tmp_path, monkeypatch):
    package = tmp_path / "contract_probe"
    _write(package / "__init__.py", '"""A probe package."""\n')
    _write(package / "top.py", "from contract_probe import middle\n")
    _write(package / "middle" / "__init__.py", '"""A probe subpackage."""\n')
    _write(package / "middle" / "inner.py", "import json\n\nfrom contract_probe import bottom, top\n")
    _write(package / "left.py", "from contract_probe import bottom\n")
    _write(package / "right.py", "from contract_probe import left\n")
    _write(package / "bottom.py", "import json\n")
    config = tmp_path / "pyproject.toml"
    config.write_text(
        "[tool.importlinter]\n"
        'root_package = "contract_probe"\n'
        "[[tool.importlinter.contracts]]\n"
        'name = "The probe layers point down"\n'
        'type = "layers"\n'
        'containers = ["contract_probe"]\n'
        "exhaustive = true\n"
        'layers = ["top", "middle", "left | right", "bottom"]\n'
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    kept, report = lint(config)

    assert not kept
    assert "contract_probe.middle.inner -> contract_probe.top (l.3)" in report, report
    assert "contract_probe.right -> contract_probe.left (l.1)" in report, report


def test_every_package_init_holds_only_a_docstring_unless_allowed():
    problems = init_problems(PACKAGE_DIR, PACKAGE, REEXPORTS)
    assert not problems, "\n".join(problems)


def test_an_init_with_code_or_a_folder_without_one_is_reported(tmp_path):
    package = tmp_path / "contract_probe"
    _write(package / "__init__.py", '"""A probe package."""\n')
    _write(package / "assigns" / "__init__.py", "x = 1\n")
    _write(package / "code_after" / "__init__.py", '"""Docstring then code."""\n\nx = 1\n')
    _write(package / "empty" / "__init__.py", "")
    _write(package / "fine" / "__init__.py", '"""Only a docstring."""\n')
    _write(package / "assigns" / "loose" / "leak.py", "import json\n")
    _write(package / "outer" / "inner" / "deep.py", "import json\n")

    problems = init_problems(package, "contract_probe")

    assert problems == [
        "assigns/loose/ holds Python files but no __init__.py, so grimp and the contracts skip it",
        "outer/ holds Python files but no __init__.py, so grimp and the contracts skip it",
        "outer/inner/ holds Python files but no __init__.py, so grimp and the contracts skip it",
        "assigns/__init__.py must hold exactly one statement, a non-empty docstring",
        "code_after/__init__.py must hold exactly one statement, a non-empty docstring",
        "empty/__init__.py must hold exactly one statement, a non-empty docstring",
    ]


def test_a_reexporting_init_that_breaks_its_rule_is_reported(tmp_path):
    package = tmp_path / "contract_probe"
    _write(package / "__init__.py", '"""A probe package."""\n')
    _write(
        package / "fine" / "__init__.py",
        '"""Re-exports."""\n\nfrom .base import A, B\nfrom .tables import C as C\n\n__all__ = ["A", "B", "C"]\n',
    )
    _write(
        package / "broken" / "__init__.py",
        "from .base import A\n"
        "from .registry import Spec\n"
        "from .base import B as Renamed\n"
        "from .base import *\n"
        "from . import base\n"
        "x = 1\n"
        '__all__ = ["A", "Z"]\n',
    )
    _write(package / "unlisted" / "__init__.py", '"""Re-exports without __all__."""\nfrom .base import A\n')
    _write(package / "twice" / "__init__.py", '"""Re-exports."""\nfrom .base import A\n__all__ = ["A", "A"]\n')
    reexports = {
        "contract_probe.fine": {"base", "tables"},
        "contract_probe.broken": {"base"},
        "contract_probe.unlisted": {"base"},
        "contract_probe.twice": {"base"},
        "contract_probe.gone": {"base"},
    }

    problems = init_problems(package, "contract_probe", reexports)

    assert problems == [
        "broken/__init__.py must start with a non-empty docstring",
        "broken/__init__.py:2 may only import names from .base and list them in __all__: from .registry import Spec",
        "broken/__init__.py:3 must import each name under its own name: from .base import B as Renamed",
        "broken/__init__.py:4 must import each name under its own name: from .base import *",
        "broken/__init__.py:5 may only import names from .base and list them in __all__: from . import base",
        "broken/__init__.py:6 may only import names from .base and list them in __all__: x = 1",
        "broken/__init__.py: __all__ must list each name it imports once, and no other",
        "twice/__init__.py: __all__ must list each name it imports once, and no other",
        "unlisted/__init__.py must list the names it re-exports in __all__",
        "contract_probe.gone is in REEXPORTS but is not a package",
    ]
