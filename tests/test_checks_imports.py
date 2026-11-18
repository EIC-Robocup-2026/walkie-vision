"""The scripts, commands and docs outside the package name walkie_vision modules that exist.

pytest never runs checks c04-c11 (only `pixi run check-env` does, on a computer
with a GPU), tools/bakeoff/propose.py or the launch file's command lines, and
nothing builds the docs against the code. A module that moves or a name that
leaves it would break them silently, so this file reads them without running
them and resolves every walkie_vision name they use (it reads the c12 checks and
tools/bakeoff/embedders.py as well, which other tests already run):

- every `import walkie_vision.X` and `from walkie_vision.X import a, b` in the
  Python files under checks/, tools/bakeoff/ and launch/, at any depth (the
  launch file imports inside a function): X is a module, and each imported name
  is an attribute or a submodule of it;
- every `-m walkie_vision.X` command in pixi.toml and in those Python files,
  as text or as a "-m", "walkie_vision.X" pair in a list: X is a module;
- in the live docs (README.md, CLAUDE.md and docs/, without the records in
  docs/setup/ and docs/architecture/decisions/), every dotted walkie_vision name
  inside backticks or a fenced code block is a module, or a module followed by
  an attribute chain, and the names of a `from walkie_vision.X import a, b` there
  are attributes or submodules of X. Prose outside code is not read, so a
  sentence may use the package name freely.

Resolving a name imports the module, which may load torch or onnxruntime; the
suite imports both already. Every unresolved name is reported at once.
"""

import ast
import functools
import importlib
import importlib.util
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PACKAGE = "walkie_vision"

SCRIPT_DIRS = ("checks", "tools/bakeoff", "launch")
COMMAND_FILES = ("pixi.toml",)
LIVE_DOCS = ("README.md", "CLAUDE.md")
DOCS_DIR = "docs"
RECORDS = ("docs/setup", "docs/architecture/decisions")

DOTTED = re.compile(rf"(?<![\w.]){PACKAGE}(?:\.[A-Za-z_]\w*)+")
COMMAND = re.compile(rf"-m\s+({PACKAGE}(?:\.[A-Za-z_]\w*)*)(?![\w.])")
FROM_IMPORT = re.compile(
    rf"\bfrom\s+({PACKAGE}(?:\.[A-Za-z_]\w*)*)\s+import\s+([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)"
)
FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
CODE_SPAN = re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)")


def is_ours(module: str) -> bool:
    """Whether module is the package or one of its modules."""
    return module == PACKAGE or module.startswith(PACKAGE + ".")


def module_problem(name: str) -> str | None:
    """None when the import system finds a module of this name; otherwise why not.

    find_spec imports the parent packages first, and raises when one of them
    is missing, is a plain module or fails to import.
    """
    try:
        spec = importlib.util.find_spec(name)
    except Exception as error:  # noqa: BLE001  Reported with the others, not raised.
        return f"no module {name}: {error!r}"
    return None if spec is not None else f"no module {name}"


@functools.cache
def resolve(dotted: str) -> str | None:
    """None when dotted names a module, or a module followed by an attribute chain;
    otherwise why it does not resolve.

    Each component is a submodule when its parent is a package that holds one of
    that name, and an attribute otherwise.
    """
    parts = dotted.split(".")
    missing = module_problem(parts[0])
    if missing is not None:
        return missing
    obj: object = importlib.import_module(parts[0])
    for depth, part in enumerate(parts[1:], start=1):
        parent = ".".join(parts[:depth])
        if (
            isinstance(obj, ModuleType)
            and hasattr(obj, "__path__")
            and module_problem(f"{obj.__name__}.{part}") is None
        ):
            try:
                obj = importlib.import_module(f"{obj.__name__}.{part}")
            except Exception as error:  # noqa: BLE001  Reported with the others, not raised.
                return f"{obj.__name__}.{part} fails to import: {error!r}"
        elif hasattr(obj, part):
            obj = getattr(obj, part)
        else:
            return f"{parent} has no attribute or submodule {part}"
    return None


def relative(path: Path, root: Path) -> str:
    """path relative to root, for the report."""
    return path.relative_to(root).as_posix()


def script_files(root: Path, dirs: Iterable[str] = SCRIPT_DIRS) -> list[Path]:
    """Every Python file under the script folders of root."""
    return sorted(source for folder in dirs for source in (root / folder).rglob("*.py"))


def live_docs(root: Path) -> list[Path]:
    """README.md, CLAUDE.md and every page under docs/ except the records."""
    docs = [root / name for name in LIVE_DOCS if (root / name).is_file()]
    records = [root / record for record in RECORDS]
    docs += [
        page
        for page in sorted((root / DOCS_DIR).rglob("*.md"))
        if not any(page.is_relative_to(record) for record in records)
    ]
    return docs


def import_problems(source: Path, root: Path) -> list[str]:
    """Every walkie_vision import in a Python file that does not resolve."""
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    where = relative(source, root)
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                missing = module_problem(alias.name) if is_ours(alias.name) else None
                if missing is not None:
                    problems.append(f"{where}:{node.lineno} import {alias.name}: {missing}")
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module and is_ours(node.module):
            missing = module_problem(node.module)
            if missing is not None:
                problems.append(f"{where}:{node.lineno} from {node.module} import ...: {missing}")
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                why = resolve(f"{node.module}.{alias.name}")
                if why is not None:
                    problems.append(f"{where}:{node.lineno} from {node.module} import {alias.name}: {why}")
    return problems


def commands(path: Path) -> list[tuple[int, str]]:
    """The (line, module) of every `-m walkie_vision.X` in a file.

    The text form covers pixi tasks and docstrings; in a Python file, a "-m"
    string followed by a module string in a list or tuple counts too.
    """
    text = path.read_text(encoding="utf-8")
    found = {
        (text.count("\n", 0, match.start()) + 1, match.group(1)) for match in COMMAND.finditer(text)
    }
    if path.suffix == ".py":
        for node in ast.walk(ast.parse(text, filename=str(path))):
            if not isinstance(node, (ast.List, ast.Tuple)):
                continue
            for flag, module in zip(node.elts, node.elts[1:]):
                if (
                    isinstance(flag, ast.Constant)
                    and flag.value == "-m"
                    and isinstance(module, ast.Constant)
                    and isinstance(module.value, str)
                    and is_ours(module.value)
                ):
                    found.add((module.lineno, module.value))
    return sorted(found)


def command_problems(paths: Iterable[Path], root: Path) -> list[str]:
    """Every `-m walkie_vision.X` command whose module does not exist."""
    return [
        f"{relative(path, root)}:{line} -m {module}: {missing}"
        for path in paths
        for line, module in commands(path)
        if (missing := module_problem(module)) is not None
    ]


def code_text(markdown: str) -> Iterator[tuple[int, str]]:
    """The (line, text) of every fenced code line and inline code span in a Markdown page."""
    fence: str | None = None
    for number, line in enumerate(markdown.splitlines(), start=1):
        opening = FENCE.match(line)
        if fence is None:
            if opening:
                fence = opening.group(1)
                continue
            for span in CODE_SPAN.finditer(line):
                yield number, span.group(2)
        elif re.fullmatch(rf"\s*{re.escape(fence[0])}{{{len(fence)},}}\s*", line):
            fence = None
        else:
            yield number, line


def doc_names(markdown: str) -> list[tuple[int, str]]:
    """The (line, dotted name) of every walkie_vision name a page's code uses.

    A dotted name counts as written; `from walkie_vision.X import a, b` adds
    walkie_vision.X.a and walkie_vision.X.b.
    """
    found = set()
    for number, text in code_text(markdown):
        found.update((number, match.group(0)) for match in DOTTED.finditer(text))
        for match in FROM_IMPORT.finditer(text):
            module = match.group(1)
            found.update((number, f"{module}.{name.strip()}") for name in match.group(2).split(","))
    return sorted(found)


def doc_problems(pages: Iterable[Path], root: Path) -> list[str]:
    """Every walkie_vision name in the pages' code that does not resolve."""
    problems = []
    for page in pages:
        for line, name in doc_names(page.read_text(encoding="utf-8")):
            why = resolve(name)
            if why is not None:
                problems.append(f"{relative(page, root)}:{line} {name}: {why}")
    return problems


def test_the_names_resolve_against_this_tree():
    spec = importlib.util.find_spec(PACKAGE)
    assert spec is not None and spec.origin is not None
    assert Path(spec.origin).is_relative_to(SRC)


def test_the_check_scripts_tools_and_launch_file_import_names_that_exist():
    sources = script_files(ROOT)
    assert sources
    problems = [problem for source in sources for problem in import_problems(source, ROOT)]
    assert not problems, "imports that do not resolve:\n" + "\n".join(problems)


def test_every_module_command_names_a_module_that_exists():
    paths = [ROOT / name for name in COMMAND_FILES] + script_files(ROOT)
    assert any(commands(path) for path in paths)
    problems = command_problems(paths, ROOT)
    assert not problems, "-m commands that do not resolve:\n" + "\n".join(problems)


def test_the_live_docs_name_modules_that_exist():
    pages = live_docs(ROOT)
    assert any(doc_names(page.read_text(encoding="utf-8")) for page in pages)
    problems = doc_problems(pages, ROOT)
    assert not problems, "names in the docs that do not resolve:\n" + "\n".join(problems)


def test_a_missing_module_or_name_in_a_script_is_reported(tmp_path):
    script = tmp_path / "checks" / "c99_probe.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import json\n"
        "import walkie_vision\n"
        "from walkie_vision import providers\n"
        "from walkie_vision.nope import x\n"
        "\n"
        "\n"
        "def main():\n"
        "    from walkie_vision.providers import Nope\n"
        "    import walkie_vision.gone\n"
        "    from walkie_vision_msgs.msg import Missing\n"
    )

    problems = import_problems(script, tmp_path)

    assert problems == [
        "checks/c99_probe.py:4 from walkie_vision.nope import ...: no module walkie_vision.nope",
        "checks/c99_probe.py:8 from walkie_vision.providers import Nope:"
        " walkie_vision.providers has no attribute or submodule Nope",
        "checks/c99_probe.py:9 import walkie_vision.gone: no module walkie_vision.gone",
    ]


def test_a_command_with_a_missing_module_is_reported(tmp_path):
    toml = tmp_path / "pixi.toml"
    toml.write_text('[tasks.probe]\ncmd = "python -m walkie_vision.nope"\n')
    launch = tmp_path / "launch" / "probe.launch.py"
    launch.parent.mkdir()
    launch.write_text(
        '"""Runs `python -m walkie_vision` and `python -m walkie_vision.gone_too`."""\n'
        "\n"
        'cmd = ["python", "-P", "-m", "walkie_vision.gone",\n'
        '       "--ros-args"]\n'
        'fine = ("python", "-m", "walkie_vision")\n'
    )

    problems = command_problems([toml, launch], tmp_path)

    assert problems == [
        "pixi.toml:2 -m walkie_vision.nope: no module walkie_vision.nope",
        "launch/probe.launch.py:1 -m walkie_vision.gone_too: no module walkie_vision.gone_too",
        "launch/probe.launch.py:3 -m walkie_vision.gone: no module walkie_vision.gone",
    ]


def test_a_missing_name_in_a_live_doc_is_reported(tmp_path):
    page = tmp_path / "docs" / "guides" / "probe.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        "# Probe\n"
        "\n"
        "Prose may say walkie_vision.prose_only without backticks.\n"
        "The thing lives in `walkie_vision.nope.thing`, beside `walkie_vision.providers`.\n"
        "\n"
        "    ```bash\n"
        "    pixi run python -c \"from walkie_vision import Nope; from walkie_vision.providers import Gone\"\n"
        "    python -m walkie_vision.gone\n"
        "    ```\n"
    )
    record = tmp_path / "docs" / "setup" / "01-probe.md"
    record.parent.mkdir(parents=True)
    record.write_text("A record keeps `walkie_vision.old_home` as written.\n")

    pages = live_docs(tmp_path)
    problems = doc_problems(pages, tmp_path)

    assert pages == [page]
    assert problems == [
        "docs/guides/probe.md:4 walkie_vision.nope.thing: walkie_vision has no attribute or submodule nope",
        "docs/guides/probe.md:7 walkie_vision.Nope: walkie_vision has no attribute or submodule Nope",
        "docs/guides/probe.md:7 walkie_vision.providers.Gone:"
        " walkie_vision.providers has no attribute or submodule Gone",
        "docs/guides/probe.md:8 walkie_vision.gone: walkie_vision has no attribute or submodule gone",
    ]
