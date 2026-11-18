"""Nothing still names, imports or serves a module or a name at the home it left.

The package moves to its concept-first layout one module at a time, and no move
leaves a shim behind: no module, package or namespace folder at the old path, no
re-export and no alias. Each move adds its rows here:

- REMOVED matches, on one line, every spelling of an import, an attribute or a
  path that names a module or a name at its old home. It sweeps the code, the build
  files and the live docs; the records (docs/setup/ and
  docs/architecture/decisions/) keep their history, and this file names the old
  homes on purpose.
- _reads_a_name_that_left reads the syntax tree of every Python file, so an import
  in parentheses over several lines, a wildcard import, an alias of the old module
  and a monkeypatch of the name at the old module count, and a docstring does not.
- _imports_a_module_that_left reads the syntax tree for imports of a module that moved
  as a whole, so `from . import x`, `from .. import x` and an import in parentheses
  over several lines count, and a docstring does not.
- _binds_one_of checks that a module a name left does not define or re-export it.
- _old_homes finds a module, a package or a namespace folder at a removed path.

Each helper has a negative control on samples in tmp_path.
"""

import ast
import importlib.machinery
import re
import sys
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PACKAGE_DIR = SRC / "walkie_vision"

CODE_AND_BUILD = [
    "src",
    "tests",
    "checks",
    "tools",
    "launch",
    "scripts",
    "config",
    ".github",
    "pixi.toml",
    "pyproject.toml",
    "mkdocs.yml",
    "README.md",
    "CLAUDE.md",
    "docs",
]
# Records keep the paths they were written with.
RECORDS = ["docs/setup", "docs/architecture/decisions"]
# Output that git ignores and a run writes, which may name a module as it was then.
GENERATED = ["checks/output"]
TEXT_SUFFIXES = {".py", ".toml", ".sh", ".yml", ".yaml", ".md", ".json", ".txt", ".cfg"}

# One alternative per module or name that left its home, in any spelling on one
# line; the syntax-tree checks below catch every other shape.
REMOVED = re.compile(
    "|".join(
        [
            # StartupError left walkie_vision.node for walkie_vision.errors. `node.`
            # also catches a test's `from walkie_vision import node as node_module`.
            r"node(_module)?[.:]StartupError\b",
            r"from (walkie_vision\.|\.+)node import [^\n]*\bStartupError\b",
            # The model file table left walkie_vision.models for walkie_vision.weights.
            # The models/ folder, models_dir, WALKIE_VISION_MODELS and the fetch-models
            # task keep their names.
            r"walkie_vision[./]models\b",
            r"from walkie_vision import [^\n]*\bmodels\b",
            r"from \.+ import [^\n]*\bmodels\b",
            r"from \.+models import",
            # The box and rectangle helpers left walkie_vision.crop, and resize_mask
            # left walkie_vision.image, for walkie_vision.boxes; crop's private _area
            # became boxes.box_area. crop.py imports the ones it still uses from boxes.
            r"\bcrop(_module)?[.:](Rect|box_iou|box_cover|box_area|outer_rect|pad_box|pad_rect|intersect_rects"
            r"|subtract_rect|_area|_overlap_area)\b",
            r"from (walkie_vision\.|\.+)crop import [^\n]*\b(Rect|box_iou|box_cover|box_area|outer_rect|pad_box"
            r"|pad_rect|intersect_rects|subtract_rect|_area|_overlap_area)\b",
            r"\bimage(_module)?[.:]resize_mask\b",
            r"from (walkie_vision\.|\.+)image import [^\n]*\bresize_mask\b",
        ]
    )
)

NODE = "walkie_vision.node"
LEFT_NODE = frozenset({"StartupError"})
MODELS = "walkie_vision.models"
CROP = "walkie_vision.crop"
LEFT_CROP = frozenset(
    {
        "Rect",
        "box_iou",
        "box_cover",
        "box_area",
        "outer_rect",
        "pad_box",
        "pad_rect",
        "intersect_rects",
        "subtract_rect",
        "_area",
        "_overlap_area",
    }
)
# The names that left crop.py and that it no longer uses: it does not even import them.
UNUSED_IN_CROP = frozenset({"pad_box", "_area", "_overlap_area"})
IMAGE = "walkie_vision.image"
LEFT_IMAGE = frozenset({"resize_mask"})


def _files_under(*relative: str) -> list[Path]:
    """Every text file at or under these paths of ROOT, without the records, the
    generated output, packaging metadata and this file."""
    skipped = [ROOT / path for path in (*RECORDS, *GENERATED)]
    files: list[Path] = []
    for name in relative:
        path = ROOT / name
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files += sorted(f for f in path.rglob("*") if f.is_file() and f.suffix in TEXT_SUFFIXES)
    # An editable install writes ignored packaging metadata that can list deleted files.
    return [
        f
        for f in files
        if f != Path(__file__).resolve()
        and not any(f.is_relative_to(folder) for folder in skipped)
        and not any(part.endswith(".egg-info") for part in f.parts)
    ]


def _mentions(files: list[Path], pattern: re.Pattern[str]) -> list[str]:
    return [
        f"{file.relative_to(ROOT) if file.is_relative_to(ROOT) else file.name}:{number}: {line.strip()}"
        for file in files
        for number, line in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1)
        if pattern.search(line)
    ]


def _imported_module(node: ast.ImportFrom, file: Path) -> str | None:
    """The absolute module an import names, resolving a relative one by the file's package."""
    if node.level == 0:
        return node.module
    if not file.is_relative_to(SRC):
        return None
    package = file.relative_to(SRC).with_suffix("").parts[: -node.level]
    return ".".join([*package, node.module] if node.module else package)


def _reads_a_name_that_left(file: Path, module: str, names: frozenset[str]) -> list[str]:
    """Where `file` imports one of `names` from `module`, reads it as an attribute of that
    module, or names it as a string next to the module in a call such as
    `monkeypatch.setattr(module, "name", ...)`, under any name the file gives the module.
    A wildcard import from `module` always counts, since it can bind any of the names.

    It reads the syntax tree rather than lines, so an import in parentheses over several
    lines counts, and a docstring that names the old home does not.
    """
    tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
    parent, _, child = module.rpartition(".")
    module_names = {module}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            module_names |= {a.asname for a in node.names if a.name == module and a.asname}
        elif isinstance(node, ast.ImportFrom) and _imported_module(node, file) == parent:
            module_names |= {a.asname or a.name for a in node.names if a.name == child}
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and _imported_module(node, file) == module:
            found += [
                f"{file.name}:{node.lineno}: imports {a.name}"
                for a in node.names
                if a.name in names or a.name == "*"
            ]
        elif isinstance(node, ast.Attribute) and node.attr in names and ast.unparse(node.value) in module_names:
            found.append(f"{file.name}:{node.lineno}: reads {ast.unparse(node)}")
        elif (
            isinstance(node, ast.Call)
            and len(node.args) >= 2
            and ast.unparse(node.args[0]) in module_names
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in names
        ):
            found.append(f"{file.name}:{node.lineno}: names {node.args[1].value}")
    return sorted(found)


def _imports_a_module_that_left(file: Path, module: str) -> list[str]:
    """Where `file` imports `module` or a module inside it: `import module`,
    `from module import name`, and `from parent import child` in any spelling, absolute
    or relative, with an alias or in parentheses over several lines.

    It reads the syntax tree rather than lines, and resolves a relative import by the
    file's package, so `from . import models` counts in walkie_vision/node.py but not in
    walkie_vision/reid/identifier.py, and a docstring that names the old home does not.
    """
    tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
    parent, _, child = module.rpartition(".")

    def inside(name: str) -> bool:
        return name == module or name.startswith(f"{module}.")

    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [f"{file.name}:{node.lineno}: imports {a.name}" for a in node.names if inside(a.name)]
        elif isinstance(node, ast.ImportFrom):
            imported = _imported_module(node, file)
            if imported is not None and inside(imported):
                found.append(f"{file.name}:{node.lineno}: imports from {imported}")
            elif imported == parent:
                found += [f"{file.name}:{node.lineno}: imports {module}" for a in node.names if a.name == child]
    return sorted(found)


def _module_level_names(tree: ast.Module, *, imports: bool = True) -> Iterator[tuple[str, int]]:
    """Each name a module binds outside its functions and classes, with its line.

    Without `imports`, only the names it defines: an imported name is left out, unless
    the import has an `as` clause (`import x as y`, `from m import x as x`), which binds
    the name on purpose and is how a module spells a re-export. A wildcard import
    (`from m import *`) is always yielded, as the name `*`: it can bind any name `m`
    exports, and which ones is not known without importing `m`.
    """
    stack: list[ast.AST] = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            yield node.name, node.lineno
            continue
        if isinstance(node, ast.Lambda):
            continue
        if isinstance(node, ast.Import | ast.ImportFrom):
            for alias in node.names:
                if imports or alias.name == "*" or alias.asname is not None:
                    yield alias.asname or alias.name.partition(".")[0], node.lineno
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            yield node.id, node.lineno
        stack.extend(ast.iter_child_nodes(node))


def _binds_one_of(file: Path, names: frozenset[str], *, imports: bool = True) -> list[str]:
    """Where `file` binds one of `names` at module level, which in a module the names
    left would put them back at their old home: a re-export or a copy.

    Without `imports`, a plain import of one of them does not count, for a module that
    still uses the names but must not define them again; an import with an `as` clause
    still counts. A wildcard import always counts, since it can bind any of them.
    """
    tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
    return sorted(
        f"{file.name}:{line}: binds {name}"
        for name, line in _module_level_names(tree, imports=imports)
        if name in names or name == "*"
    )


def _holds_code(directory: Path) -> bool:
    """Whether a directory holds a file Python could import, at any depth.

    Files under __pycache__ do not count: the import system reads them only as the cache
    of a source file next to them, so a directory that git leaves behind holding nothing
    but an ignored __pycache__ imports as an empty namespace package and serves nothing.
    """
    suffixes = tuple(importlib.machinery.all_suffixes())
    return any(
        path.name.endswith(suffixes) and "__pycache__" not in path.relative_to(directory).parts
        for path in directory.rglob("*")
    )


def _old_homes(root: Path, dotted: str) -> list[Path]:
    """What exists at a module path under root: a module, a package or a namespace folder."""
    base = root.joinpath(*dotted.split("."))
    found = []
    if base.with_suffix(".py").exists():
        found.append(base.with_suffix(".py"))
    if base.is_dir() and _holds_code(base):
        found.append(base)
    return found


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_no_code_build_file_or_live_doc_names_a_name_at_the_home_it_left():
    assert _mentions(_files_under(*CODE_AND_BUILD), REMOVED) == []


def test_the_sweep_reads_the_code_the_build_files_and_the_live_docs():
    files = set(_files_under(*CODE_AND_BUILD))
    assert {
        PACKAGE_DIR / "node.py",
        ROOT / "tests" / "test_node.py",
        ROOT / "launch" / "vision.launch.py",
        ROOT / "pixi.toml",
        ROOT / "pyproject.toml",
        ROOT / "README.md",
        ROOT / "CLAUDE.md",
        ROOT / "docs" / "architecture" / "vision.md",
    } <= files
    assert Path(__file__).resolve() not in files
    assert not any(f.is_relative_to(ROOT / record) for f in files for record in RECORDS)


def test_the_sweep_would_catch_the_startup_error_at_its_old_home(tmp_path):
    bad = _write(
        tmp_path / "bad.py",
        "from walkie_vision.node import VisionConfig, StartupError\n"
        "from .node import StartupError\n"
        "from ..node import build_providers, StartupError\n"
        "with pytest.raises(node.StartupError):\n"
        '(node_module.StartupError("OSNet is not in the Hugging Face cache"), False),\n'
        "# walkie_vision.node.StartupError is a RuntimeError\n"
        "`walkie_vision/node:StartupError` explains itself\n",
    )
    assert len(_mentions([bad], REMOVED)) == 7
    # The new home, the names that stay in the node, a longer name and prose are fine.
    good = _write(
        tmp_path / "good.py",
        "from walkie_vision.errors import StartupError\n"
        "from .errors import StartupError\n"
        "from walkie_vision import embeddings, errors, node\n"
        "with pytest.raises(errors.StartupError):\n"
        "from walkie_vision.node import VisionConfig, build_providers\n"
        "from .node import StartupLogger\n"
        "node.StartupLogger, node.StartupErrors\n"
        "StartupError (a RuntimeError) when OSNet is not cached\n"
        "raise StartupError(message)\n",
    )
    assert _mentions([good], REMOVED) == []


def test_the_sweep_skips_the_records_the_generated_output_and_other_files(tmp_path, monkeypatch):
    line = "from walkie_vision.node import StartupError\n"
    read = [
        _write(tmp_path / "src" / "walkie_vision" / "loading.py", line),
        _write(tmp_path / "docs" / "guides" / "vision-node.md", line),
        _write(tmp_path / "docs" / "architecture" / "vision.md", line),
        _write(tmp_path / "launch" / "vision.launch.py", line),
        _write(tmp_path / ".github" / "workflows" / "test.yml", line),
        _write(tmp_path / "pixi.toml", line),
        _write(tmp_path / "README.md", line),
    ]
    for skipped in (
        "docs/setup/01-vision-environment.md",
        "docs/architecture/decisions/0017-package-layout.md",
        "checks/output/c11_providers.txt",
        "src/walkie_vision.egg-info/SOURCES.txt",
        "src/walkie_vision/model.onnx",
        "src/walkie_vision/__pycache__/node.cpython-312.pyc",
    ):
        _write(tmp_path / skipped, line)
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)

    files = _files_under(*CODE_AND_BUILD)

    assert sorted(files) == sorted(read)
    assert len(_mentions(files, REMOVED)) == len(read)


def test_no_code_takes_the_startup_error_from_the_node():
    code = [file for file in _files_under(*CODE_AND_BUILD) if file.suffix == ".py"]
    assert PACKAGE_DIR / "node.py" in code
    assert [hit for file in code for hit in _reads_a_name_that_left(file, NODE, LEFT_NODE)] == []


def test_the_node_does_not_define_the_startup_error_again():
    # node.py imports StartupError to raise and catch it, but does not define it.
    assert _binds_one_of(PACKAGE_DIR / "node.py", LEFT_NODE, imports=False) == []


def test_the_model_file_table_did_not_come_back_to_models():
    assert (PACKAGE_DIR / "weights.py").is_file()
    assert _old_homes(SRC, "walkie_vision.models") == []


def test_no_code_imports_the_model_file_table_from_models():
    code = [file for file in _files_under(*CODE_AND_BUILD) if file.suffix == ".py"]
    assert PACKAGE_DIR / "node.py" in code
    assert [hit for file in code for hit in _imports_a_module_that_left(file, MODELS)] == []


def test_no_code_takes_the_box_helpers_from_crop_or_the_mask_resize_from_image():
    code = [file for file in _files_under(*CODE_AND_BUILD) if file.suffix == ".py"]
    assert {PACKAGE_DIR / "crop.py", PACKAGE_DIR / "yolo.py", ROOT / "tests" / "test_crop.py"} <= set(code)
    assert [hit for file in code for hit in _reads_a_name_that_left(file, CROP, LEFT_CROP)] == []
    assert [hit for file in code for hit in _reads_a_name_that_left(file, IMAGE, LEFT_IMAGE)] == []


def test_crop_and_image_do_not_define_the_box_helpers_or_the_mask_resize_again():
    assert (PACKAGE_DIR / "boxes.py").is_file()
    # crop.py imports the helpers it still uses from boxes, but defines none of them,
    # and does not import the ones it no longer uses. image.py uses none.
    assert _binds_one_of(PACKAGE_DIR / "crop.py", LEFT_CROP, imports=False) == []
    assert _binds_one_of(PACKAGE_DIR / "crop.py", UNUSED_IN_CROP) == []
    assert _binds_one_of(PACKAGE_DIR / "image.py", LEFT_IMAGE) == []


def test_the_tree_check_would_catch_a_module_imported_from_its_old_home(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "SRC", tmp_path / "src")
    caller = _write(
        tmp_path / "src" / "walkie_vision" / "loading.py",
        "import walkie_vision.models\n"
        "import walkie_vision.models as table\n"
        "from walkie_vision import models\n"
        "from walkie_vision import (\n"
        "    embeddings,\n"
        "    models,\n"
        ")\n"
        "from . import models as table\n"
        "from walkie_vision.models import (\n"
        "    DOWNLOADS,\n"
        ")\n"
        "from .models import HUB_MODELS\n"
        "def build():\n"
        "    from . import models\n"
        "    return models\n",
    )
    assert _imports_a_module_that_left(caller, MODELS) == [
        "loading.py:12: imports from walkie_vision.models",
        "loading.py:14: imports walkie_vision.models",
        "loading.py:1: imports walkie_vision.models",
        "loading.py:2: imports walkie_vision.models",
        "loading.py:3: imports walkie_vision.models",
        "loading.py:4: imports walkie_vision.models",
        "loading.py:8: imports walkie_vision.models",
        "loading.py:9: imports from walkie_vision.models",
    ]
    nested = _write(
        tmp_path / "src" / "walkie_vision" / "reid" / "__init__.py",
        "from .. import (\n    models,\n)\nfrom ..models import OSNET_CHECKPOINT\n",
    )
    assert _imports_a_module_that_left(nested, MODELS) == [
        "__init__.py:1: imports walkie_vision.models",
        "__init__.py:4: imports from walkie_vision.models",
    ]
    # Outside src/, an absolute import over several lines still counts.
    check = _write(tmp_path / "checks" / "c05_clip.py", "from walkie_vision import (\n    models,\n)\n")
    assert _imports_a_module_that_left(check, MODELS) == ["c05_clip.py:1: imports walkie_vision.models"]
    # The new home, a module of the same name in a subpackage, a longer module name,
    # a name called models and a docstring that names the old home are fine.
    good = _write(
        tmp_path / "src" / "walkie_vision" / "reid" / "identifier.py",
        '"""walkie_vision.models became walkie_vision.weights."""\n'
        "import walkie_vision.weights\n"
        "import walkie_vision.models_cache\n"
        "from walkie_vision import (\n"
        "    embeddings,\n"
        "    weights,\n"
        ")\n"
        "from .. import weights\n"
        "from . import models\n"
        "from ..face import models\n"
        "from ..weights import models_dir\n"
        "from walkie_vision.models_cache import entries\n",
    )
    assert _imports_a_module_that_left(good, MODELS) == []


def test_the_sweep_would_catch_the_model_file_table_at_its_old_home(tmp_path):
    bad = _write(
        tmp_path / "bad.py",
        "from walkie_vision.models import ULTRALYTICS_DIR, models_dir\n"
        "import walkie_vision.models\n"
        "from walkie_vision import embeddings, models\n"
        "from . import convert, models\n"
        "from .models import CLIP_WEIGHTS, HUB_MODELS\n"
        "from ..models import HUB_MODELS, OSNET_CHECKPOINT\n"
        'cmd = "python -m walkie_vision.models"\n'
        "| `src/walkie_vision/models.py` | Every model file and Hugging Face commit |\n"
        "Every model is pinned to the commit in walkie_vision.models.HUB_MODELS.\n",
    )
    assert len(_mentions([bad], REMOVED)) == 9
    # The new home, the models/ folder, models_dir, the environment variable, the
    # fetch-models task and a longer module name are fine.
    good = _write(
        tmp_path / "good.py",
        "from walkie_vision.weights import ULTRALYTICS_DIR, models_dir\n"
        "from walkie_vision import embeddings, weights\n"
        "from . import convert, crop, requests\n"
        "from .. import models_dir\n"
        "from .weights import CLIP_WEIGHTS, HUB_MODELS\n"
        'cmd = "python -m walkie_vision.weights"\n'
        "[tasks.fetch-models]\n"
        "Model files are kept in models/ at the repository root.\n"
        'env["WALKIE_VISION_MODELS"] = str(models_dir())\n'
        'names = args.models.split(",")\n'
        "from walkie_vision.models_cache import entries\n",
    )
    assert _mentions([good], REMOVED) == []


def test_a_relative_import_resolves_by_the_file_package():
    def imported(source: str, file: Path) -> str | None:
        (node,) = ast.parse(source).body
        assert isinstance(node, ast.ImportFrom)
        return _imported_module(node, file)

    provider = PACKAGE_DIR / "providers" / "reid" / "osnet.py"
    init = PACKAGE_DIR / "face" / "__init__.py"
    assert imported("from walkie_vision.node import StartupError", provider) == "walkie_vision.node"
    assert imported("from ..vector_tables import EMBEDDING_DIMS", provider) == "walkie_vision.providers.vector_tables"
    assert imported("from . import base", provider) == "walkie_vision.providers.reid"
    assert imported("from ...errors import StartupError", provider) == "walkie_vision.errors"
    assert imported("from .analysis import FaceAnalyser", init) == "walkie_vision.face.analysis"
    assert imported("from .. import node", init) == "walkie_vision"
    # A relative import outside src/ belongs to no package of ours.
    assert imported("from .node import StartupError", ROOT / "tests" / "test_node.py") is None


def test_the_tree_check_would_catch_a_name_read_from_its_old_home(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "SRC", tmp_path / "src")
    caller = _write(
        tmp_path / "src" / "walkie_vision" / "loading.py",
        "import walkie_vision.node\n"
        "import walkie_vision.node as live\n"
        "from walkie_vision import node as node_module\n"
        "from walkie_vision.node import (\n"
        "    VisionConfig,\n"
        "    StartupError,\n"
        ")\n"
        "from .node import StartupError as Failed\n"
        "from . import node\n"
        "a = node_module.StartupError\n"
        "b = live.StartupError\n"
        "c = walkie_vision.node.StartupError\n"
        "def build():\n"
        "    from .node import StartupError\n"
        "    return node.StartupError\n"
        'monkeypatch.setattr(node_module, "StartupError", RuntimeError)\n'
        'mock.patch.object(live, "StartupError", RuntimeError)\n'
        "from walkie_vision.node import *\n"
        "from .node import *\n",
    )
    assert _reads_a_name_that_left(caller, NODE, LEFT_NODE) == [
        "loading.py:10: reads node_module.StartupError",
        "loading.py:11: reads live.StartupError",
        "loading.py:12: reads walkie_vision.node.StartupError",
        "loading.py:14: imports StartupError",
        "loading.py:15: reads node.StartupError",
        "loading.py:16: names StartupError",
        "loading.py:17: names StartupError",
        "loading.py:18: imports *",
        "loading.py:19: imports *",
        "loading.py:4: imports StartupError",
        "loading.py:8: imports StartupError",
    ]
    # The new home, the names that stay in the node, the name next to another module
    # and a wildcard import from another module are fine, and so is a docstring that
    # names the old home.
    good = _write(
        tmp_path / "src" / "walkie_vision" / "good.py",
        '"""node.StartupError moved to walkie_vision.errors."""\n'
        "from walkie_vision import errors, node\n"
        "from walkie_vision.errors import StartupError\n"
        "from .errors import StartupError as Failed\n"
        "from walkie_vision.node import (\n"
        "    VisionConfig,\n"
        "    build_providers,\n"
        ")\n"
        "a = errors.StartupError\n"
        "b = node.VisionConfig\n"
        "c = other.StartupError\n"
        'monkeypatch.setattr(errors, "StartupError", RuntimeError)\n'
        'monkeypatch.setattr(node, "build_providers", fake)\n'
        "from walkie_vision.errors import *\n"
        "from .errors import *\n",
    )
    assert _reads_a_name_that_left(good, NODE, LEFT_NODE) == []


def test_the_tree_check_would_catch_a_name_bound_again_at_its_old_home(tmp_path):
    copy = _write(
        tmp_path / "node.py",
        "from .errors import StartupError as StartupError\n"
        "from walkie_vision.errors import *\n"
        "class StartupError(RuntimeError): ...\n"
        "StartupError = RuntimeError\n"
        "try:\n"
        "    import cv2\n"
        "except ImportError:\n"
        "    StartupError: type = RuntimeError\n",
    )
    assert _binds_one_of(copy, LEFT_NODE, imports=False) == [
        "node.py:1: binds StartupError",
        "node.py:2: binds *",
        "node.py:3: binds StartupError",
        "node.py:4: binds StartupError",
        "node.py:8: binds StartupError",
    ]
    # A module that only uses the name may import it; a name inside a function or a
    # class, or in a docstring, binds nothing at module level.
    good = _write(
        tmp_path / "good.py",
        '"""main() logs a StartupError without a traceback."""\n'
        "from .errors import StartupError\n"
        "from walkie_vision import errors\n"
        "def build():\n"
        "    StartupError = None\n"
        "    from .errors import StartupError as Failed\n"
        "    return Failed\n"
        "class Node:\n"
        "    StartupError = RuntimeError\n"
        "check = lambda StartupError: StartupError\n",
    )
    assert _binds_one_of(good, LEFT_NODE, imports=False) == []
    # With imports, the plain import counts as well.
    assert _binds_one_of(good, LEFT_NODE) == ["good.py:2: binds StartupError"]


def test_the_old_home_check_would_catch_a_module_a_package_or_a_namespace_folder(tmp_path):
    shim = "from walkie_vision.weights import DOWNLOADS\n"
    bad = tmp_path / "bad" / "walkie_vision"
    module = _write(bad / "gone.py", shim)
    package = _write(bad / "gone_package" / "__init__.py", shim).parent
    # A folder without an __init__.py still imports, as a namespace package.
    namespace = _write(bad / "gone_namespace" / "inner" / "person.py", shim).parents[1]
    # A legacy compiled module straight in the folder still imports.
    compiled = _write(bad / "gone_compiled" / "gate.pyc", "").parent
    # A module and a package of one name both count.
    both = [_write(bad / "twice.py", shim), _write(bad / "twice" / "__init__.py", shim).parent]
    assert _old_homes(bad.parent, "walkie_vision.gone") == [module]
    assert _old_homes(bad.parent, "walkie_vision.gone_package") == [package]
    assert _old_homes(bad.parent, "walkie_vision.gone_namespace") == [namespace]
    assert _old_homes(bad.parent, "walkie_vision.gone_compiled") == [compiled]
    assert _old_homes(bad.parent, "walkie_vision.twice") == both
    # The new homes are fine, and so is what a checkout leaves at the old path: git
    # removes the tracked files but keeps the ignored __pycache__, which imports nothing.
    good = tmp_path / "good" / "walkie_vision"
    for path in (
        "weights.py",
        "crops/__init__.py",
        "crops/person.py",
        "gone/__pycache__/person.cpython-312.pyc",
        "gone/data/__pycache__/x.cpython-312.pyc",
        "gone/README.md",
    ):
        _write(good / path, shim)
    assert _old_homes(good.parent, "walkie_vision.gone") == []
    assert _old_homes(good.parent, "walkie_vision.gone_package") == []
    assert _old_homes(good.parent, "walkie_vision.crops") == [good / "crops"]


def test_the_sweep_would_catch_the_box_helpers_and_the_mask_resize_at_their_old_homes(tmp_path):
    bad = _write(
        tmp_path / "bad.py",
        "from walkie_vision.crop import COLOR_NAMES, Rect\n"
        "from .crop import CropStatus, pad_rect\n"
        "from ..crop import _overlap_area\n"
        "x1, y1, x2, y2 = crop.pad_rect(box, CROP_PADDING, width, height)\n"
        "if crop.intersect_rects(crop.outer_rect(box), (0, 0, width, height)) is None:\n"
        "assert crop.box_cover(behind.box, crop.torso_box(guest)) == pytest.approx(0.4)\n"
        "return box if crop._area(box) > 0 else None\n"
        "`walkie_vision.crop.box_iou` clamps at zero\n"
        "from walkie_vision.image import image_to_bgr, resize_mask\n"
        "from .image import resize_mask\n"
        "on_raster = image.resize_mask(mask, *depth.metres.shape)\n"
        "walkie_vision.image:resize_mask keeps the mask at an equal size\n",
    )
    assert len(_mentions([bad], REMOVED)) == 12
    # The new home, the names that stay in crop and image, longer names and prose are fine.
    good = _write(
        tmp_path / "good.py",
        "from walkie_vision.boxes import Rect, box_area, pad_rect, resize_mask\n"
        "from .boxes import box_cover, box_iou, intersect_rects, outer_rect, subtract_rect\n"
        "from walkie_vision import boxes, crop, image\n"
        "from walkie_vision.crop import COLOR_NAMES, MASK_GREY, CropStatus, crop_person\n"
        "from .image import calibration_problem, camera_info_problem\n"
        "x1, y1, x2, y2 = boxes.pad_rect(box, CROP_PADDING, width, height)\n"
        "assert boxes.box_cover(behind.box, crop.torso_box(guest)) == pytest.approx(0.4)\n"
        "reduced, _, _ = crop.scale_down(pixels, CROP_MAX_SIDE)\n"
        "person_crop.box_iou, crop.box_iou_floor, crop.Rects\n"
        "image.resize_masks(masks), image.shape[:2]\n"
        "Each mask is resized with resize_mask, and each box padded with pad_rect.\n",
    )
    assert _mentions([good], REMOVED) == []


def test_the_tree_check_would_catch_a_box_helper_or_the_mask_resize_at_its_old_home(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "SRC", tmp_path / "src")
    caller = _write(
        tmp_path / "src" / "walkie_vision" / "requests.py",
        "from walkie_vision import crop as person_crop\n"
        "from . import crop, image\n"
        "from .crop import (\n"
        "    CropStatus,\n"
        "    pad_box,\n"
        ")\n"
        "from .image import resize_mask as resize\n"
        "a = crop.intersect_rects(crop.outer_rect(box), (0, 0, width, height))\n"
        "b = person_crop.box_area(box)\n"
        'monkeypatch.setattr(image, "resize_mask", fake)\n',
    )
    found = _reads_a_name_that_left(caller, CROP, LEFT_CROP) + _reads_a_name_that_left(caller, IMAGE, LEFT_IMAGE)
    assert sorted(found) == [
        "requests.py:10: names resize_mask",
        "requests.py:3: imports pad_box",
        "requests.py:7: imports resize_mask",
        "requests.py:8: reads crop.intersect_rects",
        "requests.py:8: reads crop.outer_rect",
        "requests.py:9: reads person_crop.box_area",
    ]
    # A crop that imports a helper it no longer uses, or binds one under its old name,
    # puts it back at its old home.
    crop = _write(
        tmp_path / "crop.py",
        "from .boxes import Rect, box_area, intersect_rects\n"
        "from .boxes import box_area as _area\n"
        "from .boxes import pad_box\n"
        "def _overlap_area(a, b): ...\n",
    )
    assert _binds_one_of(crop, LEFT_CROP, imports=False) == [
        "crop.py:2: binds _area",
        "crop.py:4: binds _overlap_area",
    ]
    assert _binds_one_of(crop, UNUSED_IN_CROP) == [
        "crop.py:2: binds _area",
        "crop.py:3: binds pad_box",
        "crop.py:4: binds _overlap_area",
    ]
    image = _write(tmp_path / "image.py", "import numpy as np\nfrom .boxes import resize_mask\n")
    assert _binds_one_of(image, LEFT_IMAGE) == ["image.py:2: binds resize_mask"]
    # The new home and the names that stay are fine.
    good = _write(
        tmp_path / "src" / "walkie_vision" / "pipeline.py",
        "from . import boxes, crop, image\n"
        "from .boxes import intersect_rects, resize_mask\n"
        "from .crop import COLOR_NAMES, CropStatus\n"
        "from .image import image_to_bgr\n"
        "a = boxes.pad_box(box, 0.1)\n"
        "b = crop.scale_down(pixels, 512)\n"
        'monkeypatch.setattr(crop, "MIN_BOX_HEIGHT_PX", 10.0)\n',
    )
    assert _reads_a_name_that_left(good, CROP, LEFT_CROP) == []
    assert _reads_a_name_that_left(good, IMAGE, LEFT_IMAGE) == []
