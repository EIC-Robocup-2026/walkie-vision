"""Which walkie_vision modules load a model library when imported.

Importing a module must not load torch, Ultralytics, onnxruntime, OpenCLIP or
the other model libraries in HEAVY, unless HEAVY_AT_IMPORT names the module.
Two readers depend on this. The vision node reads the MobileCLIP weight file
name from the YOLOE module to check that the model files exist before it
imports torch or Ultralytics, and the fake publisher (`pixi run
fake-detections`) is the stand-in for the object stream that runs with no
camera, GPU or model, so it must start without loading a model library.

Each module is imported in its own fresh interpreter (`python -P -c`), so the
result does not depend on import order. In one shared process, a module
checked after the face provider would find onnxruntime already loaded and could
never be blamed for it. The modules are found from the files on disk without
importing anything: pkgutil.walk_packages imports each package to descend into
it, which would load onnxruntime and torch into the walking process.

HEAVY_AT_IMPORT is exact. A module outside it that loads a HEAVY package
fails, and so does an entry that loads none (a stale entry). When a module
moves, the entry moves with it.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pkgutil
import shutil
import subprocess
import sys
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACKAGE = "walkie_vision"

HEAVY = ("torch", "torchvision", "ultralytics", "onnxruntime", "open_clip", "transformers", "timm", "insightface")

# The buffalo_l face provider loads onnxruntime. The osnet reid provider and
# the vendored OSNet network load torch. The vendored InsightFace files load
# only cv2, numpy and onnx, so they are light.
HEAVY_AT_IMPORT = frozenset(
    {
        "walkie_vision._vendor.torchreid.osnet",
        "walkie_vision.providers.face_recognition.buffalo_l",
        "walkie_vision.providers.reid.osnet",
    }
)

# Run in a fresh interpreter: argv is the module, the package whose __file__
# to report, then the names to look for in sys.modules.
_PROBE = """
import importlib, json, sys
module, package, *heavy = sys.argv[1:]
importlib.import_module(module)
print(json.dumps({"loaded": [h for h in heavy if h in sys.modules], "file": sys.modules[package].__file__}))
"""

_WORKERS = 4
_TIMEOUT_S = 60


def walk_modules(package_dir: Path, package: str) -> list[str]:
    """Every module and package under package_dir, the package itself included, imported by none."""
    names = [package]
    for info in pkgutil.iter_modules([str(package_dir)]):
        name = f"{package}.{info.name}"
        if info.ispkg:
            names += walk_modules(package_dir / info.name, name)
        else:
            names.append(name)
    return sorted(names)


def modules_on_disk(package_dir: Path, package: str) -> list[str]:
    """The dotted names of every .py file under package_dir, from the paths alone."""
    names = []
    for path in package_dir.rglob("*.py"):
        parts = path.relative_to(package_dir).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        names.append(".".join((package, *parts)))
    return sorted(names)


def _probe(module: str, package: str, heavy: Sequence[str], env: dict[str, str], cwd: Path) -> dict:
    out = subprocess.run(
        [sys.executable, "-P", "-c", _PROBE, module, package, *heavy],
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_S,
        env=env,
        cwd=cwd,
    )
    if out.returncode != 0:
        return {"error": f"{module} failed to import: {out.stderr.strip().splitlines()[-1:]}"}
    return json.loads(out.stdout.strip().splitlines()[-1])


def light_import_problems(
    package_dir: Path,
    package: str,
    heavy: Sequence[str],
    heavy_at_import: Iterable[str],
    cwd: Path,
    extra_path: Sequence[Path] = (),
) -> list[str]:
    """Import every module under package_dir in its own interpreter and list what breaks the rule."""
    allowed = set(heavy_at_import)
    modules = walk_modules(package_dir, package)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([*map(str, extra_path), *filter(None, [env.get("PYTHONPATH")])])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        results = list(pool.map(lambda m: _probe(m, package, heavy, env, cwd), modules))
    problems = [f"{name} is in the allowlist but is not a module" for name in sorted(allowed - set(modules))]
    for module, result in zip(modules, results, strict=True):
        if "error" in result:
            problems.append(result["error"])
            continue
        if not Path(result["file"]).resolve().is_relative_to(package_dir.resolve()):
            problems.append(f"{module} imported {package} from {result['file']}, not from {package_dir}")
        if result["loaded"] and module not in allowed:
            problems.append(f"{module} loads {', '.join(result['loaded'])}")
        if not result["loaded"] and module in allowed:
            problems.append(f"{module} is in the allowlist but loads none of {', '.join(heavy)}")
    return problems


def _package_dir() -> Path:
    spec = importlib.util.find_spec(PACKAGE)
    assert spec is not None and spec.submodule_search_locations is not None
    locations = list(spec.submodule_search_locations)
    assert len(locations) == 1, locations
    return Path(locations[0])


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_the_package_is_imported_from_this_checkout():
    assert _package_dir().resolve().is_relative_to(REPO / "src")


def test_every_module_loads_no_model_library_unless_allowed(tmp_path):
    problems = light_import_problems(_package_dir(), PACKAGE, HEAVY, HEAVY_AT_IMPORT, cwd=tmp_path)
    assert not problems, "\n".join(problems)


def test_the_walk_finds_every_module_file():
    # pkgutil skips a folder without __init__.py, so a namespace folder makes
    # the two lists differ.
    package_dir = _package_dir()
    assert walk_modules(package_dir, PACKAGE) == modules_on_disk(package_dir, PACKAGE)


def test_a_module_that_imports_a_heavy_package_is_reported(tmp_path):
    site = tmp_path / "site"
    _write(site / "fake_heavy.py", "")
    _write(site / "lightpkg" / "__init__.py", '"""A package."""\n')
    _write(site / "lightpkg" / "clean.py", "import json\n")
    _write(site / "lightpkg" / "loader.py", "import fake_heavy\n")
    problems = light_import_problems(
        site / "lightpkg", "lightpkg", ("fake_heavy",), (), cwd=tmp_path, extra_path=[site]
    )
    assert problems == ["lightpkg.loader loads fake_heavy"]


def test_a_heavy_package_init_makes_its_light_submodule_heavy(tmp_path):
    site = tmp_path / "site"
    _write(site / "fake_heavy.py", "")
    _write(site / "heavypkg" / "__init__.py", "import fake_heavy\n")
    _write(site / "heavypkg" / "sub.py", "import json\n")
    problems = light_import_problems(
        site / "heavypkg", "heavypkg", ("fake_heavy",), ("heavypkg",), cwd=tmp_path, extra_path=[site]
    )
    assert problems == ["heavypkg.sub loads fake_heavy"]


def test_an_allowlisted_module_that_loads_nothing_is_a_stale_entry(tmp_path):
    site = tmp_path / "site"
    _write(site / "fake_heavy.py", "")
    _write(site / "stalepkg" / "__init__.py", '"""A package."""\n')
    _write(site / "stalepkg" / "mod.py", "import json\n")
    problems = light_import_problems(
        site / "stalepkg", "stalepkg", ("fake_heavy",), ("stalepkg.mod",), cwd=tmp_path, extra_path=[site]
    )
    assert problems == ["stalepkg.mod is in the allowlist but loads none of fake_heavy"]


def test_a_heavy_import_in_a_copy_of_the_tree_is_blamed_on_its_module(tmp_path):
    # onnxruntime is the package the face provider loads, so a walk in one
    # shared process that reached pipeline after the provider would already
    # have it loaded.
    # Appended, not prepended: a from __future__ import must stay first.
    copy = tmp_path / "src" / PACKAGE
    shutil.copytree(_package_dir(), copy, ignore=shutil.ignore_patterns("__pycache__"))
    pipeline = copy / "pipeline.py"
    pipeline.write_text(pipeline.read_text() + "\nimport onnxruntime\n")
    problems = light_import_problems(
        copy, PACKAGE, HEAVY, HEAVY_AT_IMPORT, cwd=tmp_path, extra_path=[tmp_path / "src"]
    )
    assert "walkie_vision.pipeline loads onnxruntime" in problems
