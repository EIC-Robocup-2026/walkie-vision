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
  A name the old module imports and calls by its bare name may still be patched there.
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
            # The person crop left walkie_vision.crop for the walkie_vision.crops package:
            # crops.person, with the colour names and bins in crops.colour and the JPEG
            # encoding in crops.jpeg. crops, crop_person and the word crop are fine.
            r"walkie_vision[./]crop\b",
            r"from walkie_vision import [^\n]*\bcrop\b",
            r"from \.+crop import",
            r"from \.+ import [^\n]*\bcrop\b",
            # The object sampler and the padded crops left walkie_vision.embeddings for
            # walkie_vision.crops.objects, and the Lab palette shares for
            # walkie_vision.crops.colour. The CLIP provider stays in embeddings.
            r"\bembeddings(_module)?[.:](SAMPLE_PERIOD|SAMPLE_MAX_OBJECTS|SAMPLE_MIN_SIDE_PX|_short_side"
            r"|select_samples|usable_samples|CROP_PADDING|CROP_MAX_SIDE|padded_crop|encode_crop|MIN_COLOR_PIXELS"
            r"|PALETTE_RGB|_PALETTE_LAB|nearest_palette_indices|mask_color_shares)\b",
            r"from (walkie_vision\.|\.+)embeddings import [^\n]*\b(SAMPLE_PERIOD|SAMPLE_MAX_OBJECTS"
            r"|SAMPLE_MIN_SIDE_PX|_short_side|select_samples|usable_samples|CROP_PADDING|CROP_MAX_SIDE|padded_crop"
            r"|encode_crop|MIN_COLOR_PIXELS|PALETTE_RGB|_PALETTE_LAB|nearest_palette_indices|mask_color_shares)\b",
            # walkie_vision.image and walkie_vision.depth left for the walkie_vision.camera
            # package (camera.image, camera.depth and camera.mailboxes) and
            # walkie_vision.geometry, and stamp_ns for walkie_vision.convert. NewestFrame left
            # walkie_vision.node and CurrentImage left walkie_vision.requests for
            # camera.mailboxes. The words image and depth are fine. The relative
            # spellings are left to the syntax-tree check, which resolves them by the
            # file's package: inside camera/, `from .image import` is the new home.
            r"walkie_vision[./](image|depth)\b",
            r"from walkie_vision import [^\n]*\b(image|depth)\b",
            r"\bnode(_module)?[.:]NewestFrame\b",
            r"from (walkie_vision\.|\.+)node import [^\n]*\bNewestFrame\b",
            r"\brequests(_module)?[.:]CurrentImage\b",
            r"from (walkie_vision\.|\.+)requests import [^\n]*\bCurrentImage\b",
            # providers.py became the walkie_vision.providers package: one base module per
            # vision task and vector_tables, re-exported by the package, so
            # walkie_vision.providers.Detection is fine. EMBEDDING_DIMS left
            # walkie_vision.convert for providers.vector_tables, and check_model_dim for
            # providers.image_embed.base.
            r"\bproviders\.py\b",
            r"\bconvert(_module)?[.:](EMBEDDING_DIMS|check_model_dim)\b",
            r"from (walkie_vision\.|\.+)convert import [^\n]*\b(EMBEDDING_DIMS|check_model_dim)\b",
            # walkie_vision.requests became the walkie_vision.services package: the job,
            # the slot and the codes in services.jobs, and each service's checks and
            # responses in its own module, with the response builders and the message
            # bound that left walkie_vision.convert. convert's _float32_array stays, as
            # float32_array. The word requests is fine.
            r"walkie_vision[./]requests\b",
            r"from walkie_vision import [^\n]*\brequests\b",
            r"from \.+requests import",
            r"from \.+ import [^\n]*\brequests\b",
            r"\brequests\.py\b",
            r"\btest_requests\.py\b",
            r"\bconvert(_module)?[.:](fill_embed_text_response|embed_text_failure|vectors_from_response"
            r"|detect_objects_failure|embed_crop_failure|fill_embed_crop_response|bounded_message|MAX_MESSAGE_CHARS"
            r"|_float32_array)\b",
            r"from (walkie_vision\.|\.+)convert import [^\n]*\b(fill_embed_text_response|embed_text_failure"
            r"|vectors_from_response|detect_objects_failure|embed_crop_failure|fill_embed_crop_response"
            r"|bounded_message|MAX_MESSAGE_CHARS|_float32_array)\b",
            # GetPersonCrop's request check, job, mailbox and timeout cap left
            # walkie_vision.node, and its response builder left walkie_vision.convert, for
            # walkie_vision.services.get_person_crop. node.py imports the ones it uses.
            r"\bnode(_module)?[.:](CropJob|CropMailbox|_crop_answer|crop_request_problem|MAX_CROP_TIMEOUT_S)\b",
            r"from (walkie_vision\.|\.+)node import [^\n]*\b(CropJob|CropMailbox|_crop_answer|crop_request_problem"
            r"|MAX_CROP_TIMEOUT_S)\b",
            r"\bconvert(_module)?[.:]fill_person_crop_response\b",
            r"from (walkie_vision\.|\.+)convert import [^\n]*\bfill_person_crop_response\b",
            # walkie_vision.convert became walkie_vision.messages, and the QoS profiles left
            # walkie_vision.node for it. The word convert is fine: PIL's Image.convert and
            # the convert stage of the latency log keep it.
            r"walkie_vision[./]convert\b",
            r"from walkie_vision import [^\n]*\bconvert\b",
            r"from \.+convert import",
            r"from \.+ import [^\n]*\bconvert\b",
            r"\bconvert\.py\b",
            r"\btest_convert\.py\b",
            r"\bnode(_module)?[.:](IMAGE_QOS|CAMERA_INFO_QOS|OUTPUT_QOS)\b",
            r"from (walkie_vision\.|\.+)node import [^\n]*\b(IMAGE_QOS|CAMERA_INFO_QOS|OUTPUT_QOS)\b",
            # The vendored InsightFace files left walkie_vision.face for
            # walkie_vision._vendor.insightface, and the vendored OSNet network left
            # walkie_vision.reid for walkie_vision._vendor.torchreid; face and reid wrap
            # them. The relative spellings are left to the syntax-tree check, which
            # resolves them by the file's package: inside _vendor/insightface/,
            # `from . import face_align` is the new home. insightface/scrfd.py,
            # torchreid/osnet.py and the provider providers/reid/osnet.py are fine.
            r"walkie_vision[./]face[./](scrfd|arcface|face_align)\b",
            r"from walkie_vision\.face import [^\n]*\b(scrfd|arcface|face_align)\b",
            r"\bface/(scrfd|arcface|face_align)\.py\b",
            r"walkie_vision[./]reid[./]osnet\b",
            r"from walkie_vision\.reid import [^\n]*\bosnet\b",
            r"(?<!providers/)\breid/osnet\.py\b",
            r"\btest_face\.py\b",
            # walkie_vision.yolo became providers.object_detection.yoloe, with the helpers
            # both Ultralytics providers use in providers._ultralytics and the pose in
            # providers.pose_estimation.yolo_pose. yoloe, yolo_pose, yolo26 and the
            # weight file names are fine.
            r"walkie_vision[./]yolo\b",
            r"from walkie_vision import [^\n]*\byolo\b",
            r"from \.+yolo import",
            r"from \.+ import [^\n]*\byolo\b",
            r"\byolo\.py\b",
            r"\btest_yolo\.py\b",
            # The walkie_vision.face package became providers.face_recognition.buffalo_l:
            # face/heads.py with the analyser from face/analysis.py. The two buffalo_l file
            # names left the analyser for walkie_vision.weights. face_recognition,
            # face_align, insightface/ and the word face are fine.
            r"walkie_vision[./]face\b",
            r"from walkie_vision import [^\n]*\bface\b",
            r"from \.+face(\.\w+)? import",
            r"from \.+ import [^\n]*\bface\b",
            r"(?<![\w/.-])face/",
            r"\btest_heads\.py\b",
            r"\b(analysis|buffalo_l)(_module)?[.:](DETECTOR_FILE|RECOGNIZER_FILE)\b",
            r"from (walkie_vision\.|\.+)([\w.]+\.)?buffalo_l import [^\n]*\b(DETECTOR_FILE|RECOGNIZER_FILE)\b",
            # The walkie_vision.reid package became providers.reid.osnet: reid/identifier.py
            # with the loader and the input constants from reid/__init__.py. The reid vision
            # task (providers.reid, providers/reid/, and its folder named alone as reid/),
            # torchreid/, deep-person-reid and the reid stage are fine, so only the two old
            # files are rows by bare path. The relative spellings are left to the syntax-tree
            # check, which resolves them by the file's package: inside providers/,
            # `from .reid.base import` is the new home.
            r"walkie_vision[./]reid\b",
            r"from walkie_vision import [^\n]*\breid\b",
            r"(?<![\w/.-])reid/(identifier|__init__)\.py\b",
            r"\btest_reid_identifier\.py\b",
            r"\b(reid|identifier)(_module)?[.:](identifier|load_osnet|OSNetReIdentifier|osnet_input|INPUT_SIZE|MEAN"
            r"|STD)\b",
            # walkie_vision.embeddings, which held only the CLIP provider after the
            # sampler and the palette left it, became providers.image_embed.clip. No new
            # module is called embeddings, so its relative spellings are rows too. The
            # word embeddings, object_embeddings_enabled, the image_embed vision task and
            # the embed services are fine.
            r"walkie_vision[./]embeddings\b",
            r"from walkie_vision import [^\n]*\bembeddings\b",
            r"from \.+embeddings import",
            r"from \.+ import [^\n]*\bembeddings\b",
            r"\bembeddings\.py\b",
            r"\btest_embeddings\.py\b",
            r"\bembeddings(_module)?[.:](CLIP_ARCHITECTURE|CLIP_DIM|TEXT_TEMPLATE|clip_checkpoint|_build_open_clip"
            r"|ClipEmbedder)\b",
            # VisionConfig and the constants only it checks left walkie_vision.node for
            # walkie_vision.config, and build_providers with its logger protocol and its
            # helpers for walkie_vision.loading. node.py imports VisionConfig and
            # build_providers, so a test may patch build_providers on the node, which
            # looks it up there, by the module or by its dotted path in quotes; reading
            # or importing either from the node is a row (the tree check reads the
            # attribute form of build_providers in code).
            r"\bnode(_module)?[.:](VisionConfig|MAX_PAIR_WINDOW_S|TRACKER_FILES|_DEVICE"
            r"|StartupLogger|_load|_FACE_FILES|_T)\b",
            r"(?<![\w\"'])walkie_vision[./]node[.:]build_providers\b",
            r"from (walkie_vision\.|\.+)node import [^\n]*\b(VisionConfig|MAX_PAIR_WINDOW_S|TRACKER_FILES|_DEVICE"
            r"|build_providers|StartupLogger|_load|_FACE_FILES|_T)\b",
            r"\btest_build_providers\.py\b",
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
# The names that left crop.py and that its successor, crops/person.py, no longer
# uses: it does not even import them.
UNUSED_IN_CROP = frozenset({"pad_box", "_area", "_overlap_area"})
IMAGE = "walkie_vision.image"
LEFT_IMAGE = frozenset({"resize_mask"})
# crop.py became crops/person.py; the colour names and bins left it for
# crops/colour.py, and the JPEG encoding for crops/jpeg.py.
PERSON_CROP = "walkie_vision.crops.person"
LEFT_PERSON_CROP = frozenset(
    {
        "COLOR_NAMES",
        "_BLACK_MAX_V",
        "_GREY_MAX_S",
        "_WHITE_MIN_V",
        "_PALE_MAX_S",
        "_PALE_MIN_V",
        "_BROWN_MAX_V",
        "_DARK_PINK_MAX_V",
        "color_indices",
        "color_shares",
        "JPEG_QUALITY",
        "scale_down",
        "encode_jpeg",
    }
)
# The ones crops/person.py no longer uses: it does not even import them.
UNUSED_IN_PERSON_CROP = LEFT_PERSON_CROP - {"color_shares", "scale_down", "encode_jpeg"}
# The object sampler and the padded crops left embeddings.py for crops/objects.py,
# and the Lab palette shares for crops/colour.py. The CLIP provider that embeddings.py
# kept, providers/image_embed/clip.py now, uses none of them, so it does not even
# import them.
EMBEDDINGS = "walkie_vision.embeddings"
LEFT_EMBEDDINGS = frozenset(
    {
        "SAMPLE_PERIOD",
        "SAMPLE_MAX_OBJECTS",
        "SAMPLE_MIN_SIDE_PX",
        "_short_side",
        "select_samples",
        "usable_samples",
        "CROP_PADDING",
        "CROP_MAX_SIDE",
        "padded_crop",
        "encode_crop",
        "MIN_COLOR_PIXELS",
        "PALETTE_RGB",
        "_PALETTE_LAB",
        "nearest_palette_indices",
        "mask_color_shares",
    }
)
# image.py became camera/image.py. depth.py became camera/depth.py, which keeps the
# raster reading and its checks: the depth maths left it for geometry.py, the mailboxes
# for camera/mailboxes.py and stamp_ns for convert.py. camera/depth.py uses none of
# them, so it does not even import them.
CAMERA_IMAGE = "walkie_vision.camera.image"
DEPTH = "walkie_vision.depth"
CAMERA_DEPTH = "walkie_vision.camera.depth"
LEFT_DEPTH = frozenset(
    {
        "_EROSION_SHARE",
        "_MIN_EROSION_PX",
        "_MAX_EROSION_PX",
        "_MAD_TO_SIGMA",
        "_MAD_WINDOW_SIGMAS",
        "_MIN_DEPTH_WINDOW_M",
        "_FULL_AREA_SHARE",
        "_FULL_SHARPNESS_VARIANCE",
        "_MIN_SHARPNESS_PIXELS",
        "erosion_radius",
        "_erode",
        "_in_range",
        "_surviving_pixels",
        "lift_mask",
        "depth_cover",
        "frame_laplacian",
        "view_quality",
        "Bundle",
        "_WaitingColour",
        "RgbdBuffer",
        "stamp_ns",
    }
)
# The node's mailbox and the services' newest input left for camera/mailboxes.py.
# node.py imports NewestFrame to build its mailbox; requests.py uses CurrentImage no more.
LEFT_NODE_MAILBOX = frozenset({"NewestFrame"})
REQUESTS = "walkie_vision.requests"
LEFT_REQUESTS = frozenset({"CurrentImage"})
# The vector lengths left convert.py for providers/vector_tables.py, and the length
# check for providers/image_embed/base.py. convert.py imports EMBEDDING_DIMS, which it
# still uses; the length check's one caller left it for services/embed_crop.py.
CONVERT = "walkie_vision.convert"
LEFT_CONVERT = frozenset({"EMBEDDING_DIMS", "check_model_dim"})
# requests.py became the services package. The response builders and the message bound
# left convert.py for it, and _float32_array became convert's public float32_array.
# convert.py uses none of them, so it does not even import them.
LEFT_CONVERT_SERVICES = frozenset(
    {
        "fill_embed_text_response",
        "embed_text_failure",
        "vectors_from_response",
        "detect_objects_failure",
        "embed_crop_failure",
        "fill_embed_crop_response",
        "bounded_message",
        "MAX_MESSAGE_CHARS",
        "_float32_array",
    }
)
# GetPersonCrop's request check, job, mailbox and timeout cap left node.py, and its
# response builder left convert.py, for services/get_person_crop.py. node.py imports
# the ones it uses but MAX_CROP_TIMEOUT_S, and convert.py uses none of them.
LEFT_NODE_PERSON_CROP = frozenset(
    {"MAX_CROP_TIMEOUT_S", "_crop_answer", "CropJob", "CropMailbox", "crop_request_problem"}
)
LEFT_CONVERT_PERSON_CROP = frozenset({"fill_person_crop_response"})
# convert.py became messages.py, which serves none of the names that left convert.py
# before it. The QoS profiles left node.py for messages.py; node.py imports them, since
# it subscribes and publishes with them, but defines none.
MESSAGES = "walkie_vision.messages"
LEFT_NODE_QOS = frozenset({"IMAGE_QOS", "CAMERA_INFO_QOS", "OUTPUT_QOS"})
# The vendored files left face/ and reid/ for _vendor/insightface/ and
# _vendor/torchreid/. face and reid import them from there, and neither package's
# __init__ binds them under their old names.
FACE = "walkie_vision.face"
LEFT_FACE = frozenset({"scrfd", "arcface", "face_align"})
REID = "walkie_vision.reid"
LEFT_REID = frozenset({"osnet"})
VENDORED = {
    **{f"{FACE}.{name}": f"_vendor/insightface/{name}.py" for name in sorted(LEFT_FACE)},
    f"{REID}.osnet": "_vendor/torchreid/osnet.py",
}
# yolo.py became providers/object_detection/yoloe.py. The helpers both Ultralytics
# providers use left it for providers/_ultralytics.py, the private ones made public,
# and the pose for providers/pose_estimation/yolo_pose.py. yoloe.py imports the
# helpers it uses from _ultralytics but defines none of them.
YOLO = "walkie_vision.yolo"
YOLOE = "walkie_vision.providers.object_detection.yoloe"
YOLO_POSE = "walkie_vision.providers.pose_estimation.yolo_pose"
ULTRALYTICS = "walkie_vision.providers._ultralytics"
SHARED_HELPERS = frozenset(
    {"TRACKER_NAMES", "validate_tracker", "clip_box", "check_rows", "require", "reset_trackers"}
)
# The names the helpers had in yolo.py: no module defines or serves them any more.
PRIVATE_HELPERS = frozenset({"_clip_box", "_check_rows", "_require", "_reset_trackers"})
LEFT_YOLOE = SHARED_HELPERS | PRIVATE_HELPERS | {"poses_from_arrays", "YoloPoseTracker"}
# The ones yoloe.py no longer uses: it does not even import them.
UNUSED_IN_YOLOE = LEFT_YOLOE - {"validate_tracker", "clip_box", "check_rows", "require", "reset_trackers"}
# The face package became providers/face_recognition/buffalo_l.py: face/heads.py with
# the analyser from face/analysis.py, and nothing re-exports it. The two buffalo_l file
# names left the analyser for weights.py; buffalo_l.py imports them from there but
# defines neither.
BUFFALO_L = "walkie_vision.providers.face_recognition.buffalo_l"
BUFFALO_L_FILE = PACKAGE_DIR / "providers" / "face_recognition" / "buffalo_l.py"
LEFT_BUFFALO_L = frozenset({"DETECTOR_FILE", "RECOGNIZER_FILE"})
# The reid package became providers/reid/osnet.py: reid/identifier.py with the loader
# and the input constants from reid/__init__.py, and nothing re-exports it.
OSNET_FILE = PACKAGE_DIR / "providers" / "reid" / "osnet.py"
LEFT_REID_PACKAGE = frozenset({"INPUT_SIZE", "MEAN", "STD", "load_osnet", "osnet_input", "OSNetReIdentifier"})
# embeddings.py became providers/image_embed/clip.py whole, and nothing re-exports
# the CLIP provider's names.
CLIP = "walkie_vision.providers.image_embed.clip"
CLIP_FILE = PACKAGE_DIR / "providers" / "image_embed" / "clip.py"
LEFT_EMBEDDINGS_CLIP = frozenset(
    {"CLIP_ARCHITECTURE", "CLIP_DIM", "TEXT_TEMPLATE", "clip_checkpoint", "_build_open_clip", "ClipEmbedder"}
)
# The node's parameters left node.py for config.py: VisionConfig with the constants
# only it checks. The model loading left it for loading.py: build_providers with its
# logger protocol, its helper, the helper's type variable and the face file names.
# node.py imports VisionConfig and build_providers, which it uses, and defines
# neither; it uses none of the others, so it does not even import them. The node
# looks build_providers up in its own namespace when it starts, so a test patches it
# there.
CONFIG_FILE = PACKAGE_DIR / "config.py"
LOADING_FILE = PACKAGE_DIR / "loading.py"
LEFT_NODE_CONFIG = frozenset({"MAX_PAIR_WINDOW_S", "TRACKER_FILES", "_DEVICE", "VisionConfig"})
LEFT_NODE_LOADING = frozenset({"_T", "_FACE_FILES", "StartupLogger", "_load", "build_providers"})
USED_BY_NODE = frozenset({"VisionConfig", "build_providers"})
PATCHED_ON_NODE = frozenset({"build_providers"})


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


def _reads_a_name_that_left(
    file: Path, module: str, names: frozenset[str], *, patched: frozenset[str] = frozenset()
) -> list[str]:
    """Where `file` imports one of `names` from `module`, reads it as an attribute of that
    module, or names it as a string next to the module in a call such as
    `monkeypatch.setattr(module, "name", ...)`, under any name the file gives the module.
    A wildcard import from `module` always counts, since it can bind any of the names.
    A name bound to `pytest.importorskip("module")` or `importlib.import_module("module")`
    is a name for the module too.

    A name in `patched` may still be named as a string next to `module`: `module` imports
    it and looks it up in its own namespace when it runs, so that is where a test patches
    it. Importing or reading it from `module` still counts.

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
        elif (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func).rpartition(".")[2] in {"importorskip", "import_module"}
            and node.value.args
            and isinstance(node.value.args[0], ast.Constant)
            and node.value.args[0].value == module
        ):
            module_names |= {ast.unparse(target) for target in node.targets if isinstance(target, ast.Name)}
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
            and node.args[1].value not in patched
        ):
            found.append(f"{file.name}:{node.lineno}: names {node.args[1].value}")
    return sorted(found)


def _imports_a_module_that_left(file: Path, module: str) -> list[str]:
    """Where `file` imports `module` or a module inside it: `import module`,
    `from module import name`, and `from parent import child` in any spelling, absolute
    or relative, with an alias or in parentheses over several lines.

    It reads the syntax tree rather than lines, and resolves a relative import by the
    file's package, so `from . import models` counts in walkie_vision/node.py but not in
    walkie_vision/providers/reid/osnet.py, and a docstring that names the old home does not.
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
        "from walkie_vision import errors, node\n"
        "with pytest.raises(errors.StartupError):\n"
        "from walkie_vision.node import NODE_NAME, VisionNode\n"
        "from .node import LatencyLog\n"
        "node.LatencyLog, node.StartupErrors\n"
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
    person, person_tests = PACKAGE_DIR / "crops" / "person.py", ROOT / "tests" / "test_crops_person.py"
    yoloe = PACKAGE_DIR / "providers" / "object_detection" / "yoloe.py"
    assert {person, yoloe, person_tests} <= set(code)
    # crop.py is crops/person.py now: neither the old module nor its successor serves them.
    assert [hit for file in code for hit in _reads_a_name_that_left(file, CROP, LEFT_CROP)] == []
    assert [hit for file in code for hit in _reads_a_name_that_left(file, PERSON_CROP, LEFT_CROP)] == []
    # image.py is camera/image.py now: neither the old module nor its successor serves it.
    assert [hit for file in code for hit in _reads_a_name_that_left(file, IMAGE, LEFT_IMAGE)] == []
    assert [hit for file in code for hit in _reads_a_name_that_left(file, CAMERA_IMAGE, LEFT_IMAGE)] == []


def test_crop_and_image_do_not_define_the_box_helpers_or_the_mask_resize_again():
    assert (PACKAGE_DIR / "boxes.py").is_file()
    # crops/person.py (crop.py until the crops package) imports the helpers it still
    # uses from boxes, but defines none of them, and does not import the ones it no
    # longer uses. camera/image.py (image.py until the camera package) uses none.
    assert _binds_one_of(PACKAGE_DIR / "crops" / "person.py", LEFT_CROP, imports=False) == []
    assert _binds_one_of(PACKAGE_DIR / "crops" / "person.py", UNUSED_IN_CROP) == []
    assert _binds_one_of(PACKAGE_DIR / "camera" / "image.py", LEFT_IMAGE) == []


def test_the_person_crop_did_not_come_back_to_crop():
    for module in ("__init__", "person", "colour", "jpeg"):
        assert (PACKAGE_DIR / "crops" / f"{module}.py").is_file()
    assert _old_homes(SRC, CROP) == []


def test_no_code_imports_the_person_crop_from_crop():
    code = [file for file in _files_under(*CODE_AND_BUILD) if file.suffix == ".py"]
    assert {PACKAGE_DIR / "node.py", PACKAGE_DIR / "messages.py", ROOT / "tests" / "test_node.py"} <= set(code)
    assert [hit for file in code for hit in _imports_a_module_that_left(file, CROP)] == []


def test_no_code_takes_the_colour_or_jpeg_names_from_the_person_crop():
    code = [file for file in _files_under(*CODE_AND_BUILD) if file.suffix == ".py"]
    assert {PACKAGE_DIR / "node.py", PACKAGE_DIR / "fake_publisher.py", ROOT / "tests" / "test_node.py"} <= set(code)
    assert [hit for file in code for hit in _reads_a_name_that_left(file, PERSON_CROP, LEFT_PERSON_CROP)] == []


def test_the_person_crop_does_not_define_the_colour_or_jpeg_names_again():
    # crops/person.py imports color_shares, scale_down and encode_jpeg, which it
    # uses, but defines none of the names that left, and imports none of the others:
    # COLOR_NAMES stays one literal assignment, in crops/colour.py.
    person = PACKAGE_DIR / "crops" / "person.py"
    assert _binds_one_of(person, LEFT_PERSON_CROP, imports=False) == []
    assert _binds_one_of(person, UNUSED_IN_PERSON_CROP) == []


def test_no_code_takes_the_sampler_or_the_palette_from_embeddings():
    code = [file for file in _files_under(*CODE_AND_BUILD) if file.suffix == ".py"]
    assert {
        PACKAGE_DIR / "pipeline.py",
        PACKAGE_DIR / "services" / "embed_crop.py",
        PACKAGE_DIR / "fake_publisher.py",
        ROOT / "tests" / "test_pipeline.py",
        ROOT / "tests" / "test_node.py",
    } <= set(code)
    # embeddings.py is providers/image_embed/clip.py now: neither the old module nor its
    # successor serves them.
    assert [hit for file in code for hit in _reads_a_name_that_left(file, EMBEDDINGS, LEFT_EMBEDDINGS)] == []
    assert [hit for file in code for hit in _reads_a_name_that_left(file, CLIP, LEFT_EMBEDDINGS)] == []


def test_embeddings_does_not_define_or_import_the_sampler_or_the_palette():
    for module in ("objects", "colour"):
        assert (PACKAGE_DIR / "crops" / f"{module}.py").is_file()
    # embeddings.py is providers/image_embed/clip.py now. The CLIP code uses no box or
    # crop helper, so clip.py imports neither.
    assert _binds_one_of(CLIP_FILE, LEFT_EMBEDDINGS) == []
    assert _imports_a_module_that_left(CLIP_FILE, "walkie_vision.crops") == []
    assert _imports_a_module_that_left(CLIP_FILE, "walkie_vision.boxes") == []
    # Nor do the crops import the CLIP code back: an import from the crops down to the
    # providers is allowed, so no contract checks this direction.
    crops = sorted((PACKAGE_DIR / "crops").glob("*.py"))
    assert PACKAGE_DIR / "crops" / "objects.py" in crops
    for module in (EMBEDDINGS, CLIP):
        assert [hit for file in crops for hit in _imports_a_module_that_left(file, module)] == []


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
        "    messages,\n"
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
        "from walkie_vision import messages, weights\n"
        "from . import crops, messages, services\n"
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
        "    LatencyLog,\n"
        "    VisionNode,\n"
        ")\n"
        "a = errors.StartupError\n"
        "b = node.VisionNode\n"
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
    # The new home, the names that stay with the person crop and in image (at its own
    # new home, camera.image), longer names and prose are fine.
    good = _write(
        tmp_path / "good.py",
        "from walkie_vision.boxes import Rect, box_area, pad_rect, resize_mask\n"
        "from .boxes import box_cover, box_iou, intersect_rects, outer_rect, subtract_rect\n"
        "from walkie_vision import boxes, crops\n"
        "from walkie_vision.camera import image\n"
        "from walkie_vision.crops.person import MASK_GREY, CropStatus, crop_person\n"
        "from .camera.image import calibration_problem, camera_info_problem\n"
        "x1, y1, x2, y2 = boxes.pad_rect(box, CROP_PADDING, width, height)\n"
        "assert boxes.box_cover(behind.box, person.torso_box(guest)) == pytest.approx(0.4)\n"
        "reduced, _, _ = jpeg.scale_down(pixels, CROP_MAX_SIDE)\n"
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


def test_the_sweep_would_catch_the_person_crop_at_its_old_home(tmp_path):
    bad = _write(
        tmp_path / "bad.py",
        "from walkie_vision.crop import COLOR_NAMES, crop_person\n"
        "import walkie_vision.crop\n"
        "from walkie_vision import boxes, crop\n"
        "from walkie_vision import crop as person_crop\n"
        "from . import convert, crop, requests\n"
        "from .crop import CropStatus\n"
        "from ..crop import encode_jpeg\n"
        "| `src/walkie_vision/crop.py` | One person's crop, mask and colour shares |\n"
        "The pure code is in `walkie_vision.crop`, with no ROS and no model.\n",
    )
    assert len(_mentions([bad], REMOVED)) == 9
    # The new package and its modules, crop_person, the crop fields of the messages
    # and the word crop are fine.
    good = _write(
        tmp_path / "good.py",
        "from walkie_vision.crops.person import CropStatus, crop_person\n"
        "from walkie_vision.crops import colour, person\n"
        "from walkie_vision import boxes, crops\n"
        "from . import crops, messages, services\n"
        "from .crops.jpeg import encode_jpeg\n"
        "from ..crops import colour\n"
        "| `src/walkie_vision/crops/` | One person's crop, mask and colour shares |\n"
        "found = person.crop_person(image, people, track_id)\n"
        "assert det.crop.format == 'jpeg' and len(response.crop.data) > 0\n"
        "A crop of each sampled object, from the crop service.\n",
    )
    assert _mentions([good], REMOVED) == []


def test_the_tree_check_would_catch_the_person_crop_or_its_names_at_their_old_homes(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "SRC", tmp_path / "src")
    # The module that left: crop, in any spelling of an import.
    caller = _write(
        tmp_path / "src" / "walkie_vision" / "services" / "get_person_crop.py",
        "from .. import (\n"
        "    convert,\n"
        "    crop,\n"
        ")\n"
        "from ..crop import CropStatus\n"
        "import walkie_vision.crop as person_crop\n",
    )
    assert _imports_a_module_that_left(caller, CROP) == [
        "get_person_crop.py:1: imports walkie_vision.crop",
        "get_person_crop.py:5: imports from walkie_vision.crop",
        "get_person_crop.py:6: imports walkie_vision.crop",
    ]
    # The colour and JPEG names, taken from the person crop under any name.
    reader = _write(
        tmp_path / "src" / "walkie_vision" / "convert.py",
        "from .crops import person\n"
        "from walkie_vision.crops import person as person_crop\n"
        "from .crops.person import (\n"
        "    COLOR_NAMES,\n"
        "    CropStatus,\n"
        ")\n"
        "a = person.encode_jpeg(pixels, person_crop.JPEG_QUALITY)\n"
        'monkeypatch.setattr(person, "color_shares", fake)\n',
    )
    assert _reads_a_name_that_left(reader, PERSON_CROP, LEFT_PERSON_CROP) == [
        "convert.py:3: imports COLOR_NAMES",
        "convert.py:7: reads person.encode_jpeg",
        "convert.py:7: reads person_crop.JPEG_QUALITY",
        "convert.py:8: names color_shares",
    ]
    # A person crop that copies COLOR_NAMES (the alias walkie-database cannot read),
    # or imports a name it no longer uses, puts it back at its old home.
    copy = _write(
        tmp_path / "person.py",
        "from . import colour\n"
        "from .colour import COLOR_NAMES, color_shares\n"
        "from .jpeg import encode_jpeg as encode_jpeg\n"
        "COLOR_NAMES = colour.COLOR_NAMES\n"
        "def scale_down(pixels, max_side): ...\n",
    )
    assert _binds_one_of(copy, LEFT_PERSON_CROP, imports=False) == [
        "person.py:3: binds encode_jpeg",
        "person.py:4: binds COLOR_NAMES",
        "person.py:5: binds scale_down",
    ]
    assert _binds_one_of(copy, UNUSED_IN_PERSON_CROP) == [
        "person.py:2: binds COLOR_NAMES",
        "person.py:4: binds COLOR_NAMES",
    ]
    # The new homes, the package and the names that stay with the person crop are fine.
    good = _write(
        tmp_path / "src" / "walkie_vision" / "node.py",
        "from . import crops\n"
        "from .crops import colour, jpeg, person\n"
        "from .crops.colour import COLOR_NAMES\n"
        "from .crops.person import CropStatus, crop_person\n"
        "import walkie_vision.crops.person\n"
        "a = person.check_options(0.1, 512, 0.5)\n"
        "b = jpeg.encode_jpeg(pixels, jpeg.JPEG_QUALITY)\n"
        "c = colour.COLOR_NAMES.index('black')\n"
        'monkeypatch.setattr(person, "MIN_BOX_HEIGHT_PX", 10.0)\n',
    )
    assert _imports_a_module_that_left(good, CROP) == []
    assert _reads_a_name_that_left(good, PERSON_CROP, LEFT_PERSON_CROP) == []


def test_the_sweep_would_catch_the_sampler_or_the_palette_at_their_old_home(tmp_path):
    bad = _write(
        tmp_path / "bad.py",
        "from walkie_vision.embeddings import CROP_MAX_SIDE, CROP_PADDING, MIN_COLOR_PIXELS, PALETTE_RGB\n"
        "from .embeddings import ClipEmbedder, select_samples\n"
        "from ..embeddings import _PALETTE_LAB\n"
        "chosen = embeddings.select_samples(detections, counter, width, height)\n"
        "if not np.all(np.isfinite(boxes.pad_box(box, embeddings.CROP_PADDING))):\n"
        'monkeypatch.setattr(embeddings_module, "encode_crop", clock.slowed(embeddings_module.encode_crop, 3.0))\n'
        "`walkie_vision.embeddings.mask_color_shares` counts the Lab palette\n"
        "`walkie_vision/embeddings:nearest_palette_indices` takes the earlier centre\n",
    )
    assert len(_mentions([bad], REMOVED)) == 8
    # The new homes, the CLIP provider at its own new home, longer names and the other
    # PALETTE_RGB in debug_viz are fine.
    good = _write(
        tmp_path / "good.py",
        "from walkie_vision.providers.image_embed.clip import ClipEmbedder, clip_checkpoint\n"
        "from .clip import ClipEmbedder, clip_checkpoint\n"
        "from walkie_vision.crops.objects import CROP_MAX_SIDE, CROP_PADDING\n"
        "from .crops.colour import MIN_COLOR_PIXELS, PALETTE_RGB\n"
        "chosen = object_crops.select_samples(detections, counter, width, height)\n"
        'monkeypatch.setattr(objects_module, "encode_crop", clock.slowed(objects_module.encode_crop, 3.0))\n'
        "`walkie_vision.providers.image_embed.clip.ClipEmbedder` is the interim provider\n"
        "clip.ClipEmbedder, clip.CLIP_DIM, object_embeddings.select_samples\n"
        "r, g, b = debug_viz.PALETTE_RGB[0]\n",
    )
    assert _mentions([good], REMOVED) == []


def test_the_tree_check_would_catch_the_sampler_or_the_palette_at_their_old_home(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "SRC", tmp_path / "src")
    # The moved names, taken from embeddings under any name.
    reader = _write(
        tmp_path / "src" / "walkie_vision" / "services" / "embed_crop.py",
        "from .. import (\n"
        "    convert,\n"
        "    embeddings,\n"
        ")\n"
        "from ..embeddings import (\n"
        "    ClipEmbedder,\n"
        "    padded_crop,\n"
        ")\n"
        "from walkie_vision import embeddings as embeddings_module\n"
        "a = embeddings.encode_crop(pixels)\n"
        'monkeypatch.setattr(embeddings_module, "mask_color_shares", fake)\n'
        "b = embeddings.ClipEmbedder(checkpoint)\n",
    )
    assert _reads_a_name_that_left(reader, EMBEDDINGS, LEFT_EMBEDDINGS) == [
        "embed_crop.py:10: reads embeddings.encode_crop",
        "embed_crop.py:11: names mask_color_shares",
        "embed_crop.py:5: imports padded_crop",
    ]
    # An embeddings that imports, re-exports or copies a moved name, or still imports
    # the box or crop helpers, puts them back at their old home.
    copy = _write(
        tmp_path / "src" / "walkie_vision" / "embeddings.py",
        "from . import boxes\n"
        "from .crops import objects\n"
        "from .crops.objects import CROP_PADDING, padded_crop\n"
        "from .crops.colour import *\n"
        "SAMPLE_PERIOD = objects.SAMPLE_PERIOD\n"
        "def select_samples(detections, counter, width, height): ...\n",
    )
    assert _binds_one_of(copy, LEFT_EMBEDDINGS) == [
        "embeddings.py:3: binds CROP_PADDING",
        "embeddings.py:3: binds padded_crop",
        "embeddings.py:4: binds *",
        "embeddings.py:5: binds SAMPLE_PERIOD",
        "embeddings.py:6: binds select_samples",
    ]
    assert _imports_a_module_that_left(copy, "walkie_vision.crops") == [
        "embeddings.py:2: imports from walkie_vision.crops",
        "embeddings.py:3: imports from walkie_vision.crops.objects",
        "embeddings.py:4: imports from walkie_vision.crops.colour",
    ]
    assert _imports_a_module_that_left(copy, "walkie_vision.boxes") == ["embeddings.py:1: imports walkie_vision.boxes"]
    # The new homes and the CLIP names at their own new home are fine.
    good = _write(
        tmp_path / "src" / "walkie_vision" / "pipeline.py",
        "from . import messages\n"
        "from .crops import colour\n"
        "from .crops import objects as object_crops\n"
        "from .providers.image_embed import clip\n"
        "from .providers.image_embed.clip import ClipEmbedder, clip_checkpoint\n"
        "a = object_crops.select_samples(detections, counter, width, height)\n"
        "b = colour.mask_color_shares(image, mask)\n"
        "c = clip.ClipEmbedder(checkpoint)\n"
        'monkeypatch.setattr(clip, "ClipEmbedder", fake)\n',
    )
    assert _reads_a_name_that_left(good, EMBEDDINGS, LEFT_EMBEDDINGS) == []
    assert _reads_a_name_that_left(good, CLIP, LEFT_EMBEDDINGS) == []
    clean = _write(
        tmp_path / "src" / "walkie_vision" / "providers" / "image_embed" / "clip.py",
        "from ...weights import CLIP_WEIGHTS, HUB_MODELS\n"
        "from ..vector_tables import CLIP_VIT_B16\n"
        "CLIP_DIM = 512\n"
        "def clip_checkpoint(download=None): ...\n",
    )
    assert _binds_one_of(clean, LEFT_EMBEDDINGS) == []
    assert _imports_a_module_that_left(clean, "walkie_vision.crops") == []


def test_image_and_depth_did_not_come_back_to_the_package_root():
    for module in ("__init__", "image", "depth", "mailboxes"):
        assert (PACKAGE_DIR / "camera" / f"{module}.py").is_file()
    assert (PACKAGE_DIR / "geometry.py").is_file()
    assert _old_homes(SRC, IMAGE) == []
    assert _old_homes(SRC, DEPTH) == []


def test_no_code_imports_image_or_depth_from_the_package_root():
    code = [file for file in _files_under(*CODE_AND_BUILD) if file.suffix == ".py"]
    assert {
        PACKAGE_DIR / "node.py",
        PACKAGE_DIR / "pipeline.py",
        PACKAGE_DIR / "debug_viz.py",
        ROOT / "checks" / "c12_memory.py",
        ROOT / "tests" / "test_node.py",
    } <= set(code)
    assert [hit for file in code for hit in _imports_a_module_that_left(file, IMAGE)] == []
    assert [hit for file in code for hit in _imports_a_module_that_left(file, DEPTH)] == []


def test_no_code_takes_the_depth_maths_the_mailboxes_or_stamp_ns_from_camera_depth():
    code = [file for file in _files_under(*CODE_AND_BUILD) if file.suffix == ".py"]
    embed_crop = PACKAGE_DIR / "services" / "embed_crop.py"
    assert {PACKAGE_DIR / "pipeline.py", embed_crop, ROOT / "tests" / "test_geometry.py"} <= set(code)
    assert [hit for file in code for hit in _reads_a_name_that_left(file, CAMERA_DEPTH, LEFT_DEPTH)] == []


def test_camera_depth_does_not_define_or_import_the_depth_maths_the_mailboxes_or_stamp_ns():
    depth = PACKAGE_DIR / "camera" / "depth.py"
    assert _binds_one_of(depth, LEFT_DEPTH) == []
    # The raster reading needs no mailbox and no message conversion.
    assert _imports_a_module_that_left(depth, "walkie_vision.camera.mailboxes") == []
    assert _imports_a_module_that_left(depth, MESSAGES) == []


def test_no_code_takes_the_mailbox_from_the_node_or_the_current_image_from_requests():
    code = [file for file in _files_under(*CODE_AND_BUILD) if file.suffix == ".py"]
    jobs_tests = ROOT / "tests" / "test_services_jobs.py"
    assert {PACKAGE_DIR / "node.py", ROOT / "tests" / "test_node.py", jobs_tests} <= set(code)
    assert [hit for file in code for hit in _reads_a_name_that_left(file, NODE, LEFT_NODE_MAILBOX)] == []
    assert [hit for file in code for hit in _reads_a_name_that_left(file, REQUESTS, LEFT_REQUESTS)] == []


def test_the_node_and_requests_do_not_define_the_mailboxes_again():
    # node.py imports NewestFrame to build its mailbox, but does not define it.
    # The services (requests.py until the services package) do not use CurrentImage, so
    # they do not even import it.
    assert _binds_one_of(PACKAGE_DIR / "node.py", LEFT_NODE_MAILBOX, imports=False) == []
    services = sorted((PACKAGE_DIR / "services").glob("*.py"))
    assert PACKAGE_DIR / "services" / "jobs.py" in services
    assert [hit for file in services for hit in _binds_one_of(file, LEFT_REQUESTS)] == []


def test_the_sweep_would_catch_image_depth_and_the_mailboxes_at_their_old_homes(tmp_path):
    bad = _write(
        tmp_path / "bad.py",
        "from walkie_vision.image import camera_info_problem, image_to_bgr\n"
        "from walkie_vision.depth import RegisteredDepth, lift_mask\n"
        "import walkie_vision.depth\n"
        "from walkie_vision import convert, depth, image\n"
        "| `src/walkie_vision/depth.py` | Registered depth and the object geometry |\n"
        "The pairing is walkie_vision.depth.RgbdBuffer (ADR-0014).\n"
        "assert isinstance(h.node._mailbox, node_module.NewestFrame)\n"
        "from walkie_vision.node import NewestFrame, VisionConfig\n"
        "from .node import NewestFrame\n"
        "current = requests.CurrentImage(stamp_ns, max_age_s=1.0)\n"
        "from walkie_vision.requests import CurrentImage, RequestJob\n"
        "from .requests import CurrentImage\n",
    )
    assert len(_mentions([bad], REMOVED)) == 12
    # The new homes, the names that stay in node and requests, longer names and the
    # words image and depth are fine. A relative import of image or depth is left to
    # the tree check, because camera/ imports its own image and depth that way.
    good = _write(
        tmp_path / "good.py",
        "from walkie_vision.camera.image import camera_info_problem, image_to_bgr\n"
        "from walkie_vision.camera.depth import RegisteredDepth, read_registered_depth\n"
        "from walkie_vision.camera import depth, image, mailboxes\n"
        "from walkie_vision.geometry import depth_cover, lift_mask\n"
        "from .camera.image import image_to_bgr\n"
        "from ..camera.image import calibration_problem, camera_info_problem\n"
        "from .image import calibration_problem, camera_info_problem\n"
        "from . import depth, image\n"
        "from .camera.mailboxes import CurrentImage, NewestFrame, RgbdBuffer\n"
        "from . import geometry, messages, services\n"
        "| `src/walkie_vision/camera/` | The camera input and its mailboxes |\n"
        "The pairing is walkie_vision.camera.mailboxes.RgbdBuffer (ADR-0014).\n"
        "assert isinstance(h.node._mailbox, NewestFrame)\n"
        "from walkie_vision.node import NODE_NAME, VisionNode\n"
        "job = requests.RequestJob(), requests.CurrentImages\n"
        "image.shape[:2], depth.metres, walkie_vision.depth_tools\n",
    )
    assert _mentions([good], REMOVED) == []


def test_the_tree_check_would_catch_image_depth_or_the_mailboxes_at_their_old_homes(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "SRC", tmp_path / "src")
    # The modules that left: image and depth, in any spelling of an import.
    caller = _write(
        tmp_path / "src" / "walkie_vision" / "services" / "get_depth.py",
        "from .. import (\n"
        "    convert,\n"
        "    depth,\n"
        ")\n"
        "from ..image import image_to_bgr\n"
        "import walkie_vision.depth as registered\n",
    )
    assert _imports_a_module_that_left(caller, DEPTH) == [
        "get_depth.py:1: imports walkie_vision.depth",
        "get_depth.py:6: imports walkie_vision.depth",
    ]
    assert _imports_a_module_that_left(caller, IMAGE) == ["get_depth.py:5: imports from walkie_vision.image"]
    # The same relative spellings at the package root name the old homes; inside
    # camera/ they name its own image and depth, which is where they live now.
    root = _write(
        tmp_path / "src" / "walkie_vision" / "fake_camera.py",
        "from .image import image_to_bgr\nfrom . import convert, depth\n",
    )
    assert _imports_a_module_that_left(root, IMAGE) == ["fake_camera.py:1: imports from walkie_vision.image"]
    assert _imports_a_module_that_left(root, DEPTH) == ["fake_camera.py:2: imports walkie_vision.depth"]
    sibling = _write(
        tmp_path / "src" / "walkie_vision" / "camera" / "depth.py",
        "from .image import calibration_problem, camera_info_problem\nfrom . import image\n",
    )
    assert _imports_a_module_that_left(sibling, IMAGE) == []
    # The depth maths, the mailboxes and stamp_ns, taken from camera.depth under any name.
    reader = _write(
        tmp_path / "src" / "walkie_vision" / "pipeline.py",
        "from .camera import depth\n"
        "from walkie_vision.camera import depth as registered\n"
        "from .camera.depth import (\n"
        "    RegisteredDepth,\n"
        "    lift_mask,\n"
        ")\n"
        "a = depth.view_quality(laplacian, 0.5)\n"
        "b = registered.stamp_ns(header)\n"
        'monkeypatch.setattr(depth, "frame_laplacian", fake)\n',
    )
    assert _reads_a_name_that_left(reader, CAMERA_DEPTH, LEFT_DEPTH) == [
        "pipeline.py:3: imports lift_mask",
        "pipeline.py:7: reads depth.view_quality",
        "pipeline.py:8: reads registered.stamp_ns",
        "pipeline.py:9: names frame_laplacian",
    ]
    # NewestFrame taken from the node, and CurrentImage from requests.
    mailbox = _write(
        tmp_path / "src" / "walkie_vision" / "debug_viz.py",
        "from . import node, requests\n"
        "from .node import NewestFrame\n"
        "from walkie_vision.requests import (\n"
        "    CurrentImage,\n"
        ")\n"
        "a = node.NewestFrame()\n"
        "b = requests.CurrentImage(1, 2.0)\n",
    )
    assert _reads_a_name_that_left(mailbox, NODE, LEFT_NODE_MAILBOX) == [
        "debug_viz.py:2: imports NewestFrame",
        "debug_viz.py:6: reads node.NewestFrame",
    ]
    assert _reads_a_name_that_left(mailbox, REQUESTS, LEFT_REQUESTS) == [
        "debug_viz.py:3: imports CurrentImage",
        "debug_viz.py:7: reads requests.CurrentImage",
    ]
    # A camera depth that imports, re-exports or copies a name that left, a node that
    # defines the mailbox again and a requests that imports the current image put them
    # back at their old homes.
    copy = _write(
        tmp_path / "depth.py",
        "from ..geometry import lift_mask\n"
        "from ..convert import stamp_ns as stamp_ns\n"
        "from .mailboxes import *\n"
        "def view_quality(laplacian, area_share): ...\n",
    )
    assert _binds_one_of(copy, LEFT_DEPTH) == [
        "depth.py:1: binds lift_mask",
        "depth.py:2: binds stamp_ns",
        "depth.py:3: binds *",
        "depth.py:4: binds view_quality",
    ]
    node = _write(tmp_path / "node.py", "from .camera.mailboxes import RgbdBuffer\nclass NewestFrame: ...\n")
    assert _binds_one_of(node, LEFT_NODE_MAILBOX, imports=False) == ["node.py:2: binds NewestFrame"]
    requests = _write(tmp_path / "requests.py", "from .camera.mailboxes import CurrentImage\n")
    assert _binds_one_of(requests, LEFT_REQUESTS) == ["requests.py:1: binds CurrentImage"]
    # The new homes and the names that stay are fine.
    good = _write(
        tmp_path / "src" / "walkie_vision" / "node.py",
        "from . import convert, geometry, requests\n"
        "from .camera import depth, image\n"
        "from .camera.depth import RegisteredDepth, read_registered_depth\n"
        "from .camera.image import image_to_bgr\n"
        "from .camera.mailboxes import CurrentImage, NewestFrame\n"
        "from .convert import stamp_ns\n"
        "a = depth.depth_to_metres(raster, encoding)\n"
        "b = geometry.view_quality(laplacian, 0.5)\n"
        "c = requests.RequestJob()\n"
        'monkeypatch.setattr(depth, "read_registered_depth", fake)\n',
    )
    assert _imports_a_module_that_left(good, IMAGE) == []
    assert _imports_a_module_that_left(good, DEPTH) == []
    assert _reads_a_name_that_left(good, CAMERA_DEPTH, LEFT_DEPTH) == []
    assert _reads_a_name_that_left(good, REQUESTS, LEFT_REQUESTS) == []
    assert _binds_one_of(good, LEFT_NODE_MAILBOX, imports=False) == []


def test_the_provider_interfaces_did_not_come_back_to_providers_py():
    providers = PACKAGE_DIR / "providers"
    assert (providers / "vector_tables.py").is_file()
    for task in ("object_detection", "pose_estimation", "face_recognition", "reid", "image_embed"):
        assert (providers / task / "base.py").is_file()
    # The package is the new home; a providers.py beside it would be the old one.
    assert _old_homes(SRC, "walkie_vision.providers") == [providers]


def test_no_code_takes_the_vector_lengths_or_the_dim_check_from_convert():
    code = [file for file in _files_under(*CODE_AND_BUILD) if file.suffix == ".py"]
    assert {PACKAGE_DIR / "messages.py", PACKAGE_DIR / "pipeline.py", ROOT / "tests" / "test_messages.py"} <= set(code)
    # convert.py is messages.py now: neither the old module nor its successor serves them.
    assert [hit for file in code for hit in _reads_a_name_that_left(file, CONVERT, LEFT_CONVERT)] == []
    assert [hit for file in code for hit in _reads_a_name_that_left(file, MESSAGES, LEFT_CONVERT)] == []


def test_convert_does_not_define_the_vector_lengths_or_the_dim_check_again():
    # messages.py (convert.py until the rename) imports EMBEDDING_DIMS, which it still
    # uses, but defines neither. The dim check's one caller in it left for
    # services/embed_crop.py, so it does not even import that.
    assert _binds_one_of(PACKAGE_DIR / "messages.py", LEFT_CONVERT, imports=False) == []
    assert _binds_one_of(PACKAGE_DIR / "messages.py", frozenset({"check_model_dim"})) == []


def test_the_sweep_would_catch_providers_py_or_the_vector_lengths_at_their_old_homes(tmp_path):
    bad = _write(
        tmp_path / "bad.py",
        "| `src/walkie_vision/providers.py`    | The interfaces every model implements |\n"
        "- `src/walkie_vision/providers.py`: model interfaces and result types.\n"
        "expected = convert.EMBEDDING_DIMS.get(name)\n"
        "convert.check_model_dim(providers.embedder)\n"
        "from walkie_vision.convert import EMBEDDING_DIMS, to_point_cloud\n"
        "from .convert import bounded_message, check_model_dim\n"
        "from ..convert import EMBEDDING_DIMS\n"
        "`walkie_vision.convert:check_model_dim` checks the length\n",
    )
    assert len(_mentions([bad], REMOVED)) == 8
    # The new homes, the re-exported names, the names that stay in convert (messages
    # since the rename), longer file names and the bare call are fine.
    good = _write(
        tmp_path / "good.py",
        "| `src/walkie_vision/providers/`      | The interfaces every model implements |\n"
        "src/walkie_vision/providers/__init__.py re-exports them\n"
        "from walkie_vision.providers import CLIP_VIT_B16, Detection\n"
        "from walkie_vision.providers.vector_tables import EMBEDDING_DIMS\n"
        "from .providers.image_embed.base import check_model_dim\n"
        "from ..vector_tables import EMBEDDING_DIMS\n"
        "checks/c11_providers.py and tests/test_loading.py\n"
        "messages.to_point_cloud(header, points), messages.unit_vector(vector)\n"
        "from walkie_vision.messages import pose_from_person_detection\n"
        "check_model_dim(providers.embedder)\n",
    )
    assert _mentions([good], REMOVED) == []


def test_the_tree_check_would_catch_the_vector_lengths_or_the_dim_check_at_their_old_home(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "SRC", tmp_path / "src")
    reader = _write(
        tmp_path / "src" / "walkie_vision" / "services" / "embed_crop.py",
        "from .. import convert\n"
        "from walkie_vision import convert as convert_module\n"
        "from ..convert import (\n"
        "    EMBEDDING_DIMS,\n"
        "    stamp_ns,\n"
        ")\n"
        "a = convert.check_model_dim(embedder)\n"
        'monkeypatch.setattr(convert_module, "EMBEDDING_DIMS", {})\n'
        "from ..convert import *\n",
    )
    assert _reads_a_name_that_left(reader, CONVERT, LEFT_CONVERT) == [
        "embed_crop.py:3: imports EMBEDDING_DIMS",
        "embed_crop.py:7: reads convert.check_model_dim",
        "embed_crop.py:8: names EMBEDDING_DIMS",
        "embed_crop.py:9: imports *",
    ]
    # A convert that re-exports or defines them again puts them back at their old home.
    copy = _write(
        tmp_path / "convert.py",
        "from .providers.vector_tables import EMBEDDING_DIMS\n"
        "from .providers.image_embed.base import check_model_dim as check_model_dim\n"
        "EMBEDDING_DIMS = {}\n"
        "def check_model_dim(embedder): ...\n",
    )
    assert _binds_one_of(copy, LEFT_CONVERT, imports=False) == [
        "convert.py:2: binds check_model_dim",
        "convert.py:3: binds EMBEDDING_DIMS",
        "convert.py:4: binds check_model_dim",
    ]
    # The new homes, a convert that only imports and uses them, and the names that stay
    # in convert are fine.
    good = _write(
        tmp_path / "src" / "walkie_vision" / "pipeline.py",
        "from . import convert\n"
        "from .providers.image_embed.base import check_model_dim\n"
        "from .providers.vector_tables import EMBEDDING_DIMS\n"
        "a = convert.to_point_cloud(header, points)\n"
        "b = check_model_dim(embedder)\n"
        'monkeypatch.setattr(convert, "new_stream_id", fake)\n',
    )
    assert _reads_a_name_that_left(good, CONVERT, LEFT_CONVERT) == []
    clean = _write(
        tmp_path / "clean" / "convert.py",
        "from .providers.image_embed.base import check_model_dim\n"
        "from .providers.vector_tables import EMBEDDING_DIMS\n"
        "expected = EMBEDDING_DIMS.get(name)\n",
    )
    assert _binds_one_of(clean, LEFT_CONVERT, imports=False) == []


def test_requests_did_not_come_back_to_the_package_root():
    for module in ("__init__", "jobs", "embed_text", "embed_crop", "detect_objects"):
        assert (PACKAGE_DIR / "services" / f"{module}.py").is_file()
    assert _old_homes(SRC, REQUESTS) == []


def test_no_code_imports_requests():
    code = [file for file in _files_under(*CODE_AND_BUILD) if file.suffix == ".py"]
    assert {
        PACKAGE_DIR / "node.py",
        ROOT / "tests" / "test_node.py",
        ROOT / "tests" / "test_golden_messages.py",
    } <= set(code)
    assert [hit for file in code for hit in _imports_a_module_that_left(file, REQUESTS)] == []


def test_no_code_takes_the_service_builders_or_the_message_bound_from_convert():
    code = [file for file in _files_under(*CODE_AND_BUILD) if file.suffix == ".py"]
    assert {
        PACKAGE_DIR / "node.py",
        PACKAGE_DIR / "services" / "embed_crop.py",
        ROOT / "tests" / "test_golden_messages.py",
        ROOT / "tests" / "test_services_jobs.py",
    } <= set(code)
    assert [hit for file in code for hit in _reads_a_name_that_left(file, CONVERT, LEFT_CONVERT_SERVICES)] == []
    assert [hit for file in code for hit in _reads_a_name_that_left(file, MESSAGES, LEFT_CONVERT_SERVICES)] == []


def test_convert_does_not_define_or_import_the_service_builders_or_the_message_bound():
    # messages.py is convert.py renamed.
    assert _binds_one_of(PACKAGE_DIR / "messages.py", LEFT_CONVERT_SERVICES) == []


def test_the_sweep_would_catch_requests_or_the_service_builders_at_their_old_homes(tmp_path):
    bad = _write(
        tmp_path / "bad.py",
        "from walkie_vision import errors, requests\n"
        "from walkie_vision.requests import BUSY, ServiceJob\n"
        "import walkie_vision.requests\n"
        "from . import convert, requests\n"
        "from .requests import JobSlot\n"
        "from ..requests import MAX_TIMEOUT_S\n"
        "| `src/walkie_vision/requests.py` | Bounded service work |\n"
        "The pure parts are tested in tests/test_requests.py.\n"
        "return convert.embed_text_failure(jobs.DISABLED, reason)\n"
        "fail = convert.detect_objects_failure\n"
        "response.embedding = convert._float32_array(unit)\n"
        "assert len(message) <= convert.MAX_MESSAGE_CHARS\n"
        "from walkie_vision.convert import box_to_bbox, vectors_from_response\n"
        "from .convert import bounded_message\n"
        "`walkie_vision.convert:fill_embed_crop_response` fills a crop\n",
    )
    assert len(_mentions([bad], REMOVED)) == 15
    # The new homes, the names that stay in convert (messages since the rename),
    # float32_array, the requests library and the word requests are fine.
    good = _write(
        tmp_path / "good.py",
        "from walkie_vision.services import detect_objects, embed_crop, embed_text, jobs\n"
        "from walkie_vision.services.jobs import BUSY, ServiceJob\n"
        "from .services import embed_text as embed_text_service\n"
        "from .jobs import MAX_TIMEOUT_S, bounded_message\n"
        "from ..messages import float32_array, stamp_ns, unit_vector\n"
        "from walkie_vision.messages import box_to_bbox, pose_from_person_detection\n"
        "response.embedding = messages.float32_array(unit)\n"
        "| `src/walkie_vision/services/` | Bounded service work |\n"
        "The pure parts are tested in tests/test_services_jobs.py.\n"
        "import requests\n"
        "Each callback checks its request, so requests cannot stop the stream.\n"
        "messages.pose_from_person_detection(person)\n",
    )
    assert _mentions([good], REMOVED) == []


def test_the_tree_check_would_catch_requests_or_the_service_builders_at_their_old_homes(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "SRC", tmp_path / "src")
    # The module that left: requests, in any spelling of an import.
    caller = _write(
        tmp_path / "src" / "walkie_vision" / "node.py",
        "from . import (\n"
        "    convert,\n"
        "    requests,\n"
        ")\n"
        "from .requests import JobSlot\n"
        "import walkie_vision.requests as jobs\n",
    )
    assert _imports_a_module_that_left(caller, REQUESTS) == [
        "node.py:1: imports walkie_vision.requests",
        "node.py:5: imports from walkie_vision.requests",
        "node.py:6: imports walkie_vision.requests",
    ]
    # Inside services/, `from . import jobs` and `from .jobs import` name the new home.
    sibling = _write(
        tmp_path / "src" / "walkie_vision" / "services" / "embed_text.py",
        "from . import jobs\nfrom .jobs import bounded_message\nfrom .. import convert\n",
    )
    assert _imports_a_module_that_left(sibling, REQUESTS) == []
    # The builders and the message bound, taken from convert under any name.
    reader = _write(
        tmp_path / "src" / "walkie_vision" / "services" / "embed_crop.py",
        "from .. import convert\n"
        "from walkie_vision import convert as convert_module\n"
        "from ..convert import (\n"
        "    bounded_message,\n"
        "    stamp_ns,\n"
        ")\n"
        "a = convert.fill_embed_crop_response(response, header, jpeg, embedder, vector, 0.5)\n"
        "b = convert_module._float32_array(unit)\n"
        'monkeypatch.setattr(convert, "MAX_MESSAGE_CHARS", 8)\n',
    )
    assert _reads_a_name_that_left(reader, CONVERT, LEFT_CONVERT_SERVICES) == [
        "embed_crop.py:3: imports bounded_message",
        "embed_crop.py:7: reads convert.fill_embed_crop_response",
        "embed_crop.py:8: reads convert_module._float32_array",
        "embed_crop.py:9: names MAX_MESSAGE_CHARS",
    ]
    # A convert that imports, re-exports or copies one of them puts it back at its old home.
    copy = _write(
        tmp_path / "convert.py",
        "from .services.jobs import bounded_message\n"
        "from .services.embed_text import vectors_from_response as vectors_from_response\n"
        "MAX_MESSAGE_CHARS = 256\n"
        "_float32_array = float32_array\n"
        "def embed_crop_failure(code, message): ...\n",
    )
    assert _binds_one_of(copy, LEFT_CONVERT_SERVICES) == [
        "convert.py:1: binds bounded_message",
        "convert.py:2: binds vectors_from_response",
        "convert.py:3: binds MAX_MESSAGE_CHARS",
        "convert.py:4: binds _float32_array",
        "convert.py:5: binds embed_crop_failure",
    ]
    # The new homes and the names that stay in convert are fine.
    good = _write(
        tmp_path / "src" / "walkie_vision" / "pipeline.py",
        "from . import convert\n"
        "from .services import embed_text, jobs\n"
        "from .services.embed_crop import fill_embed_crop_response\n"
        "a = convert.float32_array(values)\n"
        "b = jobs.bounded_message(text)\n"
        'monkeypatch.setattr(embed_text, "EMBED_TEXT_DEADLINE_S", 0.2)\n',
    )
    assert _reads_a_name_that_left(good, CONVERT, LEFT_CONVERT_SERVICES) == []
    assert _imports_a_module_that_left(good, REQUESTS) == []
    clean = _write(
        tmp_path / "clean" / "convert.py",
        "def float32_array(values): ...\nresponse.embedding = float32_array(unit)\n",
    )
    assert _binds_one_of(clean, LEFT_CONVERT_SERVICES) == []


def test_no_code_takes_the_person_crop_service_from_the_node_or_convert():
    assert (PACKAGE_DIR / "services" / "get_person_crop.py").is_file()
    code = [file for file in _files_under(*CODE_AND_BUILD) if file.suffix == ".py"]
    assert {
        PACKAGE_DIR / "node.py",
        ROOT / "tests" / "test_node.py",
        ROOT / "tests" / "test_golden_messages.py",
        ROOT / "tests" / "test_services_get_person_crop.py",
    } <= set(code)
    assert [hit for file in code for hit in _reads_a_name_that_left(file, NODE, LEFT_NODE_PERSON_CROP)] == []
    assert [hit for file in code for hit in _reads_a_name_that_left(file, CONVERT, LEFT_CONVERT_PERSON_CROP)] == []
    assert [hit for file in code for hit in _reads_a_name_that_left(file, MESSAGES, LEFT_CONVERT_PERSON_CROP)] == []


def test_the_node_and_convert_do_not_define_the_person_crop_service_again():
    # node.py imports the job, the mailbox, the check and the answers it uses, but
    # defines none of them; the timeout cap is used only by the check, so node.py does
    # not even import it. messages.py (convert.py until the rename) does not use the
    # response builder.
    node = PACKAGE_DIR / "node.py"
    assert _binds_one_of(node, LEFT_NODE_PERSON_CROP, imports=False) == []
    assert _binds_one_of(node, frozenset({"MAX_CROP_TIMEOUT_S"})) == []
    assert _binds_one_of(PACKAGE_DIR / "messages.py", LEFT_CONVERT_PERSON_CROP) == []


def test_the_sweep_would_catch_the_person_crop_service_at_its_old_homes(tmp_path):
    bad = _write(
        tmp_path / "bad.py",
        "from walkie_vision.node import IMAGE_QOS, CropJob, CropMailbox, VisionConfig\n"
        "from walkie_vision.node import _crop_answer\n"
        "from .node import crop_request_problem\n"
        "from ..node import MAX_CROP_TIMEOUT_S\n"
        "mailbox = node.CropMailbox()\n"
        "assert node_module.MAX_CROP_TIMEOUT_S == 5.0\n"
        "`walkie_vision.node:crop_request_problem` checks the request\n"
        "response = convert.fill_person_crop_response(GetPersonCrop.Response(), people, found)\n"
        "from walkie_vision.convert import (pose_from_person_detection, fill_person_crop_response)\n"
        "from .convert import fill_person_crop_response\n",
    )
    assert len(_mentions([bad], REMOVED)) == 10
    # The new home, the names that stay in node and convert (messages since the
    # rename), longer names and the bare calls are fine.
    good = _write(
        tmp_path / "good.py",
        "from walkie_vision.services.get_person_crop import CropJob, CropMailbox, fill_person_crop_response\n"
        "from .services.get_person_crop import _crop_answer, crop_request_problem\n"
        "from .get_person_crop import MAX_CROP_TIMEOUT_S\n"
        "from walkie_vision.services import get_person_crop\n"
        "mailbox = get_person_crop.CropMailbox()\n"
        "from walkie_vision.node import EXECUTOR_THREADS, NODE_NAME, LatencyLog, VisionNode\n"
        "from walkie_vision.messages import pose_from_person_detection, to_person_detection_array\n"
        "messages.to_person_detection_array(header, info, people), node.CropJobs\n"
        "self._crops = CropMailbox()\n"
        "response = fill_person_crop_response(GetPersonCrop.Response(), people, found)\n",
    )
    assert _mentions([good], REMOVED) == []


def test_the_tree_check_would_catch_the_person_crop_service_at_its_old_homes(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "SRC", tmp_path / "src")
    # The node's names and convert's builder, taken from their old homes under any name.
    reader = _write(
        tmp_path / "src" / "walkie_vision" / "services" / "describe_crop.py",
        "from .. import convert, node\n"
        "from walkie_vision import node as node_module\n"
        "from ..node import (\n"
        "    CropJob,\n"
        "    VisionConfig,\n"
        ")\n"
        "from ..convert import (\n"
        "    fill_person_crop_response,\n"
        "    stamp_ns,\n"
        ")\n"
        "a = node._crop_answer(status, message)\n"
        'monkeypatch.setattr(node_module, "MAX_CROP_TIMEOUT_S", 1.0)\n'
        "b = convert.fill_person_crop_response(response, people, found)\n"
        "from ..node import *\n",
    )
    assert _reads_a_name_that_left(reader, NODE, LEFT_NODE_PERSON_CROP) == [
        "describe_crop.py:11: reads node._crop_answer",
        "describe_crop.py:12: names MAX_CROP_TIMEOUT_S",
        "describe_crop.py:14: imports *",
        "describe_crop.py:3: imports CropJob",
    ]
    assert _reads_a_name_that_left(reader, CONVERT, LEFT_CONVERT_PERSON_CROP) == [
        "describe_crop.py:13: reads convert.fill_person_crop_response",
        "describe_crop.py:7: imports fill_person_crop_response",
    ]
    # A node that defines one again, re-exports one or imports the cap it no longer
    # uses, and a convert that imports or copies the builder, put them back.
    node = _write(
        tmp_path / "node.py",
        "from .services.get_person_crop import CropJob, MAX_CROP_TIMEOUT_S\n"
        "from .services.get_person_crop import CropMailbox as CropMailbox\n"
        "def crop_request_problem(request): ...\n",
    )
    assert _binds_one_of(node, LEFT_NODE_PERSON_CROP, imports=False) == [
        "node.py:2: binds CropMailbox",
        "node.py:3: binds crop_request_problem",
    ]
    assert _binds_one_of(node, frozenset({"MAX_CROP_TIMEOUT_S"})) == ["node.py:1: binds MAX_CROP_TIMEOUT_S"]
    convert = _write(
        tmp_path / "convert.py",
        "from .services.get_person_crop import fill_person_crop_response\n"
        "def fill_person_crop_response(response, people, found): ...\n",
    )
    assert _binds_one_of(convert, LEFT_CONVERT_PERSON_CROP) == [
        "convert.py:1: binds fill_person_crop_response",
        "convert.py:2: binds fill_person_crop_response",
    ]
    # The new home, a node that only imports and uses them, and the names that stay
    # in node and convert are fine.
    good = _write(
        tmp_path / "src" / "walkie_vision" / "node.py",
        "from . import convert\n"
        "from .services import get_person_crop\n"
        "from .services.get_person_crop import CropJob, CropMailbox, _crop_answer, crop_request_problem\n"
        "from .services.get_person_crop import fill_person_crop_response\n"
        "a = convert.pose_from_person_detection(person)\n"
        "b = get_person_crop.MAX_CROP_TIMEOUT_S\n"
        'monkeypatch.setattr(get_person_crop, "MAX_CROP_TIMEOUT_S", 1.0)\n'
        "mailbox = CropMailbox()\n",
    )
    assert _reads_a_name_that_left(good, NODE, LEFT_NODE_PERSON_CROP) == []
    assert _reads_a_name_that_left(good, CONVERT, LEFT_CONVERT_PERSON_CROP) == []
    assert _binds_one_of(good, LEFT_NODE_PERSON_CROP, imports=False) == []


def test_convert_did_not_come_back_to_the_package_root():
    assert (PACKAGE_DIR / "messages.py").is_file()
    assert _old_homes(SRC, CONVERT) == []


def test_no_code_imports_convert():
    code = [file for file in _files_under(*CODE_AND_BUILD) if file.suffix == ".py"]
    assert {
        PACKAGE_DIR / "node.py",
        PACKAGE_DIR / "pipeline.py",
        PACKAGE_DIR / "geometry.py",
        PACKAGE_DIR / "services" / "embed_crop.py",
        ROOT / "checks" / "c12_memory.py",
        ROOT / "tests" / "test_messages.py",
    } <= set(code)
    assert [hit for file in code for hit in _imports_a_module_that_left(file, CONVERT)] == []


def test_no_code_takes_the_qos_profiles_from_the_node():
    code = [file for file in _files_under(*CODE_AND_BUILD) if file.suffix == ".py"]
    assert {
        PACKAGE_DIR / "fake_publisher.py",
        PACKAGE_DIR / "debug_viz.py",
        ROOT / "tests" / "test_node.py",
        ROOT / "tests" / "test_fake_publisher.py",
    } <= set(code)
    assert [hit for file in code for hit in _reads_a_name_that_left(file, NODE, LEFT_NODE_QOS)] == []


def test_the_node_does_not_define_the_qos_profiles_again():
    # node.py imports the three profiles, since it subscribes and publishes with them,
    # but defines none; messages.py defines all three.
    assert _binds_one_of(PACKAGE_DIR / "node.py", LEFT_NODE_QOS, imports=False) == []
    messages = _binds_one_of(PACKAGE_DIR / "messages.py", LEFT_NODE_QOS, imports=False)
    assert {hit.rpartition(" ")[2] for hit in messages} == LEFT_NODE_QOS


def test_the_sweep_would_catch_convert_or_the_qos_profiles_at_their_old_homes(tmp_path):
    bad = _write(
        tmp_path / "bad.py",
        "from walkie_vision.convert import bbox_to_box, stamp_ns\n"
        "import walkie_vision.convert\n"
        "from walkie_vision import convert, errors\n"
        "from walkie_vision import convert as convert_module\n"
        "from . import boxes, convert\n"
        "from .convert import MAX_CLOUD_POINTS, evenly_spaced_indices\n"
        "from ..convert import float32_array\n"
        "| `src/walkie_vision/convert.py`      | Model results to messages, with checks |\n"
        "- A message change needs: the `.msg` comments, `convert.py`, the field pins\n"
        "  in `tests/test_convert.py`, and a rebuild in every consuming repository\n"
        "`walkie_vision.convert.check_object_detection_array` holds these rules\n"
        "self._people_pub = self.create_publisher(PersonDetectionArray, topic, node.OUTPUT_QOS)\n"
        "assert OBSERVATION_QOS == node_module.OUTPUT_QOS\n"
        "from walkie_vision.node import IMAGE_QOS, NODE_NAME, VisionConfig, VisionNode\n"
        "from .node import CAMERA_INFO_QOS\n"
        "`walkie_vision.node:OUTPUT_QOS` is RELIABLE\n",
    )
    assert len(_mentions([bad], REMOVED)) == 16
    # The new home, the QoS profiles at it and in debug_viz, PIL's convert, the convert
    # stage of the latency log, a longer module name and the word converted are fine.
    good = _write(
        tmp_path / "good.py",
        "from walkie_vision.messages import bbox_to_box, stamp_ns\n"
        "from walkie_vision import errors, messages\n"
        "from . import boxes, messages\n"
        "from .messages import CAMERA_INFO_QOS, IMAGE_QOS, OUTPUT_QOS, stamp_ns\n"
        "from ..messages import float32_array\n"
        "| `src/walkie_vision/messages.py`     | Model results to messages, with checks, and the QoS profiles |\n"
        'bus = Image.open(assets_dir() / "bus.jpg").convert("RGB")\n'
        'cache[c.image] = ImageOps.exif_transpose(im).convert("RGB")\n'
        "ms p50/p95: wait 3/9 convert 6/8 objects 21/30\n"
        'BASE_STAGES = ("wait", "convert", "objects")\n'
        "convert_ms = (time.perf_counter() - start) * 1000.0 - depth_ms\n"
        "from walkie_vision.converters import to_ros\n"
        "from walkie_vision.node import EXECUTOR_THREADS, NODE_NAME, VisionNode\n"
        "assert OBSERVATION_QOS == messages.OUTPUT_QOS\n"
        "from walkie_vision.debug_viz import ARRAY_QOS, IMAGE_QOS, OUTPUT_QOS\n"
        "self.create_subscription(Image, cfg.image_topic, self._ring.add, debug_viz.IMAGE_QOS)\n"
        "Each image is converted to BGR before the models run.\n",
    )
    assert _mentions([good], REMOVED) == []


def test_the_tree_check_would_catch_convert_or_the_qos_profiles_at_their_old_homes(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "SRC", tmp_path / "src")
    # The module that left: convert, in any spelling of an import.
    caller = _write(
        tmp_path / "src" / "walkie_vision" / "services" / "embed_crop.py",
        "from .. import (\n"
        "    boxes,\n"
        "    convert,\n"
        ")\n"
        "from ..convert import stamp_ns\n"
        "import walkie_vision.convert as conversions\n"
        "from walkie_vision import convert as convert_module\n"
        "def answer():\n"
        "    from ..convert import bbox_to_box\n"
        "    return bbox_to_box\n",
    )
    assert _imports_a_module_that_left(caller, CONVERT) == [
        "embed_crop.py:1: imports walkie_vision.convert",
        "embed_crop.py:5: imports from walkie_vision.convert",
        "embed_crop.py:6: imports walkie_vision.convert",
        "embed_crop.py:7: imports walkie_vision.convert",
        "embed_crop.py:9: imports from walkie_vision.convert",
    ]
    # The names that left convert before the rename, taken from its successor.
    successor = _write(
        tmp_path / "src" / "walkie_vision" / "services" / "embed_text.py",
        "from .. import messages\n"
        "from ..messages import (\n"
        "    bounded_message,\n"
        "    float32_array,\n"
        ")\n"
        "a = messages.check_model_dim(embedder)\n"
        "b = messages.fill_person_crop_response(response, people, found)\n",
    )
    left_convert = LEFT_CONVERT | LEFT_CONVERT_SERVICES | LEFT_CONVERT_PERSON_CROP
    assert _reads_a_name_that_left(successor, MESSAGES, left_convert) == [
        "embed_text.py:2: imports bounded_message",
        "embed_text.py:6: reads messages.check_model_dim",
        "embed_text.py:7: reads messages.fill_person_crop_response",
    ]
    # The QoS profiles, taken from the node under any name.
    reader = _write(
        tmp_path / "src" / "walkie_vision" / "fake_publisher.py",
        "from . import node\n"
        "from walkie_vision import node as node_module\n"
        "from .node import (\n"
        "    NODE_NAME,\n"
        "    OUTPUT_QOS,\n"
        ")\n"
        "a = node.IMAGE_QOS\n"
        'monkeypatch.setattr(node_module, "CAMERA_INFO_QOS", profile)\n'
        "from .node import *\n",
    )
    assert _reads_a_name_that_left(reader, NODE, LEFT_NODE_QOS) == [
        "fake_publisher.py:3: imports OUTPUT_QOS",
        "fake_publisher.py:7: reads node.IMAGE_QOS",
        "fake_publisher.py:8: names CAMERA_INFO_QOS",
        "fake_publisher.py:9: imports *",
    ]
    # A node that defines, re-exports or star-imports a profile puts it back at its old home.
    copy = _write(
        tmp_path / "node.py",
        "from .messages import IMAGE_QOS, OUTPUT_QOS\n"
        "from .messages import CAMERA_INFO_QOS as CAMERA_INFO_QOS\n"
        "OUTPUT_QOS = QoSProfile(depth=5)\n"
        "from .messages import *\n",
    )
    assert _binds_one_of(copy, LEFT_NODE_QOS, imports=False) == [
        "node.py:2: binds CAMERA_INFO_QOS",
        "node.py:3: binds OUTPUT_QOS",
        "node.py:4: binds *",
    ]
    # The new home, a module of the same name in a subpackage, a longer module name, a
    # node that only imports and uses the profiles, and a docstring are fine.
    sibling = _write(
        tmp_path / "src" / "walkie_vision" / "camera" / "mailboxes.py",
        '"""stamp_ns came from walkie_vision.convert."""\n'
        "from .. import messages\n"
        "from ..messages import stamp_ns\n"
        "from . import convert\n"
        "from walkie_vision.messages import OUTPUT_QOS\n"
        "import walkie_vision.converters\n",
    )
    assert _imports_a_module_that_left(sibling, CONVERT) == []
    good = _write(
        tmp_path / "src" / "walkie_vision" / "node.py",
        "from . import messages\n"
        "from .messages import CAMERA_INFO_QOS, IMAGE_QOS, OUTPUT_QOS, stamp_ns\n"
        "a = messages.bbox_to_box(request.bbox)\n"
        "b = self.create_publisher(PersonDetectionArray, topic, OUTPUT_QOS)\n"
        'monkeypatch.setattr(messages, "new_stream_id", fake)\n',
    )
    assert _imports_a_module_that_left(good, CONVERT) == []
    assert _reads_a_name_that_left(good, NODE, LEFT_NODE_QOS) == []
    assert _reads_a_name_that_left(good, MESSAGES, left_convert) == []
    assert _binds_one_of(good, LEFT_NODE_QOS, imports=False) == []


def test_the_vendored_files_did_not_come_back_to_face_or_reid():
    for old, new in VENDORED.items():
        assert (PACKAGE_DIR / new).is_file()
        assert _old_homes(SRC, old) == []


def test_no_code_imports_the_vendored_files_from_face_or_reid():
    code = [file for file in _files_under(*CODE_AND_BUILD) if file.suffix == ".py"]
    assert {
        BUFFALO_L_FILE,
        OSNET_FILE,
        PACKAGE_DIR / "_vendor" / "insightface" / "arcface.py",
        ROOT / "tests" / "test_providers_face_recognition_buffalo_l.py",
        ROOT / "tests" / "test_vendor_insightface_face_align.py",
    } <= set(code)
    assert [hit for file in code for old in VENDORED for hit in _imports_a_module_that_left(file, old)] == []
    assert [hit for file in code for hit in _reads_a_name_that_left(file, FACE, LEFT_FACE)] == []
    assert [hit for file in code for hit in _reads_a_name_that_left(file, REID, LEFT_REID)] == []


def test_face_and_reid_do_not_bind_the_vendored_modules_again():
    # The face package is gone, so nothing serves the vendored modules under it. Its
    # successor, providers/face_recognition/buffalo_l.py, binds face_align, which it
    # calls through the module, and no other vendored module. The reid package is gone
    # too; its successor, providers/reid/osnet.py, binds osnet_x1_0 only.
    assert _old_homes(SRC, FACE) == []
    assert {hit.rpartition(" ")[2] for hit in _binds_one_of(BUFFALO_L_FILE, LEFT_FACE)} == {"face_align"}
    assert _old_homes(SRC, REID) == []
    assert _binds_one_of(OSNET_FILE, LEFT_REID) == []


def test_the_sweep_would_catch_the_vendored_files_at_their_old_homes(tmp_path):
    bad = _write(
        tmp_path / "bad.py",
        "from walkie_vision.face.face_align import _similarity_transform, arcface_dst\n"
        "import walkie_vision.face.scrfd\n"
        "from walkie_vision.face import HeadCropConfig, HeadFaceRecognizer, face_align\n"
        "from walkie_vision.face import arcface as vendored\n"
        "monkeypatch.setattr(walkie_vision.face.face_align, 'norm_crop', norm_crop)\n"
        '"walkie_vision.face.arcface",\n'
        "| `src/walkie_vision/face/scrfd.py` | SCRFD (vendored) |\n"
        "#   least-squares method; tests/test_face.py compares them.\n"
        "from walkie_vision.reid.osnet import osnet_x1_0\n"
        "from walkie_vision.reid import load_osnet, osnet\n"
        "The network is in `src/walkie_vision/reid/osnet.py`, under the MIT licence.\n"
        "reid/osnet.py keeps its header\n",
    )
    assert len(_mentions([bad], REMOVED)) == 12
    # The new homes, the face wrapper in buffalo_l.py, the reid wrapper in the provider
    # osnet.py, the vendored files by their new paths, the vector model names and prose
    # are fine.
    good = _write(
        tmp_path / "good.py",
        "from walkie_vision._vendor.insightface.face_align import _similarity_transform, arcface_dst\n"
        "from walkie_vision._vendor.insightface import face_align\n"
        "from walkie_vision.providers.face_recognition.buffalo_l import HeadCropConfig, HeadFaceRecognizer\n"
        "from walkie_vision.providers.face_recognition.buffalo_l import box_iou, clip_faces\n"
        "from walkie_vision.providers.reid.osnet import INPUT_SIZE, MEAN, STD, load_osnet\n"
        "from walkie_vision.providers.reid.osnet import OSNetReIdentifier, osnet_input\n"
        "from .._vendor.torchreid.osnet import osnet_x1_0\n"
        "| `src/walkie_vision/_vendor/insightface/scrfd.py` | SCRFD (vendored) |\n"
        "`_vendor/torchreid/osnet.py` keeps its header\n"
        "provider = PACKAGE_DIR / 'providers' / 'reid' / 'osnet.py'\n"
        "`src/walkie_vision/providers/reid/osnet.py` wraps the network\n"
        "| `src/walkie_vision/providers/face_recognition/buffalo_l.py` | SCRFD and ArcFace on ONNX Runtime |\n"
        'ARCFACE_W600K_R50 = "arcface_w600k_r50"\n'
        "tests/test_vendor_insightface_face_align.py compares them, and test_faces.py others.\n",
    )
    assert _mentions([good], REMOVED) == []


def test_the_tree_check_would_catch_the_vendored_files_at_their_old_homes(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "SRC", tmp_path / "src")
    # The modules that left, in any spelling of an import, absolute or relative.
    heads = _write(
        tmp_path / "src" / "walkie_vision" / "face" / "heads.py",
        "from . import (\n"
        "    analysis,\n"
        "    face_align,\n"
        ")\n"
        "from .arcface import ArcFaceONNX\n"
        "from walkie_vision.face import scrfd as detector\n"
        "import walkie_vision.face.face_align\n"
        "def recognise():\n"
        "    from .scrfd import SCRFD\n"
        "    return SCRFD\n",
    )
    found = [hit for old in VENDORED for hit in _imports_a_module_that_left(heads, old)]
    assert sorted(found) == [
        "heads.py:1: imports walkie_vision.face.face_align",
        "heads.py:5: imports from walkie_vision.face.arcface",
        "heads.py:6: imports walkie_vision.face.scrfd",
        "heads.py:7: imports walkie_vision.face.face_align",
        "heads.py:9: imports from walkie_vision.face.scrfd",
    ]
    loader = _write(
        tmp_path / "src" / "walkie_vision" / "reid" / "__init__.py",
        "from .osnet import osnet_x1_0\nfrom . import osnet\nimport walkie_vision.reid.osnet as net\n",
    )
    assert _imports_a_module_that_left(loader, f"{REID}.osnet") == [
        "__init__.py:1: imports from walkie_vision.reid.osnet",
        "__init__.py:2: imports walkie_vision.reid.osnet",
        "__init__.py:3: imports walkie_vision.reid.osnet",
    ]
    # The modules read as attributes of the package they left, under any name.
    reader = _write(
        tmp_path / "tests" / "test_heads.py",
        "from walkie_vision import face, reid as reidentification\n"
        "real = face.face_align.norm_crop\n"
        'monkeypatch.setattr(face, "arcface", fake)\n'
        "net = reidentification.osnet.osnet_x1_0()\n",
    )
    assert _reads_a_name_that_left(reader, FACE, LEFT_FACE) + _reads_a_name_that_left(reader, REID, LEFT_REID) == [
        "test_heads.py:2: reads face.face_align",
        "test_heads.py:3: names arcface",
        "test_heads.py:4: reads reidentification.osnet",
    ]
    # A face or reid __init__ that binds a vendored module puts it back at its old home.
    init = _write(
        tmp_path / "face_init.py",
        "from .._vendor.insightface import face_align\n"
        "from .._vendor.insightface import scrfd as scrfd\n"
        "from .._vendor.insightface.arcface import *\n",
    )
    assert _binds_one_of(init, LEFT_FACE) == [
        "face_init.py:1: binds face_align",
        "face_init.py:2: binds scrfd",
        "face_init.py:3: binds *",
    ]
    # The new homes, a sibling import inside the vendored package, the face wrapper in
    # buffalo_l.py, the reid wrapper in the provider osnet.py and a docstring are fine.
    good = {
        "_vendor/insightface/arcface.py": "from . import face_align\n",
        "face/analysis.py": (
            '"""Wraps what was walkie_vision.face.scrfd."""\n'
            "from .._vendor.insightface.arcface import ArcFaceONNX\n"
            "from .._vendor.insightface.scrfd import SCRFD\n"
        ),
        "face/__init__.py": "from .analysis import Face, FaceAnalyzer\nfrom .heads import HeadFaceRecognizer\n",
        "providers/face_recognition/buffalo_l.py": (
            "from ..._vendor.insightface import face_align\n"
            "from ..._vendor.insightface.arcface import ArcFaceONNX\n"
            "from ..._vendor.insightface.scrfd import SCRFD\n"
        ),
        "reid/__init__.py": "from .._vendor.torchreid.osnet import osnet_x1_0\nfrom ..weights import HUB_MODELS\n",
        "providers/reid/osnet.py": (
            "from . import base\n"
            "from ..._vendor.torchreid import osnet\n"
            "from ..._vendor.torchreid.osnet import osnet_x1_0\n"
        ),
    }
    for path, text in good.items():
        file = _write(tmp_path / "src" / "walkie_vision" / path, text)
        assert [hit for old in VENDORED for hit in _imports_a_module_that_left(file, old)] == [], path
        assert _reads_a_name_that_left(file, FACE, LEFT_FACE) == [], path
        assert _reads_a_name_that_left(file, REID, LEFT_REID) == [], path
    assert _binds_one_of(tmp_path / "src" / "walkie_vision" / "face" / "__init__.py", LEFT_FACE) == []
    assert _binds_one_of(tmp_path / "src" / "walkie_vision" / "reid" / "__init__.py", LEFT_REID) == []


def test_yolo_did_not_come_back_to_the_package_root():
    providers = PACKAGE_DIR / "providers"
    for path in ("_ultralytics.py", "object_detection/yoloe.py", "pose_estimation/yolo_pose.py"):
        assert (providers / path).is_file()
    assert _old_homes(SRC, YOLO) == []


def test_no_code_imports_yolo():
    code = [file for file in _files_under(*CODE_AND_BUILD) if file.suffix == ".py"]
    assert {
        PACKAGE_DIR / "node.py",
        ROOT / "checks" / "c11_providers.py",
        PACKAGE_DIR / "loading.py",
        ROOT / "tests" / "test_loading.py",
        ROOT / "tests" / "test_providers_ultralytics.py",
    } <= set(code)
    assert [hit for file in code for hit in _imports_a_module_that_left(file, YOLO)] == []


def test_no_code_takes_the_shared_helpers_or_the_pose_from_yoloe():
    code = [file for file in _files_under(*CODE_AND_BUILD) if file.suffix == ".py"]
    assert {
        PACKAGE_DIR / "node.py",
        PACKAGE_DIR / "providers" / "pose_estimation" / "yolo_pose.py",
        PACKAGE_DIR / "loading.py",
        ROOT / "tests" / "test_loading.py",
        ROOT / "tests" / "test_providers_ultralytics.py",
    } <= set(code)
    # yolo.py is yoloe.py now: it serves none of the names that left it, nor does the
    # pose module serve the helpers, and the helpers' old private names are gone.
    assert [hit for file in code for hit in _reads_a_name_that_left(file, YOLOE, LEFT_YOLOE)] == []
    shared = SHARED_HELPERS | PRIVATE_HELPERS
    assert [hit for file in code for hit in _reads_a_name_that_left(file, YOLO_POSE, shared)] == []
    assert [hit for file in code for hit in _reads_a_name_that_left(file, ULTRALYTICS, PRIVATE_HELPERS)] == []


def test_yoloe_and_yolo_pose_do_not_define_the_shared_helpers_again():
    providers = PACKAGE_DIR / "providers"
    yoloe, yolo_pose = providers / "object_detection" / "yoloe.py", providers / "pose_estimation" / "yolo_pose.py"
    # Both import the helpers they use from _ultralytics but define none of them, and
    # yoloe.py imports neither the ones it no longer uses nor the pose.
    assert _binds_one_of(yoloe, LEFT_YOLOE, imports=False) == []
    assert _binds_one_of(yoloe, UNUSED_IN_YOLOE) == []
    assert _binds_one_of(yolo_pose, SHARED_HELPERS | PRIVATE_HELPERS, imports=False) == []
    assert _binds_one_of(yolo_pose, PRIVATE_HELPERS | {"TRACKER_NAMES"}) == []
    # _ultralytics.py defines every shared helper, under its public name only.
    ultralytics = _binds_one_of(providers / "_ultralytics.py", SHARED_HELPERS | PRIVATE_HELPERS)
    assert {hit.rpartition(" ")[2] for hit in ultralytics} == SHARED_HELPERS


def test_the_sweep_would_catch_yolo_at_its_old_home(tmp_path):
    bad = _write(
        tmp_path / "bad.py",
        "from walkie_vision.yolo import YoloeTracker, YoloPoseTracker\n"
        "import walkie_vision.yolo\n"
        'yolo = pytest.importorskip("walkie_vision.yolo")\n'
        "from walkie_vision import errors, node, yolo\n"
        "from walkie_vision import yolo as yolo_module\n"
        "from . import boxes, yolo\n"
        "from .yolo import MOBILECLIP_FILE\n"
        "from ..yolo import TRACKER_NAMES\n"
        "| `src/walkie_vision/yolo.py`         | YOLOE-26 and YOLO26 pose with their trackers |\n"
        "one process for every model (ADR-0010). `yolo.py` and `face/heads.py` are\n"
        "`c11` and the tracker tests in `tests/test_yolo.py` again after one.\n"
        "# walkie_vision.yolo.TRACKER_NAMES, which this module does not import.\n"
        "`YoloeDetector` in `walkie_vision/yolo` |\n",
    )
    assert len(_mentions([bad], REMOVED)) == 13
    # The new homes, the sibling provider modules, the weight file names, the
    # Ultralytics classes and environment variables, and the dotted parameter names
    # of a later step are fine.
    good = _write(
        tmp_path / "good.py",
        "from walkie_vision.providers.object_detection.yoloe import YoloeDetector, YoloeTracker\n"
        "from walkie_vision.providers.pose_estimation.yolo_pose import YoloPoseTracker\n"
        "from walkie_vision.providers import _ultralytics\n"
        'yoloe = pytest.importorskip("walkie_vision.providers.object_detection.yoloe")\n'
        "from .._ultralytics import check_rows, clip_box, require, reset_trackers, validate_tracker\n"
        "from . import yoloe, yolo_pose\n"
        "from .yolo_pose import poses_from_arrays\n"
        "| `src/walkie_vision/providers/object_detection/yoloe.py` | YOLOE-26 with its tracker |\n"
        "`c11` and the tracker tests in `tests/test_providers_ultralytics.py` again after one.\n"
        'objects_weights: "yoloe-26s-seg.pt", pose_weights: "yolo26s-pose.pt"\n'
        "from ultralytics import YOLO, YOLOE\n"
        'os.environ.setdefault("YOLO_OFFLINE", "true")\n'
        "| `object_detection.yoloe.weights` | `pose_estimation.yolo_pose.tracker` |\n",
    )
    assert _mentions([good], REMOVED) == []


def test_the_tree_check_would_catch_yolo_or_its_names_at_their_old_homes(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "SRC", tmp_path / "src")
    # The module that left: yolo, in any spelling of an import.
    caller = _write(
        tmp_path / "src" / "walkie_vision" / "node.py",
        "from . import (\n"
        "    errors,\n"
        "    yolo,\n"
        ")\n"
        "from .yolo import MOBILECLIP_FILE\n"
        "import walkie_vision.yolo as detectors\n"
        "from walkie_vision import yolo as yolo_module\n"
        "def build_providers():\n"
        "    from walkie_vision.yolo import YoloeTracker\n"
        "    return YoloeTracker\n",
    )
    assert _imports_a_module_that_left(caller, YOLO) == [
        "node.py:1: imports walkie_vision.yolo",
        "node.py:5: imports from walkie_vision.yolo",
        "node.py:6: imports walkie_vision.yolo",
        "node.py:7: imports walkie_vision.yolo",
        "node.py:9: imports from walkie_vision.yolo",
    ]
    # The names that left yolo.py, taken from its successor yoloe under any name, and
    # the helpers' old private names taken from their new home.
    reader = _write(
        tmp_path / "tests" / "test_build_providers.py",
        "from walkie_vision.providers.object_detection import yoloe as detector\n"
        "from walkie_vision.providers.object_detection.yoloe import (\n"
        "    MOBILECLIP_FILE,\n"
        "    TRACKER_NAMES,\n"
        ")\n"
        "assert node.TRACKER_FILES == detector.TRACKER_NAMES\n"
        'monkeypatch.setattr(detector, "YoloPoseTracker", fake)\n'
        "from walkie_vision.providers._ultralytics import _clip_box\n"
        "from walkie_vision.providers import _ultralytics\n"
        "_ultralytics._require(path)\n"
        'yoloe = pytest.importorskip("walkie_vision.providers.object_detection.yoloe")\n'
        "assert node.TRACKER_FILES == yoloe.TRACKER_NAMES\n"
        'skipped = importlib.import_module("walkie_vision.providers.object_detection.yoloe")\n'
        'monkeypatch.setattr(skipped, "_reset_trackers", fake)\n',
    )
    assert _reads_a_name_that_left(reader, YOLOE, LEFT_YOLOE) == [
        "test_build_providers.py:12: reads yoloe.TRACKER_NAMES",
        "test_build_providers.py:14: names _reset_trackers",
        "test_build_providers.py:2: imports TRACKER_NAMES",
        "test_build_providers.py:6: reads detector.TRACKER_NAMES",
        "test_build_providers.py:7: names YoloPoseTracker",
    ]
    assert _reads_a_name_that_left(reader, ULTRALYTICS, PRIVATE_HELPERS) == [
        "test_build_providers.py:10: reads _ultralytics._require",
        "test_build_providers.py:8: imports _clip_box",
    ]
    # A yoloe that defines, re-exports or star-imports a helper or the pose puts it
    # back at its old home.
    copy = _write(
        tmp_path / "yoloe.py",
        "from .._ultralytics import check_rows, clip_box\n"
        "from .._ultralytics import TRACKER_NAMES as TRACKER_NAMES\n"
        "def _require(path):\n"
        "    return path.is_file()\n"
        "from ..pose_estimation.yolo_pose import YoloPoseTracker\n"
        "from .._ultralytics import *\n",
    )
    assert _binds_one_of(copy, LEFT_YOLOE, imports=False) == [
        "yoloe.py:2: binds TRACKER_NAMES",
        "yoloe.py:3: binds _require",
        "yoloe.py:6: binds *",
    ]
    assert _binds_one_of(copy, UNUSED_IN_YOLOE) == [
        "yoloe.py:2: binds TRACKER_NAMES",
        "yoloe.py:3: binds _require",
        "yoloe.py:5: binds YoloPoseTracker",
        "yoloe.py:6: binds *",
    ]
    # The new homes, a module named yolo inside a subpackage, modules that only import
    # and use the helpers, and a docstring are fine.
    good = {
        "node.py": (
            '"""TRACKER_FILES copies walkie_vision.yolo.TRACKER_NAMES."""\n'
            "from .providers.object_detection.yoloe import MOBILECLIP_FILE\n"
            "from walkie_vision.providers.pose_estimation.yolo_pose import YoloPoseTracker\n"
        ),
        "providers/object_detection/yoloe.py": (
            "from ...boxes import resize_mask\n"
            "from .._ultralytics import check_rows, clip_box, require, reset_trackers, validate_tracker\n"
            "from .base import Detection\n"
        ),
        "providers/pose_estimation/yolo_pose.py": (
            "from .._ultralytics import check_rows, clip_box, require, reset_trackers, validate_tracker\n"
            "from .base import COCO_KEYPOINTS, PersonPose\n"
        ),
        "providers/object_detection/registry.py": "from . import yolo\nfrom .yolo import YoloeOptions\n",
    }
    for path, text in good.items():
        file = _write(tmp_path / "src" / "walkie_vision" / path, text)
        assert _imports_a_module_that_left(file, YOLO) == [], path
        assert _reads_a_name_that_left(file, YOLOE, LEFT_YOLOE) == [], path
        assert _reads_a_name_that_left(file, ULTRALYTICS, PRIVATE_HELPERS) == [], path
    providers = tmp_path / "src" / "walkie_vision" / "providers"
    assert _binds_one_of(providers / "object_detection" / "yoloe.py", LEFT_YOLOE, imports=False) == []
    assert _binds_one_of(providers / "pose_estimation" / "yolo_pose.py", SHARED_HELPERS, imports=False) == []


def test_face_did_not_come_back_to_the_package_root():
    assert BUFFALO_L_FILE.is_file()
    assert _old_homes(SRC, FACE) == []


def test_no_code_imports_face():
    code = [file for file in _files_under(*CODE_AND_BUILD) if file.suffix == ".py"]
    assert {
        PACKAGE_DIR / "node.py",
        ROOT / "checks" / "c08_face.py",
        ROOT / "checks" / "c11_providers.py",
        PACKAGE_DIR / "loading.py",
        ROOT / "tests" / "test_loading.py",
        ROOT / "tests" / "test_providers_face_recognition_buffalo_l.py",
    } <= set(code)
    assert [hit for file in code for hit in _imports_a_module_that_left(file, FACE)] == []


def test_no_code_takes_the_face_model_file_names_from_buffalo_l():
    code = [file for file in _files_under(*CODE_AND_BUILD) if file.suffix == ".py"]
    assert {
        PACKAGE_DIR / "node.py",
        BUFFALO_L_FILE,
        PACKAGE_DIR / "loading.py",
        ROOT / "tests" / "test_loading.py",
        ROOT / "tests" / "test_providers_face_recognition_buffalo_l.py",
    } <= set(code)
    assert [hit for file in code for hit in _reads_a_name_that_left(file, BUFFALO_L, LEFT_BUFFALO_L)] == []


def test_buffalo_l_does_not_define_the_face_model_file_names_again():
    # buffalo_l.py imports both file names from weights.py, which defines them.
    assert _binds_one_of(BUFFALO_L_FILE, LEFT_BUFFALO_L, imports=False) == []
    weights = _binds_one_of(PACKAGE_DIR / "weights.py", LEFT_BUFFALO_L, imports=False)
    assert {hit.rpartition(" ")[2] for hit in weights} == LEFT_BUFFALO_L


def test_the_sweep_would_catch_face_at_its_old_home(tmp_path):
    bad = _write(
        tmp_path / "bad.py",
        "from walkie_vision.face import FaceAnalyzer, HeadCropConfig, HeadFaceRecognizer\n"
        "from walkie_vision.face.heads import box_iou  # noqa: E402\n"
        "import walkie_vision.face.analysis\n"
        'face = pytest.importorskip("walkie_vision.face")\n'
        "from walkie_vision import errors, face, node\n"
        "from walkie_vision import face as face_module\n"
        "from . import errors, face\n"
        "from .face import FaceAnalyzer\n"
        "from ..face.analysis import DETECTOR_FILE\n"
        "| `src/walkie_vision/face/heads.py`   | Faces from crops around each person's head |\n"
        "  by `face/` and `reid/`. Keep the licence headers and the change lists\n"
        "# walkie_vision.face.analysis (which imports onnxruntime).\n"
        "heads.py and analysis.py, in face/, wrap them\n"
        "Tests of tests/test_heads.py with a stub analyzer\n"
        "det_path = model_dir / analysis.DETECTOR_FILE\n"
        "rec_path = model_dir / buffalo_l.RECOGNIZER_FILE\n"
        "from walkie_vision.providers.face_recognition.buffalo_l import DETECTOR_FILE\n"
        "from .buffalo_l import FaceAnalyzer, RECOGNIZER_FILE\n",
    )
    assert len(_mentions([bad], REMOVED)) == 18
    # The new home, the names that stay in buffalo_l.py, the file names at weights.py,
    # the face recognition task, the vendored face_align, the word face, the check
    # c08_face.py and the dotted parameter names of a later step are fine.
    good = _write(
        tmp_path / "good.py",
        "from walkie_vision.providers.face_recognition.buffalo_l import FaceAnalyzer, HeadCropConfig\n"
        'buffalo_l = pytest.importorskip("walkie_vision.providers.face_recognition.buffalo_l")\n'
        "from walkie_vision.providers import face_recognition\n"
        "from . import base, buffalo_l\n"
        "from .base import Face\n"
        "from ..._vendor.insightface import face_align\n"
        "from ...weights import DETECTOR_FILE, RECOGNIZER_FILE\n"
        "from walkie_vision.weights import BUFFALO_L_DIR, DETECTOR_FILE\n"
        "providers = buffalo_l.DEFAULT_PROVIDERS\n"
        "| `src/walkie_vision/providers/face_recognition/buffalo_l.py` | SCRFD and ArcFace on ONNX Runtime |\n"
        "`providers/face_recognition/base.py` holds Face\n"
        "the head tests in `tests/test_providers_face_recognition_buffalo_l.py`\n"
        "# https://github.com/deepinsight/insightface/blob/668e8d4/python-package/insightface/utils/face_align.py\n"
        "face = Face(bbox=det[j, :4].copy(), kps=kpss[j].copy(), det_score=float(det[j, 4]))\n"
        'WARM_UP = "warming up the face models; the first run on this GPU can take minutes"\n'
        "python checks/c08_face.py\n"
        "| `face_recognition.buffalo_l.det_thresh` |\n",
    )
    assert _mentions([good], REMOVED) == []


def test_the_tree_check_would_catch_face_or_its_names_at_their_old_homes(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "SRC", tmp_path / "src")
    # The package that left, in any spelling of an import, absolute or relative.
    caller = _write(
        tmp_path / "src" / "walkie_vision" / "node.py",
        "from . import (\n"
        "    errors,\n"
        "    face,\n"
        ")\n"
        "from .face import FaceAnalyzer\n"
        "from .face.heads import box_iou\n"
        "import walkie_vision.face.analysis as analysis\n"
        "def build():\n"
        "    from walkie_vision.face import HeadFaceRecognizer\n"
        "    return HeadFaceRecognizer\n",
    )
    assert _imports_a_module_that_left(caller, FACE) == [
        "node.py:1: imports walkie_vision.face",
        "node.py:5: imports from walkie_vision.face",
        "node.py:6: imports from walkie_vision.face.heads",
        "node.py:7: imports walkie_vision.face.analysis",
        "node.py:9: imports from walkie_vision.face",
    ]
    sibling = _write(tmp_path / "src" / "walkie_vision" / "face" / "heads.py", "from .analysis import FaceAnalyzer\n")
    assert _imports_a_module_that_left(sibling, FACE) == ["heads.py:1: imports from walkie_vision.face.analysis"]
    check = _write(tmp_path / "checks" / "c11_providers.py", "from walkie_vision import (\n    face as faces,\n)\n")
    assert _imports_a_module_that_left(check, FACE) == ["c11_providers.py:1: imports walkie_vision.face"]
    # A face package or namespace folder at the old path.
    assert _old_homes(tmp_path / "src", FACE) == [tmp_path / "src" / "walkie_vision" / "face"]
    # The file names read from buffalo_l, under any name for the module.
    reader = _write(
        tmp_path / "tests" / "test_loading.py",
        "from walkie_vision.providers.face_recognition import buffalo_l as provider\n"
        "from walkie_vision.providers.face_recognition.buffalo_l import (\n"
        "    DETECTOR_FILE,\n"
        "    FaceAnalyzer,\n"
        ")\n"
        "rec = provider.RECOGNIZER_FILE\n"
        'monkeypatch.setattr(provider, "DETECTOR_FILE", "other.onnx")\n'
        'loaded = pytest.importorskip("walkie_vision.providers.face_recognition.buffalo_l")\n'
        "det = loaded.DETECTOR_FILE\n",
    )
    assert _reads_a_name_that_left(reader, BUFFALO_L, LEFT_BUFFALO_L) == [
        "test_loading.py:2: imports DETECTOR_FILE",
        "test_loading.py:6: reads provider.RECOGNIZER_FILE",
        "test_loading.py:7: names DETECTOR_FILE",
        "test_loading.py:9: reads loaded.DETECTOR_FILE",
    ]
    registry = _write(
        tmp_path / "src" / "walkie_vision" / "providers" / "face_recognition" / "registry.py",
        "from .buffalo_l import DETECTOR_FILE\nfrom . import buffalo_l\nrec = buffalo_l.RECOGNIZER_FILE\n",
    )
    assert _reads_a_name_that_left(registry, BUFFALO_L, LEFT_BUFFALO_L) == [
        "registry.py:1: imports DETECTOR_FILE",
        "registry.py:3: reads buffalo_l.RECOGNIZER_FILE",
    ]
    # A buffalo_l.py that defines or re-exports the file names puts them back.
    copy = _write(
        tmp_path / "copy" / "buffalo_l.py",
        'DETECTOR_FILE = "det_10g.onnx"\n'
        "from ...weights import RECOGNIZER_FILE as RECOGNIZER_FILE\n"
        "from ...weights import *\n",
    )
    assert _binds_one_of(copy, LEFT_BUFFALO_L, imports=False) == [
        "buffalo_l.py:1: binds DETECTOR_FILE",
        "buffalo_l.py:2: binds RECOGNIZER_FILE",
        "buffalo_l.py:3: binds *",
    ]
    # The new homes, a module named face inside another package, the vendored
    # face_align, modules that import the file names from weights, and a docstring are fine.
    good = {
        "node.py": (
            '"""_FACE_FILES copies what walkie_vision.face.analysis named."""\n'
            "from . import errors, weights\n"
            "from .providers.face_recognition.buffalo_l import FaceAnalyzer\n"
            "from .weights import DETECTOR_FILE\n"
        ),
        "providers/face_recognition/buffalo_l.py": (
            "from ..._vendor.insightface import face_align\n"
            "from ...weights import DETECTOR_FILE, RECOGNIZER_FILE\n"
            "from . import base\n"
            "from .base import Face\n"
        ),
        "providers/face_recognition/registry.py": (
            "from ...weights import DETECTOR_FILE, RECOGNIZER_FILE\n"
            "from . import buffalo_l\n"
            "providers = buffalo_l.DEFAULT_PROVIDERS\n"
        ),
        "_vendor/insightface/arcface.py": "from . import face_align\n",
        "camera/image.py": "from . import face\nfrom .face import crop\n",
    }
    for path, text in good.items():
        file = _write(tmp_path / "src" / "walkie_vision" / path, text)
        assert _imports_a_module_that_left(file, FACE) == [], path
        assert _reads_a_name_that_left(file, BUFFALO_L, LEFT_BUFFALO_L) == [], path
    good_provider = tmp_path / "src" / "walkie_vision" / "providers" / "face_recognition" / "buffalo_l.py"
    assert _binds_one_of(good_provider, LEFT_BUFFALO_L, imports=False) == []


def test_reid_did_not_come_back_to_the_package_root():
    assert OSNET_FILE.is_file()
    assert _old_homes(SRC, REID) == []
    # The reid vision task's package shares the old package's last name; it serves
    # none of the provider's names, which callers import from osnet.py.
    assert _binds_one_of(PACKAGE_DIR / "providers" / "reid" / "__init__.py", LEFT_REID_PACKAGE) == []


def test_no_code_imports_reid():
    code = [file for file in _files_under(*CODE_AND_BUILD) if file.suffix == ".py"]
    assert {
        PACKAGE_DIR / "node.py",
        PACKAGE_DIR / "providers" / "__init__.py",
        ROOT / "checks" / "c09_osnet.py",
        ROOT / "checks" / "c11_providers.py",
        PACKAGE_DIR / "loading.py",
        ROOT / "tests" / "test_loading.py",
        ROOT / "tests" / "test_providers_reid_osnet.py",
    } <= set(code)
    assert [hit for file in code for hit in _imports_a_module_that_left(file, REID)] == []


def test_the_sweep_would_catch_reid_at_its_old_home(tmp_path):
    bad = _write(
        tmp_path / "bad.py",
        "from walkie_vision.reid import INPUT_SIZE, MEAN, STD, load_osnet\n"
        "from walkie_vision.reid.identifier import OSNetReIdentifier  # noqa: E402\n"
        "import walkie_vision.reid.identifier\n"
        'identifier = pytest.importorskip("walkie_vision.reid.identifier")\n'
        "from walkie_vision import errors, node, reid\n"
        "from walkie_vision import reid as reidentification\n"
        "| `src/walkie_vision/reid/identifier.py` | OSNet on person crops, batched |\n"
        "  by `providers/face_recognition/buffalo_l.py` and `reid/identifier.py`. Keep the licence\n"
        "reid/__init__.py holds code today\n"
        '"""The tests in tests/test_reid_identifier.py use a tiny CPU module."""\n'
        "real = identifier.load_osnet\n"
        "model = reid.load_osnet('cpu')\n"
        "h, w = identifier_module.INPUT_SIZE\n"
        "pixi run python -c \"from walkie_vision.reid import load_osnet; load_osnet('cpu')\"\n",
    )
    assert len(_mentions([bad], REMOVED)) == 14
    # The new home, the reid vision task, its folder and its base module, the vendored
    # torchreid, deep-person-reid, the reid stage and provider field, the reid test file,
    # and the dotted parameter names of a later step are fine.
    good = _write(
        tmp_path / "good.py",
        "from walkie_vision.providers.reid.osnet import INPUT_SIZE, MEAN, STD, load_osnet\n"
        'osnet = pytest.importorskip("walkie_vision.providers.reid.osnet")\n'
        "from walkie_vision.providers.reid import osnet\n"
        "from walkie_vision.providers import OSNET_X1_0_MSMT17, ReIdentifier\n"
        "from .reid.base import ReIdentifier\n"
        "from . import base\n"
        "from ..._vendor.torchreid.osnet import osnet_x1_0\n"
        'monkeypatch.setattr(osnet, "load_osnet", load_osnet)\n'
        "| `src/walkie_vision/providers/reid/osnet.py` | OSNet re-identification and its weights |\n"
        "`providers/reid/base.py` holds ReIdentifier\n"
        "`providers/` holds one folder per vision task: `object_detection/`, `reid/`, `image_embed/`\n"
        "      reid/\n"
        "| `reid/` | the reid vision task: base.py, osnet.py |\n"
        "a new reid provider goes in reid/ next to osnet.py\n"
        "deep-person-reid in `_vendor/torchreid/`, wrapped by the reid provider\n"
        "# https://github.com/KaiyangZhou/deep-person-reid/blob/f8cd150/torchreid/models/osnet.py\n"
        'reid = OSNetReIdentifier(device="cpu", model=_Tiny())\n'
        "vectors = np.asarray(reid.embed(crops))\n"
        "times.reid = _ms_since(t)\n"
        'appearance_model=p.reid.model_name if vector is not None else "",\n'
        "ms p50/p95: wait 3/9 convert 6/8 objects 21/30 pose 18/25 faces 12/40 reid 9/15 build 1/2\n"
        "tests/test_providers_reid_osnet.py patches load_osnet\n"
        "| `reid.osnet.batch_size` |\n",
    )
    assert _mentions([good], REMOVED) == []


def test_the_tree_check_would_catch_reid_at_its_old_home(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "SRC", tmp_path / "src")
    # The package that left, in any spelling of an import, absolute or relative.
    caller = _write(
        tmp_path / "src" / "walkie_vision" / "node.py",
        "from . import (\n"
        "    errors,\n"
        "    reid,\n"
        ")\n"
        "from .reid import load_osnet\n"
        "from .reid.identifier import OSNetReIdentifier\n"
        "import walkie_vision.reid.identifier as identifier\n"
        "def build():\n"
        "    from walkie_vision.reid.identifier import OSNetReIdentifier\n"
        "    return OSNetReIdentifier\n",
    )
    assert _imports_a_module_that_left(caller, REID) == [
        "node.py:1: imports walkie_vision.reid",
        "node.py:5: imports from walkie_vision.reid",
        "node.py:6: imports from walkie_vision.reid.identifier",
        "node.py:7: imports walkie_vision.reid.identifier",
        "node.py:9: imports from walkie_vision.reid.identifier",
    ]
    sibling = _write(
        tmp_path / "src" / "walkie_vision" / "reid" / "identifier.py",
        "from . import INPUT_SIZE, MEAN, STD, load_osnet\n",
    )
    assert _imports_a_module_that_left(sibling, REID) == ["identifier.py:1: imports from walkie_vision.reid"]
    check = _write(
        tmp_path / "checks" / "c09_osnet.py",
        "from walkie_vision import (\n    reid as reidentification,\n)\n",
    )
    assert _imports_a_module_that_left(check, REID) == ["c09_osnet.py:1: imports walkie_vision.reid"]
    # A reid package, namespace folder or module at the old path.
    assert _old_homes(tmp_path / "src", REID) == [tmp_path / "src" / "walkie_vision" / "reid"]
    module = _write(tmp_path / "flat" / "walkie_vision" / "reid.py", "from .providers.reid.osnet import *\n")
    assert _old_homes(tmp_path / "flat", REID) == [module]
    # A reid task package that re-exports the provider's names serves them under a
    # second name.
    init = _write(
        tmp_path / "reid_init.py",
        "from .osnet import OSNetReIdentifier\nfrom .osnet import load_osnet as load_osnet\nfrom .osnet import *\n",
    )
    assert _binds_one_of(init, LEFT_REID_PACKAGE) == [
        "reid_init.py:1: binds OSNetReIdentifier",
        "reid_init.py:2: binds load_osnet",
        "reid_init.py:3: binds *",
    ]
    # The new home, the reid vision task's package and base module from inside and
    # outside providers/, a module named reid inside another package, the vendored
    # torchreid and a docstring are fine.
    good = {
        "node.py": (
            '"""OSNetReIdentifier was in walkie_vision.reid.identifier."""\n'
            "from .providers import reid\n"
            "from .providers.reid.osnet import OSNetReIdentifier\n"
        ),
        "providers/__init__.py": "from .reid.base import ReIdentifier\n",
        "providers/reid/osnet.py": (
            "from ..._vendor.torchreid.osnet import osnet_x1_0\n"
            "from ...weights import HUB_MODELS, OSNET_CHECKPOINT\n"
            "from ..vector_tables import OSNET_X1_0_MSMT17\n"
            "from . import base\n"
            "from .base import ReIdentifier\n"
        ),
        "providers/reid/registry.py": "from . import osnet\nfrom .. import reid\nfrom .osnet import load_osnet\n",
        "_vendor/torchreid/__init__.py": '"""Wrapped by walkie_vision.providers.reid.osnet."""\n',
        "camera/image.py": "from . import reid\nfrom .reid import crop\n",
    }
    for path, text in good.items():
        file = _write(tmp_path / "src" / "walkie_vision" / path, text)
        assert _imports_a_module_that_left(file, REID) == [], path


def test_the_clip_embedder_did_not_come_back_to_embeddings():
    assert CLIP_FILE.is_file()
    assert _old_homes(SRC, EMBEDDINGS) == []
    # Neither the image_embed vision task's package nor the providers package serves
    # the CLIP provider's names: callers import them from clip.py.
    for package in ("providers/image_embed", "providers"):
        assert _binds_one_of(PACKAGE_DIR / package / "__init__.py", LEFT_EMBEDDINGS_CLIP) == [], package


def test_no_code_imports_embeddings():
    code = [file for file in _files_under(*CODE_AND_BUILD) if file.suffix == ".py"]
    assert {
        PACKAGE_DIR / "node.py",
        CLIP_FILE,
        PACKAGE_DIR / "loading.py",
        ROOT / "tests" / "test_loading.py",
        ROOT / "tests" / "test_providers_image_embed_clip.py",
    } <= set(code)
    assert [hit for file in code for hit in _imports_a_module_that_left(file, EMBEDDINGS)] == []


def test_the_sweep_would_catch_embeddings_at_its_old_home(tmp_path):
    bad = _write(
        tmp_path / "bad.py",
        "from walkie_vision.embeddings import ClipEmbedder, clip_checkpoint\n"
        "from .embeddings import ClipEmbedder, clip_checkpoint\n"
        "import walkie_vision.embeddings\n"
        "from walkie_vision import embeddings, errors, node  # noqa: E402\n"
        "from walkie_vision import embeddings as clip_module\n"
        "from . import convert, embeddings\n"
        "from .. import embeddings\n"
        "| `src/walkie_vision/embeddings.py`   | The interim CLIP provider |\n"
        "  padded crops and colour shares. `embeddings.py`: the interim CLIP provider.\n"
        '"""walkie_vision.embeddings: the CLIP provider, without a GPU or a model file."""\n'
        "**The model.** `walkie_vision.embeddings.ClipEmbedder` is the interim provider:\n"
        "git mv tests/test_embeddings.py tests/test_providers_image_embed_clip.py\n"
        "clip = mock.create_autospec(embeddings.ClipEmbedder)\n"
        "assert built.dim == embeddings_module.CLIP_DIM\n"
        "`embeddings:clip_checkpoint` reads the cache\n",
    )
    assert len(_mentions([bad], REMOVED)) == 15
    # The new home, the image_embed vision task, its base module, the vector model
    # name, the embed services, the parameter, the word embeddings, the bake-off's
    # embedders.py and the new test file are fine.
    good = _write(
        tmp_path / "good.py",
        "from walkie_vision.providers.image_embed.clip import ClipEmbedder, clip_checkpoint\n"
        "from .providers.image_embed.clip import ClipEmbedder, clip_checkpoint\n"
        "from walkie_vision.providers.image_embed import clip as clip_module\n"
        "from walkie_vision.providers import CLIP_VIT_B16, Embedder\n"
        "from .image_embed.base import Embedder, check_model_dim\n"
        "from ..vector_tables import CLIP_VIT_B16\n"
        "from .services import embed_text as embed_text_service\n"
        "from . import embed_crop, jobs\n"
        'monkeypatch.setattr(clip_module, "ClipEmbedder", clip)\n'
        "clip = mock.create_autospec(clip_module.ClipEmbedder)\n"
        "| `src/walkie_vision/providers/image_embed/clip.py` | The interim CLIP provider |\n"
        "`walkie_vision.providers.image_embed.clip.ClipEmbedder` is the interim provider\n"
        "`providers/image_embed/` holds the image embedding vision task\n"
        "    object_embeddings_enabled: false\n"
        "if not self.config.object_embeddings_enabled:\n"
        "self._embeddings_enabled = bool(object_embeddings_enabled)\n"
        'print(f"text embeddings {tuple(temb.shape)}, image embeddings {tuple(iemb.shape)}")\n'
        "normalize_embeddings=True,\n"
        "The CLIP vectors match tools/bakeoff/embedders.py\n"
        '"""walkie_vision.providers.image_embed.clip: the CLIP provider."""\n'
        "tests/test_providers_image_embed_clip.py stubs open_clip\n",
    )
    assert _mentions([good], REMOVED) == []


def test_the_tree_check_would_catch_embeddings_at_its_old_home(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "SRC", tmp_path / "src")
    # The module that left, in any spelling of an import, absolute or relative.
    caller = _write(
        tmp_path / "src" / "walkie_vision" / "node.py",
        "from . import (\n"
        "    errors,\n"
        "    embeddings,\n"
        ")\n"
        "from .embeddings import ClipEmbedder\n"
        "import walkie_vision.embeddings as clip\n"
        "def build_providers():\n"
        "    from .embeddings import ClipEmbedder, clip_checkpoint\n"
        "    return ClipEmbedder, clip_checkpoint\n",
    )
    assert _imports_a_module_that_left(caller, EMBEDDINGS) == [
        "node.py:1: imports walkie_vision.embeddings",
        "node.py:5: imports from walkie_vision.embeddings",
        "node.py:6: imports walkie_vision.embeddings",
        "node.py:8: imports from walkie_vision.embeddings",
    ]
    inside = _write(
        tmp_path / "src" / "walkie_vision" / "providers" / "image_embed" / "registry.py",
        "from ... import embeddings\nfrom ...embeddings import CLIP_DIM\n",
    )
    assert _imports_a_module_that_left(inside, EMBEDDINGS) == [
        "registry.py:1: imports walkie_vision.embeddings",
        "registry.py:2: imports from walkie_vision.embeddings",
    ]
    check = _write(
        tmp_path / "checks" / "c07_clip.py",
        "from walkie_vision import (\n    embeddings as clip_module,\n)\n",
    )
    assert _imports_a_module_that_left(check, EMBEDDINGS) == ["c07_clip.py:1: imports walkie_vision.embeddings"]
    # A module, a package or a namespace folder at the old path.
    folder = tmp_path / "src" / "walkie_vision" / "embeddings"
    _write(folder / "clip.py", "CLIP_DIM = 512\n")
    assert _old_homes(tmp_path / "src", EMBEDDINGS) == [folder]
    module = _write(
        tmp_path / "flat" / "walkie_vision" / "embeddings.py", "from .providers.image_embed.clip import *\n"
    )
    assert _old_homes(tmp_path / "flat", EMBEDDINGS) == [module]
    # An image_embed package that re-exports the provider's names serves them under a
    # second name.
    init = _write(
        tmp_path / "image_embed_init.py",
        "from .clip import ClipEmbedder\nfrom .clip import clip_checkpoint as clip_checkpoint\nfrom .clip import *\n",
    )
    assert _binds_one_of(init, LEFT_EMBEDDINGS_CLIP) == [
        "image_embed_init.py:1: binds ClipEmbedder",
        "image_embed_init.py:2: binds clip_checkpoint",
        "image_embed_init.py:3: binds *",
    ]
    # The new home from inside and outside providers/, the image_embed vision task's
    # package and base module, a module named embeddings inside another package and a
    # docstring are fine.
    good = {
        "node.py": (
            '"""ClipEmbedder was in walkie_vision.embeddings."""\n'
            "from .providers import image_embed\n"
            "from .providers.image_embed.clip import ClipEmbedder, clip_checkpoint\n"
        ),
        "providers/__init__.py": "from .image_embed.base import Embedder\n",
        "providers/image_embed/clip.py": (
            "from ...weights import CLIP_WEIGHTS, HUB_MODELS\nfrom ..vector_tables import CLIP_VIT_B16\n"
        ),
        "providers/image_embed/registry.py": (
            "from . import clip\nfrom .base import Embedder\nfrom .clip import clip_checkpoint\n"
        ),
        "services/embed_text.py": "from . import jobs\nfrom ..providers.image_embed.base import Embedder\n",
        "crops/objects.py": "from . import embeddings\nfrom .embeddings import CLIP_DIM\n",
    }
    for path, text in good.items():
        file = _write(tmp_path / "src" / "walkie_vision" / path, text)
        assert _imports_a_module_that_left(file, EMBEDDINGS) == [], path


def test_the_config_and_the_loading_have_their_own_modules():
    config = _binds_one_of(CONFIG_FILE, LEFT_NODE_CONFIG, imports=False)
    assert {hit.rpartition(" ")[2] for hit in config} == LEFT_NODE_CONFIG
    loading = _binds_one_of(LOADING_FILE, LEFT_NODE_LOADING, imports=False)
    assert {hit.rpartition(" ")[2] for hit in loading} == LEFT_NODE_LOADING


def test_no_code_takes_the_config_or_the_loading_from_the_node():
    code = [file for file in _files_under(*CODE_AND_BUILD) if file.suffix == ".py"]
    assert {
        PACKAGE_DIR / "node.py",
        ROOT / "checks" / "c11_providers.py",
        ROOT / "checks" / "c12_memory.py",
        ROOT / "checks" / "c12_vision_node.py",
        ROOT / "tests" / "test_loading.py",
        ROOT / "tests" / "test_node.py",
        ROOT / "tests" / "test_launch.py",
    } <= set(code)
    left = LEFT_NODE_CONFIG | LEFT_NODE_LOADING
    assert [hit for file in code for hit in _reads_a_name_that_left(file, NODE, left, patched=PATCHED_ON_NODE)] == []
    # The node calls build_providers by its bare name, so test_node.py patches it there.
    assert any(
        hit.endswith(": names build_providers")
        for hit in _reads_a_name_that_left(ROOT / "tests" / "test_node.py", NODE, left)
    )


def test_the_node_does_not_define_the_config_or_the_loading_again():
    # node.py imports VisionConfig and build_providers, since it uses them, but defines
    # neither, and does not bind the other names at all.
    node = PACKAGE_DIR / "node.py"
    assert _binds_one_of(node, USED_BY_NODE, imports=False) == []
    assert _binds_one_of(node, (LEFT_NODE_CONFIG | LEFT_NODE_LOADING) - USED_BY_NODE) == []


def test_the_sweep_would_catch_the_config_or_the_loading_at_their_old_home(tmp_path):
    bad = _write(
        tmp_path / "bad.py",
        "from walkie_vision.node import VisionConfig, build_providers\n"
        "from .node import StartupLogger\n"
        "assert node.TRACKER_FILES == _ultralytics.TRACKER_NAMES\n"
        "for name in node._FACE_FILES:\n"
        "if window > node_module.MAX_PAIR_WINDOW_S:\n"
        "`walkie_vision.node.VisionConfig` checks every value\n"
        "    ROOT / 'tests' / 'test_build_providers.py',\n"
        "`walkie_vision/node:_load` times each model\n"
        "`walkie_vision.node.build_providers` loads the models\n",
    )
    assert len(_mentions([bad], REMOVED)) == 9
    # The new homes, the node's names that stay, a patch of build_providers on the node
    # by the module or by its dotted path, a parameter of the config, the renamed test
    # file and a call are fine.
    good = _write(
        tmp_path / "good.py",
        "from walkie_vision.config import TRACKER_FILES, VisionConfig\n"
        "from walkie_vision.loading import StartupLogger, build_providers\n"
        "from .config import VisionConfig\n"
        "from .loading import build_providers\n"
        "from walkie_vision.node import NODE_NAME, VisionNode\n"
        'monkeypatch.setattr(node_module, "build_providers", fake_build)\n'
        'monkeypatch.setattr("walkie_vision.node.build_providers", fake_build)\n'
        "with mock.patch('walkie_vision.node.build_providers', fake_build):\n"
        "tracker = config.objects_tracker\n"
        "for name in loading._FACE_FILES:\n"
        "    ROOT / 'tests' / 'test_loading.py',\n"
        "providers = build_providers(config, self.get_logger())\n",
    )
    assert _mentions([good], REMOVED) == []


def test_the_tree_check_would_catch_the_config_or_the_loading_at_their_old_home(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "SRC", tmp_path / "src")
    left = LEFT_NODE_CONFIG | LEFT_NODE_LOADING
    # The names, taken from the node under any name.
    reader = _write(
        tmp_path / "src" / "walkie_vision" / "fake_publisher.py",
        "from . import node\n"
        "from walkie_vision import node as node_module\n"
        "from .node import (\n"
        "    NODE_NAME,\n"
        "    VisionConfig,\n"
        ")\n"
        "a = node._FACE_FILES\n"
        'monkeypatch.setattr(node_module, "TRACKER_FILES", ("x.yaml",))\n'
        'monkeypatch.setattr(node_module, "build_providers", fake_build)\n'
        "from .node import *\n",
    )
    assert _reads_a_name_that_left(reader, NODE, left, patched=PATCHED_ON_NODE) == [
        "fake_publisher.py:10: imports *",
        "fake_publisher.py:3: imports VisionConfig",
        "fake_publisher.py:7: reads node._FACE_FILES",
        "fake_publisher.py:8: names TRACKER_FILES",
    ]
    # Without `patched`, the patch of build_providers on the node counts too.
    assert "fake_publisher.py:9: names build_providers" in _reads_a_name_that_left(reader, NODE, left)
    # Importing or reading build_providers from the node counts even with `patched`.
    importer = _write(
        tmp_path / "src" / "walkie_vision" / "debug_viz.py",
        "from . import node\nfrom .node import build_providers\nb = node.build_providers\n",
    )
    assert _reads_a_name_that_left(importer, NODE, left, patched=PATCHED_ON_NODE) == [
        "debug_viz.py:2: imports build_providers",
        "debug_viz.py:3: reads node.build_providers",
    ]
    # A node that defines, re-exports or star-imports one of them puts it back at its
    # old home.
    copy = _write(
        tmp_path / "node.py",
        "from .config import VisionConfig\n"
        "from .loading import build_providers as build_providers\n"
        "from .loading import _FACE_FILES\n"
        "MAX_PAIR_WINDOW_S = 0.1\n"
        "class StartupLogger: ...\n"
        "from .config import *\n",
    )
    assert _binds_one_of(copy, USED_BY_NODE, imports=False) == [
        "node.py:2: binds build_providers",
        "node.py:6: binds *",
    ]
    assert _binds_one_of(copy, left - USED_BY_NODE) == [
        "node.py:3: binds _FACE_FILES",
        "node.py:4: binds MAX_PAIR_WINDOW_S",
        "node.py:5: binds StartupLogger",
        "node.py:6: binds *",
    ]
    # A node that only imports and uses VisionConfig and build_providers, and the new
    # homes, are fine.
    good = _write(
        tmp_path / "src" / "walkie_vision" / "node.py",
        '"""VisionConfig was in walkie_vision.node."""\n'
        "from .config import VisionConfig\n"
        "from .loading import build_providers\n"
        "config = VisionConfig.from_node(self)\n"
        "providers = build_providers(config, self.get_logger())\n",
    )
    assert _binds_one_of(good, USED_BY_NODE, imports=False) == []
    assert _binds_one_of(good, left - USED_BY_NODE) == []
    assert _reads_a_name_that_left(good, NODE, left) == []
    loading = _write(
        tmp_path / "src" / "walkie_vision" / "loading.py",
        "from .config import VisionConfig\n"
        "from . import config\n"
        "a = config.TRACKER_FILES\n",
    )
    assert _reads_a_name_that_left(loading, NODE, left) == []
