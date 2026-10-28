"""Model files that are not fetched on first use (`pixi run fetch-models`).

Each file comes from a fixed release URL, is checked against its SHA-256 and
is kept in models/ at the repository root (WALKIE_VISION_MODELS overrides the
location). A file that is already there with the right checksum is not
downloaded again, so the task also passes offline once it has run.

Hugging Face models (SigLIP 2, Florence-2, EmbeddingGemma 2, OSNet, CLIP) are
not downloaded here: they download into the Hugging Face cache on first use.
HUB_MODELS pins each of them to one commit.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

BUFFALO_L_DIR = "buffalo_l"
ULTRALYTICS_DIR = "ultralytics"
_ULTRALYTICS_RELEASE = "https://github.com/ultralytics/assets/releases/download/v8.4.0"


@dataclass(frozen=True)
class Download:
    """One file to download, and where it goes under models/."""

    url: str
    sha256: str
    size: int
    dest: str
    """A file path under models/, or a directory for a zip archive."""
    licence: str
    members: dict[str, str] = field(default_factory=dict)
    """For a zip archive: the SHA-256 of each file in it."""


DOWNLOADS = (
    # InsightFace's buffalo_l pack. GitHub publishes no digest for this old
    # release asset; the checksum is that of the file InsightFace's own
    # downloader fetched from the model-zoo release on 2026-10-19, which is
    # byte-for-byte the v0.7 asset.
    Download(
        url="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        sha256="80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f",
        size=288_621_354,
        dest=BUFFALO_L_DIR,
        licence="InsightFace models: non-commercial research only",
        members={
            "det_10g.onnx": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
            "w600k_r50.onnx": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
            "1k3d68.onnx": "df5c06b8a0c12e422b2ed8947b8869faa4105387f199c477af038aa01f9a45cc",
            "2d106det.onnx": "f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf",
            "genderage.onnx": "4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb",
        },
    ),
    # Ultralytics would download these into the working directory on first
    # use; fetching them here pins the release and checks the digests GitHub
    # publishes for it. mobileclip2_b.ts is the text encoder that
    # YOLOE.set_classes() loads by file name.
    Download(
        url=f"{_ULTRALYTICS_RELEASE}/yoloe-26s-seg.pt",
        sha256="48f24206bc8680d60cbbfa296b0140da849669b9515058b72f5a945142df0654",
        size=31_072_171,
        dest=f"{ULTRALYTICS_DIR}/yoloe-26s-seg.pt",
        licence="AGPL-3.0 (Ultralytics)",
    ),
    Download(
        url=f"{_ULTRALYTICS_RELEASE}/yolo26s-pose.pt",
        sha256="a083adb42303728ae14c4bd6bd56d80da46f82fb2564dbd6f31dcc92ea321646",
        size=24_151_790,
        dest=f"{ULTRALYTICS_DIR}/yolo26s-pose.pt",
        licence="AGPL-3.0 (Ultralytics)",
    ),
    Download(
        url=f"{_ULTRALYTICS_RELEASE}/mobileclip2_b.ts",
        sha256="35d7f213e4d75f38514e4656ad3cb91158bd33e3805d8ac349f23b186f66982f",
        size=253_794_476,
        dest=f"{ULTRALYTICS_DIR}/mobileclip2_b.ts",
        licence="Apple ML Research Model licence (MobileCLIP2-B): non-commercial research only",
    ),
)


@dataclass(frozen=True)
class HubModel:
    """A Hugging Face Hub repository, pinned to one commit."""

    repo: str
    revision: str
    licence: str


# Pass `revision` whenever one of these is loaded, so that a new upload to the
# repository cannot change the results without a change here.
HUB_MODELS = {
    "siglip2": HubModel("google/siglip2-base-patch16-256", "3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab", "Apache-2.0"),
    # The transformers-native conversion of microsoft/Florence-2-base, which
    # needs no trust_remote_code.
    "florence2": HubModel("florence-community/Florence-2-base", "00921df66db728a9ceb750f5eca43e5c203a2051", "MIT"),
    "embeddinggemma2": HubModel("google/embeddinggemma-2", "914f7f89142e33e77833254d9c9b90c3cef7303b", "Apache-2.0"),
    "osnet": HubModel("kaiyangzhou/osnet", "a5c5cc037c24235cda3b21085b93ad77c9616224", "MIT"),
    # OpenAI's CLIP ViT-B/16 weights (MIT), in open_clip's format. The mirror's
    # model card says Apache-2.0.
    "clip": HubModel("timm/vit_base_patch16_clip_224.openai", "977e3dd0ec55ab8da155f2fbeb6b5f54948b6e3d", "MIT"),
}
OSNET_CHECKPOINT = (
    "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
)


def models_dir() -> Path:
    """Return the models/ directory: WALKIE_VISION_MODELS, else <repo>/models."""
    env = os.environ.get("WALKIE_VISION_MODELS")
    if env:
        return Path(env)
    root = os.environ.get("PIXI_PROJECT_ROOT")
    if root:
        return Path(root) / "models"
    raise RuntimeError("cannot find models/: set WALKIE_VISION_MODELS or run through `pixi run`")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _is_valid(path: Path, sha256: str) -> bool:
    return path.is_file() and sha256_of(path) == sha256


def _download(url: str, target: Path, size: int, sha256: str) -> None:
    """Download url to target through a .part file, and check it."""
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    print(f"  downloading {url} ({size / 2**20:.0f} MiB)", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(part, "wb") as f:
            shutil.copyfileobj(response, f, 1 << 20)
        got = sha256_of(part)
        if got != sha256:
            raise RuntimeError(f"{url}: SHA-256 is {got}, expected {sha256}")
        part.replace(target)
    finally:
        part.unlink(missing_ok=True)


def fetch(item: Download, root: Path) -> str:
    """Make sure one download is in place; return "present" or "downloaded"."""
    dest = root / item.dest
    if not item.members:
        if _is_valid(dest, item.sha256):
            return "present"
        _download(item.url, dest, item.size, item.sha256)
        return "downloaded"

    if all(_is_valid(dest / name, sha) for name, sha in item.members.items()):
        return "present"
    archive = root / Path(item.url).name
    _download(item.url, archive, item.size, item.sha256)
    try:
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for name, sha in item.members.items():
                # Only the listed names are extracted, each to a fixed path.
                with z.open(name) as src, open(dest / name, "wb") as out:
                    shutil.copyfileobj(src, out, 1 << 20)
                if sha256_of(dest / name) != sha:
                    raise RuntimeError(f"{item.url}: {name} does not match its SHA-256")
    finally:
        archive.unlink(missing_ok=True)
    return "downloaded"


def main() -> int:
    root = models_dir()
    print(f"models directory: {root}")
    failed = 0
    for item in DOWNLOADS:
        try:
            status = fetch(item, root)
        except Exception as e:  # report every file, then fail
            failed += 1
            print(f"FAILED  {item.dest}: {e}")
            continue
        print(f"{status:10s} {item.dest}  [{item.licence}]")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
