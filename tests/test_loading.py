"""walkie_vision.loading: build_providers() with every model class mocked.

No model loads: the provider classes are replaced by autospec mocks, so a
call that does not fit a real signature fails here. The model files are
empty files in a temporary folder.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

yoloe = pytest.importorskip("walkie_vision.providers.object_detection.yoloe")
yolo_pose = pytest.importorskip("walkie_vision.providers.pose_estimation.yolo_pose")
osnet = pytest.importorskip("walkie_vision.providers.reid.osnet")
buffalo_l = pytest.importorskip("walkie_vision.providers.face_recognition.buffalo_l")

import huggingface_hub  # noqa: E402
import onnxruntime  # noqa: E402
import torch  # noqa: E402
from huggingface_hub.errors import LocalEntryNotFoundError  # noqa: E402

from walkie_vision import config as config_module  # noqa: E402
from walkie_vision import errors, loading  # noqa: E402
from walkie_vision.config import VisionConfig  # noqa: E402
from walkie_vision.loading import build_providers  # noqa: E402
from walkie_vision.pipeline import Providers  # noqa: E402
from walkie_vision.providers import _ultralytics  # noqa: E402
from walkie_vision.providers.image_embed import clip as clip_module  # noqa: E402
from walkie_vision.weights import BUFFALO_L_DIR, ULTRALYTICS_DIR  # noqa: E402

CLASSES = ("chair", "sofa")
WARM_UP = "warming up the face models; the first run on this GPU can take minutes"


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


@pytest.fixture
def models(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("WALKIE_VISION_MODELS", str(tmp_path))
    config = VisionConfig(classes=CLASSES)
    for name in (config.objects_weights, yoloe.MOBILECLIP_FILE, config.pose_weights):
        _touch(tmp_path / ULTRALYTICS_DIR / name)
    for name in loading._FACE_FILES:
        _touch(tmp_path / BUFFALO_L_DIR / name)
    return tmp_path


@pytest.fixture
def manager(monkeypatch):
    """One mock that records the calls to every provider class, in order."""
    manager = mock.Mock()
    for module, name in (
        (yoloe, "YoloeTracker"),
        (yoloe, "YoloeDetector"),
        (yolo_pose, "YoloPoseTracker"),
        (osnet, "OSNetReIdentifier"),
        (buffalo_l, "FaceAnalyzer"),
        (buffalo_l, "HeadFaceRecognizer"),
    ):
        fake = mock.create_autospec(getattr(module, name))
        monkeypatch.setattr(module, name, fake)
        manager.attach_mock(fake, name)
    severity = mock.Mock()
    monkeypatch.setattr(onnxruntime, "set_default_logger_severity", severity)
    manager.attach_mock(severity, "set_default_logger_severity")
    manager.attach_mock(mock.Mock(spec=["info", "warning"]), "logger")
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: "/cache/osnet.pth")
    download = mock.Mock(side_effect=AssertionError("must not download"))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    return manager


def _model_calls(manager) -> list:
    return [c for c in manager.mock_calls if not c[0].startswith("logger")]


def test_the_tracker_names_match_the_providers():
    assert config_module.TRACKER_FILES == _ultralytics.TRACKER_NAMES


def test_every_provider_is_built_from_the_config_in_order(models, manager):
    config = VisionConfig(classes=CLASSES)
    result = build_providers(config, manager.logger)

    ultralytics = (models / ULTRALYTICS_DIR).absolute()
    expected_crop = buffalo_l.HeadCropConfig(input_sizes=(160, 256, 384, 640), max_people=6, max_face_offset=0.25)
    assert _model_calls(manager) == [
        mock.call.YoloeTracker(
            ultralytics / config.objects_weights,
            list(CLASSES),
            device="cuda:0",
            imgsz=640,
            conf=0.1,
            agnostic_nms=True,
            tracker="bytetrack.yaml",
        ),
        mock.call.YoloPoseTracker(
            ultralytics / config.pose_weights, device="cuda:0", imgsz=640, conf=0.1, tracker="botsort.yaml"
        ),
        mock.call.OSNetReIdentifier(device="cuda:0", batch_size=16),
        # ERROR and above, before the face sessions exist: a session takes the level when it is created.
        mock.call.set_default_logger_severity(3),
        mock.call.FaceAnalyzer((models / BUFFALO_L_DIR).absolute(), det_thresh=0.5),
        mock.call.HeadFaceRecognizer(manager.FaceAnalyzer.return_value, expected_crop),
    ]
    # The warm-up line comes right before the face models.
    names = [c[0] for c in manager.mock_calls]
    assert names.index("logger.info") == names.index("FaceAnalyzer") - 1
    assert manager.logger.info.call_args_list[0] == mock.call(WARM_UP)
    assert manager.logger.info.call_args_list[-1].args[0].startswith("loaded the models in ")
    manager.logger.warning.assert_not_called()

    assert result == Providers(
        objects=manager.YoloeTracker.return_value,
        pose=manager.YoloPoseTracker.return_value,
        faces=manager.HeadFaceRecognizer.return_value,
        reid=manager.OSNetReIdentifier.return_value,
    )


def test_the_second_yoloe_is_built_at_start_only_with_on_demand(models, manager):
    config = VisionConfig(classes=CLASSES, on_demand_enabled=True, objects_min_score=0.4)
    result = build_providers(config, manager.logger)
    ultralytics = (models / ULTRALYTICS_DIR).absolute()
    assert manager.YoloeDetector.mock_calls == [
        mock.call(
            ultralytics / config.objects_weights,
            list(CLASSES),
            device="cuda:0",
            imgsz=640,
            conf=0.4,
            agnostic_nms=True,
            max_det=20,
        ),
    ]
    assert result.on_demand is manager.YoloeDetector.return_value
    assert result.objects is manager.YoloeTracker.return_value
    # The streaming tracker gets exactly what it gets with the flag off.
    assert "keep_masks" not in manager.YoloeTracker.call_args.kwargs
    assert manager.YoloeTracker.call_args.kwargs["conf"] == 0.1


def test_the_second_yoloe_asks_for_masks_with_geometry(models, manager):
    build_providers(VisionConfig(classes=CLASSES, on_demand_enabled=True, geometry_enabled=True), manager.logger)
    assert manager.YoloeDetector.call_args.kwargs["keep_masks"] is True


def test_geometry_asks_the_object_detector_for_masks(models, manager):
    # The call above pins that the default config does not ask. The autospec
    # checks this keyword against the real signature.
    build_providers(VisionConfig(classes=CLASSES, geometry_enabled=True), manager.logger)
    assert manager.YoloeTracker.call_args.kwargs["keep_masks"] is True
    assert "keep_masks" not in manager.YoloPoseTracker.call_args.kwargs


@pytest.mark.parametrize("flag", ["object_crops_enabled", "object_embeddings_enabled"])
def test_sampling_asks_the_object_detector_for_masks(models, manager, monkeypatch, flag):
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: "/cache/clip.safetensors")
    monkeypatch.setattr(clip_module, "ClipEmbedder", mock.create_autospec(clip_module.ClipEmbedder))
    build_providers(VisionConfig(classes=CLASSES, **{flag: True}), manager.logger)
    assert manager.YoloeTracker.call_args.kwargs["keep_masks"] is True


def test_no_embedder_is_built_without_object_embeddings(models, manager, monkeypatch):
    # hf_hub_download raises in the manager fixture, so the cache is not read either.
    clip = mock.create_autospec(clip_module.ClipEmbedder)
    monkeypatch.setattr(clip_module, "ClipEmbedder", clip)
    assert build_providers(VisionConfig(classes=CLASSES, object_crops_enabled=True), manager.logger).embedder is None
    clip.assert_not_called()


@pytest.mark.parametrize(
    ("provider", "name", "overrides"),
    [
        ("YoloeTracker", "objects", {}),
        ("YoloeDetector", "on-demand", {"on_demand_enabled": True}),
        ("YoloPoseTracker", "pose", {}),
        ("OSNetReIdentifier", "OSNet", {}),
        ("FaceAnalyzer", "face", {}),
    ],
)
def test_a_model_that_fails_to_load_is_named_in_the_log(models, manager, provider, name, overrides):
    getattr(manager, provider).side_effect = RuntimeError("out of memory")
    with pytest.raises(RuntimeError, match="out of memory"):
        build_providers(VisionConfig(classes=CLASSES, **overrides), manager.logger)
    manager.logger.warning.assert_called_once_with(f"the {name} model failed to load: RuntimeError: out of memory")


def test_clip_that_fails_to_load_is_named_in_the_log(models, manager, monkeypatch):
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: "/cache/clip.safetensors")
    clip = mock.create_autospec(clip_module.ClipEmbedder, side_effect=RuntimeError("out of memory"))
    monkeypatch.setattr(clip_module, "ClipEmbedder", clip)
    config = VisionConfig(classes=CLASSES, device="cpu", object_embeddings_enabled=True)
    with pytest.raises(RuntimeError, match="out of memory"):
        build_providers(config, manager.logger)
    manager.logger.warning.assert_called_once_with("the CLIP model failed to load: RuntimeError: out of memory")


def test_a_clean_start_logs_no_load_failure(models, manager):
    build_providers(VisionConfig(classes=CLASSES), manager.logger)
    manager.logger.warning.assert_not_called()


def test_object_embeddings_load_clip_offline_at_the_pinned_commit(models, manager, monkeypatch):
    calls = []

    def stub(repo, filename, *, revision, local_files_only):
        calls.append((repo, filename, revision, local_files_only))
        return "/cache/clip.safetensors"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", stub)
    clip = mock.create_autospec(clip_module.ClipEmbedder)
    monkeypatch.setattr(clip_module, "ClipEmbedder", clip)

    result = build_providers(VisionConfig(classes=CLASSES, device="cpu", object_embeddings_enabled=True), manager.logger)

    assert calls == [
        (
            "timm/vit_base_patch16_clip_224.openai",
            "open_clip_model.safetensors",
            "977e3dd0ec55ab8da155f2fbeb6b5f54948b6e3d",
            True,
        )
    ]
    clip.assert_called_once_with("/cache/clip.safetensors", device="cpu")
    assert result.embedder is clip.return_value


def test_missing_clip_weights_fail_before_any_model_loads(models, manager, monkeypatch):
    def stub(*args, **kwargs):
        raise LocalEntryNotFoundError("not cached")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", stub)
    with pytest.raises(FileNotFoundError, match="pixi run fetch-models"):
        build_providers(VisionConfig(classes=CLASSES, object_embeddings_enabled=True), manager.logger)
    assert _model_calls(manager) == []


def test_the_parameters_reach_the_providers(models, manager, tmp_path):
    weights = _touch(tmp_path / "elsewhere" / "custom-seg.pt")
    _touch(weights.parent / yoloe.MOBILECLIP_FILE)
    config = VisionConfig(
        classes=CLASSES,
        device="cpu",
        objects_weights=str(weights),
        objects_imgsz=320,
        objects_conf=0.2,
        objects_agnostic_nms=False,
        objects_tracker="botsort.yaml",
        pose_imgsz=480,
        pose_conf=0.3,
        pose_tracker="bytetrack.yaml",
        face_det_thresh=0.6,
        face_input_sizes=(256, 640),
        face_max_people=3,
        face_max_offset=0.3,
        reid_batch_size=8,
    )
    build_providers(config, manager.logger)
    manager.YoloeTracker.assert_called_once_with(
        weights, list(CLASSES), device="cpu", imgsz=320, conf=0.2, agnostic_nms=False, tracker="botsort.yaml"
    )
    manager.YoloPoseTracker.assert_called_once_with(
        (models / ULTRALYTICS_DIR).absolute() / config.pose_weights,
        device="cpu",
        imgsz=480,
        conf=0.3,
        tracker="bytetrack.yaml",
    )
    manager.OSNetReIdentifier.assert_called_once_with(device="cpu", batch_size=8)
    manager.FaceAnalyzer.assert_called_once_with((models / BUFFALO_L_DIR).absolute(), det_thresh=0.6)
    crop = manager.HeadFaceRecognizer.call_args.args[1]
    assert crop == buffalo_l.HeadCropConfig(input_sizes=(256, 640), max_people=3, max_face_offset=0.3)


def test_the_cuda_cache_is_emptied_only_when_cuda_started(models, manager, monkeypatch):
    empty = mock.Mock()
    monkeypatch.setattr(torch.cuda, "empty_cache", empty)
    build_providers(VisionConfig(classes=CLASSES), manager.logger)
    empty.assert_not_called()
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    build_providers(VisionConfig(classes=CLASSES), manager.logger)
    empty.assert_called_once_with()


def test_a_missing_text_encoder_next_to_absolute_weights_fails_first(models, manager, tmp_path):
    # The weights exist in another folder, but mobileclip2_b.ts is not next to them.
    weights = _touch(tmp_path / "elsewhere" / "custom-seg.pt")
    config = VisionConfig(classes=CLASSES, objects_weights=str(weights))
    with pytest.raises(FileNotFoundError) as caught:
        build_providers(config, manager.logger)
    message = str(caught.value)
    assert str(weights.parent / yoloe.MOBILECLIP_FILE) in message
    assert "pixi run fetch-models" in message
    assert _model_calls(manager) == []


def test_every_missing_file_is_named(tmp_path, monkeypatch, manager):
    monkeypatch.setenv("WALKIE_VISION_MODELS", str(tmp_path))
    with pytest.raises(FileNotFoundError) as caught:
        build_providers(VisionConfig(classes=CLASSES), manager.logger)
    message = str(caught.value)
    for name in ("yoloe-26s-seg.pt", yoloe.MOBILECLIP_FILE, "yolo26s-pose.pt", *loading._FACE_FILES):
        assert name in message
    assert _model_calls(manager) == []


def test_osnet_downloads_once_when_it_is_not_cached(models, manager, monkeypatch):
    download = mock.Mock(return_value="/cache/osnet.pth")
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: None)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    build_providers(VisionConfig(classes=CLASSES), manager.logger)
    download.assert_called_once()
    assert "internet" in manager.logger.warning.call_args.args[0]
    manager.OSNetReIdentifier.assert_called_once()


def test_osnet_that_cannot_download_fails_before_any_model_loads(models, manager, monkeypatch):
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: None)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", mock.Mock(side_effect=OSError("no network")))
    # StartupError is a RuntimeError that main() logs without a traceback.
    with pytest.raises(errors.StartupError, match="internet") as caught:
        build_providers(VisionConfig(classes=CLASSES), manager.logger)
    assert isinstance(caught.value, RuntimeError)
    assert "no network" in str(caught.value)
    assert _model_calls(manager) == []
