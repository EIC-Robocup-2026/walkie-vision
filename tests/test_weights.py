"""walkie_vision.weights: fetching the pinned model files, with the download stubbed."""

from walkie_vision import weights

CLIP_REPO = "timm/vit_base_patch16_clip_224.openai"
CLIP_REVISION = "977e3dd0ec55ab8da155f2fbeb6b5f54948b6e3d"


def test_fetch_models_asks_for_the_pinned_clip_file(monkeypatch, tmp_path, capsys):
    calls = []

    def stub(repo, filename, *, revision):
        calls.append((repo, filename, revision))
        return "/cache/file"

    monkeypatch.setenv("WALKIE_VISION_MODELS", str(tmp_path))
    monkeypatch.setattr(weights, "DOWNLOADS", ())
    monkeypatch.setattr("huggingface_hub.hf_hub_download", stub)
    assert weights.main() == 0
    assert calls == [(CLIP_REPO, "open_clip_model.safetensors", CLIP_REVISION)]
    assert "cached" in capsys.readouterr().out


def test_a_fetch_that_fails_fails_the_task(monkeypatch, tmp_path, capsys):
    def stub(*args, **kwargs):
        raise OSError("no network")

    monkeypatch.setenv("WALKIE_VISION_MODELS", str(tmp_path))
    monkeypatch.setattr(weights, "DOWNLOADS", ())
    monkeypatch.setattr("huggingface_hub.hf_hub_download", stub)
    assert weights.main() == 1
    assert "FAILED" in capsys.readouterr().out
