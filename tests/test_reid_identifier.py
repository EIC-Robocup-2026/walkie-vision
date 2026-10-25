"""Tests of walkie_vision.reid.identifier with a tiny CPU module instead of OSNet."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from walkie_vision.providers import OSNET_X1_0_MSMT17, ReIdentifier
from walkie_vision.reid import INPUT_SIZE, MEAN, STD
from walkie_vision.reid import identifier
from walkie_vision.reid.identifier import OSNetReIdentifier, osnet_input


class _Tiny(torch.nn.Module):
    """Maps (n, 3, h, w) to (n, 512) and records each batch size."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.linear = torch.nn.Linear(3, 512)
        self.batches: list[int] = []
        self.devices: list[str] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.batches.append(x.shape[0])
        self.devices.append(x.device.type)
        return self.linear(x.mean(dim=(2, 3)))


def _crop(seed: int, h: int = 120, w: int = 50) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_osnet_input_shape_and_type():
    x = osnet_input([_crop(0), _crop(1, 300, 90)])
    assert x.shape == (2, 3, *INPUT_SIZE) == (2, 3, 256, 128)
    assert x.dtype == np.float32
    assert osnet_input([]).shape == (0, 3, 256, 128)


def test_osnet_input_is_rgb_and_normalised():
    blue = np.zeros((40, 20, 3), np.uint8)
    blue[..., 0] = 255  # BGR: pure blue
    x = osnet_input([blue])[0]
    rgb = np.array([0.0, 0.0, 1.0])
    want = (rgb - np.array(MEAN)) / np.array(STD)
    for channel in range(3):
        np.testing.assert_allclose(x[channel], want[channel], rtol=1e-6)


def test_osnet_input_resizes_to_height_by_width():
    crop = np.zeros((100, 50, 3), np.uint8)
    crop[:, :25] = 255  # left half white
    x = osnet_input([crop])[0, 0]
    assert x.shape == (256, 128)
    white, black = (1.0 - MEAN[0]) / STD[0], (0.0 - MEAN[0]) / STD[0]
    np.testing.assert_allclose(x[:, :60], white, rtol=1e-6)
    np.testing.assert_allclose(x[:, 68:], black, rtol=1e-6)


@pytest.mark.parametrize(
    "crop",
    [
        np.zeros((0, 10, 3), np.uint8),
        np.zeros((10, 0, 3), np.uint8),
        np.zeros((10, 10), np.uint8),
        np.zeros((10, 10, 4), np.uint8),
        np.zeros((10, 10, 3), np.float32),
    ],
)
def test_osnet_input_rejects_bad_crops(crop):
    with pytest.raises(ValueError):
        osnet_input([_crop(0), crop])


def test_embed_runs_in_chunks_of_batch_size():
    tiny = _Tiny()
    reid = OSNetReIdentifier(device="cpu", batch_size=2, model=tiny)
    crops = [_crop(i) for i in range(5)]
    out = reid.embed(crops)
    assert tiny.batches == [2, 2, 1] and set(tiny.devices) == {"cpu"}
    assert out.shape == (5, 512) and out.dtype == np.float32
    with torch.inference_mode():
        want = tiny(torch.from_numpy(osnet_input(crops))).numpy()
    np.testing.assert_allclose(out, want, rtol=1e-5, atol=1e-6)
    # The rows are not normalised.
    assert not np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_embed_of_nothing():
    tiny = _Tiny()
    out = OSNetReIdentifier(device="cpu", model=tiny).embed([])
    assert out.shape == (0, 512) and out.dtype == np.float32
    assert tiny.batches == []


@pytest.mark.parametrize("batch_size", [0, -1])
def test_batch_size_must_be_positive(batch_size):
    with pytest.raises(ValueError):
        OSNetReIdentifier(device="cpu", batch_size=batch_size, model=_Tiny())


def test_without_a_model_it_loads_osnet_and_warms_up(monkeypatch):
    tiny = _Tiny()
    loaded: list[str] = []

    def load_osnet(device: str) -> torch.nn.Module:
        loaded.append(device)
        return tiny

    monkeypatch.setattr(identifier, "load_osnet", load_osnet)
    OSNetReIdentifier(device="cpu")
    assert loaded == ["cpu"]
    assert tiny.batches == [2]


def test_osnet_reidentifier_is_a_reidentifier():
    reid = OSNetReIdentifier(device="cpu", model=_Tiny())
    assert isinstance(reid, ReIdentifier)
    assert reid.model_name == OSNET_X1_0_MSMT17 and reid.dim == 512
