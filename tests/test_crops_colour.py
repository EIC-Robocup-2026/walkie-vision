"""walkie_vision.crops.colour: the colour names GetPersonCrop.srv lists, the HSV colour bins and the Lab palette. No ROS graph."""

import re
from pathlib import Path

import numpy as np
import pytest

from walkie_vision.crops import colour
from walkie_vision.crops.colour import (
    COLOR_NAMES,
    MIN_COLOR_PIXELS,
    PALETTE_RGB,
    mask_color_shares,
    nearest_palette_indices,
)
from walkie_vision.fakes import box_mask

SRV = Path(__file__).resolve().parents[1] / "ros2_ws/src/walkie_vision_msgs/srv/GetPersonCrop.srv"


def test_the_colour_names_are_the_ones_the_srv_lists():
    comment = SRV.read_text().split("# The colour names", 1)[1].split("string[] color_names", 1)[0]
    listed = " ".join(line.lstrip("# ") for line in comment.splitlines()).split("in order:", 1)[1]
    names = [n.strip(" .") for n in re.split(r",| and ", listed) if n.strip(" .")]
    assert tuple(names) == COLOR_NAMES


@pytest.mark.parametrize(
    "bgr, name",
    [
        ((0, 0, 0), "black"),
        ((255, 255, 255), "white"),
        ((128, 128, 128), "grey"),
        ((0, 0, 255), "red"),
        ((0, 128, 255), "orange"),
        ((0, 255, 255), "yellow"),
        ((0, 255, 0), "green"),
        ((255, 0, 0), "blue"),
        ((255, 0, 128), "purple"),
        ((128, 0, 128), "purple"),
        ((180, 105, 255), "pink"),
        ((203, 192, 255), "pink"),
        ((19, 69, 139), "brown"),
        ((196, 228, 255), "beige"),
    ],
)
def test_colour_bins(bgr, name):
    shares = colour.color_shares(np.array([bgr], dtype=np.uint8))
    assert COLOR_NAMES[int(np.argmax(shares))] == name
    assert shares.max() == 1.0


# --- colour shares ---------------------------------------------------------


def _bgr(rgb) -> tuple[int, int, int]:
    return rgb[2], rgb[1], rgb[0]


def test_the_palette_follows_the_field_order():
    assert COLOR_NAMES[3] == "red" and PALETTE_RGB[3] == (220, 35, 35)
    assert COLOR_NAMES[-1] == "beige" and PALETTE_RGB[-1] == (210, 190, 145)
    assert len(PALETTE_RGB) == len(COLOR_NAMES) == 12


@pytest.mark.parametrize("index", range(12))
def test_each_palette_colour_counts_as_itself(index):
    image = np.full((10, 10, 3), _bgr(PALETTE_RGB[index]), dtype=np.uint8)
    shares = mask_color_shares(image, np.ones((10, 10), dtype=bool))
    assert shares.tolist() == [1.0 if i == index else 0.0 for i in range(12)]


def test_shares_use_only_the_mask_and_sum_to_one():
    image = np.full((20, 20, 3), _bgr(PALETTE_RGB[3]), dtype=np.uint8)
    image[:, 10:] = _bgr(PALETTE_RGB[7])
    mask = box_mask(20, 20, (5, 0, 15, 10))
    shares = mask_color_shares(image, mask)
    assert shares[3] == pytest.approx(0.5) and shares[7] == pytest.approx(0.5)
    assert shares.sum() == pytest.approx(1.0)


def test_shares_need_50_mask_pixels():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    assert mask_color_shares(image, box_mask(20, 20, (0, 0, 7, 7))) is None  # 49 pixels
    assert MIN_COLOR_PIXELS == 50
    assert mask_color_shares(image, box_mask(20, 20, (0, 0, 10, 5))) is not None


def test_a_tie_takes_the_earlier_centre():
    centres = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    assert nearest_palette_indices(np.array([[1.0, 0.0, 0.0]]), centres).tolist() == [0]
    assert nearest_palette_indices(np.array([[1.0, 0.0, 0.0]]), centres[::-1]).tolist() == [0]
    assert nearest_palette_indices(np.array([[1.5, 0.0, 0.0]]), centres).tolist() == [1]
