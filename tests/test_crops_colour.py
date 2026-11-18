"""walkie_vision.crops.colour: the colour names GetPersonCrop.srv lists, and the HSV colour bins. No ROS graph."""

import re
from pathlib import Path

import numpy as np
import pytest

from walkie_vision.crops import colour
from walkie_vision.crops.colour import COLOR_NAMES

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
