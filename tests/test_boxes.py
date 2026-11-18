"""walkie_vision.boxes: the rectangle maths and the mask resize."""

import numpy as np

from walkie_vision import boxes
from walkie_vision.boxes import resize_mask


def test_subtract_rect():
    rect = (0, 0, 10, 10)
    pieces = boxes.subtract_rect(rect, (3, 4, 6, 8))
    assert len(pieces) == 4
    assert sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in pieces) == 100 - 12
    assert boxes.subtract_rect(rect, (20, 20, 30, 30)) == [rect]
    assert boxes.subtract_rect(rect, (-1, -1, 11, 11)) == []
    assert boxes.subtract_rect(rect, (5, -1, 11, 11)) == [(0, 0, 5, 10)]


def test_resize_mask_takes_the_pixel_under_each_output_centre():
    mask = np.array([[True, False, False, False], [False, False, True, False]])
    # Output column i takes input column floor((i + 0.5) * 4 / 2): columns 1 and 3.
    np.testing.assert_array_equal(resize_mask(mask, 2, 2), [[False, False], [False, False]])
    # Output column i takes input column floor((i + 0.5) * 4 / 3): columns 0, 2 and 3.
    np.testing.assert_array_equal(resize_mask(mask, 2, 3), [[True, False, False], [False, True, False]])
    # Each input pixel becomes two.
    np.testing.assert_array_equal(
        resize_mask(np.array([[True, False]]), 2, 4), [[True, True, False, False], [True, True, False, False]]
    )


def test_resize_mask_of_an_equal_size_returns_the_mask():
    mask = np.zeros((3, 4), bool)
    assert resize_mask(mask, 3, 4) is mask
