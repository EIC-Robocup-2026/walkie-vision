"""walkie_vision.crop on synthetic figures: the gates, the mask and the colour shares. No ROS graph.

The guest stands in GUEST, 320 px tall in a 300 x 400 image. Other people
stand behind or in front: a box that ends lower in the image is nearer.
"""

import re
from pathlib import Path

import cv2
import numpy as np
import pytest
from walkie_vision_msgs.srv import GetPersonCrop

from walkie_vision import crop
from walkie_vision.crop import COLOR_NAMES, MASK_GREY, CropStatus, crop_person
from walkie_vision.fakes import figure_pose, paint_figure
from walkie_vision.providers import COCO_KEYPOINTS

SRV = Path(__file__).resolve().parents[1] / "ros2_ws/src/walkie_vision_msgs/srv/GetPersonCrop.srv"
H, W = 400, 300
BACKGROUND = (40, 120, 60)
RED, BLUE, WHITE, BLACK = (0, 0, 255), (255, 0, 0), (255, 255, 255), (0, 0, 0)
GUEST = (60.0, 40.0, 180.0, 360.0)
# The guest's box padded by 0.10 on each side: 12 px across, 32 px down.
CROP_ORIGIN = (48, 8)


def _image() -> np.ndarray:
    image = np.empty((H, W, 3), dtype=np.uint8)
    image[:] = BACKGROUND
    return image


def _guest(top=RED, bottom=BLUE, **kwargs):
    image = _image()
    guest = figure_pose(GUEST, track_id=3)
    paint_figure(image, guest, top, bottom, **kwargs)
    return image, guest


def _crop(image, people, track_id=3, **kwargs):
    # No scaling unless a test asks, so the pixels are exact.
    kwargs.setdefault("max_side", 1000)
    return crop_person(image, people, track_id, **kwargs)


def _share(shares, name: str) -> float:
    return float(shares[COLOR_NAMES.index(name)])


def _rects_mask(rects, shape) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=bool)
    for x0, y0, x1, y1 in np.asarray(rects, dtype=int).reshape(-1, 4):
        mask[y0:y1, x0:x1] = True
    return mask


def _greyed(pixels: np.ndarray) -> np.ndarray:
    return np.all(pixels == MASK_GREY, axis=2)


# --- the message --------------------------------------------------------------


def test_the_statuses_are_the_response_constants():
    constants = dict(re.findall(r"^uint8 ([A-Z_]+)=(\d+)$", SRV.read_text(), flags=re.MULTILINE))
    assert {k: int(v) for k, v in constants.items()} == {s.name: s.value for s in CropStatus}
    for status in CropStatus:
        assert getattr(GetPersonCrop.Response, status.name) == status.value


def test_the_colour_names_are_the_ones_the_srv_lists():
    comment = SRV.read_text().split("# The colour names", 1)[1].split("string[] color_names", 1)[0]
    listed = " ".join(line.lstrip("# ") for line in comment.splitlines()).split("in order:", 1)[1]
    names = [n.strip(" .") for n in re.split(r",| and ", listed) if n.strip(" .")]
    assert tuple(names) == COLOR_NAMES


# --- the gates ----------------------------------------------------------------


def test_a_track_that_is_not_in_the_image_is_not_in_view():
    image, guest = _guest()
    found = _crop(image, [guest], track_id=9)
    assert found.status == CropStatus.NOT_IN_VIEW
    assert found.index == -1 and found.jpeg == b""
    assert "[3]" in found.message


@pytest.mark.parametrize("height, status", [(159.0, CropStatus.TOO_SMALL), (160.0, CropStatus.OK)])
def test_a_box_under_160_px_is_too_small(height, status):
    image = _image()
    person = figure_pose((60.0, 40.0, 180.0, 40.0 + height), track_id=3)
    paint_figure(image, person, RED, BLUE)
    found = _crop(image, [person])
    assert found.status == status
    assert found.index == 0
    assert (found.jpeg != b"") == (status == CropStatus.OK)


def test_the_too_small_message_never_rounds_up_to_the_floor():
    image = _image()
    person = figure_pose((60.0, 40.0, 180.0, 199.6), track_id=3)  # 159.6 px tall
    found = _crop(image, [person])
    assert found.status == CropStatus.TOO_SMALL
    assert found.message == "the box is 159 px tall, under 160"


def test_a_box_with_no_width_is_too_small():
    image = _image()
    person = figure_pose((60.0, 40.0, 60.0, 360.0), track_id=3)  # 320 px tall
    found = _crop(image, [person])
    assert found.status == CropStatus.TOO_SMALL
    assert found.message == "the box has no width"


def test_a_nearer_person_whose_box_overlaps_occludes():
    image, guest = _guest()
    nearer = figure_pose((80.0, 60.0, 200.0, 380.0), track_id=4)  # ends 20 px lower
    found = _crop(image, [guest, nearer])
    assert found.status == CropStatus.OCCLUDED
    assert found.index == 0
    # 100 x 300 shared, out of 120 x 320 + 120 x 320 - 30000.
    assert found.max_overlap == pytest.approx(30000 / 46800)
    assert found.jpeg == b"" and found.top_shares is None


def test_a_nearer_person_over_the_legs_only_occludes_by_iou():
    # The box overlaps the guest's below the torso box (which ends at y 206.4),
    # so only the IoU gate can answer OCCLUDED.
    image, guest = _guest()
    nearer = figure_pose((60.0, 220.0, 180.0, 380.0), track_id=4)  # ends 20 px lower
    found = _crop(image, [guest, nearer])
    assert found.status == CropStatus.OCCLUDED
    assert found.torso_cover == 0.0 and found.max_overlap > crop.OCCLUDED_IOU
    # 120 x 140 shared, out of 120 x 320 + 120 x 160 - 16800.
    assert found.max_overlap == pytest.approx(16800 / 40800)


def test_a_smaller_nearer_person_over_the_torso_occludes():
    # A child in front: their box overlaps the guest's little, but covers the torso.
    image, guest = _guest()
    child = figure_pose((100.0, 150.0, 140.0, 380.0), track_id=4)
    found = _crop(image, [guest, child])
    assert found.status == CropStatus.OCCLUDED
    assert found.max_overlap < crop.OCCLUDED_IOU < found.torso_cover
    # The torso box is x 90 to 150 and y 104 to 206.4; the child covers 40 x 56.4 of it.
    assert found.torso_cover == pytest.approx(40 * 56.4 / (60 * 102.4))


def test_a_person_behind_over_40_percent_of_the_torso_is_masked_and_not_occluding():
    image = _image()
    behind = figure_pose((126.0, 20.0, 260.0, 300.0), track_id=4)  # ends 60 px higher
    image[20:300, 126:260] = WHITE
    guest = figure_pose(GUEST, track_id=3)
    paint_figure(image, guest, RED, BLUE)
    # The box covers x 126 to 150 of the torso box's x 90 to 150, at its full height.
    assert crop.box_cover(behind.box, crop.torso_box(guest)) == pytest.approx(0.4)
    found = _crop(image, [behind, guest])
    assert found.status == CropStatus.OK
    assert found.index == 1
    # Only a nearer person counts: the torso box is 40% covered, but from behind.
    assert found.max_overlap == 0.0 and found.torso_cover == 0.0
    # The part of the box inside the crop, minus the guest's box: above it and right of it.
    assert sorted(map(tuple, found.masked.astype(int).tolist())) == [(78, 12, 144, 32), (132, 32, 144, 292)]
    greyed = _greyed(found.pixels)
    assert np.array_equal(greyed, _rects_mask(found.masked, found.pixels.shape))
    # Inside the guest's box, the person behind stays: the accepted residual.
    x, y = 170 - CROP_ORIGIN[0], 250 - CROP_ORIGIN[1]
    assert tuple(found.pixels[y, x]) == WHITE
    assert _share(found.top_shares, "red") == 1.0


def test_the_guest_box_is_never_greyed_and_masked_boxes_match_the_grey_pixels():
    image = _image()
    guest_box = (60.5, 40.3, 180.6, 360.2)
    guest = figure_pose(guest_box, track_id=3)
    others = [
        figure_pose((130.4, 18.6, 250.7, 290.2), track_id=4),  # behind, on the right
        figure_pose((0.0, 100.0, 66.0, 390.0), track_id=5),  # nearer, on the left, barely overlapping
        figure_pose((150.0, 0.0, 300.0, 120.0), track_id=6),  # far behind, at the top
    ]
    for other in others:
        paint_figure(image, other, WHITE, BLACK)
    paint_figure(image, guest, RED, BLUE)
    found = _crop(image, [*others, guest])
    assert found.status == CropStatus.OK
    assert len(found.masked) >= 3
    greyed = _greyed(found.pixels)
    assert np.array_equal(greyed, _rects_mask(found.masked, found.pixels.shape))
    # Every pixel that the guest's box touches, also in part, keeps its value.
    x0, y0 = int(np.floor(guest_box[0])) - CROP_ORIGIN[0], int(np.floor(guest_box[1])) - CROP_ORIGIN[1]
    x1, y1 = int(np.ceil(guest_box[2])) - CROP_ORIGIN[0], int(np.ceil(guest_box[3])) - CROP_ORIGIN[1]
    assert not greyed[y0:y1, x0:x1].any()
    np.testing.assert_array_equal(
        found.pixels[y0:y1, x0:x1], image[y0 + CROP_ORIGIN[1] : y1 + CROP_ORIGIN[1], x0 + CROP_ORIGIN[0] : x1 + CROP_ORIGIN[0]]
    )


def test_mask_others_false_greys_nothing():
    image, guest = _guest()
    behind = figure_pose((126.0, 20.0, 260.0, 300.0), track_id=4)
    found = _crop(image, [guest, behind], mask_others=False)
    assert found.status == CropStatus.OK
    assert found.masked.shape == (0, 4)
    assert not _greyed(found.pixels).any()


# --- the colour shares --------------------------------------------------------


def test_the_shares_of_a_solid_figure():
    image, guest = _guest(RED, BLUE)
    found = _crop(image, [guest])
    assert found.status == CropStatus.OK
    assert _share(found.top_shares, "red") == 1.0
    assert _share(found.bottom_shares, "blue") == 1.0
    assert found.top_shares.sum() == pytest.approx(1.0) and found.bottom_shares.sum() == pytest.approx(1.0)
    assert found.top_left == 1.0 and found.bottom_left == 1.0


def test_the_shares_of_a_striped_figure():
    image, guest = _guest(RED, BLUE, top_stripe=WHITE, bottom_stripe=BLACK, stripe_px=4)
    found = _crop(image, [guest])
    assert _share(found.top_shares, "red") == pytest.approx(0.5, abs=0.08)
    assert _share(found.top_shares, "white") == pytest.approx(0.5, abs=0.08)
    assert _share(found.bottom_shares, "blue") == pytest.approx(0.5, abs=0.08)
    assert _share(found.bottom_shares, "black") == pytest.approx(0.5, abs=0.08)
    assert _share(found.top_shares, "red") + _share(found.top_shares, "white") == pytest.approx(1.0)
    assert found.top_shares.sum() == pytest.approx(1.0) and found.bottom_shares.sum() == pytest.approx(1.0)


def test_a_grey_mask_over_a_red_region_leaves_it_red():
    # The left shoulder and the left knee lie outside the guest's box, over a
    # person behind, so the mask greys part of each region.
    image = _image()
    behind = figure_pose((175.0, 30.0, 280.0, 330.0), track_id=4)
    image[30:330, 175:280] = WHITE
    guest = figure_pose(GUEST, track_id=3)
    guest.keypoints[COCO_KEYPOINTS.index("left_shoulder"), 0] = 260.0
    guest.keypoints[COCO_KEYPOINTS.index("left_knee"), 0] = 260.0
    paint_figure(image, guest, RED, RED)

    found = _crop(image, [guest, behind])
    unmasked = _crop(image, [guest, behind], mask_others=False)
    assert found.status == CropStatus.OK
    greyed = _greyed(found.pixels)
    # The torso is painted red at x 185, y 120, right of the guest's box: it is greyed.
    x, y = 185 - CROP_ORIGIN[0], 120 - CROP_ORIGIN[1]
    assert greyed[y, x] and tuple(unmasked.pixels[y, x]) == RED
    assert not greyed[40 - CROP_ORIGIN[1] : 360 - CROP_ORIGIN[1], 60 - CROP_ORIGIN[0] : 180 - CROP_ORIGIN[0]].any()
    assert _share(found.top_shares, "red") == 1.0 and _share(found.top_shares, "grey") == 0.0
    assert _share(found.bottom_shares, "red") == 1.0 and _share(found.bottom_shares, "grey") == 0.0
    # Part of each region lies outside the box and is left out: about a fifth
    # of the torso and a quarter of the thighs.
    assert found.top_left == pytest.approx(0.79, abs=0.05)
    assert found.bottom_left == pytest.approx(0.74, abs=0.05)
    assert unmasked.top_left == found.top_left and unmasked.bottom_left == found.bottom_left
    np.testing.assert_array_equal(unmasked.top_shares, found.top_shares)


@pytest.mark.parametrize(
    "hidden, top_gated, bottom_gated",
    [
        ("left_shoulder", True, False),
        ("right_knee", False, True),
        ("left_hip", True, True),
        ("nose", False, False),
    ],
)
def test_a_hidden_keypoint_gates_its_region(hidden, top_gated, bottom_gated):
    image, guest = _guest()
    guest.keypoint_scores[COCO_KEYPOINTS.index(hidden)] = 0.3
    found = _crop(image, [guest])
    assert found.status == CropStatus.OK
    assert (found.top_shares is None) == top_gated
    assert (found.bottom_shares is None) == bottom_gated
    assert (found.top_left == 0.0) == top_gated
    assert (found.bottom_left == 0.0) == bottom_gated


def test_min_keypoint_score_gates_both_regions():
    image, guest = _guest()  # every keypoint scores 0.9
    found = _crop(image, [guest], min_keypoint_score=0.95)
    assert found.status == CropStatus.OK and found.jpeg
    assert found.top_shares is None and found.bottom_shares is None


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
    shares = crop.color_shares(np.array([bgr], dtype=np.uint8))
    assert COLOR_NAMES[int(np.argmax(shares))] == name
    assert shares.max() == 1.0


# --- the crop -----------------------------------------------------------------


def test_the_crop_is_scaled_down_with_its_keypoints_and_masks():
    image, guest = _guest()
    behind = figure_pose((126.0, 20.0, 260.0, 300.0), track_id=4)
    full = _crop(image, [guest, behind])
    assert full.pixels.shape == (384, 144, 3)  # never scaled up
    np.testing.assert_allclose(full.keypoints, guest.keypoints - CROP_ORIGIN)

    small = _crop(image, [guest, behind], max_side=192)
    assert small.pixels.shape == (192, 72, 3)
    np.testing.assert_allclose(small.keypoints, full.keypoints * 0.5)
    np.testing.assert_allclose(small.masked, full.masked * 0.5)
    decoded = cv2.imdecode(np.frombuffer(small.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (192, 72, 3)
    # The shares come from the full-resolution pixels.
    np.testing.assert_array_equal(small.top_shares, full.top_shares)


def test_the_crop_keeps_no_frame():
    image, guest = _guest()
    behind = figure_pose((126.0, 20.0, 260.0, 300.0), track_id=4)
    before = image.copy()
    found = _crop(image, [guest, behind])
    # The input image is not greyed, and nothing in the result is a view of it.
    np.testing.assert_array_equal(image, before)
    for value in vars(found).values():
        if isinstance(value, np.ndarray):
            assert not np.shares_memory(value, image)


# --- the helpers --------------------------------------------------------------


@pytest.mark.parametrize(
    "other_box, own_box, own_scores, nearer",
    [
        # 3% of 400 is 12 px.
        ((0.0, 0.0, 50.0, 373.0), (0.0, 0.0, 50.0, 360.0), 0.9, True),
        ((0.0, 0.0, 50.0, 371.0), (0.0, 0.0, 50.0, 360.0), 0.9, False),
        ((0.0, 0.0, 50.0, 300.0), (0.0, 0.0, 50.0, 360.0), 0.9, False),
        # Cut off by the image's bottom edge.
        ((0.0, 0.0, 50.0, 400.0), (0.0, 0.0, 50.0, 395.0), 0.9, True),
        ((0.0, 0.0, 50.0, 395.0), (0.0, 0.0, 50.0, 400.0), 0.9, False),
        # Both cut off: nearer only when it hides a torso keypoint.
        ((0.0, 0.0, 300.0, 400.0), (60.0, 40.0, 180.0, 400.0), 0.3, True),
        ((0.0, 0.0, 300.0, 400.0), (60.0, 40.0, 180.0, 400.0), 0.9, False),
        ((250.0, 0.0, 300.0, 400.0), (60.0, 40.0, 180.0, 400.0), 0.3, False),
    ],
)
def test_is_nearer(other_box, own_box, own_scores, nearer):
    own = figure_pose(own_box, keypoint_score=own_scores)
    assert crop.is_nearer(figure_pose(other_box), own, H, min_keypoint_score=0.5) == nearer


def test_subtract_rect():
    rect = (0, 0, 10, 10)
    pieces = crop.subtract_rect(rect, (3, 4, 6, 8))
    assert len(pieces) == 4
    assert sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in pieces) == 100 - 12
    assert crop.subtract_rect(rect, (20, 20, 30, 30)) == [rect]
    assert crop.subtract_rect(rect, (-1, -1, 11, 11)) == []
    assert crop.subtract_rect(rect, (5, -1, 11, 11)) == [(0, 0, 5, 10)]


@pytest.mark.parametrize(
    "kwargs, name",
    [
        ({"padding": -0.1}, "padding"),
        ({"padding": float("nan")}, "padding"),
        ({"max_side": 0}, "max_side"),
        ({"min_keypoint_score": 1.5}, "min_keypoint_score"),
    ],
)
def test_options_out_of_range_raise(kwargs, name):
    image, guest = _guest()
    with pytest.raises(ValueError, match=name):
        _crop(image, [guest], **kwargs)
