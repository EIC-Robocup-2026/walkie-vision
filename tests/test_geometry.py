"""walkie_vision.geometry: lifting a mask to points and scoring a view, without ROS communication or a GPU."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest
from builtin_interfaces.msg import Time
from sensor_msgs.msg import Image

from walkie_vision.camera.depth import RegisteredDepth, depth_to_metres
from walkie_vision.fakes import box_mask
from walkie_vision.geometry import depth_cover, erosion_radius, frame_laplacian, lift_mask, view_quality
from walkie_vision.messages import MAX_CLOUD_POINTS, MIN_CLOUD_POINTS

FRAME = "zed_head_left_camera_frame_optical"
BASE_MS = 100_000  # Stamps start at 100 s: a zero stamp never pairs.


def _stamped(ms: float, frame: str = FRAME) -> Image:
    msg = Image()
    ns = round(ms * 1_000_000)
    msg.header.stamp = Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)
    msg.header.frame_id = frame
    return msg


def _raster(values, encoding: str, *, bigendian: bool = False, padding: int = 0) -> Image:
    """A depth image with these values, in the given byte order and with padded rows."""
    code = {"32FC1": "f4", "16UC1": "u2"}[encoding]
    array = np.asarray(values, dtype=(">" if bigendian else "<") + code)
    rows = [row.tobytes() + b"\xee" * padding for row in array]
    msg = _stamped(BASE_MS)
    msg.height, msg.width = array.shape
    msg.encoding = encoding
    msg.is_bigendian = int(bigendian)
    msg.step = int(array.shape[1]) * int(code[1]) + padding
    msg.data = b"".join(rows)
    return msg


# --- lifting a mask to points --------------------------------------------------

RASTER_H, RASTER_W = 40, 60
FX, FY, CX, CY = 100.0, 120.0, 30.0, 20.0
MIN_M, MAX_M = 0.2, 3.0


def _plane(value: float = 1.0, height: int = RASTER_H, width: int = RASTER_W) -> np.ndarray:
    return np.full((height, width), value, np.float32)


def _box(x1: int, y1: int, x2: int, y2: int, height: int = RASTER_H, width: int = RASTER_W) -> np.ndarray:
    return box_mask(height, width, (x1, y1, x2, y2))


def _lift(metres: np.ndarray, mask: np.ndarray, *, min_m: float = MIN_M, max_m: float = MAX_M) -> np.ndarray:
    return lift_mask(mask, RegisteredDepth(np.asarray(metres, np.float32), FX, FY, CX, CY), min_m, max_m)


def _pixels(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the integer rows and columns the points came from, by inverting the back-projection."""
    rows = np.rint(points[:, 1] * FY / points[:, 2] + CY).astype(int)
    cols = np.rint(points[:, 0] * FX / points[:, 2] + CX).astype(int)
    return rows, cols


def test_a_plane_lifts_to_the_back_projection_of_the_eroded_mask():
    # 20 x 20 px gives a radius of 1, so one pixel goes on every side.
    points = _lift(_plane(2.0), _box(10, 5, 30, 25))
    rows, cols = np.mgrid[6:24, 11:29]
    expected = np.column_stack(
        ((cols.ravel() - CX) * 2.0 / FX, (rows.ravel() - CY) * 2.0 / FY, np.full(rows.size, 2.0))
    )
    assert points.dtype == np.float32
    np.testing.assert_allclose(points, expected, atol=1e-6)


def test_a_tilted_plane_keeps_its_slope():
    metres = np.broadcast_to(1.0 + 0.001 * np.arange(RASTER_H, dtype=np.float32)[:, None], (RASTER_H, RASTER_W))
    points = _lift(metres, _box(10, 5, 30, 35))
    rows, _ = _pixels(points)
    assert len(points) == 18 * 28
    np.testing.assert_allclose(points[:, 2], 1.0 + 0.001 * rows, atol=1e-6)


def test_a_cube_face_keeps_its_size_and_drops_the_background():
    # A cube's front face at z = 1 m on a wall at 3 m: x from -0.1 to 0.2 m and y from -0.15 to 0.15 m.
    fx, cx, cy = 100.0, 60.0, 40.0
    metres = np.full((80, 120), 3.0, np.float32)
    metres[25:55, 50:80] = 1.0
    # The mask overshoots the face by one pixel on each side. 32 px gives a radius of 2.
    mask = box_mask(80, 120, (49, 24, 81, 56))
    points = lift_mask(mask, RegisteredDepth(metres, fx, fx, cx, cy), MIN_M, MAX_M)
    np.testing.assert_array_equal(points[:, 2], 1.0)
    assert len(points) == 28 * 28
    np.testing.assert_allclose(
        [points[:, 0].min(), points[:, 0].max(), points[:, 1].min(), points[:, 1].max()],
        [(51 - cx) / fx, (78 - cx) / fx, (26 - cy) / fx, (53 - cy) / fx],
        atol=1e-6,
    )


def test_the_whole_mask_is_still_cut_by_the_depth_window():
    # 20 x 3 px: erosion leaves 18 pixels, under the floor. The whole mask has 60,
    # and its top row overshoots onto a wall at 3 m.
    metres = _plane(3.0)
    metres[10:12, 10:30] = 1.0
    points = _lift(metres, _box(10, 9, 30, 12))
    np.testing.assert_array_equal(points[:, 2], 1.0)
    assert len(points) == 2 * 20


def test_points_are_in_the_optical_frame():
    # The optical centre is at pixel (30, 20). Right is +x, down is +y, forward is +z.
    right_below = _lift(_plane(), _box(40, 25, 55, 40))
    assert (right_below[:, 0] > 0).all() and (right_below[:, 1] > 0).all() and (right_below[:, 2] > 0).all()
    left_above = _lift(_plane(), _box(5, 0, 25, 15))
    assert (left_above[:, 0] < 0).all() and (left_above[:, 1] < 0).all() and (left_above[:, 2] > 0).all()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0, MIN_M - 0.01, MAX_M + 0.5])
def test_a_bad_depth_gives_no_point(bad):
    metres = _plane()
    metres[10, 20] = bad
    points = _lift(metres, _box(10, 5, 30, 25))
    assert len(points) == 18 * 18 - 1
    assert np.isfinite(points).all()
    rows, cols = _pixels(points)
    assert not ((rows == 10) & (cols == 20)).any()


def test_the_range_limits_keep_a_depth_on_the_limit():
    assert len(_lift(_plane(1.0), _box(10, 5, 30, 25), min_m=1.0, max_m=1.0 + 1e-3)) == 18 * 18
    assert len(_lift(_plane(1.0), _box(10, 5, 30, 25), min_m=0.5, max_m=1.0)) == 18 * 18
    assert len(_lift(_plane(1.0), _box(10, 5, 30, 25), min_m=1.5, max_m=2.0)) == 0
    assert len(_lift(_plane(3.0), _box(10, 5, 30, 25), min_m=0.2, max_m=2.0)) == 0


@pytest.mark.parametrize(("millimetres", "kept"), [(199, False), (200, True), (3000, True), (3001, False)])
def test_16uc1_and_32fc1_keep_the_same_depths_at_the_default_range_limits(millimetres, kept):
    # MIN_M and MAX_M are the node's default range, 200 mm and 3000 mm, and both limits are inclusive.
    mask = np.ones((1, 1), bool)
    for raster in (_raster([[millimetres]], "16UC1"), _raster([[millimetres / 1000]], "32FC1")):
        assert depth_cover(mask, depth_to_metres(raster), MIN_M, MAX_M) == (1.0 if kept else 0.0), raster.encoding


@pytest.mark.parametrize("millimetres", [200, 3000])
def test_a_16uc1_depth_on_a_range_limit_lifts_like_the_same_32fc1_depth(millimetres):
    from_millimetres = _raster(np.full((RASTER_H, RASTER_W), millimetres), "16UC1")
    from_metres = _raster(np.full((RASTER_H, RASTER_W), millimetres / 1000), "32FC1")
    points = _lift(depth_to_metres(from_millimetres), _box(10, 5, 30, 25))
    assert len(points) == 18 * 18
    np.testing.assert_array_equal(points, _lift(depth_to_metres(from_metres), _box(10, 5, 30, 25)))


def test_a_depth_far_from_the_median_is_dropped_by_the_mad_window():
    # Three levels in equal shares: the median is 1.0 and the MAD 0.02, so the window is 0.0889 m.
    metres = _plane()
    metres[np.arange(RASTER_H) % 3 == 0, :] = 0.98
    metres[np.arange(RASTER_H) % 3 == 2, :] = 1.02
    metres[12, 15] = 1.08
    metres[12, 16] = 1.2
    points = _lift(metres, _box(10, 5, 30, 25))
    rows, cols = _pixels(points)
    assert ((rows == 12) & (cols == 15)).any()
    assert not ((rows == 12) & (cols == 16)).any()
    assert len(points) == 18 * 18 - 1


def test_a_zero_mad_still_keeps_small_noise_within_one_centimetre():
    metres = _plane()
    metres[12, 15] = 1.008
    metres[12, 16] = 1.02
    points = _lift(metres, _box(10, 5, 30, 25))
    rows, cols = _pixels(points)
    assert ((rows == 12) & (cols == 15)).any()
    assert not ((rows == 12) & (cols == 16)).any()


def test_a_thin_mask_falls_back_to_the_whole_mask():
    # 3 x 20 px: erosion leaves 18 pixels, under the floor. The whole mask has 60.
    points = _lift(_plane(), _box(10, 5, 30, 8))
    assert len(points) == 60


def test_the_fallback_applies_the_same_depth_rules():
    # 9 x 9 px: erosion leaves 49 pixels, all without a depth. The ring has 32, of which one is bad.
    metres = _plane()
    metres[6:13, 11:18] = float("nan")
    metres[5, 10] = 0.0
    points = _lift(metres, _box(10, 5, 19, 14))
    assert len(points) == 31
    assert (points[:, 2] == 1.0).all()


@pytest.mark.parametrize(("pixels", "expected"), [(MIN_CLOUD_POINTS - 1, 0), (MIN_CLOUD_POINTS, MIN_CLOUD_POINTS)])
def test_the_point_floor_of_the_whole_mask(pixels, expected):
    # One row: erosion leaves nothing, so the whole mask decides.
    assert len(_lift(_plane(), _box(0, 10, pixels, 11))) == expected


def test_thirty_eroded_points_stay_and_twenty_nine_fall_back():
    # 12 x 5 px: erosion leaves 10 x 3 = 30 pixels, and the whole mask has 60.
    mask = _box(10, 5, 22, 10)
    assert len(_lift(_plane(), mask)) == MIN_CLOUD_POINTS
    metres = _plane()
    metres[6, 11] = float("nan")
    assert len(_lift(metres, mask)) == 59


def test_too_few_points_give_an_empty_array_of_the_right_shape():
    for mask in (np.zeros((RASTER_H, RASTER_W), bool), _box(0, 10, 5, 11)):
        points = _lift(_plane(), mask)
        assert points.shape == (0, 3) and points.dtype == np.float32


def test_the_cap_keeps_evenly_spaced_pixels_in_row_major_order():
    # 100 x 100 px gives a radius of 5, so 90 x 90 = 8100 pixels survive.
    metres = _plane(1.0, 200, 300)
    mask = _box(50, 50, 150, 150, 200, 300)
    points = _lift(metres, mask)
    assert len(points) == MAX_CLOUD_POINTS
    rows, cols = _pixels(points)
    chosen = np.arange(MAX_CLOUD_POINTS) * 8100 // MAX_CLOUD_POINTS
    all_rows, all_cols = np.mgrid[55:145, 55:145]
    np.testing.assert_array_equal(rows, all_rows.ravel()[chosen])
    np.testing.assert_array_equal(cols, all_cols.ravel()[chosen])
    np.testing.assert_array_equal(points, _lift(metres, mask))


def test_exactly_the_cap_keeps_every_pixel():
    # 54 x 44 px gives a radius of 2, so 50 x 40 = 2000 pixels survive.
    points = _lift(_plane(1.0, 200, 300), _box(50, 50, 104, 94, 200, 300))
    assert len(points) == MAX_CLOUD_POINTS
    rows, cols = _pixels(points)
    all_rows, all_cols = np.mgrid[52:92, 52:102]
    np.testing.assert_array_equal(rows, all_rows.ravel())
    np.testing.assert_array_equal(cols, all_cols.ravel())


@pytest.mark.parametrize(
    ("height", "width", "radius"),
    [
        (1, 1, 1),
        (10, 100, 1),
        (20, 20, 1),
        (30, 30, 2),
        (50, 50, 2),  # 2.5 goes to the even side.
        (70, 70, 4),  # 3.5 goes to the even side.
        (100, 100, 5),
        (400, 400, 5),
        (20, 400, 1),
        (400, 20, 1),
    ],
)
def test_the_erosion_radius(height, width, radius):
    mask = np.zeros((500, 500), bool)
    mask[:height, :width] = True
    assert erosion_radius(mask) == radius


def test_the_erosion_radius_uses_the_extent_not_the_pixel_count():
    assert erosion_radius(np.eye(100, dtype=bool)) == 5


def test_an_empty_mask_has_no_erosion_radius():
    assert erosion_radius(np.zeros((5, 5), bool)) == 0


def test_a_mask_on_the_edge_of_the_raster_is_eroded_from_the_edge():
    # The whole raster: 40 px gives a radius of 2, and outside the raster counts as not in the mask.
    points = _lift(_plane(), np.ones((RASTER_H, RASTER_W), bool))
    rows, cols = _pixels(points)
    assert (rows.min(), rows.max(), cols.min(), cols.max()) == (2, 37, 2, 57)


def test_depth_cover_counts_the_valid_pixels_before_erosion():
    metres = _plane()
    mask = _box(0, 0, 10, 10)
    assert depth_cover(mask, metres, MIN_M, MAX_M) == 1.0
    # The outer ring would be gone after erosion, and it still counts.
    metres[0, :10] = float("nan")
    metres[1, :10] = 0.0
    metres[2, :10] = MAX_M + 1.0
    assert depth_cover(mask, metres, MIN_M, MAX_M) == pytest.approx(0.7)


def test_depth_cover_of_an_empty_mask_is_zero():
    assert depth_cover(np.zeros((RASTER_H, RASTER_W), bool), _plane(), MIN_M, MAX_M) == 0.0


# --- view quality ----------------------------------------------------------------

Q_H, Q_W = 100, 100
CENTRED_BOX = np.array([25.0, 25.0, 75.0, 75.0])
FLAT = np.zeros((Q_H, Q_W), np.float32)


def _noise(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (Q_H, Q_W, 3), dtype=np.uint8)


def _q_box(x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    return box_mask(Q_H, Q_W, (x1, y1, x2, y2))


def test_frame_laplacian_uses_the_recipe():
    image = _noise()
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    expected = cv2.Laplacian(grey, cv2.CV_32F, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    result = frame_laplacian(image)
    assert result.dtype == np.float32 and result.shape == (Q_H, Q_W)
    np.testing.assert_array_equal(result, expected)


def test_a_flat_image_has_no_laplacian_even_at_the_border():
    assert not frame_laplacian(np.full((Q_H, Q_W, 3), 200, np.uint8)).any()


def test_view_quality_is_the_mean_of_its_four_terms():
    # A tenth of the image gives a full area term. The box is centred, half the
    # mask has depth, and a flat image has no sharpness: 0.25 * (1 + 1 + 0.5 + 0).
    assert view_quality(_q_box(0, 45, 100, 55), np.array([0.0, 45.0, 100.0, 55.0]), FLAT, 0.5) == pytest.approx(0.625)


def test_the_area_term_is_the_square_root_of_the_share():
    # 250 px is a quarter of the full area. A box on the corner is not centred.
    score = view_quality(_q_box(0, 0, 25, 10), np.array([0.0, 0.0, 0.0, 0.0]), FLAT, 0.0)
    assert score == pytest.approx(0.25 * 0.5)


@pytest.mark.parametrize(
    ("box", "centred"),
    [
        ((50.0, 50.0, 50.0, 50.0), 1.0),
        ((0.0, 0.0, 0.0, 0.0), 0.0),
        ((100.0, 100.0, 100.0, 100.0), 0.0),
        ((100.0, 50.0, 100.0, 50.0), 1.0 - 1.0 / math.sqrt(2.0)),
    ],
)
def test_the_centred_term_falls_with_the_distance_from_the_centre(box, centred):
    # Area, depth and sharpness are all 0 or known: a tenth of the image gives 1.
    score = view_quality(_q_box(0, 0, 100, 10), np.array(box), FLAT, 0.0)
    assert score == pytest.approx(0.25 * (1.0 + centred))


def test_more_depth_cover_scores_higher_by_a_quarter_of_the_difference():
    mask = _q_box(25, 25, 75, 75)
    low, high = view_quality(mask, CENTRED_BOX, FLAT, 0.25), view_quality(mask, CENTRED_BOX, FLAT, 0.75)
    assert high - low == pytest.approx(0.125)


def test_a_sharper_view_scores_higher():
    image = _noise()
    mask = _q_box(25, 25, 75, 75)
    sharp = view_quality(mask, CENTRED_BOX, frame_laplacian(image), 0.0)
    blurred = view_quality(mask, CENTRED_BOX, frame_laplacian(cv2.GaussianBlur(image, (9, 9), 3.0)), 0.0)
    assert sharp > blurred


def test_a_mask_over_an_edge_scores_higher_than_a_mask_over_a_flat_part():
    # The same mask size and box, so only the sharpness term can differ.
    image = np.zeros((Q_H, Q_W, 3), np.uint8)
    image[:, 50:] = 255
    laplacian = frame_laplacian(image)
    over_edge = view_quality(_q_box(40, 0, 60, 20), CENTRED_BOX, laplacian, 0.0)
    over_flat = view_quality(_q_box(70, 0, 90, 20), CENTRED_BOX, laplacian, 0.0)
    assert over_edge > over_flat


def test_sharpness_needs_fifty_mask_pixels():
    laplacian = frame_laplacian(_noise())
    below = _q_box(25, 25, 74, 26)  # 49 px
    enough = _q_box(25, 25, 75, 26)  # 50 px
    score_below = view_quality(below, CENTRED_BOX, laplacian, 0.0)
    score_enough = view_quality(enough, CENTRED_BOX, laplacian, 0.0)
    assert score_below == pytest.approx(0.25 * (math.sqrt(49 / 1000) + 1.0))
    # The noise image's Laplacian variance is far above the full mark.
    assert score_enough == pytest.approx(0.25 * (math.sqrt(50 / 1000) + 1.0 + 1.0))


def test_an_empty_mask_scores_zero():
    assert view_quality(np.zeros((Q_H, Q_W), bool), CENTRED_BOX, FLAT, 1.0) == 0.0


def test_the_best_view_scores_one():
    mask = np.ones((Q_H, Q_W), bool)
    assert view_quality(mask, np.array([0.0, 0.0, 100.0, 100.0]), frame_laplacian(_noise()), 1.0) == 1.0


def test_every_score_is_finite_and_in_range():
    rng = np.random.default_rng(3)
    laplacian = frame_laplacian(_noise())
    for _ in range(200):
        x1, x2 = sorted(rng.integers(0, Q_W + 1, 2).tolist())
        y1, y2 = sorted(rng.integers(0, Q_H + 1, 2).tolist())
        score = view_quality(_q_box(x1, y1, x2, y2), np.array([x1, y1, x2, y2], float), laplacian, float(rng.random()))
        assert math.isfinite(score) and 0.0 <= score <= 1.0
