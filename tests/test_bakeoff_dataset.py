"""Data set files, crops and size buckets of the embedding bake-off (tools/bakeoff/dataset.py)."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

_TOOL = str(Path(__file__).resolve().parents[1] / "tools" / "bakeoff")
if _TOOL not in sys.path:
    sys.path.insert(0, _TOOL)

from dataset import (  # noqa: E402
    Crop,
    DatasetError,
    Object,
    Query,
    check,
    cut_crops,
    downscale,
    list_images,
    make_buckets,
    read_manifest,
    read_objects,
    read_queries,
    similar_pairs,
    write_manifest,
)

OBJECTS_CSV = """object_id,label,pair,note
red_mug_1,mug,red mugs,left
red_mug_2,mug,red mugs,right
cup_blue,cup,,
apple,apple,,
"""


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def objects(tmp_path):
    return read_objects(_write(tmp_path / "objects.csv", OBJECTS_CSV))


def test_read_objects(objects):
    assert [o.object_id for o in objects] == ["red_mug_1", "red_mug_2", "cup_blue", "apple"]
    assert objects[0] == Object("red_mug_1", "mug", "red mugs", "left")
    assert similar_pairs(objects) == [("red mugs", "red_mug_1", "red_mug_2")]


def test_similar_pairs_of_a_group_of_three():
    objs = [Object(i, "mug", "g") for i in ("a", "b", "c")]
    assert similar_pairs(objs) == [("g", "a", "b"), ("g", "a", "c"), ("g", "b", "c")]


@pytest.mark.parametrize(
    "text, message",
    [
        ("object_id,label\na,mug\na,cup\n", "appears twice"),
        ("object_id,label\nred mug,mug\n", "must be letters"),
        ("object_id,label\n-,mug\n", "must be letters"),
        ("object_id,label\na,\n", "label is empty"),
        ("object_id,pair\na,x\n", "missing column(s) label"),
        ("object_id,label\n", "no objects"),
    ],
)
def test_read_objects_errors(tmp_path, text, message):
    with pytest.raises(DatasetError, match=message.replace("(", r"\(").replace(")", r"\)")):
        read_objects(_write(tmp_path / "objects.csv", text))


def test_read_queries(tmp_path, objects):
    path = _write(
        tmp_path / "queries.csv",
        'query,object_ids\nthe red mug,red_mug_1 red_mug_2\n"a cup, blue",cup_blue\nan apple,apple apple\n',
    )
    assert read_queries(path, objects) == [
        Query("the red mug", ("red_mug_1", "red_mug_2")),
        Query("a cup, blue", ("cup_blue",)),
        Query("an apple", ("apple",)),
    ]


@pytest.mark.parametrize(
    "text, message",
    [
        ("query,object_ids\nthe pear,pear\n", "unknown object id"),
        ("query,object_ids\nan apple,apple\nan apple,cup_blue\n", "appears twice"),
        ("query,object_ids\nan apple,\n", "object_ids is empty"),
    ],
)
def test_read_queries_errors(tmp_path, objects, text, message):
    with pytest.raises(DatasetError, match=message):
        read_queries(_write(tmp_path / "queries.csv", text), objects)


MANIFEST_HEADER = "crop_id,image,x1,y1,x2,y2,label,score,object_id\n"


def test_read_manifest_keeps_reviewed_rows_in_order(tmp_path, objects):
    path = _write(
        tmp_path / "manifest.csv",
        MANIFEST_HEADER
        + "f2_00,f2.png,10,10,50,40,mug,0.910,red_mug_1\n"
        + "f1_01,f1.png,0,0,20,20,mug,0.400,-\n"
        + "f1_00,f1.png,5,6,25,36,cup,0.800,cup_blue\n"
        + "extra,f1.png,1,2,3,4,,,apple\n",
    )
    crops = read_manifest(path, objects)
    assert [c.crop_id for c in crops] == ["extra", "f1_00", "f2_00"]
    assert crops[1] == Crop("f1_00", "f1.png", (5, 6, 25, 36), "cup_blue", "cup", 0.8)
    assert crops[0].score is None


def test_read_manifest_lists_unreviewed_rows(tmp_path, objects):
    path = _write(
        tmp_path / "manifest.csv",
        MANIFEST_HEADER + "a,f.png,0,0,9,9,mug,0.9,\nb,f.png,0,0,9,9,cup,0.8,cup_blue\nc,f.png,0,0,9,9,mug,0.7,\n",
    )
    with pytest.raises(DatasetError) as e:
        read_manifest(path, objects)
    assert "2 row(s) not reviewed" in str(e.value)
    assert "line 2 (a, proposed mug)" in str(e.value)
    assert "line 4 (c, proposed mug)" in str(e.value)


@pytest.mark.parametrize(
    "row, message",
    [
        ("a,f.png,0,0,9,9,mug,0.9,pear", "unknown object_id"),
        ("a,f.png,9,0,9,9,mug,0.9,apple", "needs 0 <= x1 < x2"),
        ("a,f.png,-1,0,9,9,mug,0.9,apple", "needs 0 <= x1 < x2"),
        ("a,f.png,0.5,0,9,9,mug,0.9,apple", "not a whole number"),
        ("a,f.png,0,0,9,9,mug,high,apple", "not a number"),
        ("a,f.png,0,0,9,9,mug,0.9,apple\na,f.png,0,0,9,9,mug,0.9,apple", "appears twice"),
        ("a,f.png,0,0,9,9,mug,0.9,-", "no rows are kept"),
    ],
)
def test_read_manifest_errors(tmp_path, objects, row, message):
    with pytest.raises(DatasetError, match=message):
        read_manifest(_write(tmp_path / "manifest.csv", MANIFEST_HEADER + row + "\n"), objects)


def test_write_then_read_manifest(tmp_path, objects):
    crops = [Crop("x", "f.png", (1, 2, 30, 40), "apple", "apple", 0.5), Crop("y", "f.png", (0, 0, 5, 5), "cup_blue")]
    write_manifest(tmp_path / "m.csv", crops)
    assert read_manifest(tmp_path / "m.csv", objects) == crops


def test_check_errors_and_warnings(objects):
    queries = [Query("the red mug", ("red_mug_1", "red_mug_2")), Query("a cup", ("cup_blue",))]
    crops = [Crop("a", "f.png", (0, 0, 9, 9), "red_mug_1"), Crop("b", "f.png", (0, 0, 9, 9), "cup_blue")]
    warnings = check(objects, queries, crops)
    assert any("red_mug_2 has no crop" in w for w in warnings)
    assert any("object apple has no crop" in w for w in warnings)
    assert any("pair red mugs" in w for w in warnings)
    with pytest.raises(DatasetError, match="none of its objects"):
        check(objects, [Query("an apple", ("apple",))], crops)


def test_downscale_sets_the_longest_side():
    im = Image.new("RGB", (100, 50))
    assert downscale(im, 32).size == (32, 16)
    assert downscale(im, 100).size == (100, 50)
    assert downscale(im, 101) is None
    assert downscale(Image.new("RGB", (7, 300)), 48).size == (1, 48)


def test_downscale_averages_pixels():
    # Left half black, right half 200: area averaging keeps both halves exact.
    a = np.zeros((32, 64, 3), np.uint8)
    a[:, 32:] = 200
    small = np.asarray(downscale(Image.fromarray(a), 32))
    assert small.shape == (16, 32, 3)
    assert (small[:, :16] == 0).all() and (small[:, 16:] == 200).all()


def test_buckets_skip_crops_that_are_too_small():
    native = [Image.new("RGB", (40, 10)), Image.new("RGB", (30, 70)), Image.new("RGB", (100, 100))]
    buckets = make_buckets(native)
    assert {b: idx for b, (idx, _) in buckets.items()} == {
        "native": [0, 1, 2],
        32: [0, 1, 2],
        48: [1, 2],
        64: [1, 2],
        96: [2],
    }
    assert [im.size for im in buckets[64][1]] == [(27, 64), (64, 64)]


def test_cut_crops(tmp_path):
    a = np.arange(20 * 30 * 3, dtype=np.uint32).reshape(20, 30, 3).astype(np.uint8)
    Image.fromarray(a).save(tmp_path / "f.png")
    crops = cut_crops(tmp_path, [Crop("a", "f.png", (3, 4, 13, 9), "x")])
    np.testing.assert_array_equal(np.asarray(crops[0]), a[4:9, 3:13])
    with pytest.raises(DatasetError, match="outside f.png"):
        cut_crops(tmp_path, [Crop("a", "f.png", (0, 0, 31, 5), "x")])
    with pytest.raises(DatasetError, match="not found"):
        cut_crops(tmp_path, [Crop("a", "g.png", (0, 0, 3, 5), "x")])


def test_cut_crops_applies_the_exif_orientation(tmp_path):
    # Orientation 6 stores the pixels turned by 90 degrees. cv2.imread, and
    # so YOLOE and the review overlays, turn them back; the crops must too.
    stored = np.zeros((40, 60, 3), np.uint8)
    stored[:, :30] = (255, 0, 0)
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.fromarray(stored).save(tmp_path / "phone.jpg", exif=exif, quality=95)
    shown = cv2.imread(str(tmp_path / "phone.jpg"))[..., ::-1]
    assert shown.shape[:2] == (60, 40)
    (crop,) = cut_crops(tmp_path, [Crop("a", "phone.jpg", (0, 0, 40, 30), "x")])
    np.testing.assert_allclose(np.asarray(crop, dtype=float), shown[:30].astype(float), atol=8)


def test_list_images_refuses_names_that_differ_only_in_the_suffix(tmp_path):
    for name in ("a.png", "a.jpg", "b.png", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    with pytest.raises(DatasetError, match="a.jpg, a.png"):
        list_images(tmp_path)
    (tmp_path / "a.jpg").unlink()
    assert [p.name for p in list_images(tmp_path)] == ["a.png", "b.png"]
