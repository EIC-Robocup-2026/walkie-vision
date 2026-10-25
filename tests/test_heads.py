"""Tests of walkie_vision.face.heads with a stub analyzer: no ONNX model is loaded."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from walkie_vision.face import HeadCropConfig, HeadFaceRecognizer, face_align
from walkie_vision.face.heads import box_iou, clip_faces, head_crop, input_size_for, pick_face
from walkie_vision.fakes import hash_vector
from walkie_vision.providers import ARCFACE_W600K_R50, PersonFaceRecognizer, PersonPose

WIDTH, HEIGHT = 1280, 720
CONFIG = HeadCropConfig()


def _pose(
    box: list[float],
    head: tuple[float, float] | None,
    spread: float = 30.0,
    head_score: float = 0.9,
    shoulders: float | None = None,
    shoulder_score: float = 0.9,
) -> PersonPose:
    """A person whose head keypoints have their mean at head and span spread in x (spread / 2 in y)."""
    kp = np.zeros((17, 2), dtype=np.float32)
    scores = np.full(17, 0.05, dtype=np.float32)
    x1, y1, x2, y2 = box
    kp[:] = ((x1 + x2) / 2, (y1 + y2) / 2)
    if head is not None:
        cx, cy = head
        s = spread
        kp[0] = (cx, cy)  # nose
        kp[1], kp[2] = (cx - s / 4, cy - s / 4), (cx + s / 4, cy - s / 4)  # eyes
        kp[3], kp[4] = (cx - s / 2, cy + s / 4), (cx + s / 2, cy + s / 4)  # ears
        scores[:5] = head_score
        if shoulders is not None:
            kp[5], kp[6] = (cx - shoulders / 2, cy + 2 * s), (cx + shoulders / 2, cy + 2 * s)
            scores[5:7] = shoulder_score
    return PersonPose(box=np.array(box, dtype=np.float64), score=0.9, keypoints=kp, keypoint_scores=scores)


# --- head_crop ---------------------------------------------------------------


def test_head_crop_is_centred_on_the_head_points_and_sized_by_their_spread():
    box, centre, side = head_crop(_pose([100, 100, 400, 700], (250, 200), spread=30), (HEIGHT, WIDTH), CONFIG)
    np.testing.assert_allclose(centre, [250, 200])
    assert side == pytest.approx(4.0 * 30)
    assert box == (190, 140, 310, 260)


def test_head_crop_uses_only_confident_head_points():
    pose = _pose([100, 100, 400, 700], (250, 200), spread=40)
    pose.keypoint_scores[3:5] = 0.1  # ears not seen
    _, centre, side = head_crop(pose, (HEIGHT, WIDTH), CONFIG)
    # Nose and eyes: mean y is cy - s/6, spread is s/2 in x and s/4 in y.
    np.testing.assert_allclose(centre, [250, 200 - 40 / 6], atol=1e-4)
    assert side == pytest.approx(4.0 * 20)


def test_head_crop_side_from_the_shoulders_when_wider():
    _, _, side = head_crop(_pose([100, 100, 400, 700], (250, 200), spread=30, shoulders=200), (HEIGHT, WIDTH), CONFIG)
    assert side == pytest.approx(1.2 * 200)
    # Shoulders below the score threshold do not count.
    pose = _pose([100, 100, 400, 700], (250, 200), spread=30, shoulders=200, shoulder_score=0.2)
    assert head_crop(pose, (HEIGHT, WIDTH), CONFIG)[2] == pytest.approx(120)


def test_head_crop_falls_back_to_the_box():
    pose = _pose([100, 100, 300, 700], (200, 200))
    pose.keypoint_scores[1:5] = 0.0  # only the nose
    _, centre, side = head_crop(pose, (HEIGHT, WIDTH), CONFIG)
    np.testing.assert_allclose(centre, [200, 200])
    assert side == pytest.approx(min(0.8 * 200, 0.4 * 600))
    pose = _pose([100, 100, 500, 300], (200, 150))
    pose.keypoint_scores[1:5] = 0.0
    assert head_crop(pose, (HEIGHT, WIDTH), CONFIG)[2] == pytest.approx(0.4 * 200)


def test_head_crop_has_a_minimum_side():
    _, _, side = head_crop(_pose([100, 100, 400, 700], (250, 200), spread=5), (HEIGHT, WIDTH), CONFIG)
    assert side == 64
    box, _, side = head_crop(
        _pose([100, 100, 400, 700], (250, 200), spread=5), (HEIGHT, WIDTH), HeadCropConfig(min_crop_px=100)
    )
    assert side == 100 and box == (200, 150, 300, 250)


def test_head_crop_rounds_outwards_and_clips_at_the_edges():
    box, _, _ = head_crop(_pose([0, 0, 400, 700], (100.5, 200.25), spread=16), (HEIGHT, WIDTH), CONFIG)
    assert box == (68, 168, 133, 233)
    box, _, side = head_crop(_pose([0, 0, 400, 700], (10, 10), spread=30), (HEIGHT, WIDTH), CONFIG)
    assert side == 120 and box == (0, 0, 70, 70)
    box, _, _ = head_crop(_pose([1000, 0, 1280, 720], (1275, 715), spread=30), (HEIGHT, WIDTH), CONFIG)
    assert box == (1215, 655, 1280, 720)


def test_head_crop_is_none_without_a_head():
    assert head_crop(_pose([100, 100, 400, 700], None), (HEIGHT, WIDTH), CONFIG) is None
    pose = _pose([100, 100, 400, 700], (250, 200), head_score=0.29, shoulders=200)
    assert head_crop(pose, (HEIGHT, WIDTH), CONFIG) is None


def test_head_crop_is_none_when_almost_nothing_is_in_the_image():
    pose = _pose([1200, 0, 1280, 720], (WIDTH + 30, 200), spread=10)
    assert head_crop(pose, (HEIGHT, WIDTH), CONFIG) is None  # 2 px left


# --- input_size_for and HeadCropConfig ----------------------------------------


@pytest.mark.parametrize(
    ("w", "h", "size"),
    [(160, 100, 160), (161, 10, 256), (10, 256, 256), (257, 257, 384), (640, 1, 640), (641, 641, 640), (0, 0, 160)],
)
def test_input_size_for(w, h, size):
    assert input_size_for(w, h, (160, 256, 384, 640)) == size


def test_input_size_for_unsorted_sizes():
    assert input_size_for(100, 100, (640, 160, 256)) == 160


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_sizes": ()},
        {"input_sizes": (100,)},
        {"input_sizes": (160, 0)},
        {"input_sizes": (-32,)},
        {"input_sizes": (160.0,)},
        {"max_people": -1},
        {"dedupe_iou": 0.0},
        {"dedupe_iou": 1.5},
        {"max_face_offset": 0.0},
        {"max_face_offset": 1.01},
        {"max_face_offset": -0.1},
    ],
)
def test_head_crop_config_rejects(kwargs):
    with pytest.raises(ValueError):
        HeadCropConfig(**kwargs)


def test_head_crop_config_accepts_the_bounds_and_normalises_sizes():
    cfg = HeadCropConfig(input_sizes=[np.int64(32), 64], max_people=0, dedupe_iou=1.0, max_face_offset=1.0)
    assert cfg.input_sizes == (32, 64) and all(type(s) is int for s in cfg.input_sizes)
    assert HeadCropConfig(input_sizes=(160, 256, 384, 640), max_people=6, max_face_offset=0.25) == HeadCropConfig()


# --- clip_faces, pick_face, box_iou --------------------------------------------


def test_clip_faces():
    det = np.array(
        [[600, 100, 700, 200, 0.9], [-50, -40, -10, -5, 0.8], [10, 10, 50, 60, 0.7]], dtype=np.float32
    )
    kpss = np.stack([np.full((5, 2), 650.0), np.full((5, 2), -30.0), np.full((5, 2), 30.0)])
    out_det, out_kps = clip_faces(det, kpss, (480, 640))
    np.testing.assert_allclose(out_det, [[600, 100, 640, 200, 0.9], [10, 10, 50, 60, 0.7]], rtol=1e-6)
    # Landmarks are not clipped; the dropped face's landmarks go with it.
    np.testing.assert_array_equal(out_kps, [np.full((5, 2), 650.0), np.full((5, 2), 30.0)])
    # The inputs are not changed.
    assert det[0, 2] == 700


def test_clip_faces_with_no_faces():
    det, kps = clip_faces(np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32), (480, 640))
    assert det.shape == (0, 5) and kps.shape == (0, 5, 2)


def _face_box(cx: float, cy: float, score: float = 0.9, w: float = 20, h: float = 26) -> list[float]:
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, score]


def test_pick_face_needs_the_centre_inside_the_grown_box():
    person = np.array([100.0, 100.0, 200.0, 400.0])  # grown: x 90..210, y 70..430
    head = np.array([205.0, 120.0])
    assert pick_face(np.array([_face_box(215, 120)]), head, person, 30) is None
    assert pick_face(np.array([_face_box(208, 120)]), head, person, 30) == 0
    assert pick_face(np.array([_face_box(210, 120)]), head, person, 30) == 0  # the edge counts as inside


def test_pick_face_distance_gate():
    person = np.array([0.0, 0.0, 400.0, 400.0])
    head = np.array([200.0, 100.0])
    boxes = np.array([_face_box(250, 100)])
    assert pick_face(boxes, head, person, 49.9) is None
    assert pick_face(boxes, head, person, 50.0) == 0


def test_pick_face_nearest_wins_and_ties_go_to_the_higher_score():
    person = np.array([0.0, 0.0, 400.0, 400.0])
    head = np.array([200.0, 100.0])
    boxes = np.array([_face_box(212, 100, 0.99), _face_box(205, 100, 0.6), _face_box(250, 100, 0.99)])
    assert pick_face(boxes, head, person, 30) == 1
    tie = np.array([_face_box(190, 100, 0.6), _face_box(210, 100, 0.9), _face_box(200, 110, 0.7)])
    assert pick_face(tie, head, person, 30) == 1
    assert pick_face(np.zeros((0, 5)), head, person, 30) is None


def test_box_iou():
    assert box_iou(np.array([0, 0, 10, 10]), np.array([0, 0, 10, 10])) == pytest.approx(1.0)
    assert box_iou(np.array([0, 0, 10, 10]), np.array([5, 0, 15, 10])) == pytest.approx(50 / 150)
    assert box_iou(np.array([0, 0, 10, 10]), np.array([20, 20, 30, 30])) == 0.0
    assert box_iou(np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])) == 0.0


# --- faces_for with a stub analyzer --------------------------------------------


def _coded_image(width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    """An image whose every pixel encodes its own x and y, so a crop knows where it came from."""
    x = np.broadcast_to(np.arange(width)[None, :], (height, width))
    y = np.broadcast_to(np.arange(height)[:, None], (height, width))
    return np.stack([x % 256, y % 256, (x // 256) * 16 + y // 256], axis=-1).astype(np.uint8)


def _crop_origin(crop: np.ndarray) -> tuple[int, int]:
    b0, b1, b2 = (int(v) for v in crop[0, 0])
    return (b2 // 16) * 256 + b0, (b2 % 16) * 256 + b1


@dataclass
class _ImageFace:
    bbox: tuple[float, float, float, float]
    score: float = 0.9

    def kps(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox
        w, h = x2 - x1, y2 - y1
        rel = [(0.3, 0.4), (0.7, 0.4), (0.5, 0.6), (0.35, 0.8), (0.65, 0.8)]
        return np.array([(x1 + a * w, y1 + b * h) for a, b in rel], dtype=np.float32)

    def centre(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2, (y1 + y2) / 2


class _StubDetector:
    """Finds every face of the image whose centre is inside the crop, in crop pixels, like SCRFD would."""

    def __init__(self, faces: list[_ImageFace]) -> None:
        self.faces = faces
        self.calls: list[tuple[tuple[int, ...], tuple[int, int] | None, int]] = []

    def detect(self, img, input_size=None, max_num=0, metric="default"):
        self.calls.append((img.shape, input_size, max_num))
        ox, oy = _crop_origin(img)
        h, w = img.shape[:2]
        rows, kps = [], []
        for f in self.faces:
            cx, cy = f.centre()
            if ox <= cx < ox + w and oy <= cy < oy + h:
                x1, y1, x2, y2 = f.bbox
                rows.append([x1 - ox, y1 - oy, x2 - ox, y2 - oy, f.score])
                kps.append(f.kps() - (ox, oy))
        return np.array(rows, dtype=np.float32).reshape(-1, 5), np.array(kps, dtype=np.float32).reshape(-1, 5, 2)


class _StubRecognizer:
    input_size = (112, 112)

    def __init__(self) -> None:
        self.batches: list[int] = []

    def get_feat(self, imgs):
        imgs = imgs if isinstance(imgs, list) else [imgs]
        self.batches.append(len(imgs))
        return np.stack([hash_vector(np.ascontiguousarray(i).tobytes(), 512) for i in imgs]) * 3.0


class _StubAnalyzer:
    def __init__(self, faces: list[_ImageFace]) -> None:
        self.detector = _StubDetector(faces)
        self.recognizer = _StubRecognizer()


def _recognizer(faces: list[_ImageFace], **config) -> tuple[HeadFaceRecognizer, _StubAnalyzer]:
    stub = _StubAnalyzer(faces)
    return HeadFaceRecognizer(stub, HeadCropConfig(**config), warm_up=False), stub


def test_faces_are_in_whole_image_pixels_and_inside_the_image():
    f1 = _ImageFace((225, 170, 275, 230), 0.88)
    f2 = _ImageFace((1240, 170, 1295, 230), 0.77)  # runs past the right edge
    people = [_pose([100, 100, 400, 700], (250, 200)), _pose([1100, 100, 1280, 700], (1265, 200))]
    rec, stub = _recognizer([f1, f2])
    out = rec.faces_for(_coded_image(), people)
    assert len(out) == 2 and all(f is not None for f in out)
    np.testing.assert_allclose(out[0].bbox, f1.bbox)
    np.testing.assert_allclose(out[0].kps, f1.kps(), atol=1e-4)
    assert out[0].det_score == pytest.approx(0.88)
    np.testing.assert_allclose(out[1].bbox, [1240, 170, 1280, 230])
    # The landmarks are where they really are, not clipped.
    np.testing.assert_allclose(out[1].kps, f2.kps(), atol=1e-4)
    for f in out:
        assert 0 <= f.bbox[0] < f.bbox[2] <= WIDTH and 0 <= f.bbox[1] < f.bbox[3] <= HEIGHT
        assert f.embedding.shape == (512,) and f.embedding.dtype == np.float32
        # Not normalised: convert does that.
        assert np.linalg.norm(f.embedding) == pytest.approx(3.0, rel=1e-5)
    assert not np.allclose(out[0].embedding, out[1].embedding)
    # One SCRFD call per person, at the smallest size that holds a 120 px crop.
    assert [c[1:] for c in stub.detector.calls] == [((160, 160), 0), ((160, 160), 0)]
    assert stub.detector.calls[0][0] == (120, 120, 3)
    assert stub.recognizer.batches == [2]


def test_max_people_keeps_the_largest_people():
    faces = [_ImageFace((140, 170, 160, 230)), _ImageFace((540, 170, 560, 230)), _ImageFace((940, 170, 960, 230))]
    people = [
        _pose([100, 150, 200, 400], (150, 200)),  # smallest
        _pose([400, 100, 700, 700], (550, 200)),  # largest
        _pose([850, 100, 1050, 700], (950, 200)),
    ]
    rec, stub = _recognizer(faces, max_people=2)
    out = rec.faces_for(_coded_image(), people)
    assert out[0] is None and out[1] is not None and out[2] is not None
    np.testing.assert_allclose(out[1].bbox, faces[1].bbox)
    np.testing.assert_allclose(out[2].bbox, faces[2].bbox)
    assert len(stub.detector.calls) == 2

    rec, stub = _recognizer(faces, max_people=0)
    assert rec.faces_for(_coded_image(), people) == [None, None, None]
    assert stub.detector.calls == [] and stub.recognizer.batches == []


def test_a_face_found_for_two_people_goes_to_the_nearer_head():
    face = _ImageFace((480, 170, 530, 230))  # centre (505, 200)
    people = [
        _pose([450, 100, 650, 700], (520, 200)),  # 15 px from the face; larger box, so searched first
        _pose([400, 100, 600, 690], (500, 200)),  # 5 px from the face
    ]
    rec, stub = _recognizer([face])
    out = rec.faces_for(_coded_image(), people)
    assert out[0] is None
    assert out[1] is not None
    np.testing.assert_allclose(out[1].bbox, face.bbox)
    assert stub.recognizer.batches == [1]


def test_a_neighbours_face_is_not_given_to_a_person_whose_head_is_far_from_it():
    face_b = _ImageFace((420, 180, 470, 240))  # centre (445, 210)
    a = _pose([300, 100, 500, 700], (400, 200))  # crop 340..460 holds B's face, 46 px from A's head
    b = _pose([380, 100, 580, 700], (445, 210))
    rec, _ = _recognizer([face_b])
    out = rec.faces_for(_coded_image(), [a, b])
    assert out[0] is None
    np.testing.assert_allclose(out[1].bbox, face_b.bbox)
    # Also when B has no box of their own.
    assert rec.faces_for(_coded_image(), [a]) == [None]
    # With a looser gate the face would go to A: the gate is what stops it.
    loose, _ = _recognizer([face_b], max_face_offset=0.5)
    assert loose.faces_for(_coded_image(), [a])[0] is not None


def test_a_face_that_cannot_be_aligned_becomes_none(monkeypatch):
    faces = [_ImageFace((225, 170, 275, 230)), _ImageFace((625, 170, 675, 230)), _ImageFace((1025, 170, 1075, 230))]
    people = [_pose([100, 100, 400, 700], (250, 200)), _pose([500, 100, 800, 700], (650, 200)),
              _pose([900, 100, 1200, 700], (1050, 200))]
    bad = faces[1].kps()
    real = face_align.norm_crop

    def norm_crop(img, landmark, image_size=112):
        if np.allclose(landmark, bad, atol=1e-3):
            raise ValueError("cannot align: the landmarks are degenerate")
        return real(img, landmark, image_size=image_size)

    monkeypatch.setattr(face_align, "norm_crop", norm_crop)
    rec, stub = _recognizer(faces)
    out = rec.faces_for(_coded_image(), people)
    assert out[1] is None
    assert out[0].embedding.shape == (512,) and out[2].embedding.shape == (512,)
    assert stub.recognizer.batches == [2]


def test_faces_for_keeps_the_input_order_and_length():
    faces = [_ImageFace((225, 170, 275, 230)), _ImageFace((925, 170, 975, 230))]
    people = [
        _pose([850, 100, 1050, 700], (950, 200)),
        _pose([600, 100, 700, 700], None),  # no head keypoints
        _pose([100, 100, 400, 700], (250, 200)),
        _pose([700, 100, 800, 700], (750, 200)),  # a head but no face
    ]
    rec, stub = _recognizer(faces)
    out = rec.faces_for(_coded_image(), people)
    assert len(out) == 4
    np.testing.assert_allclose(out[0].bbox, faces[1].bbox)
    assert out[1] is None and out[3] is None
    np.testing.assert_allclose(out[2].bbox, faces[0].bbox)
    assert stub.recognizer.batches == [2]
    assert rec.faces_for(_coded_image(), []) == []
    assert stub.recognizer.batches == [2]


def test_faces_for_with_no_faces_skips_the_recogniser():
    rec, stub = _recognizer([])
    assert rec.faces_for(_coded_image(), [_pose([100, 100, 400, 700], (250, 200))]) == [None]
    assert len(stub.detector.calls) == 1 and stub.recognizer.batches == []


def test_warm_up_runs_every_input_and_batch_size():
    stub = _StubAnalyzer([])
    HeadFaceRecognizer(stub, HeadCropConfig(input_sizes=(160, 320), max_people=3))
    assert [(c[0], c[1]) for c in stub.detector.calls] == [((160, 160, 3), (160, 160)), ((320, 320, 3), (320, 320))]
    assert stub.recognizer.batches == [1, 2, 3]
    stub = _StubAnalyzer([])
    HeadFaceRecognizer(stub, HeadCropConfig(max_people=0))
    assert len(stub.detector.calls) == 4 and stub.recognizer.batches == [1]
    stub = _StubAnalyzer([])
    HeadFaceRecognizer(stub, warm_up=False)
    assert stub.detector.calls == [] and stub.recognizer.batches == []


def test_head_face_recognizer_is_a_person_face_recognizer():
    stub = _StubAnalyzer([])
    rec = HeadFaceRecognizer(stub, warm_up=False)
    assert isinstance(rec, PersonFaceRecognizer)
    assert rec.analyzer is stub
    assert rec.model_name == ARCFACE_W600K_R50 and rec.dim == 512
    assert rec.config == HeadCropConfig()
