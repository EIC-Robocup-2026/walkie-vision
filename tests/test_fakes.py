import numpy as np
import pytest

from walkie_vision.fakes import (
    FakeEmbedder,
    FakeFaceRecognizer,
    FakeObjectDetector,
    FakePoseEstimator,
    FakeReIdentifier,
    hash_vector,
)
from walkie_vision.providers import (
    Detection,
    Embedder,
    Face,
    FaceRecognizer,
    ObjectDetector,
    PersonPose,
    PoseEstimator,
    ReIdentifier,
)

IMAGE = np.zeros((480, 640, 3), dtype=np.uint8)


def _pose() -> PersonPose:
    return PersonPose(
        box=np.array([100.0, 50.0, 300.0, 470.0]),
        score=0.9,
        keypoints=np.arange(34, dtype=np.float32).reshape(17, 2),
        keypoint_scores=np.full(17, 0.8, dtype=np.float32),
        track_id=4,
    )


def test_fakes_implement_the_protocols():
    assert isinstance(FakeObjectDetector([]), ObjectDetector)
    assert isinstance(FakeEmbedder(), Embedder)
    assert isinstance(FakePoseEstimator([]), PoseEstimator)
    assert isinstance(FakeReIdentifier(), ReIdentifier)
    assert isinstance(FakeFaceRecognizer([]), FaceRecognizer)


def test_fake_model_names_cannot_match_a_database_table():
    for provider in (FakeEmbedder(), FakeReIdentifier(), FakeFaceRecognizer([])):
        assert provider.model_name.startswith("fake_")


def test_hash_vector_is_deterministic_and_unit_length():
    a = hash_vector("a red mug", 512)
    assert a.dtype == np.float32 and a.shape == (512,)
    assert np.isclose(np.linalg.norm(a), 1.0)
    np.testing.assert_array_equal(a, hash_vector(b"a red mug", 512))
    # Different inputs give nearly orthogonal vectors.
    assert abs(float(a @ hash_vector("a blue mug", 512))) < 0.2


def test_fake_embedder_shapes_and_determinism():
    embedder = FakeEmbedder(dim=64)
    texts = embedder.embed_texts(["mug", "chair", "mug"])
    assert texts.shape == (3, 64) and texts.dtype == np.float32
    np.testing.assert_array_equal(texts[0], texts[2])
    images = embedder.embed_images([IMAGE, IMAGE[:240]])
    assert images.shape == (2, 64)
    # The same bytes in another shape are another image.
    assert not np.array_equal(images[0], images[1])
    assert embedder.embed_texts([]).shape == (0, 64)


def test_fake_detector_filters_by_class_and_clips_boxes():
    detector = FakeObjectDetector(
        [
            Detection(box=np.array([600.0, 400.0, 700.0, 500.0]), label="mug", score=0.8, track_id=3),
            Detection(box=np.array([10.0, 10.0, 50.0, 50.0]), label="chair", score=0.7),
            Detection(box=np.array([700.0, 10.0, 800.0, 50.0]), label="mug", score=0.6),
        ]
    )
    assert detector.detect(IMAGE) == []
    detector.set_classes(["mug"])
    assert detector.classes == ["mug"]
    found = detector.detect(IMAGE)
    # The third mug lies outside the image.
    assert len(found) == 1
    np.testing.assert_array_equal(found[0].box, [600, 400, 640, 480])
    assert found[0].track_id == 3
    # The given objects are not changed.
    np.testing.assert_array_equal(detector.objects[0].box, [600, 400, 700, 500])


def test_fake_pose_estimator_returns_copies():
    estimator = FakePoseEstimator([_pose()])
    people = estimator.estimate(IMAGE)
    people[0].keypoints[0] = -1
    assert estimator.estimate(IMAGE)[0].keypoints[0, 0] == 0


def test_fake_reidentifier_gives_equal_vectors_for_equal_crops():
    reid = FakeReIdentifier(dim=128)
    crop = np.full((256, 128, 3), 7, dtype=np.uint8)
    vectors = reid.embed([crop, crop.copy(), crop + 1])
    assert vectors.shape == (3, 128)
    np.testing.assert_array_equal(vectors[0], vectors[1])
    assert not np.array_equal(vectors[0], vectors[2])


@pytest.mark.parametrize("given", [None, np.ones(512, dtype=np.float32)])
def test_fake_face_recognizer_fills_missing_embeddings(given):
    face = Face(bbox=np.array([10.0, 20.0, 60.0, 90.0]), kps=np.zeros((5, 2)), det_score=0.9, embedding=given)
    recognizer = FakeFaceRecognizer([face])
    first, second = recognizer.recognize(IMAGE)[0], recognizer.recognize(IMAGE)[0]
    assert first.embedding.shape == (512,)
    np.testing.assert_array_equal(first.embedding, second.embedding)
    if given is not None:
        np.testing.assert_array_equal(first.embedding, given)
    assert face.embedding is given
