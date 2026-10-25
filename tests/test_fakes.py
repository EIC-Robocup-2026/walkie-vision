import numpy as np
import pytest

from walkie_vision.fakes import (
    FakeEmbedder,
    FakeFaceRecognizer,
    FakeObjectDetector,
    FakePersonFaceRecognizer,
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
    PersonFaceRecognizer,
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


def _person(box: list[float]) -> PersonPose:
    return PersonPose(
        box=np.array(box),
        score=0.9,
        keypoints=np.zeros((17, 2), dtype=np.float32),
        keypoint_scores=np.full(17, 0.8, dtype=np.float32),
    )


def _face(box: list[float], embedding: np.ndarray | None = None) -> Face:
    return Face(bbox=np.array(box), kps=np.zeros((5, 2)), det_score=0.8, embedding=embedding)


def test_fake_person_face_recognizer_implements_the_protocol():
    recognizer = FakePersonFaceRecognizer([])
    assert isinstance(recognizer, PersonFaceRecognizer)
    assert recognizer.model_name.startswith("fake_")
    assert recognizer.faces_for(IMAGE, []) == []


def test_fake_person_face_recognizer_assigns_by_centre_in_input_order():
    left = _face([110.0, 60.0, 150.0, 110.0])
    right = _face([410.0, 60.0, 450.0, 110.0])
    recognizer = FakePersonFaceRecognizer([right, left])
    people = [_person([100.0, 50.0, 300.0, 470.0]), _person([600.0, 0.0, 640.0, 40.0]), _person([400.0, 50.0, 600.0, 470.0])]
    faces = recognizer.faces_for(IMAGE, people)
    assert len(faces) == 3
    np.testing.assert_array_equal(faces[0].bbox, left.bbox)
    assert faces[1] is None
    np.testing.assert_array_equal(faces[2].bbox, right.bbox)
    # Copies: the given faces are not changed.
    faces[0].bbox[0] = -1
    assert left.bbox[0] == 110.0 and left.embedding is None


def test_fake_person_face_recognizer_counts_the_box_edge_as_inside():
    # The centre (100, 50) lies on the person's top-left corner.
    recognizer = FakePersonFaceRecognizer([_face([90.0, 40.0, 110.0, 60.0])])
    assert recognizer.faces_for(IMAGE, [_person([100.0, 50.0, 300.0, 470.0])])[0] is not None


def test_fake_person_face_recognizer_never_gives_one_face_to_two_people():
    face = _face([150.0, 100.0, 190.0, 150.0])
    recognizer = FakePersonFaceRecognizer([face])
    # Both boxes contain the face's centre; the first person gets it.
    faces = recognizer.faces_for(IMAGE, [_person([100.0, 50.0, 300.0, 470.0]), _person([120.0, 60.0, 320.0, 470.0])])
    assert faces[0] is not None and faces[1] is None


def test_fake_person_face_recognizer_embeds_like_the_fake_face_recognizer():
    face = _face([110.0, 60.0, 150.0, 110.0])
    given = np.ones(512, dtype=np.float32)
    recognizer = FakePersonFaceRecognizer([face, _face([410.0, 60.0, 450.0, 110.0], embedding=given)])
    people = [_person([100.0, 50.0, 300.0, 470.0]), _person([400.0, 50.0, 600.0, 470.0])]
    faces = recognizer.faces_for(IMAGE, people)
    np.testing.assert_array_equal(faces[0].embedding, FakeFaceRecognizer([face]).recognize(IMAGE)[0].embedding)
    # The key that FakeFaceRecognizer has always used, so its vectors do not change.
    expected = hash_vector(b"face " + np.asarray(face.bbox, dtype=np.float64).tobytes(), 512)
    np.testing.assert_array_equal(faces[0].embedding, expected)
    np.testing.assert_array_equal(faces[0].embedding, recognizer.faces_for(IMAGE, people)[0].embedding)
    np.testing.assert_array_equal(faces[1].embedding, given)
