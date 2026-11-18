"""walkie_vision.pipeline's object sampling: crops, vectors and colour shares, with the fakes. No ROS graph.

A fake embedder stands in for CLIP, so nothing downloads or loads a model.
"""

from __future__ import annotations

import numpy as np
import pytest
from builtin_interfaces.msg import Time
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header

from walkie_vision.convert import bbox_to_box
from walkie_vision.crops import colour, objects
from walkie_vision.crops.colour import PALETTE_RGB
from walkie_vision.crops.objects import encode_crop, padded_crop
from walkie_vision.fakes import (
    FakeEmbedder,
    FakeObjectDetector,
    FakePersonFaceRecognizer,
    FakePoseEstimator,
    FakeReIdentifier,
    box_mask,
    hash_vector,
)
from walkie_vision.pipeline import Providers, TrackIdBase, VisionPipeline
from walkie_vision.providers import CLIP_VIT_B16, Detection

CLASSES = ["chair", "sofa"]
FRAME = "zed_head_left_camera_frame_optical"
WIDTH, HEIGHT = 64, 48
BOX = (10.0, 8.0, 50.0, 40.0)  # 40 x 32 px
IMAGE = np.random.default_rng(0).integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)


def _detection(track_id=10, score=0.8, box=BOX, mask=None, label="chair") -> Detection:
    return Detection(box=np.array(box, dtype=np.float64), label=label, score=score, track_id=track_id, mask=mask)


def _header() -> Header:
    return Header(stamp=Time(sec=12, nanosec=345), frame_id=FRAME)


def _info() -> CameraInfo:
    info = CameraInfo()
    info.width, info.height = WIDTH, HEIGHT
    info.k = [50.0, 0.0, 32.0, 0.0, 50.0, 24.0, 0.0, 0.0, 1.0]
    return info


class RecordingEmbedder(FakeEmbedder):
    """A fake embedder that keeps what it was given."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.images: list[np.ndarray] = []

    def embed_images(self, images):
        self.images.extend(np.array(image) for image in images)
        return super().embed_images(images)


class FixedEmbedder(RecordingEmbedder):
    """Returns the given rows, whatever the images are."""

    def __init__(self, rows) -> None:
        super().__init__()
        self.rows = np.asarray(rows, dtype=np.float32)

    def embed_images(self, images):
        self.images.extend(np.array(image) for image in images)
        return self.rows


def _providers(detections, embedder=None) -> Providers:
    return Providers(
        objects=FakeObjectDetector(detections, classes=CLASSES),
        pose=FakePoseEstimator([]),
        faces=FakePersonFaceRecognizer([]),
        reid=FakeReIdentifier(),
        embedder=embedder,
    )


def _pipeline(detections, embedder=None, *, crops=False, vectors=False, allow_fake=True, start_ns=0) -> VisionPipeline:
    return VisionPipeline(
        _providers(detections, embedder),
        CLASSES,
        TrackIdBase(start_ns=start_ns),
        object_crops_enabled=crops,
        object_embeddings_enabled=vectors,
        allow_fake_embedder=allow_fake,
    )


def _sampled(detection) -> bool:
    return bool(detection.crop.data) or bool(detection.embedding) or bool(detection.color_shares)


def _bgr(rgb) -> tuple[int, int, int]:
    return rgb[2], rgb[1], rgb[0]


# --- the pipeline ----------------------------------------------------------


def test_with_both_flags_off_no_sampling_work_runs(monkeypatch):
    def refuse(*args):
        raise AssertionError("no sampling work with the flags off")

    for name in ("select_samples", "padded_crop", "encode_crop"):
        monkeypatch.setattr(objects, name, refuse)
    monkeypatch.setattr(colour, "mask_color_shares", refuse)
    embedder = RecordingEmbedder()
    mask = box_mask(HEIGHT, WIDTH, BOX)
    result = _pipeline([_detection(mask=mask)], embedder).process(IMAGE, _header(), _info())
    assert embedder.images == []
    assert not _sampled(result.objects.detections[0])


def test_a_pipeline_without_the_flags_needs_no_embedder():
    result = _pipeline([_detection()]).process(IMAGE, _header(), _info())
    assert result.objects.detections[0].embedding_model == ""


def test_embeddings_need_an_embedder():
    with pytest.raises(ValueError, match="needs an embedder"):
        _pipeline([_detection()], vectors=True)


def test_the_real_path_refuses_a_fake_embedder():
    with pytest.raises(ValueError, match="must not run with the real models"):
        _pipeline([_detection()], FakeEmbedder(), vectors=True, allow_fake=False)


def test_the_real_path_accepts_a_real_model_name():
    _pipeline([_detection()], FakeEmbedder(model_name=CLIP_VIT_B16), vectors=True, allow_fake=False)


def test_the_real_path_refuses_clip_with_the_wrong_length():
    embedder = FakeEmbedder(model_name=CLIP_VIT_B16, dim=8)
    with pytest.raises(ValueError, match="512"):
        _pipeline([_detection()], embedder, vectors=True, allow_fake=False)


def test_the_counter_starts_at_zero_and_counts_processed_images():
    pipeline = _pipeline([_detection(track_id=8)], RecordingEmbedder(), vectors=True)
    sampled = []
    for _ in range(13):
        sampled.append(_sampled(pipeline.process(IMAGE, _header(), _info()).objects.detections[0]))
    assert [i for i, s in enumerate(sampled) if s] == [2, 12]
    assert pipeline.processed_images == 13


def test_a_failed_image_does_not_count():
    class FailingOnce(FakeObjectDetector):
        failed = False

        def detect(self, image):
            if not self.failed:
                self.failed = True
                raise RuntimeError("the detector failed")
            return super().detect(image)

    providers = Providers(
        objects=FailingOnce([_detection(track_id=10)], classes=CLASSES),
        pose=FakePoseEstimator([]),
        faces=FakePersonFaceRecognizer([]),
        reid=FakeReIdentifier(),
        embedder=RecordingEmbedder(),
    )
    pipeline = VisionPipeline(
        providers, CLASSES, TrackIdBase(start_ns=0), object_embeddings_enabled=True, allow_fake_embedder=True
    )
    with pytest.raises(RuntimeError):
        pipeline.process(IMAGE, _header(), _info())
    assert pipeline.processed_images == 0
    assert _sampled(pipeline.process(IMAGE, _header(), _info()).objects.detections[0])


def test_the_sampler_uses_the_published_track_id():
    # The offset is 5 ms, so tracker id 5 is published as 10 and tracker id 10 as 15.
    pipeline = _pipeline(
        [_detection(track_id=5), _detection(track_id=10, label="sofa")], RecordingEmbedder(), vectors=True, start_ns=5_000_000
    )
    first, second = pipeline.process(IMAGE, _header(), _info()).objects.detections
    assert (first.track_id, second.track_id) == (10, 15)
    assert _sampled(first) and not _sampled(second)


def test_only_the_sampled_detections_are_filled():
    detections = [_detection(10 * (i + 1), score, mask=box_mask(HEIGHT, WIDTH, BOX)) for i, score in enumerate([0.5, 0.9, 0.7, 0.6])]
    detections.append(_detection(track_id=3, score=1.0))  # the top score, but not its turn
    embedder = RecordingEmbedder()
    objects = _pipeline(detections, embedder, crops=True, vectors=True).process(IMAGE, _header(), _info()).objects.detections
    assert len(embedder.images) == 3
    for i, detection in enumerate(objects):
        if i in (1, 2, 3):
            assert detection.crop.data and detection.embedding and detection.color_shares
            assert detection.embedding_model == "fake_embedder"
        else:
            assert not _sampled(detection) and detection.embedding_model == ""


def test_the_embedding_and_the_crop_come_from_the_same_pixels():
    embedder = RecordingEmbedder()
    result = _pipeline([_detection()], embedder, crops=True, vectors=True).process(IMAGE, _header(), _info())
    detection = result.objects.detections[0]
    source = padded_crop(IMAGE, np.array(BOX))
    assert len(embedder.images) == 1
    np.testing.assert_array_equal(embedder.images[0], source)
    assert bytes(detection.crop.data) == encode_crop(source)
    expected = hash_vector(b"image " + repr(source.shape).encode() + source.tobytes(), 512)
    np.testing.assert_allclose(np.array(detection.embedding), expected, atol=1e-6)


def test_crops_work_without_embeddings():
    embedder = RecordingEmbedder()
    result = _pipeline([_detection(mask=box_mask(HEIGHT, WIDTH, BOX))], embedder, crops=True).process(IMAGE, _header(), _info())
    detection = result.objects.detections[0]
    assert embedder.images == []
    assert bytes(detection.crop.data[:2]) == b"\xff\xd8"
    assert detection.crop.format == "jpeg"
    assert detection.crop.header == result.objects.header
    assert len(detection.color_shares) == 12
    assert not detection.embedding and detection.embedding_model == ""


def test_embeddings_work_without_crops():
    result = _pipeline([_detection()], RecordingEmbedder(), vectors=True).process(IMAGE, _header(), _info())
    detection = result.objects.detections[0]
    assert len(detection.embedding) == 512 and not detection.crop.data


def test_a_detection_without_a_mask_has_no_colour_shares():
    result = _pipeline([_detection()], crops=True).process(IMAGE, _header(), _info())
    detection = result.objects.detections[0]
    assert detection.crop.data and not detection.color_shares


def test_colour_comes_from_the_original_mask_not_the_padded_crop():
    image = np.full((HEIGHT, WIDTH, 3), _bgr(PALETTE_RGB[7]), dtype=np.uint8)
    x1, y1, x2, y2 = (int(v) for v in BOX)
    image[y1:y2, x1:x2] = _bgr(PALETTE_RGB[3])
    result = _pipeline([_detection(mask=box_mask(HEIGHT, WIDTH, BOX))], crops=True).process(image, _header(), _info())
    assert result.objects.detections[0].color_shares[3] == pytest.approx(1.0)


def test_a_mask_of_the_wrong_size_raises():
    with pytest.raises(ValueError, match="mask has shape"):
        _pipeline([_detection(mask=np.ones((4, 4), dtype=bool))], crops=True).process(IMAGE, _header(), _info())


def test_vectors_become_unit_length():
    rows = np.zeros((1, 512), dtype=np.float32)
    rows[0, 0] = 3.0
    rows[0, 1] = 4.0
    detection = _pipeline([_detection()], FixedEmbedder(rows), vectors=True).process(IMAGE, _header(), _info()).objects.detections[0]
    assert detection.embedding[0] == pytest.approx(0.6) and detection.embedding[1] == pytest.approx(0.8)
    assert np.linalg.norm(detection.embedding) == pytest.approx(1.0)


@pytest.mark.parametrize("value", [0.0, float("nan"), float("inf")])
def test_a_bad_vector_is_refused_and_the_crop_stays(value):
    rows = np.full((1, 512), value, dtype=np.float32)
    result = _pipeline([_detection()], FixedEmbedder(rows), crops=True, vectors=True).process(IMAGE, _header(), _info())
    detection = result.objects.detections[0]
    assert not detection.embedding and detection.embedding_model == ""
    assert detection.crop.data
    assert result.dropped == {"embedding": 1}


def test_the_wrong_shape_from_the_embedder_fails_the_image():
    pipeline = _pipeline([_detection()], FixedEmbedder(np.ones((2, 512))), vectors=True)
    with pytest.raises(ValueError, match="embedder returned shape"):
        pipeline.process(IMAGE, _header(), _info())
    assert pipeline.processed_images == 0


def test_an_embedder_that_raises_fails_the_image():
    class Broken(FakeEmbedder):
        def embed_images(self, images):
            raise RuntimeError("out of memory")

    pipeline = _pipeline([_detection()], Broken(), vectors=True)
    with pytest.raises(RuntimeError, match="out of memory"):
        pipeline.process(IMAGE, _header(), _info())


def test_an_unsampled_image_does_not_call_the_embedder():
    embedder = RecordingEmbedder()
    _pipeline([_detection(track_id=3)], embedder, crops=True, vectors=True).process(IMAGE, _header(), _info())
    assert embedder.images == []


def test_the_box_in_the_message_stays_unpadded():
    result = _pipeline([_detection()], crops=True).process(IMAGE, _header(), _info())
    np.testing.assert_allclose(bbox_to_box(result.objects.detections[0].bbox), BOX)
