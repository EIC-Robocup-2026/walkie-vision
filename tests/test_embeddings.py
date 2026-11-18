"""Sampling, crops, colour shares and the CLIP provider, without a GPU or a model file.

A fake embedder stands in for CLIP in the pipeline's tests. The provider's own
tests inject a stub loader and a stub open_clip, so nothing downloads or loads
a model.
"""

from __future__ import annotations

import sys
import types

import cv2
import numpy as np
import pytest
import torch
from builtin_interfaces.msg import Time
from huggingface_hub.errors import LocalEntryNotFoundError
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header

from walkie_vision import embeddings
from walkie_vision.convert import bbox_to_box
from walkie_vision.crops.colour import COLOR_NAMES
from walkie_vision.embeddings import (
    CROP_MAX_SIDE,
    MIN_COLOR_PIXELS,
    PALETTE_RGB,
    ClipEmbedder,
    clip_checkpoint,
    encode_crop,
    mask_color_shares,
    nearest_palette_indices,
    padded_crop,
    select_samples,
)
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

CLIP_REPO = "timm/vit_base_patch16_clip_224.openai"
CLIP_REVISION = "977e3dd0ec55ab8da155f2fbeb6b5f54948b6e3d"
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


# --- the sampler -----------------------------------------------------------


@pytest.mark.parametrize(
    "track_id, counter, sampled",
    [(10, 0, True), (1, 0, False), (9, 0, False), (1, 9, True), (2, 9, False), (7, 13, True), (0, 0, True), (0, 1, False)],
)
def test_a_track_is_eligible_once_in_ten_images(track_id, counter, sampled):
    assert select_samples([_detection(track_id)], counter, WIDTH, HEIGHT) == ([0] if sampled else [])


@pytest.mark.parametrize(
    "box, sampled",
    [
        ((0.0, 0.0, 32.0, 32.0), True),
        ((0.0, 0.0, 31.9, 40.0), False),
        ((0.0, 0.0, 40.0, 31.0), False),
        # A box that hangs off the image counts only its pixels inside it.
        ((-30.0, 0.0, 10.0, 40.0), False),
    ],
)
def test_a_box_needs_32_pixels_on_its_short_side(box, sampled):
    assert select_samples([_detection(box=box)], 0, WIDTH, HEIGHT) == ([0] if sampled else [])


def test_at_most_three_detections_are_sampled_by_score():
    scores = [0.5, 0.9, 0.7, 0.6, 0.8]
    detections = [_detection(10 * (i + 1), score) for i, score in enumerate(scores)]
    assert select_samples(detections, 0, WIDTH, HEIGHT) == [1, 4, 2]


def test_equal_scores_break_by_track_id_then_index():
    detections = [_detection(30, 0.5), _detection(10, 0.5), _detection(20, 0.5), _detection(10, 0.5)]
    assert select_samples(detections, 0, WIDTH, HEIGHT) == [1, 3, 2]


# --- padding and JPEG ------------------------------------------------------


def test_the_crop_grows_15_percent_on_each_side():
    image = np.arange(1000 * 1000 * 3, dtype=np.uint32).reshape(1000, 1000, 3).astype(np.uint8)
    crop = padded_crop(image, np.array([100.0, 200.0, 300.0, 600.0]))
    # 30 px left and right of a 200 px width, 60 px above and below a 400 px height.
    assert crop.shape[:2] == (520, 260)
    np.testing.assert_array_equal(crop, image[140:660, 70:330])


def test_the_crop_is_clipped_to_the_image():
    crop = padded_crop(IMAGE, np.array([0.0, 0.0, 64.0, 48.0]))
    assert crop.shape == IMAGE.shape
    crop = padded_crop(IMAGE, np.array([50.0, 40.0, 64.0, 48.0]))
    np.testing.assert_array_equal(crop, IMAGE[38:48, 47:64])


def test_a_box_outside_the_image_has_no_crop():
    with pytest.raises(ValueError, match="no pixels"):
        padded_crop(IMAGE, np.array([200.0, 200.0, 240.0, 240.0]))


def test_a_jpeg_has_its_header_and_is_reduced_but_never_enlarged():
    small = encode_crop(IMAGE)
    assert small[:2] == b"\xff\xd8"
    assert cv2.imdecode(np.frombuffer(small, np.uint8), cv2.IMREAD_COLOR).shape == IMAGE.shape

    large = np.zeros((800, 1024, 3), dtype=np.uint8)
    decoded = cv2.imdecode(np.frombuffer(encode_crop(large), np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (400, CROP_MAX_SIDE, 3)


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


# --- the pipeline ----------------------------------------------------------


def test_with_both_flags_off_no_sampling_work_runs(monkeypatch):
    def refuse(*args):
        raise AssertionError("no sampling work with the flags off")

    for name in ("select_samples", "padded_crop", "encode_crop", "mask_color_shares"):
        monkeypatch.setattr(embeddings, name, refuse)
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


# --- the CLIP provider, with stubs -----------------------------------------


def test_the_weights_load_from_the_cache_at_the_pinned_commit():
    calls = []

    def stub(repo, filename, *, revision, local_files_only):
        calls.append((repo, filename, revision, local_files_only))
        return "/cache/open_clip_model.safetensors"

    assert clip_checkpoint(stub) == "/cache/open_clip_model.safetensors"
    assert calls == [(CLIP_REPO, "open_clip_model.safetensors", CLIP_REVISION, True)]


def test_weights_that_are_not_cached_say_what_to_run():
    def stub(*args, **kwargs):
        raise LocalEntryNotFoundError("not cached")

    with pytest.raises(FileNotFoundError, match="pixi run fetch-models"):
        clip_checkpoint(stub)


def test_the_loader_uses_quickgelu_and_the_local_file(monkeypatch):
    calls = []

    class Model:
        evaluating = False

        def eval(self):
            self.evaluating = True

    model = Model()

    def create(name, *, pretrained, device):
        calls.append((name, pretrained, device))
        return model, "train transform", "preprocess"

    stub = types.SimpleNamespace(create_model_and_transforms=create, get_tokenizer=lambda name: f"tokenizer {name}")
    monkeypatch.setitem(sys.modules, "open_clip", stub)

    built = ClipEmbedder("/cache/weights.safetensors", device="cpu")

    assert calls == [("ViT-B-16-quickgelu", "/cache/weights.safetensors", "cpu")]
    assert model.evaluating
    assert built._preprocess == "preprocess"
    assert built._tokenizer == "tokenizer ViT-B-16-quickgelu"


class StubClip:
    """Stands in for the open_clip model, preprocessing and tokenizer."""

    def __init__(self, image_rows=None) -> None:
        self.pil_images = []
        self.tokenized: list[str] = []
        self.image_rows = image_rows

    def build(self, checkpoint: str, device: str):
        assert checkpoint == "/cache/weights.safetensors" and device == "cpu"
        return self, self.preprocess, self.tokenizer

    def preprocess(self, image):
        self.pil_images.append(image)
        return torch.zeros(3, 224, 224)

    def tokenizer(self, texts):
        self.tokenized.extend(texts)
        return torch.zeros(len(texts), 77, dtype=torch.long)

    def encode_image(self, batch):
        if self.image_rows is not None:
            return torch.tensor(self.image_rows, dtype=torch.float32)
        rows = torch.zeros(batch.shape[0], 512)
        rows[:, 0] = 3.0
        rows[:, 1] = 4.0
        return rows

    def encode_text(self, batch):
        return torch.ones(batch.shape[0], 512)


def _clip(stub: StubClip) -> ClipEmbedder:
    return ClipEmbedder("/cache/weights.safetensors", device="cpu", build=stub.build)


def test_the_provider_names_the_model_and_its_length():
    embedder = _clip(StubClip())
    assert embedder.model_name == "clip_vit_b16" and embedder.dim == 512


def test_clip_gets_rgb_pixels():
    stub = StubClip()
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[..., 0] = 200  # blue in BGR
    _clip(stub).embed_images([image])
    assert stub.pil_images[0].getpixel((0, 0)) == (0, 0, 200)


def test_image_rows_are_unit_length_float32():
    vectors = _clip(StubClip()).embed_images([IMAGE, IMAGE])
    assert vectors.shape == (2, 512) and vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-6)


def test_a_zero_row_stays_zero_for_the_caller_to_refuse():
    rows = np.zeros((1, 512), dtype=np.float32).tolist()
    vectors = _clip(StubClip(rows)).embed_images([IMAGE])
    assert not vectors.any()


def test_texts_always_use_the_template():
    stub = StubClip()
    vectors = _clip(stub).embed_texts(["red mug", "cup {with} braces"])
    assert stub.tokenized == ["a photo of red mug.", "a photo of cup {with} braces."]
    assert vectors.shape == (2, 512)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-6)


def test_empty_batches_do_not_reach_the_model():
    stub = StubClip()
    embedder = _clip(stub)
    assert embedder.embed_images([]).shape == (0, 512)
    assert embedder.embed_texts([]).shape == (0, 512)
    assert stub.pil_images == [] and stub.tokenized == []
