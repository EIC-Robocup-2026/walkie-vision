"""Golden digests of the stream and service messages. No ROS graph.

Each case builds messages the way the node and the fake publisher do, with
the fakes on a painted synthetic image, and compares their digests with the
values frozen in GOLDEN. The values were computed on the flat package layout
(ad7c833), before its modules moved: moving code changes no digest, so a
changed digest means a published or answered message changed.

A digest is not taken over the CDR bytes: rclpy leaves the alignment padding
after strings uninitialized, so the same message serializes to different
bytes from call to call. cdr() sends the message through serialize_message
and back, so the digest sees the wire's values (float32 rounding, bounded
types), and hashes the result field by field. The digests depend on the
locked numpy and OpenCV; a dependency update that changes one needs the new
value checked and frozen by hand.
"""

from __future__ import annotations

import array
import hashlib
import struct
from pathlib import Path

import numpy as np
import pytest
from builtin_interfaces.msg import Time
from rclpy.serialization import deserialize_message, serialize_message
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header
from walkie_vision_msgs.srv import DetectObjects, EmbedText, GetPersonCrop

from walkie_vision.camera.depth import RegisteredDepth
from walkie_vision.convert import (
    check_object_detection_array,
    fill_person_crop_response,
    pose_from_person_detection,
)
from walkie_vision.crops.person import CropStatus, crop_person
from walkie_vision.fake_publisher import DEFAULT_SCENE, FakeStream, load_scene
from walkie_vision.fakes import (
    FakeEmbedder,
    FakeObjectDetector,
    FakePersonFaceRecognizer,
    FakePoseEstimator,
    FakeReIdentifier,
    box_mask,
    figure_pose,
    paint_figure,
)
from walkie_vision.node import _crop_answer
from walkie_vision.pipeline import Providers, TrackIdBase, VisionPipeline
from walkie_vision.providers import Detection, Face
from walkie_vision.services.detect_objects import detect_objects_failure
from walkie_vision.services.embed_crop import embed_crop_answer, embed_crop_failure
from walkie_vision.services.embed_text import embed_text_failure, fill_embed_text_response, vectors_from_response
from walkie_vision.services.jobs import (
    BUSY,
    DISABLED,
    INFERENCE_FAILED,
    INVALID,
    MODEL_UNAVAILABLE,
    NO_IMAGE,
    STALE_IMAGE,
    TIMEOUT,
)

REPO = Path(__file__).resolve().parents[1]
STREAM_ID = "00000000-0000-4000-8000-000000000000"
CODES = (INVALID, NO_IMAGE, STALE_IMAGE, BUSY, TIMEOUT, DISABLED, MODEL_UNAVAILABLE, INFERENCE_FAILED)
H, W = 240, 320
CAMERA_FRAME = "zed_head_left_camera_frame_optical"
CHAIR = (130.0, 150.0, 175.0, 200.0)
SOFA = (280.0, 170.0, 318.0, 236.0)
CUP = (60.0, 160.0, 100.0, 200.0)
CLASSES = ["chair", "sofa"]


def _feed(value, h) -> None:
    """Hash a message value field by field, with exact bit patterns and no CDR padding."""
    if hasattr(value, "get_fields_and_field_types"):
        h.update(b"{" + type(value).__name__.encode())
        for name in value.get_fields_and_field_types():
            h.update(b"." + name.encode() + b"=")
            _feed(getattr(value, name), h)
        h.update(b"}")
    elif isinstance(value, array.array):
        h.update(b"a" + value.typecode.encode() + len(value).to_bytes(8, "little") + value.tobytes())
    elif isinstance(value, np.ndarray):
        h.update(b"n" + value.dtype.str.encode() + repr(value.shape).encode() + np.ascontiguousarray(value).tobytes())
    elif isinstance(value, (bytes, bytearray)):
        h.update(b"b" + len(value).to_bytes(8, "little") + bytes(value))
    elif isinstance(value, (list, tuple)):
        h.update(b"[" + len(value).to_bytes(8, "little"))
        for item in value:
            _feed(item, h)
    elif isinstance(value, str):
        data = value.encode()
        h.update(b"s" + len(data).to_bytes(8, "little") + data)
    elif isinstance(value, bool):
        h.update(b"T" if value else b"F")
    elif isinstance(value, int):
        h.update(b"i" + str(value).encode() + b";")
    elif isinstance(value, float):
        h.update(b"d" + struct.pack("<d", value))
    else:
        raise TypeError(f"cannot hash a {type(value).__name__}")


def cdr(msg) -> str:
    """Digest of a message as a subscriber receives it: the first 24 hex characters of a SHA-256."""
    h = hashlib.sha256()
    _feed(deserialize_message(serialize_message(msg), type(msg)), h)
    return h.hexdigest()[:24]


def _figures():
    """The painted image and the two figures standing in it, track ids 3 and 4."""
    rng = np.random.default_rng(7)
    image = np.clip(rng.normal(110, 25, (H, W, 3)), 0, 255).astype(np.uint8)
    a = figure_pose((20.0, 20.0, 110.0, 230.0), track_id=3)
    b = figure_pose((180.0, 25.0, 270.0, 228.0), track_id=4, score=0.8)
    paint_figure(image, a, top=(0, 0, 200), bottom=(200, 60, 0), top_stripe=(255, 255, 255))
    paint_figure(image, b, top=(0, 180, 0), bottom=(40, 40, 40))
    image[150:200, 130:175] = (30, 90, 160)
    image[170:236, 280:318] = (120, 30, 120)
    return image, a, b


def _camera_info() -> CameraInfo:
    info = CameraInfo()
    info.header.frame_id = "elsewhere"
    info.width, info.height = W, H
    info.k = [200.0, 0.0, 160.0, 0.0, 200.0, 120.0, 0.0, 0.0, 1.0]
    info.p = [200.0, 0.0, 160.0, 0.0, 0.0, 200.0, 120.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    return info


def _depth() -> RegisteredDepth:
    ys, xs = np.mgrid[0:H, 0:W]
    metres = (1.2 + 0.002 * xs + 0.001 * ys).astype(np.float32)
    metres[160:190, 140:165] = 0.9
    return RegisteredDepth(metres, 200.0, 200.0, 160.0, 120.0)


def _header(i: int) -> Header:
    return Header(stamp=Time(sec=50 + i, nanosec=1000 * i), frame_id=CAMERA_FRAME)


def _detection(box, label, score, track_id, mask=True) -> Detection:
    m = box_mask(H, W, box) if mask else None
    return Detection(box=np.array(box, dtype=np.float64), label=label, score=score, track_id=track_id, mask=m)


def _providers(people, embedder=None, on_demand=None, masks=False) -> Providers:
    face = Face(bbox=np.array([45.0, 25.0, 80.0, 60.0]), kps=np.zeros((5, 2)), det_score=0.85)
    return Providers(
        objects=FakeObjectDetector(
            [_detection(CHAIR, "chair", 0.81, 5, masks), _detection(SOFA, "sofa", 0.62, 6, masks)], classes=CLASSES
        ),
        pose=FakePoseEstimator(people),
        faces=FakePersonFaceRecognizer([face]),
        reid=FakeReIdentifier(),
        embedder=embedder,
        on_demand=on_demand,
    )


def _frames(pipeline, image, n, depth):
    """Run n frames; return a row per frame and the last frame's result."""
    info = _camera_info()
    rows = []
    for i in range(n):
        r = pipeline.process(image, _header(i), info, depth)
        check_object_detection_array(r.objects)
        rows.append({
            "people": cdr(r.people),
            "objects": cdr(r.objects),
            "dropped": dict(r.dropped),
            "duplicates": r.duplicates,
            "sampled": r.sampled,
        })
    return rows, r


def _person_crop(image, people, track_id, **options) -> dict:
    """GetPersonCrop's response for one track of a published people array, as the node builds it."""
    found = crop_person(image, [pose_from_person_detection(p) for p in people.people], track_id, **options)
    response = fill_person_crop_response(GetPersonCrop.Response(), people, found)
    return {"status": CropStatus(response.status).name, "digest": cdr(response)}


def _default_crop_options() -> dict:
    request = GetPersonCrop.Request()
    return {
        "padding": float(request.padding),
        "max_side": int(request.max_side),
        "mask_others": bool(request.mask_others),
        "min_keypoint_score": float(request.min_keypoint_score),
    }


def _pipeline_cases() -> dict:
    image, a, b = _figures()
    depth = _depth()
    out: dict = {}

    plain = VisionPipeline(_providers([a, b]), CLASSES, TrackIdBase(start_ns=0), stream_id=STREAM_ID)
    out["stream-plain"], last_plain = _frames(plain, image, 3, None)
    rich = VisionPipeline(
        _providers([a, b], embedder=FakeEmbedder(), masks=True),
        CLASSES,
        TrackIdBase(start_ns=0),
        stream_id=STREAM_ID,
        geometry_enabled=True,
        object_crops_enabled=True,
        object_embeddings_enabled=True,
        allow_fake_embedder=True,
        people_dedupe_iou=0.7,
        people_min_score=0.1,
        objects_min_score=0.1,
    )
    out["stream-geometry-crops-embeddings"], _ = _frames(rich, image, 12, depth)

    on_demand = VisionPipeline(
        _providers(
            [a, b],
            embedder=FakeEmbedder(),
            on_demand=FakeObjectDetector(
                [_detection(CUP, "cup", 0.7, 9), _detection(CHAIR, "chair", 0.5, 4)], classes=["cup"]
            ),
        ),
        CLASSES,
        TrackIdBase(start_ns=0),
        stream_id=STREAM_ID,
        on_demand_enabled=True,
        geometry_enabled=True,
        object_crops_enabled=True,
        object_embeddings_enabled=True,
        allow_fake_embedder=True,
    )
    on_demand.prepare_on_demand(["cup", "chair"])
    od = on_demand.detect_on_demand(image, _header(0), _camera_info(), depth)
    check_object_detection_array(od.objects)
    out["detect-objects-on-demand"] = {"objects": cdr(od.objects), "dropped": dict(od.dropped), "sampled": od.sampled}
    out["DetectObjects-success"] = cdr(DetectObjects.Response(success=True, objects=od.objects))

    people = last_plain.people
    published = [int(p.track_id) for p in people.people]
    defaults = _default_crop_options()
    out["GetPersonCrop-track-0"] = _person_crop(image, people, published[0], **defaults)
    out["GetPersonCrop-track-1"] = _person_crop(image, people, published[1], **defaults)
    out["GetPersonCrop-missing"] = _person_crop(image, people, 123456, **defaults)
    out["GetPersonCrop-small-unmasked"] = _person_crop(
        image, people, published[0], padding=0.0, max_side=64, mask_others=False, min_keypoint_score=0.2
    )

    # A third figure in front of the first (its box reaches the image's bottom
    # edge) and a figure too short for a crop.
    near = figure_pose((60.0, 40.0, 140.0, 239.5), track_id=7)
    tiny = figure_pose((285.0, 20.0, 315.0, 120.0), track_id=8)
    crowded = VisionPipeline(_providers([a, b, near, tiny]), CLASSES, TrackIdBase(start_ns=0), stream_id=STREAM_ID)
    _, last_crowded = _frames(crowded, image, 1, None)
    crowd = last_crowded.people
    crowd_ids = [int(p.track_id) for p in crowd.people]
    out["GetPersonCrop-occluded"] = _person_crop(image, crowd, crowd_ids[0], **defaults)
    out["GetPersonCrop-too-small"] = _person_crop(image, crowd, crowd_ids[3], **defaults)
    for status in CropStatus:
        out[f"GetPersonCrop-answer-{status.name}"] = cdr(_crop_answer(status, f"status {status.name}"))

    embedder = FakeEmbedder()
    text = fill_embed_text_response(EmbedText.Response(), embedder, ["red chair", "a cup", "a photo of a sofa."])
    out["EmbedText-ok"] = cdr(text)
    out["EmbedText-vectors"] = hashlib.sha256(np.ascontiguousarray(vectors_from_response(text)).tobytes()).hexdigest()[
        :24
    ]
    out["EmbedText-empty"] = cdr(fill_embed_text_response(EmbedText.Response(), embedder, []))
    header = _header(1)
    out["EmbedCrop-with-depth"] = cdr(embed_crop_answer(embedder, image, header, np.array(CHAIR), depth, 0.2, 3.0))
    out["EmbedCrop-without-depth"] = cdr(embed_crop_answer(embedder, image, header, np.array(SOFA), None, 0.2, 3.0))
    for code in CODES:
        out[f"EmbedText-failure-{code}"] = cdr(embed_text_failure(code, "x" * 300))
        out[f"EmbedCrop-failure-{code}"] = cdr(embed_crop_failure(code, f"because {code}"))
        out[f"DetectObjects-failure-{code}"] = cdr(detect_objects_failure(code, f"because {code}"))
    return out


def _fake_stream_cases() -> dict:
    scene = load_scene(DEFAULT_SCENE)
    variants = {
        "fake-stream-crops": dict(crops_enabled=True),
        "fake-stream-no-crops": dict(crops_enabled=False),
        "fake-stream-crops-fake-clip-512": dict(crops_enabled=True, embedding_model="fake_clip", embedding_dim=512),
        "fake-stream-no-crops-fake-small-16": dict(
            crops_enabled=False, embedding_model="fake_small", embedding_dim=16
        ),
    }
    out: dict = {
        "fake-stream-default-scene": DEFAULT_SCENE.relative_to(REPO).as_posix(),
        "fake-stream-tf-static": cdr(scene.camera.transform(Time(sec=1))),
    }
    for name, kwargs in variants.items():
        stream = FakeStream(scene, stream_id=STREAM_ID, **kwargs)
        out[name] = [cdr(stream.array(i, Time(sec=1000 + i, nanosec=i))) for i in range(stream.frame_count)]
    return out


GOLDEN: dict = {
    "DetectObjects-failure-BUSY": "68fac875b03d046553c9422c",
    "DetectObjects-failure-DISABLED": "8470ad2d99ab49424151b14c",
    "DetectObjects-failure-INFERENCE_FAILED": "a64ddd3fa9096d8499ea3e03",
    "DetectObjects-failure-INVALID": "3766472935585fb40f303d13",
    "DetectObjects-failure-MODEL_UNAVAILABLE": "d9b3d126360e6bc0516a58cd",
    "DetectObjects-failure-NO_IMAGE": "afd4525257fac94b0cd4bb76",
    "DetectObjects-failure-STALE_IMAGE": "ae286f87de3ff8495b42c961",
    "DetectObjects-failure-TIMEOUT": "6223492dd98ada616b15e35e",
    "DetectObjects-success": "98205a293e3bfdd302f571f1",
    "EmbedCrop-failure-BUSY": "9b3e8e3eb5f91e421747760b",
    "EmbedCrop-failure-DISABLED": "e9d3e085430c520823accdba",
    "EmbedCrop-failure-INFERENCE_FAILED": "e73bbe8323079de93b6228ce",
    "EmbedCrop-failure-INVALID": "02acd77c9e5393590e28130b",
    "EmbedCrop-failure-MODEL_UNAVAILABLE": "2c43c56a3f617340bee936be",
    "EmbedCrop-failure-NO_IMAGE": "97d52079b6f4a39d45cd5c5c",
    "EmbedCrop-failure-STALE_IMAGE": "1ff9dd961c1c55ec19ec0b79",
    "EmbedCrop-failure-TIMEOUT": "f87834722bc7af4f868944a5",
    "EmbedCrop-with-depth": "443609ce87e3bdb23e570258",
    "EmbedCrop-without-depth": "ec618e8a35625a0a3741ae7d",
    "EmbedText-empty": "05772bb948977f40939eabf9",
    "EmbedText-failure-BUSY": "1e36a4f4fe687dab6804e920",
    "EmbedText-failure-DISABLED": "3f96b1f7f274c19c734e5a82",
    "EmbedText-failure-INFERENCE_FAILED": "9a3fd478741579d086698c36",
    "EmbedText-failure-INVALID": "f199293f15f31725dd2c1a76",
    "EmbedText-failure-MODEL_UNAVAILABLE": "ba127610bdaab408fa2ccf4c",
    "EmbedText-failure-NO_IMAGE": "f71682399aeaeda272dd80dc",
    "EmbedText-failure-STALE_IMAGE": "bddd87a986083ec8f22a110c",
    "EmbedText-failure-TIMEOUT": "dbdfeb4e7677bacec79f3957",
    "EmbedText-ok": "aec2526ffad9ed722c8fc0d5",
    "EmbedText-vectors": "5333b318a940cb9976559bea",
    "GetPersonCrop-answer-FAILED": "5ca553f945495a85d56c7f4b",
    "GetPersonCrop-answer-NOT_IN_VIEW": "566cd4ea0afffc641e89be98",
    "GetPersonCrop-answer-NO_IMAGE": "012bc9af510a00ecfaad8cb0",
    "GetPersonCrop-answer-OCCLUDED": "6dd9d4093d728507f9dcc098",
    "GetPersonCrop-answer-OK": "20588dd72939267be50e58d6",
    "GetPersonCrop-answer-TIMEOUT": "0dff5956b0593521a2c069cd",
    "GetPersonCrop-answer-TOO_SMALL": "0d642ff77fdb942c1bc6c501",
    "GetPersonCrop-missing": {"digest": "91f62b646a644e6be44c69f3", "status": "NOT_IN_VIEW"},
    "GetPersonCrop-occluded": {"digest": "33897c33a55e5663a1e52479", "status": "OCCLUDED"},
    "GetPersonCrop-small-unmasked": {"digest": "16d67000efc1b143068d9819", "status": "OK"},
    "GetPersonCrop-too-small": {"digest": "15b24ce64cf81a153ded7445", "status": "TOO_SMALL"},
    "GetPersonCrop-track-0": {"digest": "8e069b3aa97792af2c371a95", "status": "OK"},
    "GetPersonCrop-track-1": {"digest": "a65dff740a559fed8779e15d", "status": "OK"},
    "detect-objects-on-demand": {"dropped": {}, "objects": "05e00627495f5bbefc026db9", "sampled": 2},
    "fake-stream-crops": [
        "0520d26f08885a152694c71b", "d7ceba9cdab10066066e3dae", "dbb7078ba9f3dbaac5e10175",
        "85600caaedd5d88e06599308", "31a5e2507b0afffb69320b85", "3a3725e979bd367733409415",
        "98ac8f2f87b85724f795f665", "f2d16b266312cf7195a3f4b3", "b25cd4037714ecee955acf56",
        "d716b8863136b74a37e4e53c", "b8ab1785a44ed7b20d7dd862", "d01ae77d59a965f8fa961064",
        "29a91dc2a4875bbfa652d2af", "83d4380632d48639bd1e33eb", "9e012c7369264518faa07e31",
        "c9a8c30572ad9888d6716649", "e8d985d5eeb8f1e57987ba87", "552ab3276210f993fed5636d",
        "c0ceec3d3f97a3dae0d47846", "21732e91829776bb1f8525c8", "c705dc5f609c9c805e553f7f",
        "b12bd668219a6962093aa8d4", "87217d223cc7ec86ee2ec9d7", "eb339ff4fe3cb5759d59b172",
        "995304e9c838e6bf5062ee4b", "6c7682420981e7f2aab21809", "51889df8813d229431e1e675",
        "ac398de25dbb0160887a7462", "e25caf4c79bae981ff6e003f", "e3ec7147ff8e85640682a41d",
    ],
    "fake-stream-crops-fake-clip-512": [
        "66035ca8936395397140cb69", "3f4b914cc3f0f8801d569f15", "422d32143938b90378dcc37a",
        "1e8f1246a77ffdd6e7b8e89a", "82e1e7ad5cb2a5b7dfa17009", "83cfd5a26335851b30113878",
        "29d2808ec38037a213cd4b6f", "d736c2f4db3cda728199f86f", "cd83c6ccd9922591feb71e1f",
        "ebd30869219aae0f3390b316", "30276c448d46a9f65b5ec9f8", "83a823b5627cc5816ae748df",
        "7a3035eeba3e335c15cdae90", "8656f00d03964831d5a099e9", "97e53d36935ec9d682f4a14a",
        "c9a8c30572ad9888d6716649", "e8d985d5eeb8f1e57987ba87", "552ab3276210f993fed5636d",
        "c0ceec3d3f97a3dae0d47846", "21732e91829776bb1f8525c8", "260ee1b6b7454599b4c94354",
        "293c8878f1f011750c38bb8a", "c61f331d43d7c285258e4941", "1b713059c3ae0cd2039a05d2",
        "6682e4b777dcd6f9320366ee", "0cf6a38228454702b8238daa", "2569c3beda11ebac77b97e14",
        "2c4c76332c06a15bd52bbd6e", "ea254fbb96665abe3fa47dae", "0a78960cf1d0cb51dcba9c83",
    ],
    "fake-stream-default-scene": "config/fakes/boxes.json",
    "fake-stream-no-crops": [
        "8f7bca5b3149e622b6b22c03", "3c256dc9ad786c18582bd4f9", "b1fea7e2b1bd786310fdefe2",
        "185a881d348847cba6da08c0", "9cc9ff557291a49413a4caec", "34f9352d587c0986044c3f62",
        "55ff84d823adf8fd81c22a44", "d676e73a82dfaa1b3b00e8ee", "f04af98f20cfb11dfbd0469b",
        "a465fe414bdcb55f6b1ae354", "b1e31ecc2596683d4eb692d6", "fa784de19219efb10f6f061f",
        "920f20d8b2d42a9377aae76e", "dce9aa22751d16098eab3a88", "4f5aaa6ba09c2d6c501bd45a",
        "c9a8c30572ad9888d6716649", "e8d985d5eeb8f1e57987ba87", "552ab3276210f993fed5636d",
        "c0ceec3d3f97a3dae0d47846", "21732e91829776bb1f8525c8", "8f78804c0063cd3f12162be8",
        "4a1ea19abf530e748f2ea861", "c2532f07b1afe52d6d532dcc", "cf407c06054ece9bf09f660e",
        "bc9afebdb3bb51c9dc64261d", "66b86684bd42bff44ae6cae5", "6310e63b0c559f22d42d41f5",
        "1e3d6e1ac9d398ac17255d41", "ce0ccbfb81cbffbce4695f87", "57d801dc577e101ca230de3b",
    ],
    "fake-stream-no-crops-fake-small-16": [
        "9373112f756f7f7bcd4c8b33", "b0bf5c2d2e3e09a5c135e47d", "f4ea93c790a4f03da55dbbb5",
        "3dca18085df93f788d2d2193", "2c213d510ce8ced89df8bdc3", "cb9b3a72537936fa4dc6fbda",
        "7d9f8f8825c4da562ffd8608", "197d0ab6786495d8501ba2c2", "e6fbb0c94c5f8fb3da211639",
        "785810660de161e4c2743cf0", "e51d915f9ccc1072ec205516", "6fbcec9f0fe0d29671caa88b",
        "b86695cd278171fea7802ee1", "81532c9e941d511b5e9841e6", "9aaf4fea9c411645d5fb6a52",
        "c9a8c30572ad9888d6716649", "e8d985d5eeb8f1e57987ba87", "552ab3276210f993fed5636d",
        "c0ceec3d3f97a3dae0d47846", "21732e91829776bb1f8525c8", "3a658d4e51089ce8f0763b77",
        "b09cb2efe4d82fb7993c8b12", "78513048d296e0146568e62f", "aa7e06fcfb01fca937bdf200",
        "6bbdf6bb1a5e1ea5383ebff5", "b2d89ea3322b2adf741a5f28", "7d391a2f1867e870b8712f6b",
        "acc9cf2b28db2b955d9a1b9b", "c373e3fabc5b80ef6c7a543c", "83cc1c0c5bf6ce6d478ba02b",
    ],
    "fake-stream-tf-static": "6fb0e0d165c08e174ab26585",
    "stream-geometry-crops-embeddings": [
        {"dropped": {}, "duplicates": 0, "objects": "3d2ed9b49263fbda85d10e8b", "people": "81eaf9b2377bd88e10ca32c1", "sampled": 0},
        {"dropped": {}, "duplicates": 0, "objects": "56564cb2847c01993e6fdb7f", "people": "a4ae0691b2deab96470544f7", "sampled": 0},
        {"dropped": {}, "duplicates": 0, "objects": "e0c46f33619e1b799905b9df", "people": "c99dc9f5a383a61850fec423", "sampled": 0},
        {"dropped": {}, "duplicates": 0, "objects": "b23898cc388d7048b6f70ac8", "people": "0a53f55bee0563980b3f2895", "sampled": 0},
        {"dropped": {}, "duplicates": 0, "objects": "e1ea9eeeb2a5424b87432647", "people": "f1d62a9bddc70f8254755b3f", "sampled": 1},
        {"dropped": {}, "duplicates": 0, "objects": "1f64a49cb5fe305be604cc1e", "people": "dba9a489e333e60b8ff2148d", "sampled": 1},
        {"dropped": {}, "duplicates": 0, "objects": "acdd07fe382095086bb59545", "people": "23dd759dabecf222d37ca683", "sampled": 0},
        {"dropped": {}, "duplicates": 0, "objects": "1795717e5a51be4a208c6de9", "people": "c17b87bcaa391bc9ac8917db", "sampled": 0},
        {"dropped": {}, "duplicates": 0, "objects": "da326616af8dad78027a409a", "people": "de0f82cb57361a75cdbcd606", "sampled": 0},
        {"dropped": {}, "duplicates": 0, "objects": "b66afd78e6d8bc18f07d4a98", "people": "68664cdbf6e2cf0328193bf4", "sampled": 0},
        {"dropped": {}, "duplicates": 0, "objects": "bbe67317072b45f768516863", "people": "d06a316ebaa55d395d0b1a9a", "sampled": 0},
        {"dropped": {}, "duplicates": 0, "objects": "5f7c296afcf4df8ad4fd4138", "people": "8be878c569526d504476da76", "sampled": 0},
    ],
    "stream-plain": [
        {"dropped": {}, "duplicates": 0, "objects": "d7a69ef761f28c236cf5e306", "people": "81eaf9b2377bd88e10ca32c1", "sampled": 0},
        {"dropped": {}, "duplicates": 0, "objects": "04e6e814e3d072f3a25a7c31", "people": "a4ae0691b2deab96470544f7", "sampled": 0},
        {"dropped": {}, "duplicates": 0, "objects": "f472eb6c0aba4dbb9a809f70", "people": "c99dc9f5a383a61850fec423", "sampled": 0},
    ],
}
"""The frozen value of each case: a message digest, or the digests and counts of a run of frames."""


@pytest.fixture(scope="module")
def computed() -> dict:
    return {**_pipeline_cases(), **_fake_stream_cases()}


def test_every_case_has_a_golden_value(computed):
    assert sorted(computed) == sorted(GOLDEN)


@pytest.mark.parametrize("case", sorted(GOLDEN))
def test_message_matches_its_golden_digest(computed, case):
    assert computed[case] == GOLDEN[case]
