"""walkie_vision.services.get_person_crop: the GetPersonCrop response from one person crop. No ROS graph."""

import numpy as np
import pytest
from rclpy.serialization import deserialize_message, serialize_message
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header
from walkie_vision_msgs.srv import GetPersonCrop

from walkie_vision.crops import person
from walkie_vision.crops.colour import COLOR_NAMES
from walkie_vision.crops.person import crop_person
from walkie_vision.fakes import figure_pose, paint_figure
from walkie_vision.messages import pose_from_person_detection, to_person_detection, to_person_detection_array
from walkie_vision.services.get_person_crop import fill_person_crop_response


def _header() -> Header:
    header = Header(frame_id="camera_optical_frame")
    header.stamp.sec = 1_792_400_000
    header.stamp.nanosec = 250_000_000
    return header


def _camera_info() -> CameraInfo:
    info = CameraInfo(width=2208, height=1242, distortion_model="plumb_bob")
    info.header = _header()
    info.k = [1050.0, 0.0, 1104.0, 0.0, 1050.0, 621.0, 0.0, 0.0, 1.0]
    return info


def _crop_scene():
    """A 2208 x 1242 image with the guest, track 7, and a person behind, track 8."""
    image = np.full((1242, 2208, 3), 60, dtype=np.uint8)
    guest = figure_pose((800.0, 200.0, 1100.0, 1000.0), track_id=7)
    behind = figure_pose((1050.0, 150.0, 1300.0, 900.0), track_id=8)
    paint_figure(image, guest, (0, 0, 255), (255, 0, 0))
    people = to_person_detection_array(_header(), _camera_info(), [to_person_detection(p) for p in (behind, guest)])
    return image, people


def test_person_crop_round_trip():
    image, people = _crop_scene()
    poses = [pose_from_person_detection(p) for p in people.people]
    found = crop_person(image, poses, 7)
    response = fill_person_crop_response(GetPersonCrop.Response(), people, found)
    back = deserialize_message(serialize_message(response), GetPersonCrop.Response)
    assert back.status == GetPersonCrop.Response.OK and back.message == ""
    assert back.header == people.header and back.crop.header == people.header
    assert back.person.track_id == 7
    assert back.crop.format == "jpeg" and bytes(back.crop.data) == found.jpeg
    assert max(found.pixels.shape[:2]) == 512
    np.testing.assert_allclose(np.asarray(back.crop_keypoints).reshape(17, 2), found.keypoints, rtol=1e-5)
    masked = np.asarray(back.masked_boxes).reshape(-1, 4)
    assert len(masked) == 2
    np.testing.assert_allclose(masked, found.masked, rtol=1e-5)
    assert list(back.color_names) == list(COLOR_NAMES)
    assert len(back.top_shares) == len(back.bottom_shares) == len(COLOR_NAMES)
    assert back.top_shares[COLOR_NAMES.index("red")] == pytest.approx(1.0)
    assert back.bottom_shares[COLOR_NAMES.index("blue")] == pytest.approx(1.0)
    assert back.top_left == pytest.approx(1.0) and back.bottom_left == pytest.approx(1.0)


def test_a_gated_region_gives_empty_shares_in_the_response():
    image, people = _crop_scene()
    poses = [pose_from_person_detection(p) for p in people.people]
    poses[1].keypoint_scores[13] = 0.1  # the guest's left knee
    found = crop_person(image, poses, 7)
    response = fill_person_crop_response(GetPersonCrop.Response(), people, found)
    assert response.status == GetPersonCrop.Response.OK
    assert len(response.top_shares) == len(COLOR_NAMES)
    assert len(response.bottom_shares) == 0 and response.bottom_left == 0.0


def test_an_occluded_person_crop_gives_the_person_and_the_overlaps():
    image, people = _crop_scene()
    # Its box ends 100 px lower than the guest's: nearer.
    nearer = to_person_detection(figure_pose((850.0, 300.0, 1150.0, 1100.0), track_id=9))
    people.people.append(nearer)
    found = crop_person(image, [pose_from_person_detection(p) for p in people.people], 7)
    response = fill_person_crop_response(GetPersonCrop.Response(), people, found)
    assert response.status == GetPersonCrop.Response.OCCLUDED and "nearer" in response.message
    assert response.header == people.header and response.person.track_id == 7
    assert response.max_overlap == pytest.approx(found.max_overlap) and response.max_overlap > 0.3
    assert response.torso_cover == pytest.approx(found.torso_cover)
    assert len(response.crop.data) == 0 and len(response.masked_boxes) == 0
    assert len(response.color_names) == 0 and len(response.top_shares) == 0


@pytest.mark.parametrize(
    ("track_id", "status"),
    [(9, GetPersonCrop.Response.NOT_IN_VIEW), (8, GetPersonCrop.Response.TOO_SMALL)],
)
def test_a_person_crop_without_a_crop(track_id, status, monkeypatch):
    image, people = _crop_scene()
    # Track 8 is 750 px tall: too small only when the floor is above that.
    monkeypatch.setattr(person, "MIN_BOX_HEIGHT_PX", 800.0)
    found = crop_person(image, [pose_from_person_detection(p) for p in people.people], track_id)
    response = fill_person_crop_response(GetPersonCrop.Response(), people, found)
    assert response.status == status and response.message
    assert response.header == people.header
    assert response.person.track_id == (0 if status == GetPersonCrop.Response.NOT_IN_VIEW else 8)
    assert len(response.crop.data) == 0 and len(response.crop_keypoints) == 0
    assert len(response.color_names) == 0 and len(response.top_shares) == 0
