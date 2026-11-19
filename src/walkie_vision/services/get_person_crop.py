"""GetPersonCrop: the request check, the job and its one-slot mailbox, and the responses (ADR-0011).

The callback checks the request with crop_request_problem(), leaves a CropJob
in the CropMailbox and waits for the worker's answer. The worker answers from
the next image and its published people with fill_person_crop_response().
"""

from __future__ import annotations

import math
import threading

import numpy as np
from sensor_msgs.msg import CompressedImage
from walkie_vision_msgs.msg import PersonDetectionArray
from walkie_vision_msgs.srv import GetPersonCrop

from ..crops import person
from ..crops.colour import COLOR_NAMES
from ..crops.person import CropStatus, PersonCrop
from ..messages import float32_array

MAX_CROP_TIMEOUT_S = 5.0
"""The longest timeout_s a GetPersonCrop request may ask for: the service answers one request at a time."""


def _crop_answer(status: CropStatus, message: str) -> GetPersonCrop.Response:
    """A GetPersonCrop response with only a status and a message."""
    return GetPersonCrop.Response(status=int(status), message=message)


class CropJob:
    """One GetPersonCrop request, from the service callback to the worker and back.

    The job gets one answer: the worker's through finish(), or none when
    wait() stops waiting first. A lock decides which comes first.
    """

    def __init__(self, request: GetPersonCrop.Request, frames_received: int) -> None:
        self.request = request
        self.frames_received = frames_received
        """The node's frames_received when the request arrived, to tell NO_IMAGE from TIMEOUT."""
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._response: GetPersonCrop.Response | None = None
        self._given_up = False

    def finish(self, response: GetPersonCrop.Response) -> bool:
        """Answer the job. False when it has an answer already, or nobody waits for one."""
        with self._lock:
            if self._given_up or self._response is not None:
                return False
            self._response = response
        self._done.set()
        return True

    def wait(self, timeout: float) -> GetPersonCrop.Response | None:
        """Return the answer, waiting up to timeout seconds; None when there is none, and then no answer is taken."""
        self._done.wait(timeout)
        with self._lock:
            if self._response is None:
                self._given_up = True
            return self._response


class CropMailbox:
    """A one-slot mailbox for the GetPersonCrop job that waits for the next image.

    The service's callback group is mutually exclusive, so at most one job
    waits at a time.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._job: CropJob | None = None
        self._closed = False

    def put(self, job: CropJob) -> bool:
        """Leave a job for the worker. False when the mailbox is closed."""
        with self._lock:
            if self._closed:
                return False
            self._job = job
            return True

    def take(self) -> CropJob | None:
        """Return the waiting job without waiting; None when there is none."""
        with self._lock:
            job, self._job = self._job, None
            return job

    def withdraw(self, job: CropJob) -> None:
        """Remove job if it still waits."""
        with self._lock:
            if self._job is job:
                self._job = None

    def close(self) -> None:
        """Refuse new jobs, and answer the waiting one with FAILED."""
        with self._lock:
            self._closed = True
            job, self._job = self._job, None
        if job is not None:
            job.finish(_crop_answer(CropStatus.FAILED, "the node is shutting down"))


def crop_request_problem(request: GetPersonCrop.Request) -> str | None:
    """Return why a GetPersonCrop request is refused, or None when it is valid."""
    if request.track_id == 0:
        return "track_id 0 is not a track"
    timeout = float(request.timeout_s)
    if not (math.isfinite(timeout) and 0.0 < timeout <= MAX_CROP_TIMEOUT_S):
        return f"timeout_s must be more than 0 and at most {MAX_CROP_TIMEOUT_S:g}, not {timeout}"
    try:
        person.check_options(float(request.padding), int(request.max_side), float(request.min_keypoint_score))
    except ValueError as e:
        return str(e)
    return None


def fill_person_crop_response(
    response: GetPersonCrop.Response, people: PersonDetectionArray, found: PersonCrop
) -> GetPersonCrop.Response:
    """Fill a GetPersonCrop response from walkie_vision.crops.person.crop_person()'s result.

    people is the published array of the image that found came from. The
    header is set for every status that found can have; the person and the
    overlaps once the track is found; the crop and the shares only for OK.
    """
    response.status = int(found.status)
    response.message = found.message
    response.header = people.header
    if found.index < 0:
        return response
    response.person = people.people[found.index]
    response.max_overlap = float(found.max_overlap)
    response.torso_cover = float(found.torso_cover)
    if found.status != CropStatus.OK:
        return response
    response.crop = CompressedImage(header=people.header, format="jpeg", data=found.jpeg)
    response.crop_keypoints = float32_array(np.asarray(found.keypoints).reshape(-1))
    response.masked_boxes = float32_array(np.asarray(found.masked).reshape(-1))
    response.color_names = list(COLOR_NAMES)
    if found.top_shares is not None:
        response.top_shares = float32_array(found.top_shares)
        response.top_left = float(found.top_left)
    if found.bottom_shares is not None:
        response.bottom_shares = float32_array(found.bottom_shares)
        response.bottom_left = float(found.bottom_left)
    return response
