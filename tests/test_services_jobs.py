"""walkie_vision.services.jobs: the job, the slot and the bound on a failure message.

Each service's checks and responses are tested in its own tests/test_services_*.py.
The newest input and the wake are tested in tests/test_camera_mailboxes.py,
and the services themselves through the node in tests/test_node.py.
"""

from __future__ import annotations

import time

import pytest
from walkie_vision_msgs.srv import EmbedText

from walkie_vision.fakes import FakeEmbedder
from walkie_vision.services.embed_crop import embed_crop_failure
from walkie_vision.services.embed_text import embed_text_failure, fill_embed_text_response
from walkie_vision.services.jobs import MAX_MESSAGE_CHARS, MODEL_UNAVAILABLE, TIMEOUT, JobSlot, ServiceJob

LONG_S = 30.0


def _job(deadline_s: float = LONG_S) -> ServiceJob:
    return ServiceJob(EmbedText.Request(), time.monotonic() + deadline_s, embed_text_failure)


# --- the job ---------------------------------------------------------------


def test_a_job_answered_in_time_returns_the_answer():
    job = _job()
    answer = EmbedText.Response(success=True)
    assert job.finish(answer)
    assert job.wait() is answer
    assert not job.given_up


def test_a_job_keeps_its_first_answer():
    job = _job()
    first = EmbedText.Response(success=True)
    assert job.finish(first)
    assert not job.finish(EmbedText.Response(message="later"))
    assert job.wait() is first


def test_a_job_past_its_deadline_takes_no_late_answer():
    job = _job(deadline_s=0.0)
    assert job.wait() is None
    assert job.given_up
    assert job.remaining() <= 0
    assert not job.finish(EmbedText.Response(success=True))
    assert job.wait() is None


def test_a_job_refuses_an_answer_after_its_deadline_that_reaches_the_lock_first():
    now = [100.0]
    job = ServiceJob(EmbedText.Request(texts=["cup"]), 100.1, embed_text_failure, clock=lambda: now[0])
    # The worker finishes late, before the waiting callback wakes and gives up.
    now[0] = 100.2
    assert not job.finish(EmbedText.Response(success=True))
    assert job.wait() is None
    assert job.given_up


def test_a_job_on_the_monotonic_clock_refuses_a_late_answer_before_its_caller_gives_up():
    job = _job(deadline_s=-0.1)
    assert not job.finish(EmbedText.Response(success=True))
    assert job.wait() is None


@pytest.mark.parametrize(("finished_at", "taken"), [(100.0999, True), (100.1, False)])
def test_a_job_takes_an_answer_only_before_its_deadline(finished_at, taken):
    now = [100.0]
    job = ServiceJob(EmbedText.Request(texts=["cup"]), 100.1, embed_text_failure, clock=lambda: now[0])
    answer = EmbedText.Response(success=True)
    now[0] = finished_at
    assert job.finish(answer) is taken
    # The caller may read the answer later. Only the time of finish() counts.
    now[0] = 100.2
    assert job.wait() is (answer if taken else None)


# --- the slot --------------------------------------------------------------


def test_the_slot_holds_one_job_and_wakes_the_worker():
    wakes = []
    slot = JobSlot(lambda: wakes.append(1))
    first, second = _job(), _job()
    assert slot.admit(first)
    assert wakes == [1]
    assert not slot.admit(second)
    assert wakes == [1]
    assert slot.claim() is first
    assert slot.claim() is None


def test_a_claimed_job_holds_the_slot_until_its_release():
    slot = JobSlot(lambda: None)
    job = _job()
    assert slot.admit(job)
    assert slot.claim() is job
    # The callback gave up, but the model still runs.
    slot.withdraw(job)
    assert not slot.admit(_job())
    slot.release(job)
    assert slot.admit(_job())


def test_withdrawing_an_unclaimed_job_frees_the_slot():
    slot = JobSlot(lambda: None)
    job = _job()
    assert slot.admit(job)
    slot.withdraw(job)
    assert slot.claim() is None
    assert slot.admit(_job())


def test_an_old_job_cannot_free_the_slot_of_a_new_one():
    slot = JobSlot(lambda: None)
    old, new = _job(), _job()
    assert slot.admit(old)
    slot.release(old)
    assert slot.admit(new)
    slot.release(old)
    slot.withdraw(old)
    assert not slot.admit(_job())


def test_closing_the_slot_answers_its_job_and_refuses_new_ones():
    slot = JobSlot(lambda: None)
    job = _job()
    assert slot.admit(job)
    assert slot.claim() is job
    slot.close()
    assert slot.closed
    answer = job.wait()
    assert not answer.success and answer.message.startswith(MODEL_UNAVAILABLE)
    assert not slot.admit(_job())
    # The worker's answer after that is the late one.
    assert not job.finish(EmbedText.Response(success=True))


def test_closing_the_slot_after_the_deadline_leaves_its_caller_the_timeout():
    now = [100.0]
    slot = JobSlot(lambda: None)
    job = ServiceJob(EmbedText.Request(texts=["cup"]), 100.1, embed_text_failure, clock=lambda: now[0])
    assert slot.admit(job)
    assert slot.claim() is job
    now[0] = 100.2
    slot.close()
    assert slot.closed
    # The deadline has passed, so the callback answers TIMEOUT, as it would without the shutdown.
    assert job.wait() is None
    assert not slot.admit(_job())


# --- the bound on a failure message ----------------------------------------


def test_a_failure_message_is_bounded():
    long = "x" * 10_000
    assert len(embed_text_failure(TIMEOUT, long).message) == MAX_MESSAGE_CHARS
    crop_failure = embed_crop_failure(TIMEOUT, long)
    assert len(crop_failure.message) == MAX_MESSAGE_CHARS
    assert crop_failure.error_code == TIMEOUT
    assert not crop_failure.success and len(crop_failure.embedding) == 0 and len(crop_failure.crop.data) == 0


def test_an_embedder_failure_message_is_bounded():
    class Broken(FakeEmbedder):
        def embed_texts(self, texts):
            raise RuntimeError("x" * 10_000)

    response = fill_embed_text_response(EmbedText.Response(), Broken(), ["mug"])
    assert not response.success and len(response.message) == MAX_MESSAGE_CHARS
