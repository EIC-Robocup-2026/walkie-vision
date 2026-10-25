"""The track id base: ids from a later run never meet an earlier run's."""

import pytest

from walkie_vision.pipeline import TRACK_ID_SPACE, TrackIdBase

MS = 1_000_000


def test_zero_stays_zero_and_negative_ids_raise():
    base = TrackIdBase(start_ns=123 * MS)
    assert base.map(0) == 0
    with pytest.raises(ValueError):
        base.map(-1)


def test_a_zero_start_maps_ids_to_themselves():
    base = TrackIdBase(start_ns=0)
    assert base.offset == 0
    assert [base.map(k) for k in (1, 2, 3, 1000)] == [1, 2, 3, 1000]


def test_the_offset_is_the_start_time_in_ms():
    assert TrackIdBase(start_ns=5 * MS + 999_999).offset == 5
    assert TrackIdBase(start_ns=(TRACK_ID_SPACE + 7) * MS).offset == 7


def test_ids_wrap_without_reaching_zero():
    base = TrackIdBase(start_ns=(TRACK_ID_SPACE - 1) * MS)
    assert base.offset == TRACK_ID_SPACE - 1
    assert base.map(1) == TRACK_ID_SPACE
    assert base.map(2) == 1
    assert all(base.map(k) != 0 for k in range(1, 5000))


@pytest.mark.parametrize("start_ms", [0, 1, TRACK_ID_SPACE // 2, TRACK_ID_SPACE - 1, 1_793_000_000_000])
def test_every_id_fits_in_uint32(start_ms):
    base = TrackIdBase(start_ns=start_ms * MS)
    for local in (1, 2, 10_000, TRACK_ID_SPACE - 1, TRACK_ID_SPACE, 2**40):
        assert 1 <= base.map(local) <= 2**32 - 1


def test_the_default_start_is_now():
    base = TrackIdBase()
    assert 0 <= base.offset < TRACK_ID_SPACE


def test_two_runs_ten_seconds_apart_never_share_an_id():
    start = 1_793_000_000_000 * MS
    first = TrackIdBase(start_ns=start)
    second = TrackIdBase(start_ns=start + 10_000 * MS)
    earlier = {first.map(k) for k in range(1, 10_000)}
    later = {second.map(k) for k in range(1, 10_000)}
    assert len(earlier) == 9_999
    assert earlier.isdisjoint(later)
