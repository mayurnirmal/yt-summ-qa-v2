"""
Tests video/segment state tracking, including resume-after-interruption behavior.
Uses an in-memory-style temp DB file per test — no shared state between tests.
"""
import pytest
from app.database import sqlite as db


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(db, "DB_PATH", str(db_path))
    db.init_db()
    yield


def test_create_and_get_video_status():
    db.create_video_record("vid1", "Title", "Channel", 300)
    assert db.get_video_status("vid1") == "pending"


def test_update_video_status():
    db.create_video_record("vid1", "Title", "Channel", 300)
    db.update_video_status("vid1", "transcribing")
    assert db.get_video_status("vid1") == "transcribing"


def test_update_video_status_rejects_unknown_status():
    db.create_video_record("vid1", "Title", "Channel", 300)
    with pytest.raises(ValueError):
        db.update_video_status("vid1", "not_a_real_status")


def test_get_video_status_returns_none_for_unknown_video():
    assert db.get_video_status("nonexistent") is None


def test_segment_status_tracking_for_resume():
    db.create_video_record("vid1", "Title", "Channel", 300)
    db.mark_segment_status("vid1", 0, "transcribed")
    db.mark_segment_status("vid1", 1, "transcribed")
    db.mark_segment_status("vid1", 2, "pending")

    done = db.get_transcribed_segment_indices("vid1")
    assert done == {0, 1}


def test_mark_segment_status_upserts_on_conflict():
    db.create_video_record("vid1", "Title", "Channel", 300)
    db.mark_segment_status("vid1", 0, "pending")
    db.mark_segment_status("vid1", 0, "transcribed")

    done = db.get_transcribed_segment_indices("vid1")
    assert done == {0}