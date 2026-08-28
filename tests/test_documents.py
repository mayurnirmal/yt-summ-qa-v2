"""
Tests transcript-to-Document chunking and timestamp mapping.
Uses an inline fake transcript — no file I/O needed for the mapping logic itself.
"""
import json
import pytest
from app.rag import documents as docs_module


FAKE_TRANSCRIPT = [
    {"text": "This is the first part of the video.", "start": 0.0, "end": 3.0},
    {"text": "Now we move on to the second part.", "start": 3.5, "end": 6.5},
    {"text": "And finally the third and last part.", "start": 7.0, "end": 10.0},
]


@pytest.fixture
def fake_transcript_file(tmp_path, monkeypatch):
    monkeypatch.setattr(docs_module, "TRANSCRIPT_DIR", str(tmp_path))
    path = tmp_path / "vid123.json"
    with open(path, "w") as f:
        json.dump(FAKE_TRANSCRIPT, f)
    return "vid123"


def test_load_transcript(fake_transcript_file):
    result = docs_module.load_transcript(fake_transcript_file)
    assert result == FAKE_TRANSCRIPT


def test_load_transcript_missing_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(docs_module, "TRANSCRIPT_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        docs_module.load_transcript("nonexistent")


def test_transcript_to_text_joins_chunks():
    text = docs_module.transcript_to_text_with_timestamps(FAKE_TRANSCRIPT)
    assert text == "This is the first part of the video. Now we move on to the second part. And finally the third and last part."


def test_find_timestamp_for_offset_start():
    text = docs_module.transcript_to_text_with_timestamps(FAKE_TRANSCRIPT)
    offset = text.find("second part")
    timestamp = docs_module._find_timestamp_for_offset(FAKE_TRANSCRIPT, offset)
    assert timestamp == 3.5


def test_build_documents_creates_chunks_with_metadata(fake_transcript_file, monkeypatch):
    monkeypatch.setattr(docs_module, "CHUNK_SIZE", 50)
    monkeypatch.setattr(docs_module, "CHUNK_OVERLAP", 10)

    video_metadata = {"title": "Test Video", "channel": "Test Channel"}
    result = docs_module.build_documents(fake_transcript_file, video_metadata)

    assert len(result) > 1  # small chunk_size forces multiple documents
    for doc in result:
        assert doc.metadata["video_id"] == fake_transcript_file
        assert doc.metadata["title"] == "Test Video"
        assert doc.metadata["channel"] == "Test Channel"
        assert "start_timestamp" in doc.metadata