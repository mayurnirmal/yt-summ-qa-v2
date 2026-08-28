"""
Tests audio segmentation logic.
Uses a generated silent audio file so no real video download is needed in CI.
"""
import pytest
from pathlib import Path
from pydub import AudioSegment

from app.ingestion.audio import segment_audio, SEGMENT_LENGTH_MS


@pytest.fixture
def sample_audio(tmp_path):
    """Generates a 25-min silent mp3 to test multi-segment splitting."""
    audio = AudioSegment.silent(duration=25 * 60 * 1000)  # 25 minutes
    audio_path = tmp_path / "test_audio.mp3"
    audio.export(audio_path, format="mp3")
    return str(audio_path), "test_video_id"


def test_segment_audio_creates_correct_count(sample_audio, monkeypatch, tmp_path):
    audio_path, video_id = sample_audio
    monkeypatch.setattr("app.ingestion.audio.AUDIO_DIR", str(tmp_path))

    segments = segment_audio(audio_path, video_id)

    # 25 min at 10-min segments = 3 segments (10, 10, 5)
    assert len(segments) == 3
    for path in segments:
        assert Path(path).exists()


def test_segment_audio_last_segment_shorter(sample_audio, monkeypatch, tmp_path):
    audio_path, video_id = sample_audio
    monkeypatch.setattr("app.ingestion.audio.AUDIO_DIR", str(tmp_path))

    segments = segment_audio(audio_path, video_id)
    last_segment = AudioSegment.from_file(segments[-1])

    assert len(last_segment) <= SEGMENT_LENGTH_MS