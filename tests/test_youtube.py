"""
tests/test_youtube.py

Unit tests for app/ingestion/youtube.py — specifically extract_video_id(),
the "URL → video ID" test called out in spec section 22.1.

Why no network calls here:
    get_video_metadata() and download_audio() hit YouTube directly, so they
    are intentionally NOT tested live in this file. Per spec section 23,
    CI should not download real videos — those two functions get mocked
    tests in a follow-up (feature/audio-segmentation prep), not real
    network calls in the default test run.
"""
import pytest
from app.ingestion.youtube import extract_video_id, InvalidYouTubeURLError

@pytest.mark.parametrize("url,expected_id", [
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30s", "dQw4w9WgXcQ"),
    ("https://www.youtube.com/watch?v=o1tIvfplhHc&list=RDo1tIvfplhHc&start_radio=1","o1tIvfplhHc")
])
def test_extract_video_id_valid(url, expected_id):
    assert extract_video_id(url) == expected_id

def test_extract_video_id_invalid():
    with pytest.raises(InvalidYouTubeURLError):
        extract_video_id("https://example.com/not-a-video")