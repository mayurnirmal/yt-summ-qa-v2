"""
Tests transcription output shape and timestamp-offsetting logic.
Mocks WhisperModel entirely — no real model load or audio needed in CI.
"""
import json
import pytest
from unittest.mock import MagicMock, patch

from app.ingestion.transcription import transcribe_video, transcribe_segment


def _fake_whisper_segment(text, start, end):
    seg = MagicMock()
    seg.text = text
    seg.start = start
    seg.end = end
    return seg


@patch("app.ingestion.transcription._get_model")
def test_transcribe_segment_returns_expected_shape(mock_get_model):
    mock_model = MagicMock()
    mock_model.transcribe.return_value = (
        [_fake_whisper_segment("hello world", 0.0, 2.5)],
        None,
    )
    mock_get_model.return_value = mock_model

    result = transcribe_segment("fake_path.mp3")

    assert result == [{"text": "hello world", "start": 0.0, "end": 2.5}]
 
    
@patch("app.ingestion.transcription._save_full_transcript")
@patch("app.ingestion.transcription._save_segment_cache")
@patch("app.ingestion.transcription.transcribe_segment")
def test_transcribe_video_offsets_timestamps_across_segments(
    mock_transcribe_segment,
    mock_save_segment,
    mock_save_full,
):
    # segment 0: one chunk at 0-5s
    # segment 1: one chunk at 0-3s
    # second segment should become 600-603s

    mock_transcribe_segment.side_effect = [
        [{"text": "first segment", "start": 0.0, "end": 5.0}],
        [{"text": "second segment", "start": 0.0, "end": 3.0}],
    ]

    result = transcribe_video(
        ["seg_000.mp3", "seg_001.mp3"],
        "video123"
    )

    assert result[0]["start"] == 0.0
    assert result[1]["start"] == 600.0
    assert result[1]["end"] == 603.0

    mock_save_full.assert_called_once()