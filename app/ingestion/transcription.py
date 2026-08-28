"""
Transcribes audio segments to text using faster-whisper, with per-segment timestamps.
Consumes segment paths from audio.py — output feeds directly into rag/documents.py.
"""
import json
import logging
from pathlib import Path
from faster_whisper import WhisperModel
from app.database.sqlite import get_transcribed_segment_indices, mark_segment_status, update_video_status


from app.config import WHISPER_MODEL, TRANSCRIPT_DIR

logger = logging.getLogger(__name__)

_model = None  # lazy-loaded singleton — loading whisper is expensive, don't reload per segment


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        logger.info(f"Loading whisper model: {WHISPER_MODEL}")
        _model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    return _model


def transcribe_segment(segment_path: str) -> list[dict]:
    """Transcribe one audio segment. Returns list of {text, start, end} per whisper segment."""
    model = _get_model()
    segments, _info = model.transcribe(segment_path, beam_size=5)

    result = [
        {"text": seg.text.strip(), "start": seg.start, "end": seg.end}
        for seg in segments
    ]
    return result


def transcribe_video(segment_paths: list[str], video_id: str) -> list[dict]:
    already_done = get_transcribed_segment_indices(video_id)
    update_video_status(video_id, "transcribing")

    full_transcript = []
    time_offset = 0.0
    segment_length_sec = 10 * 60

    for i, segment_path in enumerate(segment_paths):
        if i in already_done:
            logger.info(f"Skipping segment {i} for {video_id} — already transcribed")
            time_offset += segment_length_sec
            continue

        chunks = transcribe_segment(segment_path)
        for chunk in chunks:
            chunk["start"] += time_offset
            chunk["end"] += time_offset
            full_transcript.append(chunk)

        mark_segment_status(video_id, i, "transcribed")
        time_offset += segment_length_sec

    update_video_status(video_id, "transcribed")
    _save_transcript(full_transcript, video_id)
    return full_transcript


def _save_transcript(transcript: list[dict], video_id: str) -> str:
    """Persist transcript as JSON under data/transcripts/<video_id>.json — resumable checkpoint."""
    out_dir = Path(TRANSCRIPT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_id}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved transcript for {video_id} -> {out_path}")
    return str(out_path)