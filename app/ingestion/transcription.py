"""
Transcribes audio segments to text using faster-whisper, with per-segment timestamps.
Consumes segment paths from audio.py — output feeds directly into rag/documents.py.
"""
# app/ingestion/transcription.py — replace _save_transcript and transcribe_video
import json
import logging
import os
from pathlib import Path
from faster_whisper import WhisperModel

from app.config import WHISPER_MODEL, TRANSCRIPT_DIR
from app.database.sqlite import get_transcribed_segment_indices, mark_segment_status, update_video_status
from faster_whisper import WhisperModel, BatchedInferencePipeline

logger = logging.getLogger(__name__)

_model = None
_batched_model = None

# app/ingestion/transcription.py — updated _get_model()
def _get_model() -> BatchedInferencePipeline:
    global _model, _batched_model
    if _batched_model is None:
        logger.info(f"Loading whisper model: {WHISPER_MODEL}")
        _model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8", cpu_threads=os.cpu_count())
        _batched_model = BatchedInferencePipeline(model=_model)
    return _batched_model

def transcribe_segment(segment_path: str) -> list[dict]:
    model = _get_model()
    segments, _info = model.transcribe(segment_path, beam_size=5, batch_size=16, vad_filter=True)
    return [{"text": seg.text.strip(), "start": seg.start, "end": seg.end} for seg in segments]

def _segment_cache_path(video_id: str, segment_index: int) -> Path:
    return Path(TRANSCRIPT_DIR) / video_id / f"segment_{segment_index:03d}.json"


def _save_segment_cache(chunks: list[dict], video_id: str, segment_index: int):
    path = _segment_cache_path(video_id, segment_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def _load_segment_cache(video_id: str, segment_index: int) -> list[dict]:
    path = _segment_cache_path(video_id, segment_index)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def transcribe_video(segment_paths: list[str], video_id: str) -> list[dict]:
    already_done = get_transcribed_segment_indices(video_id)
    update_video_status(video_id, "transcribing")

    full_transcript = []
    time_offset = 0.0
    segment_length_sec = 10 * 60

    for i, segment_path in enumerate(segment_paths):
        if i in already_done:
            try:
                chunks = _load_segment_cache(video_id, i)
                logger.info(f"Loaded cached segment {i} for {video_id}")
            except FileNotFoundError:
                # DB says done but cache is missing (stale/inconsistent state) — redo it
                logger.warning(f"Segment {i} marked transcribed but no cache found — re-transcribing")
                chunks = [
                    {**c, "start": c["start"] + time_offset, "end": c["end"] + time_offset}
                    for c in transcribe_segment(segment_path)
                ]
                _save_segment_cache(chunks, video_id, i)
        else:
            chunks = [
                {**c, "start": c["start"] + time_offset, "end": c["end"] + time_offset}
                for c in transcribe_segment(segment_path)
            ]
            _save_segment_cache(chunks, video_id, i)
            mark_segment_status(video_id, i, "transcribed")

        full_transcript.extend(chunks)
        time_offset += segment_length_sec

    update_video_status(video_id, "transcribed")
    _save_full_transcript(full_transcript, video_id)
    return full_transcript


def _save_full_transcript(transcript: list[dict], video_id: str) -> str:
    """Assembled convenience copy for documents.py to read — segment caches remain the source of truth for resume."""
    out_path = Path(TRANSCRIPT_DIR) / f"{video_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved transcript for {video_id} -> {out_path}")
    return str(out_path)