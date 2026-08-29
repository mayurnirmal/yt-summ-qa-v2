"""
Orchestrates the full ingest-to-queryable pipeline for one video, resuming from wherever
the DB says it left off instead of redoing completed stages.
Replaces the manual step-by-step calls in the app.py smoke test with one entry point.
"""
import logging
from pathlib import Path

from app.config import AUDIO_DIR, VECTORSTORE_DIR
from app.database.sqlite import (
    create_video_record, update_video_status, get_video_status,
)
from app.ingestion.youtube import get_video_metadata, download_audio, extract_video_id
from app.ingestion.audio import segment_audio
from app.ingestion.transcription import transcribe_video
from app.rag.documents import build_documents
from app.rag.vectorstore import build_vectorstore, load_vectorstore
from app.rag.retriever import get_retriever

logger = logging.getLogger(__name__)


def process_video(url: str):
    """Runs (or resumes) the full pipeline for a video. Returns a retriever ready for querying."""
    video_id = extract_video_id(url)
    status = get_video_status(video_id)

    # already fully processed — just load the existing index, skip everything else
    if status == "completed" and _vectorstore_exists(video_id):
        logger.info(f"{video_id} already completed — loading cached vectorstore")
        return get_retriever(load_vectorstore(video_id))

    metadata = get_video_metadata(url)
    create_video_record(video_id, metadata["title"], metadata["channel"], metadata["duration"])

    try:
        audio_path = _get_or_download_audio(url, video_id)
        segments = segment_audio(audio_path, video_id)
        transcribe_video(segments, video_id)  # internally resumes per-segment already

        if _vectorstore_exists(video_id):
            logger.info(f"Reusing existing vectorstore for {video_id}")
            store = load_vectorstore(video_id)
        else:
            documents = build_documents(video_id, metadata)
            update_video_status(video_id, "embedding")
            store = build_vectorstore(documents, video_id)

        update_video_status(video_id, "completed")
        return get_retriever(store)

    except Exception as e:
        logger.error(f"Pipeline failed for {video_id}: {e}")
        update_video_status(video_id, "failed", error_message=str(e))
        raise


def _vectorstore_exists(video_id: str) -> bool:
    return (Path(VECTORSTORE_DIR) / video_id).exists()


def _get_or_download_audio(url: str, video_id: str) -> str:
    """Skips download if full_audio.mp3 already exists on disk."""
    existing = Path(AUDIO_DIR) / video_id / "full_audio.mp3"
    if existing.exists():
        logger.info(f"Reusing existing audio for {video_id}")
        return str(existing)
    update_video_status(video_id, "downloading")
    return download_audio(url, video_id)