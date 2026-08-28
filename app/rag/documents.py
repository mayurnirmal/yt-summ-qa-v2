"""
Converts a video's transcript into chunked LangChain Document objects with metadata.
Output feeds directly into embeddings.py/vectorstore.py — every chunk carries the metadata retrieval will filter/cite on.
"""
import json
import logging
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.config import TRANSCRIPT_DIR, CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def load_transcript(video_id: str) -> list[dict]:
    """Load the saved transcript JSON for a video (list of {text, start, end})."""
    path = Path(TRANSCRIPT_DIR) / f"{video_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No transcript found for {video_id} at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def transcript_to_text_with_timestamps(transcript: list[dict]) -> str:
    """Joins transcript chunks into one text blob. Kept separate from splitting so it's testable on its own."""
    return " ".join(chunk["text"] for chunk in transcript)


def _find_timestamp_for_offset(transcript: list[dict], char_offset: int) -> float:
    """Maps a character offset in the joined text back to the nearest transcript segment's start time."""
    running_len = 0
    for chunk in transcript:
        chunk_len = len(chunk["text"]) + 1  # +1 for the joining space
        if running_len + chunk_len > char_offset:
            return chunk["start"]
        running_len += chunk_len
    return transcript[-1]["start"] if transcript else 0.0


def build_documents(video_id: str, video_metadata: dict) -> list[Document]:
    """Chunk a video's transcript into Documents, each tagged with video_id, title, channel, and a start timestamp."""
    transcript = load_transcript(video_id)
    full_text = transcript_to_text_with_timestamps(transcript)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    # split_text doesn't return offsets, so track position via find() from a moving cursor
    chunks = splitter.split_text(full_text)
    documents = []
    cursor = 0

    for chunk_text in chunks:
        offset = full_text.find(chunk_text, cursor)
        if offset == -1:
            offset = cursor  # fallback if overlap makes find() miss — rare, but don't crash
        timestamp = _find_timestamp_for_offset(transcript, offset)

        documents.append(Document(
            page_content=chunk_text,
            metadata={
                "video_id": video_id,
                "title": video_metadata.get("title"),
                "channel": video_metadata.get("channel"),
                "start_timestamp": timestamp,
            },
        ))
        cursor = offset + 1  # allow overlap without re-finding the same spot

    logger.info(f"Built {len(documents)} documents for {video_id}")
    return documents