"""
Tracks per-video and per-segment processing state so pipeline runs are resumable.
Every ingestion/RAG step checks here before redoing work — this is what makes interrupted runs cheap to restart.
"""
import sqlite3
import logging
from contextlib import contextmanager
from pathlib import Path

from app.config import DB_PATH

logger = logging.getLogger(__name__)

VIDEO_STATUSES = ("pending", "downloading", "segmenting", "transcribing", "transcribed", "embedding", "completed", "failed")


@contextmanager
def get_connection():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create tables if they don't exist. Safe to call on every app startup."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                video_id TEXT PRIMARY KEY,
                title TEXT,
                channel TEXT,
                duration INTEGER,
                status TEXT NOT NULL DEFAULT 'pending',
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS segments (
                video_id TEXT NOT NULL,
                segment_index INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                PRIMARY KEY (video_id, segment_index),
                FOREIGN KEY (video_id) REFERENCES videos(video_id)
            )
        """)
    logger.info("Database initialized")


def create_video_record(video_id: str, title: str, channel: str, duration: int):
    """Insert a new video row, or ignore if it already exists (resume case)."""
    with get_connection() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO videos (video_id, title, channel, duration) VALUES (?, ?, ?, ?)",
            (video_id, title, channel, duration),
        )


def update_video_status(video_id: str, status: str, error_message: str | None = None):
    """Update a video's pipeline status. Raises ValueError on an unrecognized status."""
    if status not in VIDEO_STATUSES:
        raise ValueError(f"Unknown status '{status}', must be one of {VIDEO_STATUSES}")
    with get_connection() as conn:
        conn.execute(
            "UPDATE videos SET status = ?, error_message = ?, updated_at = CURRENT_TIMESTAMP WHERE video_id = ?",
            (status, error_message, video_id),
        )


def get_video_status(video_id: str) -> str | None:
    """Returns current status string, or None if the video isn't in the DB yet."""
    with get_connection() as conn:
        row = conn.execute("SELECT status FROM videos WHERE video_id = ?", (video_id,)).fetchone()
        return row["status"] if row else None


def mark_segment_status(video_id: str, segment_index: int, status: str):
    """Upsert a segment's transcription status — used for resume-mid-video."""
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO segments (video_id, segment_index, status) VALUES (?, ?, ?)
               ON CONFLICT(video_id, segment_index) DO UPDATE SET status = excluded.status""",
            (video_id, segment_index, status),
        )


def get_transcribed_segment_indices(video_id: str) -> set[int]:
    """Returns the set of segment indices already marked 'transcribed' — used to skip redone work on resume."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT segment_index FROM segments WHERE video_id = ? AND status = 'transcribed'",
            (video_id,),
        ).fetchall()
        return {row["segment_index"] for row in rows}